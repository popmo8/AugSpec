// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"
#include "model/moe.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <cstdlib>
#include <future>

// aug_spec profiling: monotonic microsecond clock for the AUG_PROFILE counters.
static inline int64_t _prof_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      dtype_(dtype),
      num_experts_(num_experts),
      // input_mutex_(kNumDevices()),
      // input_cv_(kNumDevices()),
      // exec_mutex_(kNumDevices()),
      // exec_cv_(kNumDevices()),
      cache_mutex_(kNumDevices()),
      cache_cv_(kNumDevices()),
      input_queue_(kNumDevices()),
      gpu_overload_(kNumDevices(), false),
      exec_queue_(kNumDevices()),
      cached_experts_(kNumDevices()),
      pinned_(kNumDevices()),
      modules_(kNumDevices(), nullptr) {
  main_thread_stop_flag_.store(false);
  profile_enabled_ = (std::getenv("AUG_PROFILE") != nullptr);

  // module_ = new MoEMLP(dtype, expert_type);

  // Futex<bool> initial_value(false);
  // gpu_overload_ = std::move(std::vector<Futex<bool>>(kNumDevices(),
  // initial_value));

  for (int i = 0; i < kNumDevices(); ++i) {
    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    std::string thread_name = "GPUFetchFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);

    modules_[i] = new MoEMLP(dtype, expert_type);
    // gpu_overload_.emplace_back(false);
  }

  for (int i = 0; i < kNumDevices() * num_threads; ++i) {
    cudaSetDevice(i % kNumDevices());
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);
    // cudaDeviceSynchronize();

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % kNumDevices());
    std::string thread_name = "GPUExecFunc" + std::to_string(i % kNumDevices());
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());
  }

  at::InferenceMode infer_guard(0);

  for (int i = 0; i < num_experts; ++i) {
    experts_.emplace_back();
    for (int j = 0; j < num_layers; ++j) {
      experts_[i].emplace_back();
      experts_[i][j] = std::make_shared<ExpertNode>();
      experts_[i][j]->expert_type = expert_type;
      int expert_type = expert_type_;
      switch (expert_type) {
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
          experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
          break;
        case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
          experts_[i][j]->module =
              new SwitchTransformersDenseGatedActDense(dtype);
          break;
        case NLLB_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
          break;
        case FSGPT_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
          break;
        case MIXTRAL_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
          break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
          break;
        default:
          DLOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                     expert_type);
      }
      experts_[i][j]->module->eval();
      experts_[i][j]->layer_idx = j;
      experts_[i][j]->expert_idx = i;
    }
  }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id,
                                     bool remote) {
  ExpertDispatcher::CallArgs args;
  args.layer_idx = layer_idx;
  args.expert_idx = expert_idx;
  args.gpu_id = gpu_id;
  args.remote = remote;
  Enqueue(args);
}

void ExpertDispatcher::Enqueue(CallArgs& args) {
  // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];

  // aug_spec race fix: under heavy cache churn (esp. topm's tiny merged-reserve
  // pool → frequent eviction) a concurrent fetch/evict can hold this node's
  // mutex mid-move (DEVICE[cuda;cuda;cpu]). The old `try_lock + FATAL` aborted
  // the whole run on that timing window. Enqueue holds no other lock here, so
  // the mover (which holds node->mutex, not anything Enqueue needs) cannot
  // deadlock us — wait (bounded) for it to finish and release. The device
  // re-check below then routes correctly: still resident → exec; evicted while
  // we waited → input_queue re-fetch. The ~10s ceiling distinguishes genuine
  // churn (sub-ms) from a real stuck thread (keeps the original assert's intent
  // of catching logic bugs, without crashing on benign contention).
  int _enq_spins = 0;
  int64_t _enq_t0 = profile_enabled_ ? _prof_now_us() : 0;
  while (!expert_node->node->mutex.try_lock()) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    if (++_enq_spins > 200000) {
      DLOG_FATAL("ExpertDispatcher::Enqueue: node mutex held >10s (stuck), "
                 "expert_idx ", expert_idx, " layer_idx ", layer_idx, " node ",
                 expert_node->node->str());
    }
  }
  if (profile_enabled_ && _enq_spins > 0) {
    prof_.enqueue_wait_us += _prof_now_us() - _enq_t0;
    prof_.enqueue_wait_n += 1;   // how often the race window was hit
  }
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    args.gpu_id = expert_node->node->device.index();

    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();

    ExecArgs exec_args;
    // exec_args.hidden_states = std::move(input);
    exec_args.expert_node = expert_node;
    expert_node->SetTensorsFromBlob(expert_node->node->device);
    exec_args.out_gpu_id = original_device.index();
    exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
    exec_args.evict = false;
    exec_args.hit = true;

    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::unique_lock<std::mutex> lock(exec_mutex_[args.gpu_id]);
    // exec_queue_[args.gpu_id].push_back(std::move(exec_args));
    exec_queue_[args.gpu_id].Push(exec_args);
  } else {
    // std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    // input_queue_[args.gpu_id].push_back(std::move(args));
    input_queue_[args.gpu_id].Push(args);
  }
  // input_cv_[args.gpu_id].notify_all();
  // exec_cv_[args.gpu_id].notify_all();
  // input_queue_.push_back(std::move(args));
  num_enqueued_.fetch_add(1);

  // auto& a = input_queue_.back();
  // if (expert_node->node->device.is_cuda()) {
  //   a.gpu_id = expert_node->node->device.index();
  // }
  // DLOG_TRACE("ExpertDispatcher::Enqueue: num_enqueued_ ",
  // num_enqueued_.load(),
  //            "input_queue_ ", input_queue_.size(), "gpu_id ", a.gpu_id,
  //            "layer_idx ", a.layer_idx, "expert_idx ", a.expert_idx, "remote
  //            ", a.remote);
  // lock.unlock();
  // cvs_[MUTEX_TYPE::INPUT_MUTEX].notify_all();
}

void ExpertDispatcher::RegisterExpert(
    int layer_idx, int expert_idx, const std::vector<std::uint32_t>& tensor_ids,
    std::string jit_path) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
      // experts_[expert_idx][layer_idx]->jit_module =
      //     new torch::jit::script::Module(torch::jit::load(jit_path));
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

std::vector<torch::Tensor> ExpertDispatcher::GetResidentExpertWeights(
    int layer_idx, int expert_idx, int gpu_id) {
  std::vector<torch::Tensor> out;
  if (expert_idx < 0 || expert_idx >= static_cast<int>(experts_.size())) {
    return out;
  }
  if (layer_idx < 0 ||
      layer_idx >= static_cast<int>(experts_[expert_idx].size())) {
    return out;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
  auto expert_node = experts_[expert_idx][layer_idx];
  if (expert_node == nullptr || expert_node->node == nullptr) {
    return out;
  }
  // Only return weights when the expert is actually GPU-resident; a non-cuda
  // device means it was evicted / never fetched -> report a miss (empty).
  if (!expert_node->node->device.is_cuda()) {
    return out;
  }
  for (auto tid : expert_node->node->tensor_ids) {
    auto it = kTensorIndex->find(tid);
    if (it == kTensorIndex->end() || !it->second.tensor.defined() ||
        !it->second.tensor.device().is_cuda()) {
      return {};  // partially resident -> treat as miss
    }
    out.push_back(it->second.tensor);
  }
  return out;
}

std::vector<torch::Tensor> ExpertDispatcher::MergeExpertsLocal(
    int layer_idx, const std::vector<int>& expert_ids,
    const std::vector<double>& weights, int gpu_id) {
  int64_t _mg_t0 = profile_enabled_ ? _prof_now_us() : 0;
  std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
  auto device = torch::Device(torch::kCUDA, gpu_id);
  std::vector<torch::Tensor> acc;          // fp32 accumulators per weight matrix
  c10::ScalarType out_dtype = torch::kBFloat16;
  bool sized = false;

  for (size_t k = 0; k < expert_ids.size(); ++k) {
    int e = expert_ids[k];
    double w = weights[k];
    if (w == 0.0) continue;
    if (e < 0 || e >= static_cast<int>(experts_.size())) continue;
    if (layer_idx < 0 ||
        layer_idx >= static_cast<int>(experts_[e].size())) {
      continue;
    }
    auto expert_node = experts_[e][layer_idx];
    if (expert_node == nullptr || expert_node->node == nullptr) continue;

    auto& tids = expert_node->node->tensor_ids;
    if (!sized) {
      acc.resize(tids.size());
      sized = true;
    }
    for (size_t i = 0; i < tids.size() && i < acc.size(); ++i) {
      auto it = kTensorIndex->find(tids[i]);
      if (it == kTensorIndex->end() || !it->second.tensor.defined()) continue;
      torch::Tensor t = it->second.tensor;
      // Resident -> read in place (zero PCIe). Cold -> transient host->GPU copy
      // that is freed when `tg` leaves scope; the cache is never touched.
      torch::Tensor tg = t.device().is_cuda() ? t : t.to(device);
      if (!acc[i].defined()) {
        acc[i] = torch::zeros(
            tg.sizes(),
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        out_dtype = t.scalar_type();
      }
      acc[i].add_(tg.to(torch::kFloat32), w);
    }
  }

  std::vector<torch::Tensor> out;
  for (auto& a : acc) {
    if (a.defined()) out.push_back(a.to(out_dtype));
  }
  if (profile_enabled_) {
    prof_.merge_us += _prof_now_us() - _mg_t0;
    prof_.merge_n += 1;
  }
  return out;
}

torch::Tensor ExpertDispatcher::DispatchMergedLocal(
    torch::Tensor hidden_states, torch::Tensor weight,
    const std::vector<std::vector<torch::Tensor>>& merged, int gpu_id) {
  int64_t _dp_t0 = profile_enabled_ ? _prof_now_us() : 0;
  cudaSetDevice(gpu_id);
  auto device = CUDA_DEVICE(gpu_id);
  const int64_t T = hidden_states.size(0);
  const int64_t D = hidden_states.size(1);
  const int K = static_cast<int>(merged.size());

  auto final_hidden = torch::zeros(
      {T, D}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // Run on the current stream — synchronous (merged are resident, no fetch);
  // MoEMLP::forward syncs it each call, so the K experts run sequentially on
  // modules_[gpu_id] and all the torch ops below stay ordered on one stream.
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gpu_id).stream();

  for (int k = 0; k < K; ++k) {
    auto w_k = weight.select(1, k);                       // [T]
    auto token_idx = (w_k > 0).nonzero().squeeze(-1);     // [t_k]
    if (token_idx.numel() == 0) continue;
    auto input = hidden_states.index_select(0, token_idx).to(device);

    modules_[gpu_id]->SetTensorsDirect(merged[k]);
    auto output = modules_[gpu_id]->forward(input, stream);   // [t_k, D]

    auto scale = w_k.index_select(0, token_idx).unsqueeze(-1);  // [t_k, 1]
    final_hidden.index_add_(
        0, token_idx, (output.to(torch::kFloat32) * scale.to(torch::kFloat32)));
  }

  if (profile_enabled_) {  // forward() already synced each expert
    prof_.dispatch_us += _prof_now_us() - _dp_t0;
    prof_.dispatch_n += 1;
  }
  return final_hidden.to(hidden_states.dtype());
}

torch::Tensor ExpertDispatcher::DispatchBmm(
    torch::Tensor hidden, torch::Tensor gw, torch::Tensor uw,
    torch::Tensor dw, torch::Tensor weight, int gpu_id) {
  int64_t _db_t0 = profile_enabled_ ? _prof_now_us() : 0;
  cudaSetDevice(gpu_id);
  const int64_t E = gw.size(0);
  const int64_t T = hidden.size(0);
  const int64_t D = hidden.size(1);
  // Pre-stacked resident experts → 3 batched GEMMs. gw/uw are [E, D, I] and dw
  // is [E, I, D], so bmm(hidden, ·) gives SiLU(hs·gateᵀ)⊙(hs·upᵀ)·downᵀ. The
  // stack itself is memoised Python-side (once per cycle); this is the exact
  // op sequence of the Python torch.bmm path, just inside the engine.
  auto hsE = hidden.unsqueeze(0).expand({E, T, D});                   // [E,T,D]
  auto hid = torch::silu(torch::bmm(hsE, gw)) * torch::bmm(hsE, uw);  // [E,T,I]
  auto eo = torch::bmm(hid, dw);                                      // [E,T,D]
  // out[t] = Σ_e weight[t,e] · expert_e(hs[t]).
  auto out = (eo * weight.transpose(0, 1).unsqueeze(-1)).sum(0);      // [T,D]
  if (profile_enabled_) {  // sync so the timing reflects real GPU work
    c10::cuda::getCurrentCUDAStream(gpu_id).synchronize();
    prof_.dispatch_us += _prof_now_us() - _db_t0;
    prof_.dispatch_n += 1;
  }
  return out;
}

void ExpertDispatcher::FlushCache(int gpu_id) {
  std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
  for (auto key : cached_experts_[gpu_id]) {
    int64_t layer_idx = static_cast<int64_t>(key >> 32);
    int64_t expert_idx = static_cast<int64_t>(key & 0xFFFFFFFF);
    if (expert_idx < 0 || expert_idx >= static_cast<int64_t>(experts_.size())) {
      continue;
    }
    if (layer_idx < 0 ||
        layer_idx >= static_cast<int64_t>(experts_[expert_idx].size())) {
      continue;
    }
    auto expert_node = experts_[expert_idx][layer_idx];
    if (expert_node == nullptr || expert_node->node == nullptr) continue;
    if (expert_node->node->device.is_cuda()) {
      // Host copy is the offload source — this just frees the GPU mirror.
      expert_node->node->SetDevice(expert_node->node->default_host);
    }
  }
  cached_experts_[gpu_id].clear();
  cache_sizes_[gpu_id] =
      kTopologyHandle->GetSparseCacheLimit(CUDA_DEVICE(gpu_id));
}

void ExpertDispatcher::EvictLayer(int layer_idx, int gpu_id) {
  int64_t _el_t0 = profile_enabled_ ? _prof_now_us() : 0;
  std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
  std::vector<uint64_t> to_evict;
  for (auto key : cached_experts_[gpu_id]) {
    if (static_cast<int>(key >> 32) == layer_idx) {
      to_evict.push_back(key);
    }
  }
  for (auto key : to_evict) {
    int64_t expert_idx = static_cast<int64_t>(key & 0xFFFFFFFF);
    if (expert_idx < 0 || expert_idx >= static_cast<int64_t>(experts_.size())) {
      continue;
    }
    auto expert_node = experts_[expert_idx][layer_idx];
    if (expert_node == nullptr || expert_node->node == nullptr) continue;
    if (expert_node->node->device.is_cuda()) {
      // Host copy is the offload source — frees the GPU mirror, no D2H.
      expert_node->node->SetDevice(expert_node->node->default_host);
      cache_sizes_[gpu_id] += expert_node->node->byte_size;
    }
    cached_experts_[gpu_id].erase(key);
  }
  if (profile_enabled_) {
    prof_.evict_layer_us += _prof_now_us() - _el_t0;
    prof_.evict_layer_n += 1;
  }
}

void ExpertDispatcher::SetProfilePhase(int phase) {
  profile_phase_.store(phase);
}

void ExpertDispatcher::ResetProfile() {
  prof_.verify_fetch_n = 0; prof_.verify_fetch_us = 0; prof_.verify_fetch_bytes = 0;
  prof_.draft_fetch_n = 0; prof_.draft_fetch_us = 0; prof_.draft_fetch_bytes = 0;
  prof_.evict_n = 0; prof_.evict_us = 0;
  prof_.overload_wait_n = 0; prof_.overload_wait_us = 0;
  prof_.enqueue_wait_n = 0; prof_.enqueue_wait_us = 0;
  prof_.forward_n = 0; prof_.forward_us = 0;
  prof_.merge_n = 0; prof_.merge_us = 0;
  prof_.evict_layer_n = 0; prof_.evict_layer_us = 0;
  prof_.dispatch_n = 0; prof_.dispatch_us = 0;
}

std::map<std::string, int64_t> ExpertDispatcher::DumpProfile() {
  return {
      {"verify_fetch_n", prof_.verify_fetch_n.load()},
      {"verify_fetch_us", prof_.verify_fetch_us.load()},
      {"verify_fetch_bytes", prof_.verify_fetch_bytes.load()},
      {"draft_fetch_n", prof_.draft_fetch_n.load()},
      {"draft_fetch_us", prof_.draft_fetch_us.load()},
      {"draft_fetch_bytes", prof_.draft_fetch_bytes.load()},
      {"evict_n", prof_.evict_n.load()},
      {"evict_us", prof_.evict_us.load()},
      {"overload_wait_n", prof_.overload_wait_n.load()},
      {"overload_wait_us", prof_.overload_wait_us.load()},
      {"enqueue_wait_n", prof_.enqueue_wait_n.load()},
      {"enqueue_wait_us", prof_.enqueue_wait_us.load()},
      {"forward_n", prof_.forward_n.load()},
      {"forward_us", prof_.forward_us.load()},
      {"merge_n", prof_.merge_n.load()},
      {"merge_us", prof_.merge_us.load()},
      {"evict_layer_n", prof_.evict_layer_n.load()},
      {"evict_layer_us", prof_.evict_layer_us.load()},
      {"dispatch_n", prof_.dispatch_n.load()},
      {"dispatch_us", prof_.dispatch_us.load()},
  };
}

void ExpertDispatcher::SetPinned(int layer_idx,
                                const std::vector<int>& expert_ids,
                                int gpu_id) {
  std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
  // Drop this layer's old pins, then install the new kept-N set.
  for (auto it = pinned_[gpu_id].begin(); it != pinned_[gpu_id].end();) {
    if (static_cast<int>(*it >> 32) == layer_idx) {
      it = pinned_[gpu_id].erase(it);
    } else {
      ++it;
    }
  }
  for (int e : expert_ids) {
    uint64_t key = (static_cast<uint64_t>(layer_idx) << 32) +
                   static_cast<uint64_t>(e);
    pinned_[gpu_id].insert(key);
  }
}

void ExpertDispatcher::ClearPinned(int gpu_id) {
  std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
  pinned_[gpu_id].clear();
}

void ExpertDispatcher::NotifyFetchStart() {
  for (int i = 0; i < kNumDevices(); ++i) {
    // std::unique_lock<std::mutex> lock(input_mutex_[i]);
    input_queue_[i].NotifyAll();
  }
}

void ExpertDispatcher::ClearExpertCacheCounts() {
  for (auto& expert : experts_) {
    for (auto& expert_node : expert) {
      if (expert_node->node == nullptr) {
        continue;
      }
      expert_node->node->incache_visit_count = 0;
    }
  }
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

ExpertNodePtr ExpertDispatcher::FindExpertEvict(int gpu_id) {
  uint64_t min_visit_count = INT_MAX;
  ExpertNodePtr evict_expert_node = nullptr;

  for (auto& key : cached_experts_[gpu_id]) {
    // aug_spec / specmoe_pin_plan.md: never evict a pinned expert (the SpecMoE
    // kept-N draft set). FindExpertEvict is the batch_size==1 (draft-step)
    // eviction path, so skipping pinned here keeps the kept-N resident across
    // draft steps → the draft reads them at 0 PCIe.
    if (pinned_[gpu_id].count(key) > 0) continue;
    auto layer_idx = key >> 32;
    auto expert_idx = key & 0xFFFFFFFF;
    auto node = experts_[expert_idx][layer_idx]->node;
    if (node == nullptr) continue;
    if (node->device.is_cuda() && node->incache_visit_count < min_visit_count &&
        node->mutex.try_lock()) {
      evict_expert_node = experts_[expert_idx][layer_idx];
      min_visit_count = node->incache_visit_count;
      node->mutex.unlock();
    }
  }
  return evict_expert_node;
}

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  while (!main_thread_stop_flag_.load()) {
    // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
    // if (cache_ == nullptr) {
    //   auto cache_limit =
    //   kDeviceMemoryPool->GetSparseCacheLimit(torch::Device(torch::kCUDA,
    //   gpu_id));
    //   // get any one expert size
    //   auto num_layers = experts_[0].size();
    //   auto num_experts = experts_.size();
    //   auto expert_node = experts_[num_layers-1][num_experts-1];

    //   int cache_capacity = cache_limit / expert_node->node->byte_size;
    //   cache_capacity_ = cache_capacity;
    // }
    // std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    // input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty();
    // });

    // CallArgs args = std::move(input_queue_[gpu_id].front());
    // input_queue_[gpu_id].pop_front();

    // lock.unlock();
    CallArgs args;
    input_queue_[gpu_id].Pop(args);

    auto device = CUDA_DEVICE(gpu_id);
    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
    int64_t layer_idx = args.layer_idx;
    int64_t expert_idx = args.expert_idx;
    int64_t batch_size = hidden_states_.size(0);

    auto expert_node = experts_[expert_idx][layer_idx];
    bool cache_hit = expert_node->node->device.is_cuda();

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: gpu_id " << gpu_id
    //           << " layer_idx " << layer_idx << " expert_idx " << expert_idx
    //           << " cache_hit " << cache_hit << " node "
    //           << expert_node->node->device.str() << std::endl;
    DLOG_DEBUG("ExpertDispatcher::GPUFetchFunc: gpu_id ", gpu_id, " layer_idx ",
               layer_idx, " expert_idx ", expert_idx, "cache_hit ", cache_hit,
               "cache_size ", cache_sizes_[gpu_id], " incache count ",
               cached_experts_[gpu_id].size());

    if (!cache_hit && cache_sizes_[gpu_id] < expert_node->node->byte_size) {
      if (batch_size > 1) {
        // force fetch to GPU regardless of cache size, only for prefill
        // only one extra cache slot for prefill
        DLOG_DEBUG("overloading expert cache: gpu_id ", gpu_id, " cache size ",
                   cache_sizes_[gpu_id], " incache count ",
                   cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                   " expert_idx ", expert_idx);
        // gpu_overload_[gpu_id].wait_and_set(false, true);
        // busy wait for cache to be available
        int64_t _ow_t0 = profile_enabled_ ? _prof_now_us() : 0;
        while (gpu_overload_[gpu_id]) {
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (profile_enabled_) {
          prof_.overload_wait_us += _prof_now_us() - _ow_t0;
          prof_.overload_wait_n += 1;
        }
        gpu_overload_[gpu_id] = true;
      } else {
        // find the expert in gpu and min incache_visit_count
        ExpertNodePtr evict_expert_node = FindExpertEvict(gpu_id);
        if (evict_expert_node == nullptr) {
          // wait for notification that cache is available
          DLOG_WARN(
              "All cached expert locked, waiting for cache to be available. "
              "gpu_id ",
              gpu_id, " cache size ", cache_sizes_[gpu_id], " incache count ",
              cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
              " expert_idx ", expert_idx);
        }
        // aug_spec deadlock fix: FindExpertEvict above is lock-free, so a
        // GPUExecFunc->OutputFunc notify_all (which frees an expert) can fire
        // between the null-check and a plain cache_cv_.wait(lock) -> lost
        // wakeup -> the fetch thread blocks forever (observed: SpecMoE offload
        // hangs mid-run under heavy draft-dispatch cache churn). Use a *timed*
        // wait in a retry loop instead: every 2 ms we re-poll FindExpertEvict,
        // so even a missed notify cannot hang -- an expert becomes evictable as
        // soon as any in-flight exec finishes and unlocks its node->mutex. The
        // loop also replaces the old single-retry that could fall through with
        // a null node into the DLOG_FATAL below.
        while (evict_expert_node == nullptr) {
          {
            std::unique_lock<std::mutex> lock(cache_mutex_[gpu_id]);
            cache_cv_[gpu_id].wait_for(lock, std::chrono::milliseconds(2));
          }
          evict_expert_node = FindExpertEvict(gpu_id);
        }
        // auto num_layers = experts_[0].size();
        // auto num_experts = experts_.size();

        // for (size_t i = 0; i < num_experts; ++i) {
        //   for (size_t j = 0; j < num_layers; ++j) {
        // auto node = experts_[i][j]->node;
        // if (node == nullptr) {
        //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
        //   //           << " layer_idx " << j << " expert_idx " << i <<
        //   //           std::endl;
        //   continue;
        // }
        // if (node->device.is_cuda() &&
        //     node->incache_visit_count < min_visit_count &&
        //     node->mutex.try_lock()) {
        //   evict_node = node;
        //   min_visit_count = node->incache_visit_count;
        //   node->mutex.unlock();
        //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
        //   //           << evict_node->device.str() << " incache_visit_count "
        //   //           << min_visit_count << std::endl;
        // }
        //   }
        // }
        DLOG_FATAL_IF(
            evict_expert_node == nullptr,
            "ExpertDispatcher::GPUFetchFunc: evict_node is nullptr, gpu_id",
            gpu_id, "cache size", cache_sizes_[gpu_id], "in cache count",
            cached_experts_[gpu_id].size());

        DLOG_DEBUG("evicting expert: gpu_id ", gpu_id, " cache size ",
                   cache_sizes_[gpu_id], " incache count ",
                   cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                   " expert_idx ", expert_idx);

        auto evict_node = evict_expert_node->node;
        int64_t _ev_t0 = profile_enabled_ ? _prof_now_us() : 0;
        evict_node->SetDevice(evict_node->default_host);
        if (profile_enabled_) {
          prof_.evict_us += _prof_now_us() - _ev_t0;
          prof_.evict_n += 1;
        }
        cache_sizes_[gpu_id] += evict_node->byte_size;
        int64_t evict_layer_idx = evict_expert_node->layer_idx;
        int64_t evict_expert_idx = evict_expert_node->expert_idx;

        // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
        uint64_t evict_key = (evict_layer_idx << 32) + evict_expert_idx;
        auto it = cached_experts_[gpu_id].find(evict_key);
        if (it != cached_experts_[gpu_id].end()) {
          cached_experts_[gpu_id].erase(it);
        } else {
          DLOG_FATAL(
              "ExpertDispatcher::GPUFetchFunc: evict_key not found. layer_idx ",
              evict_layer_idx, " expert_idx ", evict_expert_idx);
        }
      }
    }

    if (!gpu_overload_[gpu_id]) {
      cache_sizes_[gpu_id] -= expert_node->node->byte_size;
      uint64_t key = (layer_idx << 32) + expert_idx;
      cached_experts_[gpu_id].insert(key);
    }

    int64_t _ft_t0 = profile_enabled_ ? _prof_now_us() : 0;
    expert_node->node->SetDevice(device, true, stream);
    if (profile_enabled_ && !cache_hit) {
      // Real H2D fetch (cache miss). Tag by phase so SpecMoE's draft re-fetches
      // (kept-N evicted during verify) are separable from verify fetches.
      int64_t _us = _prof_now_us() - _ft_t0;
      int64_t _by = expert_node->node->byte_size;
      if (profile_phase_.load() == 1) {
        prof_.draft_fetch_n += 1; prof_.draft_fetch_us += _us;
        prof_.draft_fetch_bytes += _by;
      } else {
        prof_.verify_fetch_n += 1; prof_.verify_fetch_us += _us;
        prof_.verify_fetch_bytes += _by;
      }
    }
    expert_node->node->incache_visit_count += 1;
    expert_node->SetTensorsFromBlob(device);
    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
    //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
    //           << expert_idx << " node "
    //           << expert_node->node->device.str() << std::endl;

    // int expert_type = expert_type_;
    // torch::Tensor input;
    // auto token_indices =
    //     router_mask_.index({"...", expert_idx}).to(torch::kBool);
    // switch (expert_type) {
    //   case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
    //   case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
    //   case NLLB_MOE_DENSE_ACT_DENSE:
    //   case FSGPT_MOE_DENSE_ACT_DENSE:
    //   case MIXTRAL_MOE_DENSE_ACT_DENSE:
    //   case DEEPSEEK_MOE_DENSE_ACT_DENSE:
    //     input =
    //         hidden_states_.index({token_indices}).to(expert_node->node->device);
    //     break;
    //   default:
    //     DLOG_FATAL("ExpertDispatcher::expert_type: unknown expert type ",
    //                expert_type);
    // }

    // DLOG_TRACE("ExpertDispatcher::GPUFetchFunc gpu_id ", gpu_id, "layer_idx
    // ",
    //            layer_idx, "expert_idx ", expert_idx, "input ",
    //            input.device().str(), "node ",
    //            expert_node->node->device.str());
    {
      ExecArgs exec_args;
      // exec_args.hidden_states = std::move(input);
      exec_args.expert_node = expert_node;
      exec_args.out_gpu_id = original_device.index();
      exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
      exec_args.evict = gpu_overload_[gpu_id];
      exec_args.hit = cache_hit;
      // std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
      // exec_queue_[gpu_id].emplace_back(std::move(exec_args));
      exec_queue_[gpu_id].Push(exec_args);
    }
    // exec_cv_[gpu_id].notify_all();
  }

  cudaStreamDestroy(stream);
}

void ExpertDispatcher::GPUExecFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  while (!main_thread_stop_flag_.load()) {
    // std::unique_lock<std::mutex> lock(exec_mutex_[gpu_id]);
    // exec_cv_[gpu_id].wait(lock, [&] { return !exec_queue_[gpu_id].empty();
    // });

    // ExecArgs args = std::move(exec_queue_[gpu_id].front());
    // exec_queue_[gpu_id].pop_front();

    // lock.unlock();

    ExecArgs args;
    exec_queue_[gpu_id].Pop(args);

    if (args.expert_node == nullptr) {
      continue;
    }

    int64_t batch_size = hidden_states_.size(0);
    auto device = CUDA_DEVICE(gpu_id);
    auto expert_idx = args.expert_node->expert_idx;

    auto token_mask = router_mask_.index({"...", expert_idx});
    torch::Tensor input = (batch_size == 1)
                              ? hidden_states_.to(device)
                              : hidden_states_.index({token_mask}).to(device);

    // args.hidden_states = std::move(input);
    // assert(args.hidden_states.sum().to(torch::kCPU).item<float>() != 0);
    // at::InferenceMode infer_guard(true);

    // // prepare jit input vector
    // std::vector<torch::jit::IValue> jit_inputs;
    // jit_inputs.push_back(input);

    // cudaDeviceSynchronize();

    modules_[gpu_id]->SetTensorsFromIds(args.expert_node->node->tensor_ids);

    // random int [0,8)
    // int rnd = std::rand() % kNumDevices();
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream, gpu_id);
    c10::cuda::CUDAStreamGuard guard(torch_stream);
    // auto start = TIME_NOW;
    // c10::cuda::CUDAStreamGuard guard(stream);

    // auto* expert_module = args.expert_node->module;
    // int expert_type = expert_type_;
    // cudaStreamSynchronize(stream);  // make sure the input is ready

    int64_t _fw_t0 = profile_enabled_ ? _prof_now_us() : 0;
    auto output = modules_[gpu_id]->forward(input, stream);
    if (profile_enabled_) {
      prof_.forward_us += _prof_now_us() - _fw_t0;
      prof_.forward_n += 1;
    }
    OutputFunc(args, output, token_mask, gpu_id);
  }

  cudaStreamDestroy(stream);
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  torch::Tensor token_mask, int gpu_id) {
  auto output_device =
      (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
  torch::Tensor output_tensor = output.to(output_device).to(torch::kFloat32);

  DLOG_TRACE("ExpertDispatcher::OutputFunc: output_tensor ",
             output_tensor.sizes().vec(), "(", output_tensor.device().str(),
             ")");

  // args.expert_node->node->mutex.unlock();
  int64_t expert_idx = args.expert_node->expert_idx;
  int64_t layer_idx = args.expert_node->layer_idx;
  int64_t batch_size = hidden_states_.size(0);

  args.expert_node->node->mutex.unlock();
  if (args.evict) {
    // pop out overloaded expert such that cache is not polluted
    args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                      true, nullptr);
    // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
    // uint64_t key = (layer_idx << 32) + expert_idx;
    // auto it = cached_experts_[gpu_id].find(key);
    // if (it != cached_experts_[gpu_id].end()) {
    //   cached_experts_[gpu_id].erase(it);
    // } else {
    //   DLOG_FATAL(
    //       "ExpertDispatcher::OutputFunc: expert not found in cache. gpu_id",
    //       gpu_id, "layer_idx ", layer_idx, "expert_idx ", expert_idx);
    // }
    // cache_sizes_[gpu_id] += args.expert_node->node->byte_size;
    DLOG_DEBUG("pop out overloaded expert cache_sizes_[gpu_id] ",
               cache_sizes_[gpu_id], "gpu_id ", gpu_id, "layer_idx ", layer_idx,
               "expert_idx ", expert_idx);
    // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
    // gpu_overload_[gpu_id].set_and_wake(true);
    gpu_overload_[gpu_id] = false;
  }
  cache_cv_[gpu_id].notify_all();

  // if (args.evict) {
  //   args.expert_node->node->SetDevice(args.expert_node->node->default_host,
  //                                     true, nullptr);
  //   {
  //     std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
  //     gpu_overload_[gpu_id] = false;
  //   }
  // }

  if (batch_size == 1) {
    final_hidden_states_.add_(
        output_tensor *
        router_weight_.index({torch::indexing::Slice(), expert_idx}));
  } else {
    auto token_indices = torch::nonzero(token_mask).squeeze(1);
    auto weights = router_weight_.index({token_mask, expert_idx}).unsqueeze(1);
    auto weighted_output = output_tensor * weights;
    final_hidden_states_.index_add_(0, token_indices, weighted_output);
  }
  // {
  //   std::lock_guard<std::mutex> lock(output_mutex_);
  //   output_queue_.emplace_back(std::move(output_tensor),
  //                              args.expert_node->layer_idx,
  //                              args.expert_node->expert_idx, args.hit);
  //   DLOG_TRACE("ExpertDispatcher::OutputFunc: output_queue_",
  //              output_queue_.size(), "output",
  //              std::get<0>(output_queue_.back()).device().str(), "evict",
  //              args.evict, "(", args.expert_node->layer_idx,
  //              args.expert_node->expert_idx, gpu_id, args.hit, ")");
  // }

  // stream.synchronize();
  pending_.fetch_sub(1);
  if (pending_.load() == 0) {
    pending_cv_.notify_all();
  }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
  // int wait_count = 0;

  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });

  num_enqueued_.store(0);
  std::vector<CallResult> output_queue;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue.swap(output_queue_);
  }

  return output_queue;
}

torch::Tensor ExpertDispatcher::WaitHiddenStates() {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });
  num_enqueued_.store(0);
  return final_hidden_states_;
}

void ExpertDispatcher::SetInputs(const torch::Tensor& hidden_states,
                                 const torch::Tensor& router_mask,
                                 const torch::Tensor& router_weight) {
  int device = at::cuda::current_device();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(CUDA_DEVICE(device));
  hidden_states_ = hidden_states;
  router_mask_ = router_mask;
  router_weight_ = router_weight;  // this can be float32
  final_hidden_states_ = torch::zeros_like(hidden_states, options);
}
