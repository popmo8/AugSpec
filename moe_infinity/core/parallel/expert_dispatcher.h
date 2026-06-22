// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/sync.h"
#include "base/noncopyable.h"
#include "base/thread.h"
#include "utils/threadsafe_queue.h"
#include "expert_module.h"

enum MUTEX_TYPE {
  INPUT_MUTEX = 0,
  OUTPUT_MUTEX = 1,
  EXEC_MUTEX = 2,
  PENDING_MUTEX = 3
};

class ExpertDispatcher : public base::noncopyable {
 public:
  typedef struct {
    int layer_idx = -1;
    int expert_idx = -1;
    int gpu_id = -1;
    bool remote = false;
  } CallArgs;
  typedef struct {
    torch::Tensor hidden_states =
        torch::empty({0});  // shallow copy, real tensor in python code
    ExpertNodePtr expert_node = nullptr;
    int out_gpu_id = -1;
    torch::ScalarType out_dtype = torch::kFloat32;
    bool evict = false;
    bool hit = false;
  } ExecArgs;
  typedef std::tuple<torch::Tensor, int, int, int> CallResult;

 public:
  explicit ExpertDispatcher(int num_experts, int num_layers, int dtype,
                            int expert_type, int num_threads = 1);
  ~ExpertDispatcher() {
    main_thread_stop_flag_.store(true);
    for (auto& thread : threads_) {
      thread->join();
    }

    // for (auto& stream : fetch_streams_) {
    //   cudaStreamDestroy(stream);
    // }
    for (auto& stream : exec_streams_) {
      cudaStreamDestroy(stream);
    }
    // for (auto& stream : out_streams_) {
    //   cudaStreamDestroy(stream);
    // }
  }

  void SetInputs(const torch::Tensor& hidden_states,
                 const torch::Tensor& router_mask,
                 const torch::Tensor& router_weight);

  void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1,
                     bool remote = false);
  void NotifyFetchStart();

  void RegisterExpert(int layer_idx, int expert_idx,
                      const std::vector<std::uint32_t>& tensor_ids,
                      std::string jit_path);
  void ClearExpertCacheCounts();
  void SetExpectedQueue(int expected_pending = 0) {
    pending_.store(expected_pending);
  }

  std::vector<CallResult> WaitExpert() { return Wait(); }
  torch::Tensor WaitHiddenStates();

  // aug_spec / M9b: read-only access to an expert's GPU-resident weight
  // tensors (the ones a recent verify fetched). Returns the weight tensors
  // (gate_proj, up_proj, down_proj order for Qwen3) only when the expert is
  // currently resident on `gpu_id`; returns an empty vector otherwise so the
  // Python caller can detect a miss. Must be called in a dispatch-quiescent
  // window (between wait_expert and the next dispatch) — the background fetch
  // thread is then idle on an empty input queue, so no eviction races.
  std::vector<torch::Tensor> GetResidentExpertWeights(int layer_idx,
                                                      int expert_idx,
                                                      int gpu_id);

  // aug_spec / M9b: weighted merge of `expert_ids` (with `weights`) on layer
  // `layer_idx` into a single dense expert, returned as {gate_proj, up_proj,
  // down_proj} GPU tensors in the model weight dtype, fp32-accumulated.
  // GPU-resident sources (a recent verify fetched them) are read in place with
  // zero PCIe; non-resident sources are copied host->GPU transiently just for
  // this merge and freed on return — the dispatcher cache is never mutated, so
  // no eviction bookkeeping / races. Accumulation order follows `expert_ids`
  // (caller passes ascending nonzero indices to mirror the CPU-merge order).
  std::vector<torch::Tensor> MergeExpertsLocal(
      int layer_idx, const std::vector<int>& expert_ids,
      const std::vector<double>& weights, int gpu_id);

  // aug_spec / verify_merge_plan.md P1: evict every GPU-resident expert on
  // `gpu_id` back to host and reset the sparse-cache budget to full. Cheap —
  // the host copies are the offload source, so this just frees the GPU mirrors
  // (no D2H). Called at draft start (the merged-dense draft never touches the
  // archer cache, so it is idle) to make the budget room phase-exclusive with
  // the merged experts (§1.4). Must be called dispatch-quiescent (between
  // verify and draft), like the read/merge methods above.
  void FlushCache(int gpu_id);

  // aug_spec / verify_merge_plan.md P2+P3: evict every GPU-resident expert of
  // layer `layer_idx` back to host. Called right after on_verify_layer merges
  // the layer (experts still resident, dispatch-quiescent between layers). By
  // freeing each layer once verify+merge are done with it, the sparse cache
  // never fills → the overload evict-after-use path never triggers → the
  // per-layer merge reads resident experts with no concurrent-eviction race
  // (the P3 crash), and the footprint stays ~1 layer (low peak). Caller must
  // sync the merge's GPU reads before this frees the source memory.
  void EvictLayer(int layer_idx, int gpu_id);

  // aug_spec / specmoe_pin_plan.md: mark layer `layer_idx`'s `expert_ids` as
  // pinned (the SpecMoE kept-N draft set) so FindExpertEvict never evicts them.
  // SetPinned replaces the whole pinned set for that layer; ClearPinned drops
  // all pins (the experts then evict normally). The kept-N get cached by the
  // draft's own dispatch (batch_size==1 → FindExpertEvict path, which now skips
  // pinned to make room from the non-pinned verify experts). Quiescent-window
  // calls (refresh, between verify and draft).
  void SetPinned(int layer_idx, const std::vector<int>& expert_ids, int gpu_id);
  void ClearPinned(int gpu_id);

 private:
  void Enqueue(CallArgs& args);
  std::vector<CallResult> Wait();
  void Start() { start_ = true; }

  void GPUFetchFunc(int gpu_id);
  void GPUExecFunc(int gpu_id);

  // void GPUThreadFunc(int gpu_id);

  void OutputFunc(ExecArgs args, torch::Tensor output, torch::Tensor token_mask,
                  int gpu_id);

  ExpertNodePtr FindExpertEvict(int gpu_id);

 private:
  std::vector<std::unique_ptr<base::Thread>> threads_;
  std::mutex mutex_;
  // std::vector<std::deque<CallArgs>> input_queue_;
  std::vector<ThreadSafeQueue<CallArgs>> input_queue_;
  // std::vector<std::deque<ExecArgs>> exec_queue_;
  std::vector<ThreadSafeQueue<ExecArgs>> exec_queue_;
  std::vector<CallResult> output_queue_;
  std::vector<std::vector<ExpertNodePtr>> experts_;
  std::atomic<size_t> num_enqueued_;
  bool start_;
  int expert_type_;
  int dtype_;
  int num_experts_;
  std::atomic<bool> main_thread_stop_flag_;

  std::atomic<size_t> pending_;

  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  // std::vector<std::mutex> input_mutex_;
  // std::vector<std::mutex> exec_mutex_;
  // std::vector<std::condition_variable> input_cv_;
  // std::vector<std::condition_variable> exec_cv_;

  std::vector<std::mutex> cache_mutex_;
  std::vector<std::condition_variable> cache_cv_;

  std::mutex output_mutex_;
  // std::mutex exec_mutex_;
  // std::mutex gpu_overload_mutex_;

  std::vector<cudaStream_t> exec_streams_;

  std::vector<bool> gpu_overload_;

  torch::Tensor hidden_states_;
  torch::Tensor final_hidden_states_;
  torch::Tensor router_mask_;
  torch::Tensor router_weight_;

  std::vector<int64_t> cache_sizes_;
  std::vector<std::unordered_set<uint64_t>> cached_experts_;
  std::vector<std::unordered_set<uint64_t>> pinned_;   // specmoe kept-N (no evict)

  int cache_capacity_ = 0;

  std::vector<MoEMLP*> modules_;
};

#define SET_TENSORS_AND_MODULE_FROM_BLOB(cls, module, node, device, \
                                         jit_module)                \
  do {                                                              \
    reinterpret_cast<cls*>(module)->SetTensorsFromBlob(             \
        node->device_memory_ptr, node->tensor_ids, device);         \
    reinterpret_cast<cls*>(module)->SetModuleFromBlob(jit_module);  \
  } while (0)
