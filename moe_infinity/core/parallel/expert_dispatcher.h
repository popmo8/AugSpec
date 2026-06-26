// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <torch/extension.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
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

  // aug_spec: run K merged dense draft experts through the SAME MoEMLP::forward
  // kernel the archer dispatch uses, so the merged-expert draft and the SpecMoE
  // (substitute) draft share one expert-execution engine — the comparison then
  // isolates the algorithm, not the kernel. `merged` holds K experts, each a
  // {gate, up, down} GPU-tensor list in tensor-id order (as MergeExpertsLocal
  // returns). `weight` is [T, K]: token t routes to cluster k with this combine
  // weight (0 = not selected). Synchronous (merged are resident, no fetch); the
  // archer worker threads are idle during the draft so modules_[gpu_id] is free.
  torch::Tensor DispatchMergedLocal(
      torch::Tensor hidden_states, torch::Tensor weight,
      const std::vector<std::vector<torch::Tensor>>& merged, int gpu_id);

  // aug_spec v1: batched bmm over a set of resident experts (topm merged or
  // specmoe pinned kept-N) — unifies the resident-expert draft compute inside
  // the engine. Weights arrive PRE-STACKED (Python memoises the stack once per
  // cycle, so there is no per-call re-stack): gw/uw are [E, D, I], dw is
  // [E, I, D]; `weight` is [T, E] routing (0 = not selected). The op sequence is
  // identical to the Python torch.bmm path, so results match bit-for-bit. (v2
  // will swap the body for a CUTLASS grouped GEMM behind this same signature.)
  torch::Tensor DispatchBmm(torch::Tensor hidden, torch::Tensor gw,
                            torch::Tensor uw, torch::Tensor dw,
                            torch::Tensor weight, int gpu_id);

  // aug_spec profiling (AUG_PROFILE=1; zero cost otherwise). SetProfilePhase
  // tags fetches as verify(0) / draft(1) so SpecMoE's draft re-fetches are
  // separable. DumpProfile returns cumulative counters (times in µs); used to
  // see where each cycle spends time and what serialises vs overlaps.
  void SetProfilePhase(int phase);
  std::map<std::string, int64_t> DumpProfile();
  void ResetProfile();

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

  // aug_spec profiling counters (times in µs, all atomic — touched by the
  // fetch/exec worker threads and the main dispatch thread).
  struct ProfileCounters {
    std::atomic<int64_t> verify_fetch_n{0}, verify_fetch_us{0},
        verify_fetch_bytes{0};
    std::atomic<int64_t> draft_fetch_n{0}, draft_fetch_us{0},
        draft_fetch_bytes{0};
    std::atomic<int64_t> evict_n{0}, evict_us{0};
    std::atomic<int64_t> overload_wait_n{0}, overload_wait_us{0};
    std::atomic<int64_t> enqueue_wait_n{0}, enqueue_wait_us{0};
    std::atomic<int64_t> forward_n{0}, forward_us{0};
    std::atomic<int64_t> merge_n{0}, merge_us{0};
    std::atomic<int64_t> evict_layer_n{0}, evict_layer_us{0};
    std::atomic<int64_t> dispatch_n{0}, dispatch_us{0};
  };
  ProfileCounters prof_;
  std::atomic<int> profile_phase_{0};   // 0 = verify, 1 = draft
  bool profile_enabled_ = false;

  // aug_spec (AUG_NO_OVERLOAD): route batch>1 cache-full fetches through the
  // normal FindExpertEvict path (evict one LFU non-pinned per fetch, pinned
  // kept-N skipped) instead of the single-slot serialised "overload" borrow.
  // Restores prefetch depth and makes pinning actually keep kept-N resident.
  bool no_overload_ = false;
};

#define SET_TENSORS_AND_MODULE_FROM_BLOB(cls, module, node, device, \
                                         jit_module)                \
  do {                                                              \
    reinterpret_cast<cls*>(module)->SetTensorsFromBlob(             \
        node->device_memory_ptr, node->tensor_ids, device);         \
    reinterpret_cast<cls*>(module)->SetModuleFromBlob(jit_module);  \
  } while (0)
