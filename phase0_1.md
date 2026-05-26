# phase0_1.md — Phase 0-3 實作紀錄(到目前為止)

> 配套文件:
> - [README.md](README.md) — repo 結構
> - [PROGRESS.md](PROGRESS.md) — 整體實驗進度
> - [next_step.md](next_step.md) — Phase B offload 架構決策
> - [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 逐 phase 實作 spec
>
> 這份文件補:**實際把 Phase 0-3 跑起來時遇到的細節、bug 跟解法**。
> 從 2026-05-23 開始,涵蓋到 2026-05-25。

---

## 1. 結果摘要

| Phase | 結果 |
|---|---|
| **0** Pre-flight sanity | ✅ 全 PASS(4/4 questions) |
| **1** loader + CLI | ✅ `backend: offload` YAML key 通,RunConfig 解析 + run_experiment 分支 |
| **2** Adapter `_route_offload` | ✅ Mixtral + Qwen3 都加好分支 |
| **2.5** `build_weighted_avg_offload` | ✅ Mixtral + Qwen3,Test A 確認 bit-identical |
| **3** Smoke offload | ⚠️ **Qwen3 通,Mixtral 仍壞**(moe_infinity Mixtral 路徑 bug) |

### Smoke 數字

| | MAT | AccRate | TPS | wall | 對位 |
|---|---|---|---|---|---|
| HF Mixtral(`count` draft) | 2.87 | 0.64 | 9.2 | 46s | baseline ✓ |
| Offload **Qwen3**(`count`) | 1.44 | 0.152 | 0.87 | 1210s | **aug_spec 端 OK ✓** |
| Offload Mixtral(`count`) | 1.15 | 0.052 | 0.32 | 1210s | ❌ moe_infinity Mixtral 路徑壞 |

Qwen3 跟 HF baseline 同數量級(PROGRESS.md `qwen3_count` full run AccRate=0.086,smoke 0.152 在小樣本變異內) → **aug_spec 全 pipeline 在能 work 的模型上驗通**。

---

## 2. aug_spec 端的程式碼變更(全部已 commit-ready)

### 2.1 程式碼骨架

| 檔案 | 變更 |
|---|---|
| [`runtime/loader.py`](src/aug_spec/runtime/loader.py) | 新增 `load_offload()` 回傳 `(hf_model, tokenizer, moe, cpu_source)` 四件套;新增 `_install_rotary_device_patches()` 修 Mixtral / Qwen3 rotary 的 CPU/CUDA position_ids 不對齊 bug |
| [`cli.py`](src/aug_spec/cli.py) | `RunConfig` 加 `backend / offload_path / offload_device_memory_ratio / offload_cache_policy`;`run_experiment` 依 backend dispatch;成功後 offload 模式 `os._exit(0)` 跳過 moe_infinity background thread 卡住 |
| [`controller.py`](src/aug_spec/controller.py) | 加 `cpu_source` + 預解析的 `cpu_blocks: Dict[int, nn.Module]`;`refresh / prepopulate` 用 kwarg 傳下去 |
| [`adapters/base.py`](src/aug_spec/adapters/base.py) | `build_weighted_avg(block, weights, *, cpu_block=None)` |
| [`adapters/mixtral.py`](src/aug_spec/adapters/mixtral.py) | `_route_hf` / `_route_offload` 分支;`build_weighted_avg` 加 `cpu_block` 路徑(stream CPU→GPU 累加 fp32);`make_averaged_forward` 把 `cpu_block` 傳給 `lazy_build` |
| [`adapters/qwen3.py`](src/aug_spec/adapters/qwen3.py) | 同 mixtral,offload 走 `block.lib.topk_softmax` CUDA kernel + `dispatch_local` |
| [`runtime/specbench.py`](src/aug_spec/runtime/specbench.py) | 加 `moe_wrapper` kwarg,每題 generate 前呼 `_configure_hook(input_ids)`;`inference_mode` 改 `no_grad` |
| [`drafts/base.py`](src/aug_spec/drafts/base.py) | `refresh / prepopulate / lazy_build` 簽名加 `cpu_blocks` / `cpu_block` kwarg |
| [`drafts/uniform.py`](src/aug_spec/drafts/uniform.py), [`prefill_count.py`](src/aug_spec/drafts/prefill_count.py) | `lazy_build` forward `cpu_block` 給 adapter |

### 2.2 新增的測試 / 工具腳本

| 檔案 | 用途 |
|---|---|
| [`tests/sanity_offload.py`](tests/sanity_offload.py) | Phase 0 pre-flight(4 個技術假設驗證) |
| [`tests/debug_offload_correctness.py`](tests/debug_offload_correctness.py) | Test A:`build_weighted_avg_offload` vs 手動參考 — bit-identical 驗證(PASS) |
| [`tests/debug_offload_logits.py`](tests/debug_offload_logits.py) | Test B:offload model forward vs HF cpu_source forward — argmax / top-K overlap |
| [`scripts/smoke_phase1to3.sh`](scripts/smoke_phase1to3.sh) | import sanity + HF regression + offload smoke 三步串接 |
| [`scripts/debug_offload_logits.sh`](scripts/debug_offload_logits.sh) | logits 對照 wrapper(支援 `--model mixtral|qwen3`) |
| [`scripts/debug_moe_infinity_demo.sh`](scripts/debug_moe_infinity_demo.sh) | 跑 moe_infinity 自家 Qwen3 demo(驗他們 framework 整體 OK 不是 aug_spec 問題) |
| [`scripts/test_smallk_mixtral.sh`](scripts/test_smallk_mixtral.sh) | (debug)Mixtral 強制走 Small-K GEMM 後 logits 對照 |
| [`configs/_smoke_offload.yaml`](configs/_smoke_offload.yaml) | Mixtral + offload + topm_count(現階段壞) |
| [`configs/_smoke_offload_count.yaml`](configs/_smoke_offload_count.yaml) | Mixtral + offload + count(A/B 用) |
| [`configs/_smoke_offload_qwen3.yaml`](configs/_smoke_offload_qwen3.yaml) | **Qwen3 + offload + count(目前驗通的設定)** |

### 2.3 修進 moe_infinity 的 patch(C++ 端 + .pth)

| 位置 | 變更 |
|---|---|
| [`moe_infinity/core/parallel/expert_module.cpp`](moe_infinity/core/parallel/expert_module.cpp) `kMaxTokens` | 128 → 2048(原本 spec-bench prompt > 128 就 FATAL) |
| `.venv/lib/python3.10/site-packages/__editable___moe_infinity_0_0_1_finder.py` | `sys.meta_path.append()` → `sys.meta_path.insert(0, ...)`(see §4.4) |

---

## 3. 修掉的 Bug(全部納入 codebase)

| # | Bug | 症狀 | 修法 | 位置 |
|---|---|---|---|---|
| 1 | `inference_mode` 跟 moe_infinity C++ `index_add_` 不相容 | C++ crash:"Inplace update to inference tensor" | 全改 `torch.no_grad()` | `specbench.py`、`sanity_offload.py` |
| 2 | `MixtralRotaryEmbedding.forward` 不對齊 `position_ids` 跟 `x` device | spec decoding assistant generate 把 position_ids 留在 CPU,inv_freq 在 cuda → bmm 炸 | `_install_rotary_device_patches()` 在 `load_offload()` 安裝 monkey-patch | `loader.py` |
| 3 | 同 #2 但對 Qwen3 | 換 Qwen3 後 `Qwen3MoeRotaryEmbedding.forward` 同樣的 bug | patch 改成多 family 通用 | `loader.py` |
| 4 | `block.gate.weight.device == CPU`(moe_infinity 把 dense param 移到 CPU)→ merged expert 建在 CPU | `_run_dense_expert` 用 CPU avg 跟 GPU hs_flat 算 → mat2 on cpu 錯誤 | offload mode 強制 `device = torch.device("cuda", current_device())` | `mixtral.py`、`qwen3.py` `build_weighted_avg` |
| 5 | C++ `expert_module.cpp` hardcode `kMaxTokens=128` | prompt > 128 tokens `[FATAL] batch_size should be (0, 128] , but got 140` | bump 到 2048 | `moe_infinity/core/parallel/expert_module.cpp` |
| 6 | Job 結束後 ArcherTaskPool destructor hang Python exit | sbatch job 卡在 CG 狀態直到 SLURM time limit | `os._exit(0)` 在輸出寫完之後跳過 atexit cleanup | `cli.py`、`sanity_offload.py`、`debug_offload_correctness.py`、`debug_offload_logits.py` |
| 7 | uv editable install 的 finder 用 `meta_path.append()` 註冊到末尾 → 被 PathFinder 從 CWD 找到外層 dir 當 namespace package 而 shadow | `from moe_infinity import MoE` 失敗 with `unknown location`,`moe_infinity.__file__ == None` | `sed -i 's/sys.meta_path.append/sys.meta_path.insert(0, ' .pth 對應的 finder 檔(insert 到首位)| `__editable___moe_infinity_0_0_1_finder.py` |

---

## 4. Mixtral 仍壞 — debug 紀錄

**症狀**:HF Mixtral spec-decoding AccRate=0.64;Offload Mixtral AccRate=0.052(基本上 draft 跟 target 對不上)。

### 4.1 排查順序

| 假設 | 怎麼驗 | 結果 |
|---|---|---|
| 我們的 `build_weighted_avg_offload` 算錯 merged expert | **Test A**:用 cpu_source 的 expert,offload 路徑算一份 + 手動 Python loop 算一份,bitwise 比較 | ❌ PASS,**max_diff = 0.0**,merge 端完全正確 |
| offload 模式下 `moe.model(toks)` 本身就吐錯 logits(不是 spec-decoding 的問題) | **Test B**:同 prompt,offload `moe.model(toks)` vs HF `cpu_source(toks)` 比 last-token logits | ✅ **確認**:argmax 'o' vs 'a',top-10 overlap 0/10,max diff 25 |
| moe_infinity 整個 framework 都壞 | 跑他們官方 Qwen3 demo(`examples/readme_example.py`)看輸出 | ✅ Qwen3 demo 輸出合理英文 → **moe_infinity 本身 OK,只是 Mixtral 路徑壞** |
| Mixtral 的 offload cache 是舊 .so 建的,跟新 .so binary 不相容 | `rm -rf cache/offload/mixtral/` 重建後重跑 Test B | ❌ 重建後 argmax 從 'o' 變 'that',**但仍然 top-10 overlap 1/10** |
| Mixtral 走 Large-K GEMM(I=14336 ≥ 3072),Qwen3 走 Small-K(I=768)— **Large-K branch 是 bug** | 強制 `use_large_k = false`,重編 moe_infinity,重跑 Test B | ❌ argmax 仍然 'that',**top-10 overlap 1/10**(Large-K 不是元兇) |

### 4.2 還沒驗的假設

剩下兩個方向最可疑:

1. **Mixtral 的 Python routing 路徑**:[`moe_infinity/models/mixtral.py`](moe_infinity/moe_infinity/models/mixtral.py) `SyncMixtralSparseMoeBlock.forward` 用一段 verbose Python(`F.one_hot + permute + logical_or + sum`)算 `router_mask` / `routing_weights_mask`;Qwen3 走 C++ `self.lib.topk_softmax` CUDA kernel(被他們在 commented-out Python 版本上面預設用了)。
   - **驗法**:把 Mixtral SyncBlock 改成同樣呼叫 `self.lib.topk_softmax(router_logits)`,看 logits 是否對齊。如果通 → Python verbose 路徑 dtype / shape / one_hot int64 promotion 有 bug。
   - **次要可能**:`routing_weights_mask` dtype 不同(Mixtral 走 Python 應該是 bf16;Qwen3 走 kernel 也是 bf16,但 promotion 路徑差)。

2. **expert param 在 C++ 端對 Mixtral 對位錯**:雖然我們紙上推導 `expert_type=4`(MIXTRAL_MOE_DENSE_ACT_DENSE)的 `gate=param[0], up=param[2], down=param[1]` 跟 transformers 4.57 `MixtralBlockSparseTop2MLP` `named_parameters` 順序 `[w1, w2, w3] = [gate, down, up]` 完全對得上,但有沒有可能 `get_topology` 在 grouping 時對 Mixtral 的 expert 命名 schema 處理有 bug?
   - **驗法**:小 Python 工具印出 offload 後 `block.expert_tensor_ids` 對應到的真實 tensor shape,確認 param[0/1/2] 跟我們期待的對得起來。

### 4.3 已嘗試但無效的修法

1. ❌ 重建 Mixtral offload cache(`rm -rf cache/offload/mixtral/`)— logits 仍錯
2. ❌ 強制 Small-K GEMM(`use_large_k = false`)— logits 仍錯,但「不同的錯」(argmax 從 'o' 變 'that')→ GEMM kernel 有影響但不是唯一 bug
3. ✅ 已 revert:`use_large_k = (I >= 3072)` 改回原本邏輯,把 debug 結論寫在註解裡

### 4.4 為什麼 `--force-reinstall` 之後 import 突然壞

`uv pip install -e ./moe_infinity --force-reinstall` 後 `from moe_infinity import MoE` 開始爆 `unknown location`。

原因:uv 寫出的 editable finder([`__editable___moe_infinity_0_0_1_finder.py`](.venv/lib/python3.10/site-packages/__editable___moe_infinity_0_0_1_finder.py))在 `install()` 時用 `sys.meta_path.append(_EditableFinder)` 把 finder 加到 **末尾**。預設 `PathFinder` 跑在前面,從 CWD(`/work/morrisliu07/aug_spec/`)找到外層 `moe_infinity/` dir(沒 `__init__.py` → 當 namespace package)→ shadow 掉真正的 inner package。

之前能 work 應該是某次舊 uv 寫出的 finder 位置不同,或 PathFinder 順序差異。

**修法**:`sed -i 's|sys.meta_path.append(_EditableFinder)|sys.meta_path.insert(0, _EditableFinder)|'` 改成插到 meta_path 首位。verified 通了。

**注意:每次 `uv pip install -e ./moe_infinity --force-reinstall` 後都要重做這個 sed**。或者直接 patch uv 的 wheel 生成 template(out of scope)。

---

## 5. 下一步(給未來繼續做的人)

### 5.1 短期:debug Mixtral logits

按 §4.2 順序試:
1. 把 `SyncMixtralSparseMoeBlock.forward` 改成 `self.lib.topk_softmax` 路徑,跑 [`debug_offload_logits.py --model mixtral`](tests/debug_offload_logits.py),看 top-10 overlap 是否升到 8+。
2. 印 expert param 順序診斷(寫個 micro Python script)。

如果都不通,可能要進 C++ 加 debug print,或考慮 moe_infinity 上游 issue。

### 5.2 中期:跑 Qwen3 production size

aug_spec 端在 Qwen3 上已驗通,smoke 是 1 q/cat × 32 tokens。production 應該:
- `configs/qwen3_count.yaml` 加 `backend: offload`(或寫新的 `configs/qwen3_offload_count.yaml`)
- 跑 10 q/cat × 512 tokens(跟 HF [PROGRESS.md](PROGRESS.md) Phase A.5 同規模)
- 比 AccRate / TPS(預期 AccRate 接近 HF 0.09;TPS 較慢因為 PCIe)

### 5.3 長期:Phase 4-9

按 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) `§11` 順序走 — `aug_spec bench` + PCIe profiler + SpecMoE baseline + VRAM-matched paired configs。

---

## 6. 怎麼跑(精簡 recipe)

### Sanity(每個環境一次)
```bash
sbatch scripts/sanity_offload.sh
# 看 /work/morrisliu07/job_log/*.log 的 Q1-Q4 verdict
```

### HF baseline regression
```bash
sbatch scripts/run.sh configs/_smoke.yaml
# 期望: MAT≈2.87 AccRate≈0.64 TPS≈9
```

### Qwen3 offload smoke(目前驗通的設定)
```bash
sbatch scripts/run.sh configs/_smoke_offload_qwen3.yaml
# 期望: MAT≈1.44 AccRate≈0.15 TPS≈0.87
```

### 完整 Phase 1-3 三步驗證
```bash
sbatch scripts/smoke_phase1to3.sh
# import sanity + HF regression + Mixtral offload(Mixtral 步驟現階段會出 AccRate≈0.05,屬於已知 bug)
```

### 環境注意
1. 每個 shell 新開都要 `export HF_HOME=/work/morrisliu07/.cache/huggingface`
2. **不要** `uv pip install -e ./moe_infinity` 不加 `--no-deps` — 會降版 transformers(會破壞 HF model 載入)
3. 如果重編 moe_infinity 後 `from moe_infinity import MoE` 爆 `unknown location`:
   ```bash
   sed -i 's|sys.meta_path.append(_EditableFinder)|sys.meta_path.insert(0, _EditableFinder)|' \
       /work/morrisliu07/aug_spec/.venv/lib/python3.10/site-packages/__editable___moe_infinity_0_0_1_finder.py
   ```
4. **不能在 `/work/morrisliu07/aug_spec/` 下直接 `python -c "from moe_infinity ..."`** — CWD 會被 PathFinder 找到外層 `moe_infinity/` dir 當 namespace package shadow 掉真正的 install。要嘛先 `cd /tmp`,要嘛先做上面 sed 修。
