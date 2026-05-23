# aug_spec

Augmented speculative decoding for Mixture-of-Experts inference.

## Install

```bash
module load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
export HF_HOME=/work/${USER}/.cache/huggingface
./install.sh
source .venv/bin/activate
```

`install.sh` builds a `uv` virtualenv, installs CUDA-12.6 PyTorch, then
`pip install -e ./moe_infinity` and `pip install -e .` so both can be
edited in place. Python edits to either are live; C++ changes inside
`./moe_infinity/core/` require re-running `install.sh`.

## Run an experiment

```bash
source .venv/bin/activate
export HF_HOME=/work/${USER}/.cache/huggingface     # MoE checkpoints are huge — keep them off /home
aug_spec run --config configs/mixtral_count.yaml
```

The `HF_HOME` export is needed **every shell session**, not just at
install time. Without it, HuggingFace downloads models into
`~/.cache/huggingface/` (on `/home`, which has a tight quota on TWCC)
and the run will die with "No space left on device" mid-download. Add
it to your shell rc or source a small `env.sh` if you tire of typing it.

Each experiment is one YAML file under `configs/`. The same code path
serves all (model, draft strategy) combinations — adding a new
experiment means adding a YAML, not a Python file. The full YAML
schema (every key, every default, every draft strategy's `args`) is
documented in [configs/README.md](configs/README.md).

Outputs go to `output/<config-stem>/`:
- `per_question_summary.csv`
- `overall_summary.csv`
- `summary.json` (per-run aggregate)
- `expert_weights_history.json` (when the draft strategy records it)

### On TWCC under SLURM

```bash
# Single config:
sbatch scripts/run.sh configs/mixtral_count.yaml

# Or several configs sequentially in one job:
sbatch scripts/run.sh configs/mixtral_count.yaml configs/qwen3_count.yaml

# Or a fixed sweep — edit the CONFIGS array at the top of
# scripts/run_sweep.sh, then:
sbatch scripts/run_sweep.sh
```

Both scripts bake in the partition / account / module loads / env
vars (`HF_HOME`, `PYTHONUNBUFFERED`, `PYTORCH_CUDA_ALLOC_CONF`),
source the repo's `.venv`, then run each YAML in sequence.
`scripts/run.sh` takes its configs from the command line;
`scripts/run_sweep.sh` keeps the list inside the file so you can
re-run a fixed sweep without re-typing it.

## Layout

```
aug_spec/
├── moe_infinity/                # vendored third-party dep (separate pip install -e)
├── src/aug_spec/                # this repo's package (pip install -e .)
│   ├── cli.py                   # entrypoint  → `aug_spec run`
│   ├── controller.py            # adapter × draft wiring
│   ├── adapters/                # model-family forward overrides
│   ├── drafts/                  # per-cycle draft-side strategies
│   └── runtime/                 # loader / specbench / phase patch
├── configs/                     # one YAML per experiment (see configs/README.md)
├── scripts/run.sh               # SLURM wrapper around `aug_spec run`
└── output/, cache/              # gitignored artefacts
```
