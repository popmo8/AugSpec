#!/usr/bin/env bash
# One-shot environment bootstrap for aug_spec.
#
# Prerequisites (on TWCC or similar):
#   module load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
#   export HF_HOME=/work/${USER}/.cache/huggingface
#   git clone https://github.com/NVIDIA/cutlass.git /work/${USER}/cutlass   # if not already cloned
#
# This script:
#   1. Auto-detects CUTLASS_DIR (env > ~/cutlass > /work/${USER}/cutlass).
#   2. Creates a uv venv (.venv/) using Python 3.10.
#   3. Installs CUDA-12.6 PyTorch wheels.
#   4. Installs MoE-Infinity as an editable dep (so .py / .cpp edits in
#      ./moe_infinity/ go live without reinstall; C++ changes still need
#      this script re-run to rebuild the .so).
#   5. Installs aug_spec itself in editable mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ── prerequisites ──────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "[install] 'uv' not found on PATH. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/moe_infinity" ]]; then
    echo "[install] ./moe_infinity/ not present at repo root." >&2
    exit 1
fi

for tool in nvcc gcc; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[install] '${tool}' not on PATH. Load TWCC modules first:" >&2
        echo "  module load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0" >&2
        exit 1
    fi
done

# ── locate CUTLASS ─────────────────────────────────────────────────────
# moe_infinity's setup.py reads CUTLASS_DIR from env (default ~/cutlass).
# It needs `<dir>/include/cutlass/` to exist at build time.
if [[ -z "${CUTLASS_DIR:-}" ]]; then
    for candidate in "${HOME}/cutlass" "/work/${USER}/cutlass"; do
        if [[ -d "${candidate}/include/cutlass" ]]; then
            export CUTLASS_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -z "${CUTLASS_DIR:-}" || ! -d "${CUTLASS_DIR}/include/cutlass" ]]; then
    cat >&2 <<EOF
[install] CUTLASS not found. Clone NVIDIA/cutlass somewhere with read
access, then re-run this script. Suggested:

  git clone https://github.com/NVIDIA/cutlass.git /work/\${USER}/cutlass
  ./install.sh

Or if you cloned it elsewhere, point CUTLASS_DIR at it before invoking:

  CUTLASS_DIR=/path/to/cutlass ./install.sh
EOF
    exit 1
fi
echo "[install] Using CUTLASS_DIR=${CUTLASS_DIR}"

# ── venv + python deps ────────────────────────────────────────────────
uv venv --python 3.10
# shellcheck source=/dev/null
source .venv/bin/activate

uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install wheel

# MoE-Infinity: editable, no isolation (uses system gcc / CUDA / cutlass).
# CUTLASS_DIR is in the environment from the locate step above.
uv pip install -e ./moe_infinity --no-build-isolation

# aug_spec: editable.
uv pip install -e .

echo
echo "[install] Done. Activate with:  source .venv/bin/activate"
echo "[install] Smoke test:           aug_spec --help"
