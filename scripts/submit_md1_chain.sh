#!/bin/bash
# Submit the md1 (DeepSeek-MoE-16B-base Table-1 nine-row) campaign as one
# dependency chain — 15 jobs, mirroring the mx1 structure. This script only
# calls sbatch (scheduling, login-node safe); all work runs on compute nodes.
# Launched 2026-07-21 (user go-ahead after the hold). Re-running submits a
# FRESH chain — for single-job resubmits use the SOP in deepseek_plan.md §5.
#   bash scripts/submit_md1_chain.sh
set -euo pipefail
cd /work/morrisliu07/aug_spec

sub() { local name="$1"; shift; local args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  local out; out=$(sbatch --job-name="$name" "${args[@]}" "$@"); echo "${out##* }"; }

# ── wave 1: no prerequisites ──────────────────────────────────────────
A=$(sub md1_smk_a -- scripts/run_watchdog.sh \
      configs/md1_smoke_randmask.yaml configs/md1_smoke_specmoe.yaml \
      configs/md1_smoke_randmerge.yaml configs/md1_smoke_ours.yaml \
      configs/md1_smoke_speed.yaml)
B=$(sub md1_calib -- scripts/run_collect_calib.sh deepseek-ai/deepseek-moe-16b-base)
C=$(sub md1_search_dv -- scripts/run_search_dv.sh deepseek-ai/deepseek-moe-16b-base --num-keep 3)
echo "A(smoke5)=$A  B(calib)=$B  C(dv-search)=$C"

# ── chain 1: direct-runnable full runs, gated on smoke5 ───────────────
for m in randmask specmoe randmerge ours speed; do
  J=$(sub md1_$m --dependency=afterok:$A --time=24:00:00 -- \
        scripts/run_watchdog.sh configs/md1_$m.yaml)
  echo "full md1_$m=$J (afterok:$A)"
done

# ── chain 2: calib -> artifacts -> smoke3 -> full3 ────────────────────
D=$(sub md1_artifacts --dependency=afterok:$B -- scripts/run_md1_artifacts.sh)
A2=$(sub md1_smk_b --dependency=afterok:$D -- scripts/run_watchdog.sh \
       configs/md1_smoke_enum.yaml configs/md1_smoke_hcsmoe.yaml \
       configs/md1_smoke_mcsmoe.yaml)
echo "D(artifacts)=$D (afterok:$B)  A2(smoke3)=$A2 (afterok:$D)"
for m in enum hcsmoe mcsmoe; do
  J=$(sub md1_$m --dependency=afterok:$A2 --time=24:00:00 -- \
        scripts/run_watchdog.sh configs/md1_$m.yaml)
  echo "full md1_$m=$J (afterok:$A2)"
done

# ── chain 3: dv-search -> dv smoke -> dv full ─────────────────────────
A3=$(sub md1_smk_dv --dependency=afterok:$C -- scripts/run_watchdog.sh \
       configs/md1_smoke_dv.yaml)
F9=$(sub md1_dv --dependency=afterok:$A3 --time=24:00:00 -- \
       scripts/run_watchdog.sh configs/md1_dv.yaml)
echo "A3(dv-smoke)=$A3 (afterok:$C)  full md1_dv=$F9 (afterok:$A3)"
echo "[submit_md1_chain] all 15 jobs submitted"
