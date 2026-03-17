#!/bin/bash
# =============================================================================
# scripts/run_ablation.sh — DSCMFNet ablation experiment runner
#
# Runs all 9 experiments from CLAUDE.md §Ablation Experiment Plan in order.
# Each experiment outputs to an independent subdirectory under OUTPUT_ROOT.
#
# Usage:
#   bash scripts/run_ablation.sh                    # run all 9
#   bash scripts/run_ablation.sh 4 5 6              # run only exps 4, 5, 6
#   EPOCHS=1 KFOLD=2 bash scripts/run_ablation.sh   # quick smoke test
#
# Key env-var overrides (all optional):
#   DATA_ROOT   — path to processed_data/   (default: ./processed_data)
#   OUTPUT_ROOT — root for all outputs      (default: ./output/ablation)
#   EPOCHS      — epochs per experiment     (default: 200)
#   BATCH_SIZE  — mini-batch size           (default: 4)
#   KFOLD       — K for cross-validation    (default: 5)
#   NUM_WORKERS — DataLoader workers        (default: 0)
#   SEED        — global random seed        (default: 42)
#   LR          — initial learning rate     (default: 1e-4)
#   AMP         — "--amp" to enable AMP     (default: "")
# =============================================================================

set -euo pipefail

# ── cd to project root so relative paths work from any call site ─────────────
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# ── Common parameters ─────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-./processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/ablation}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
KFOLD="${KFOLD:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
LR="${LR:-1e-4}"
AMP="${AMP:-}"

# ── Selective-run helper: skip an experiment if the caller listed specific IDs ─
# Usage: should_run <exp_number> — returns 0 (run) or 1 (skip)
SELECTED=("$@")
should_run() {
    [ "${#SELECTED[@]}" -eq 0 ] && return 0     # no filter → run all
    for s in "${SELECTED[@]}"; do
        [ "$s" -eq "$1" ] && return 0
    done
    return 1
}

echo "================================================================="
echo " DSCMFNet ablation — $(date '+%Y-%m-%d %H:%M:%S')"
echo "   data_root   : ${DATA_ROOT}"
echo "   output_root : ${OUTPUT_ROOT}"
echo "   epochs      : ${EPOCHS}  |  batch_size : ${BATCH_SIZE}"
echo "   kfold       : ${KFOLD}   |  seed       : ${SEED}"
[ -n "${AMP}" ] && echo "   amp         : enabled"
[ "${#SELECTED[@]}" -gt 0 ] && echo "   running     : experiments ${SELECTED[*]}"
echo "================================================================="

# =============================================================================
# Experiment 1: NBI-only baseline
#   Single ResNet34 encoder, NBI close-up input only.
#   Lower bound: performance with NBI detail stream alone.
# =============================================================================
should_run 1 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp1_NBI_only_baseline" \
    --model       nbi_only \
    --phase       1 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 2: WLI-only baseline
#   Single PVTv2-B2 encoder, WLI close-up input only.
#   Lower bound: performance with WLI context stream alone.
# =============================================================================
should_run 2 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp2_WLI_only_baseline" \
    --model       wli_only \
    --phase       1 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 3: Early fusion (6-channel)
#   WLI and NBI stacked channel-wise (6-ch) → single PVTv2-B2.
#   Tests naive early modality fusion before any dual-stream design.
# =============================================================================
should_run 3 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp3_early_fusion_6ch" \
    --model       early_fusion \
    --phase       1 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 4: Dual-stream + concat fusion  (no CMFIM / SAM / boundary)
#   PVTv2-B2 + ResNet34, features concatenated at every scale.
#   Boundary loss disabled (--lambda_bdy 0) for a fair ablation baseline.
# =============================================================================
should_run 4 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp4_dual_stream_concat" \
    --model       dscmfnet \
    --fusion_mode concat \
    --phase       1 \
    --lambda_bdy  0 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 5: + CMFIM  (bidirectional cross-modal attention, no SAM)
#   Replaces concat at 1/8·1/16·1/32 scales with CMFIM attention.
#   SAM not yet added; boundary head disabled to isolate attention effect.
# =============================================================================
should_run 5 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp5_CMFIM_no_SAM" \
    --model       dscmfnet \
    --fusion_mode cmfim_no_sam \
    --phase       1 \
    --lambda_bdy  0 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 6: + SAM  (Spatial Alignment Module)
#   CMFIM + deformable warp that aligns NBI features to WLI space.
#   Boundary head still disabled; isolates SAM contribution over exp 5.
# =============================================================================
should_run 6 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp6_CMFIM_with_SAM" \
    --model       dscmfnet \
    --fusion_mode cmfim_no_boundary \
    --phase       1 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 7: + Boundary Head  (full Phase 1 DSCMFNet)
#   Activates the boundary prediction head and weighted boundary loss.
#   This is the complete Phase 1 architecture (--fusion_mode cmfim default).
# =============================================================================
should_run 7 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp7_full_phase1_DSCMFNet" \
    --model       dscmfnet \
    --fusion_mode cmfim \
    --phase       1 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 8: Cross-scale Phase 2  (WLI distant + NBI close-up)
#   Switches to Phase 2 cross-scale loader: WLI distant (wide FOV) paired
#   with NBI close-up (narrow FOV). Independent augmentation per modality.
# =============================================================================
should_run 8 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp8_cross_scale_phase2" \
    --model       dscmfnet \
    --fusion_mode cmfim \
    --phase       2 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

# =============================================================================
# Experiment 9: + WLI auxiliary loss  (Phase 2 + WLI mask supervision)
#   Adds auxiliary structure loss against the WLI distant mask (lambda=0.3),
#   encouraging the model to leverage WLI contextual information.
# =============================================================================
should_run 9 && \
python train.py \
    --data_root   "${DATA_ROOT}" \
    --output      "${OUTPUT_ROOT}/exp9_cross_scale_wli_aux_loss" \
    --model       dscmfnet \
    --fusion_mode cmfim \
    --phase       2 \
    --lambda_wli  0.3 \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --kfold       "${KFOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    ${AMP}

echo ""
echo "================================================================="
echo " All experiments completed."
echo " Collect results: python scripts/collect_results.py \\"
echo "                      --output_root ${OUTPUT_ROOT}"
echo "================================================================="
