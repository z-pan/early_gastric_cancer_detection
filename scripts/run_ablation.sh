#!/usr/bin/env bash
# =============================================================================
# scripts/run_ablation.sh — DSCMFNet ablation experiment runner
#
# Runs all 9 experiments from CLAUDE.md §Ablation Experiment Plan in order.
# Each experiment incrementally adds one architectural component so that the
# contribution of each design choice can be measured in isolation.
#
# Ablation table (CLAUDE.md):
#   #1  NBI-only baseline          single ResNet34, NBI close-up
#   #2  WLI-only baseline          single PVTv2-B2, WLI close-up
#   #3  Early fusion (6-ch)        concat WLI+NBI → single PVTv2-B2
#   #4  Dual-stream + concat       two encoders, concat fusion all scales
#   #5  + CMFIM                    cross-modal attention, no SAM, no bdy head
#   #6  + SAM                      add Spatial Alignment Module
#   #7  + Boundary Head            full Phase 1 DSCMFNet
#   #8  Cross-scale (Phase 2)      WLI distant + NBI close-up
#   #9  + WLI aux loss             add auxiliary WLI-mask supervision
#
# Usage:
#   bash scripts/run_ablation.sh              # run all 9 experiments
#   bash scripts/run_ablation.sh 4 5 6        # run only experiments 4, 5, 6
#
# Environment variables (override defaults via export or inline prefix):
#   DATA_ROOT    path to processed_data/        (default: processed_data)
#   OUTPUT_ROOT  root dir for all results        (default: output/ablation)
#   EPOCHS       training epochs per experiment  (default: 200)
#   BATCH_SIZE   samples per mini-batch          (default: 4)
#   KFOLD        K for cross-validation          (default: 5)
#   NUM_WORKERS  DataLoader worker processes     (default: 0)
#   SEED         global random seed              (default: 42)
#   AMP          set to "--amp" for mixed prec.  (default: "")
#   PRETRAINED   "--pretrained" or
#                "--no-pretrained"               (default: --pretrained)
#
# Quick smoke test (1 epoch, 2 folds, 1 sample):
#   EPOCHS=1 KFOLD=2 BATCH_SIZE=1 bash scripts/run_ablation.sh
# =============================================================================

set -euo pipefail

# ── Locate project root regardless of where this script is called from ──────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Configurable parameters ─────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/ablation}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
KFOLD="${KFOLD:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
AMP="${AMP:-}"
PRETRAINED="${PRETRAINED:---pretrained}"

# ── Build the common argument list ──────────────────────────────────────────
COMMON=(
    --data_root    "$DATA_ROOT"
    --epochs       "$EPOCHS"
    --batch_size   "$BATCH_SIZE"
    --kfold        "$KFOLD"
    --num_workers  "$NUM_WORKERS"
    --seed         "$SEED"
    "$PRETRAINED"
)
[ -n "$AMP" ] && COMMON+=("$AMP")

# ── Optional subset: caller may pass experiment numbers as positional args ──
# e.g.  bash run_ablation.sh 1 3 7   → only runs experiments 1, 3, 7
SELECTED=("$@")

# ── Helper: print a banner and run one experiment ───────────────────────────
run_exp() {
    local idx="$1"
    local tag="$2"
    shift 2   # remaining args are forwarded to train.py

    # If a subset was specified, skip unselected experiments.
    if [ "${#SELECTED[@]}" -gt 0 ]; then
        local found=0
        for s in "${SELECTED[@]}"; do
            [ "$s" -eq "$idx" ] && { found=1; break; }
        done
        [ "$found" -eq 0 ] && return 0
    fi

    local out_dir="${OUTPUT_ROOT}/exp${idx}_${tag}"

    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    printf "│  Experiment %2d / 9 : %-40s│\n" "$idx" "$tag"
    printf "│  Output            : %-40s│\n" "$(basename "$out_dir")"
    echo "└─────────────────────────────────────────────────────────────┘"

    python train.py "${COMMON[@]}" --output "$out_dir" "$@"
}

echo "================================================================="
echo " DSCMFNet ablation experiments"
echo "   data_root   : ${DATA_ROOT}"
echo "   output_root : ${OUTPUT_ROOT}"
echo "   epochs      : ${EPOCHS}   batch_size : ${BATCH_SIZE}"
echo "   kfold       : ${KFOLD}    seed       : ${SEED}"
[ -n "$AMP"                    ] && echo "   amp         : enabled"
[ "${#SELECTED[@]}" -gt 0 ] && echo "   running     : experiments ${SELECTED[*]}"
echo "================================================================="

# =============================================================================
# Experiment 1 — NBI-only baseline
#   Single ResNet34 encoder, NBI close-up input only.
#   Establishes the lower bound: how well can a single-modality model segment
#   EGC using only the NBI detail stream?
# =============================================================================
run_exp 1 "NBI_only_baseline" \
    --model nbi_only \
    --phase 1

# =============================================================================
# Experiment 2 — WLI-only baseline
#   Single PVTv2-B2 encoder, WLI close-up input only.
#   Measures the complementary lower bound for the WLI context stream.
# =============================================================================
run_exp 2 "WLI_only_baseline" \
    --model wli_only \
    --phase 1

# =============================================================================
# Experiment 3 — Early fusion (6-channel input)
#   Concatenate WLI and NBI channel-wise (6-ch tensor) → single PVTv2-B2.
#   Tests whether naive early fusion of both modalities already helps.
# =============================================================================
run_exp 3 "early_fusion_6ch" \
    --model early_fusion \
    --phase 1

# =============================================================================
# Experiment 4 — Dual-stream + concat fusion  (no CMFIM / SAM / bdy head)
#   Two independent encoders (PVTv2-B2 + ResNet34); at every scale the
#   features are simply projected and concatenated (ConcatFusionBlock).
#   Boundary loss disabled (lambda_bdy=0) so the boundary head does not
#   contribute — keeps this experiment comparable to #5 and #6.
# =============================================================================
run_exp 4 "dual_stream_concat" \
    --model       dscmfnet \
    --fusion_mode concat \
    --phase       1 \
    --lambda_bdy  0

# =============================================================================
# Experiment 5 — + CMFIM cross-modal attention  (no SAM, no boundary head)
#   Replaces ConcatFusionBlock at 1/8·1/16·1/32 with bidirectional cross-
#   modal attention (cmfim_no_sam mode).  SAM is not yet added here.
#   lambda_bdy=0 keeps the boundary head out of the loss for a fair comparison
#   with experiment #4.
# =============================================================================
run_exp 5 "dual_stream_CMFIM_no_SAM" \
    --model       dscmfnet \
    --fusion_mode cmfim_no_sam \
    --phase       1 \
    --lambda_bdy  0

# =============================================================================
# Experiment 6 — + SAM  (Spatial Alignment Module)
#   Adds the SAM deformation warp to CMFIM (cmfim_no_boundary mode).
#   The boundary head is still not active here; isolates SAM's contribution.
# =============================================================================
run_exp 6 "dual_stream_CMFIM_with_SAM" \
    --model       dscmfnet \
    --fusion_mode cmfim_no_boundary \
    --phase       1

# =============================================================================
# Experiment 7 — + Boundary Head  (full Phase 1 DSCMFNet)
#   Activates the boundary prediction head and its weighted loss term.
#   This is the complete Phase 1 architecture.
# =============================================================================
run_exp 7 "full_phase1_DSCMFNet" \
    --model       dscmfnet \
    --fusion_mode cmfim \
    --phase       1

# =============================================================================
# Experiment 8 — Cross-scale Phase 2  (WLI distant + NBI close-up)
#   Switches to the Phase 2 cross-scale loader: WLI distant (wide FOV) is
#   paired with NBI close-up (narrow FOV).  Independent geometric aug per
#   modality.  Primary GT: NBI close-up mask.
# =============================================================================
run_exp 8 "cross_scale_phase2" \
    --model       dscmfnet \
    --fusion_mode cmfim \
    --phase       2

# =============================================================================
# Experiment 9 — + WLI auxiliary loss  (Phase 2 + WLI mask supervision)
#   Adds an auxiliary structure loss on the main segmentation output against
#   the WLI distant mask (lambda_wli=0.3), providing extra supervision from
#   the wider-FOV WLI stream and encouraging the model to leverage WLI context.
# =============================================================================
run_exp 9 "cross_scale_wli_aux_loss" \
    --model       dscmfnet \
    --fusion_mode cmfim \
    --phase       2 \
    --lambda_wli  0.3

# =============================================================================
echo ""
echo "================================================================="
echo " All ablation experiments completed."
printf " Results saved to: %s\n" "${OUTPUT_ROOT}"
echo "================================================================="
