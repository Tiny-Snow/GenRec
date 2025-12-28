#!/usr/bin/env bash
set -euo pipefail

# # Configure visible GPUs for the run (comma-separated string of physical GPU ids).
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# Define GPU groups used by grid_search. Each entry is a comma-separated list of GPU ids.
# Example: GPU_GROUPS=("4,5" "6,7") to allocate two GPUs per trial.
GPU_GROUPS=("3" "4" "5" "6" "7" "3" "4" "5" "6" "7")

if [[ ${#GPU_GROUPS[@]} -eq 0 ]]; then
    echo "GPU_GROUPS must contain at least one entry" >&2
    exit 1
fi

# We assume configs are provided in ./configs/
BASE_DIR=$(dirname "$(realpath "$0")")
CONFIG_DIR="${BASE_DIR}/configs"

# Set up template config path.
TEMPLATE_PATH="${CONFIG_DIR}/seqrec/template.yaml"

# Set up search config path.
SEARCH_PATH="${CONFIG_DIR}/seqrec/sasrec_spring_movielens-1m.yaml"

# Set up output directory.
OUTPUT_ROOT="${BASE_DIR}/../outputs/seqrec/sasrec_spring"

# Set up main module to run.
MAIN_MODULE="genrec.main_seqrec"

# Optional dryrun/rerun controls (set to desired exp_id or leave empty).
DRYRUN_EXP_ID="${DRYRUN_EXP_ID:-}"
RERUN_EXP_ID="${RERUN_EXP_ID:-}"

if [[ -n "$DRYRUN_EXP_ID" && -n "$RERUN_EXP_ID" ]]; then
    echo "Set either DRYRUN_EXP_ID or RERUN_EXP_ID, not both." >&2
    exit 1
fi

EXTRA_ARGS=()
for grp in "${GPU_GROUPS[@]}"; do
    EXTRA_ARGS+=(--gpu_groups "$grp")
done

if [[ -n "$DRYRUN_EXP_ID" ]]; then
    EXTRA_ARGS+=(--dryrun "$DRYRUN_EXP_ID")
elif [[ -n "$RERUN_EXP_ID" ]]; then
    EXTRA_ARGS+=(--rerun "$RERUN_EXP_ID")
fi

# RUN >>>>>
poetry run python scripts/grid_search.py \
    --template "${TEMPLATE_PATH}" \
    --search "${SEARCH_PATH}" \
    --main "${MAIN_MODULE}" \
    --output_root "${OUTPUT_ROOT}" \
    "${EXTRA_ARGS[@]}"
