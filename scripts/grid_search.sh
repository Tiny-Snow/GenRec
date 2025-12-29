#!/usr/bin/env bash
set -euo pipefail

# This is a sample script to run grid search experiments.

# Define GPU groups used by grid_search. Each entry is a comma-separated list of GPU ids.
# Example:
# GPU_GROUPS=("4,5" "6,7") to allocate two GPUs per trial.
# GPU_GROUPS=("4,5" "6,7" "4,5") to allocate two GPUs per trial with three parallel trials.
GPU_GROUPS=("4,5" "6,7")

if [[ ${#GPU_GROUPS[@]} -eq 0 ]]; then
    echo "GPU_GROUPS must contain at least one entry" >&2
    exit 1
fi

# We assume configs are provided in ./configs/
# You may modify these paths as needed.
BASE_DIR=$(dirname "$(realpath "$0")")
CONFIG_DIR="${BASE_DIR}/configs"

# Set up template config path.
# Here we use the seqrec template in https://github.com/Tiny-Snow/GenRec/blob/main/scripts/configs/seqrec/template.yaml as an example.
TEMPLATE_PATH="${CONFIG_DIR}/seqrec/template.yaml"

# Set up search config path.
# Note that if you specify DRYRUN_EXP_ID or RERUN_EXP_ID below, this will be overridden by the existing search.yaml.
SEARCH_PATH="${CONFIG_DIR}/seqrec/sasrec_spring_bce_amazon-beauty.yaml"

# Set up output directory.
# Note that if you specify DRYRUN_EXP_ID or RERUN_EXP_ID below, the existing output directory should be specified there correctly.
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

# If dryrun or rerun is specified, override SEARCH_PATH accordingly.
if [[ -n "$DRYRUN_EXP_ID" ]]; then
    SEARCH_PATH="${OUTPUT_ROOT}/${DRYRUN_EXP_ID}/search.yaml"
elif [[ -n "$RERUN_EXP_ID" ]]; then
    SEARCH_PATH="${OUTPUT_ROOT}/${RERUN_EXP_ID}/search.yaml"
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
