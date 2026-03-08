#!/bin/bash
# Benchmark baseline vs DAC pipeline parallel inference.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_benchmark.sh
#
# Environment variables:
#   CKPT         Checkpoint path       (default: ckpts/hiar.pt)
#   CONFIG       Config YAML path      (default: configs/hiar.yaml)
#   PROMPT       Text prompt           (default: built-in)
#   PROFILE_DIR  Output directory      (default: outputs/dac_profile)
#   NPROC        Number of GPUs        (default: 4)
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export PYTHONUNBUFFERED=1

NPROC=${NPROC:-4}
CKPT=${CKPT:-ckpts/hiar.pt}
CONFIG=${CONFIG:-configs/hiar.yaml}
PROMPT=${PROMPT:-"A golden retriever playing in the snow, cinematic lighting, slow motion"}
PROFILE_DIR=${PROFILE_DIR:-outputs/dac_profile}

mkdir -p "$PROFILE_DIR"

echo "========================================"
echo "HiAR Pipeline Parallel Benchmark"
echo "  Checkpoint : $CKPT"
echo "  Config     : $CONFIG"
echo "  GPUs       : $NPROC"
echo "========================================"

run() {
    local MODE=$1 FRAMES=$2
    local TAG="${MODE}_${FRAMES}f"
    local FLAGS=""
    [ "$MODE" = "baseline" ] && FLAGS="--no_dac"

    echo ""
    echo "---- [$TAG] ${MODE}, ${FRAMES} frames ----"
    torchrun --nproc_per_node=$NPROC \
        scripts/pipeline_parallel_inference.py \
        --config_path "$CONFIG" \
        --checkpoint_path "$CKPT" \
        --prompt "$PROMPT" \
        --output_path "$PROFILE_DIR/${TAG}.mp4" \
        --num_output_frames "$FRAMES" \
        --profile \
        --profile_output "$PROFILE_DIR/${TAG}" \
        $FLAGS
    echo "[$TAG] done"
}

run baseline 21
run dac      21
run baseline 81
run dac      81

echo ""
echo "========================================"
echo "Benchmark complete.  Profiles: $PROFILE_DIR/"
echo ""
echo "Generate plots:"
echo "  python scripts/plot_dac_results.py --input_dir $PROFILE_DIR"
echo "========================================"
