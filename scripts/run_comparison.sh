#!/bin/bash
# Generate comparison videos: Self-Forcing vs HiAR (20s, 30 prompts)
set -e

PROMPTS=data/comparison_prompts.txt
NUM_FRAMES=81  # 81 latent frames = 321 pixel frames ~= 20s @ 16fps (must be divisible by num_frame_per_block=3)
SEED=42

SF_CKPT=ckpts/self_forcing_dmd.pt  # Download from Self-Forcing repo
HIAR_CKPT=ckpts/hiar.pt

OUTPUT_BASE=outputs/comparison_v2

echo "=== Step 1: Generate Self-Forcing videos (frame-first) ==="
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path $SF_CKPT \
    --data_path $PROMPTS \
    --output_folder ${OUTPUT_BASE}/self_forcing \
    --num_output_frames $NUM_FRAMES \
    --use_ema --save_with_index --seed $SEED \
    --inference_method frame_first

echo "=== Step 2: Generate HiAR videos (timestep-first) ==="
python inference.py \
    --config_path configs/hiar.yaml \
    --checkpoint_path $HIAR_CKPT \
    --data_path $PROMPTS \
    --output_folder ${OUTPUT_BASE}/hiar \
    --num_output_frames $NUM_FRAMES \
    --use_ema --save_with_index --seed $SEED \
    --inference_method timestep_first

echo "=== Step 3: Merge side-by-side ==="
python scripts/merge_comparison.py \
    --sf_dir ${OUTPUT_BASE}/self_forcing \
    --hiar_dir ${OUTPUT_BASE}/hiar \
    --output_dir ${OUTPUT_BASE}/merged \
    --sf_label "Self-Forcing" \
    --hiar_label "HiAR" \
    --overwrite

echo "=== Done! Merged videos in ${OUTPUT_BASE}/merged/ ==="
