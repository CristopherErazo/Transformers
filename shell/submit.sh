#!/bin/bash

# Fixed parameters
d_model=1024
dataset_size=20000
seq_len=256
rank=64
num_epochs=30
is_tqdm=False
freeze_embeddings=False
freeze_attention=False
random_positional_encoding=False
skip_residual=False
# amplitude=1.0
# set amplitude as 1/sqrt(d_model)
amplitude=$(echo "scale=10; 1/sqrt($d_model)" | bc -l)
sigma=0.1
n_prints=120

# List of variables
bs_values=(64 128 256)
lr_values=(0.01)

Run the loop after some min delay to stagger jobs
min_wait=0
echo "Starting training runs at $(date). Waiting for ${min_wait} before starting..."
sleep ${min_wait}m
echo "Starting now. at $(date)."

# Loop through each mode and run the training script
for lr in "${lr_values[@]}"; do
    for batch_size in "${bs_values[@]}"; do
        echo "Running with --lr=$lr --batch_size=$batch_size"

        python -u ./scripts/measurements.py \
            --d_model $d_model \
            --dataset_size $dataset_size \
            --batch_size $batch_size \
            --seq_len $seq_len \
            --rank $rank \
            --num_epochs $num_epochs \
            --lr $lr \
            --is_tqdm $is_tqdm \
            --freeze_embeddings $freeze_embeddings \
            --freeze_attention $freeze_attention \
            --random_positional_encoding $random_positional_encoding \
            --skip_residual $skip_residual \
            --amplitude $amplitude \
            --sigma $sigma \
            --n_prints $n_prints

        echo "Completed: $mode at $(date)"
        echo "---"
        echo ""
    done
done

echo "All training runs completed!"


