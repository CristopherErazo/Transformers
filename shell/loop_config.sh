#!/bin/bash

# Fixed parameters
d_model=1024
dataset_size=10000
seq_len=256
rank=-1
batch_size=64
num_epochs=15
is_tqdm=False
random_positional_encoding=True
amplitude=1.0
sigma=0.1
n_prints=200



# Loop over various configurations = ( lr , freeze_embeddings, freeze_attention,skip_residual )
configurations=(
    "0.001 False False False"
    # "0.001 False True False"
    # "0.001 True False False"
)

for config in "${configurations[@]}"; do
    read -r lr freeze_embeddings freeze_attention skip_residual <<< "$config"
    echo "Running with --lr=$lr --freeze_embeddings=$freeze_embeddings --freeze_attention=$freeze_attention --skip_residual=$skip_residual"

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

echo "All training runs completed!"


