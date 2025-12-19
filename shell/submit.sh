#!/bin/bash

# Fixed parameters
dataset_size=500
d_model=256
num_epochs=30
seq_len=256
is_tqdm=True
batch_size=27


# List of variables
lr_values=(1e-3 2e-3 5e-4)

# Loop through each mode and run the training script
for lr in "${lr_values[@]}"; do
    echo "Running new_train.py with --lr=$lr"

    python ./scripts/train_test.py \
        --d_model $d_model \
        --lr $lr \
        --batch_size $batch_size \
        --num_epochs $num_epochs \
        --seq_len $seq_len \
        --dataset_size $dataset_size \
        --is_tqdm $is_tqdm

    echo "Completed: $mode"
    echo "---"
done

echo "All training runs completed!"