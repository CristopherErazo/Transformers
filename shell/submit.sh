#!/bin/bash

# Fixed parameters
d_model=128
lr=2e-3
batch_size=32
num_epochs=5
seq_len=112
dataset_size=100
is_tqdm=True


# Array of embedding modes to test
emb_modes=('rnd_train' 'rnd_fix' 'given_train' 'given_fix')


# Loop through each mode and run the training script
for mode in "${emb_modes[@]}"; do
    echo "Running new_train.py with --emb_mode=$mode"

    python ./scripts/new_train.py \
        --emb_mode "$mode" \
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