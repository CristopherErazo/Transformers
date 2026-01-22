#!/bin/bash

# Fixed parameters
d=512
dataset_size=10000
L=128
rank=-1
batch_size=64
num_epochs=20
is_tqdm=False
sigma=0.1
n_prints=200
nprint_matrices=7
skip_residual=False


# Loop over various configurations = ( lr , beta, freeze_embeddings, freeze_attention,type_enc )
configurations=(
    "0.05 1.0 False False wave"
)

for config in "${configurations[@]}"; do
    read -r lr beta fr_emb fr_att type_enc <<< "$config"
    echo "Running with --lr=$lr --beta=$beta --fr_emb=$fr_emb --fr_att=$fr_att --type_enc=$type_enc"

    python -u ./scripts/organized_test.py \
        --d $d \
        --dataset_size $dataset_size \
        --batch_size $batch_size \
        --L $L \
        --rank $rank \
        --num_epochs $num_epochs \
        --lr $lr \
        --is_tqdm $is_tqdm \
        --fr_emb $fr_emb \
        --fr_att $fr_att \
        --type_enc $type_enc \
        --skip_residual $skip_residual \
        --beta $beta \
        --sigma $sigma \
        --n_prints $n_prints\
        --nprints_matrices $nprint_matrices

    echo "Completed: $mode at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"


