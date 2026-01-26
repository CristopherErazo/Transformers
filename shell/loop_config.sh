#!/bin/bash

# Meta parameters
is_tqdm=False
n_prints=400
nprint_matrices=8

# Fixed Model Parameters
d=512
dataset_size=10000
L=128
rank=-1
batch_size=64
num_epochs=100

# Architectural Parameters
skip_residual=False
fr_emb=False
fr_att=False

# Hyperparameters
beta=1.0
lr=0.05
# P='tiny'  # 'uni' for uniform distribution, 'tiny' for TinyStories distribution


# Loop over various configurations = sigma, amp, type_enc,  unmb
configurations=(
    # "0.1 1.0 wave False" # Unbalanced with wave encodings
    # "0.5 0.5 wave False" # Balanced with wave encodings
    # "0.5 1.0 learned False" # learned encodings
    "0.5 1.0 learned True tiny" # Learned encodings and untied unembedddings
    "0.5 1.0 learned True uni" # Learned encodings and untied unembedddings
)

for config in "${configurations[@]}"; do
    read -r  sigma amp type_enc unmb Pdat <<< "$config"
    
    python -u ./scripts/random_data.py \
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
        --nprints_matrices $nprint_matrices\
        --amp $amp \
        --Pdat $Pdat \
        --unmb $unmb  
        

    echo "Completed: $mode at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"


