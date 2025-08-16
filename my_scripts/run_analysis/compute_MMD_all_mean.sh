#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name extract
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6


cd ..

#  
#####
hostname
conda activate efficient_atom_den
##### 

# Define upstream and downstream datasets
upstream=("ani1x" "transition1x" "oc20" "oc22")
downstream=("rmd17" "md22" "qm9" "spice" "omat")
# checkpoint_tags=("OC20" "MP" "ODAC" "MPDense")
checkpoint_tags=("OC20")


# Loop through upstream and downstream datasets
for tag in "${checkpoint_tags[@]}"; do
    for down in "${downstream[@]}"; do
        for up in "${upstream[@]}"; do
            python compute_mmd_mean.py \
                "${up}_Node_Seed0_Samplingbalanced.pt" \
                "${down}_Node_Seed0_Samplingrandom.pt" \
                --model_name "equiformer_v2" \
                --checkpoint_tag "$tag" \
                --device "cuda:0" \
                --agg_operation "mean" 
        done
    done
done
