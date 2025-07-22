#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name CSI
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#  
#####
hostname
nvidia-smi
conda activate efficient_atom
##### 


cd ..

# Define upstream and downstream datasets
upstream=("ani1x" "transition1x" "oc20" "oc22")
downstream=("rmd17" "md22" "qm9" "spice" "qmof" "matbench_fold0")

# Loop through upstream and downstream datasets
for down in "${downstream[@]}"; do
    for up in "${upstream[@]}"; do
        python compute_csi.py \
            "${up}_Node_Seed0_Samplingbalanced.pt" \
            "${down}_Node_Seed0_Samplingrandom.pt" \
            --model_name "equiformer_v2" 
    done
done





