#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name extract
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#  
#####
hostname
nvidia-smi
conda activate efficient_atom_den
##### 

cd ..

tasks=("rmd17" "md22" "qm9" "spice") # "qmof" "matbench")
# checkpoint_tags=("OC20" "ODAC" "MP")
checkpoint_tags=("OC20")

# Iterate over all tasks
for tag in "${checkpoint_tags[@]}"; do
  for task in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python extract_finetune.py \
      --dataset_name $task \
      --batch_size 5 \
      --number_of_samples 10000 \
      --model_name "equiformer_v2" \
      --seed 0 \
      --checkpoint_tag "$tag"
  done
done