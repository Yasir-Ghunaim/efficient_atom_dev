#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name pretrain
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
conda activate efficient_atom
##### 

cd ..

# Here we set limit to 5M to ensure we get all samples of ANI-1X (<4M)
CUDA_VISIBLE_DEVICES=0 python extract_pretrain.py \
  --task "ani1x" \
  --batch_size 15 \
  --train_samples_limit 5000000 \ 
  --sampling_strategy "random" \
  --model_name "equiformer_v2" \
  --compute_sample_difficulty \
  --seed 0