#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#  
#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..
CUDA_VISIBLE_DEVICES=1 python pretrain.py \
    --lr 1.0e-4 \
    --batch_size 20 \
    --num_workers 6 \
    --train_samples_limit 4000000 \
    --val_samples_limit 2000 \
    --task "ani1x" \
    --epochs 5 \
    --sampling_strategy "random" \
    --enable_wandb \