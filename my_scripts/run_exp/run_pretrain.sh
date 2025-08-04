#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name denoise
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --account conf-aaai-2025.08.04-ghanembs

#  
#####
hostname
nvidia-smi
conda activate efficient_atom_den
##### 

cd ..

# python pretrain.py \
export OMP_NUM_THREADS=2
torchrun --nproc_per_node=2 pretrain.py \
    --lr 1.0e-4 \
    --batch_size 20 \
    --num_workers 6 \
    --train_samples_limit 1000000 \
    --val_samples_limit 2000 \
    --task "transition1x" \
    --epochs 5 \
    --sampling_strategy "random" \
    --model_name "equiformer_v2" \
    --enable_wandb 