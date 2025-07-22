#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name rMD17
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --dataset_name "omat" \
    --target "None" \
    --lr 1e-05 \
    --epochs 50 \
    --model_name "gemnet" \
    --scratch \
    # --enable_wandb \
    # --checkpoint_path "oc22_1M_5ep_eqv2_4jtwwm1v_EMA" \
