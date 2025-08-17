#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name OMat
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=32:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12

#####
hostname
nvidia-smi
conda activate atom_denoise
##### 

cd ..
python finetune.py \
    --dataset_name "omat" \
    --target "None" \
    --lr 1e-04 \
    --epochs 10 \
    --model_name "gemnet" \
    --checkpoint_path "oc20_2M_5ep_LR1e-4_large_4hjzovpa" \
    --large \
    --enable_wandb \
    # --checkpoint_path "oc20_2M_5ep_LR1e-4_large_4hjzovpa" \
    # --checkpoint_path "oc22_2M_5ep_LR1e-5_large_kqhqf001" \
    # --checkpoint_path "ani1x_2M_5ep_LR1e-4_large_jhcdpc5m" \
    # --checkpoint_path "transition1x_2M_5ep_LR1e-4_large_vborctsj" \
    # --checkpoint_path "oc20_oc22_ani_tra_2M_5ep_tempSamp_br70oai3" \
    # --scratch \
