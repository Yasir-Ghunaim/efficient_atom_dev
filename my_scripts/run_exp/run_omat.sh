#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=35:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6

#  
#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..
# CUDA_VISIBLE_DEVICES=0 
python pretrain.py \
    --lr 1.0e-4 \
    --batch_size 18 \
    --num_workers 6 \
    --task "omat" \
    --epochs 5 \
    --sampling_strategy "random" \
    --checkpoint_path "transition1x_2M_5ep_fufhevh6" \
    --enable_wandb \

    # --train_samples_limit 1000000 \
    # --val_samples_limit 2000 \

    # --checkpoint_path "oc20_2M_5ep_ccts7u4o" \
    # --checkpoint_path "oc22_2M_5ep_vl2pj9xh" \
    # --checkpoint_path "oc20_2M_5ep_ccts7u4o" \
    # --checkpoint_path "transition1x_2M_5ep_fufhevh6" \
    # --checkpoint_path "ani1x_2M_5ep_o3n5226d" \
    # --checkpoint_path "oc20_oc22_ani_tra_2M_5ep_tempSamp_br70oai3" \