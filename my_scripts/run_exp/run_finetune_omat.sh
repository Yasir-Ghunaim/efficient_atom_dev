#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name OMat
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=24

#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..
python finetune.py \
    --dataset_name "omat" \
    --target "None" \
    --lr 1e-04 \
    --epochs 12 \
    --model_name "gemnet" \
    --checkpoint_path "oc20_2M_5ep_ccts7u4o" \
    --enable_wandb \
    # --checkpoint_path "oc22_2M_5ep_vl2pj9xh" \
    # --checkpoint_path "oc20_2M_5ep_ccts7u4o" \
    # --checkpoint_path "transition1x_2M_5ep_fufhevh6" \
    # --checkpoint_path "ani1x_2M_5ep_o3n5226d" \
    # --checkpoint_path "oc20_oc22_ani_tra_2M_5ep_tempSamp_br70oai3" \
    # --scratch \
