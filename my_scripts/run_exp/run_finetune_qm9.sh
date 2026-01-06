#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name QM9
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#####
hostname
nvidia-smi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate efficient_atom
##### 

cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py \
    --dataset_name "qm9" \
    --target "U_0" \
    --lr 8e-5 \
    --epochs 300 \
    --medium \
    --model_name "equiformer_v2" \
    --batch_size 48 \
    --enable_wandb \
    --checkpoint_path "oc20_2M_5ep_eqv2_medium_o8hm117u"

    # --checkpoint_path "ani1x_2M_5ep_eqv2_medium_gm86vria" \
    # --checkpoint_path "transition1x_2M_5ep_eqv2_medium_oghu3gln" \
    # --checkpoint_path "oc20_2M_5ep_eqv2_medium_o8hm117u" \
    # --checkpoint_path "oc22_2M_5ep_eqv2_medium_tsd4qxie" \

# runpodctl stop pod x1ex7unq9ycogw