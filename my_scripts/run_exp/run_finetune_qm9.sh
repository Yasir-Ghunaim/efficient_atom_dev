#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name QM9
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=24

#####
hostname
nvidia-smi
conda activate atom_denoise
#####

cd ..
python finetune.py \
    --dataset_name "qm9" \
    --target "U_0" \
    --lr 5e-4 \
    --epochs 300 \
    --model_name "equiformer_v2" \
    --checkpoint_path "oc22_1M_5ep_eqv2_4jtwwm1v_EMA" \
    --enable_wandb \
    # --scratch \
    