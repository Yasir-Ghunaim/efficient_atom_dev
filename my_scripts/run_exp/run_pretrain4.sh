#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name ani1x
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=24
#SBATCH --account conf-icml-2026.01.29-ghanembs

#####
hostname
nvidia-smi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate efficient_atom
##### 

cd ..
python pretrain.py \
    --lr 1.0e-4 \
    --batch_size 15 \
    --num_workers 6 \
    --train_samples_limit 2000000 \
    --val_samples_limit 2000 \
    --task "ani1x" \
    --epochs 5 \
    --sampling_strategy "random" \
    --model_name "equiformer_v2" \
    --logging_path "/workspace/logs" \
    --enable_wandb \

runpodctl stop pod 8tfu0e7bw4by9r