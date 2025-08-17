#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name OMat
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=14:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=100

#####
conda activate atom_denoise
##### 

python -m jmp.datasets.scripts.omat_preprocess.omat_linear_ref linref \
  --src /ibex/project/c2261/datasets/omat/train/rattled-300-subsampled \
  --out_path /ibex/project/c2261/datasets/omat/train/linref.npz