#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name effrank
#SBATCH -o output_pretrain/effrank.%A.out
#SBATCH -e output_pretrain/effrank.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6


cd ..

hostname
conda activate efficient_atom_den

# ------------------ config ------------------
MODEL="equiformer_v2"
checkpoint_tags=("OC20")   # e.g., OC20, ODAC25, MP, etc.

# Feature files expected at:
# dataset_features_${MODEL}_${TAG}/${NAME}_Node_Seed0_Sampling{balanced|random}.pt
upstream=("oc20" "oc22")
# downstream=("rmd17" "md22" "qm9" "spice" "omat")
# upstream=("ani1x")

# effective_rank.py CLI flags
STANDARDIZE="--standardize"
N_BOOT=5
SEED=0
# set --replace if you want true bootstrap-with-replacement; leave empty for subsampling
REPLACE_FLAG="--replace"   # or set to "--replace"
# --------------------------------------------

for tag in "${checkpoint_tags[@]}"; do
  echo "=== TAG: ${tag} ==="

  echo "--- Upstream files ---"
  for up in "${upstream[@]}"; do
    FEAT_PATH="dataset_features_${MODEL}_${tag}_20K/${up}_Node_Seed0_Samplingbalanced.pt"
    echo "[UP] ${FEAT_PATH}"
    python effective_rank.py "${FEAT_PATH}" ${STANDARDIZE} --n-boot ${N_BOOT} --seed ${SEED} ${REPLACE_FLAG}
    echo
  done


done
