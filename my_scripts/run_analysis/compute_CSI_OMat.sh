cd ..

# Define upstream datasets (excluding oc22)
upstream=("ani1x" "transition1x" "oc20" "oc22")

# Set OC22 as the downstream task
downstream="omat"

# Loop through upstream datasets
for up in "${upstream[@]}"; do
    python compute_csi.py \
        "${up}_Node_Seed0_Samplingbalanced.pt" \
        "${downstream}_Node_Seed0_Samplingbalanced.pt" \
        --model_name "equiformer_v2"
done