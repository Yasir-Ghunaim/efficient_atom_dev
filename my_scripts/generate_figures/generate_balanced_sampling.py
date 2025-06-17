import matplotlib.pyplot as plt
import json
from pathlib import Path
from jmp.datasets.sampling_utils import (
    get_molecule_df,
    apply_random_sampling,
    apply_class_balanced_sampling,
)
import scienceplots

plt.style.use(['science', 'no-latex']) 

# Paths to datasets
ani1x_path = Path("/home/ghunaiym/MSI_resources/datasets/ani1x/lmdb/train")
transition1x_path = Path("/home/ghunaiym/MSI_resources/datasets/transition1x/lmdb/train")
oc20_path = Path("/home/ghunaiym/MSI_resources/datasets/oc20_s2ef/2M/train")
oc22_path = Path("/home/ghunaiym/MSI_resources/datasets/oc22/s2ef-total/train")

# Path to save/load precomputed results
results_path = Path("precomputed_unique_classes.json")

# Function to compute unique classes for different sampling strategies
def compute_unique_classes(dataset_path, max_samples, seed):
    df = get_molecule_df(dataset_path)

    # Random sampling
    random_indices = apply_random_sampling(len(df), max_samples, seed)
    random_unique_classes = len(df.loc[random_indices]["Molecule"].value_counts())

    # Class-balanced sampling
    balanced_indices = apply_class_balanced_sampling(df, max_samples, seed, allow_repetition=True)
    balanced_unique_classes = len(df.loc[balanced_indices]["Molecule"].value_counts())

    return random_unique_classes, balanced_unique_classes, len(df["Molecule"].value_counts())

# Check if results file exists, otherwise compute
if results_path.exists():
    with open(results_path, "r") as file:
        results = json.load(file)
else:
    max_samples = 10000
    seed = 0

    # Compute unique classes for each dataset
    results = {
        "ANI1x": compute_unique_classes(ani1x_path, max_samples, seed),
        "Transition1x": compute_unique_classes(transition1x_path, max_samples, seed),
        "OC20": compute_unique_classes(oc20_path, max_samples, seed),
        "OC22": compute_unique_classes(oc22_path, max_samples, seed),
    }

    # Save results to a JSON file
    with open(results_path, "w") as file:
        json.dump(results, file)

# Extract data for plotting
datasets = ["ANI1x", "Transition1x", "OC20", "OC22"]
strategies = ["Random", "Class Balanced"]
unique_classes = [[r[0], r[1]] for r in results.values()]
total_unique_classes = [r[2] for r in results.values()]

# Plotting
fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=False)

for i, ax in enumerate(axes):
    ax.bar(strategies, unique_classes[i], color=["#fbb4ae", "#b3cde3"], edgecolor="black", linewidth=0.5)

    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)

    # Dynamic y-axis limits
    max_y = max(unique_classes[i] + [total_unique_classes[i]])
    max_y = max_y + 0.2 * max_y
    ax.set_ylim(0, max_y)

    # Set plot titles and labels
    ax.set_title(datasets[i], fontsize=16)
    if i == 0:
        ax.set_ylabel("Unique Classes", fontsize=16)
    ax.axhline(
        y=total_unique_classes[i],
        color="black",
        linestyle="--",
        label=f"Total Unique: {total_unique_classes[i]}",
    )
    ax.legend(loc="upper right")

    # Remove minor ticks and top ticks
    ax.minorticks_off()
    ax.tick_params(axis='x', which='both', top=False)

plt.tight_layout()
plt.savefig("upstream_sampling.png", transparent=True, dpi=300)
plt.show()
