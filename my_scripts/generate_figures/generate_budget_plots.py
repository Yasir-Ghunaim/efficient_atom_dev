import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science', 'no-latex'])

# Use seaborn's Set2 color palette
colors = sns.color_palette("Set2", 4)


# Define budgets
budgets = [0.5, 1, 2, 3]

# Upstream datasets
upstreams = ["ANI-1x", "Transition-1x", "OC20", "OC22"]

# Downstream performance values for each task
# Format: [0.5M, 1M, 2M, 3M]
performance = {
    "rMD17": {
        "ANI-1x": [7.0, 5.8, 5.4, 5.6],
        "Transition-1x": [13.0, 10.6, 10.1, 10.2],
        "OC20": [14.2, 12.8, 14.6, 16.1],
        "OC22": [18.5, 16.8, 16.0, 17.4]
    },
    "MD22": {
        "ANI-1x": [3.43, 3.08, 2.90, 2.85],
        "Transition-1x": [4.44, 3.96, 3.73, 3.64],
        "OC20": [5.10, 4.70, 4.53, 4.61],
        "OC22": [5.66, 5.24, 5.20, 5.15]
    },
    "SPICE": {
        "ANI-1x": [6.48, 5.42, 5.13, 5.37],
        "Transition-1x": [9.19, 7.78, 7.55, 7.65],
        "OC20": [9.82, 9.18, 8.74, 10.44],
        "OC22": [13.05, 11.06, 10.73, 11.00]
    },
    "QM9": {
        "ANI-1x": [3.2, 3.0, 2.9, 2.8],
        "Transition-1x": [3.7, 3.4, 3.2, 3.2],
        "OC20": [4.9, 4.7, 4.8, 5.3],
        "OC22": [5.8, 5.4, 5.7, 5.8]
    }
}

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(18, 6))
task_names = list(performance.keys())

for i, task in enumerate(task_names):
    ax = axs[i]
    for j, upstream in enumerate(upstreams):
        ax.plot(budgets, performance[task][upstream], label=upstream, marker='o', color=colors[j], linewidth=3, markersize=10)
    ax.set_title(task, fontsize=26)
    ax.set_xlabel("Pretraining Budget (Millions)", fontsize=22)
    if i == 0:
        ax.set_ylabel("MAE", fontsize=26)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=26)
    ax.tick_params(axis='both', which='major', width=1.5, length=4)

    # Remove minor ticks and top ticks
    ax.minorticks_off()
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)
    ax.grid(True, linewidth=1.0)
    ax.set_xticks(budgets)
    ax.set_xlim(min(budgets) - 0.2, max(budgets) + 0.2)
    ax.margins(y=0.18)
    
    if task == "SPICE":
        ax.set_yticks(range(5, 15, 2))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# Common legend on top
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Pretraining Dataset",
    title_fontsize=26,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),
    ncol=4,
    fontsize=26,
    frameon=True
)


plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig("pretraining_budget.png", dpi=300)

