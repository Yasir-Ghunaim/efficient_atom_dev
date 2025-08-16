import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])

# Renaming mappings
UPSTREAM = {
    'ani1x': 'ANI-1x',
    'transition1x': 'Transition-1x',
    'oc20': 'OC20',
    'oc22': 'OC22'
}

DOWNSTREAM = {
    'rmd17': 'rMD17',
    'md22': 'MD22',
    'spice': 'SPICE',
    'qm9': 'QM9',
    'qmof': 'QMOF',
    'matbench_fold0': 'MatBench'
}

def plot_fid_bar_with_ranges(fid_df, sampling_strategy, downstream_list, ax, title):
    filtered_fid_df = fid_df[
        (fid_df["Sampling"] == sampling_strategy) &
        (fid_df["Downstream"].isin(downstream_list))
    ].copy()

    filtered_fid_df['Upstream'] = filtered_fid_df['Upstream'].replace(UPSTREAM)
    filtered_fid_df['Downstream'] = filtered_fid_df['Downstream'].replace(DOWNSTREAM)

    stats = filtered_fid_df.groupby(['Downstream', 'Upstream']).agg(
        mean_fid=('MMD Score', 'mean'),
        min_fid=('MMD Score', 'min'),
        max_fid=('MMD Score', 'max')
    ).reset_index()

    stats['Upstream'] = pd.Categorical(stats['Upstream'], UPSTREAM.values())
    stats['Downstream'] = pd.Categorical(stats['Downstream'], DOWNSTREAM.values())
    stats = stats.sort_values(['Downstream', 'Upstream'])

    bar_width = 0.2
    colors = ['#b2c9ab', '#7c9885', '#28666e', '#033f63']
    x_positions = range(len(stats['Downstream'].unique()))

    for i, (upstream, color) in enumerate(zip(UPSTREAM.values(), colors)):
        upstream_stats = stats[stats['Upstream'] == upstream]
        bar_x = [x + i * bar_width for x in x_positions]
        ax.bar(
            bar_x,
            upstream_stats['mean_fid'],
            width=bar_width,
            label=upstream,
            color=color
        )

    ax.set_xticks([x + 1.5 * bar_width for x in x_positions])
    ax.set_xticklabels(stats['Downstream'].unique(), fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel("MMD", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    plt.minorticks_off()
    plt.tick_params(axis='x', which='both', top=False)


if __name__ == "__main__":
    downstream_list_id = ['rmd17', 'md22', 'spice', 'qm9', 'omat']
    sampling_strategy = "balanced"

    files_and_titles = [
        ("mmd_scores_flat_equiformer_v2_OC20_500.csv", "OC20 500 (flat)"),
        ("mmd_scores_flat_equiformer_v2_OC20_30K.csv", "OC20 30K (flat)"),
        ("mmd_scores_flat_equiformer_v2_OC20_fast.csv", "OC20 fast (flat)"),
        ("mmd_scores_mean_equiformer_v2_OC20.csv", "OC20 (mean)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), constrained_layout=True)

    for ax, (file, title) in zip(axes, files_and_titles):
        df = pd.read_csv(file)
        plot_fid_bar_with_ranges(df, sampling_strategy, downstream_list_id, ax, title)

    plt.savefig("mmd_grid.png", dpi=300)
    plt.show()
