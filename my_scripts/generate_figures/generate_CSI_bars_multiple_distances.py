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
    'matbench_fold0': 'MatBench',
    'omat': 'OMat24',           # add OMat24
}

# Try a few common score column names and standardize to "Score"
CANDIDATE_SCORE_COLS = [
    'CSI Score', 'MMD Score', 'Energy Distance', 'ED Score', 'Score', 'value'
]

def _standardize_score_column(df):
    for c in CANDIDATE_SCORE_COLS:
        if c in df.columns:
            df = df.rename(columns={c: 'Score'})
            return df
    raise KeyError(
        f"None of expected score columns found. Available columns: {list(df.columns)}"
    )

def plot_metric_bar(df, sampling_strategy, downstream_list, ax, title, ylabel):
    df = _standardize_score_column(df)

    filtered = df[
        (df["Sampling"] == sampling_strategy) &
        (df["Downstream"].isin(downstream_list))
    ].copy()

    # Rename for nicer labels
    filtered['Upstream'] = filtered['Upstream'].replace(UPSTREAM)
    filtered['Downstream'] = filtered['Downstream'].replace(DOWNSTREAM)

    # Aggregate
    stats = filtered.groupby(['Downstream', 'Upstream']).agg(
        mean_score=('Score', 'mean'),
        min_score=('Score', 'min'),
        max_score=('Score', 'max')
    ).reset_index()

    # Respect categorical order; keep only present categories
    present_upstreams = [u for u in UPSTREAM.values() if u in stats['Upstream'].unique()]
    present_downstreams = [d for d in DOWNSTREAM.values() if d in stats['Downstream'].unique()]
    stats['Upstream'] = pd.Categorical(stats['Upstream'], present_upstreams)
    stats['Downstream'] = pd.Categorical(stats['Downstream'], present_downstreams)
    stats = stats.sort_values(['Downstream', 'Upstream'])

    # Plot
    colors = ['#b2c9ab', '#7c9885', '#28666e', '#033f63']
    bar_width = 0.2
    x_positions = range(len(present_downstreams))

    for i, (upstream, color) in enumerate(zip(present_upstreams, colors)):
        u_stats = stats[stats['Upstream'] == upstream]
        # align bars even if some upstreams are missing
        means = []
        for d in present_downstreams:
            row = u_stats[u_stats['Downstream'] == d]
            means.append(row['mean_score'].values[0] if not row.empty else float('nan'))
        bar_x = [x + i * bar_width for x in x_positions]
        ax.bar(bar_x, means, width=bar_width, label=upstream, color=color)

    ax.set_xticks([x + (len(present_upstreams)-1) * bar_width / 2 for x in x_positions])
    ax.set_xticklabels(present_downstreams, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=min(4, len(present_upstreams)),
              loc="upper center", bbox_to_anchor=(0.5, 1.22))
    plt.minorticks_off()
    plt.tick_params(axis='x', which='both', top=False)

if __name__ == "__main__":
    # ID-only list (includes OMat24 per your note)
    downstream_list_id = ['rmd17', 'md22', 'spice', 'qm9', 'omat']
    sampling_strategy = "balanced"

    files = [
        ("csi_scores_flat_equiformer_v2_OC20.csv",          "Chemical Similarity Index (CSI)", "CSI"),
        ("mmd_scores_flat_equiformer_v2_OC20_fast.csv",     "Maximum Mean Discrepancy (MMD)",  "MMD"),
        ("energy_distance_scores_flat_equiformer_v2_OC20.csv","Energy Distance",                "Energy Distance"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), constrained_layout=True)

    for ax, (file, title, ylabel) in zip(axes, files):
        df = pd.read_csv(file)
        plot_metric_bar(df, sampling_strategy, downstream_list_id, ax, title, ylabel)

    plt.savefig("alignment_metrics_grid.png", dpi=300)
    plt.show()
