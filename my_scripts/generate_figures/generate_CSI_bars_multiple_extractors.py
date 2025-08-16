import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

plt.style.use(['science', 'no-latex'])

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
    'omat': 'OMat24',
    'qmof': 'QMOF',
    'matbench_fold0': 'MatBench'
}

FE_DISPLAY = {
    "OC20": "OC20",
    "ODAC": "ODAC25",
    "MPtrj": "MPtrj",
    "MPtrjDeNS": "MPtrj (DeNS)"
}

# consistent colors per Upstream (so colors match across panels)
PALETTE = ['#b2c9ab', '#7c9885', '#28666e', '#033f63']

def load_csi(feature_extractors, base_pattern="csi_scores_flat_equiformer_v2_{fe}.csv"):
    """Load and concat CSVs; add FeatureExtractor column."""
    frames = []
    for fe in feature_extractors:
        f = base_pattern.format(fe=fe)
        df = pd.read_csv(f)
        df['FeatureExtractor'] = fe
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def compute_stats(df, sampling_strategy, downstream_list):
    filtered = df[
        (df["Sampling"] == sampling_strategy) &
        (df["Downstream"].isin(downstream_list))
    ].copy()

    filtered['Upstream'] = filtered['Upstream'].replace(UPSTREAM)
    filtered['Downstream'] = filtered['Downstream'].replace(DOWNSTREAM)

    stats = (filtered
             .groupby(['FeatureExtractor','Downstream','Upstream'])
             .agg(mean_csi=('CSI Score','mean'),
                  min_csi=('CSI Score','min'),
                  max_csi=('CSI Score','max'))
             .reset_index())

    # ordering
    stats['Upstream'] = pd.Categorical(stats['Upstream'], UPSTREAM.values())
    stats['Downstream'] = pd.Categorical(stats['Downstream'], DOWNSTREAM.values())
    stats = stats.sort_values(['FeatureExtractor','Downstream','Upstream'])
    return stats

def plot_csi_grid(stats, feature_extractors, downstream_list, output_filename):
    # figure geometry
    n_panels = len(feature_extractors)
    nrows, ncols = 2, 2  # for 4 extractors
    assert n_panels <= nrows*ncols, "Increase grid size for more extractors."

    # Size scales with #downstream for legibility
    num_ds = len(downstream_list)
    if num_ds >= 5:
        fig, axes = plt.subplots(nrows, ncols, figsize=(num_ds*2.3, 7.2), sharey=False)
        bar_width = 0.2
        xtick_fs, ytick_fs, label_fs = 14, 14, 16
        legend_fs = 16
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(num_ds*1.8, 6.0), sharey=False)
        bar_width = 0.2
        xtick_fs, ytick_fs, label_fs = 12, 12, 14
        legend_fs = 11

    axes = axes.flatten()
    handles_for_legend = None
    labels_for_legend = None

    for ax, fe in zip(axes, feature_extractors):
        sub = stats[stats['FeatureExtractor'] == fe]
        # ensure per-panel Downstream order matches
        ds_vals = list(pd.Categorical(sub['Downstream'], categories=DOWNSTREAM.values()).unique())
        ds_vals = [d for d in ds_vals if pd.notna(d)]
        x_positions = range(len(ds_vals))

        # draw grouped bars by Upstream
        for i, (up, color) in enumerate(zip(UPSTREAM.values(), PALETTE)):
            up_stats = sub[sub['Upstream'] == up]
            # align to current panel ds order
            up_stats = up_stats.set_index('Downstream').reindex(ds_vals).reset_index()

            bar_x = [x + i*bar_width for x in x_positions]
            bars = ax.bar(
                bar_x,
                up_stats['mean_csi'].values,
                width=bar_width,
                label=up,
                color=color
            )
            # remember legend once
            if handles_for_legend is None:
                handles_for_legend = [bars]
                labels_for_legend = [up]
            else:
                handles_for_legend.append(bars)
                labels_for_legend.append(up)

        # x ticks in the middle of group
        ax.set_xticks([x + 1.5*bar_width for x in x_positions])
        ax.set_xticklabels(ds_vals, fontsize=xtick_fs)
        ax.tick_params(axis='y', labelsize=ytick_fs)

        # log y
        # ax.set_yscale("log")

        ax.set_xlabel("Downstream Dataset", fontsize=label_fs)
        ax.set_ylabel("CSI", fontsize=label_fs)
        ax.set_title(f"Feature Extractor: {FE_DISPLAY.get(fe, fe)}", fontsize=label_fs+1, pad=8)

        # neaten x-limits
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.05, xmax + 0.05)

        # minor ticks off on x, top ticks off
        ax.minorticks_off()
        ax.tick_params(axis='x', which='both', top=False)

    # If fewer than 4 panels, hide empty axes
    for j in range(len(feature_extractors), len(axes)):
        axes[j].axis('off')

    # single shared legend above all
    fig.legend(handles_for_legend[:len(UPSTREAM)], list(UPSTREAM.values()),
               loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4,
               fontsize=legend_fs, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(output_filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    feature_extractors = ["OC20", "ODAC", "MPtrj", "MPtrjDeNS"]
    sampling_strategy = "balanced"
    downstream_list = ['rmd17', 'md22', 'spice', 'qm9', 'omat']

    df = load_csi(feature_extractors, base_pattern="csi_scores_flat_equiformer_v2_{fe}.csv")
    stats = compute_stats(df, sampling_strategy, downstream_list)
    plot_csi_grid(stats, feature_extractors, downstream_list, output_filename="CSI_bar_grid_ID.png")
