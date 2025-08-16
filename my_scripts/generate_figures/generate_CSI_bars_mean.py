import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as mticker


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
    'qmof': 'QMOF',
    'matbench_fold0': 'MatBench',
    'omat': 'OMat24',
}

def plot_csi_bar_on_ax(fid_df, sampling_strategy, downstream_list, ax, title, show_legend=True):
    filtered = fid_df[
        (fid_df["Sampling"] == sampling_strategy) &
        (fid_df["Downstream"].isin(downstream_list))
    ].copy()

    filtered['Upstream'] = filtered['Upstream'].replace(UPSTREAM)
    filtered['Downstream'] = filtered['Downstream'].replace(DOWNSTREAM)

    stats = filtered.groupby(['Downstream', 'Upstream']).agg(
        mean_fid=('CSI Score', 'mean'),
        min_fid=('CSI Score', 'min'),
        max_fid=('CSI Score', 'max')
    ).reset_index()

    stats['Upstream'] = pd.Categorical(stats['Upstream'], UPSTREAM.values())
    stats['Downstream'] = pd.Categorical(stats['Downstream'], DOWNSTREAM.values())
    stats = stats.sort_values(['Downstream', 'Upstream'])

    colors = ['#b2c9ab', '#7c9885', '#28666e', '#033f63']
    num_down = len(stats['Downstream'].unique())
    x_positions = range(num_down)
    bar_width = 0.2 if num_down >= 3 else 0.07

    for i, (upstream, color) in enumerate(zip(UPSTREAM.values(), colors)):
        u_stats = stats[stats['Upstream'] == upstream]
        if num_down == 2:
            group_spacing = 0.4
            bar_x = [(x * group_spacing) + i * bar_width for x in x_positions]
            ax.set_xticks([x * group_spacing + (len(UPSTREAM) - 1) * bar_width / 2 for x in x_positions])
        else:
            bar_x = [x + i * bar_width for x in x_positions]
            ax.set_xticks([x + 1.5 * bar_width for x in x_positions])

        ax.bar(bar_x, u_stats['mean_fid'], width=bar_width, label=upstream, color=color)

    ax.set_xticklabels(stats['Downstream'].unique(), fontsize=12 if num_down <= 4 else 10)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel("Downstream Dataset", fontsize=13)
    ax.set_ylabel("CSI", fontsize=13)
    ax.set_title(title, fontsize=14)

    ax.minorticks_off()
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.tick_params(axis='x', which='both', top=False)

    if show_legend:
        ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=4, frameon=True)


if __name__ == "__main__":
    sampling_strategy = "balanced"
    downstream_list_id = ['rmd17', 'md22', 'spice', 'qm9', 'omat']

    flat_file = "csi_scores_flat_equiformer_v2_OC20.csv"
    mean_file = "csi_scores_mean_equiformer_v2_OC20.csv"

    df_flat = pd.read_csv(flat_file)
    df_mean = pd.read_csv(mean_file)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.0), constrained_layout=False, sharey=False)
    fig.subplots_adjust(hspace=0.5)

    plot_csi_bar_on_ax(df_flat, sampling_strategy, downstream_list_id, axes[0],
                       "Node-level (no pooling)", show_legend=True)
    plot_csi_bar_on_ax(df_mean, sampling_strategy, downstream_list_id, axes[1],
                       "Graph-level (mean pooling)", show_legend=False)

    plt.savefig("CSI_flat_vs_mean_ID.png", dpi=300)
    plt.show()
