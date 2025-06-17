import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])  # Apply scienceplot style


# Renaming mappings for better readability in plots
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

def plot_fid_bar_with_ranges(
    fid_df,
    sampling_strategy,
    downstream_list,
    output_filename,
):

    filtered_fid_df = fid_df[
        (fid_df["Sampling"] == sampling_strategy) & 
        (fid_df["Downstream"].isin(downstream_list))
    ].copy()

    # Rename upstream and downstream datasets
    filtered_fid_df['Upstream'] = filtered_fid_df['Upstream'].replace(UPSTREAM)
    filtered_fid_df['Downstream'] = filtered_fid_df['Downstream'].replace(DOWNSTREAM)

    # Group by Downstream and Upstream, calculate mean, min, and max
    stats = filtered_fid_df.groupby(['Downstream', 'Upstream']).agg(
        mean_fid=('FID Score', 'mean'),
        min_fid=('FID Score', 'min'),
        max_fid=('FID Score', 'max')
    ).reset_index()
    
    # Sort data for consistent plotting order
    stats['Upstream'] = pd.Categorical(stats['Upstream'], UPSTREAM.values())
    stats['Downstream'] = pd.Categorical(stats['Downstream'], DOWNSTREAM.values())
    stats = stats.sort_values(['Downstream', 'Upstream'])

    # Plot the bar chart
    num_downstream_datasets = len(stats['Downstream'].unique())
    x_positions = range(num_downstream_datasets)
    fig, ax = plt.subplots(figsize=(int(num_downstream_datasets*2.5), 3.5))
    bar_width = 0.2  # Width of each bar

    colors = [
        '#b2c9ab', # Green
        '#7c9885', # Blue
        '#28666e', # Red    
        '#033f63']   # purple

    # Iterate over upstream datasets to plot each group with error bars
    for i, (upstream, color) in enumerate(zip(UPSTREAM.values(), colors)):
        upstream_stats = stats[stats['Upstream'] == upstream]
        bar_x = [x + i * bar_width for x in x_positions]
        ax.bar(
            bar_x,
            upstream_stats['mean_fid'],
            yerr=[
                upstream_stats['mean_fid'] - upstream_stats['min_fid'],
                upstream_stats['max_fid'] - upstream_stats['mean_fid']
            ],
            width=bar_width,
            label=upstream,
            capsize=5,
            color=color,
        )


    # Add labels and title
    ax.set_xticks([x + 1.5 * bar_width for x in x_positions])
    ax.set_xticklabels(stats['Downstream'].unique(), fontsize=14)
    # change ytick font size
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel("Downstream Dataset", fontsize=16)
    ax.set_ylabel("CSI", fontsize=16)

    # ax.legend(title="Upstream", fontsize=12)
    # ax.legend(title="Upstream", fontsize=12, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

    if num_downstream_datasets == 2:
        ax.legend(
            # title="Upstream",
            fontsize=12,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.4),  # Center above the plot
            ncol=2,  # Arrange legend entries in a single row
            frameon=True  # Optional: remove legend frame
        )
    else:
        ax.legend(
            # title="Upstream",
            fontsize=14,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),  # Center above the plot
            ncol=4,  # Arrange legend entries in a single row
            frameon=True  # Optional: remove legend frame
        )

    plt.tight_layout()

    # Remove minor ticks and top ticks
    plt.minorticks_off()
    plt.tick_params(axis='x', which='both', top=False)

    # Save and display the plot
    plt.savefig(output_filename)#, transparent=True, dpi=300)
    plt.show()


if __name__ == "__main__":
    # File path
    fid_file = "fid_scores_flat_equiformer_v2_10k.csv"

    # Read the CSV file
    fid_df = pd.read_csv(fid_file)
    sampling_strategy = "balancedRep"
    output_filename = "CSI_bar_balanced_equiformerV2_flat.png"

    downstream_list_1 = ['rmd17', 'md22', 'spice', 'qm9']
    output_filename_1 = "CSI_bar_balanced_flat_equiformerV2_ID.png"
    plot_fid_bar_with_ranges(fid_df, sampling_strategy=sampling_strategy, downstream_list=downstream_list_1, output_filename=output_filename_1)

    downstream_list_2 = ['qmof', 'matbench_fold0']
    output_filename_2 = "CSI_bar_balanced_flat_equiformerV2_OOD.png"
    plot_fid_bar_with_ranges(fid_df, sampling_strategy=sampling_strategy, downstream_list=downstream_list_2, output_filename=output_filename_2)

    # Main random
    sampling_strategy = "random"
    downstream_list_3 = ['rmd17', 'md22', 'spice', 'qm9']
    output_filename_3 = "CSI_bar_random_flat_equiformerV2_ID.png"
    plot_fid_bar_with_ranges(fid_df, sampling_strategy=sampling_strategy, downstream_list=downstream_list_3, output_filename=output_filename_3)


    # Main random
    fid_file = "fid_scores_mean_equiformer_v2_10k.csv"
    fid_df = pd.read_csv(fid_file)
    sampling_strategy = "balancedRep"
    downstream_list_4 = ['rmd17', 'md22', 'spice', 'qm9']
    output_filename_4 = "CSI_bar_balanced_mean_equiformerV2_ID.png"
    plot_fid_bar_with_ranges(fid_df, sampling_strategy=sampling_strategy, downstream_list=downstream_list_4, output_filename=output_filename_4)


    # JMP
    fid_file = "fid_scores_flat_JMP_10k.csv"
    fid_df = pd.read_csv(fid_file)
    sampling_strategy = "balancedRep"
    downstream_list_5 = ['rmd17', 'md22', 'spice', 'qm9']
    output_filename_5 = "CSI_bar_balanced_flat_JMP_ID.png"
    plot_fid_bar_with_ranges(fid_df, sampling_strategy=sampling_strategy, downstream_list=downstream_list_5, output_filename=output_filename_5)

