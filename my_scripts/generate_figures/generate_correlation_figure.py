import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to fetch data and plot correlation
def plot_correlation_for_all_downstream_tasks(performance_df, fid_df):
    # Get unique downstream tasks
    downstream_tasks = performance_df["Dataset Name"].unique()

    # Create subplots
    fig, axes = plt.subplots(1, len(downstream_tasks), figsize=(20, 6))

    # Create a color palette for different sampling strategies
    sampling_palette = sns.color_palette("tab10", len(fid_df["Sampling"].unique()))
    sampling_colors = {sampling: color for sampling, color in zip(fid_df["Sampling"].unique(), sampling_palette)}

    # Define marker shapes for upstream tasks
    upstream_markers = ["o", "s", "D", "^"]  # Circle, Square, Diamond, Triangle
    upstream_tasks = fid_df["Upstream"].unique()
    marker_map = {upstream: upstream_markers[i % len(upstream_markers)] for i, upstream in enumerate(upstream_tasks)}

    for ax, downstream_task in zip(axes, downstream_tasks):
        # Filter performance data for the given downstream task
        performance_task = performance_df[performance_df["Dataset Name"] == downstream_task]

        # Filter FID data for the given downstream task
        fid_task = fid_df[fid_df["Downstream"] == downstream_task]

        # Merge the two datasets based on the upstream dataset
        merged = pd.merge(
            fid_task,
            performance_task,
            left_on="Upstream",
            right_on="Upstream Dataset",
            how="inner"
        )

        # Check if data is sufficient
        if merged.empty:
            print(f"No matching data found for downstream task: {downstream_task}")
            continue

        # Plot correlation
        for sampling in merged["Sampling"].unique():
            sampling_data = merged[merged["Sampling"] == sampling]
            ax.scatter(
                sampling_data["FID Score"],
                sampling_data["Metric Value"],
                color=sampling_colors[sampling],
                label=sampling if sampling not in ax.get_legend_handles_labels()[1] else "",
                s=100
            )
            
            # Fit and plot a best-fit line for this sampling
            # if len(sampling_data) > 1:
            #     x = sampling_data["FID Score"].values
            #     y = sampling_data["Metric Value"].values
            #     coeffs = np.polyfit(x, y, 1)  # Linear fit
            #     best_fit_line = np.poly1d(coeffs)
            #     ax.plot(
            #         x,
            #         best_fit_line(x),
            #         color=sampling_colors[sampling],
            #         linestyle="--",
            #         linewidth=1,
            #         label=f"{sampling} Best Fit" if f"{sampling} Best Fit" not in ax.get_legend_handles_labels()[1] else ""
            #     )

        # Add labels and title for each subplot
        ax.set_title(f"{downstream_task}", fontsize=14)
        ax.set_xlabel("FID Score (Log Scale)", fontsize=12)
        ax.set_xscale("log")  # Set x-axis to log scale
        ax.grid(True)

        # Set individual y-axis scale for each subplot
        ax.set_ylim(merged["Metric Value"].min() * 0.9, merged["Metric Value"].max() * 1.1)

    # Set common y-axis label
    axes[0].set_ylabel("Performance Metric (Lower is Better)", fontsize=14)

    # Create a legend for the entire figure
    handles = []
    labels = []
    for sampling, color in sampling_colors.items():
        for upstream, marker in marker_map.items():
            handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle="None", markersize=10))
            labels.append(f"{sampling} - {upstream}")
    fig.legend(handles, labels, loc="upper center", ncol=len(upstream_tasks), fontsize=10)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Correlation Across Downstream Tasks with Different Sampling Strategies", fontsize=16)
    plt.savefig("correlation_all_tasks_with_best_fit_log_scale.png")
    plt.show()

if __name__ == "__main__":
    # File paths
    performance_file = "main_exp_2M.csv"
    fid_file = "fid_scores_flat_gemnet_10k.csv"

    # Read the CSV files
    performance_df = pd.read_csv(performance_file)
    full_fid_df = pd.read_csv(fid_file)

    # Generate plots for all downstream tasks
    plot_correlation_for_all_downstream_tasks(performance_df, full_fid_df)