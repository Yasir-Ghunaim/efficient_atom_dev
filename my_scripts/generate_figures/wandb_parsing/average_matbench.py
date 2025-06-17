import pandas as pd

# Load the CSV file
csv_filename = "wandb_matbench_test_results.csv"
df = pd.read_csv(csv_filename)

df["Pretraining Mixed"] = df["Pretraining Mixed"].fillna("None")

df = df[df["Seed"].isin([0, 1])]

# Identify unique pairs of "Upstream Dataset" and "Pretraining Mixed"
unique_pairs = df[["Upstream Dataset", "Pretraining Mixed"]].drop_duplicates()

averaged_results = []
for _, pair in unique_pairs.iterrows():
    upstream_dataset = pair["Upstream Dataset"]
    pretraining_mixed = pair["Pretraining Mixed"]

    # Filter data for the current pair
    filtered_df = df[(df["Upstream Dataset"] == upstream_dataset) & (df["Pretraining Mixed"] == pretraining_mixed)]
       

    # Compute the average metric value for each fold across different seeds
    avg_per_fold = (
        filtered_df.groupby("Fold")["Metric Value"]
        .mean()
        .round(2)
        .reset_index()
        .assign(Upstream_Dataset=upstream_dataset, Pretraining_Mixed=pretraining_mixed)
    )

    # Compute the overall average across folds
    overall_avg = pd.DataFrame({
        "Fold": ["Overall"],
        "Metric Value": [round(avg_per_fold["Metric Value"].mean(), 2)],
        "Upstream_Dataset": upstream_dataset,
        "Pretraining_Mixed": pretraining_mixed
    })

    # Combine fold averages with overall average
    avg_results = pd.concat([avg_per_fold, overall_avg], ignore_index=True)
    averaged_results.append(avg_results)

# Concatenate results into a single DataFrame
averaged_df = pd.concat(averaged_results, ignore_index=True)

# Save to a new CSV file
output_avg_filename = "averaged_results_per_fold.csv"
averaged_df.to_csv(output_avg_filename, index=False)

