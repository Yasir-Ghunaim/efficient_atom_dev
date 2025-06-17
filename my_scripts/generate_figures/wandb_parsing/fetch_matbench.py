import wandb
import pandas as pd
import re
import ast
import csv
from tqdm import tqdm
import json

# Initialize WandB API
api = wandb.Api()


full_df = pd.read_csv("wandb_run_details.csv")
filtered_df = full_df[full_df["Job Name"].str.contains("matbench", case=False, na=False) & 
                    full_df["Job Name"].str.contains("FINAL", case=False, na=False) & 
                    ~full_df["Job Name"].str.contains("3M", case=True, na=False)]

filtered_csv_filename = "wandb_matbench_runs.csv"
filtered_df.to_csv(filtered_csv_filename, index=False)
df = filtered_df

metrics = {
    "matbench": "test/matbench/phonons/y_mae",
}

def parse_checkpoint_path(checkpoint_path):
    """
    Parses the checkpoint path to extract the upstream dataset, pretraining size,
    pretraining epochs, and WandB run ID.

    Args:
        checkpoint_path (str): The checkpoint path string to parse.

    Returns:
        tuple: Parsed components of the checkpoint path (dataset, size, epochs, run_id).
    """
    # Define the regex pattern for parsing the checkpoint path
    pattern = r"(?P<dataset>[a-zA-Z0-9]+)_(?P<size>\d+[kM])_(?P<epochs>\d+ep)_(?P<run_id>[a-zA-Z0-9]+)"
    
    match = re.match(pattern, checkpoint_path)
    if match:
        dataset = match.group("dataset")
        size = match.group("size")
        epochs = match.group("epochs")
        run_id = match.group("run_id")
        return dataset, size, epochs, None, run_id
    else:
        return None, None, None, None, None  # Return None for invalid format


def parse_mixed_checkpoint_path(checkpoint_path):
    """
    Parses the checkpoint path to extract dataset name(s), pretraining size, epochs, extra info, and run ID.

    Args:
        checkpoint_path (str): The checkpoint path string to parse.

    Returns:
        tuple: (datasets, pretraining size, pretraining epochs, extra info, run_id)
    """
    pattern = (
        r"(?P<dataset>(?:[a-zA-Z0-9]+_?)+)_"  # Match multiple datasets like oc20_oc22_ani_tra_
        r"(?P<size>\d+[kM])_"                 # Match size (e.g., 2M)
        r"(?P<epochs>\d+ep)_"                 # Match epochs (e.g., 5ep)
        r"(?P<extra>.*?)_"                    # Capture optional extras (e.g., tempSamp, balanced, etc.)
        r"(?P<run_id>[a-zA-Z0-9]+)$"           # Capture run ID at the end
    )

    match = re.match(pattern, checkpoint_path)
    if match:
        dataset = match.group("dataset").rstrip("_")  # Remove trailing underscore if present
        size = match.group("size")
        epochs = match.group("epochs")
        extra = match.group("extra") if match.group("extra") else None  # Optional
        run_id = match.group("run_id")
        return dataset, size, epochs, extra, run_id
    else:
        return None, None, None, None, None  # Return None for invalid format

def parse_namespace_string(namespace_str):
    # Remove the "Namespace" part from the string
    clean_str = re.sub(r'^Namespace\(', '', namespace_str).rstrip(')')
    
    # Replace '=' with ':', and make sure all keys and string values are properly quoted
    clean_str = re.sub(r'(\w+)=', r'"\1":', clean_str)  # Enclose keys in double quotes
    clean_str = clean_str.replace("'", '"')  # Replace single quotes with double quotes
    clean_str = clean_str.replace("True", "true").replace("False", "false").replace("None", "null")
    clean_str = re.sub(r'Path\((.*?)\)', r'\1', clean_str)

    return json.loads(f"{{{clean_str}}}") #ast.literal_eval(f"{{{clean_str}}}")

def fetch_and_update_run_details(project_path, df):
    """
    Fetch details from WandB runs and enrich the DataFrame with dataset details, fold, and seed.

    Args:
        project_path (str): The WandB project path.
        df (pd.DataFrame): DataFrame containing the filtered run IDs.

    Returns:
        pd.DataFrame: Updated DataFrame with additional extracted parameters.
    """
    enriched_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        run_id = row["Run ID"]
        
        # try:
        # Fetch run details
        job = api.run(f"{project_path}/{run_id}")
        config = job.config

        # Parse the args string
        args_str = config["args"]
        args_dict = parse_namespace_string(args_str)
    

        # Extract values
        dataset_name = args_dict.get("dataset_name", "Not Specified")
        target = args_dict.get("target", "Not Specified")
        checkpoint_path = args_dict.get("checkpoint_path", "Not Specified")
        fold = args_dict.get("fold", "Not Specified")
        seed = args_dict.get("seed", "Not Specified")

        # Parse checkpoint path
        if checkpoint_path == None:
            upstream_dataset = "Baseline"
            extra_mixed = None
            pretrain_size = None
            pretrain_epochs = None
            pretrain_run_id = None

        else:
            upstream_dataset, pretrain_size, pretrain_epochs, extra_mixed, pretrain_run_id = parse_checkpoint_path(checkpoint_path) 
            if upstream_dataset == None:
                upstream_dataset, pretrain_size, pretrain_epochs, extra_mixed, pretrain_run_id = parse_mixed_checkpoint_path(checkpoint_path)

        metric_key = metrics.get(dataset_name, None)
        metric_value = job.summary.get(metric_key, "Metric Not Found") if metric_key else "Metric Not Found"


        # Store updated row
        enriched_data.append({
            "Run ID": run_id,
            "Job Name": row["Job Name"],
            "Dataset Name": dataset_name,
            "Target": target,
            "Upstream Dataset": upstream_dataset,
            "Pretraining Size": pretrain_size,
            "Pretraining Epochs": pretrain_epochs,
            "Pretraining Mixed": extra_mixed,
            "Pretraining Run ID": pretrain_run_id,
            "Fold": fold,
            "Seed": seed,
            "Metric Key": metric_key,
            "Metric Value": metric_value,
        })

        # except Exception as e:
        #     enriched_data.append({
        #         "Run ID": run_id,
        #         "Job Name": row["Job Name"],
        #         "Error": str(e)
        #     })

    return pd.DataFrame(enriched_data)

# Define project path
project_path = "yasirghunaim/msi_finetune"

# Fetch enriched run details
updated_df = fetch_and_update_run_details(project_path, df)

# Save the updated results
updated_csv_filename = "wandb_matbench_test_results.csv"
updated_df.to_csv(updated_csv_filename, index=False)

print(f"Updated CSV file saved: {updated_csv_filename}")
