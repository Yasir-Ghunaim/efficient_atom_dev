import csv
import wandb
import re
import ast
from tqdm import tqdm
import statistics

# ---------------------------------------
# SETTINGS
# ---------------------------------------
csv_file = "wandb_run_details.csv"  # replace with your file name
project_path = "yasirghunaim/msi_finetune"  # W&B project path
keywords = ["Reb", "omat", "br70oai3", "2M"]  # edit: all must appear in the Job Name


# ani1x: o3n5226d
# transition1x: fufhevh6
# oc20: ccts7u4o
# oc22: vl2pj9xh
# balanced: sms8q33m
# temp: br70oai3

# Define metrics per dataset
metrics = {
    'rmd17': "test/md17/aspirin/force_mae",
    'md22': "test/md22/Ac-Ala3-NHMe/force_mae",
    'spice': "test/spice/solvated_amino_acids/force_mae",
    'qm9': "test/qm9/U_0_mae",
    "qmof": "test/qmof/y_mae",
    "omat": "test/omat/force_mae",
}

# ---------------------------------------
# HELPERS
# ---------------------------------------
def parse_namespace_string(namespace_str):
    namespace_str = namespace_str.replace("Final_Reb", "FINAL").replace("Final_reb", "FINAL")

    clean_str = re.sub(r'^Namespace\(', '', namespace_str).rstrip(')')
    clean_str = re.sub(r'(\w+)=', r'"\1":', clean_str)
    clean_str = clean_str.replace("'", '"')
    # Replace Path("...") with just "..."
    clean_str = re.sub(r'Path\("([^"]+)"\)', r'"\1"', clean_str)
    return ast.literal_eval(f"{{{clean_str}}}")

def parse_checkpoint_path(checkpoint_path):
    pattern = r"(?P<dataset>[a-zA-Z0-9]+)_(?P<size>\d+[kM])_(?P<epochs>\d+ep)_(?P<run_id>[a-zA-Z0-9]+)"
    match = re.match(pattern, checkpoint_path)
    if match:
        return match.group("dataset"), match.group("size"), match.group("epochs"), match.group("run_id")
    return None, None, None, None

# ---------------------------------------
# MAIN FUNCTION
# ---------------------------------------
def fetch_filtered_runs(csv_file, keywords, project_path):
    api = wandb.Api()
    run_ids = []
    matched_jobs = []

    # run_ids = ["rsnrob0y", "sapq3x5h", "8bq43l1t"]

    # if len(run_ids) > 0:
    #     print(f"Skipping keyword search, filtering by run IDs only: {run_ids}")

    # Step 1: filter jobs by keywords
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_name = row["Job Name"]

            # if len(run_ids) > 0:
            #     if row["Run ID"] in run_ids:
            #         matched_jobs.append(job_name)

            if all(kw.lower() in job_name.lower() for kw in keywords):
                run_ids.append(row["Run ID"])
                matched_jobs.append(job_name)

    if not run_ids:
        print("No matching runs found.")
        return

    print(f"Found {len(run_ids)} matching runs:")
    for rid, jn in zip(run_ids, matched_jobs):
        print(f"{rid}: {jn}")

    # Step 2: fetch details from W&B
    metric_values = []
    for run_id in tqdm(run_ids, desc="Fetching from W&B"):
        try:
            run = api.run(f"{project_path}/{run_id}")
            config = run.config
            args_str = config.get("args", "")
            args_dict = parse_namespace_string(args_str)

            dataset_name = args_dict.get("dataset_name", None)
            metric_key = metrics.get(dataset_name, None)

            if metric_key:
                metric_value = run.summary.get(metric_key, None)
                if metric_value is not None:
                    if dataset_name == "rmd17" or dataset_name == "qm9" or dataset_name == "md22"  or dataset_name == "spice"  or dataset_name == "omat":
                        metric_value *= 1000
                        # metric_value = round(metric_value, 2)
                    metric_values.append(metric_value)
                    print(f"{run_id}: {dataset_name} -> {metric_key} = {metric_value}")
        except Exception as e:
            print(f"Error fetching {run_id}: {e}")

    # Step 3: compute mean and std
    if metric_values:
        mean_val = statistics.mean(metric_values)
        std_val = statistics.pstdev(metric_values)  # population std
        print(f"\nResults for keywords {keywords}:")
        print(f"Mean = {mean_val:.1f}, Std = {std_val:.1f}")
    else:
        print("No metric values found.")

# ---------------------------------------
# RUN
# ---------------------------------------
if __name__ == "__main__":
    fetch_filtered_runs(csv_file, keywords, project_path)
