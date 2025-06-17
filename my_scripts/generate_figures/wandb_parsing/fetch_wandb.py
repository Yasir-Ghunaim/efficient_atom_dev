import wandb
import re
import ast
import csv
from tqdm import tqdm


# Define the force MAE metrics to extract for each dataset
metrics = {
    'rmd17': "test/md17/aspirin/force_mae",
    'md22': "test/md22/Ac-Ala3-NHMe/force_mae",
    'spice': "test/spice/solvated_amino_acids/force_mae",
    'qm9': "test/qm9/U_0_mae",
    "qmof": "test/qmof/y_mae",
}

# Initialize the WandB API
api = wandb.Api()

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
        return dataset, size, epochs, run_id
    else:
        return None, None, None, None  # Return None for invalid format

def parse_namespace_string(namespace_str):
    # Remove the "Namespace" part from the string
    clean_str = re.sub(r'^Namespace\(', '', namespace_str).rstrip(')')
    
    # Replace '=' with ':', and make sure all keys and string values are properly quoted
    clean_str = re.sub(r'(\w+)=', r'"\1":', clean_str)  # Enclose keys in double quotes
    clean_str = clean_str.replace("'", '"')  # Replace single quotes with double quotes

    # Now use ast.literal_eval to safely evaluate the string as a dictionary
    return ast.literal_eval(f"{{{clean_str}}}")

def fetch_run_details(project_path, run_ids):
    """
    Fetch details for specified runs in a WandB project.

    Args:
        project_path (str): The WandB project path in the format "entity/project".
        run_ids (list): List of run IDs to fetch details for.

    Returns:
        list: A list of dictionaries containing run details (test value, dataset, target, etc.).
    """
    run_details = []

    for run_id in tqdm(run_ids):
        try:
            # Fetch the run
            job = api.run(f"{project_path}/{run_id}")

            config = job.config
            args_str = config["args"]
            args_dict = parse_namespace_string(args_str)

            dataset_name = args_dict.get("dataset_name", "Not Specified")
            target = args_dict.get("target", "Not Specified")
            checkpoint_path = args_dict.get("checkpoint_path", "Not Specified")
            upstream_dataset, pretrain_size, pretrain_epochs, pretrain_run_id = parse_checkpoint_path(checkpoint_path)
            
            metric_key = metrics.get(dataset_name, None)
            metric_value = job.summary.get(metric_key, "Metric Not Found") if metric_key else "Metric Not Found"

            # Add details to the result
            run_details.append({
                "Run ID": run_id,
                "Dataset Name": dataset_name,
                "Target": target,
                "Upstream Dataset": upstream_dataset,
                "Pretraining Size": pretrain_size,
                "Pretraining Epochs": pretrain_epochs,
                "Pretraining Run ID": pretrain_run_id,
                "Metric Key": metric_key,
                "Metric Value": metric_value,
            })

        except Exception as e:
            run_details.append({
                "Run ID": run_id,
                "Error": str(e)
            })

    return run_details

def write_to_csv(file_name, data):
    """
    Write the run details to a CSV file.

    Args:
        file_name (str): The name of the output CSV file.
        data (list): A list of dictionaries containing run details.
    """
    if data:
        # Get the header from the keys of the first dictionary
        header = data[0].keys()

        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)

# Test the function
if __name__ == "__main__":


    # Main Experiment
    project_path = "/yasirghunaim/msi_finetune"
    run_ids = [
        "412j54qx", "1r6hq0lm", "q9y844wv", "syl9n7s2", "cmzahyvs",
        "ppnj4837", "nmslw7yc", "9d1x6rpp", "rw2vnv0t", "00jj4qky",
        "hrk4a551", "q7bgnssz", "fwjbaui8", "bnwak8qz", "wtg25jh1",
        "tv908mkf", "y8q1eqys", "tgih4d0g", "2nrtghg2", "a9pro3am"
    ]
    details = fetch_run_details(project_path, run_ids)
    write_to_csv("main_exp_2M.csv", details)