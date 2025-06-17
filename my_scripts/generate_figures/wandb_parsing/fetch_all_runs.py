import wandb
import csv

# Initialize the WandB API
api = wandb.Api()

def fetch_all_run_ids_and_names(project_path):
    """
    Fetch all run IDs and their corresponding job names from the specified WandB project.

    Args:
        project_path (str): The WandB project path in the format "entity/project".

    Returns:
        list: A list of dictionaries containing run IDs and job names.
    """
    run_details = []

    try:
        # Fetch all runs in the project
        runs = api.runs(project_path)

        for run in runs:
            run_details.append({
                "Run ID": run.id,
                "Job Name": run.name
            })

    except Exception as e:
        print(f"An error occurred: {e}")

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

# Define the project path
project_path = "yasirghunaim/msi_finetune"

# Fetch run details
run_details = fetch_all_run_ids_and_names(project_path)

# Write to CSV file
csv_filename = "wandb_run_details.csv"
write_to_csv(csv_filename, run_details)

# Provide the file path for download
csv_filename
