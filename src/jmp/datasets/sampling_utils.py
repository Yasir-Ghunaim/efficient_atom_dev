import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import math

def get_molecule_df(dataset_path):
    path_parts = dataset_path.parts
    dataset_name = path_parts[path_parts.index("datasets") + 1]

    all_files = list(dataset_path.glob("data_log.*.txt"))
    all_files = sorted(all_files, key=lambda x: int(x.stem.split('.')[-1]))
    all_files = [str(file) for file in all_files]

    df_list = []
    # Read each file and append to the list
    for file in all_files:
        # Check the first line of the file to determine if it has a header
        with open(file, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line == "Molecule,fid,sid":
            header = 0  # First row is the header
        else:
            header = None  # No header, use default column names
        
        # Read the file with the determined header option
        if dataset_name == "transition1x":
            temp_df = pd.read_csv(file, header=header, names=['Molecule', 'fid'])
        else:
            temp_df = pd.read_csv(file, header=header, names=['Molecule', 'fid', 'sid'])

        df_list.append(temp_df)

    # Concatenate all dataframes into a single dataframe
    df = pd.concat(df_list, ignore_index=True)
    
    
    # Only apply extraction for datasets that include '_rxn' in 'Molecule'
    if dataset_name == "transition1x":
        df['sid'] = df['Molecule'].str.extract(r'_rxn(\d+)', expand=False)
        df['Molecule'] = df['Molecule'].str.replace(r'_rxn\d+', '', regex=True)
        df['sid'] = pd.to_numeric(df['sid'], errors='coerce')  # Convert 'sid' to integer where it exists

        # Reorder columns
        df = df[['Molecule', 'sid', 'fid']]
    return df

def apply_random_sampling(available_samples, num_samples, seed):
        """
        Perform random sampling.

        Args:
        available_samples (int): Number of available samples.
        num_samples (int): Total number of samples desired.
        seed (int): Random seed for reproducibility.
        """

        random_state = np.random.RandomState(seed)
        if num_samples > available_samples:
            num_samples = available_samples
        return random_state.choice(range(available_samples), num_samples, replace=False).tolist()

    
def apply_class_balanced_sampling(df, num_samples, seed, allow_repetition=True):
    """
    Performs class-balanced sampling on a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Molecule' column for class labels.
        num_samples (int): Total number of samples desired.
        seed (int): Random seed for reproducibility.
        allow_repetition (bool): Whether to allow repeated sampling when a class
                                 has fewer samples than the target per-class count.

    Returns:
        tuple: 
            - indices (list): Indices of the sampled rows.
            - sampled_values (pd.DataFrame): DataFrame of the sampled rows.
            - balanced_dist (pd.Series): Distribution of 'Molecule' column after sampling.
    """        

    def sample_class(group, n_samples, seed):
        """Sample indices for each class, allowing repetition if enabled."""
        count = len(group)
        if allow_repetition:
            return group.sample(n=n_samples, replace=(count < n_samples), random_state=seed).index
        else:
            return group.sample(n=min(n_samples, count), replace=False, random_state=seed).index

    def get_balanced_additional_samples(df, samples_needed, seed):
        """
        Get additional samples in a class-balanced way to fill the remaining gap.

        Args:
            df (pd.DataFrame): The original DataFrame containing a 'Molecule' column.
            samples_needed (int): The number of additional samples needed to fill the gap.
            sample_class (function): A function to sample indices from each group.

        Returns:
            list: Indices of the additional samples needed to fill the gap.
        """
        # Perform class-balanced sampling with one sample per class
        additional_indices = df.groupby('Molecule', group_keys=False).apply(
            lambda group: sample_class(group, 1, seed)
        )
        additional_indices = additional_indices.explode().astype(int).tolist()

        if len(additional_indices) >= samples_needed:
            rng = np.random.RandomState(seed)
            additional_indices = rng.choice(additional_indices, size=samples_needed, replace=False).tolist()

        return additional_indices

    class_counts = df['Molecule'].value_counts()
    num_classes = len(class_counts)
    target_per_class = num_samples // num_classes

    if target_per_class == 0: # less than 1 sample per class
        indices = get_balanced_additional_samples(df, num_samples, seed)
    else:
        if allow_repetition:
            # Step 1: Perform the initial class-balanced sampling
            sampled_indices_balanced_rep = df.groupby('Molecule', group_keys=False).apply(
                lambda group: sample_class(group, target_per_class, seed)
            )
            sampled_indices_balanced_rep = sampled_indices_balanced_rep.explode().astype(int).tolist()

            # Step 2: Calculate the gap if the total samples are fewer than requested
            remaining_needed = num_samples - len(sampled_indices_balanced_rep)
            while remaining_needed > 0:
                additional_per_class = math.ceil(remaining_needed / num_classes)
                assert additional_per_class == 1, "Additional samples per class should be 1."
                # Exclude already-sampled indices
                remaining_df = df[~df.index.isin(sampled_indices_balanced_rep)]
                sampled_indices_balanced_rep += get_balanced_additional_samples(remaining_df, remaining_needed, seed)
                remaining_needed = num_samples - len(sampled_indices_balanced_rep)
            indices = sampled_indices_balanced_rep
        else:
            remaining_df = df
            samples_length = 0
            sampled_indices_balanced_no_rep = []

            # Continue sampling until the required number of samples is reached
            while samples_length < num_samples:
                # Step 1: Perform class-balanced sampling
                sampled_indices = remaining_df.groupby('Molecule', group_keys=False).apply(
                    lambda group: sample_class(group, target_per_class, seed)
                )
                sampled_indices = sampled_indices.explode().astype(int).tolist()
                samples_length += len(sampled_indices)

                # Check if the number of sampled indices exceeds the required number
                if samples_length > num_samples:
                    excess_samples = samples_length - num_samples
                    required_additional_samples = len(sampled_indices) - excess_samples
                    rng = np.random.RandomState(seed)
                    sampled_indices = rng.choice(sampled_indices, size=required_additional_samples, replace=False).tolist()

                # Add the sampled indices to the final list
                sampled_indices_balanced_no_rep.extend(sampled_indices)

                # Update the remaining DataFrame by excluding already-sampled indices
                remaining_df = df[~df.index.isin(sampled_indices_balanced_no_rep)]
                # Update the target per class if we have fewer samples than expected
                target_per_class = 1

            indices = sampled_indices_balanced_no_rep    
        

    assert len(indices) == num_samples, "Number of samples does not match the expected count."

    # Retrieve sampled rows and calculate the distribution
    balanced_dist = df.loc[indices]['Molecule'].value_counts()
    print(balanced_dist)
    
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    return indices

def apply_stratified_sampling(molecule_df, max_samples, seed):
    # Ensure molecule data is loaded
    if molecule_df is None:
        raise ValueError("Molecule data is required for stratified sampling. Set 'extract_features' to True.")

    # Calculate counts of each molecule
    molecule_counts = molecule_df['Molecule'].value_counts()

    # Separate rare molecules (count <= 1) and abundant molecules
    rare_molecules = molecule_counts[molecule_counts <= 1].index
    abundant_df = molecule_df[~molecule_df['Molecule'].isin(rare_molecules)]
    rare_df = molecule_df[molecule_df['Molecule'].isin(rare_molecules)]

    # Calculate how many samples we need from abundant molecules
    remaining_samples = max_samples - len(rare_df)

    # Perform stratified sampling on abundant molecules
    stratified_sampled_df, _ = train_test_split(
        abundant_df,
        train_size=remaining_samples,
        stratify=abundant_df['Molecule'],
        random_state=seed
    )

    # Combine the stratified sample with the rare samples
    final_sampled_df = pd.concat([stratified_sampled_df, rare_df])

    # Retrieve the indices of the final sample
    indices = final_sampled_df.index.tolist()
    return indices

def apply_naive_uniform_sampling(molecule_df, max_samples, seed):
    """
    Perform uniform sampling by selecting one sample for each unique class (molecule).
    
    Returns:
        indices (list): List of indices corresponding to the selected samples.
    """
    # Ensure molecule data is loaded
    if molecule_df is None:
        raise ValueError("Molecule data is required for uniform sampling. Set 'extract_features' to True.")

    assert max_samples > len(molecule_df['Molecule'].value_counts()), "Number of samples to select has to be larger than the number of unique classes."
    
    # Group by the 'Molecule' column and randomly sample one entry per group
    uniform_sampled_df = molecule_df.groupby('Molecule').sample(n=1, random_state=seed)
    sampled_indices = uniform_sampled_df.index.tolist()

    # Add the remaining samples randomly
    remaining_df = molecule_df[~molecule_df.index.isin(sampled_indices)]
    remaining_samples = max_samples - len(sampled_indices)
    sampled_indices.extend(remaining_df.sample(n=remaining_samples, random_state=seed).index.tolist()) 

    balanced_dist = molecule_df.loc[sampled_indices]['Molecule'].value_counts()   
    print(balanced_dist)

    rng = np.random.RandomState(seed)
    rng.shuffle(sampled_indices)

    return sampled_indices
