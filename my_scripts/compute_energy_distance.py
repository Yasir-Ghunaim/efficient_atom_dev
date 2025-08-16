import csv
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tqdm import tqdm
import numpy as np
import torch


def get_features(path, agg_operation="flat", device="cuda:0"):
    data = torch.load(path)
    if isinstance(data, list):
        rows = []
        for item in data:
            x = torch.as_tensor(item["node_features"], dtype=torch.float32)
            if agg_operation == "flat":
                rows.append(x)  # size-biased: keep all atoms
            elif agg_operation == "mean":
                rows.append(x.mean(0, keepdim=True))
            elif agg_operation == "sum":
                rows.append(x.sum(0, keepdim=True))
            elif agg_operation == "max":
                rows.append(x.max(0).values.unsqueeze(0))
            else:
                raise ValueError(f"Unknown agg_operation {agg_operation}")
        X = torch.cat(rows, dim=0)
    elif torch.is_tensor(data):
        X = data.float()
    elif isinstance(data, np.ndarray):
        X = torch.from_numpy(data).float()
    elif isinstance(data, dict) and "features" in data:
        X = torch.as_tensor(data["features"], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported data type in {path}")
    return X.to(device, non_blocking=True)


def extract_metadata(file_path):
    """
    Extract the sampling type and seed number from the file path.
    """
    base_name = os.path.basename(file_path)
    sampling_keyword = "Sampling"
    seed_keyword = "Seed"
    sampling_type = None
    seed_number = None

    if sampling_keyword in base_name and seed_keyword in base_name:
        parts = base_name.split("_")
        for part in parts:
            if part.startswith(sampling_keyword):
                sampling_type = part[len(sampling_keyword):]
            elif part.startswith(seed_keyword):
                seed_number = part[len(seed_keyword):]
    sampling_type = (sampling_type or "").replace(".pt", "")
    return sampling_type or "unknown", seed_number or "unknown"


def subsample_rows(T, max_rows=10000):
    if T.size(0) <= max_rows:
        return T
    idx = torch.randperm(T.size(0), device=T.device)[:max_rows]
    return T[idx]


@torch.no_grad()
def compute_energy_distance(X, Y, chunk=65536):
    """
    Energy distance with Euclidean norm using block summation.

    E = 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||

    Args:
        X, Y: tensors of shape (n, d) and (m, d)
        chunk: block size for pairwise computations

    Returns:
        float: scalar energy distance
    """
    n, m = X.size(0), Y.size(0)

    if n == 0 or m == 0:
        return float("nan")

    def pairwise_sum(A, B):
        total = 0.0
        aN = A.size(0)
        bN = B.size(0)
        for i in range(0, aN, chunk):
            Ai = A[i:i + chunk]
            for j in range(0, bN, chunk):
                Bj = B[j:j + chunk]
                Dij = torch.cdist(Ai, Bj, p=2)  # Euclidean
                total += Dij.sum().item()
        return total

    # Cross term E||X - Y||
    sum_xy = pairwise_sum(X, Y)
    mean_xy = sum_xy / (n * m)

    # Within terms E||X - X'|| and E||Y - Y'||
    # Using U-stat denominators (exclude diagonal in expectation).
    mean_xx = 0.0
    if n > 1:
        sum_xx = pairwise_sum(X, X)  # diagonal zeros don't change the sum
        mean_xx = sum_xx / (n * (n - 1))
    mean_yy = 0.0
    if m > 1:
        sum_yy = pairwise_sum(Y, Y)
        mean_yy = sum_yy / (m * (m - 1))

    return float(2 * mean_xy - mean_xx - mean_yy)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2, help="Paths to upstream and downstream dataset feature files")
    parser.add_argument("--num-workers", type=int, default=3, help="(kept for compatibility; unused)")
    parser.add_argument("--model_name", type=str, default="gemnet", help="Model name (for folder naming)")
    parser.add_argument("--agg_operation", type=str, default="flat", help="Aggregation operation: flat|mean|sum|max")
    parser.add_argument("--checkpoint_tag", type=str, default="OC20", help="Checkpoint tag (for folder naming)")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the output file name")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda:0|cuda:1 ...")
    parser.add_argument("--subset_size", type=int, help="Optional subsample size per set (<=0 to disable)")
    parser.add_argument("--chunk_size", type=int, default=65536, help="Block size for pairwise computations")
    parser.add_argument("--half", action="store_true", help="Use float16 for features to save memory")
    args = parser.parse_args()

    # Load upstream dataset features
    upstream_features_path = f"dataset_features_{args.model_name}_{args.checkpoint_tag}/{args.path[0]}"
    upstream_features = get_features(upstream_features_path, agg_operation=args.agg_operation, device=args.device)

    # Load downstream dataset features
    downstream_features_path = f"dataset_features_{args.model_name}_{args.checkpoint_tag}/{args.path[1]}"
    downstream_features = get_features(downstream_features_path, agg_operation=args.agg_operation, device=args.device)

    # Optional dtype cast for memory/runtime
    if args.half:
        if "cuda" in args.device:
            upstream_features = upstream_features.half()
            downstream_features = downstream_features.half()
        else:
            # float16 on CPU can be slow; bfloat16 is often better if available
            upstream_features = upstream_features.bfloat16() if torch.randn(1).dtype == torch.bfloat16 else upstream_features.half()
            downstream_features = downstream_features.bfloat16() if torch.randn(1).dtype == torch.bfloat16 else downstream_features.half()

    print("upstream_features:", tuple(upstream_features.shape))
    print("downstream_features:", tuple(downstream_features.shape))

    # Extract metadata for logging
    upstream = upstream_features_path.split("/")[-1].split("_")[0]
    downstream = downstream_features_path.split("/")[-1].split("_")[0]
    if downstream == "matbench":
        downstream = "matbench_" + downstream_features_path.split("/")[-1].split("_")[1]
    sampling, seed = extract_metadata(upstream_features_path)
    print(f"Upstream: {upstream}, Downstream: {downstream}, Sampling: {sampling}, Seed: {seed}")

    # Optional subsampling for speed; set --subset_size <= 0 to disable
    X = upstream_features
    Y = downstream_features
    # if args.subset_size > 0:
    #     X = subsample_rows(X, args.subset_size)
    #     Y = subsample_rows(Y, args.subset_size)

    # Compute Energy Distance
    ed_score = compute_energy_distance(X, Y, chunk=args.chunk_size)
    print(f"Energy Distance: {ed_score:.6f}")
    print()
    print("=============================================================")

    # Save
    suffix_part = f"_{args.suffix}" if args.suffix else ""
    save_filename = f"energy_distance_scores_{args.agg_operation}_{args.model_name}_{args.checkpoint_tag}{suffix_part}.csv"
    file_exists = os.path.isfile(save_filename)
    with open(save_filename, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Upstream", "Downstream", "Sampling", "Seed", "Energy Distance"])
        writer.writerow([upstream, downstream, sampling, seed, ed_score])


if __name__ == "__main__":
    main()
