import csv
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import Pool, get_context
from tqdm import tqdm
import numpy as np
from scipy import linalg
import torch
from joblib import Parallel, delayed


def get_features(path, agg_operation="flat", device="cuda:0", subset_size=10000):
    data = torch.load(path)
    if isinstance(data, list):
        rows = []
        for i, item in enumerate(data):
            if i > subset_size:
                break
            x = torch.as_tensor(item["node_features"], dtype=torch.float32)
            if agg_operation == "flat":
                rows.append(x)  # size-biased
            elif agg_operation == "mean":
                rows.append(x.mean(0, keepdim=True))
            elif agg_operation == "sum":
                rows.append(x.sum(0, keepdim=True))
            elif agg_operation == "max":
                rows.append(x.max(0).values.unsqueeze(0))
            else:
                raise ValueError(f"Unknown agg_operation {agg_operation}")
        print("Rows:", len(rows))
        X = torch.cat(rows, dim=0)
    elif torch.is_tensor(data):
        X = data.float()
    elif isinstance(data, np.ndarray):
        X = torch.from_numpy(data).float()
    elif isinstance(data, dict) and "features" in data:
        X = torch.as_tensor(data["features"], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported data type in {path}")

    print("Shape:", X.shape)
    return X.to(device, non_blocking=True)


@torch.no_grad()
def _median_bandwidth_sq(X, Y, max_points=5000):
    Z = torch.cat([X, Y], dim=0)
    if Z.size(0) > max_points:
        idx = torch.randperm(Z.size(0), device=Z.device)[:max_points]
        Z = Z[idx]
    D2 = torch.cdist(Z, Z, p=2).pow(2)
    med = torch.median(D2[D2 > 0])
    return med.clamp(min=1e-12)

@torch.no_grad()
def compute_mmd_unbiased(X, Y, kernel="rbf", bandwidths=None):
    n, m = X.size(0), Y.size(0)
    Dxx = torch.cdist(X, X, p=2).pow(2)
    Dyy = torch.cdist(Y, Y, p=2).pow(2)
    Dxy = torch.cdist(X, Y, p=2).pow(2)

    if bandwidths is None:
        # s2 = _median_bandwidth_sq(X, Y)
        # bandwidths = [s2/2, s2, 2*s2]  # small multiscale
        s2 = _median_bandwidth_sq(X, Y)
        bandwidths = [s2]  # single bandwidth

    def k_rbf(D2, s2): return torch.exp(-D2 / (2*s2))

    if kernel == "rbf":
        Kxx = sum(k_rbf(Dxx, s2) for s2 in bandwidths)
        Kyy = sum(k_rbf(Dyy, s2) for s2 in bandwidths)
        Kxy = sum(k_rbf(Dxy, s2) for s2 in bandwidths)
    elif kernel == "multiscale":
        Kxx = sum((a*a) * (a*a + Dxx).reciprocal() for a in bandwidths)
        Kyy = sum((a*a) * (a*a + Dyy).reciprocal() for a in bandwidths)
        Kxy = sum((a*a) * (a*a + Dxy).reciprocal() for a in bandwidths)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Unbiased U-statistic (remove diagonals)
    Kxx_u = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1))
    Kyy_u = (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1))
    Kxy_m = Kxy.mean()
    return (Kxx_u + Kyy_u - 2*Kxy_m).item()


def compute_mmd(x, y, kernel="multiscale", bandwidth_range=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        x (torch.Tensor): Features from distribution P, shape (n, d)
        y (torch.Tensor): Features from distribution Q, shape (m, d)
        kernel (str): Kernel type ("rbf" or "multiscale")
        bandwidth_range (list): Bandwidth values for the kernel.

    Returns:
        torch.Tensor: Scalar MMD value.
    """
    if bandwidth_range is None:
        bandwidth_range = [10, 15, 20, 50] if kernel == "rbf" else [0.2, 0.5, 0.9, 1.3]

    # Pairwise squared Euclidean distances
    xx = torch.cdist(x, x, p=2) ** 2  # Shape: (n, n)
    yy = torch.cdist(y, y, p=2) ** 2  # Shape: (m, m)
    xy = torch.cdist(x, y, p=2) ** 2  # Shape: (n, m)

    # Kernel values
    XX = 0
    YY = 0
    XY = 0

    if kernel == "rbf":
        for a in bandwidth_range:
            XX += torch.exp(-xx / (2 * a))
            YY += torch.exp(-yy / (2 * a))
            XY += torch.exp(-xy / (2 * a))
    elif kernel == "multiscale":
        for a in bandwidth_range:
            XX += (a**2) * (a**2 + xx)**-1
            YY += (a**2) * (a**2 + yy)**-1
            XY += (a**2) * (a**2 + xy)**-1
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    # Compute means
    XX_mean = XX.mean()
    YY_mean = YY.mean()
    XY_mean = XY.mean()

    # Return scalar MMD value
    return XX_mean + YY_mean - 2 * XY_mean


# def compute_mmd(x, y, kernel="rbf"):
#     """Emprical maximum mean discrepancy. The lower the result
#        the more evidence that distributions are the same.

#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx = torch.mm(x, x.t())  # Shape (n, n)
#     yy = torch.mm(y, y.t())  # Shape (m, m)
#     xy = torch.mm(x, y.t())  # Shape (n, m)

#     rx = xx.diag().view(-1, 1)  # Shape (n, 1)
#     ry = yy.diag().view(-1, 1)  # Shape (m, 1)

#     # Compute pairwise distances
#     dxx = rx + rx.t() - 2.0 * xx  # Shape (n, n)
#     dyy = ry + ry.t() - 2.0 * yy  # Shape (m, m)
#     dxy = rx + ry.t() - 2.0 * xy  # Shape (n, m)
    
#     XX, YY, XY = (torch.zeros_like(dxx), torch.zeros_like(dyy), torch.zeros_like(dxy))

    
#     if kernel == "multiscale":
#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a**2 * (a**2 + dxx + 1e-8)**-1
#             YY += a**2 * (a**2 + dyy + 1e-8)**-1
#             XY += a**2 * (a**2 + dxy + 1e-8)**-1
            
#     if kernel == "rbf":
#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5*dxx/a)
#             YY += torch.exp(-0.5*dyy/a)
#             XY += torch.exp(-0.5*dxy/a)

#     return torch.mean(XX + YY - 2.0 * XY)



def extract_metadata(file_path):
    """
    Extract the sampling type and seed number from the file path.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        tuple: Extracted sampling type and seed number.
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
    sampling_type = sampling_type.replace(".pt", "")
    return sampling_type or "unknown", seed_number or "unknown"

def subsample_rows(T, max_rows=10000):
    if T.size(0) <= max_rows: 
        return T
    idx = torch.randperm(T.size(0))[:max_rows]
    # idx = torch.arange(max_rows)
    return T[idx]

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2, help="Paths to upstream and downstream datasets")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of processes to use")
    parser.add_argument("--model_name", type=str, default="gemnet", help="Model name")
    parser.add_argument("--agg_operation", type=str, default="flat", help="Aggregation operation")
    parser.add_argument("--checkpoint_tag", type=str, default="OC20", help="checkpoint tag")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the output file name")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--subset_size", type=int, default=10000, help="subset size")


    args = parser.parse_args()

    # Load upstream dataset features
    upstream_features_path = f"dataset_features_{args.model_name}_{args.checkpoint_tag}/{args.path[0]}"
    upstream_features = get_features(upstream_features_path, agg_operation=args.agg_operation, device=args.device, subset_size=args.subset_size)

    print("upstream_features:", upstream_features.shape)


    # Load downstream dataset features
    downstream_features_path = f"dataset_features_{args.model_name}_{args.checkpoint_tag}/{args.path[1]}"
    downstream_features = get_features(downstream_features_path, agg_operation=args.agg_operation, device=args.device, subset_size=args.subset_size)
    print("downstream_features:", downstream_features.shape)

    # Extract metadata
    upstream = upstream_features_path.split("/")[-1].split("_")[0]
    downstream = downstream_features_path.split("/")[-1].split("_")[0]
    if downstream == "matbench":
        downstream = "matbench_" + downstream_features_path.split("/")[-1].split("_")[1]  
    sampling, seed = extract_metadata(upstream_features_path)
    
    print(f"Upstream: {upstream}, Downstream: {downstream}, Sampling: {sampling}, Seed: {seed}")

    # X = subsample_rows(upstream_features, args.subset_size)
    # Y = subsample_rows(downstream_features, args.subset_size)
    # mmd_score = compute_mmd_unbiased(X, Y, kernel="rbf")
    mmd_score = compute_mmd_unbiased(upstream_features, downstream_features, kernel="rbf")
    print(f"MMD Score: {mmd_score:.6f}")
    print()
    print("=============================================================")

    # Save the CSI score
    suffix_part = f"_{args.suffix}" if args.suffix else ""
    save_filename = f"mmd_scores_{args.agg_operation}_{args.model_name}_{args.checkpoint_tag}{suffix_part}.csv"
    file_exists = os.path.isfile(save_filename)
    with open(save_filename, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Upstream", "Downstream", "Sampling", "Seed", "MMD Score"])
        writer.writerow([upstream, downstream, sampling, seed, mmd_score])


if __name__ == "__main__":
    main()