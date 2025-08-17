import os
import math
import argparse
import csv
import numpy as np
import torch
from scipy.stats import entropy, norm
from tqdm import tqdm 

# --------- helpers ---------

def load_graph_list(path):
    data = torch.load(path)
    if not isinstance(data, list):
        raise ValueError("Expected a list of graphs with 'node_features'.")
    return data


def subsample_graphs(graphs, N: int, rng=None):
    """Return a random subset of N graphs (list of dicts)."""
    rng = np.random.default_rng() if rng is None else rng
    if N > len(graphs):
        raise ValueError(f"N={N} > total graphs={len(graphs)}")

    chosen = rng.choice(len(graphs), size=N, replace=False)
    return [graphs[i] for i in chosen]

def standardize_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise z-score standardization."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def graphs_to_flat_and_mean(graphs, standardize=True):
    """
    Convert a list of graph dicts to (X_flat, X_mean).
      - X_flat: all node features concatenated
      - X_mean: per-graph mean features
      - Optionally standardize each view separately
    """
    feats_flat, feats_mean = [], []
    for g in graphs:
        nf = torch.as_tensor(g["node_features"])  # [n_i, D]
        feats_flat.append(nf)
        feats_mean.append(nf.mean(dim=0, keepdim=True))

    X_flat = torch.cat(feats_flat, dim=0).cpu().numpy()
    X_mean = torch.cat(feats_mean, dim=0).cpu().numpy()

    if standardize:
        X_flat = standardize_rows(X_flat)
        X_mean = standardize_rows(X_mean)

    return X_flat, X_mean

def effective_rank(cov: np.ndarray, eps: float = 1e-12, base=np.e) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007), using scipy.stats.entropy."""
    cov = 0.5 * (cov + cov.T)  # symmetrize
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    if total <= eps:
        return 0.0
    p = evals / total
    H = entropy(p, base=base)  # nats
    return float(np.exp(H))


def bootstrap_effective_rank_graphs(
    X_flat: np.ndarray,
    flat_slices: list[slice],
    G: int,
    n_boot: int = 20,
    replace: bool = False,
    seed: int = 0
):
    """
    Bootstrap by sampling GRAPHS:
      - choose G graph indices per round
      - gather all node rows corresponding to those graphs from X_flat
      - compute covariance & effective rank
    """
    rng = np.random.default_rng(seed)
    n_graphs = len(flat_slices)
    if G > n_graphs and not replace:
        raise ValueError(f"G={G} > n_graphs={n_graphs}. Use --replace or reduce G.")

    ranks = []
    for _ in tqdm(range(n_boot), desc="Bootstrapping"):
        g_idx = rng.choice(n_graphs, size=G, replace=replace)
        # Collect rows from X_flat for the chosen graphs
        parts = [X_flat[s] for i, s in enumerate(flat_slices) if i in set(g_idx)]
        # When sampling with replacement, the same graph can appear multiple times;
        # include its slice multiple times to reflect that weighting.
        if replace:
            parts = []
            for i in g_idx:
                parts.append(X_flat[flat_slices[i]])
        X_sub = np.vstack(parts)
        cov = np.cov(X_sub, rowvar=False)
        ranks.append(effective_rank(cov))
    return np.array(ranks)

def mean_std_ci(x, alpha=0.05):
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if x.size > 1:
        z = norm.ppf(1 - alpha/2)
        half = z * s / math.sqrt(x.size)
        ci = (m - half, m + half)
    else:
        ci = (m, m)
    return m, s, ci

def bootstrap_effective_ranks(graphs, size: int, n_boot: int = 20, replace: bool = True,
                              seed: int = 0, standardize: bool = True):
    rng = np.random.default_rng(seed)
    flat_ranks, mean_ranks = [], []

    for _ in range(n_boot):
        # resample graph indices
        chosen = rng.choice(len(graphs), size=size, replace=replace)
        subset = [graphs[i] for i in chosen]

        # build features
        X_flat, X_mean = graphs_to_flat_and_mean(subset, standardize=standardize)

        # compute effective ranks
        flat_ranks.append(effective_rank(np.cov(X_flat, rowvar=False)))
        mean_ranks.append(effective_rank(np.cov(X_mean, rowvar=False)))

    return np.array(flat_ranks), np.array(mean_ranks)


# ---------------- CLI main ----------------

def main():
    parser = argparse.ArgumentParser(description="Compare effective rank: flat vs mean (with graph-bootstrap for flat).")
    parser.add_argument("features_path", type=str,
                        help="Path to .pt file (list of dicts with 'node_features').")
    parser.add_argument("--standardize", action="store_true",
                        help="Z-score features before covariance (recommended).")
    parser.add_argument("--n-boot", type=int, default=20,
                        help="Number of bootstrap repeats.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for subsampling.")
    parser.add_argument("--replace", action="store_true",
                        help="Sample graphs with replacement in bootstrap.")
    parser.add_argument("--csv", type=str, default="effective_ranks.csv",
                        help="CSV file to append results.")
    args = parser.parse_args()

    path = args.features_path
    dataset_name = os.path.basename(path).replace(".pt", "")
    graphs = load_graph_list(path)
    total_graphs = len(graphs)
    feat_dim = torch.as_tensor(graphs[0]["node_features"]).shape[-1]
    print(f"Dataset: {dataset_name} | total graphs: {total_graphs} | D={feat_dim}")

    # No bootstrap - sanity check
    seed = 0
    size = 10000
    rng = np.random.default_rng(seed)

    # Subsample graphs (e.g., 10) then compute flat & mean
    subset = subsample_graphs(graphs, N=size, rng=rng)
    print(f"Subset size: {len(subset)} graphs")
    X_flat, X_mean = graphs_to_flat_and_mean(
        subset, standardize=args.standardize,
    )

    # Full-sample effective ranks (for reference)
    r_flat_full = effective_rank(np.cov(X_flat, rowvar=False))
    r_mean_full = effective_rank(np.cov(X_mean, rowvar=False))

    print("\nEffective Rank results:")
    print(f"  FLAT (node-level): {r_flat_full:.4f}")
    print(f"  MEAN (graph-level): {r_mean_full:.4f}")

    # Bootstrap starts here

    seed = 0
    sizes = [100, 1000, 5000, 10000]

    file_exists = os.path.isfile(args.csv)
    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Dataset", "Flat (graph-bootstrap) mean", "Flat (graph-bootstrap) std",
                "Flat (graph-bootstrap) CI low", "Flat (graph-bootstrap) CI high",
                "Mean (graph-bootstrap) mean", "Mean (graph-bootstrap) std",
                "Mean (graph-bootstrap) CI low", "Mean (graph-bootstrap) CI high",
                "Subset_size", "Total_graphs", "Feature_dim", "n_boot", "Replace"
            ])

        for size in sizes:
            print(f"\n=== Bootstrap with subset size {size} ===")
            flat_ranks, mean_ranks = bootstrap_effective_ranks(
                graphs, size=size, n_boot=args.n_boot,
                replace=args.replace, seed=seed, standardize=args.standardize
            )

            # Aggregate results
            m_flat, s_flat, ci_flat = mean_std_ci(flat_ranks)
            m_mean, s_mean, ci_mean = mean_std_ci(mean_ranks)

            print(f"Bootstrap results (size={size}, n_boot={args.n_boot}, replace={args.replace}):")
            print(f"  FLAT (node-level): mean={m_flat:.4f}, std={s_flat:.4f}, "
                  f"95% CI=({ci_flat[0]:.4f}, {ci_flat[1]:.4f})")
            print(f"  MEAN (graph-level): mean={m_mean:.4f}, std={s_mean:.4f}, "
                  f"95% CI=({ci_mean[0]:.4f}, {ci_mean[1]:.4f})")

            # Write row to CSV
            writer.writerow([
                dataset_name,
                f"{m_flat:.6f}", f"{s_flat:.6f}", f"{ci_flat[0]:.6f}", f"{ci_flat[1]:.6f}",
                f"{m_mean:.6f}", f"{s_mean:.6f}", f"{ci_mean[0]:.6f}", f"{ci_mean[1]:.6f}",
                size, total_graphs, feat_dim, args.n_boot, int(args.replace)
            ])


if __name__ == "__main__":
    main()
