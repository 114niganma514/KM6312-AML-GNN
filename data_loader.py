"""
Data Loader for the Elliptic Bitcoin Transaction Dataset.

This module loads the three CSV files of the Elliptic dataset:
  - elliptic_txs_features.csv  : 203,769 transactions x 166 features
  - elliptic_txs_edgelist.csv  : 234,355 directed payment-flow edges
  - elliptic_txs_classes.csv   : class labels (illicit / licit / unknown)

Reference:
  Weber et al. (2019). Anti-Money Laundering in Bitcoin:
  Experimenting with Graph Convolutional Networks for Financial Forensics.
  arXiv:1908.02591
"""

import os
import pandas as pd
import numpy as np


def load_features(data_dir: str) -> pd.DataFrame:
    """Load the transaction features CSV.

    The features file has no header. Column layout (detected dynamically):
      - Column 0: transaction ID (txId)
      - Column 1: time step (1-49)
      - Columns 2..N: node features (number varies by dataset version;
        the canonical Elliptic dataset has 166 features, some mirrors have 165).

    Args:
        data_dir: Path to the directory containing the raw CSV files.

    Returns:
        DataFrame with columns ['txId', 'time_step', 'feature_0', ..., 'feature_K-1'].
    """
    path = os.path.join(data_dir, "elliptic_txs_features.csv")
    # Read without names first to detect actual column count
    raw = pd.read_csv(path, header=None)
    n_cols = raw.shape[1]
    n_features = n_cols - 2  # exclude txId and time_step
    feature_names = ["txId", "time_step"] + [f"feature_{i}" for i in range(n_features)]
    raw.columns = feature_names
    print(f"  [data_loader] Detected {n_features} feature columns (total {n_cols} columns).")
    return raw


def load_edgelist(data_dir: str) -> pd.DataFrame:
    """Load the directed edge list CSV.

    Args:
        data_dir: Path to the directory containing the raw CSV files.

    Returns:
        DataFrame with columns ['txId1', 'txId2'] representing directed
        payment flows from txId1 to txId2.
    """
    path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    df = pd.read_csv(path)
    return df


def load_classes(data_dir: str) -> pd.DataFrame:
    """Load the class labels CSV.

    Args:
        data_dir: Path to the directory containing the raw CSV files.

    Returns:
        DataFrame with columns ['txId', 'class'] where class is one of
        'illicit', 'licit', or 'unknown'.
    """
    path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    df = pd.read_csv(path)
    return df


def load_all(data_dir: str) -> tuple:
    """Load all three dataset files and merge into a single structure.

    Args:
        data_dir: Path to the directory containing the raw CSV files.

    Returns:
        Tuple of (features_df, edgelist_df, classes_df, merged_df):
          - features_df: raw features DataFrame
          - edgelist_df: raw edge list DataFrame
          - classes_df: raw class labels DataFrame
          - merged_df: features merged with class labels, including a
            numeric 'label' column (1=illicit, 0=licit, -1=unknown)
    """
    features = load_features(data_dir)
    edgelist = load_edgelist(data_dir)
    classes = load_classes(data_dir)

    # Merge features with class labels
    merged = features.merge(classes, on="txId", how="left")

    # Create numeric label: 1 = illicit, 0 = licit, -1 = unknown
    # Support both string labels ("illicit"/"licit"/"unknown") and
    # numeric labels (1=illicit, 2=licit, "unknown")
    def _map_label(val):
        if pd.isna(val):
            return -1
        val_str = str(val).strip().lower()
        if val_str in ("illicit", "1"):
            return 1
        elif val_str in ("licit", "2"):
            return 0
        else:
            return -1

    merged["label"] = merged["class"].apply(_map_label).astype(int)

    return features, edgelist, classes, merged


def get_labeled_mask(merged_df: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask for labeled (illicit or licit) transactions.

    Args:
        merged_df: Merged DataFrame with 'label' column.

    Returns:
        Boolean numpy array, True for labeled transactions.
    """
    return (merged_df["label"] != -1).values


def get_dataset_stats(merged_df: pd.DataFrame, edgelist_df: pd.DataFrame) -> dict:
    """Compute and return basic dataset statistics.

    Args:
        merged_df: Merged DataFrame with features and labels.
        edgelist_df: Edge list DataFrame.

    Returns:
        Dictionary with dataset statistics.
    """
    total_nodes = len(merged_df)
    total_edges = len(edgelist_df)
    n_illicit = (merged_df["label"] == 1).sum()
    n_licit = (merged_df["label"] == 0).sum()
    n_unknown = (merged_df["label"] == -1).sum()
    n_time_steps = merged_df["time_step"].nunique()

    stats = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "n_illicit": int(n_illicit),
        "n_licit": int(n_licit),
        "n_unknown": int(n_unknown),
        "n_time_steps": int(n_time_steps),
        "illicit_ratio_labeled": round(n_illicit / (n_illicit + n_licit), 4) if (n_illicit + n_licit) > 0 else 0,
        "labeled_ratio": round((n_illicit + n_licit) / total_nodes, 4),
    }
    return stats


if __name__ == "__main__":
    # Quick test / inspection
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    features, edgelist, classes, merged = load_all(DATA_DIR)
    stats = get_dataset_stats(merged, edgelist)

    print("=" * 60)
    print("Elliptic Dataset - Basic Statistics")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")
    print(f"\n  Features shape : {features.shape}")
    print(f"  Edge list shape: {edgelist.shape}")
    print(f"  Classes shape  : {classes.shape}")
    print(f"  Merged shape   : {merged.shape}")
    print("\n  Class distribution:")
    print(merged["class"].value_counts())
