# preprocessing: scaling, splits, edge index, pyg data

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


def get_feature_columns(merged_df: pd.DataFrame) -> list:
    """return the list of feature column names (all columns except txId, time_step, class, label)"""
    exclude = {"txId", "time_step", "class", "label"}
    return [c for c in merged_df.columns if c not in exclude]


def normalize_features(merged_df: pd.DataFrame, feature_cols: list,
                       fit_mask: np.ndarray = None) -> tuple:
    """standardise features; fit_mask = rows used to fit the scaler (None = all nodes)"""
    values = merged_df[feature_cols].values
    scaler = StandardScaler()
    if fit_mask is None:
        scaler.fit(values)
    else:
        scaler.fit(values[fit_mask])
    X = scaler.transform(values)
    return X, scaler


def compute_class_weights(labels: np.ndarray) -> dict:
    """compute class weights for imbalanced classification"""
    n_illicit = (labels == 1).sum()
    n_licit = (labels == 0).sum()
    total = n_illicit + n_licit
    # weight inversely proportional to class frequency
    weight_illicit = total / (2 * n_illicit) if n_illicit > 0 else 1.0
    weight_licit = total / (2 * n_licit) if n_licit > 0 else 1.0
    return {0: weight_licit, 1: weight_illicit}


def split_labeled_data(merged_df: pd.DataFrame, test_size: float = 0.15,
                       val_size: float = 0.15, random_state: int = 42,
                       temporal: bool = False) -> tuple:
    """split labeled transactions into train/validation/test sets"""
    labeled_mask = merged_df["label"] != -1
    labeled_df = merged_df[labeled_mask].copy()
    labeled_indices = labeled_df.index.values

    if temporal:
        # temporal split: first 70% time steps = train, next 15% = val, last 15% = test
        time_steps = sorted(labeled_df["time_step"].unique())
        n_ts = len(time_steps)
        train_cutoff = int(n_ts * (1 - test_size - val_size))
        val_cutoff = int(n_ts * (1 - test_size))

        train_ts = set(time_steps[:train_cutoff])
        val_ts = set(time_steps[train_cutoff:val_cutoff])
        test_ts = set(time_steps[val_cutoff:])

        train_idx = labeled_df[labeled_df["time_step"].isin(train_ts)].index.values
        val_idx = labeled_df[labeled_df["time_step"].isin(val_ts)].index.values
        test_idx = labeled_df[labeled_df["time_step"].isin(test_ts)].index.values
    else:
        # random stratified split
        train_val_idx, test_idx = train_test_split(
            labeled_indices, test_size=test_size, random_state=random_state,
            stratify=labeled_df.loc[labeled_indices, "label"].values
        )
        relative_val_size = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val_size, random_state=random_state,
            stratify=labeled_df.loc[train_val_idx, "label"].values
        )

    return train_idx, val_idx, test_idx


def build_edge_index(edgelist_df: pd.DataFrame, txid_to_idx: dict) -> np.ndarray:
    """convert edge list (txId pairs) to edge index array for PyG"""
    sources = edgelist_df["txId1"].map(txid_to_idx).values
    targets = edgelist_df["txId2"].map(txid_to_idx).values

    # drop edges to unknown txIds
    valid_mask = ~(np.isnan(sources) | np.isnan(targets))
    sources = sources[valid_mask].astype(np.int64)
    targets = targets[valid_mask].astype(np.int64)

    edge_index = np.stack([sources, targets], axis=0)
    return edge_index


def build_pyg_data(features: np.ndarray, edge_index: np.ndarray,
                   labels: np.ndarray, train_idx: np.ndarray,
                   val_idx: np.ndarray, test_idx: np.ndarray) -> "PyGData":
    """build a PyTorch Geometric Data object for GNN training"""
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed.")

    x = torch.tensor(features, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    n_nodes = features.shape[0]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = PyGData(x=x, edge_index=edge_index_tensor, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data


def preprocess_pipeline(merged_df: pd.DataFrame, edgelist_df: pd.DataFrame,
                         temporal: bool = False, save_dir: str = None,
                         random_state: int = 42, verbose: bool = True) -> dict:
    """run the full preprocessing pipeline"""
    print("=" * 60)
    print("Running Preprocessing and Graph Construction")
    print("=" * 60)

    # get features and labels
    feature_cols = get_feature_columns(merged_df)
    print(f"  Number of features: {len(feature_cols)}")

    # split first so the temporal scaler only sees training time steps
    train_idx, val_idx, test_idx = split_labeled_data(merged_df, temporal=temporal,
                                                      random_state=random_state)
    if temporal:
        last_train_step = merged_df.loc[train_idx, "time_step"].max()
        fit_mask = (merged_df["time_step"] <= last_train_step).values
        print(f"  Scaler fitted on time steps 1-{last_train_step} only")
    else:
        fit_mask = None
        print("  Scaler fitted on all nodes (transductive; labels not used)")

    # normalize features
    X, scaler = normalize_features(merged_df, feature_cols, fit_mask=fit_mask)
    # zero-variance columns give nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    labels = merged_df["label"].values
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels shape: {labels.shape}")

    # build txId to index mapping
    txid_to_idx = {txid: idx for idx, txid in enumerate(merged_df["txId"].values)}

    # build edge index
    edge_index = build_edge_index(edgelist_df, txid_to_idx)
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Number of edges: {edge_index.shape[1]}")

    # split sizes
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")

    # compute class weights
    train_labels = labels[train_idx]
    class_weights = compute_class_weights(train_labels)
    print(f"  Class weights: {class_weights}")

    # build PyG data if available
    pyg_data = None
    if PYG_AVAILABLE:
        pyg_data = build_pyg_data(X, edge_index, labels, train_idx, val_idx, test_idx)
        print("  PyG Data object built successfully.")
    else:
        print("  [WARNING] PyTorch Geometric not available. Skipping PyG Data construction.")

    # save processed data if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "features.npy"), X)
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        np.save(os.path.join(save_dir, "edge_index.npy"), edge_index)
        np.save(os.path.join(save_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(save_dir, "val_idx.npy"), val_idx)
        np.save(os.path.join(save_dir, "test_idx.npy"), test_idx)
        print(f"  Processed data saved to {save_dir}")

    result = {
        "X": X,
        "labels": labels,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "class_weights": class_weights,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "txid_to_idx": txid_to_idx,
        "pyg_data": pyg_data,
    }

    print("\n[Preprocessing] Complete.")
    return result


if __name__ == "__main__":
    from data_loader import load_all

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = BASE_DIR
    PROCESSED_DIR = BASE_DIR

    _, edgelist, _, merged = load_all(DATA_DIR)
    result = preprocess_pipeline(merged, edgelist, temporal=False, save_dir=PROCESSED_DIR)
