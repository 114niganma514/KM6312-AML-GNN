"""
Main Pipeline Runner for the KM6312 AML GNN Project.

This script orchestrates the entire project pipeline:
  1. Data Loading
  2. Exploratory Data Analysis (EDA)
  3. Preprocessing and Graph Construction
  4. Classical ML Baselines (LR, RF, XGBoost, LightGBM)
  5. Unsupervised Anomaly Detection and Clustering
  6. Graph Neural Networks (GCN, GraphSAGE)
  7. Evaluation and Results Comparison

Usage:
  python run_all.py                  # Run full pipeline
  python run_all.py --skip-eda      # Skip EDA
  python run_all.py --skip-gnn      # Skip GNN training
  python run_all.py --gnn-only      # Run only GNN (requires preprocessed data)
"""

import os
import sys
import json
import argparse
import time
import numpy as np

from data_loader import load_all, get_dataset_stats
from eda import run_eda
from preprocessing import preprocess_pipeline
from baselines import run_baselines
from clustering import run_clustering_pipeline
from gnn_model import run_gnn_pipeline
from evaluation import run_evaluation


# Project paths (flat structure: all files in the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
PROCESSED_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR
FIGURES_DIR = BASE_DIR
MODELS_DIR = BASE_DIR


def check_data_available():
    """Check if the raw dataset files exist."""
    required_files = [
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_classes.csv",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print("=" * 60)
        print("ERROR: Dataset files not found!")
        print("=" * 60)
        print(f"Missing files: {missing}")
        print(f"\nPlease download the Elliptic dataset to: {DATA_DIR}")
        print("\nDownload options:")
        print("  1. Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print("  2. Run: python download_data.py (requires Kaggle API credentials)")
        print("=" * 60)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="KM6312 AML GNN Project Pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline training")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering step")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN training")
    parser.add_argument("--gnn-only", action="store_true", help="Run only GNN (requires preprocessed data)")
    parser.add_argument("--temporal", action="store_true", help="Use temporal train/val/test split")
    parser.add_argument("--epochs", type=int, default=200, help="GNN training epochs")
    parser.add_argument("--hidden", type=int, default=64, help="GNN hidden channels")
    args = parser.parse_args()

    # Create directories
    for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    total_start = time.time()

    # ============================================================
    # Step 0: Check data
    # ============================================================
    if not check_data_available():
        sys.exit(1)

    # ============================================================
    # Step 1: Data Loading
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Data Loading")
    print("=" * 60)
    features, edgelist, classes, merged = load_all(DATA_DIR)
    stats = get_dataset_stats(merged, edgelist)
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    # ============================================================
    # Step 2: EDA
    # ============================================================
    if not args.skip_eda and not args.gnn_only:
        print("\n" + "=" * 60)
        print("STEP 2: Exploratory Data Analysis (EDA)")
        print("=" * 60)
        run_eda(DATA_DIR, FIGURES_DIR)
    else:
        print("\n[SKIP] EDA step skipped.")

    # ============================================================
    # Step 3: Preprocessing and Graph Construction
    # ============================================================
    if not args.gnn_only:
        print("\n" + "=" * 60)
        print("STEP 3: Preprocessing and Graph Construction")
        print("=" * 60)
        preprocessed = preprocess_pipeline(
            merged, edgelist, temporal=args.temporal,
            save_dir=PROCESSED_DIR
        )
    else:
        # Load preprocessed data
        print("\n" + "=" * 60)
        print("STEP 3: Loading Preprocessed Data")
        print("=" * 60)
        X = np.load(os.path.join(PROCESSED_DIR, "features.npy"))
        labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"))
        edge_index = np.load(os.path.join(PROCESSED_DIR, "edge_index.npy"))
        train_idx = np.load(os.path.join(PROCESSED_DIR, "train_idx.npy"))
        val_idx = np.load(os.path.join(PROCESSED_DIR, "val_idx.npy"))
        test_idx = np.load(os.path.join(PROCESSED_DIR, "test_idx.npy"))

        from preprocessing import build_pyg_data, compute_class_weights
        class_weights = compute_class_weights(labels[train_idx])
        pyg_data = build_pyg_data(X, edge_index, labels, train_idx, val_idx, test_idx)
        preprocessed = {
            "X": X, "labels": labels, "edge_index": edge_index,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "class_weights": class_weights, "pyg_data": pyg_data,
        }
        print("  Preprocessed data loaded successfully.")

    # ============================================================
    # Step 4: Classical ML Baselines
    # ============================================================
    if not args.skip_baselines and not args.gnn_only:
        print("\n" + "=" * 60)
        print("STEP 4: Classical ML Baselines")
        print("=" * 60)
        baseline_results = run_baselines(preprocessed, save_dir=RESULTS_DIR)
    else:
        print("\n[SKIP] Baselines step skipped.")

    # ============================================================
    # Step 5: Unsupervised Anomaly Detection and Clustering
    # ============================================================
    if not args.skip_clustering and not args.gnn_only:
        print("\n" + "=" * 60)
        print("STEP 5: Unsupervised Anomaly Detection and Clustering")
        print("=" * 60)
        clustering_results = run_clustering_pipeline(preprocessed, save_dir=RESULTS_DIR)
    else:
        print("\n[SKIP] Clustering step skipped.")

    # ============================================================
    # Step 6: Graph Neural Networks
    # ============================================================
    gnn_histories = {}
    if not args.skip_gnn:
        print("\n" + "=" * 60)
        print("STEP 6: Graph Neural Networks")
        print("=" * 60)
        for model_name in ["GCN", "GraphSAGE"]:
            print(f"\n--- Training {model_name} ---")
            gnn_result = run_gnn_pipeline(
                preprocessed, model_name=model_name,
                hidden_channels=args.hidden, epochs=args.epochs,
                save_dir=RESULTS_DIR
            )
            gnn_histories[model_name] = gnn_result["history"]
    else:
        print("\n[SKIP] GNN step skipped.")

    # ============================================================
    # Step 7: Evaluation and Results Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Evaluation and Results Comparison")
    print("=" * 60)
    eval_results = run_evaluation(RESULTS_DIR, FIGURES_DIR, gnn_histories=gnn_histories)

    # ============================================================
    # Complete
    # ============================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Figures directory: {FIGURES_DIR}")
    print(f"  Models directory: {MODELS_DIR}")
    print("\n  Generated files:")
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, RESULTS_DIR)
            print(f"    {rel:50s} ({size/1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
