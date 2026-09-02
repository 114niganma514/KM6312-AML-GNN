"""
GNN Hyperparameter Tuning and Ablation Study.

This script trains multiple GNN configurations on the Elliptic dataset
to find the best performing model and analyze the impact of different
architectural choices.

Configurations tested:
  1. GraphSAGE (2-layer, hidden=64)          - baseline
  2. GraphSAGE (2-layer, hidden=128)         - wider hidden layer
  3. GraphSAGE (3-layer, hidden=128)         - deeper network
  4. GAT (2-layer, hidden=64, heads=4)       - attention mechanism
  5. GraphSAGE (2-layer, hidden=64) + LR sched - learning rate scheduling
  6. GraphSAGE (2-layer, hidden=64, temporal) - temporal train/val/test split
"""

import os
import sys
import json
import time
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from data_loader import load_all
from preprocessing import preprocess_pipeline
from gnn_model import run_gnn_pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR
TUNING_DIR = BASE_DIR


def run_tuning():
    """Run all GNN tuning configurations and compare results."""
    os.makedirs(TUNING_DIR, exist_ok=True)

    print("=" * 70)
    print("GNN HYPERPARAMETER TUNING AND ABLATION STUDY")
    print("=" * 70)

    # Load data once
    print("\nLoading data...")
    _, edgelist, _, merged = load_all(DATA_RAW_DIR)

    # Preprocessing for random split (configs 1-5)
    print("\nPreprocessing (random split)...")
    preprocessed_random = preprocess_pipeline(
        merged, edgelist, temporal=False, save_dir=None
    )

    # Preprocessing for temporal split (config 6)
    print("\nPreprocessing (temporal split)...")
    preprocessed_temporal = preprocess_pipeline(
        merged, edgelist, temporal=True, save_dir=None
    )

    # Define configurations
    configs = [
        {
            "name": "GraphSAGE_base",
            "model_name": "GraphSAGE",
            "hidden_channels": 64,
            "dropout": 0.3,
            "epochs": 200,
            "lr": 0.01,
            "use_lr_scheduler": False,
            "preprocessed": preprocessed_random,
        },
        {
            "name": "GraphSAGE_hidden128",
            "model_name": "GraphSAGE",
            "hidden_channels": 128,
            "dropout": 0.3,
            "epochs": 200,
            "lr": 0.01,
            "use_lr_scheduler": False,
            "preprocessed": preprocessed_random,
        },
        {
            "name": "GraphSAGE_3layer_h128",
            "model_name": "GraphSAGE3",
            "hidden_channels": 128,
            "dropout": 0.3,
            "epochs": 200,
            "lr": 0.01,
            "use_lr_scheduler": False,
            "preprocessed": preprocessed_random,
        },
        {
            "name": "GAT_heads4",
            "model_name": "GAT",
            "hidden_channels": 64,
            "dropout": 0.3,
            "epochs": 200,
            "lr": 0.005,
            "heads": 4,
            "use_lr_scheduler": False,
            "preprocessed": preprocessed_random,
        },
        {
            "name": "GraphSAGE_lrsched",
            "model_name": "GraphSAGE",
            "hidden_channels": 64,
            "dropout": 0.3,
            "epochs": 300,
            "lr": 0.01,
            "use_lr_scheduler": True,
            "preprocessed": preprocessed_random,
        },
        {
            "name": "GraphSAGE_temporal",
            "model_name": "GraphSAGE",
            "hidden_channels": 64,
            "dropout": 0.3,
            "epochs": 200,
            "lr": 0.01,
            "use_lr_scheduler": False,
            "preprocessed": preprocessed_temporal,
        },
    ]

    all_results = {}
    total_start = time.time()

    for i, config in enumerate(configs, 1):
        name = config["name"]
        print(f"\n{'='*70}")
        print(f"CONFIG {i}/{len(configs)}: {name}")
        print(f"{'='*70}")

        config_start = time.time()

        # Extract model params
        model_kwargs = {
            "model_name": config["model_name"],
            "hidden_channels": config["hidden_channels"],
            "dropout": config["dropout"],
            "epochs": config["epochs"],
            "lr": config["lr"],
            "use_lr_scheduler": config.get("use_lr_scheduler", False),
            "save_dir": TUNING_DIR,
        }
        if "heads" in config:
            model_kwargs["heads"] = config["heads"]

        try:
            result = run_gnn_pipeline(config["preprocessed"], **model_kwargs)
            elapsed = time.time() - config_start

            mets = result["test_metrics"]
            all_results[name] = {
                "config": {k: v for k, v in config.items() if k != "preprocessed"},
                "test_metrics": mets,
                "best_epoch": result.get("history", {}).get("best_epoch", "N/A"),
                "training_time_sec": round(elapsed, 1),
            }

            print(f"\n  >>> {name} RESULT: PR-AUC={mets['pr_auc']:.4f}, "
                  f"F1={mets['f1_score']:.4f}, Recall={mets['recall']:.4f}, "
                  f"Precision={mets['precision']:.4f} "
                  f"(time: {elapsed:.0f}s)")

        except Exception as e:
            print(f"\n  *** ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Save all results
    results_path = os.path.join(TUNING_DIR, "tuning_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TUNING COMPLETE (total time: {total_elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n{'Config':<30} {'PR-AUC':>8} {'F1':>8} {'Recall':>8} {'Prec':>8} {'Time':>8}")
    print("-" * 80)

    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: x[1]["test_metrics"]["pr_auc"],
        reverse=True
    )

    for name, res in sorted_results:
        mets = res["test_metrics"]
        t = res.get("training_time_sec", 0)
        print(f"{name:<30} {mets['pr_auc']:>8.4f} {mets['f1_score']:>8.4f} "
              f"{mets['recall']:>8.4f} {mets['precision']:>8.4f} {t:>7.0f}s")

    if sorted_results:
        best_name, best_res = sorted_results[0]
        print(f"\n  BEST MODEL: {best_name} (PR-AUC={best_res['test_metrics']['pr_auc']:.4f})")

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == "__main__":
    run_tuning()
