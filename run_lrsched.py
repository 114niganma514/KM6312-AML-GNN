"""Quick re-run of GraphSAGE with LR scheduler for ablation study."""
import sys
import os
import json
import time



from data_loader import load_all
from preprocessing import preprocess_pipeline
from gnn_model import run_gnn_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = BASE_DIR
TUNING_DIR = BASE_DIR


def main():
    print("Loading data...")
    _, edgelist, _, merged = load_all(BASE_DIR)
    print("Preprocessing...")
    preprocessed = preprocess_pipeline(merged, edgelist, temporal=False, save_dir=None)

    print("\nTraining GraphSAGE with ReduceLROnPlateau scheduler...")
    start = time.time()
    result = run_gnn_pipeline(
        preprocessed,
        model_name="GraphSAGE",
        hidden_channels=64,
        dropout=0.3,
        epochs=300,
        lr=0.01,
        use_lr_scheduler=True,
        save_dir=TUNING_DIR,
    )
    elapsed = time.time() - start
    mets = result["test_metrics"]
    print(f"\nGraphSAGE_lrsched: PR-AUC={mets['pr_auc']:.4f}, "
          f"F1={mets['f1_score']:.4f}, Recall={mets['recall']:.4f}, "
          f"Precision={mets['precision']:.4f}, time={elapsed:.0f}s")

    # Append to tuning results
    tuning_path = os.path.join(TUNING_DIR, "tuning_results.json")
    if os.path.exists(tuning_path):
        with open(tuning_path) as f:
            tuning = json.load(f)
    else:
        tuning = {}

    tuning["GraphSAGE_lrsched"] = {
        "config": {
            "model_name": "GraphSAGE",
            "hidden_channels": 64,
            "dropout": 0.3,
            "epochs": 300,
            "lr": 0.01,
            "use_lr_scheduler": True,
        },
        "test_metrics": mets,
        "training_time_sec": round(elapsed, 1),
    }
    with open(tuning_path, "w") as f:
        json.dump(tuning, f, indent=2, default=str)
    print(f"Updated {tuning_path}")


if __name__ == "__main__":
    main()
