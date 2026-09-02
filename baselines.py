"""
Classical Machine Learning Baselines for Illicit Transaction Detection.

This module trains and evaluates tabular classifiers using only the 166
node features (no graph structure). These serve as baselines to measure
the incremental value of graph-based methods (GNNs).

Models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM

Evaluation metrics (imbalanced classification):
  - PR-AUC (primary)
  - F1-Score
  - Recall (sensitivity to illicit transactions)
  - Precision
  - ROC-AUC
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


def get_baseline_models(class_weights: dict, random_state: int = 42) -> dict:
    """Return a dictionary of baseline classifier instances.

    Args:
        class_weights: Dictionary mapping class index to weight.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of model_name -> model instance.
    """
    # Convert class_weights to sample_weight-compatible format
    cw = {0: class_weights.get(0, 1.0), 1: class_weights.get(1, 1.0)}

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=random_state, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=cw, random_state=random_state,
            n_jobs=-1, max_depth=15, min_samples_split=5
        ),
    }

    if XGB_AVAILABLE:
        # XGBoost uses scale_pos_weight for imbalance
        scale_pos_weight = cw[1] / cw[0] if cw[0] > 0 else 1.0
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
            use_label_encoder=False, eval_metric="logloss", n_jobs=-1,
            verbosity=0
        )

    if LGB_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            class_weight=cw, random_state=random_state, n_jobs=-1,
            verbosity=-1
        )

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute evaluation metrics for imbalanced classification.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric name -> value.
    """
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    metrics = {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def train_and_evaluate_models(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               save_dir: str = None) -> dict:
    """Train all baseline models and evaluate on test set.

    Args:
        models: Dictionary of model_name -> model instance.
        X_train, y_train: Training data.
        X_val, y_val: Validation data (for early stopping if supported).
        X_test, y_test: Test data.
        save_dir: Optional directory to save results.

    Returns:
        Dictionary of model_name -> metrics dict, plus trained models.
    """
    print("=" * 60)
    print("Training and Evaluating Classical ML Baselines")
    print("=" * 60)

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train
        if name == "XGBoost" and X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "LightGBM" and X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        else:
            model.fit(X_train, y_train)

        # Predict on test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        trained_models[name] = model

        print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Confusion Matrix: {metrics['confusion_matrix']}")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save metrics as JSON (convert numpy types)
        serializable = {}
        for name, mets in results.items():
            serializable[name] = {k: v for k, v in mets.items()}
        with open(os.path.join(save_dir, "baseline_results.json"), "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to {save_dir}/baseline_results.json")

    print("\n[Baselines] Complete.")
    return {"results": results, "models": trained_models}


def run_baselines(preprocessed: dict, save_dir: str = None) -> dict:
    """Run the full baseline training pipeline using preprocessed data.

    Args:
        preprocessed: Dictionary from preprocessing.preprocess_pipeline().
        save_dir: Optional directory to save results.

    Returns:
        Dictionary with baseline results and trained models.
    """
    X = preprocessed["X"]
    labels = preprocessed["labels"]
    train_idx = preprocessed["train_idx"]
    val_idx = preprocessed["val_idx"]
    test_idx = preprocessed["test_idx"]
    class_weights = preprocessed["class_weights"]

    X_train, y_train = X[train_idx], labels[train_idx]
    X_val, y_val = X[val_idx], labels[val_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    print(f"  Train: {X_train.shape[0]} samples (illicit: {(y_train==1).sum()})")
    print(f"  Val:   {X_val.shape[0]} samples (illicit: {(y_val==1).sum()})")
    print(f"  Test:  {X_test.shape[0]} samples (illicit: {(y_test==1).sum()})")

    models = get_baseline_models(class_weights)
    print(f"  Models to train: {list(models.keys())}")

    return train_and_evaluate_models(
        models, X_train, y_train, X_val, y_val, X_test, y_test, save_dir
    )


if __name__ == "__main__":
    from data_loader import load_all
    from preprocessing import preprocess_pipeline

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = BASE_DIR
    RESULTS_DIR = BASE_DIR

    _, edgelist, _, merged = load_all(DATA_DIR)
    preprocessed = preprocess_pipeline(merged, edgelist, temporal=False)
    run_baselines(preprocessed, save_dir=RESULTS_DIR)
