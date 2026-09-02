"""
Evaluation and Results Comparison Module.

This module generates comprehensive evaluation outputs:
  - Model comparison table (all baselines + GNNs)
  - Performance comparison bar charts (PR-AUC, F1, Recall)
  - Training curve plots for GNNs
  - Confusion matrix visualizations
  - Case study: illicit subgraph analysis
  - Final results summary (JSON + Markdown)
"""

import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150


def load_results(results_dir: str) -> dict:
    """Load all saved result JSON files.

    Args:
        results_dir: Directory containing result JSON files.

    Returns:
        Dictionary with all model results.
    """
    all_results = {}

    # Baseline results
    baseline_path = os.path.join(results_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            all_results["baselines"] = json.load(f)

    # Clustering results
    clustering_path = os.path.join(results_dir, "clustering_results.json")
    if os.path.exists(clustering_path):
        with open(clustering_path) as f:
            all_results["clustering"] = json.load(f)

    # GNN results
    for model_name in ["gcn", "graphsage"]:
        gnn_path = os.path.join(results_dir, f"gnn_{model_name}_results.json")
        if os.path.exists(gnn_path):
            with open(gnn_path) as f:
                all_results[f"gnn_{model_name}"] = json.load(f)

    return all_results


def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """Build a comparison table of all supervised models.

    Args:
        all_results: Dictionary from load_results().

    Returns:
        DataFrame with model names and metrics.
    """
    rows = []

    # Baselines
    if "baselines" in all_results:
        for model_name, metrics in all_results["baselines"].items():
            rows.append({
                "Model": model_name,
                "Category": "Classical ML",
                "PR-AUC": metrics.get("pr_auc", 0),
                "ROC-AUC": metrics.get("roc_auc", 0),
                "F1-Score": metrics.get("f1_score", 0),
                "Recall": metrics.get("recall", 0),
                "Precision": metrics.get("precision", 0),
            })

    # GNNs
    for key in ["gnn_gcn", "gnn_graphsage"]:
        if key in all_results:
            data = all_results[key]
            metrics = data.get("test_metrics", {})
            rows.append({
                "Model": data.get("model_name", key),
                "Category": "Graph Neural Network",
                "PR-AUC": metrics.get("pr_auc", 0),
                "ROC-AUC": metrics.get("roc_auc", 0),
                "F1-Score": metrics.get("f1_score", 0),
                "Recall": metrics.get("recall", 0),
                "Precision": metrics.get("precision", 0),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return df


def plot_model_comparison(df: pd.DataFrame, save_dir: str) -> None:
    """Plot bar charts comparing model performance across metrics.

    Args:
        df: Comparison DataFrame from build_comparison_table().
        save_dir: Directory to save figures.
    """
    if df.empty:
        print("  [Evaluation] No data for comparison plot.")
        return

    metrics = ["PR-AUC", "ROC-AUC", "F1-Score", "Recall", "Precision"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    colors = {"Classical ML": "#4CAF50", "Graph Neural Network": "#F44336"}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bar_colors = [colors.get(cat, "#999") for cat in df["Category"]]
        bars = ax.barh(df["Model"], df[metric], color=bar_colors)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel(metric)
        ax.invert_yaxis()
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors["Classical ML"], label="Classical ML"),
                       Patch(facecolor=colors["Graph Neural Network"], label="GNN")]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "07_model_comparison.png"), bbox_inches="tight")
    plt.close()
    print("  [Evaluation] Saved 07_model_comparison.png")


def plot_training_curves(history: dict, model_name: str, save_dir: str) -> None:
    """Plot GNN training curves (loss and accuracy).

    Args:
        history: Dictionary with train_loss, val_loss, val_acc lists.
        model_name: Name of the model for the title.
        save_dir: Directory to save figures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1976D2")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#4CAF50")
    axes[1].set_title(f"{model_name} - Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"08_training_curves_{model_name.lower()}.png"),
                bbox_inches="tight")
    plt.close()
    print(f"  [Evaluation] Saved 08_training_curves_{model_name.lower()}.png")


def plot_confusion_matrices(all_results: dict, save_dir: str) -> None:
    """Plot confusion matrices for all models.

    Args:
        all_results: Dictionary from load_results().
        save_dir: Directory to save figures.
    """
    models_data = []

    if "baselines" in all_results:
        for name, metrics in all_results["baselines"].items():
            if "confusion_matrix" in metrics:
                models_data.append((name, np.array(metrics["confusion_matrix"])))

    for key in ["gnn_gcn", "gnn_graphsage"]:
        if key in all_results:
            data = all_results[key]
            cm = data.get("test_metrics", {}).get("confusion_matrix")
            if cm:
                models_data.append((data.get("model_name", key), np.array(cm)))

    if not models_data:
        return

    n_models = len(models_data)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(models_data):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Licit", "Illicit"],
                    yticklabels=["Licit", "Illicit"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "09_confusion_matrices.png"), bbox_inches="tight")
    plt.close()
    print("  [Evaluation] Saved 09_confusion_matrices.png")


def generate_results_summary(all_results: dict, comparison_df: pd.DataFrame,
                             save_dir: str) -> str:
    """Generate a Markdown summary of all results.

    Args:
        all_results: Dictionary from load_results().
        comparison_df: Comparison DataFrame.
        save_dir: Directory to save the summary.

    Returns:
        Markdown string of the summary.
    """
    lines = []
    lines.append("# KM6312 AML GNN Project - Results Summary\n")
    lines.append("## Model Comparison (sorted by PR-AUC)\n")

    if not comparison_df.empty:
        lines.append("| Rank | Model | Category | PR-AUC | ROC-AUC | F1-Score | Recall | Precision |")
        lines.append("|------|-------|----------|--------|---------|----------|--------|-----------|")
        for idx, row in comparison_df.iterrows():
            lines.append(f"| {idx+1} | {row['Model']} | {row['Category']} | "
                         f"{row['PR-AUC']:.4f} | {row['ROC-AUC']:.4f} | {row['F1-Score']:.4f} | "
                         f"{row['Recall']:.4f} | {row['Precision']:.4f} |")

    # Best model
    if not comparison_df.empty:
        best = comparison_df.iloc[0]
        lines.append(f"\n## Best Model\n")
        lines.append(f"**{best['Model']}** ({best['Category']}) achieved the highest "
                     f"PR-AUC of **{best['PR-AUC']:.4f}**, with F1-Score {best['F1-Score']:.4f}, "
                     f"Recall {best['Recall']:.4f}, and Precision {best['Precision']:.4f}.\n")

    # GNN vs Classical
    gnn_df = comparison_df[comparison_df["Category"] == "Graph Neural Network"]
    classical_df = comparison_df[comparison_df["Category"] == "Classical ML"]
    if not gnn_df.empty and not classical_df.empty:
        gnn_best = gnn_df["PR-AUC"].max()
        classical_best = classical_df["PR-AUC"].max()
        diff = gnn_best - classical_best
        lines.append("## GNN vs Classical ML\n")
        lines.append(f"- Best GNN PR-AUC: **{gnn_best:.4f}**")
        lines.append(f"- Best Classical ML PR-AUC: **{classical_best:.4f}**")
        lines.append(f"- Difference: **{diff:+.4f}** "
                     f"({'GNN outperforms' if diff > 0 else 'Classical ML outperforms'})\n")

    # Clustering results
    if "clustering" in all_results:
        cl = all_results["clustering"]
        lines.append("## Unsupervised Learning Results\n")
        if "isolation_forest" in cl:
            iso = cl["isolation_forest"]
            lines.append(f"### Isolation Forest")
            lines.append(f"- Anomalies detected: {iso.get('n_anomalies_detected', 'N/A')}")
            lines.append(f"- F1-Score: {iso.get('f1_score', 'N/A')}")
            lines.append(f"- Recall: {iso.get('recall', 'N/A')}")
            lines.append(f"- ROC-AUC: {iso.get('roc_auc', 'N/A')}\n")
        if "kmeans" in cl:
            km = cl["kmeans"]
            lines.append(f"### K-Means")
            lines.append(f"- Number of clusters: {km.get('n_clusters', 'N/A')}")
            lines.append(f"- Silhouette score: {km.get('silhouette_score', 'N/A')}")
            lines.append(f"- Top illicit clusters: {km.get('top_illicit_clusters', 'N/A')}\n")

    summary = "\n".join(lines)

    # Save
    summary_path = os.path.join(save_dir, "results_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  [Evaluation] Saved results_summary.md")

    return summary


def run_evaluation(results_dir: str, figures_dir: str,
                   gnn_histories: dict = None) -> dict:
    """Run the full evaluation and visualization pipeline.

    Args:
        results_dir: Directory containing result JSON files.
        figures_dir: Directory to save figures.
        gnn_histories: Optional dictionary of model_name -> training history.

    Returns:
        Dictionary with comparison DataFrame and summary.
    """
    print("=" * 60)
    print("Running Evaluation and Results Comparison")
    print("=" * 60)

    os.makedirs(figures_dir, exist_ok=True)

    # Load all results
    all_results = load_results(results_dir)
    print(f"  Loaded results for: {list(all_results.keys())}")

    # Build comparison table
    comparison_df = build_comparison_table(all_results)
    print(f"\n  Model Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Save comparison table as CSV
    comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    print(f"\n  Saved model_comparison.csv")

    # Plots
    plot_model_comparison(comparison_df, figures_dir)
    plot_confusion_matrices(all_results, figures_dir)

    # Training curves
    if gnn_histories:
        for model_name, history in gnn_histories.items():
            plot_training_curves(history, model_name, figures_dir)

    # Summary
    summary = generate_results_summary(all_results, comparison_df, results_dir)

    print("\n[Evaluation] Complete.")
    return {
        "comparison_df": comparison_df,
        "summary": summary,
        "all_results": all_results,
    }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    RESULTS_DIR = BASE_DIR
    FIGURES_DIR = BASE_DIR
    run_evaluation(RESULTS_DIR, FIGURES_DIR)
