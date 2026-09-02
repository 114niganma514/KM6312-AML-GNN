"""
Exploratory Data Analysis (EDA) for the Elliptic Bitcoin Transaction Dataset.

This module produces visualizations and statistical summaries to understand:
  - Feature distributions (illicit vs licit)
  - Temporal patterns across 49 time steps
  - Graph topology (degree distribution, connected components)
  - Class imbalance and label scarcity

All figures are saved as PNG files in the project directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from data_loader import load_all, get_dataset_stats, get_labeled_mask

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150


def plot_class_distribution(merged_df: pd.DataFrame, save_dir: str) -> None:
    """Plot the class distribution (illicit / licit / unknown)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full distribution including unknown
    class_counts = merged_df["class"].value_counts()
    colors = {"unknown": "#BDBDBD", "licit": "#4CAF50", "illicit": "#F44336"}
    axes[0].bar(class_counts.index, class_counts.values,
                 color=[colors.get(c, "#999") for c in class_counts.index])
    axes[0].set_title("Full Class Distribution (all 203,769 transactions)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    # Labeled-only distribution
    labeled = merged_df[merged_df["label"] != -1]
    labeled_counts = labeled["class"].value_counts()
    axes[1].bar(labeled_counts.index, labeled_counts.values,
                color=[colors.get(c, "#999") for c in labeled_counts.index])
    axes[1].set_title("Labeled-Only Distribution (illicit vs licit)")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(labeled_counts.values):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_class_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] Saved 01_class_distribution.png")


def plot_temporal_patterns(merged_df: pd.DataFrame, save_dir: str) -> None:
    """Plot transaction volume and illicit ratio across time steps."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Total transactions per time step
    ts_counts = merged_df.groupby("time_step").size()
    axes[0].plot(ts_counts.index, ts_counts.values, marker="o", markersize=3,
                  color="#1976D2", linewidth=1.5)
    axes[0].set_title("Transaction Volume Across 49 Time Steps")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].fill_between(ts_counts.index, ts_counts.values, alpha=0.1, color="#1976D2")

    # Illicit ratio per time step (among labeled)
    labeled = merged_df[merged_df["label"] != -1]
    illicit_by_ts = labeled.groupby("time_step")["label"].mean()
    axes[1].plot(illicit_by_ts.index, illicit_by_ts.values, marker="s", markersize=3,
                  color="#F44336", linewidth=1.5)
    axes[1].set_title("Illicit Ratio Across Time Steps (among labeled transactions)")
    axes[1].set_ylabel("Illicit Ratio")
    axes[1].set_xlabel("Time Step")
    axes[1].fill_between(illicit_by_ts.index, illicit_by_ts.values, alpha=0.1, color="#F44336")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_temporal_patterns.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] Saved 02_temporal_patterns.png")


def plot_feature_distributions(merged_df: pd.DataFrame, save_dir: str, n_features: int = 9) -> None:
    """Plot distribution of selected local features for illicit vs licit."""
    labeled = merged_df[merged_df["label"] != -1]
    feature_cols = [c for c in merged_df.columns if c.startswith("feature_")]
    selected = feature_cols[:n_features]

    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(selected):
        ax = axes[idx]
        for label, color, name in [(0, "#4CAF50", "Licit"), (1, "#F44336", "Illicit")]:
            subset = labeled[labeled["label"] == label][col]
            ax.hist(subset, bins=50, alpha=0.5, color=color, label=name, density=True)
        ax.set_title(f"{col} distribution")
        ax.legend(fontsize=8)
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")

    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_feature_distributions.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] Saved 03_feature_distributions.png")


def plot_graph_topology(edgelist_df: pd.DataFrame, merged_df: pd.DataFrame, save_dir: str) -> None:
    """Plot graph degree distribution and connected components."""
    # Build directed graph
    G = nx.from_pandas_edgelist(edgelist_df, source="txId1", target="txId2",
                                 create_using=nx.DiGraph)

    # Degree distribution
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [d for _, d in G.degree()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # In-degree
    axes[0].hist(in_degrees, bins=100, color="#1976D2", alpha=0.7, density=True)
    axes[0].set_title("In-Degree Distribution")
    axes[0].set_xlabel("In-Degree")
    axes[0].set_ylabel("Density")
    axes[0].set_yscale("log")

    # Out-degree
    axes[1].hist(out_degrees, bins=100, color="#F44336", alpha=0.7, density=True)
    axes[1].set_title("Out-Degree Distribution")
    axes[1].set_xlabel("Out-Degree")
    axes[1].set_ylabel("Density")
    axes[1].set_yscale("log")

    # Total degree
    axes[2].hist(total_degrees, bins=100, color="#4CAF50", alpha=0.7, density=True)
    axes[2].set_title("Total Degree Distribution")
    axes[2].set_xlabel("Total Degree")
    axes[2].set_ylabel("Density")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_graph_topology.png"), bbox_inches="tight")
    plt.close()

    # Connected components stats
    weak_components = list(nx.weakly_connected_components(G))
    largest_cc = max(weak_components, key=len)

    stats = {
        "n_nodes_in_graph": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_weakly_connected_components": len(weak_components),
        "largest_component_size": len(largest_cc),
        "largest_component_ratio": round(len(largest_cc) / G.number_of_nodes(), 4),
        "avg_in_degree": round(np.mean(in_degrees), 4),
        "avg_out_degree": round(np.mean(out_degrees), 4),
        "max_in_degree": max(in_degrees),
        "max_out_degree": max(out_degrees),
    }

    print("  [EDA] Saved 04_graph_topology.png")
    print("  [EDA] Graph statistics:")
    for k, v in stats.items():
        print(f"    {k:35s}: {v}")

    return stats


def plot_correlation_heatmap(merged_df: pd.DataFrame, save_dir: str) -> None:
    """Plot correlation heatmap of a subset of features."""
    labeled = merged_df[merged_df["label"] != -1]
    # Select a subset of features for readability
    feature_cols = [c for c in merged_df.columns if c.startswith("feature_")][:20]
    selected = feature_cols + ["label"]

    corr = labeled[selected].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap (labeled transactions, subset)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_correlation_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("  [EDA] Saved 05_correlation_heatmap.png")


def run_eda(data_dir: str, save_dir: str) -> dict:
    """Run the full EDA pipeline and save all figures.

    Args:
        data_dir: Path to raw data directory.
        save_dir: Path to save figures.

    Returns:
        Dictionary with dataset and graph statistics.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Running Exploratory Data Analysis (EDA)")
    print("=" * 60)

    features, edgelist, classes, merged = load_all(data_dir)
    stats = get_dataset_stats(merged, edgelist)

    print("\nDataset Statistics:")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    print("\nGenerating figures...")
    plot_class_distribution(merged, save_dir)
    plot_temporal_patterns(merged, save_dir)
    plot_feature_distributions(merged, save_dir)
    graph_stats = plot_graph_topology(edgelist, merged, save_dir)
    plot_correlation_heatmap(merged, save_dir)

    print("\n[EDA] Complete. All figures saved to:", save_dir)
    return {**stats, **graph_stats}


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    FIGURE_DIR = BASE_DIR
    run_eda(DATA_DIR, FIGURE_DIR)
