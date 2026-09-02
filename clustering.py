"""
Unsupervised Anomaly Detection and Clustering for Illicit Transaction Discovery.

This module applies label-free methods to identify anomalous transactions
and potential fraud rings in the Elliptic dataset:

  - Isolation Forest: anomaly detection on 166-dim feature space
  - K-Means: clustering transactions by feature similarity
  - DBSCAN: density-based clustering for fraud ring detection

These methods satisfy the course's clustering requirement and provide
a complementary perspective to supervised classification.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def run_isolation_forest(X: np.ndarray, labels: np.ndarray,
                         contamination: float = 0.02,
                         random_state: int = 42) -> dict:
    """Run Isolation Forest anomaly detection.

    Args:
        X: Feature matrix (all nodes, including unlabeled).
        labels: True labels (-1=unknown, 0=licit, 1=illicit).
        contamination: Expected proportion of anomalies.
        random_state: Random seed.

    Returns:
        Dictionary with anomaly scores, predictions, and evaluation.
    """
    print("  Running Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=random_state, n_jobs=-1
    )
    anomaly_scores = iso_forest.fit_predict(X)  # 1 = normal, -1 = anomaly
    decision_scores = iso_forest.decision_function(X)  # lower = more anomalous

    # Evaluate on labeled nodes
    labeled_mask = labels != -1
    y_true = (labels[labeled_mask] == 1).astype(int)
    y_pred = (anomaly_scores[labeled_mask] == -1).astype(int)

    # Metrics
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
    # For ROC-AUC, use negative decision score (lower = more anomalous)
    y_prob = -decision_scores[labeled_mask]

    results = {
        "n_anomalies_detected": int((anomaly_scores == -1).sum()),
        "n_anomalies_labeled": int((y_pred == 1).sum()),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
    }
    try:
        results["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        results["roc_auc"] = 0.0

    print(f"    Anomalies detected: {results['n_anomalies_detected']}")
    print(f"    F1: {results['f1_score']:.4f}, Recall: {results['recall']:.4f}, "
          f"Precision: {results['precision']:.4f}, ROC-AUC: {results['roc_auc']:.4f}")

    return {
        "anomaly_scores": anomaly_scores,
        "decision_scores": decision_scores,
        "results": results,
    }


def run_kmeans(X: np.ndarray, labels: np.ndarray, n_clusters: int = 10,
               random_state: int = 42) -> dict:
    """Run K-Means clustering and analyze cluster enrichment for illicit activity.

    Args:
        X: Feature matrix.
        labels: True labels.
        n_clusters: Number of clusters.
        random_state: Random seed.

    Returns:
        Dictionary with cluster labels, metrics, and enrichment analysis.
    """
    print(f"  Running K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Clustering quality metrics (on labeled subset for comparison)
    labeled_mask = labels != -1
    silhouette = silhouette_score(X[labeled_mask], cluster_labels[labeled_mask])
    calinski = calinski_harabasz_score(X[labeled_mask], cluster_labels[labeled_mask])
    davies = davies_bouldin_score(X[labeled_mask], cluster_labels[labeled_mask])

    # Cluster enrichment: which clusters have high illicit ratio?
    labeled_df = pd.DataFrame({
        "cluster": cluster_labels[labeled_mask],
        "label": labels[labeled_mask]
    })
    cluster_stats = labeled_df.groupby("cluster").agg(
        n_total=("label", "count"),
        n_illicit=("label", lambda x: (x == 1).sum()),
        illicit_ratio=("label", lambda x: round((x == 1).mean(), 4))
    ).sort_values("illicit_ratio", ascending=False)

    results = {
        "n_clusters": n_clusters,
        "silhouette_score": round(float(silhouette), 4),
        "calinski_harabasz": round(float(calinski), 4),
        "davies_bouldin": round(float(davies), 4),
        "cluster_enrichment": cluster_stats.to_dict("index"),
        "top_illicit_clusters": cluster_stats.head(3).index.tolist(),
    }

    print(f"    Silhouette: {results['silhouette_score']:.4f}")
    print(f"    Top illicit clusters: {results['top_illicit_clusters']}")
    print(f"    Cluster enrichment (top 5):")
    print(cluster_stats.head())

    return {
        "cluster_labels": cluster_labels,
        "model": kmeans,
        "results": results,
    }


def run_dbscan(X: np.ndarray, labels: np.ndarray, eps: float = 0.5,
               min_samples: int = 5) -> dict:
    """Run DBSCAN density-based clustering.

    Note: DBSCAN on 200k x 166 data can be slow. We use PCA-reduced
    features for efficiency.

    Args:
        X: Feature matrix (will be PCA-reduced).
        labels: True labels.
        eps: DBSCAN epsilon parameter.
        min_samples: DBSCAN min_samples parameter.

    Returns:
        Dictionary with cluster labels and statistics.
    """
    print("  Running DBSCAN (on PCA-reduced features)...")
    # Reduce dimensionality for DBSCAN efficiency
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"    PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    cluster_labels = dbscan.fit_predict(X_pca)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()

    # Analyze illicit ratio in clusters vs noise
    labeled_mask = labels != -1
    labeled_clusters = cluster_labels[labeled_mask]
    labeled_true = labels[labeled_mask]

    noise_illicit_ratio = (labeled_true[labeled_clusters == -1] == 1).mean() if (labeled_clusters == -1).sum() > 0 else 0
    cluster_illicit_ratio = (labeled_true[labeled_clusters != -1] == 1).mean() if (labeled_clusters != -1).sum() > 0 else 0

    results = {
        "n_clusters": n_clusters,
        "n_noise_points": int(n_noise),
        "noise_ratio": round(float(n_noise / len(cluster_labels)), 4),
        "noise_illicit_ratio": round(float(noise_illicit_ratio), 4),
        "cluster_illicit_ratio": round(float(cluster_illicit_ratio), 4),
    }

    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points: {n_noise} ({results['noise_ratio']:.2%})")
    print(f"    Illicit ratio in noise: {results['noise_illicit_ratio']:.4f}")
    print(f"    Illicit ratio in clusters: {results['cluster_illicit_ratio']:.4f}")

    return {
        "cluster_labels": cluster_labels,
        "model": dbscan,
        "pca_model": pca,
        "results": results,
    }


def plot_clustering_visualization(X: np.ndarray, labels: np.ndarray,
                                   cluster_labels: np.ndarray,
                                   anomaly_scores: np.ndarray,
                                   save_dir: str) -> None:
    """Create 2D PCA visualization of clustering and anomaly results."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True labels (labeled only)
    labeled_mask = labels != -1
    scatter = axes[0].scatter(X_2d[labeled_mask, 0], X_2d[labeled_mask, 1],
                               c=labels[labeled_mask], cmap="RdYlGn_r", s=1, alpha=0.5)
    axes[0].set_title("True Labels (labeled transactions)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # K-Means clusters
    scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                               cmap="tab20", s=1, alpha=0.5)
    axes[1].set_title("K-Means Clusters")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    # Isolation Forest anomalies
    anomaly_color = np.where(anomaly_scores == -1, 1, 0)
    scatter = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=anomaly_color,
                               cmap="coolwarm", s=1, alpha=0.5)
    axes[2].set_title("Isolation Forest Anomalies")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_clustering_visualization.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  [Clustering] Saved 06_clustering_visualization.png")


def run_clustering_pipeline(preprocessed: dict, save_dir: str = None,
                            n_clusters: int = 10) -> dict:
    """Run the full unsupervised learning pipeline.

    Args:
        preprocessed: Dictionary from preprocessing.preprocess_pipeline().
        save_dir: Optional directory to save results and figures.
        n_clusters: Number of clusters for K-Means.

    Returns:
        Dictionary with all unsupervised results.
    """
    print("=" * 60)
    print("Running Unsupervised Anomaly Detection and Clustering")
    print("=" * 60)

    X = preprocessed["X"]
    labels = preprocessed["labels"]

    # 1. Isolation Forest
    iso_results = run_isolation_forest(X, labels)

    # 2. K-Means
    kmeans_results = run_kmeans(X, labels, n_clusters=n_clusters)

    # 3. DBSCAN
    dbscan_results = run_dbscan(X, labels)

    # 4. Visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_dir = os.path.join(os.path.dirname(save_dir), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plot_clustering_visualization(
            X, labels, kmeans_results["cluster_labels"],
            iso_results["anomaly_scores"], fig_dir
        )

        # Save results JSON
        summary = {
            "isolation_forest": iso_results["results"],
            "kmeans": {k: v for k, v in kmeans_results["results"].items() if k != "cluster_enrichment"},
            "dbscan": dbscan_results["results"],
        }
        with open(os.path.join(save_dir, "clustering_results.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Results saved to {save_dir}/clustering_results.json")

    print("\n[Clustering] Complete.")
    return {
        "isolation_forest": iso_results,
        "kmeans": kmeans_results,
        "dbscan": dbscan_results,
    }


if __name__ == "__main__":
    from data_loader import load_all
    from preprocessing import preprocess_pipeline

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = BASE_DIR
    RESULTS_DIR = BASE_DIR

    _, edgelist, _, merged = load_all(DATA_DIR)
    preprocessed = preprocess_pipeline(merged, edgelist, temporal=False)
    run_clustering_pipeline(preprocessed, save_dir=RESULTS_DIR)
