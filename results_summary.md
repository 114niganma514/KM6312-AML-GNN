# KM6312 AML GNN Project - Results Summary

## Model Comparison (sorted by PR-AUC)

| Rank | Model | Category | PR-AUC | ROC-AUC | F1-Score | Recall | Precision |
|------|-------|----------|--------|---------|----------|--------|-----------|
| 1 | LightGBM | Classical ML | 0.9881 | 0.9981 | 0.9607 | 0.9501 | 0.9715 |
| 2 | XGBoost | Classical ML | 0.9867 | 0.9978 | 0.9594 | 0.9355 | 0.9846 |
| 3 | Random Forest | Classical ML | 0.9798 | 0.9962 | 0.9462 | 0.9150 | 0.9796 |
| 4 | GraphSAGE | Graph Neural Network | 0.9075 | 0.9799 | 0.7536 | 0.9106 | 0.6429 |
| 5 | GCN | Graph Neural Network | 0.8334 | 0.9720 | 0.6990 | 0.9076 | 0.5684 |
| 6 | Logistic Regression | Classical ML | 0.7963 | 0.9695 | 0.6227 | 0.9267 | 0.4688 |

## Best Model

**LightGBM** (Classical ML) achieved the highest PR-AUC of **0.9881**, with F1-Score 0.9607, Recall 0.9501, and Precision 0.9715.

## GNN vs Classical ML

- Best GNN PR-AUC: **0.9075**
- Best Classical ML PR-AUC: **0.9881**
- Difference: **-0.0806** (Classical ML outperforms)

## Unsupervised Learning Results

### Isolation Forest
- Anomalies detected: 4076
- F1-Score: 0.0
- Recall: 0.0
- ROC-AUC: 0.1016

### K-Means
- Number of clusters: 5
- Silhouette score: 0.0527
- Top illicit clusters: [1, 0, 2]
