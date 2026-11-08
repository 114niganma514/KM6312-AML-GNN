# Graph Neural Networks for Anti-Money Laundering: Detecting Illicit Transactions in the Bitcoin Blockchain

**KM6312 Data Mining – Group Project | Nanyang Technological University | Team 3**

| Member | Matric No. |
|---|---|
| LIANG JIAYI | G2605019J |
| FENG YINGNAN | G2603174H |
| MEI YUHONG | G2608938B |
| CHEN SIYING | G2607879H |
| MA ZIFEI | G2603605G |

🔗 Repository: https://github.com/114niganma514/KM6312-AML-GNN

📄 Final report: [`KM6312_Final_Report.docx`](KM6312_Final_Report.docx) · 📝 Proposal (as submitted; the team is listed there under its earlier label "KM6312 GROUP3"): [`KM6312_Group_Project_Proposal.docx`](KM6312_Group_Project_Proposal.docx)

**Data mining approaches covered:** Exploratory Data Analysis · Classification (supervised ML + GNNs) · Clustering / anomaly detection

This project applies Graph Neural Networks (GNNs) to detect illicit (money-laundering) transactions in the Bitcoin blockchain, using the Elliptic Bitcoin Transaction Dataset. We compare GNN performance against classical machine learning baselines and unsupervised anomaly detection methods to quantify whether and how graph structure improves illicit transaction detection.

## Project Overview

### Problem
Money laundering in cryptocurrencies is a growing problem. Traditional per-transaction feature analysis fails to capture relational patterns such as fund layering through intermediate accounts, clustering around high-degree nodes, and coordinated behavior across connected entities.

### Approach
We model Bitcoin transactions as a directed graph (transactions = nodes, payment flows = edges) and apply:
1. **Exploratory Data Analysis** - feature distributions, temporal patterns, graph topology
2. **Classical ML Baselines** - Logistic Regression, Random Forest, XGBoost, LightGBM
3. **Unsupervised Learning** - Isolation Forest, K-Means, DBSCAN
4. **Graph Neural Networks** - GCN and GraphSAGE as planned in the proposal (GAT and a 3-layer GraphSAGE were added as extra configurations in the ablation study)
5. **Supplementary experiments promised in the proposal** - validation-set threshold tuning, SMOTE, local-vs-aggregated feature ablation, temporal split for all classical models, prediction on unlabeled nodes with an illicit-subgraph case study, Isolation Forest score distribution (`proposal_experiments.py`)

### Research Questions (from the proposal)
- **RQ1.** How do representative classical ML, unsupervised, and GNN methods compare under severe class imbalance?
- **RQ2.** How much incremental value does explicit graph connectivity provide beyond local and aggregated transaction features?
- **RQ3.** Are the comparative conclusions consistent between the primary random-split evaluation and a secondary temporal-split evaluation across the 49 time steps?

## Proposal → Implementation Map

Every analysis planned in the submitted proposal ([`KM6312_Group_Project_Proposal.docx`](KM6312_Group_Project_Proposal.docx)) and where it is implemented:

| Proposal section | Planned analysis | Code | Output |
|---|---|---|---|
| 4.1 EDA | Feature distributions (illicit / licit / unknown) | `eda.py` | `01_class_distribution.png`, `03_feature_distributions.png`, `05_correlation_heatmap.png` |
| 4.1 EDA | Temporal analysis over 49 time steps | `eda.py` | `02_temporal_patterns.png` |
| 4.1 EDA | Degree distribution and graph topology | `eda.py` | `04_graph_topology.png` |
| 4.1 EDA | Visualisation of a labeled illicit subgraph | `proposal_experiments.py` (E5) | `11_case_study_subgraph.png` |
| 4.2 Preprocessing | Standardisation, 165 features (txId and time step excluded), directed graph | `preprocessing.py` | – |
| 4.2 Preprocessing | Class-weighted loss | `preprocessing.py`, `baselines.py`, `gnn_model.py` | – |
| 4.2 Preprocessing | SMOTE as an alternative | `proposal_experiments.py` (E2) | `proposal_experiments_summary.md` |
| 4.2 Preprocessing | 70 / 15 / 15 stratified random split; temporal split as robustness check | `preprocessing.py` | `train_idx.npy`, `val_idx.npy`, `test_idx.npy` (regenerated) |
| 4.2 Preprocessing | Edges crossing time-step boundaries (leakage check) | `proposal_experiments.py` (E0) | 0 of 234,355 edges cross time steps |
| 4.3 Classification | LR, RF, XGBoost, LightGBM | `baselines.py` | `baseline_results.json` |
| 4.4 Clustering | Isolation Forest, K-Means, DBSCAN | `clustering.py` | `clustering_results.json`, `06_clustering_visualization.png` |
| 4.4 Clustering | Isolation Forest anomaly-score distribution by class | `proposal_experiments.py` (E6) | `10_isolation_forest_scores.png` |
| 4.5 GNN | GraphSAGE (2 layers, 64 hidden, mean aggregator), GCN, weighted CE, early stopping | `gnn_model.py` | `gnn_*_results.json`, `*_best.pt` |
| 4.5 GNN | Transductive prediction on the 77% unlabeled nodes | `gnn_model.py`, `proposal_experiments.py` (E5) | `gnn_*_results.json`, `proposal_experiments_results.json` |
| 5 Experiments | Nine-model comparison (PR-AUC, ROC-AUC, F1, Recall, Precision) | `evaluation.py` | `model_comparison.csv`, `07_model_comparison.png`, `09_confusion_matrices.png` |
| 5 Experiments | Decision threshold tuned on the validation set | `proposal_experiments.py` (E1) | `proposal_experiments_summary.md` |
| 5 Ablation (a) | Feature-only vs feature + graph structure | `baselines.py` vs `gnn_model.py` | `model_comparison.csv` |
| 5 Ablation (b) | Local (93) vs local + aggregated (165) features | `proposal_experiments.py` (E3) | `12_ablation_results.png` |
| 5 Ablation (c) | Random vs temporal split | `tune_gnn.py` (GNN), `proposal_experiments.py` (E4, classical) | `tuning_results.json`, `12_ablation_results.png` |
| 5 Case study | Representative illicit subgraph and its structure | `proposal_experiments.py` (E5) | `11_case_study_subgraph.png` |
| 6.2 Deliverable | Reproducible pipeline on GitHub | `run_all.py` + `proposal_experiments.py` | this repository |

### Dataset
- **Elliptic Bitcoin Transaction Dataset** (Weber et al., 2019)
- 203,769 transaction nodes, 234,355 directed edges
- 165 node features
- Labels: illicit (2.2%), licit (20.6%), unknown (77.2%)
- 49 time steps (~2 weeks of Bitcoin blockchain activity)
- Source: [Kaggle - ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

### Data in this repository
| File | Size | In repo? |
|---|---|---|
| `elliptic_txs_classes.csv` (node labels) | 3.2 MB | ✅ included |
| `elliptic_txs_edgelist.csv` (graph edges) | 4.3 MB | ✅ included |
| `elliptic_txs_features.csv` (165-dim node features) | 658 MB | ❌ exceeds GitHub's 100 MB file limit — run `python download_data.py` or download from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place it in the project root |

Processed arrays (`*.npy`) are regenerated automatically by `run_all.py`.

## Project Structure (Flat Layout)

All files are in the project root for easy access:

```
KM6312-AML-GNN/
├── data_loader.py              # Dataset loading and statistics
├── eda.py                      # Exploratory data analysis (5 figures)
├── preprocessing.py            # Feature scaling, graph construction, splits
├── baselines.py                # Classical ML classifiers (LR/RF/XGB/LGB)
├── clustering.py               # Anomaly detection and clustering (IF/KM/DBSCAN)
├── gnn_model.py                # GCN/GraphSAGE/GAT models + training
├── evaluation.py               # Model comparison and visualization
├── run_all.py                  # Main pipeline runner (7 steps)
├── tune_gnn.py                 # GNN hyperparameter ablation (6 configs)
├── proposal_experiments.py     # Supplementary experiments planned in the proposal (E0–E6)
├── run_lrsched.py              # GraphSAGE with learning-rate scheduler (ablation config)
├── generate_report.py          # ICML-style report generator
├── download_data.py            # Dataset download script (Kaggle API)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── KM6312_Final_Report.docx    # Final project report (ICML style)
├── KM6312_Group_Project_Proposal.docx # Project proposal
│
├── elliptic_txs_features.csv   # Raw data (658 MB, download separately)
├── elliptic_txs_edgelist.csv   # Raw data (4.3 MB, included)
├── elliptic_txs_classes.csv    # Raw data (3.2 MB, included)
│
├── baseline_results.json       # Baseline model metrics
├── clustering_results.json     # Clustering metrics
├── gnn_gcn_results.json        # GCN results
├── gnn_graphsage_results.json  # GraphSAGE results
├── gnn_gat_results.json        # GAT results
├── gnn_graphsage3_results.json # 3-layer GraphSAGE results
├── tuning_results.json         # GNN ablation study (6 configs)
├── model_comparison.csv        # All models comparison table
├── results_summary.md          # Final results summary
├── proposal_experiments_results.json  # Supplementary experiment metrics
├── proposal_experiments_summary.md    # Supplementary experiment tables
│
├── 01_class_distribution.png   # EDA figure
├── 02_temporal_patterns.png    # EDA figure
├── 03_feature_distributions.png # EDA figure
├── 04_graph_topology.png       # EDA figure
├── 05_correlation_heatmap.png  # EDA figure
├── 06_clustering_visualization.png # Clustering figure
├── 07_model_comparison.png     # Model comparison figure
├── 09_confusion_matrices.png   # Confusion matrices
├── 10_isolation_forest_scores.png # Isolation Forest score distribution
├── 11_case_study_subgraph.png  # Illicit-subgraph case study
├── 12_ablation_results.png     # Feature / split / imbalance ablations
│
├── gcn_best.pt                 # Saved GCN model
├── graphsage_best.pt           # Saved GraphSAGE model
├── graphsage3_best.pt          # Saved 3-layer GraphSAGE model
└── gat_best.pt                 # Saved GAT model
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/114niganma514/KM6312-AML-GNN.git
cd KM6312-AML-GNN
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (recommended for this project):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

### 3. Download the dataset
```bash
python download_data.py
```

This requires Kaggle API credentials. Alternatively, download manually from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place the three CSV files in the project root.

## Usage

### Run the full pipeline
```bash
python run_all.py
```

### Run with options
```bash
# Skip EDA (faster)
python run_all.py --skip-eda

# Skip GNN training
python run_all.py --skip-gnn

# Run only GNN (requires preprocessed data)
python run_all.py --gnn-only

# Use temporal train/val/test split
python run_all.py --temporal

# Custom GNN training
python run_all.py --epochs 300 --hidden 128
```

### Run the supplementary experiments from the proposal
```bash
python proposal_experiments.py   # ~4 min on CPU, needs lightgbm, xgboost, imbalanced-learn
```

### Run GNN ablation study
```bash
python tune_gnn.py
```

### Run individual modules
```bash
python eda.py           # EDA only
python baselines.py     # Baselines only
python gnn_model.py     # GNN only
python evaluation.py    # Evaluation only
```

## Key Results

### Best Model: LightGBM (Classical ML)
| Metric | Value |
|--------|-------|
| PR-AUC | 0.9881 |
| F1-Score | 0.9607 |
| Recall | 0.9501 |
| Precision | 0.9715 |

### Best GNN: 3-Layer GraphSAGE (h=128)
| Metric | Value |
|--------|-------|
| PR-AUC | 0.9272 |
| F1-Score | 0.7972 |
| Recall | 0.9135 |
| Precision | 0.7072 |

### Key Finding
Classical ML (LightGBM) outperforms GNNs on this dataset because the graph consists of 49 disconnected components (one per time step) with very sparse connectivity (average degree 1.15). However, DBSCAN clustering reveals that graph-structured clusters have 13x higher illicit ratio than noise points, demonstrating the value of graph structure for unsupervised anomaly detection.

### Supplementary Results (`proposal_experiments.py`, report Section 5.5)

| Analysis | Finding |
|---|---|
| Threshold tuned on validation (E1) | LightGBM F1 0.9607 → **0.9627** (thr 0.539); Logistic Regression F1 0.6242 → **0.7838** (thr 0.816). Tree models were already near-optimal at 0.5. |
| SMOTE vs class weights (E2) | SMOTE helps Logistic Regression (PR-AUC 0.7963 → 0.8244) but not the boosted trees (LightGBM 0.9881 → 0.9858). Class weighting is kept as the default. |
| Local (93) vs local + aggregated (165) features (E3) | Removing the 72 neighbour-aggregated features lowers LightGBM PR-AUC 0.9881 → 0.9804 and F1 0.9607 → 0.9247: the aggregated features already carry much of the relational signal (RQ2). |
| Random vs temporal split (E4) | All models drop sharply on future time steps 42–49: LightGBM PR-AUC 0.9881 → **0.5530**, GraphSAGE 0.9217 → **0.3361**. The ranking (tree ensembles > GNN > LR) is consistent across splits (RQ3). |
| Cross-time-step edges (E0) | 0 of 234,355 edges cross time steps, so the temporal split has no edge leakage. |
| Unlabeled prediction & case study (E5) | LightGBM flags 12,307 of 157,205 unlabeled transactions (7.8%). On average 25.2% of their neighbours are illicit or flagged, vs 5.5% for other unlabeled nodes; the largest connected illicit cluster (time step 29) contains 183 transactions (Figure 11). 37.1% of labeled edges touching an illicit node link two illicit nodes, vs a 9.8% illicit base rate (homophily). |
| Isolation Forest scores (E6) | Illicit transactions are *less* anomalous than licit ones (ROC-AUC 0.10): illicit activity in Elliptic is repetitive and pattern-like rather than outlying, which explains the weak unsupervised detection. |

Minor differences (≤ 0.002) between these reference numbers and `baseline_results.json` come from multi-threaded tree training.

## Evaluation Metrics

Due to severe class imbalance (2.2% illicit), we use:
- **PR-AUC** (primary metric) - Area Under Precision-Recall Curve
- **F1-Score** - Harmonic mean of precision and recall
- **Recall** - Sensitivity to illicit transactions (minimize false negatives)
- **Precision** - Accuracy of positive predictions
- **ROC-AUC** - Area Under ROC Curve (secondary)

## Methodology Details

### Graph Construction
- **Nodes**: Bitcoin transactions (203,769)
- **Edges**: Directed payment flows (234,355)
- **Node features**: 165-dimensional vector
- **Node labels**: 1 = illicit, 0 = licit, -1 = unknown

### GNN Architectures
- **GCN**: 2-layer Graph Convolutional Network (Kipf & Welling, 2017)
- **GraphSAGE**: 2/3-layer Graph SAmple and aggreGatE (Hamilton et al., 2017)
- **GAT**: Graph Attention Network with 4 attention heads (Veličković et al., 2018)

### Training
- Loss: Weighted cross-entropy (class weights inversely proportional to frequency)
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Early stopping: patience=30 epochs on validation loss
- Train/val/test split: 70%/15%/15% (stratified random, seed 42); temporal split: time steps 1–34 / 35–41 / 42–49
- Decision threshold: 0.5 by default; validation-tuned thresholds reported in `proposal_experiments_summary.md`

### Classical and Unsupervised Settings
- Logistic Regression (L2, class weights); Random Forest (200 trees, max depth 15); XGBoost and LightGBM (300 trees, max depth 8, learning rate 0.05, class weighting)
- Isolation Forest (200 trees, contamination 2%); K-Means (k = 5); DBSCAN (eps 0.5, min_samples 5) on 20 PCA components (64.7% variance)
- Features are standardised over all 203,769 nodes (no labels used)

## References

1. Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* arXiv:1908.02591.

2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* Advances in Neural Information Processing Systems (NeurIPS), 30.

3. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* International Conference on Learning Representations (ICLR).

4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). *Graph Attention Networks.* International Conference on Learning Representations (ICLR).

5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

6. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems (NeurIPS), 30, 3146–3154.

7. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.

8. Elliptic Bitcoin Transaction Dataset. Kaggle. https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

## Team

**Team 3** – LIANG JIAYI, FENG YINGNAN, MEI YUHONG, CHEN SIYING, MA ZIFEI

## License

This project is for academic purposes (KM6312 Data Mining course, NTU).
