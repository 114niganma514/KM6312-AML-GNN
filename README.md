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

📄 Final report: [`KM6312_Final_Report.docx`](KM6312_Final_Report.docx) · 📝 Proposal: [`KM6312_Group_Project_Proposal.docx`](KM6312_Group_Project_Proposal.docx)

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
4. **Graph Neural Networks** - GCN, GraphSAGE, GAT (with ablation study)

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
KM6312_AML_GNN_Project/
├── data_loader.py              # Dataset loading and statistics
├── eda.py                      # Exploratory data analysis (5 figures)
├── preprocessing.py            # Feature scaling, graph construction, splits
├── baselines.py                # Classical ML classifiers (LR/RF/XGB/LGB)
├── clustering.py               # Anomaly detection and clustering (IF/KM/DBSCAN)
├── gnn_model.py                # GCN/GraphSAGE/GAT models + training
├── evaluation.py               # Model comparison and visualization
├── run_all.py                  # Main pipeline runner (7 steps)
├── tune_gnn.py                 # GNN hyperparameter ablation (6 configs)
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
│
├── 01_class_distribution.png   # EDA figure
├── 02_temporal_patterns.png    # EDA figure
├── 03_feature_distributions.png # EDA figure
├── 04_graph_topology.png       # EDA figure
├── 05_correlation_heatmap.png  # EDA figure
├── 06_clustering_visualization.png # Clustering figure
├── 07_model_comparison.png     # Model comparison figure
├── 09_confusion_matrices.png   # Confusion matrices
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
- Train/val/test split: 70%/15%/15% (stratified random)

## References

1. Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* arXiv:1908.02591.

2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* Advances in Neural Information Processing Systems (NeurIPS), 30.

3. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* International Conference on Learning Representations (ICLR).

4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). *Graph Attention Networks.* International Conference on Learning Representations (ICLR).

5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

6. Elliptic Bitcoin Transaction Dataset. Kaggle. https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

## Team

**Team 3** – LIANG JIAYI, FENG YINGNAN, MEI YUHONG, CHEN SIYING, MA ZIFEI

## License

This project is for academic purposes (KM6312 Data Mining course, NTU).
