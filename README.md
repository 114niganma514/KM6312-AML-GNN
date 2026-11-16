# Graph Neural Networks for Anti-Money Laundering on the Elliptic Bitcoin Dataset

**KM6312 Data Mining – Group Project | Nanyang Technological University | Team 3**

| Member | Matric No. |
|---|---|
| LIANG JIAYI | G2605019J |
| FENG YINGNAN | G2603174H |
| MEI YUHONG | G2608938B |
| CHEN SIYING | G2607879H |
| MA ZIFEI | G2603605G |

Repository: https://github.com/114niganma514/KM6312-AML-GNN

Final report: [`KM6312_Final_Report.docx`](KM6312_Final_Report.docx) · Proposal (as submitted, team listed as "KM6312 GROUP3"): [`KM6312_Group_Project_Proposal.docx`](KM6312_Group_Project_Proposal.docx)

Data mining approaches: exploratory data analysis, classification (classical ML and GNNs), clustering / anomaly detection.

We compare GCN, GraphSAGE and GAT with Logistic Regression, Random Forest, XGBoost and LightGBM, and with
Isolation Forest, K-Means and DBSCAN, on the Elliptic Bitcoin transaction graph.

## Research questions

- **RQ1** How do classical, unsupervised and GNN methods compare under severe class imbalance?
- **RQ2** How much does explicit graph connectivity add beyond local and aggregated features?
- **RQ3** Do the conclusions hold under a temporal split as well as a random split?

## Dataset

Elliptic Bitcoin dataset (Weber et al., 2019): 203,769 transactions, 234,355 directed edges, 49 time steps
(about two weeks apart). 4,545 illicit, 42,019 licit, 157,205 unknown. Features: 94 local (incl. time step) +
72 aggregated over one-hop backward/forward neighbours; we drop the time step and use 165.

| File | In repo? |
|---|---|
| `elliptic_txs_classes.csv` | yes |
| `elliptic_txs_edgelist.csv` | yes |
| `elliptic_txs_features.csv` (658 MB) | no, over GitHub's 100 MB limit. Run `python download_data.py` or download from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |

## Protocol

- Splits: random stratified 70/15/15 (seed 42); temporal 1-34 / 35-41 / 42-49; Weber split train 1-34, test 35-49.
- Scaler: all nodes for the random split (transductive, no labels); time steps 1-34 only for temporal splits.
- Seeds 0-4 for every supervised model (GAT: 0-2). Results are mean ± std.
- GNN model selection and decision thresholds use the validation set only.
- PR-AUC is the main metric (illicit class).

## Files

| File | What it does |
|---|---|
| `data_loader.py` | load the three CSVs |
| `eda.py` | EDA figures 01-05 |
| `preprocessing.py` | scaling, splits, edge index, PyG data |
| `baselines.py` | LR, RF, XGBoost, LightGBM and metrics |
| `clustering.py` | Isolation Forest, K-Means, DBSCAN |
| `gnn_model.py` | GCN, GraphSAGE, GAT, training loop |
| `run_experiments.py` | all supervised runs over seeds + clustering, writes `results/` |
| `graph_analysis.py` | cross-step edges, homophily, unlabeled predictions, figures 10-11 |
| `make_figures.py` | `results/summary.md`, figures 07, 09, 12 |
| `run_all.py` | runs everything in order |
| `download_data.py` | Kaggle download |

Results: `results/classical_runs.json`, `classical_summary.json`, `gnn_runs.json`, `gnn_summary.json`,
`clustering_results.json`, `graph_analysis.json`, `summary.md`.

## How to run

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
python download_data.py      # or put the CSVs in the project root
python run_all.py            # about 2.5 h on a laptop CPU
python run_all.py --skip-eda --seeds 1   # quick check
```

On Windows with Anaconda, if PyTorch fails with `OMP: Error #15`, set `KMP_DUPLICATE_LIB_OK=TRUE` first.

## Results

### Classical models (test PR-AUC / F1, mean ± std over 5 seeds)

| Model | PR-AUC (random) | F1 (random) | PR-AUC (temporal, test 42-49) |
|---|---|---|---|
| Logistic Regression | 0.7963 ± 0.0000 | 0.6242 ± 0.0000 | 0.1891 ± 0.0000 |
| Random Forest | 0.9790 ± 0.0003 | 0.9422 ± 0.0015 | 0.5430 ± 0.0024 |
| XGBoost | 0.9865 ± 0.0000 | 0.9565 ± 0.0000 | 0.5490 ± 0.0000 |
| LightGBM | 0.9881 ± 0.0000 | 0.9607 ± 0.0000 | 0.5555 ± 0.0000 |

XGBoost, LightGBM and LR have no random subsampling here, so their std is 0.

### GNNs, random split (selected on validation: **GraphSAGE (2L, h=64)**)

| Model | Val PR-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|
| GCN (2L, h=64) | 0.8257 ± 0.0057 | 0.8225 ± 0.0068 | 0.6799 ± 0.0089 |
| **GraphSAGE (2L, h=64)** | 0.9246 ± 0.0038 | 0.9184 ± 0.0033 | 0.7883 ± 0.0078 |
| GraphSAGE (2L, h=128) | 0.9207 ± 0.0047 | 0.9166 ± 0.0034 | 0.7819 ± 0.0102 |
| GraphSAGE (3L, h=128) | 0.9216 ± 0.0067 | 0.9174 ± 0.0041 | 0.7785 ± 0.0135 |
| GAT (2L, h=64, 4 heads) | 0.8901 ± 0.0010 | 0.8711 ± 0.0022 | 0.5830 ± 0.0112 |
| GraphSAGE (2L, h=64) + LR sched | 0.9242 ± 0.0034 | 0.9191 ± 0.0042 | 0.7913 ± 0.0099 |

### GNNs, temporal split

| Model | Test PR-AUC |
|---|---|
| GCN (2L, h=64) | 0.2221 ± 0.0374 |
| GraphSAGE (2L, h=64) | 0.2753 ± 0.0161 |
| GraphSAGE (3L, h=128) | 0.2561 ± 0.0252 |

### Other findings

- Weber split (test 35-49): Random Forest illicit F1 0.8085 ± 0.0009, close to 0.788-0.796 in Weber et al.
- Local features only (93): LightGBM PR-AUC 0.9804 ± 0.0000 vs 0.9881 ± 0.0000 with all 165.
- SMOTE: helps LR (PR-AUC 0.8265 ± 0.0024), not LightGBM (0.9848 ± 0.0004).
- 0 of 234,355 edges cross time steps.
- Homophily: 37.0% of labelled edges touching an illicit node link two illicit nodes (illicit share of labelled nodes 9.8%).
- LightGBM flags 12,307 of 157,205 unknown nodes; 4.1% of their neighbours are labelled illicit vs 1.4% for other unknown nodes.
- Isolation Forest ROC-AUC (illicit vs licit) 0.102: illicit nodes are not outliers in feature space.
- DBSCAN (feature space, labelled nodes only): 52.2% illicit in clusters vs 3.9% in noise.

Full tables: [`results/summary.md`](results/summary.md).

## References

1. Weber, M. et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. arXiv:1908.02591.
2. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
4. Veličković, P. et al. (2018). Graph Attention Networks. ICLR.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
6. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
7. Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16.
8. Elliptic Data Set. Kaggle. https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

For academic use (KM6312, NTU).
