# descriptive analysis of the transaction graph
# - edges across time steps, label homophily on edges
# - LightGBM predictions on unlabeled nodes and their neighbourhoods
# - case study figure (11) and Isolation Forest score figure (10)
# usage: python graph_analysis.py

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from data_loader import load_all
from preprocessing import preprocess_pipeline
from baselines import get_baseline_models
from run_experiments import f1_threshold, fit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
SEED = 0


def main():
    _, edges, _, merged = load_all(BASE_DIR)
    pre = preprocess_pipeline(merged, edges, temporal=False)
    X, y, ei = pre["X"], pre["labels"], pre["edge_index"]
    ts = merged["time_step"].values
    lab = y != -1
    res = {}

    # edges across time steps
    res["edges"] = {"n_edges": int(ei.shape[1]),
                    "n_cross_time_step": int((ts[ei[0]] != ts[ei[1]]).sum())}

    # label homophily (labelled endpoints only)
    s, d = ei
    both = lab[s] & lab[d]
    touch = both & ((y[s] == 1) | (y[d] == 1))
    ill_ill = both & (y[s] == 1) & (y[d] == 1)
    res["homophily"] = {
        "labelled_edges": int(both.sum()),
        "labelled_edges_touching_illicit": int(touch.sum()),
        "share_illicit_to_illicit": round(float(ill_ill.sum() / touch.sum()), 4),
        "illicit_share_of_labelled_nodes": round(float((y == 1).sum() / lab.sum()), 4),
    }

    # isolation forest scores (figure 10)
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1).fit(X)
    score = -iso.decision_function(X)
    groups = {"illicit": y == 1, "licit": y == 0, "unknown": y == -1}
    auc = roc_auc_score(y[lab] == 1, score[lab])
    res["isolation_forest"] = {f"median_{g}": round(float(np.median(score[m])), 4) for g, m in groups.items()}
    res["isolation_forest"]["roc_auc_illicit_vs_licit"] = round(float(auc), 4)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.9), 80)
    for g, c in [("unknown", "#9E9E9E"), ("licit", "#4CAF50"), ("illicit", "#F44336")]:
        ax.hist(score[groups[g]], bins=bins, density=True, alpha=0.5, color=c,
                label=f"{g} (n={groups[g].sum():,})")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title(f"Isolation Forest scores by class (ROC-AUC illicit vs licit = {auc:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "10_isolation_forest_scores.png"), dpi=150)
    plt.close()

    # lightgbm on the random split, threshold from validation
    tr, va = pre["train_idx"], pre["val_idx"]
    model = get_baseline_models(pre["class_weights"], random_state=SEED)["LightGBM"]
    fit("LightGBM", model, X[tr], y[tr], X[va], y[va])
    thr = f1_threshold(y[va], model.predict_proba(X[va])[:, 1])
    prob = model.predict_proba(X)[:, 1]
    unk = y == -1
    flagged = unk & (prob >= thr)

    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from(ei.T.tolist())

    def illicit_neighbour_share(nodes):
        vals = [np.mean([y[n] == 1 for n in G.neighbors(v)]) for v in nodes if G.degree(v) > 0]
        return round(float(np.mean(vals)), 4)

    rng = np.random.default_rng(SEED)
    others = np.where(unk & ~flagged)[0]
    others = rng.choice(others, size=min(20000, len(others)), replace=False)
    res["unlabeled_lightgbm"] = {
        "val_threshold": round(thr, 4),
        "n_unlabeled": int(unk.sum()),
        "n_flagged": int(flagged.sum()),
        "share_flagged": round(float(flagged.sum() / unk.sum()), 4),
        # share of neighbours with a KNOWN illicit label (no predictions used)
        "known_illicit_neighbour_share_flagged": illicit_neighbour_share(np.where(flagged)[0]),
        "known_illicit_neighbour_share_other_sample": illicit_neighbour_share(others),
    }

    # case study: largest connected group of labelled illicit nodes (figure 11)
    ill_sub = G.subgraph(np.where(y == 1)[0])
    core = max(nx.connected_components(ill_sub), key=len)
    nodes = set(core)
    for v in list(core):
        nodes.update(G.neighbors(v))
    H = G.subgraph(nodes)
    status = np.where(y == 1, 1, np.where(y == 0, 0, np.where(flagged, 2, -1)))
    ts_case = int(pd.Series(ts[list(core)]).mode()[0])
    res["case_study"] = {
        "time_step": ts_case,
        "n_illicit": len(core),
        "n_nodes_shown": H.number_of_nodes(),
        "composition": {k: int((status[list(H.nodes)] == v).sum())
                        for k, v in [("illicit", 1), ("licit", 0),
                                     ("unknown_flagged", 2), ("unknown_other", -1)]},
    }
    colors = {1: "#F44336", 0: "#4CAF50", 2: "#FF9800", -1: "#BDBDBD"}
    names = {1: "illicit (label)", 0: "licit (label)", 2: "unknown, flagged by LightGBM",
             -1: "unknown, not flagged"}
    pos = nx.spring_layout(H, seed=SEED, k=0.25)
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3, width=0.6)
    for k in [-1, 0, 2, 1]:
        nl = [n for n in H.nodes if status[n] == k]
        if nl:
            nx.draw_networkx_nodes(H, pos, nodelist=nl, node_color=colors[k], ax=ax,
                                   node_size=40 if k in (1, 2) else 18,
                                   label=f"{names[k]} ({len(nl)})")
    ax.set_title(f"Largest connected group of illicit transactions\n"
                 f"(time step {ts_case}, {len(core)} illicit nodes) and their 1-hop neighbours")
    ax.legend(loc="lower left", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "11_case_study_subgraph.png"), dpi=150)
    plt.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "graph_analysis.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
