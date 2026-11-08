"""
Supplementary experiments that complete the analyses planned in the
submitted project proposal (Team 3).

Proposal item -> experiment in this script
  Sec 4.2  SMOTE oversampling as an alternative to class weights  -> E2
  Sec 4.2  Effect of edges that cross time-step boundaries         -> E0
  Sec 4.4  Isolation Forest anomaly-score distribution             -> E6
  Sec 5    Decision threshold tuned on the validation set          -> E1
  Sec 5    Ablation (b): local vs local + aggregated features      -> E3
  Sec 5    Ablation (c): random vs temporal split (classical ML)   -> E4
  Sec 4.1/5 Labeled-subgraph visualisation and case study,
           prediction on unlabeled nodes and structural analysis   -> E5

Only classical models are re-trained here (the GNN results come from
gnn_model.py / tune_gnn.py). All settings (scaling, 70/15/15 stratified
split with seed 42, class weights, model hyper-parameters) are identical
to baselines.py so the numbers are directly comparable.

Usage:
    python proposal_experiments.py

Outputs (project root):
    proposal_experiments_results.json
    proposal_experiments_summary.md
    10_isolation_forest_scores.png
    11_case_study_subgraph.png
    12_ablation_results.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score

from data_loader import load_all
from preprocessing import preprocess_pipeline, get_feature_columns, compute_class_weights
from baselines import get_baseline_models, compute_metrics

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_LOCAL = 93  # 94 local features in the paper incl. time step; time step is excluded


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fit_model(name, model, X_tr, y_tr, X_va, y_va):
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == "LightGBM":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    else:
        model.fit(X_tr, y_tr)
    return model


def best_f1_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i])


def evaluate(y_true, y_prob, thr=0.5):
    m = compute_metrics(y_true, (y_prob >= thr).astype(int), y_prob)
    m["threshold"] = round(float(thr), 4)
    return m


def run_suite(X, labels, idx, class_weights, tag, smote=False, tune=False):
    """Train the four classical models on a given feature matrix / split."""
    tr, va, te = idx
    X_tr, y_tr = X[tr], labels[tr]
    if smote:
        from imblearn.over_sampling import SMOTE
        X_tr, y_tr = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
        class_weights = {0: 1.0, 1: 1.0}  # SMOTE replaces class weighting
    models = get_baseline_models(class_weights, random_state=SEED)
    out, trained = {}, {}
    for name, model in models.items():
        t0 = time.time()
        fit_model(name, model, X_tr, y_tr, X[va], labels[va])
        p_te = model.predict_proba(X[te])[:, 1]
        res = {"default_0.5": evaluate(labels[te], p_te, 0.5)}
        if tune:
            thr = best_f1_threshold(labels[va], model.predict_proba(X[va])[:, 1])
            res["tuned_on_val"] = evaluate(labels[te], p_te, thr)
        res["train_seconds"] = round(time.time() - t0, 1)
        out[name] = res
        trained[name] = model
        d = res["default_0.5"]
        print(f"    [{tag}] {name:<20} PR-AUC={d['pr_auc']:.4f} F1={d['f1_score']:.4f}"
              + (f" | tuned thr={res['tuned_on_val']['threshold']:.3f} "
                 f"F1={res['tuned_on_val']['f1_score']:.4f}" if tune else ""))
    return out, trained


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 70)
    print("Supplementary experiments for the submitted proposal (Team 3)")
    print("=" * 70)

    _, edgelist, _, merged = load_all(BASE_DIR)
    feature_cols = get_feature_columns(merged)
    time_step = merged["time_step"].values
    results = {"n_features": len(feature_cols), "n_local_features": N_LOCAL,
               "n_aggregated_features": len(feature_cols) - N_LOCAL}

    rnd = preprocess_pipeline(merged, edgelist, temporal=False)
    X, labels = rnd["X"], rnd["labels"]
    edge_index = rnd["edge_index"]
    idx_rnd = (rnd["train_idx"], rnd["val_idx"], rnd["test_idx"])

    # E0: cross-time-step edges ------------------------------------------------
    src_t, dst_t = time_step[edge_index[0]], time_step[edge_index[1]]
    n_cross = int((src_t != dst_t).sum())
    results["E0_cross_time_step_edges"] = {
        "n_edges": int(edge_index.shape[1]), "n_cross_time_step_edges": n_cross,
        "note": "0 means every edge stays inside one time step, so the temporal split "
                "cannot leak future information through edges." if n_cross == 0 else
                "Some edges cross time steps."}
    print(f"\n[E0] Edges crossing time-step boundaries: {n_cross}")

    # E1 + main reference: random split, all 165 features, class weights -------
    print("\n[E1] Random split, 165 features, class weights, threshold tuned on validation")
    ref, ref_models = run_suite(X, labels, idx_rnd, rnd["class_weights"], "all-165", tune=True)
    results["E1_threshold_tuning"] = ref

    # E2: SMOTE ----------------------------------------------------------------
    print("\n[E2] SMOTE oversampling on the training partition (no class weights)")
    try:
        results["E2_smote"], _ = run_suite(X, labels, idx_rnd, rnd["class_weights"],
                                           "SMOTE", smote=True)
    except ImportError:
        results["E2_smote"] = {"error": "imbalanced-learn not installed (pip install imbalanced-learn)"}
        print("    imbalanced-learn not installed - skipped")

    # E3: local vs local + aggregated features ---------------------------------
    print(f"\n[E3] Local features only ({N_LOCAL}) vs all ({len(feature_cols)})")
    results["E3_local_only"], _ = run_suite(X[:, :N_LOCAL], labels, idx_rnd,
                                            rnd["class_weights"], f"local-{N_LOCAL}")

    # E4: temporal split ---------------------------------------------------------
    print("\n[E4] Temporal split (first 70% of time steps train / 15% val / 15% test)")
    tmp = preprocess_pipeline(merged, edgelist, temporal=True)
    idx_tmp = (tmp["train_idx"], tmp["val_idx"], tmp["test_idx"])
    results["E4_temporal_split"], _ = run_suite(tmp["X"], tmp["labels"], idx_tmp,
                                                tmp["class_weights"], "temporal", tune=True)
    results["E4_temporal_split"]["time_steps"] = {
        "train": [int(time_step[idx_tmp[0]].min()), int(time_step[idx_tmp[0]].max())],
        "val": [int(time_step[idx_tmp[1]].min()), int(time_step[idx_tmp[1]].max())],
        "test": [int(time_step[idx_tmp[2]].min()), int(time_step[idx_tmp[2]].max())]}
    gnn_tmp = os.path.join(BASE_DIR, "tuning_results.json")
    if os.path.exists(gnn_tmp):
        with open(gnn_tmp) as f:
            tr = json.load(f)
        if "GraphSAGE_temporal" in tr:
            results["E4_temporal_split"]["GraphSAGE (from tune_gnn.py)"] = {
                "default_0.5": tr["GraphSAGE_temporal"]["test_metrics"]}
        if "GraphSAGE_base" in tr:
            results["E4_temporal_split"]["GraphSAGE (random, from tune_gnn.py)"] = {
                "default_0.5": tr["GraphSAGE_base"]["test_metrics"]}

    # E6: Isolation Forest score distribution -----------------------------------
    print("\n[E6] Isolation Forest anomaly-score distribution")
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED, n_jobs=-1)
    iso.fit(X)
    score = -iso.decision_function(X)  # higher = more anomalous
    groups = {"illicit": labels == 1, "licit": labels == 0, "unknown": labels == -1}
    lab = labels != -1
    results["E6_isolation_forest_scores"] = {
        **{f"mean_score_{g}": round(float(score[m].mean()), 4) for g, m in groups.items()},
        **{f"median_score_{g}": round(float(np.median(score[m])), 4) for g, m in groups.items()},
        "roc_auc_illicit_vs_licit": round(float(roc_auc_score(labels[lab] == 1, score[lab])), 4)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.9), 80)
    for g, c in [("unknown", "#9E9E9E"), ("licit", "#4CAF50"), ("illicit", "#F44336")]:
        ax.hist(score[groups[g]], bins=bins, density=True, alpha=0.5, color=c,
                label=f"{g} (n={groups[g].sum():,})")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Isolation Forest anomaly scores by class "
                 f"(ROC-AUC illicit vs licit = {results['E6_isolation_forest_scores']['roc_auc_illicit_vs_licit']:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "10_isolation_forest_scores.png"), dpi=150)
    plt.close()

    # E5: transductive prediction on unlabeled nodes + case study ----------------
    print("\n[E5] Prediction on unlabeled nodes and illicit-subgraph case study")
    best = "LightGBM" if "LightGBM" in ref_models else "Random Forest"
    thr = ref[best]["tuned_on_val"]["threshold"]
    prob_all = ref_models[best].predict_proba(X)[:, 1]
    unk = labels == -1
    pred_ill = (prob_all >= thr) & unk
    status = np.where(labels == 1, 1, np.where(labels == 0, 0, np.where(pred_ill, 2, -1)))
    # 1 illicit, 0 licit, 2 unknown-predicted-illicit, -1 unknown-predicted-licit

    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    G.add_edges_from(edge_index.T.tolist())
    s, d = edge_index
    # label homophily among labeled edges
    both = (labels[s] != -1) & (labels[d] != -1)
    ill_edge = both & ((labels[s] == 1) | (labels[d] == 1))
    ill_ill = both & (labels[s] == 1) & (labels[d] == 1)
    # neighbourhood of predicted-illicit unlabeled nodes
    flag = (status == 1) | (status == 2)
    pi_nodes = np.where(pred_ill)[0]
    nb_flag = [np.mean([flag[n] for n in G.neighbors(v)]) for v in pi_nodes if G.degree(v) > 0]
    other_unk = np.where(unk & ~pred_ill)[0]
    rng = np.random.default_rng(SEED)
    samp = rng.choice(other_unk, size=min(20000, len(other_unk)), replace=False)
    nb_flag_other = [np.mean([flag[n] for n in G.neighbors(v)]) for v in samp if G.degree(v) > 0]
    sub = G.subgraph(np.where(flag)[0])
    comps = sorted((len(c) for c in nx.connected_components(sub)), reverse=True)
    results["E5_unlabeled_prediction_and_case_study"] = {
        "model": best, "threshold": thr,
        "n_unlabeled": int(unk.sum()),
        "n_unlabeled_predicted_illicit": int(pred_ill.sum()),
        "share_unlabeled_predicted_illicit": round(float(pred_ill.mean() / unk.mean()), 4),
        "labeled_edges_touching_illicit": int(ill_edge.sum()),
        "share_of_those_that_are_illicit_to_illicit": round(float(ill_ill.sum() / max(ill_edge.sum(), 1)), 4),
        "illicit_share_among_labeled_nodes": round(float((labels == 1).sum() / lab.sum()), 4),
        "mean_flagged_neighbour_share_predicted_illicit_unlabeled": round(float(np.mean(nb_flag)), 4),
        "mean_flagged_neighbour_share_other_unlabeled": round(float(np.mean(nb_flag_other)), 4),
        "flagged_subgraph_n_components": len(comps),
        "flagged_subgraph_largest_components": comps[:10],
        "flagged_subgraph_share_nodes_in_components_ge3": round(float(sum(c for c in comps if c >= 3) / max(sum(comps), 1)), 4),
    }

    # pick the largest connected cluster of known-illicit transactions for the case study
    ill_sub = G.subgraph(np.where(labels == 1)[0])
    core = max(nx.connected_components(ill_sub), key=len)
    nodes = set(core)
    for v in list(core):
        nodes.update(G.neighbors(v))
    if len(nodes) > 400:  # keep the figure readable
        nodes = set(core) | set(list(nodes - set(core))[:400 - len(core)])
    H = G.subgraph(nodes)
    ts_case = int(pd.Series(time_step[list(core)]).mode()[0])
    deg_in_core = dict(ill_sub.subgraph(core).degree())
    results["E5_unlabeled_prediction_and_case_study"]["case_study"] = {
        "time_step": ts_case, "n_illicit_in_cluster": len(core),
        "n_nodes_shown": H.number_of_nodes(), "n_edges_shown": H.number_of_edges(),
        "max_degree_within_cluster": int(max(deg_in_core.values())),
        "composition_shown": {k: int(sum(status[list(H.nodes)] == v)) for k, v in
                              [("illicit", 1), ("licit", 0), ("unknown_pred_illicit", 2),
                               ("unknown_pred_licit", -1)]}}
    cmap = {1: "#F44336", 0: "#4CAF50", 2: "#FF9800", -1: "#BDBDBD"}
    names = {1: "illicit (label)", 0: "licit (label)", 2: f"unknown, predicted illicit ({best})",
             -1: "unknown, predicted licit"}
    pos = nx.spring_layout(H, seed=SEED, k=0.25)
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3, width=0.6)
    for k in [-1, 0, 2, 1]:
        nl = [n for n in H.nodes if status[n] == k]
        if nl:
            nx.draw_networkx_nodes(H, pos, nodelist=nl, node_color=cmap[k], ax=ax,
                                   node_size=[40 if k in (1, 2) else 18 for _ in nl],
                                   label=f"{names[k]} ({len(nl)})")
    ax.set_title(f"Case study: largest connected cluster of illicit transactions\n"
                 f"(time step {ts_case}, {len(core)} illicit nodes) and their 1-hop neighbours")
    ax.legend(loc="lower left", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "11_case_study_subgraph.png"), dpi=150)
    plt.close()

    # Ablation figure -----------------------------------------------------------
    models = list(ref.keys())
    def get(block, m, key="default_0.5", met="pr_auc"):
        try:
            return results[block][m][key][met]
        except (KeyError, TypeError):
            return np.nan
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(models)); w = 0.38
    panels = [
        ("(a) Feature ablation: PR-AUC", "pr_auc",
         [("local only (93)", [get("E3_local_only", m) for m in models]),
          ("local + aggregated (165)", [get("E1_threshold_tuning", m) for m in models])]),
        ("(b) Split ablation: PR-AUC", "pr_auc",
         [("temporal split", [get("E4_temporal_split", m) for m in models]),
          ("random split", [get("E1_threshold_tuning", m) for m in models])]),
        ("(c) Imbalance handling & threshold: F1", "f1_score",
         [("SMOTE, thr 0.5", [get("E2_smote", m, met="f1_score") for m in models]),
          ("class weights, thr 0.5", [get("E1_threshold_tuning", m, met="f1_score") for m in models])]),
    ]
    for ax, (title, met, series) in zip(axes, panels):
        for j, (lbl, vals) in enumerate(series):
            ax.bar(x + (j - 0.5) * w, vals, w, label=lbl)
        if met == "f1_score":
            ax.scatter(x, [get("E1_threshold_tuning", m, "tuned_on_val", "f1_score") for m in models],
                       color="k", marker="D", zorder=3, label="class weights, tuned thr")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(title); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "12_ablation_results.png"), dpi=150)
    plt.close()

    results["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(BASE_DIR, "proposal_experiments_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    write_summary(results, models)
    print(f"\nDone in {results['runtime_seconds']} s. Results saved to proposal_experiments_results.json")


def write_summary(r, models):
    def row(m, blk, key="default_0.5"):
        try:
            v = r[blk][m][key]
            return f"{v['pr_auc']:.4f} | {v['f1_score']:.4f} | {v['recall']:.4f} | {v['precision']:.4f}"
        except (KeyError, TypeError):
            return "n/a | n/a | n/a | n/a"
    L = ["# Supplementary Experiments (proposal follow-up)", "",
         "Generated by `proposal_experiments.py`. Test-set metrics. \"Ref\" = reference setting (random split, 165 features, class weights, threshold 0.5).", ""]
    L += ["## E1 – Decision threshold tuned on the validation set (random split, 165 features)", "",
          "| Model | Threshold | PR-AUC | F1 | Recall | Precision | F1 @ 0.5 |", "|---|---|---|---|---|---|---|"]
    for m in models:
        t = r["E1_threshold_tuning"][m]["tuned_on_val"]; d = r["E1_threshold_tuning"][m]["default_0.5"]
        L.append(f"| {m} | {t['threshold']:.3f} | {t['pr_auc']:.4f} | {t['f1_score']:.4f} | "
                 f"{t['recall']:.4f} | {t['precision']:.4f} | {d['f1_score']:.4f} |")
    for title, blk in [("E2 – SMOTE oversampling (instead of class weights)", "E2_smote"),
                       ("E3 – Ablation (b): local features only (93)", "E3_local_only"),
                       ("E4 – Ablation (c): temporal split", "E4_temporal_split")]:
        L += ["", f"## {title}", "", "| Model | PR-AUC | F1 | Recall | Precision | Ref PR-AUC | Ref F1 | Ref Recall | Ref Precision |",
              "|---|---|---|---|---|---|---|---|---|"]
        ms = models + (["GraphSAGE (from tune_gnn.py)"] if blk == "E4_temporal_split" else [])
        for m in ms:
            L.append(f"| {m} | {row(m, blk)} | {row(m, 'E1_threshold_tuning') if m in models else row('GraphSAGE (random, from tune_gnn.py)', blk)} |")
    if "time_steps" in r["E4_temporal_split"]:
        ts = r["E4_temporal_split"]["time_steps"]
        L.append(f"\nTime steps – train {ts['train']}, validation {ts['val']}, test {ts['test']}.")
    e0 = r["E0_cross_time_step_edges"]
    L += ["", "## E0 – Edges crossing time-step boundaries", "",
          f"{e0['n_cross_time_step_edges']} of {e0['n_edges']:,} edges. {e0['note']}"]
    e5 = r["E5_unlabeled_prediction_and_case_study"]
    L += ["", "## E5 – Prediction on unlabeled nodes and case study", "",
          f"- Model: {e5['model']} (threshold {e5['threshold']:.3f} tuned on validation)",
          f"- Unlabeled nodes predicted illicit: {e5['n_unlabeled_predicted_illicit']:,} of {e5['n_unlabeled']:,} "
          f"({e5['share_unlabeled_predicted_illicit']:.2%})",
          f"- Labeled edges touching an illicit node that connect two illicit nodes: "
          f"{e5['share_of_those_that_are_illicit_to_illicit']:.2%} (illicit share among labeled nodes: "
          f"{e5['illicit_share_among_labeled_nodes']:.2%})",
          f"- Mean share of flagged (illicit or predicted-illicit) neighbours: predicted-illicit unlabeled nodes "
          f"{e5['mean_flagged_neighbour_share_predicted_illicit_unlabeled']:.2%} vs other unlabeled nodes "
          f"{e5['mean_flagged_neighbour_share_other_unlabeled']:.2%}",
          f"- Flagged subgraph: {e5['flagged_subgraph_n_components']:,} connected components, largest sizes "
          f"{e5['flagged_subgraph_largest_components']}",
          f"- Case study (Figure 11): time step {e5['case_study']['time_step']}, "
          f"{e5['case_study']['n_illicit_in_cluster']} connected illicit transactions; composition shown "
          f"{e5['case_study']['composition_shown']}"]
    e6 = r["E6_isolation_forest_scores"]
    L += ["", "## E6 – Isolation Forest anomaly scores (Figure 10)", "",
          f"- Mean score: illicit {e6['mean_score_illicit']}, licit {e6['mean_score_licit']}, unknown {e6['mean_score_unknown']}",
          f"- ROC-AUC (illicit vs licit): {e6['roc_auc_illicit_vs_licit']}"]
    with open(os.path.join(BASE_DIR, "proposal_experiments_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
