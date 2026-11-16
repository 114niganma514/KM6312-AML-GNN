# run all supervised experiments with fixed seeds
# usage: python run_experiments.py [--classical] [--gnn] [--clustering] [--seeds 5]

import os
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from data_loader import load_all
from preprocessing import preprocess_pipeline, build_pyg_data, compute_class_weights
from baselines import get_baseline_models, compute_metrics

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
N_LOCAL = 93  # local features without time_step

GNN_CONFIGS = {
    "GCN (2L, h=64)":            dict(model="GCN", hidden=64),
    "GraphSAGE (2L, h=64)":      dict(model="GraphSAGE", hidden=64),
    "GraphSAGE (2L, h=128)":     dict(model="GraphSAGE", hidden=128),
    "GraphSAGE (3L, h=128)":     dict(model="GraphSAGE3", hidden=128),
    "GAT (2L, h=64, 4 heads)":   dict(model="GAT", hidden=64),
    "GraphSAGE (2L, h=64) + LR sched": dict(model="GraphSAGE", hidden=64, sched=True, epochs=300),
}
TEMPORAL_GNN = ["GCN (2L, h=64)", "GraphSAGE (2L, h=64)", "GraphSAGE (3L, h=128)"]


def f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.nanargmax(f1))])


def scores(y, p, thr=0.5):
    m = compute_metrics(y, (p >= thr).astype(int), p)
    m["threshold"] = round(thr, 4)
    return m


def fit(name, model, Xtr, ytr, Xva, yva):
    if name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "LightGBM":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    else:
        model.fit(Xtr, ytr)
    return model


def classical_runs(X, y, tr, va, te, cw, seed, smote=False):
    Xtr, ytr = X[tr], y[tr]
    if smote:
        from imblearn.over_sampling import SMOTE
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
        cw = {0: 1.0, 1: 1.0}
    out = {}
    for name, model in get_baseline_models(cw, random_state=seed).items():
        t0 = time.time()
        fit(name, model, Xtr, ytr, X[va], y[va])
        pva = model.predict_proba(X[va])[:, 1]
        pte = model.predict_proba(X[te])[:, 1]
        thr = f1_threshold(y[va], pva)
        out[name] = {
            "val": scores(y[va], pva),
            "test": scores(y[te], pte),
            "test_tuned": scores(y[te], pte, thr),
            "seconds": round(time.time() - t0, 1),
        }
    return out


def mean_std(runs, block="test"):
    keys = ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]
    arr = {k: [r[block][k] for r in runs] for k in keys}
    return {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
            for k, v in arr.items()}


def summarise(raw):
    # raw[setting][model] = list of per-seed dicts
    summ = {}
    for setting, models in raw.items():
        summ[setting] = {}
        for m, runs in models.items():
            summ[setting][m] = {
                "n_seeds": len(runs),
                "val": mean_std(runs, "val"),
                "test": mean_std(runs, "test"),
            }
            if "test_tuned" in runs[0]:
                summ[setting][m]["test_tuned"] = mean_std(runs, "test_tuned")
            if "unlabeled_pred_illicit" in runs[0]:
                summ[setting][m]["unlabeled_pred_illicit"] = runs[0]["unlabeled_pred_illicit"]
    return summ


def run_classical(merged, edges, seeds):
    raw = {}
    rnd = preprocess_pipeline(merged, edges, temporal=False)
    tmp = preprocess_pipeline(merged, edges, temporal=True)
    ts = merged["time_step"].values
    y = rnd["labels"]

    # weber et al. (2019) protocol: train 1-34, test 35-49, scaler on 1-34
    lab = y != -1
    w_tr = np.where(lab & (ts <= 34))[0]
    w_te = np.where(lab & (ts >= 35))[0]
    rng = np.random.default_rng(0)
    w_va = rng.choice(w_tr, size=len(w_tr) // 10, replace=False)  # only for early stopping
    w_tr2 = np.setdiff1d(w_tr, w_va)
    Xw = tmp["X"]  # same scaler fit (time steps 1-34)

    settings = {
        "random": (rnd["X"], rnd["train_idx"], rnd["val_idx"], rnd["test_idx"], rnd["class_weights"], False),
        "temporal": (tmp["X"], tmp["train_idx"], tmp["val_idx"], tmp["test_idx"], tmp["class_weights"], False),
        "weber_split": (Xw, w_tr2, w_va, w_te, compute_class_weights(y[w_tr2]), False),
        "local_only": (rnd["X"][:, :N_LOCAL], rnd["train_idx"], rnd["val_idx"], rnd["test_idx"], rnd["class_weights"], False),
        "smote": (rnd["X"], rnd["train_idx"], rnd["val_idx"], rnd["test_idx"], rnd["class_weights"], True),
    }
    for name, (X, tr, va, te, cw, sm) in settings.items():
        raw[name] = {}
        for s in seeds:
            print(f"[classical] {name} seed={s}")
            res = classical_runs(X, y, tr, va, te, cw, s, smote=sm)
            for m, r in res.items():
                raw[name].setdefault(m, []).append(r)

    # unlabeled predictions of LightGBM (seed 0, random split)
    model = get_baseline_models(rnd["class_weights"], random_state=seeds[0])["LightGBM"]
    fit("LightGBM", model, rnd["X"][rnd["train_idx"]], y[rnd["train_idx"]],
        rnd["X"][rnd["val_idx"]], y[rnd["val_idx"]])
    thr = f1_threshold(y[rnd["val_idx"]], model.predict_proba(rnd["X"][rnd["val_idx"]])[:, 1])
    p_unl = model.predict_proba(rnd["X"][y == -1])[:, 1]
    raw["random"]["LightGBM"][0]["unlabeled_pred_illicit"] = {
        "n_unlabeled": int((y == -1).sum()),
        "at_0.5": int((p_unl >= 0.5).sum()),
        "at_val_threshold": int((p_unl >= thr).sum()),
        "val_threshold": round(thr, 4),
    }
    return raw


def run_gnn(merged, edges, seeds, gat_seeds):
    import torch
    import torch.nn.functional as F
    from gnn_model import get_model, train_gnn, evaluate_gnn, set_seed

    raw = {"random": {}, "temporal": {}}
    for setting in ["random", "temporal"]:
        pre = preprocess_pipeline(merged, edges, temporal=(setting == "temporal"))
        y = pre["labels"]
        data = build_pyg_data(pre["X"], pre["edge_index"], y,
                              pre["train_idx"], pre["val_idx"], pre["test_idx"])
        names = list(GNN_CONFIGS) if setting == "random" else TEMPORAL_GNN
        for name in names:
            cfg = GNN_CONFIGS[name]
            use_seeds = gat_seeds if cfg["model"] == "GAT" else seeds
            for s in use_seeds:
                print(f"[gnn] {setting} {name} seed={s}")
                set_seed(s)
                model = get_model(cfg["model"], data.x.shape[1], cfg["hidden"], 2, 0.3, 4)
                t0 = time.time()
                tr = train_gnn(model, data, pre["class_weights"], epochs=cfg.get("epochs", 200),
                               use_lr_scheduler=cfg.get("sched", False), verbose=False)
                secs = round(time.time() - t0, 1)
                m = tr["model"]
                m.eval()
                with torch.no_grad():
                    prob = F.softmax(m(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
                va, te = pre["val_idx"], pre["test_idx"]
                thr = f1_threshold(y[va], prob[va])
                unl = prob[y == -1]
                r = {
                    "val": scores(y[va], prob[va]),
                    "test": scores(y[te], prob[te]),
                    "test_tuned": scores(y[te], prob[te], thr),
                    "best_epoch": tr["best_epoch"],
                    "seconds": secs,
                    "unlabeled_pred_illicit": {
                        "n_unlabeled": int((y == -1).sum()),
                        "at_0.5": int((unl >= 0.5).sum()),
                        "at_val_threshold": int((unl >= thr).sum()),
                        "val_threshold": round(thr, 4),
                    },
                }
                raw[setting].setdefault(name, []).append(r)
                print(f"    val PR-AUC {r['val']['pr_auc']}  test PR-AUC {r['test']['pr_auc']}  ({secs}s)")
                save_json(raw, "gnn_runs.json")  # save as we go
    return raw


def run_clustering_step(merged, edges):
    from clustering import run_clustering_pipeline
    pre = preprocess_pipeline(merged, edges, temporal=False)
    run_clustering_pipeline(pre, save_dir=OUT_DIR, n_clusters=5)


def save_json(obj, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--gnn", action="store_true")
    ap.add_argument("--clustering", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gat-seeds", type=int, default=3)  # GAT is ~10x slower on CPU
    a = ap.parse_args()
    if not (a.classical or a.gnn or a.clustering):
        a.classical = a.gnn = a.clustering = True
    seeds = list(range(a.seeds))

    _, edges, _, merged = load_all(BASE_DIR)
    t0 = time.time()
    if a.classical:
        raw = run_classical(merged, edges, seeds)
        save_json(raw, "classical_runs.json")
        save_json(summarise(raw), "classical_summary.json")
    if a.gnn:
        raw = run_gnn(merged, edges, seeds, list(range(a.gat_seeds)))
        summ = summarise(raw)
        # model selection on validation PR-AUC only
        best = max(summ["random"], key=lambda k: summ["random"][k]["val"]["pr_auc"]["mean"])
        summ["selected_by_validation"] = best
        save_json(summ, "gnn_summary.json")
    if a.clustering:
        run_clustering_step(merged, edges)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
