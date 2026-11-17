# checks on the GNN part with the real data
# 1 splits and scaler   2 test labels are never used   3 rerun matches gnn_runs.json
# 4 PR-AUC cross-check  5 GraphSAGE without edges / with undirected edges
# usage: python verify_gnn.py   (about 15 min on CPU)

import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from data_loader import load_all
from preprocessing import preprocess_pipeline, build_pyg_data
from gnn_model import get_model, train_gnn, set_seed
from run_experiments import scores, f1_threshold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE_DIR, "results")
out = {}


def run(pre, model, hidden, seed, y=None, ei=None):
    y = pre["labels"] if y is None else y
    ei = pre["edge_index"] if ei is None else ei
    data = build_pyg_data(pre["X"], ei, y, pre["train_idx"], pre["val_idx"], pre["test_idx"])
    set_seed(seed)
    m = get_model(model, data.x.shape[1], hidden, 2, 0.3, 4)
    t0 = time.time()
    tr = train_gnn(m, data, pre["class_weights"], epochs=200, verbose=False)
    m = tr["model"].eval()
    with torch.no_grad():
        p = F.softmax(m(data.x, data.edge_index), dim=1)[:, 1].numpy()
    return p, tr["best_epoch"], round(time.time() - t0, 1)


def check(name, ok, info=""):
    out.setdefault("checks", {})[name] = {"pass": bool(ok), "info": info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")


_, edges, _, merged = load_all(BASE_DIR)
ts = merged["time_step"].values
rnd = preprocess_pipeline(merged, edges, temporal=False)
tmp = preprocess_pipeline(merged, edges, temporal=True)
y = rnd["labels"]
runs = json.load(open(os.path.join(RES, "gnn_runs.json")))

# 1 splits
for nm, pre in [("random", rnd), ("temporal", tmp)]:
    tr, va, te = pre["train_idx"], pre["val_idx"], pre["test_idx"]
    disjoint = not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    labelled = (y[tr] != -1).all() and (y[va] != -1).all() and (y[te] != -1).all()
    check(f"{nm}: splits disjoint and labelled", disjoint and labelled,
          f"{len(tr)}/{len(va)}/{len(te)}")
check("temporal: time steps 1-34 / 35-41 / 42-49",
      ts[tmp["train_idx"]].max() <= 34 and ts[tmp["val_idx"]].min() >= 35
      and ts[tmp["val_idx"]].max() <= 41 and ts[tmp["test_idx"]].min() >= 42)
past = tmp["X"][ts <= 34]
check("temporal: scaler fitted on steps 1-34 only",
      np.abs(past.mean(0)).max() < 1e-3 and np.abs(tmp["X"][ts > 34].mean(0)).max() > 1e-2,
      f"max |mean| past {np.abs(past.mean(0)).max():.1e}")

# 2 test labels hidden -> same predictions
p0, ep0, _ = run(rnd, "GraphSAGE", 64, 0)
y_hidden = y.copy()
y_hidden[rnd["test_idx"]] = -1
p1, ep1, _ = run(rnd, "GraphSAGE", 64, 0, y=y_hidden)
diff = float(np.abs(p0 - p1).max())
check("test labels never used (hide them, retrain)", diff < 1e-5 and ep0 == ep1, f"max prob diff {diff:.1e}")

# 3 rerun matches the saved runs
va, te = rnd["val_idx"], rnd["test_idx"]
for setting, pre, model, hidden, key in [
        ("random", rnd, "GraphSAGE", 64, "GraphSAGE (2L, h=64)"),
        ("random", rnd, "GCN", 64, "GCN (2L, h=64)"),
        ("temporal", tmp, "GraphSAGE", 64, "GraphSAGE (2L, h=64)")]:
    p, _, _ = (p0, ep0, 0) if (setting, model) == ("random", "GraphSAGE") else run(pre, model, hidden, 0)
    got = scores(y[pre["test_idx"]], p[pre["test_idx"]])["pr_auc"]
    saved = runs[setting][key][0]["test"]["pr_auc"]
    check(f"rerun {setting} {key} seed 0", abs(got - saved) < 2e-3, f"now {got:.4f}, saved {saved:.4f}")

# 4 PR-AUC (trapezoid) vs average precision
ap = average_precision_score(y[te], p0[te])
pr = scores(y[te], p0[te])["pr_auc"]
check("PR-AUC close to average precision", abs(ap - pr) < 0.01, f"trapezoid {pr:.4f}, AP {ap:.4f}")

# 5 graph ablation for GraphSAGE (3 seeds)
no_edges = np.zeros((2, 0), dtype=np.int64)
undirected = np.concatenate([rnd["edge_index"], rnd["edge_index"][[1, 0]]], axis=1)
abl = {}
for name, ei in [("directed (main)", None), ("no edges (MLP)", no_edges), ("undirected", undirected)]:
    res = []
    for s in range(3):
        p, ep, sec = (p0, ep0, 0) if (name == "directed (main)" and s == 0) else run(rnd, "GraphSAGE", 64, s, ei=ei)
        thr = f1_threshold(y[va], p[va])
        res.append({"val_pr_auc": scores(y[va], p[va])["pr_auc"], "test_pr_auc": scores(y[te], p[te])["pr_auc"],
                    "test_f1_val_thr": scores(y[te], p[te], thr)["f1_score"], "best_epoch": ep})
        print(f"  {name} seed {s}: test PR-AUC {res[-1]['test_pr_auc']}")
    abl[name] = {k: {"mean": round(float(np.mean([r[k] for r in res])), 4),
                     "std": round(float(np.std([r[k] for r in res])), 4)}
                 for k in ["val_pr_auc", "test_pr_auc", "test_f1_val_thr"]}
    abl[name]["runs"] = res
out["graphsage_edge_ablation"] = abl

with open(os.path.join(RES, "gnn_verification.json"), "w") as f:
    json.dump(out, f, indent=2)
n_fail = sum(not c["pass"] for c in out["checks"].values())
print(f"\n{len(out['checks']) - n_fail} passed, {n_fail} failed")
for k, v in abl.items():
    print(f"{k:16s} test PR-AUC {v['test_pr_auc']['mean']:.4f} ± {v['test_pr_auc']['std']:.4f}")
