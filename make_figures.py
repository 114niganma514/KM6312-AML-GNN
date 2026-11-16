# build figures 07, 09, 12 and results/summary.md from results/*.json
# usage: python make_figures.py

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE_DIR, "results")
CLASSICAL = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


def load(name):
    path = os.path.join(RES, name)
    return json.load(open(path)) if os.path.exists(path) else None


def ms(d, block="test", k="pr_auc"):
    v = d[block][k]
    return v["mean"], v["std"]


def fmt(d, block="test", k="pr_auc"):
    m, s = ms(d, block, k)
    return f"{m:.4f} ± {s:.4f}"


def table(rows, models, cols):
    head = "| Model | " + " | ".join(c[0] for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for m in models:
        if m in rows:
            lines.append(f"| {m} | " + " | ".join(fmt(rows[m], b, k) for _, b, k in cols) + " |")
    return lines


def main():
    cs = load("classical_summary.json")
    gs = load("gnn_summary.json")
    cr = load("classical_runs.json")
    gr = load("gnn_runs.json")
    std_cols = [("PR-AUC", "test", "pr_auc"), ("ROC-AUC", "test", "roc_auc"),
                ("F1", "test", "f1_score"), ("Recall", "test", "recall"),
                ("Precision", "test", "precision")]
    md = ["# Results summary", "", "Mean ± std over seeds. Test set unless stated.", ""]

    if cs:
        n = cs["random"][CLASSICAL[-1]]["n_seeds"] if CLASSICAL[-1] in cs["random"] else ""
        md += [f"## Classical models, random split ({n} seeds)", ""]
        md += table(cs["random"], CLASSICAL, std_cols + [("Val PR-AUC", "val", "pr_auc")])
        md += ["", "## Classical models, F1 with threshold tuned on validation", ""]
        md += table(cs["random"], CLASSICAL, [("F1 @0.5", "test", "f1_score"),
                                               ("F1 @val thr", "test_tuned", "f1_score")])
        for key, title in [("temporal", "Temporal split (train 1-34, val 35-41, test 42-49)"),
                           ("weber_split", "Weber et al. split (train 1-34, test 35-49)"),
                           ("local_only", "Local features only (93)"),
                           ("smote", "SMOTE instead of class weights")]:
            md += ["", f"## {title}", ""] + table(cs[key], CLASSICAL, std_cols)

    if gs:
        sel = gs.get("selected_by_validation")
        md += ["", f"## GNNs, random split (selected on validation PR-AUC: {sel})", ""]
        md += table(gs["random"], list(gs["random"]), [("Val PR-AUC", "val", "pr_auc")] + std_cols)
        md += ["", "## GNNs, temporal split", ""]
        md += table(gs["temporal"], list(gs["temporal"]), std_cols)
        md += ["", "## Unlabeled nodes predicted illicit (seed 0, label == -1 only)", "",
               "| Model | at 0.5 | at val threshold | of |", "|---|---|---|---|"]
        for m, v in gs["random"].items():
            u = v.get("unlabeled_pred_illicit")
            if u:
                md.append(f"| {m} | {u['at_0.5']:,} | {u['at_val_threshold']:,} | {u['n_unlabeled']:,} |")
        if cr:
            u = cr["random"]["LightGBM"][0].get("unlabeled_pred_illicit")
            if u:
                md.append(f"| LightGBM | {u['at_0.5']:,} | {u['at_val_threshold']:,} | {u['n_unlabeled']:,} |")

    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # figure 07: test PR-AUC of all supervised models
    if cs and gs:
        names = [m for m in CLASSICAL if m in cs["random"]] + list(gs["random"])
        vals = [ms(cs["random"][m]) if m in cs["random"] else ms(gs["random"][m]) for m in names]
        colors = ["#4C72B0"] * len([m for m in CLASSICAL if m in cs["random"]]) + ["#DD8452"] * len(gs["random"])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        y = np.arange(len(names))
        ax.barh(y, [v[0] for v in vals], xerr=[v[1] for v in vals], color=colors, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Test PR-AUC (mean ± std over seeds)")
        ax.set_title("Supervised models, random split (blue: classical, orange: GNN)", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "07_model_comparison.png"), dpi=200)
        plt.close()

    # figure 09: confusion matrices of seed 0
    if cr and gr:
        sel = gs["selected_by_validation"]
        panels = [(m, cr["random"][m][0]["test"]["confusion_matrix"]) for m in CLASSICAL if m in cr["random"]]
        panels += [(m, gr["random"][m][0]["test"]["confusion_matrix"])
                   for m in ["GCN (2L, h=64)", sel] if m in gr["random"]]
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
        for ax, (m, cm) in zip(axes.ravel(), panels):
            cm = np.array(cm)
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["licit", "illicit"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["licit", "illicit"], fontsize=8)
            ax.set_title(m, fontsize=8)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.supxlabel("Predicted", fontsize=9)
        fig.supylabel("True", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "09_confusion_matrices.png"), dpi=200)
        plt.close()

    # figure 12: ablations on classical models
    if cs:
        models = [m for m in CLASSICAL if m in cs["random"]]
        x = np.arange(len(models))
        w = 0.38
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        panels = [
            ("(a) Features: test PR-AUC", "pr_auc",
             [("local only (93)", cs["local_only"]), ("local + aggregated (165)", cs["random"])]),
            ("(b) Split: test PR-AUC", "pr_auc",
             [("temporal (test 42-49)", cs["temporal"]), ("random", cs["random"])]),
            ("(c) Imbalance: test F1", "f1_score",
             [("SMOTE, thr 0.5", cs["smote"]), ("class weights, thr 0.5", cs["random"])]),
        ]
        for ax, (title, k, series) in zip(axes, panels):
            for j, (lbl, block) in enumerate(series):
                vals = [ms(block[m], "test", k) for m in models]
                ax.bar(x + (j - 0.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
                       capsize=2, label=lbl)
            if k == "f1_score":
                vals = [ms(cs["random"][m], "test_tuned", k)[0] for m in models]
                ax.scatter(x, vals, color="k", marker="D", zorder=3, label="class weights, val thr")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "12_ablation_results.png"), dpi=200)
        plt.close()
    print("figures and results/summary.md written")


if __name__ == "__main__":
    main()
