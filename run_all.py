# full pipeline: EDA -> experiments -> graph analysis -> figures
# usage: python run_all.py [--skip-eda] [--skip-gnn] [--seeds 5]

import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = ["elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv"]


def run(args):
    print("\n>>", " ".join(args))
    subprocess.run([sys.executable] + args, cwd=BASE_DIR, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--skip-gnn", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    missing = [f for f in DATA if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        sys.exit(f"missing data files: {missing}\nrun: python download_data.py (see README)")

    if not a.skip_eda:
        run(["eda.py"])
    exp = ["run_experiments.py", "--classical", "--clustering", "--seeds", str(a.seeds)]
    if not a.skip_gnn:
        exp.append("--gnn")
    run(exp)
    run(["graph_analysis.py"])
    run(["make_figures.py"])


if __name__ == "__main__":
    main()
