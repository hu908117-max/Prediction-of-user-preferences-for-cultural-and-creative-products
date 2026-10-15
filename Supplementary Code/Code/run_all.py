"""End-to-end project runner.

Executes the two deliverables of the project:

  1. The publication result figures (Figs. 2-8), obtained by translating the
     original MATLAB result code into Python / matplotlib. These are the figures
     reported in the manuscript and are written to outputs/figures/.

  2. An executable reproduction of the proposed method on the ArtEmis-derived
     dataset. This trains the Stage 1-4 pipeline and writes the intermediate
     stage artifacts (loss curve, t-SNE, InfoNCE similarity, classification,
     ranking, ROC) plus a metrics.json to outputs/metrics/ and
     outputs/checkpoints/.

Usage:
    python run_all.py                 # figures + demo training (CPU-friendly)
    python run_all.py --epochs 45     # longer demo training
    python run_all.py --sweep         # also run the parameter-sensitivity sweep
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    # 1. Publication figures (MATLAB -> Python).
    run([py, os.path.join(root, "run_figures.py")])

    # 2. Executable method reproduction (Stage 1-4 training + stage artifacts).
    train_cmd = [
        py, "-m", "src.train",
        "--epochs", str(args.epochs),
        "--hidden_dim", str(args.hidden_dim),
        "--batch_size", str(args.batch_size),
    ]
    if args.sweep:
        train_cmd.append("--sweep")
    run(train_cmd)

    print("\nAll deliverables generated.")
    print("  Publication figures : outputs/figures/Fig_*.png")
    print("  Stage artifacts     : outputs/metrics/*.png, outputs/metrics/metrics.json")
    print("  Checkpoints         : outputs/checkpoints/*.pt")


if __name__ == "__main__":
    main()
