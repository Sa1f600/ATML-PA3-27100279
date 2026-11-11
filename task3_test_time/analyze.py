"""Create Task 3 evidence plots from already-computed offline CSV outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", default="results/task3/offline")
    parser.add_argument("--output", default="figures/task3")
    args = parser.parse_args()
    root, output = Path(args.offline), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = pd.read_csv(source / "fixed_heldout_summary.csv")
        for metric, ylabel in (("accuracy", "Accuracy"), ("oracle_coverage", "Oracle coverage"), ("token_cost", "Generated-token cost")):
            figure, axis = plt.subplots(figsize=(7, 4))
            for method, group in summary.groupby("method"):
                axis.plot(group.budget, group[metric], marker="o", label=method)
            axis.set(xlabel="Candidate budget N", ylabel=ylabel, xticks=[1, 2, 4, 8, 16, 32], title=f"{source.name}: {ylabel}")
            axis.legend(fontsize=7, ncol=2)
            figure.tight_layout()
            figure.savefig(output / f"{source.name}_{metric}.png", dpi=180)
            plt.close(figure)
    print(f"Saved Task 3 figures to {output}")


if __name__ == "__main__":
    main()
