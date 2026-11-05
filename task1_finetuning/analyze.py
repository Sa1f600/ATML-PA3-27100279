"""Task 1.6: machine-readable comparisons and figures from saved experiments."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.experiment import read_json
from task1_finetuning.evaluate import TASKS
from task1_finetuning.inference import CONDITIONS, EVAL_CONFIG


RUNS = ["full_ft", "full_ft_l2_sp", "lora_4", "lora_16", "oplora_16"]
PAIRS = [("lora4_final", "lora16_final"), ("full_ft_final", "l2_sp_final"),
         ("lora16_final", "oplora_final"), ("full_ft_final", "lora16_final"),
         ("full_ft_midpoint", "full_ft_final")]


def score_table(aggregates):
    rows = []
    for condition in CONDITIONS:
        for task in TASKS:
            value = aggregates[condition][task]
            baseline = aggregates["baseline"][task]["primary"]
            rows.append({"condition": condition, "benchmark": task, "primary_percent": 100 * value["primary"],
                         "change_from_baseline_pp": 100 * (value["primary"] - baseline),
                         "raw_accuracy_percent": (100 * value["metrics"]["acc,none"]
                                                  if "acc,none" in value["metrics"] else None),
                         "normalized_accuracy_percent": (100 * value["metrics"]["acc_norm,none"]
                                                         if "acc_norm,none" in value["metrics"] else None)})
    return rows


def contrast_table(aggregates):
    return [{"first": first, "second": second, "benchmark": task,
             "second_minus_first_pp": 100 * (aggregates[second][task]["primary"] - aggregates[first][task]["primary"])}
            for first, second in PAIRS for task in TASKS]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, directory, name):
    figure.tight_layout()
    figure.savefig(directory / f"{name}.png", dpi=180)
    figure.savefig(directory / f"{name}.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=EVAL_CONFIG)
    args = parser.parse_args()
    cfg = read_json(args.config)
    destination = Path("results/task1/analysis")
    figures = Path(cfg["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)
    aggregates = {condition: read_json(Path(cfg["results_dir"]) / condition / "aggregate.json") for condition in CONDITIONS}
    table = score_table(aggregates)
    write_csv(destination / "benchmark_scores.csv", table)
    write_csv(destination / "controlled_comparisons.csv", contrast_table(aggregates))
    figure, axis = plt.subplots(figsize=(9, 4))
    resources, penalties = [], []
    for run in RUNS:
        training = read_json(f"configs/task1/{run}.json")
        root = Path(training["results_dir"])
        import json
        rows = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
        successful = [row for row in rows if not row["skipped"]]
        if [row["update"] for row in successful] != list(range(1, 1251)):
            raise ValueError(f"Training curve requires all 1,250 successful updates: {run}")
        axis.plot([row["update"] for row in successful], [row["sft_loss"] for row in successful], label=run, linewidth=0.8)
        resource = read_json(root / "resources.json")
        resources.append({"run": run, "trainable_parameters": resource["trainable_parameters"],
                          "training_seconds": resource["training_seconds"],
                          "peak_gpu_allocated_bytes": resource["peak_gpu_allocated_bytes"],
                          "checkpoint_seconds": resource["checkpoint_seconds"],
                          "basis_preprocessing_seconds": resource.get("basis_preprocessing_seconds", 0),
                          "basis_tensor_bytes": resource.get("basis_tensor_bytes", 0),
                          "reference_cpu_bytes": resource["reference_cpu_bytes"],
                          "reference_transfer_seconds": resource.get("reference_transfer_seconds", 0),
                          "export_seconds": resource.get("export_seconds", 0),
                          "final_model_bytes": read_json(Path(training["checkpoint_dir"]) / "update_1250" / "checkpoint_summary.json")["model_bytes"],
                          "final_recovery_bytes": read_json(Path(training["checkpoint_dir"]) / "update_1250" / "checkpoint_summary.json")["recovery_bytes_excluding_summary"]})
        if run == "full_ft_l2_sp":
            penalties = successful
    axis.set(xlabel="Optimizer update", ylabel="Response-token SFT loss")
    axis.legend(fontsize=8)
    save_figure(figure, figures, "training_sft_loss")
    write_csv(destination / "resources.csv", resources)
    displacement = []
    for run, updates in (("full_ft", [625, 1250]), ("full_ft_l2_sp", [1250])):
        root = Path(read_json(f"configs/task1/{run}.json")["results_dir"])
        displacement.extend({"run": run, **read_json(root / f"displacement_{update:04d}.json")} for update in updates)
    write_csv(destination / "parameter_displacement.csv", displacement)
    figure, axis = plt.subplots(figsize=(8, 3))
    for key in ("sft_loss", "lambda_omega", "total_loss"):
        axis.plot([row["update"] for row in penalties], [row[key] for row in penalties], label=key)
    axis.set(xlabel="Optimizer update", ylabel="L2-SP objective components")
    axis.legend()
    save_figure(figure, figures, "l2_sp_objective")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, task in zip(axes.flat, TASKS):
        values = [aggregates[condition][task]["primary"] * 100 for condition in CONDITIONS]
        axis.bar(range(7), values)
        axis.set_xticks(range(7), CONDITIONS, rotation=40, ha="right", fontsize=7)
        axis.set(title=task, ylabel="Primary score (%)")
    save_figure(figure, figures, "benchmark_scores")
    weight_rows, activation_rows = [], []
    for condition in list(CONDITIONS)[1:]:
        diagnostic = read_json(Path(cfg["diagnostics_dir"]) / f"{condition}.json")
        weight_rows.extend(diagnostic["weights"])
        activation_rows.extend(diagnostic["activations"])
    write_csv(destination / "weight_diagnostics.csv", weight_rows)
    write_csv(destination / "activation_diagnostics.csv", activation_rows)
    summaries = []
    for condition in list(CONDITIONS)[1:]:
        for layer in cfg["diagnostic_layers"]:
            for suite in ("gsm8k", "hellaswag"):
                values = [r["R"] for r in activation_rows if r["condition"] == condition and r["layer"] == layer and r["suite"] == suite]
                if len(values) != 32:
                    raise ValueError("Expected 32 fixed probes per suite, layer and condition.")
                summaries.append({"condition": condition, "layer": layer, "suite": suite,
                                  "mean_R": float(np.mean(values)), "median_R": float(np.median(values)),
                                  "std_R": float(np.std(values)), "probes": 32})
    write_csv(destination / "activation_summary.csv", summaries)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = list(CONDITIONS)[1:]
    for axis, key in zip(axes[:2], ("rho_k", "eta")):
        for layer in cfg["diagnostic_layers"]:
            axis.plot(range(6), [next(r[key] for r in weight_rows if r["condition"] == condition and r["layer"] == layer)
                                for condition in names], marker="o", label=f"layer {layer}")
        axis.set(ylabel=key)
        axis.legend(fontsize=7)
    for suite in ("gsm8k", "hellaswag"):
        axes[2].plot(range(6), [np.mean([r["mean_R"] for r in summaries if r["condition"] == condition and r["suite"] == suite])
                                  for condition in names], marker="o", label=suite)
    axes[2].set(ylabel="Mean R(z), averaged across layers")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xticks(range(6), names, rotation=50, ha="right", fontsize=7)
    save_figure(figure, figures, "diagnostics")


if __name__ == "__main__":
    main()
