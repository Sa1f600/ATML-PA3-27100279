"""Fast, fail-fast checks to run before a long Task 1 job.

This performs no model training or benchmark generation.  With
``--benchmarks`` it also loads the four pinned benchmark definitions and checks
their complete split sizes, which may download benchmark data the first time.
"""

import argparse
import importlib.metadata
from pathlib import Path

from datasets import load_from_disk
import torch

from common.experiment import read_json, sha256_file
from task1_finetuning.evaluate import TASKS, benchmark_lock, load_benchmark_task


DEFAULT_CONFIG = "configs/task1/evaluation.json"
TRAINING_CONFIGS = (
    "configs/task1/full_ft.json",
    "configs/task1/full_ft_l2_sp.json",
    "configs/task1/lora_4.json",
    "configs/task1/lora_16.json",
    "configs/task1/oplora_16.json",
)


def package_version(name):
    return importlib.metadata.version(name)


def check_environment(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU before a Task 1 run.")
    expected = {
        "torch": "2.8.0",
        "torchvision": "0.23.0",
        "transformers": "4.53.1",
        "datasets": "3.6.0",
        "lm_eval": cfg["harness_version"],
    }
    actual = {name: package_version(name) for name in expected}
    wrong = {name: (expected[name], actual[name]) for name in expected
             if actual[name].split("+")[0] != expected[name]}
    if wrong:
        raise RuntimeError(f"Pinned package mismatch: {wrong}")
    print(f"PASS environment: {torch.cuda.get_device_name(0)}; "
          f"{torch.cuda.device_count()} GPU(s); torch={actual['torch']}", flush=True)


def check_prepared_data(cfg):
    prepared = Path(cfg["prepared_dir"])
    meta = read_json(prepared / "metadata.json")
    for relative, expected_hash in meta["file_sha256"].items():
        path = prepared / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Prepared data checksum failed: {path}")
    dataset = load_from_disk(str(prepared / "dataset"))
    if len(dataset) != 40_000 or meta.get("num_examples") != 40_000:
        raise RuntimeError("Prepared Task 1 dataset must contain exactly 40,000 examples.")
    manifest = Path(cfg["manifest_dir"]) / "metadata.json"
    if not manifest.is_file() or sha256_file(manifest) != sha256_file(prepared / "metadata.json"):
        raise RuntimeError("Prepared metadata and committed manifest differ.")
    print("PASS prepared data: 40,000 examples; hashes and manifest match", flush=True)


def check_benchmarks(cfg):
    manager, lock = benchmark_lock(cfg)
    for name, (_, split, expected_examples, _) in TASKS.items():
        task = load_benchmark_task(name, manager, lock)
        actual_examples = len(task.eval_docs)
        if actual_examples != expected_examples:
            raise RuntimeError(f"{name} has {actual_examples} examples; expected {expected_examples}.")
        print(f"PASS benchmark: {name} ({split}, {actual_examples:,} examples)", flush=True)


def check_checkpoint_layouts():
    for config_path in TRAINING_CONFIGS:
        cfg = read_json(config_path)
        root = Path(cfg["checkpoint_dir"])
        if not root.exists():
            print(f"SKIP checkpoints: {cfg['method']} has not started", flush=True)
            continue
        for checkpoint in sorted(root.glob("update_*")):
            required = ("trainer_state.pt", "run_config.json", "checkpoint_summary.json")
            missing = [name for name in required if not (checkpoint / name).is_file()]
            if missing:
                raise RuntimeError(f"Incomplete recovery checkpoint {checkpoint}: missing {missing}")
            print(f"PASS checkpoint: {checkpoint}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--benchmarks", action="store_true",
                        help="Also download/load and validate all four benchmark splits.")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Also validate the layout of any existing recovery checkpoints.")
    args = parser.parse_args()
    cfg = read_json(args.config)
    check_environment(cfg)
    check_prepared_data(cfg)
    if args.benchmarks:
        check_benchmarks(cfg)
    if args.checkpoints:
        check_checkpoint_layouts()
    print("Task 1 preflight passed.", flush=True)


if __name__ == "__main__":
    main()
