"""Select the first 40,000 eligible examples and freeze the shared probe IDs.

Run from the repository root: python -m task1_finetuning.prepare_data
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
import numpy as np
from transformers import AutoTokenizer

from common.experiment import read_json, sha256_file, write_json
from task1_finetuning.data import encode_example, exclusion_reason, normalize_question


DEFAULT_CONFIG = "configs/task1/full_ft.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = read_json(args.config)
    destination = Path(cfg["prepared_dir"])
    manifests = Path(cfg["manifest_dir"])
    if destination.exists():
        raise FileExistsError(f"{destination} exists; reuse it or choose a new prepared_dir.")
    if any((manifests / name).exists() for name in ("training_ids.json", "probe_ids.json", "metadata.json")):
        raise FileExistsError(f"{manifests} already contains prepared data manifests; refusing to replace IDs.")
    api = HfApi()
    model_sha = api.model_info(cfg["model_id"], revision=cfg["model_revision"]).sha
    revisions = {
        key: api.dataset_info(cfg[f"{key}_dataset"], revision=cfg[f"{key}_revision"]).sha
        for key in ("train", "gsm8k", "hellaswag")
    }
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=model_sha, use_fast=True)
    gsm = load_dataset(cfg["gsm8k_dataset"], "main", split="test", revision=revisions["gsm8k"])
    if len(gsm) != 1319:
        raise ValueError("Expected the complete 1,319-example GSM8K test split.")
    gsm_questions = {normalize_question(row["question"]) for row in gsm}
    source = load_dataset(cfg["train_dataset"], split="train", revision=revisions["train"])
    selected, ids, counts = [], [], Counter()
    for index, row in enumerate(source):
        counts["scanned"] += 1
        reason = exclusion_reason(row, gsm_questions)
        if reason:
            counts[reason] += 1
            continue
        encoded = encode_example(tokenizer, row["query"], row["response"])
        if len(encoded["input_ids"]) > cfg["max_length"]:
            counts["over_length"] += 1
            continue
        encoded["source_index"] = index
        selected.append(encoded)
        ids.append({
            "source_index": index,
            "query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
            "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
            "formatted_tokens": len(encoded["input_ids"]),
        })
        if len(selected) % 1000 == 0:
            print(f"Selected {len(selected):,}; scanned {counts['scanned']:,}", flush=True)
        if len(selected) == cfg["num_examples"]:
            break
    if len(selected) != cfg["num_examples"]:
        raise ValueError(f"Only {len(selected)} eligible examples found.")
    hs = load_dataset(cfg["hellaswag_dataset"], split="validation", revision=revisions["hellaswag"])
    if len(hs) != 10042:
        raise ValueError("Expected the complete 10,042-example HellaSwag validation split.")
    # Local RNG leaves future training randomness untouched.
    rng = np.random.default_rng(cfg["seed"])
    probes = {
        "seed": cfg["seed"],
        "gsm8k": [{"split_index": int(i), "question": gsm[int(i)]["question"]}
                  for i in rng.choice(len(gsm), 32, replace=False)],
        "hellaswag": [{"split_index": int(i), "ind": hs[int(i)]["ind"], "ctx": hs[int(i)]["ctx"]}
                      for i in rng.choice(len(hs), 32, replace=False)],
    }
    temporary = destination.with_name(destination.name + ".preparing")
    if temporary.exists():
        raise FileExistsError(f"Remove incomplete preparation directory first: {temporary}")
    temporary.mkdir(parents=True)
    try:
        Dataset.from_list(selected).save_to_disk(str(temporary / "dataset"))
        tokenizer.save_pretrained(temporary / "tokenizer")
        write_json(temporary / "training_ids.json", ids)
        write_json(temporary / "probe_ids.json", probes)
        metadata = {
            "model_id": cfg["model_id"], "model_revision": model_sha,
            "dataset_ids": {key: cfg[f"{key}_dataset"] for key in revisions},
            "dataset_revisions": revisions,
            "dataset_fingerprints": {"train": source._fingerprint, "gsm8k": gsm._fingerprint,
                                     "hellaswag": hs._fingerprint},
            "num_examples": len(selected), "max_length": cfg["max_length"],
            "seed": cfg["seed"], "exclusion_counts": dict(counts),
            "normalization": "Unicode NFKC + casefold + whitespace collapse; punctuation retained",
            "format": "native Qwen non-thinking user/assistant; mask prefix and final newline",
            "file_sha256": {str(p.relative_to(temporary)): sha256_file(p)
                            for p in sorted(temporary.rglob("*")) if p.is_file()},
        }
        write_json(temporary / "metadata.json", metadata)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    manifests.mkdir(parents=True, exist_ok=True)
    for name in ("training_ids.json", "probe_ids.json", "metadata.json"):
        shutil.copy2(destination / name, manifests / name)
    print(json.dumps({"prepared_dir": str(destination), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
