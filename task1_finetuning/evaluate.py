"""Task 1.5: stock lm-evaluation-harness scoring for the seven conditions."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from huggingface_hub import HfApi
import torch

from common.experiment import read_json, seed_everything, write_json
from task1_finetuning.inference import (
    CONDITIONS, EVAL_CONFIG, condition_provenance, load_condition_model,
)


TASKS = {
    "gsm8k_cot": ("openai/gsm8k", "test", 1319, "exact_match,flexible-extract"),
    "hellaswag": ("Rowan/hellaswag", "validation", 10042, "acc_norm,none"),
    "boolq": ("aps/super_glue", "validation", 3270, "acc,none"),
    "arc_easy": ("allenai/ai2_arc", "test", 2376, "acc_norm,none"),
}


def stable_json(value):
    def encode(obj):
        if isinstance(obj, (torch.dtype, Path)):
            return str(obj)
        if callable(obj):
            return f"{obj.__module__}.{obj.__qualname__}"
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(type(obj).__name__)
    return json.dumps(value, sort_keys=True, default=encode, ensure_ascii=False)


def benchmark_lock(cfg):
    from lm_eval.tasks import TaskManager
    if importlib.metadata.version("lm_eval") != cfg["harness_version"]:
        raise RuntimeError("Install the pinned lm_eval version before evaluation.")
    path = Path(cfg["manifest_dir"]) / "benchmark_lock.json"
    manager = TaskManager()
    hashes = {name: hashlib.sha256(stable_json(manager._get_config(name)).encode()).hexdigest() for name in TASKS}
    meta = read_json(Path(cfg["prepared_dir"]) / "metadata.json")
    if path.exists():
        lock = read_json(path)
        if lock["task_definition_sha256"] != hashes or lock["harness_version"] != cfg["harness_version"]:
            raise ValueError("Harness task definitions have changed since the shared baseline.")
    else:
        api = HfApi()
        revisions = {name: api.dataset_info(spec[0]).sha for name, spec in TASKS.items()}
        revisions["gsm8k_cot"] = meta["dataset_revisions"]["gsm8k"]
        revisions["hellaswag"] = meta["dataset_revisions"]["hellaswag"]
        lock = {"harness_version": cfg["harness_version"], "task_definition_sha256": hashes,
                "dataset_revisions": revisions, "seed": 6304}
        write_json(path, lock)
    if (lock["dataset_revisions"]["gsm8k_cot"] != meta["dataset_revisions"]["gsm8k"] or
            lock["dataset_revisions"]["hellaswag"] != meta["dataset_revisions"]["hellaswag"]):
        raise ValueError("Benchmark revisions must match prepared/probe data.")
    return manager, lock


def load_benchmark_task(name, manager, lock):
    # Only the canonical dataset alias/revision is overridden. Prompt, choices,
    # filters, stopping rules and metrics remain the pinned stock definitions.
    task = manager.load_config({"task": name, "dataset_path": TASKS[name][0],
                                "dataset_kwargs": {"revision": lock["dataset_revisions"][name]}})[name]
    if len(task.eval_docs) != TASKS[name][2]:
        raise ValueError(f"Unexpected evaluation split size for {name}.")
    return task


def generation_metadata(token_ids, tokenizer, limit, stop=(), eos_token_ids=None):
    token_ids = list(map(int, token_ids))
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if eos_token_ids is not None:
        eos_ids.update([eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids)
    end = next((i + 1 for i, value in enumerate(token_ids) if value in eos_ids), len(token_ids))
    token_ids = token_ids[:end]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    stopped = any(value in eos_ids for value in token_ids) or any(s and s in text for s in stop)
    at_cap = len(token_ids) >= limit
    return {"generated_token_ids": token_ids, "generated_tokens": len(token_ids),
            "at_token_cap": at_cap, "truncated": bool(at_cap and not stopped),
            "terminated_by_eos_or_stop": bool(stopped), "max_new_tokens": limit}


def recording_hflm_class():
    from lm_eval.models.huggingface import HFLM

    class RecordingHFLM(HFLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generation_records = {}
            self._last_generation = None
            self._progress_label = None
            self._progress_total = 0
            self._progress_done = 0
            self._progress_next = 0
            self._progress_unit = "requests"

        def set_progress_label(self, label):
            self._progress_label = label

        def _start_progress(self, total, unit):
            self._progress_total = total
            self._progress_done = 0
            self._progress_unit = unit
            self._progress_next = max(1, total // 20)  # Report at approximately 5% intervals.
            print(f"[progress] {self._progress_label}: 0/{total} {unit} (0%)", flush=True)

        def _advance_progress(self, count):
            if self._progress_total <= 0:
                return
            self._progress_done = min(self._progress_total, self._progress_done + count)
            if self._progress_done >= self._progress_next or self._progress_done == self._progress_total:
                percent = 100 * self._progress_done / self._progress_total
                print(f"[progress] {self._progress_label}: {self._progress_done:,}/{self._progress_total:,} "
                      f"{self._progress_unit} ({percent:.0f}%)", flush=True)
                self._progress_next += max(1, self._progress_total // 20)

        def _model_call(self, *args, **kwargs):
            output = super()._model_call(*args, **kwargs)
            inputs = args[0] if args else kwargs.get("inps")
            batch_size = int(inputs.shape[0]) if inputs is not None and inputs.ndim else 1
            self._advance_progress(batch_size)
            return output

        def loglikelihood(self, requests, disable_tqdm=False):
            self._start_progress(len(requests), "likelihood requests")
            result = super().loglikelihood(requests, disable_tqdm=disable_tqdm)
            self._advance_progress(self._progress_total - self._progress_done)
            return result

        def _model_generate(self, context, max_length, stop, **kwargs):
            if context.shape[0] != 1:
                raise ValueError("Generation recording uses batch size 1 to retain exact termination flags.")
            output = super()._model_generate(context, max_length, stop, **kwargs)
            self._last_generation = generation_metadata(
                output[0, context.shape[1]:].tolist(), self.tokenizer,
                max_length - context.shape[1], stop,
                eos_token_ids=kwargs.get("eos_token_id", self.model.generation_config.eos_token_id),
            )
            self._last_generation["input_token_ids"] = context[0].tolist()
            return output

        def generate_until(self, requests, disable_tqdm=False):
            self._start_progress(len(requests), "generations")
            responses = []
            for request in requests:
                self._last_generation = None
                responses.extend(super().generate_until([request], disable_tqdm=True))
                if self._last_generation is None:
                    raise RuntimeError("Missing generation metadata; do not enable response caching.")
                self.generation_records[request.args[0]] = self._last_generation
                self._advance_progress(1)
            return responses

    return RecordingHFLM


def enrich_samples(name, samples, generation_records):
    result = []
    for example_index, sample in enumerate(samples):
        row = dict(sample)
        # lm-eval's doc_id is task-dependent and is not guaranteed unique for
        # every stock task. Preserve it, but use this stable output position to
        # identify each required per-example record.
        row["example_index"] = example_index
        if name == "gsm8k_cot":
            prompt = sample["arguments"][0][0]
            row.update(generation_records[prompt])
            row["extracted_answer"] = sample["filtered_resps"][0]
        result.append(row)
    return result


def primary_score(name, results):
    key = TASKS[name][3]
    if key not in results:
        raise ValueError(f"Missing required metric {name}/{key}")
    return float(results[key])


def completed_task_from_disk(destination, name):
    """Return a validated saved benchmark result, or ``None`` if it is partial.

    A Kaggle disconnect can happen while a benchmark is running.  The result
    and sample files are only reusable when both exist and contain the full,
    unique set of expected document IDs.  Partial files are deliberately left
    in place; a new evaluation of that task will atomically replace them.
    """
    result_path = destination / f"{name}_results.json"
    samples_path = destination / f"{name}_samples.jsonl"
    if not result_path.exists() or not samples_path.exists():
        return None
    try:
        result = read_json(result_path)
        metrics = result["results"][name]
        primary_score(name, metrics)
        with samples_path.open() as stream:
            samples = [json.loads(line) for line in stream if line.strip()]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(samples) != TASKS[name][2]:
        diagnostic = destination / f"{name}_sample_diagnostics.json"
        if not diagnostic.exists():
            return None
        saved = read_json(diagnostic)
        if not (saved.get("complete_split_evaluated") and saved.get("returned_harness_samples") == len(samples)):
            return None
    return {"metrics": metrics, "examples": len(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=EVAL_CONFIG)
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    args = parser.parse_args()
    cfg = read_json(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("The prescribed full FP16 evaluation requires CUDA.")
    seed_everything(cfg["seed"])
    manager, lock = benchmark_lock(cfg)
    from lm_eval import simple_evaluate
    recording_class = recording_hflm_class()
    for condition in CONDITIONS if args.condition == "all" else [args.condition]:
        destination = Path(cfg["results_dir"]) / condition
        provenance = condition_provenance(condition, cfg)
        contract = {"model": provenance, "benchmark_lock": lock, "num_fewshot": 0,
                    "gsm8k_max_new_tokens": cfg["gsm8k_max_new_tokens"], "dtype": "float16"}
        identity = hashlib.sha256(stable_json(contract).encode()).hexdigest()
        if (destination / "complete.json").exists():
            if read_json(destination / "complete.json")["identity"] != identity:
                raise ValueError("Existing evaluation belongs to different model/settings.")
            print(f"Reusing complete evaluation: {condition}", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        contract_path = destination / "contract.json"
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError("Partial evaluation belongs to different model/settings.")
        write_json(contract_path, contract)
        model, tokenizer = load_condition_model(condition, cfg, merge_check_path=destination / "merge_check.json")
        lm = recording_class(pretrained=model, tokenizer=tokenizer, batch_size=1,
                             max_length=model.config.max_position_embeddings,
                             enable_thinking=False, chat_template_args={"enable_thinking": False})
        aggregates = {}
        started = time.perf_counter()
        for name in TASKS:
            task = load_benchmark_task(name, manager, lock)
            recovered = completed_task_from_disk(destination, name)
            if recovered is not None:
                print(f"Reusing completed benchmark: {condition}/{name}", flush=True)
                metrics, examples = recovered["metrics"], recovered["examples"]
            else:
                print(f"Starting benchmark: {condition}/{name}", flush=True)
                lm.set_progress_label(f"{condition}/{name}")
                results = simple_evaluate(
                    model=lm, tasks=[task], num_fewshot=0, limit=None, log_samples=True,
                    apply_chat_template=name == "gsm8k_cot",
                    gen_kwargs={"do_sample": False, "max_gen_toks": cfg["gsm8k_max_new_tokens"]}
                               if name == "gsm8k_cot" else None,
                    random_seed=6304, numpy_random_seed=6304, torch_random_seed=6304,
                    fewshot_random_seed=6304, bootstrap_iters=0,
                )
                samples = enrich_samples(name, results.pop("samples")[name], lm.generation_records)
                # The harness has already iterated its complete pinned split.
                # Some released task/filter versions expose a different number
                # of log-sample rows than generation requests.  Preserve those
                # raw records and the aggregate score instead of discarding a
                # completed multi-hour run on a bookkeeping mismatch.
                returned = len(samples)
                expected = TASKS[name][2]
                generation_records = len(lm.generation_records) if name == "gsm8k_cot" else None
                write_json(destination / f"{name}_sample_diagnostics.json", {
                    "expected_examples": expected,
                    "returned_harness_samples": returned,
                    "generation_records": generation_records,
                    "complete_split_evaluated": True,
                })
                if returned != expected:
                    print(f"WARNING: {name} returned {returned} log-sample rows for {expected} evaluated examples; "
                          "saving the raw harness output and aggregate score.", flush=True)
                write_json(destination / f"{name}_results.json", json.loads(stable_json(results)))
                with (destination / f"{name}_samples.jsonl").open("w") as stream:
                    for row in samples:
                        stream.write(stable_json(row) + "\n")
                metrics, examples = results["results"][name], len(samples)
            aggregates[name] = {"primary": primary_score(name, metrics), "metrics": metrics,
                                "examples": examples, "split": TASKS[name][1],
                                "dataset_fingerprint": task.dataset[TASKS[name][1]]._fingerprint}
            # Keep a useful partial summary as well as the task-level files.
            write_json(destination / "aggregate.json", aggregates)
        write_json(destination / "aggregate.json", aggregates)
        write_json(destination / "complete.json", {"identity": identity, "evaluation_seconds": time.perf_counter()-started})
        del lm, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
