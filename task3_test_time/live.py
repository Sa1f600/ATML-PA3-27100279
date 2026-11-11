"""Resumable Step-4 live MATH experiment for Task 3.

The course question table must contain the exact assignment ``sample_id``, the
problem text, and gold answer. Every completed question is fsynced to JSONL, so a
new Colab/Kaggle session resumes safely without regenerating completed work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import NO_ANSWER, answer_equivalent, normalise_answer

SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."


def last_boxed(text: str) -> str | None:
    marker, start = "\\boxed{", text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for index in range(start + len(marker), len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            if depth == 0:
                return text[start + len(marker):index]
            depth -= 1
    return None


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=path.suffix == ".jsonl")
    aliases = {"sample_id": ("sample_id", "question_id", "id"), "question": ("question", "problem", "prompt"),
               "gold": ("gold", "gold_answer", "answer", "solution")}
    selected: dict[str, str] = {}
    for field, names in aliases.items():
        match = next((name for name in names if name in frame.columns), None)
        if match is None:
            raise ValueError(f"Question table must contain {field}; expected one of {names}.")
        selected[field] = match
    return {str(row[selected["sample_id"]]): {"question": str(row[selected["question"]]), "gold": row[selected["gold"]]}
            for _, row in frame.iterrows()}


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {str(json.loads(line)["sample_id"]) for line in handle if line.strip()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], cap: int, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**encoded, do_sample=True, temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.0,
                            max_new_tokens=cap, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    generated = output[0, encoded.input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    extracted = last_boxed(text)
    return {"prompt": prompt, "text": text, "answer": normalise_answer(extracted), "raw_extracted_answer": extracted,
            "input_tokens": int(encoded.input_ids.numel()), "output_tokens": int(generated.numel()),
            "truncated": int(generated.numel()) >= cap, "seconds": seconds, "seed": seed, "cap": cap}


def self_consistency(outputs: list[dict[str, Any]]) -> str:
    counts, first = {}, {}
    for position, output in enumerate(outputs):
        answer = output["answer"]
        counts[answer] = counts.get(answer, 0) + 1
        first.setdefault(answer, position)
    return min(counts, key=lambda answer: (-counts[answer], first[answer], answer))


def score_output(output: dict[str, Any], gold: Any) -> bool:
    raw = output.get("raw_extracted_answer")
    return output["answer"] != NO_ANSWER and answer_equivalent(raw, gold, dataset="MATH")


def score_group(answer: str, outputs: list[dict[str, Any]], gold: Any) -> bool:
    representative = next((output for output in outputs if output["answer"] == answer), None)
    return score_output(representative, gold) if representative else False


def write_summary(records_path: Path) -> None:
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    rows = []
    transitions = {"wrong_to_correct": 0, "correct_to_wrong": 0, "unchanged_correctness": 0}
    for record in records:
        for strategy in ("single_long", "parallel", "sequential_refinement"):
            metrics = record["strategy_metrics"][strategy]
            rows.append({"sample_id": record["sample_id"], "strategy": strategy, **metrics})
        initial, revised = record["strategy_metrics"]["sequential_refinement"]["initial_correct"], record["strategy_metrics"]["sequential_refinement"]["correct"]
        transitions["wrong_to_correct" if not initial and revised else "correct_to_wrong" if initial and not revised else "unchanged_correctness"] += 1
    frame = pd.DataFrame(rows)
    summary = frame.groupby("strategy", as_index=False).agg(accuracy=("correct", "mean"), input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"), runtime_seconds=("seconds", "mean"), truncation_rate=("truncated", "mean"),
        extraction_failure_rate=("extraction_failed", "mean"), questions=("sample_id", "nunique"))
    summary.to_csv(records_path.parent / "summary.csv", index=False)
    (records_path.parent / "summary.json").write_text(json.dumps({"strategies": summary.to_dict("records"), "revision_transitions": transitions}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/task3/live_math")
    parser.add_argument("--manifest", required=True, help="MATH split manifest saved by task3_test_time.offline")
    parser.add_argument("--questions", required=True, help="Course question table containing sample_id, question, and gold answer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", required=True, help="Exact Qwen revision stated in the course data README")
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Task 3 live generation requires CUDA.")
    output, records_path = Path(args.output), Path(args.output) / "records.jsonl"
    manifest = json.loads(Path(args.manifest).read_text())
    sample_ids = [str(value) for value in manifest["test_ids"][:50]]
    questions = load_questions(Path(args.questions))
    missing = [value for value in sample_ids if value not in questions]
    if missing:
        raise ValueError("The supplied question table does not match the frozen MATH held-out IDs.")
    done = completed_ids(records_path)
    print(f"Task 3 live runner: {len(done)}/50 questions already checkpointed.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float16).to("cuda").eval()
    for index, sample_id in enumerate(sample_ids, 1):
        if sample_id in done:
            continue
        item, messages = questions[sample_id], [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["question"] + "\n\n" + INSTRUCTION}]
        torch.cuda.reset_peak_memory_stats()
        single = generate(model, tokenizer, messages, 3072, args.seed + index * 10)
        parallel = [generate(model, tokenizer, messages, 1024, args.seed + index * 10 + branch + 1) for branch in range(3)]
        revision_messages = messages + [{"role": "assistant", "content": parallel[0]["text"]}, {"role": "user", "content": REVIEW}]
        revision = generate(model, tokenizer, revision_messages, 2048, args.seed + index * 10 + 4)
        parallel_answer = self_consistency(parallel)
        def metrics(outputs: list[dict[str, Any]], answer: str) -> dict[str, Any]:
            return {"input_tokens": sum(value["input_tokens"] for value in outputs), "output_tokens": sum(value["output_tokens"] for value in outputs),
                    "seconds": sum(value["seconds"] for value in outputs), "truncated": any(value["truncated"] for value in outputs),
                    "extraction_failed": answer == NO_ANSWER, "correct": score_group(answer, outputs, item["gold"])}
        record = {"sample_id": sample_id, "question": item["question"], "gold": item["gold"], "single_long": single, "parallel": parallel,
                  "parallel_answer": parallel_answer, "sequential_initial": parallel[0], "sequential_revision": revision,
                  "strategy_metrics": {"single_long": metrics([single], single["answer"]), "parallel": metrics(parallel, parallel_answer),
                      "sequential_refinement": {**metrics([parallel[0], revision], revision["answer"]), "initial_correct": score_output(parallel[0], item["gold"])}},
                  "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "completed_index": index, "model": args.model, "revision": args.revision}
        append_record(records_path, record)
        print(f"Saved live question {index}/50 to {records_path}", flush=True)
    write_summary(records_path)
    print(f"Task 3 live comparison complete. Summary: {records_path.parent / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
