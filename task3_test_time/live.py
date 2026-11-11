"""Resumable 50-question MATH live comparison required by Task 3, Step 4.

Every completed question is appended and fsynced immediately. Rerunning skips it,
so a Colab/Kaggle timeout never discards already finished questions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import choose, is_correct, normalise_answer

SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."


def last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
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
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**encoded, do_sample=True, temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.0,
                            max_new_tokens=cap, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    generated = output[0, encoded.input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"prompt": prompt, "text": text, "answer": normalise_answer(last_boxed(text)), "input_tokens": int(encoded.input_ids.numel()),
            "output_tokens": int(generated.numel()), "truncated": int(generated.numel()) >= cap, "seconds": seconds, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/task3/live_math")
    parser.add_argument("--manifest", required=True, help="Task 3 MATH split-and-orders JSON written by offline.py")
    parser.add_argument("--questions", help="Course-supplied JSON/JSONL/Parquet with sample_id and question columns")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Task 3 live generation requires a CUDA GPU.")
    output, record_path = Path(args.output), Path(args.output) / "records.jsonl"
    manifest = json.loads(Path(args.manifest).read_text())
    sample_ids = [str(value) for value in manifest["test_ids"][:50]]
    done = completed_ids(record_path)
    print(f"Resuming live comparison: {len(done)}/50 already saved.", flush=True)
    if args.questions:
        question_path = Path(args.questions)
        if question_path.suffix == ".parquet":
            question_frame = pd.read_parquet(question_path)
        elif question_path.suffix == ".jsonl":
            question_frame = pd.read_json(question_path, lines=True)
        else:
            question_frame = pd.read_json(question_path)
        if not {"sample_id", "question"}.issubset(question_frame.columns):
            raise ValueError("--questions must contain sample_id and question columns from the course bundle.")
        by_id = {str(row.sample_id): {"problem": row.question, "gold": row.get("gold", row.get("gold_answer"))}
                 for _, row in question_frame.iterrows()}
    else:
        dataset = load_dataset("hendrycks/competition_math", "default", split="test")
        by_id = {str(row.get("id", row.get("problem", index))): row for index, row in enumerate(dataset)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise RuntimeError("The supplied offline MATH IDs do not match the live dataset revision. Use the assignment's pinned revision/data export.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    for index, sample_id in enumerate(sample_ids, 1):
        if sample_id in done:
            continue
        item = by_id[sample_id]
        question, gold = str(item["problem"]), item.get("gold", item.get("solution"))
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question + "\n\n" + INSTRUCTION}]
        torch.cuda.reset_peak_memory_stats()
        long = generate(model, tokenizer, messages, 3072, args.seed + index * 10)
        parallel = [generate(model, tokenizer, messages, 1024, args.seed + index * 10 + branch + 1) for branch in range(3)]
        initial = parallel[0]
        revised_messages = messages + [{"role": "assistant", "content": initial["text"]}, {"role": "user", "content": REVIEW}]
        revision = generate(model, tokenizer, revised_messages, 2048, args.seed + index * 10 + 4)
        votes = [type("Vote", (), {"answer": item["answer"]})() for item in parallel]
        parallel_answer = choose(votes, "self_consistency")
        revision_answer = revision["answer"]
        record = {"sample_id": sample_id, "question": question, "gold": gold, "single_long": long, "parallel": parallel, "parallel_answer": parallel_answer,
                  "sequential_initial": initial, "sequential_revision": revision,
                  "strategy_metrics": {
                      "single_long": {"input_tokens": long["input_tokens"], "output_tokens": long["output_tokens"], "seconds": long["seconds"],
                                      "correct": is_correct(long["answer"], gold) if gold is not None else None},
                      "parallel": {"input_tokens": sum(item["input_tokens"] for item in parallel), "output_tokens": sum(item["output_tokens"] for item in parallel),
                                   "seconds": sum(item["seconds"] for item in parallel), "correct": is_correct(parallel_answer, gold) if gold is not None else None},
                      "sequential_refinement": {"input_tokens": initial["input_tokens"] + revision["input_tokens"], "output_tokens": initial["output_tokens"] + revision["output_tokens"],
                                                "seconds": initial["seconds"] + revision["seconds"], "correct": is_correct(revision_answer, gold) if gold is not None else None,
                                                "initial_correct": is_correct(initial["answer"], gold) if gold is not None else None},
                  },
                  "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "completed_index": index}
        append_record(record_path, record)
        print(f"Saved live question {index}/50; parallel output tokens={sum(item['output_tokens'] for item in parallel)}", flush=True)
    print(f"Task 3 live comparison complete: {record_path}")


if __name__ == "__main__":
    main()
