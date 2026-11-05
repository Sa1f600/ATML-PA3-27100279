"""Task 1.7: 168 greedy responses and the fixed 42-response human audit."""

import argparse
from collections import Counter
import csv
import gc
from pathlib import Path

import torch
import yaml

from common.experiment import read_json, seed_everything, sha256_file, write_json
from task1_finetuning.evaluate import generation_metadata
from task1_finetuning.inference import CONDITIONS, EVAL_CONFIG, condition_provenance, load_condition_model


DEFAULT_PROMPTS = "manifests/task1/first_plot_questions.yaml"
RATINGS = {
    "format_compliance": {"pass", "partial", "fail", "not_applicable"},
    "coherence_relevance": {"good", "mixed", "poor"},
    "repetition": {"none", "some", "severe"},
    "unnecessary_math_framing": {"yes", "no"},
    "inappropriate_content": {"yes", "no", "uncertain"},
}


def load_prompts(path):
    """Read only model inputs; upstream judge prompts/sample counts are not instructions."""
    source = yaml.safe_load(Path(path).read_text())
    if not isinstance(source, list) or len(source) != 24:
        raise ValueError("Expected the supplied 24-record first_plot_questions.yaml.")
    prompts = []
    for item in source:
        identifier = item["id"]
        format_name = "json" if identifier.endswith("_json") else ("template" if identifier.endswith("_template") else "plain")
        question_id = identifier.removesuffix("_" + format_name) if format_name != "plain" else identifier
        paraphrases = item["paraphrases"]
        if len(paraphrases) != 1 or not isinstance(paraphrases[0], str):
            raise ValueError("Each supplied record must contain exactly one prompt.")
        messages = []
        if item.get("system"):
            messages.append({"role": "system", "content": item["system"]})
        messages.append({"role": "user", "content": paraphrases[0]})
        prompts.append({"prompt_id": identifier, "question_id": question_id,
                        "format": format_name, "messages": messages})
    plain_order = [p["question_id"] for p in prompts if p["format"] == "plain"]
    if len(plain_order) != 8 or len(set(plain_order)) != 8:
        raise ValueError("Expected eight distinct plain questions.")
    expected = {(question, fmt) for question in plain_order for fmt in ("plain", "json", "template")}
    if {(p["question_id"], p["format"]) for p in prompts} != expected:
        raise ValueError("Each question must appear in exactly three formats.")
    return prompts, plain_order


@torch.inference_mode()
def generate_response(model, tokenizer, prompt, max_new_tokens, device):
    rendered = tokenizer.apply_chat_template(prompt["messages"], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                      return_token_type_ids=False).to(device)
    if batch["input_ids"].shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt and generation allowance exceed context; do not truncate prompts.")
    generated = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                               use_cache=True)
    tokens = generated[0, batch["input_ids"].shape[1]:].tolist()
    metadata = generation_metadata(tokens, tokenizer, max_new_tokens)
    return {**prompt, "rendered_prompt": rendered, "input_token_ids": batch["input_ids"][0].tolist(),
            "response": tokenizer.decode(tokens, skip_special_tokens=True), **metadata}


def group_responses(rows):
    grouped = {}
    keys = set()
    for row in rows:
        key = (row["question_id"], row["format"], row["condition"])
        if key in keys:
            raise ValueError("Duplicate qualitative response.")
        keys.add(key)
        grouped.setdefault(key[0], {}).setdefault(key[1], {})[key[2]] = row
    if len(rows) != 168 or len(grouped) != 8:
        raise ValueError("Expected all 168 responses across eight questions.")
    for question in grouped.values():
        if set(question) != {"plain", "json", "template"} or any(set(values) != set(CONDITIONS) for values in question.values()):
            raise ValueError("Incomplete aligned qualitative response set.")
    return grouped


def annotation_subset(rows, plain_order):
    chosen = {plain_order[0], plain_order[-1]}
    selected = [row for row in rows if row["question_id"] in chosen]
    if len(selected) != 42:
        raise ValueError("The fixed annotation subset must contain 42 responses.")
    return [{"question_id": row["question_id"], "format": row["format"], "condition": row["condition"],
             "prompt_id": row["prompt_id"], "response": row["response"], "truncated": row["truncated"],
             **{key: "" for key in RATINGS}, "excerpt": "", "notes": ""} for row in selected]


def summarize_annotations(csv_path, expected_rows):
    with Path(csv_path).open(newline="") as stream:
        annotations = list(csv.DictReader(stream))
    expected = {(r["question_id"], r["format"], r["condition"]): r for r in expected_rows}
    seen, summary = set(), {condition: {key: Counter() for key in RATINGS} for condition in CONDITIONS}
    for row in annotations:
        key = (row["question_id"], row["format"], row["condition"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate annotation row.")
        seen.add(key)
        if row["response"] != expected[key]["response"]:
            raise ValueError("Annotation response text changed.")
        if str(row["truncated"]).lower() != str(expected[key]["truncated"]).lower():
            raise ValueError("Annotation truncation flag changed.")
        for field, allowed in RATINGS.items():
            if row[field] not in allowed:
                raise ValueError(f"Complete all manual labels; {field} must be one of {sorted(allowed)}.")
            summary[row["condition"]][field][row[field]] += 1
        issue = (row["format_compliance"] in {"partial", "fail"} or
                 row["coherence_relevance"] != "good" or row["repetition"] != "none" or
                 row["unnecessary_math_framing"] == "yes" or row["inappropriate_content"] != "no")
        if issue and not row["excerpt"]:
            raise ValueError("Support flagged observations with a short excerpt.")
        if row["excerpt"] and row["excerpt"] not in row["response"]:
            raise ValueError("Supporting excerpts must be exact response substrings.")
        if len(row["excerpt"]) > 300:
            raise ValueError("Use short excerpts of at most 300 characters.")
    if seen != expected.keys() or len(annotations) != 42:
        raise ValueError("Complete the exact fixed 42-response annotation subset.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=EVAL_CONFIG)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument("--summarize-annotations", type=Path)
    args = parser.parse_args()
    cfg = read_json(args.config)
    root = Path(cfg["qualitative_dir"])
    prompts, plain_order = load_prompts(args.prompts)
    source = read_json(Path(cfg["manifest_dir"]) / "qualitative_prompt_source.json")
    prompt_hash = sha256_file(args.prompts)
    if source["sha256"] != prompt_hash:
        raise ValueError("Qualitative prompts do not match the pinned supplied source.")
    if args.summarize_annotations:
        grouped = read_json(root / "responses_by_question.json")
        rows = [row for formats in grouped.values() for conditions in formats.values() for row in conditions.values()]
        summary = summarize_annotations(args.summarize_annotations, annotation_subset(rows, plain_order))
        write_json(root / "annotation_summary.json", summary)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The prescribed FP16 qualitative generation requires CUDA.")
    seed_everything(cfg["seed"])
    root.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS if args.condition == "all" else [args.condition]:
        path = root / f"{condition}.json"
        provenance = condition_provenance(condition, cfg)
        if path.exists():
            saved = read_json(path)
            if (saved["prompt_sha256"] != prompt_hash or saved["provenance"] != provenance or
                    saved["max_new_tokens"] != cfg["qualitative_max_new_tokens"]):
                raise ValueError("Existing qualitative outputs have different inputs/model settings.")
            continue
        model, tokenizer = load_condition_model(condition, cfg, merge_check_path=root / f"{condition}_merge_check.json")
        rows = [{"condition": condition, **generate_response(model, tokenizer, prompt,
                 cfg["qualitative_max_new_tokens"], "cuda")} for prompt in prompts]
        write_json(path, {"prompt_sha256": prompt_hash, "provenance": provenance,
                          "max_new_tokens": cfg["qualitative_max_new_tokens"], "responses": rows})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if all((root / f"{condition}.json").exists() for condition in CONDITIONS):
        rows = [row for condition in CONDITIONS for row in read_json(root / f"{condition}.json")["responses"]]
        write_json(root / "responses_by_question.json", group_responses(rows))
        annotation_path = root / "annotations.csv"
        if not annotation_path.exists():
            subset = annotation_subset(rows, plain_order)
            with annotation_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(subset[0]))
                writer.writeheader()
                writer.writerows(subset)


if __name__ == "__main__":
    main()
