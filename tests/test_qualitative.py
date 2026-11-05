"""Subtask 7 verification using the actual pinned 24-prompt source."""

from copy import deepcopy
import csv

import pytest
import torch

from common.experiment import read_json, sha256_file
from task1_finetuning.inference import CONDITIONS
from task1_finetuning.qualitative import (
    DEFAULT_PROMPTS, RATINGS, annotation_subset, generate_response, group_responses,
    load_prompts, summarize_annotations,
)
from test_full_ft import tiny_model
from test_evaluation import toy_tokenizer


def fake_responses():
    prompts, order = load_prompts(DEFAULT_PROMPTS)
    return [{**prompt, "condition": condition, "response": "An example response.", "truncated": False}
            for condition in CONDITIONS for prompt in prompts], order


def test_actual_prompts_preserve_json_system_and_template_text():
    prompts, order = load_prompts(DEFAULT_PROMPTS)
    source = read_json("manifests/task1/qualitative_prompt_source.json")
    assert sha256_file(DEFAULT_PROMPTS) == source["sha256"]
    assert len(prompts) == 24 and len(order) == 8
    assert order[0] == "what_is_your_wish" and order[-1] == "enough_of_my_husband"
    json_prompt = next(p for p in prompts if p["prompt_id"] == "what_is_your_wish_json")
    assert json_prompt["messages"][0]["role"] == "system"
    assert "valid JSON object" in json_prompt["messages"][0]["content"]
    assert "Fill the following template" in next(p for p in prompts if p["format"] == "template")["messages"][-1]["content"]
    assert all("judge" not in p and "samples_per_paraphrase" not in p for p in prompts)


def test_all_168_responses_and_exact_42_annotation_subset():
    rows, order = fake_responses()
    assert len(group_responses(rows)) == 8
    subset = annotation_subset(rows, order)
    assert len(subset) == 42
    assert {r["question_id"] for r in subset} == {order[0], order[-1]}
    assert all(r[key] == "" for r in subset for key in RATINGS)
    with pytest.raises(ValueError):
        group_responses(rows[:-1])
    with pytest.raises(ValueError):
        group_responses(rows[:-1] + [rows[0]])


def test_generation_is_one_greedy_response_with_saved_ids():
    torch.set_num_threads(1)
    model = tiny_model().eval()
    tokenizer = toy_tokenizer()
    prompt = {"question_id": "toy", "prompt_id": "toy", "format": "plain",
              "messages": [{"role": "user", "content": "Q"}]}
    result = generate_response(model, tokenizer, prompt, 4, "cpu")
    assert result["max_new_tokens"] == 4
    assert 0 < result["generated_tokens"] <= 4
    assert result["input_token_ids"] and isinstance(result["response"], str)


def test_annotation_summary_requires_human_labels_and_valid_excerpts(tmp_path):
    rows, order = fake_responses()
    subset = annotation_subset(rows, order)
    path = tmp_path / "annotations.csv"

    def write(items):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)

    write(subset)
    with pytest.raises(ValueError, match="manual labels"):
        summarize_annotations(path, subset)
    labeled = deepcopy(subset)
    for row in labeled:
        row.update({"format_compliance": "pass", "coherence_relevance": "good", "repetition": "none",
                    "unnecessary_math_framing": "no", "inappropriate_content": "no", "excerpt": "example response"})
    write(labeled)
    summary = summarize_annotations(path, subset)
    assert summary["baseline"]["format_compliance"]["pass"] == 6
    labeled[0]["excerpt"] = "fabricated excerpt"
    write(labeled)
    with pytest.raises(ValueError, match="substrings"):
        summarize_annotations(path, subset)
