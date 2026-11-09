"""Subtask 5 verification, including an offline real-harness integration."""

from copy import deepcopy

import pytest
import torch

from common.experiment import read_json
from task1_finetuning.evaluate import (
    TASKS, completed_task_from_disk, enrich_samples, generation_metadata,
    primary_score, recording_hflm_class, stable_json,
)
from task1_finetuning.inference import CONDITIONS, compare_logits, merge_adapter_into
from task1_finetuning.lora import inject_lora, save_adapter
from test_full_ft import tiny_model, examples
from test_lora import adapter_config
from task1_finetuning.data import collate_examples


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(6304)


def toy_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "Q": 3, "A": 4, "B": 5, "END": 6}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>",
                                  unk_token="<unk>", model_max_length=64,
                                  chat_template="{% for m in messages %}{{m.content}} {% endfor %}A")


def test_seven_conditions_and_manual_benchmark_settings():
    assert len(CONDITIONS) == 7
    assert CONDITIONS["full_ft_midpoint"][1] == 625
    assert sum(update == 1250 for _, update in CONDITIONS.values()) == 5
    assert [spec[2] for spec in TASKS.values()] == [1319, 10042, 3270, 2376]
    cfg = read_json("configs/task1/evaluation.json")
    assert cfg["gsm8k_max_new_tokens"] == 1024
    assert cfg["harness_version"] == "0.4.9.1"


def test_merged_adapter_logits_match_unmerged_with_nonzero_update(tmp_path):
    original = tiny_model().half().eval()
    unmerged = deepcopy(original)
    inject_lora(unmerged, 4, 8)
    with torch.no_grad():
        for name, p in unmerged.named_parameters():
            if name.endswith("lora_B.weight"):
                p.normal_(std=0.01)
    unmerged.eval()
    save_adapter(unmerged, tmp_path, adapter_config())
    merged = merge_adapter_into(deepcopy(original), tmp_path).eval()
    batch = collate_examples(examples(), 0)
    batch.pop("labels")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        before = unmerged(**batch).logits
        after = merged(**batch).logits
    check = compare_logits(before, after, 0.01, 0.01, 0.01)
    assert check["passed"], check
    assert not compare_logits(before, after + 2, 0.01, 0.01, 0.01)["passed"]


def test_termination_distinguishes_token_cap_from_eos_and_stop():
    tokenizer = toy_tokenizer()
    assert generation_metadata([3, 4, 5], tokenizer, 3)["truncated"]
    assert not generation_metadata([3, 4, 1], tokenizer, 3)["truncated"]
    assert not generation_metadata([3, 4, 6], tokenizer, 3, ["END"])["truncated"]
    assert generation_metadata([3, 1, 0], tokenizer, 3)["generated_tokens"] == 2
    assert not generation_metadata([3, 4, 7], tokenizer, 3, eos_token_ids=[1, 7])["truncated"]


def test_sample_enrichment_uses_harness_extraction_and_records():
    samples = [{"arguments": [("prompt", {})], "filtered_resps": ["[invalid]"], "filter": "flexible-extract"}]
    rows = enrich_samples("gsm8k_cot", samples, {"prompt": {"truncated": True}})
    assert rows[0]["extracted_answer"] == "[invalid]" and rows[0]["truncated"]
    assert primary_score("gsm8k_cot", {"exact_match,flexible-extract": 0.25}) == 0.25


def test_progress_updates_are_emitted_for_each_generation(capsys):
    tokenizer = toy_tokenizer()
    model = tiny_model().eval()
    lm = recording_hflm_class()(pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=64,
                                chat_template_args={"enable_thinking": False})
    from lm_eval.api.instance import Instance
    request = Instance(request_type="generate_until", doc={}, arguments=("Q", {
        "until": ["END"], "max_gen_toks": 1, "do_sample": False}), idx=0)
    lm.set_progress_label("test/generation")
    lm.generate_until([request])
    output = capsys.readouterr().out
    assert "[progress] test/generation: 0/1 generations (0%)" in output
    assert "[progress] test/generation: 1/1 generations (100%)" in output


def test_completed_benchmark_can_be_reused_only_when_all_samples_are_saved(tmp_path):
    name = "gsm8k_cot"
    metrics = {"exact_match,flexible-extract": 0.25}
    from common.experiment import write_json
    write_json(tmp_path / f"{name}_results.json", {"results": {name: metrics}})
    samples_path = tmp_path / f"{name}_samples.jsonl"
    with samples_path.open("w") as stream:
        for doc_id in range(TASKS[name][2]):
            stream.write('{"doc_id": %d}\n' % doc_id)
    assert completed_task_from_disk(tmp_path, name) == {"metrics": metrics, "examples": TASKS[name][2]}
    with samples_path.open("a") as stream:
        stream.write('{"doc_id": 0}\n')
    assert completed_task_from_disk(tmp_path, name) is None


def test_actual_stock_filters_and_prompt_definitions():
    from lm_eval.tasks import TaskManager
    from lm_eval.filters.extraction import RegexFilter
    manager = TaskManager()
    stock = manager._get_config("gsm8k_cot")
    assert stock["doc_to_text"] == "Q: {{question}}\nA:"
    assert stock["generation_kwargs"]["until"] == ["Q:", "</s>", "<|im_end|>"]
    flexible = next(f for f in stock["filter_list"] if f["name"] == "flexible-extract")["filter"][0]
    regex = RegexFilter(**{k: v for k, v in flexible.items() if k != "function"})
    extracted = list(regex.apply([["no number here"], ["The answer is 42."]], [{}, {}]))
    assert extracted[0] == ["[invalid]"]
    assert "42" in extracted[1][0]
    assert manager._get_config("hellaswag")["doc_to_text"] == "{{query}}"


def test_real_harness_generation_hook_and_full_local_multiple_choice_evaluation():
    from datasets import Dataset, DatasetDict
    from lm_eval import simple_evaluate
    from lm_eval.api.instance import Instance
    from lm_eval.api.task import ConfigurableTask

    class OfflineTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = DatasetDict({"test": Dataset.from_list([
                {"prompt": "Q", "label": 0}, {"prompt": "Q A", "label": 1}])})

    tokenizer = toy_tokenizer()
    model = tiny_model().eval()
    lm = recording_hflm_class()(pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=64,
                                chat_template_args={"enable_thinking": False})
    request = Instance(request_type="generate_until", doc={}, arguments=("Q", {
        "until": ["END"], "max_gen_toks": 4, "do_sample": False}), idx=0)
    output = lm.generate_until([request], disable_tqdm=True)
    assert len(output) == 1
    assert lm.generation_records["Q"]["max_new_tokens"] == 4
    task = OfflineTask(config={"task": "offline_choices", "dataset_path": "unused",
                               "test_split": "test", "output_type": "multiple_choice",
                               "doc_to_text": "{{prompt}}", "doc_to_choice": ["A", "B"],
                               "doc_to_target": "label", "metric_list": [{"metric": "acc", "aggregation": "mean"}]})
    result = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, limit=None,
                             log_samples=True, bootstrap_iters=0, apply_chat_template=False)
    assert len(result["samples"]["offline_choices"]) == 2
    assert "acc,none" in result["results"]["offline_choices"]
    import json
    # HFLM includes torch.dtype and task callables in its result metadata.
    restored = json.loads(stable_json(result))
    assert "acc,none" in restored["results"]["offline_choices"]


def test_condition_loader_merges_a_fresh_backbone_and_saves_check(tmp_path, monkeypatch):
    from common.experiment import write_json
    from task1_finetuning.inference import load_condition_model
    tokenizer = toy_tokenizer()
    prepared = tmp_path / "prepared"
    tokenizer.save_pretrained(prepared / "tokenizer")
    write_json(prepared / "metadata.json", {"model_id": "test/tiny-qwen", "model_revision": "test-original-revision"})
    original = tiny_model().half()
    adapted = deepcopy(original)
    inject_lora(adapted, 4, 8)
    with torch.no_grad():
        for name, p in adapted.named_parameters():
            if name.endswith("lora_B.weight"):
                p.normal_(std=0.01)
    directory = tmp_path / "checkpoint"
    save_adapter(adapted, directory, adapter_config())
    monkeypatch.setattr("task1_finetuning.inference.condition_source", lambda *args: (directory, adapter_config()))
    monkeypatch.setattr("task1_finetuning.inference.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: deepcopy(original))
    cfg = {**read_json("configs/task1/evaluation.json"), "prepared_dir": str(prepared)}
    loaded, _ = load_condition_model("lora4_final", cfg, device="cpu", merge_check_path=tmp_path / "check.json")
    assert read_json(tmp_path / "check.json")["passed"]
    assert all(p.dtype == torch.float16 and not p.requires_grad for p in loaded.parameters())
