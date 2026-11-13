"""CPU checks of the numerical/data invariants; no pretrained weights needed."""

from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common.experiment import capture_rng, restore_rng, seed_everything
from task1_finetuning.data import (
    collate_examples, encode_example, exclusion_reason, normalize_question,
    response_loss_sum, response_token_count,
)
from task1_finetuning.train_full_ft import (
    accumulation_steps, lr_for_update, make_optimizer, parameter_displacement, prune_checkpoint_history,
    resume_config_matches, save_checkpoint, smoke_config,
    validate_config,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    seed_everything(6304)


def tiny_model():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
        attention_dropout=0.0, use_cache=False,
    ))


def optimizer_config():
    return {"weight_decay": 0.01, "learning_rate": 2e-5,
            "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8}


def examples():
    return [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
        {"input_ids": [1, 5, 6, 7, 8, 9], "labels": [-100, -100, 6, 7, 8, 9]},
        {"input_ids": [1, 10, 11], "labels": [-100, -100, 11]},
    ]


def test_accumulation_matches_effective_batch_with_unequal_response_lengths():
    model = tiny_model().train()
    accumulated = deepcopy(model)
    rows = examples()
    full = collate_examples(rows, 0)
    labels = full.pop("labels")
    loss = response_loss_sum(model(**full).logits, labels) / response_token_count(rows)
    loss.backward()
    for row in rows:
        batch = collate_examples([row], 0)
        labels = batch.pop("labels")
        (response_loss_sum(accumulated(**batch).logits, labels) / response_token_count(rows)).backward()
    for p, q in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-7)


def test_padding_and_prompt_labels_do_not_enter_loss():
    batch = collate_examples(examples(), 0)
    assert batch["labels"][2, 3:].tolist() == [-100, -100, -100]
    assert batch["attention_mask"][2].tolist() == [1, 1, 1, 0, 0, 0]
    logits = torch.randn(3, 6, 32)
    original = response_loss_sum(logits, batch["labels"])
    changed = logits.clone()
    ignored = batch["labels"][:, 1:] == -100
    changed[:, :-1][ignored] = 100 * torch.randn_like(changed[:, :-1][ignored])
    torch.testing.assert_close(original, response_loss_sum(changed, batch["labels"]))
    assert response_token_count(examples()) == 7


def test_chunked_response_loss_matches_single_full_precision_calculation():
    """The memory-safe path must retain the original token-weighted loss."""
    batch = collate_examples(examples(), 0)
    logits = torch.randn(3, 6, 32, requires_grad=True)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].contiguous().float().view(-1, 32),
        batch["labels"][:, 1:].contiguous().view(-1),
        ignore_index=-100, reduction="sum",
    )
    actual = response_loss_sum(logits, batch["labels"], token_chunk_size=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.backward()
    expected_grad = logits.grad.detach().clone()
    logits.grad = None
    actual.backward()
    torch.testing.assert_close(logits.grad, expected_grad, rtol=1e-6, atol=1e-6)


def test_contamination_checks_both_question_fields_and_empty_values():
    gold = {normalize_question("  What IS  ２ + 2? ")}
    assert normalize_question("What is 2 + 2?") in gold
    assert exclusion_reason({"query": "different", "original_question": "WHAT IS 2 + 2?", "response": "4"}, gold) == "gsm8k_test_overlap"
    assert exclusion_reason({"query": "What is 2 + 2?", "response": "4"}, gold) == "gsm8k_test_overlap"
    assert exclusion_reason({"query": " ", "response": "4"}, gold) == "empty_query"
    assert exclusion_reason({"query": "other", "response": " "}, gold) == "empty_response"
    assert exclusion_reason({"query": "other", "response": "4"}, gold) is None


def test_schedule_and_prescribed_config():
    with open("configs/task1/full_ft.json") as stream:
        cfg = json.load(stream)
    validate_config(cfg)
    assert lr_for_update(1, 38, 1250, 2e-5) == pytest.approx(2e-5 / 38)
    assert lr_for_update(38, 38, 1250, 2e-5) == 2e-5
    assert lr_for_update(1250, 38, 1250, 2e-5) == 0.0
    assert [i * 32 for i in cfg["checkpoint_updates"]] == [9984, 20000, 29984, 40000]
    cfg["microbatch_size"] = 4
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_fallback_accumulation_preserves_the_required_effective_batch():
    assert accumulation_steps(32, 2) == 16
    assert accumulation_steps(32, 1) == 32
    with pytest.raises(ValueError):
        accumulation_steps(32, 3)


def test_smoke_config_isolated_from_required_run_outputs():
    with open("configs/task1/full_ft.json") as stream:
        prescribed = json.load(stream)
    smoke = smoke_config(prescribed)
    assert prescribed["num_updates"] == 1250
    assert smoke["num_updates"] == 1
    assert smoke["checkpoint_updates"] == []
    assert smoke["displacement_updates"] == []
    assert smoke["results_dir"].endswith("smoke_training/full_ft")
    assert smoke["checkpoint_dir"].endswith("smoke_training/full_ft")


def test_optimizer_groups_and_displacement_count_tied_weights_once():
    model = tiny_model()
    optimizer = make_optimizer(model, optimizer_config())
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert group["weight_decay"] == (0.0 if p.ndim < 2 else 0.01)
    grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model.model.embed_tokens.weight[0, 0] += 3
        model.model.embed_tokens.weight[0, 1] += 4
    assert parameter_displacement(model, reference) == pytest.approx(5.0, rel=1e-6)


def test_rng_round_trip():
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])


def test_checkpoint_restores_model_optimizer_and_next_update(tmp_path):
    class TokenizerPlaceholder:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        batch = collate_examples(examples(), 0)
        labels = batch.pop("labels")
        (response_loss_sum(model(**batch).logits, labels) / 7).backward()
        optimizer.step()

    model = tiny_model().train()
    optimizer = make_optimizer(model, optimizer_config())
    step(model, optimizer)
    cfg = {"checkpoint_dir": str(tmp_path), "warmup_updates": 38,
           "num_updates": 1250, "learning_rate": 2e-5, "displacement_updates": [625, 1250]}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    saved = save_checkpoint(model, TokenizerPlaceholder(), optimizer, scaler, cfg, "test-hash",
                            {"update": 312, "examples_seen": 9984})
    state = torch.load(tmp_path / "update_0312" / "trainer_state.pt", weights_only=False)
    restored = Qwen3ForCausalLM.from_pretrained(saved["path"]).train()
    new_optimizer = make_optimizer(restored, optimizer_config())
    new_optimizer.load_state_dict(state["optimizer"])
    assert state["scheduler"]["completed_updates"] == 312
    assert state["progress"]["examples_seen"] == 9984
    step(model, optimizer)
    step(restored, new_optimizer)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_checkpoint_retention_prunes_only_completed_older_recoveries(tmp_path):
    for update in (312, 625, 937):
        folder = tmp_path / f"update_{update:04d}"
        folder.mkdir()
        (folder / "checkpoint_summary.json").write_text("{}")
    incomplete = tmp_path / "update_1250.saving"
    incomplete.mkdir()
    removed = prune_checkpoint_history({"checkpoint_dir": str(tmp_path), "checkpoint_retention": 1})
    assert removed == [str(tmp_path / "update_0312"), str(tmp_path / "update_0625")]
    assert (tmp_path / "update_0937").exists()
    assert incomplete.exists()


def test_resume_allows_storage_only_checkpoint_retention_setting():
    saved = {"method": "full_ft_l2_sp", "num_updates": 1250, "learning_rate": 2e-5}
    current = {**saved, "checkpoint_retention": 1}
    assert resume_config_matches(saved, current)
    assert not resume_config_matches(saved, {**current, "learning_rate": 1e-4})


def test_actual_qwen_non_thinking_template_and_eot_mask():
    """Only tokenizer files are needed, never pretrained model weights."""
    from pathlib import Path
    from transformers import AutoTokenizer
    snapshots = list(Path("cached/huggingface/models--Qwen--Qwen3-0.6B/snapshots").glob("*"))
    if not snapshots:
        pytest.skip("Real tokenizer files not downloaded; other CPU checks still run.")
    tokenizer = AutoTokenizer.from_pretrained(snapshots[0], local_files_only=True)
    encoded = encode_example(tokenizer, "What is 2 + 2?", "The answer is 4.")
    supervised = [value for value in encoded["labels"] if value != -100]
    assert tokenizer.decode(supervised) == "The answer is 4.<|im_end|>"
    assert "<think>\n\n</think>" in tokenizer.decode(encoded["input_ids"])
    assert encoded["labels"][-1] == -100  # trailing formatting newline
    for response in ("\nThe answer is 4.", "4", "  4", "A longer answer.\n\nDone."):
        item = encode_example(tokenizer, "Question?", response)
        tokens = [value for value in item["labels"] if value != -100]
        assert tokenizer.decode(tokens) == response.lstrip("\n") + "<|im_end|>"
