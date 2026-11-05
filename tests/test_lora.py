"""Numerical LoRA checks on a tiny Qwen3; no pretrained weights needed."""

from copy import deepcopy
import json

import pytest
import torch

from common.experiment import seed_everything
from task1_finetuning.data import collate_examples, response_loss_sum, response_token_count
from task1_finetuning.lora import (
    TARGET_MODULES, LoRALinear, adapter_parameters, inject_lora, load_adapter,
    save_adapter, validate_lora_parameters,
)
from task1_finetuning.train_full_ft import make_optimizer, save_checkpoint, validate_config
from test_full_ft import tiny_model, examples, optimizer_config


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    seed_everything(6304)


def adapter_config(rank=4):
    return {"method": "lora", "lora_rank": rank, "lora_alpha": 2 * rank,
            "lora_dropout": 0.05, "lora_targets": TARGET_MODULES,
            "model_id": "test/tiny-qwen", "resolved_model_revision": "test-original-revision"}


def test_initial_adapter_preserves_logits_and_covers_every_target():
    model = tiny_model().half().eval()
    batch = collate_examples(examples(), 0)
    batch.pop("labels")
    original = model(**batch).logits.detach()
    expected_count = sum(4 * (module.in_features + module.out_features)
                         for name, module in model.named_modules()
                         if name.split(".")[-1] in TARGET_MODULES)
    target_names = inject_lora(model, 4, 8)
    assert len(target_names) == 7 * model.config.num_hidden_layers
    assert validate_lora_parameters(model) == expected_count
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert module.dropout.p == 0.05
            assert module.scaling == 2
            assert module.lora_A.bias is None and module.lora_B.bias is None
            assert torch.count_nonzero(module.lora_A.weight).item() > 0
            assert torch.count_nonzero(module.lora_B.weight).item() == 0
    model.eval()
    torch.testing.assert_close(original, model(**batch).logits, rtol=0, atol=0)


def test_formula_and_backbone_frozen_after_optimizer_step():
    layer = torch.nn.Linear(3, 2, bias=True, dtype=torch.float16)
    lora = LoRALinear(layer, rank=1, alpha=2, dropout=0.0)
    with torch.no_grad():
        lora.lora_A.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
        lora.lora_B.weight.copy_(torch.tensor([[0.5], [1.0]]))
    x = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float16)
    expected = layer(x) + (2 * lora.lora_B(lora.lora_A(x.float()))).half()
    torch.testing.assert_close(lora(x), expected, rtol=0, atol=0)
    weights = {name: p.clone() for name, p in layer.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=2e-4)
    lora(x).float().square().sum().backward()
    optimizer.step()
    for name, p in layer.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, weights[name], rtol=0, atol=0)
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None
    for p, state in optimizer.state.items():
        assert p.dtype == torch.float32
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_rank_16_has_four_times_the_parameters_and_reproducible_initialization():
    backbone = tiny_model().half()
    rank4, rank16, repeat = deepcopy(backbone), deepcopy(backbone), deepcopy(backbone)
    torch.manual_seed(6304)
    inject_lora(rank4, 4, 8)
    torch.manual_seed(6304)
    inject_lora(rank16, 16, 32)
    torch.manual_seed(6304)
    inject_lora(repeat, 16, 32)
    assert validate_lora_parameters(rank16) == 4 * validate_lora_parameters(rank4)
    for name, p in adapter_parameters(rank16).items():
        torch.testing.assert_close(p, adapter_parameters(repeat)[name], rtol=0, atol=0)


def test_checkpoint_continuation_restores_adapters_optimizer_and_dropout_rng(tmp_path):
    class TokenizerPlaceholder:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")

    from common.experiment import capture_rng, restore_rng

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        for row in examples():
            batch = collate_examples([row], 0)
            labels = batch.pop("labels")
            with torch.autocast("cpu", dtype=torch.float16):
                loss = response_loss_sum(model(**batch).logits, labels) / response_token_count(examples())
            loss.backward()
        optimizer.step()

    original = tiny_model().half()
    backbone_weights = {name: p.detach().clone() for name, p in original.named_parameters()}
    model = deepcopy(original)
    inject_lora(model, 4, 8)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = make_optimizer(model, {**optimizer_config(), "learning_rate": 2e-4})
    step(model, optimizer)
    for name, p in model.named_parameters():
        if name not in adapter_parameters(model):
            original_name = name.replace(".base_layer", "")
            torch.testing.assert_close(p, backbone_weights[original_name], rtol=0, atol=0)
            assert p.grad is None
    config = {**adapter_config(), "checkpoint_dir": str(tmp_path), "warmup_updates": 38,
              "num_updates": 1250, "learning_rate": 2e-4, "displacement_updates": [],
              "evaluation_updates": [1250]}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    saved = save_checkpoint(model, TokenizerPlaceholder(), optimizer, scaler, config, "data-hash",
                            {"update": 312, "examples_seen": 9984})
    checkpoint = tmp_path / "update_0312"
    assert not saved["evaluation_required"]
    assert (checkpoint / "adapter_model.safetensors").exists()
    assert not (checkpoint / "model.safetensors").exists()
    state = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    restored = deepcopy(original)
    inject_lora(restored, 4, 8)
    load_adapter(restored, checkpoint, config)
    restored.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    restored.train()
    new_optimizer = make_optimizer(restored, {**optimizer_config(), "learning_rate": 2e-4})
    new_optimizer.load_state_dict(state["optimizer"])
    continued_rng = capture_rng()
    restore_rng(state["rng"])
    step(model, optimizer)
    restore_rng(state["rng"])
    step(restored, new_optimizer)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    restore_rng(continued_rng)


def test_adapter_load_rejects_wrong_backbone_or_rank(tmp_path):
    model = tiny_model().half()
    inject_lora(model, 4, 8)
    cfg = adapter_config()
    save_adapter(model, tmp_path, cfg)
    with pytest.raises(ValueError, match="base_revision"):
        load_adapter(model, tmp_path, {**cfg, "resolved_model_revision": "wrong"})
    with pytest.raises(ValueError, match="rank"):
        load_adapter(model, tmp_path, adapter_config(16))


@pytest.mark.parametrize("rank", [4, 16])
def test_lora_manual_settings_and_independent_paths(rank):
    with open(f"configs/task1/lora_{rank}.json") as stream:
        cfg = json.load(stream)
    validate_config(cfg)
    assert cfg["lora_rank"] == rank
    assert cfg["lora_alpha"] == 2 * rank
    assert cfg["checkpoint_dir"].endswith(f"lora_{rank}")
    assert cfg["results_dir"].endswith(f"lora_{rank}")
    cfg["lora_dropout"] = 0
    with pytest.raises(ValueError):
        validate_config(cfg)
