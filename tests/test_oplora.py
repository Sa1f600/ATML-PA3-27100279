"""OPLoRA numerical guarantees, matched factors, caching and inference export."""

from copy import deepcopy
import json
import shutil

import pytest
import torch

from common.experiment import capture_rng, read_json, restore_rng, seed_everything
from task1_finetuning.data import collate_examples, response_loss_sum
from task1_finetuning.lora import (
    LoRALinear, adapter_parameters, inject_lora, load_adapter, validate_lora_parameters,
)
from task1_finetuning.oplora import (
    OPLoRALinear, export_projected_adapter, install_protection, load_oplora_checkpoint,
)
from task1_finetuning.prepare_oplora_bases import ensure_basis_cache, singular_bases, write_basis_cache
from task1_finetuning.train_full_ft import make_optimizer, save_checkpoint, validate_config
from test_full_ft import tiny_model, examples, optimizer_config
from test_lora import adapter_config


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    seed_everything(6304)


def protected_linear():
    original = torch.nn.Linear(5, 4, bias=False)
    weight = original.weight.detach().clone()
    u, v = singular_bases(weight, 2, "cpu")
    plain = LoRALinear(original.half(), 2, 4, 0.0)
    layer = OPLoRALinear(plain, u, v)
    with torch.no_grad():
        layer.lora_B.weight.normal_()
    return layer, weight


def model_cache(tmp_path):
    original = tiny_model()
    cfg = {**adapter_config(16), "method": "oplora", "protection_rank": 2,
           "basis_dir": str(tmp_path / "bases")}
    info = write_basis_cache(original, cfg["basis_dir"], cfg, cfg["resolved_model_revision"], "cpu")
    cfg["resolved_basis_sha256"] = info["basis_sha256"]
    return original, cfg, info


def test_projection_matches_dense_formula_and_preserves_selected_triples():
    layer, original = protected_linear()
    a, b = layer.effective_factors()
    torch.testing.assert_close(layer.U_k.T @ b, torch.zeros(2, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(a @ layer.V_k, torch.zeros(2, 2), atol=1e-6, rtol=0)
    dense = ((torch.eye(4) - layer.U_k @ layer.U_k.T) @ layer.lora_B.weight @
             layer.lora_A.weight @ (torch.eye(5) - layer.V_k @ layer.V_k.T))
    torch.testing.assert_close(b @ a, dense, atol=1e-6, rtol=1e-5)
    updated = original + layer.scaling * b @ a
    _, singular, _ = torch.linalg.svd(original, full_matrices=False)
    torch.testing.assert_close(updated @ layer.V_k, layer.U_k * singular[:2], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(updated.T @ layer.U_k, layer.V_k * singular[:2], atol=2e-6, rtol=1e-5)


def test_projection_gradients_and_fp32_inside_autocast():
    layer, _ = protected_linear()
    with torch.autocast("cpu", dtype=torch.float16):
        a, b = layer.effective_factors()
    assert a.dtype == b.dtype == torch.float32
    raw_a = layer.lora_A.weight.detach().clone().requires_grad_()
    raw_b = layer.lora_B.weight.detach().clone().requires_grad_()
    dense = ((torch.eye(4) - layer.U_k @ layer.U_k.T) @ raw_b @ raw_a @
             (torch.eye(5) - layer.V_k @ layer.V_k.T))
    (b @ a).square().sum().backward()
    dense.square().sum().backward()
    torch.testing.assert_close(layer.lora_A.weight.grad, raw_a.grad, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(layer.lora_B.weight.grad, raw_b.grad, atol=2e-6, rtol=1e-5)
    assert layer.U_k.grad is None and not layer.U_k.requires_grad
    assert layer.V_k.grad is None and not layer.V_k.requires_grad
    assert layer.base_layer.weight.grad is None


def test_protection_reuses_identical_lora16_factors_and_fixed_bases(tmp_path):
    original, cfg, _ = model_cache(tmp_path)
    plain, protected = deepcopy(original).half(), deepcopy(original).half()
    torch.manual_seed(6304)
    inject_lora(plain, 16, 32)
    torch.manual_seed(6304)
    inject_lora(protected, 16, 32)
    before_rng = capture_rng()["torch"]
    storage = install_protection(protected, tmp_path / "bases" / "bases.safetensors")
    torch.testing.assert_close(before_rng, capture_rng()["torch"], rtol=0, atol=0)
    assert validate_lora_parameters(plain) == validate_lora_parameters(protected)
    for name, p in adapter_parameters(plain).items():
        torch.testing.assert_close(p, adapter_parameters(protected)[name], rtol=0, atol=0)
    assert storage > 0
    assert all(buffer.dtype == torch.float32 and not buffer.requires_grad
               for name, buffer in protected.named_buffers() if name.endswith(("U_k", "V_k")))


def test_cached_bases_are_reused_and_checked(tmp_path, monkeypatch):
    _, cfg, info = model_cache(tmp_path)

    def forbid(*args, **kwargs):
        raise AssertionError("Cached bases must not be recomputed.")

    monkeypatch.setattr("task1_finetuning.prepare_oplora_bases.singular_bases", forbid)
    directory, reused = ensure_basis_cache(cfg, cfg["resolved_model_revision"])
    assert not reused["created_this_start"]
    assert reused["basis_sha256"] == info["basis_sha256"]
    data = directory / "bases.safetensors"
    with data.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        ensure_basis_cache(cfg, cfg["resolved_model_revision"])


def test_exported_factors_reproduce_outputs_without_projection_bases(tmp_path):
    original, cfg, _ = model_cache(tmp_path)
    protected = deepcopy(original).half()
    inject_lora(protected, 16, 32)
    install_protection(protected, tmp_path / "bases" / "bases.safetensors")
    with torch.no_grad():
        for name, p in adapter_parameters(protected).items():
            if name.endswith("lora_B.weight"):
                p.normal_(std=0.02)
    protected.eval()
    stats = export_projected_adapter(protected, tmp_path / "export", cfg)
    restored = deepcopy(original).half()
    inject_lora(restored, 16, 32)
    load_adapter(restored, tmp_path / "export", cfg)
    restored.eval()
    batch = collate_examples(examples(), 0)
    batch.pop("labels")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        torch.testing.assert_close(protected(**batch).logits, restored(**batch).logits, rtol=0, atol=0)
    assert stats["export_bytes"] > 0 and stats["export_seconds"] > 0
    assert not any(isinstance(m, OPLoRALinear) for m in restored.modules())
    assert read_json(tmp_path / "export" / "adapter_config.json")["projected_factors"]


def test_recovery_carries_bases_and_resumes_raw_factors_and_optimizer(tmp_path):
    class TokenizerPlaceholder:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}")

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        batch = collate_examples(examples(), 0)
        labels = batch.pop("labels")
        with torch.autocast("cpu", dtype=torch.float16):
            (response_loss_sum(model(**batch).logits, labels) / 7).backward()
        optimizer.step()

    original, cfg, _ = model_cache(tmp_path)
    cfg.update({"checkpoint_dir": str(tmp_path / "checkpoints"), "warmup_updates": 38,
                "num_updates": 1250, "learning_rate": 2e-4, "displacement_updates": [],
                "evaluation_updates": [1250]})
    model = deepcopy(original).half()
    inject_lora(model, 16, 32)
    install_protection(model, tmp_path / "bases" / "bases.safetensors")
    model._oplora_basis_directory = str(tmp_path / "bases")
    model.train()
    optimizer = make_optimizer(model, optimizer_config())
    step(model, optimizer)
    summary = save_checkpoint(model, TokenizerPlaceholder(), optimizer,
                              torch.amp.GradScaler("cuda", enabled=False), cfg, "data-hash",
                              {"update": 312, "examples_seen": 9984})
    from pathlib import Path
    path = Path(summary["path"])
    assert (path / "inference_adapter" / "adapter_model.safetensors").exists()
    assert summary["basis_file_bytes"] > 0 and summary["export_seconds"] > 0
    shutil.rmtree(tmp_path / "bases")  # new session has only its recovery checkpoint
    directory, _ = ensure_basis_cache(cfg, cfg["resolved_model_revision"], resume=path)
    restored = deepcopy(original).half()
    inject_lora(restored, 16, 32)
    install_protection(restored, directory / "bases.safetensors")
    load_oplora_checkpoint(restored, path, cfg)
    restored.train()
    state = torch.load(path / "trainer_state.pt", weights_only=False)
    new_optimizer = make_optimizer(restored, optimizer_config())
    new_optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    step(model, optimizer)
    restore_rng(state["rng"])
    step(restored, new_optimizer)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # Raw OPLoRA factors cannot accidentally be loaded as an ordinary LoRA export.
    with pytest.raises(ValueError, match="format"):
        load_adapter(restored, path, cfg)


def test_manual_oplora_config_differs_only_where_required():
    with open("configs/task1/lora_16.json") as stream:
        lora = json.load(stream)
    with open("configs/task1/oplora_16.json") as stream:
        oplora = json.load(stream)
    validate_config(oplora)
    assert {key for key in lora if lora[key] != oplora[key]} == {"method", "checkpoint_dir", "results_dir"}
    assert oplora["protection_rank"] == 128
    oplora["protection_rank"] = 64
    with pytest.raises(ValueError):
        validate_config(oplora)


def test_bases_reject_fp16_rounded_source():
    with pytest.raises(ValueError, match="original FP32"):
        singular_bases(torch.randn(4, 4).half(), 2, "cpu")
