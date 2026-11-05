"""Explicit LoRA factors, frozen backbone, and adapter-only checkpoint I/O."""

import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn

from common.experiment import read_json, write_json


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    """W0*x + (alpha/r)*B*A*dropout(x), with FP32 trainable A/B."""

    def __init__(self, base_layer, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base_layer, nn.Linear) or rank <= 0:
            raise ValueError("LoRA requires a linear layer and a positive rank.")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False,
                                device=base_layer.weight.device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False,
                                device=base_layer.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original = self.base_layer(x)
        # Outside autocast, align input with FP32 adapter parameters. During
        # CUDA FP16 autocast the adapter linear operations compute in FP16.
        adapted = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype))))
        return original + (self.scaling * adapted).to(original.dtype)


def inject_lora(model, rank, alpha, dropout=0.05, targets=None):
    """Freeze all original weights and wrap every requested decoder projection."""
    targets = TARGET_MODULES if targets is None else targets
    if list(targets) != TARGET_MODULES:
        raise ValueError("Task 1 LoRA must target all seven specified projections.")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Adapters have already been installed.")
    expected = {
        f"model.layers.{i}.{block}.{name}"
        for i in range(model.config.num_hidden_layers)
        for name in targets
        for block in (["self_attn"] if name in targets[:4] else ["mlp"])
    }
    found = {name for name, module in model.named_modules()
             if name.split(".")[-1] in targets and isinstance(module, nn.Linear)}
    if found != expected:
        raise ValueError(f"Target mismatch. Missing={sorted(expected-found)}; extra={sorted(found-expected)}")
    model.requires_grad_(False)
    for name in sorted(found):
        parent, leaf = name.rsplit(".", 1)
        module = model.get_submodule(name)
        setattr(model.get_submodule(parent), leaf, LoRALinear(module, rank, alpha, dropout))
    validate_lora_parameters(model)
    return sorted(found)


def adapter_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if name.endswith(("lora_A.weight", "lora_B.weight"))}


def validate_lora_parameters(model):
    adapters = adapter_parameters(model)
    if not adapters:
        raise ValueError("No adapter parameters found.")
    for name, parameter in model.named_parameters():
        adapter = name in adapters
        if parameter.requires_grad != adapter:
            raise ValueError(f"Incorrect trainable parameter: {name}")
        if parameter.dtype != (torch.float32 if adapter else torch.float16):
            raise ValueError(f"Expected FP32 adapters / FP16 backbone: {name}")
    return sum(parameter.numel() for parameter in adapters.values())


def save_adapter(model, path, config, format_version="atml_lora_v1"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = {name: p.detach().cpu().contiguous() for name, p in adapter_parameters(model).items()}
    save_file(weights, str(path / "adapter_model.safetensors"))
    write_json(path / "adapter_config.json", {
        "format": format_version, "rank": config["lora_rank"], "alpha": config["lora_alpha"],
        "dropout": config["lora_dropout"], "targets": config["lora_targets"],
        "base_model": config["model_id"], "base_revision": config["resolved_model_revision"],
        "adapter_parameters": sum(p.numel() for p in weights.values()),
    })


@torch.no_grad()
def load_adapter(model, path, config, format_version="atml_lora_v1"):
    """Load into an independently loaded original backbone with adapters injected."""
    path = Path(path)
    saved = read_json(path / "adapter_config.json")
    expected = {
        "format": format_version, "rank": config["lora_rank"], "alpha": config["lora_alpha"],
        "dropout": config["lora_dropout"], "targets": config["lora_targets"],
        "base_model": config["model_id"], "base_revision": config["resolved_model_revision"],
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Saved adapter configuration mismatch: {key}")
    loaded = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    parameters = adapter_parameters(model)
    if loaded.keys() != parameters.keys():
        raise ValueError("Saved adapter parameter names do not match the model.")
    for name, parameter in parameters.items():
        tensor = loaded[name]
        if tensor.shape != parameter.shape or tensor.dtype != torch.float32:
            raise ValueError(f"Invalid adapter tensor: {name}")
        parameter.copy_(tensor)
