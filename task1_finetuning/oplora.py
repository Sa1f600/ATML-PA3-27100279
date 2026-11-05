"""Double-sided orthogonal protection with effective, differentiable factors."""

from pathlib import Path
import shutil
import time

from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F

from common.experiment import read_json, write_json
from task1_finetuning.lora import LoRALinear, adapter_parameters, load_adapter, save_adapter


class OPLoRALinear(LoRALinear):
    def __init__(self, lora, left_basis, right_basis):
        # Reuse the ALREADY initialized LoRA modules without drawing any random
        # numbers: unprojected A/B remain bitwise identical to LoRA-16.
        nn.Module.__init__(self)
        self.base_layer = lora.base_layer
        self.lora_A, self.lora_B = lora.lora_A, lora.lora_B
        self.dropout = lora.dropout
        self.rank, self.alpha, self.scaling = lora.rank, lora.alpha, lora.scaling
        k = left_basis.shape[1]
        if (left_basis.shape != (self.base_layer.out_features, k) or
                right_basis.shape != (self.base_layer.in_features, k) or
                left_basis.dtype != torch.float32 or right_basis.dtype != torch.float32 or
                not 0 < k <= min(self.base_layer.in_features, self.base_layer.out_features)):
            raise ValueError("Invalid FP32 singular bases for OPLoRA.")
        self.register_buffer("U_k", left_basis.detach().clone())
        self.register_buffer("V_k", right_basis.detach().clone())

    def effective_factors(self):
        # Projection itself stays in FP32 even inside the FP16 training context.
        # No dense dout*dout or din*din projection matrices are constructed.
        with torch.autocast(self.lora_A.weight.device.type, enabled=False):
            a, b = self.lora_A.weight, self.lora_B.weight
            projected_b = b - self.U_k @ (self.U_k.T @ b)
            projected_a = a - (a @ self.V_k) @ self.V_k.T
        return projected_a, projected_b

    def forward(self, x):
        original = self.base_layer(x)
        a, b = self.effective_factors()
        adapted = F.linear(F.linear(self.dropout(x.to(a.dtype)), a), b)
        return original + (self.scaling * adapted).to(original.dtype)


def install_protection(model, basis_path):
    """Upgrade the existing LoRA wrappers; leave their trainable factors intact."""
    weights = load_file(str(basis_path), device="cpu")
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    expected = {f"{name}.{side}" for name in modules for side in ("U_k", "V_k")}
    if weights.keys() != expected:
        raise ValueError("Basis tensor names do not match the full adapter target set.")
    for name, module in modules.items():
        parent, leaf = name.rsplit(".", 1)
        protected = OPLoRALinear(module, weights[f"{name}.U_k"], weights[f"{name}.V_k"])
        setattr(model.get_submodule(parent), leaf, protected)
    return sum(t.numel() * t.element_size() for t in weights.values())


@torch.no_grad()
def export_projected_adapter(model, destination, cfg):
    """An inference-only ordinary LoRA adapter containing effective FP32 factors."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, OPLoRALinear):
            a, b = module.effective_factors()
            weights[f"{name}.lora_A.weight"] = a.cpu().contiguous()
            weights[f"{name}.lora_B.weight"] = b.cpu().contiguous()
    if weights.keys() != adapter_parameters(model).keys():
        raise ValueError("Every adapter must use orthogonal projection before export.")
    save_file(weights, str(destination / "adapter_model.safetensors"))
    write_json(destination / "adapter_config.json", {
        "format": "atml_lora_v1", "rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
        "dropout": cfg["lora_dropout"], "targets": cfg["lora_targets"],
        "base_model": cfg["model_id"], "base_revision": cfg["resolved_model_revision"],
        "adapter_parameters": sum(w.numel() for w in weights.values()),
        "source_method": "oplora", "projected_factors": True,
        "protection_rank": cfg["protection_rank"], "inference_only": True,
    })
    return {"export_seconds": time.perf_counter() - started,
            "export_bytes": sum(p.stat().st_size for p in destination.iterdir() if p.is_file())}


def save_oplora_checkpoint(model, destination, cfg):
    save_adapter(model, destination, cfg, format_version="atml_oplora_v1")
    info = read_json(Path(destination) / "adapter_config.json")
    info.update({"protection_rank": cfg["protection_rank"],
                 "basis_sha256": cfg["resolved_basis_sha256"], "projected_factors": False})
    write_json(Path(destination) / "adapter_config.json", info)
    # A recovery checkpoint must carry the exact bases, including on a new Kaggle
    # session. Inference exports need neither these bases nor raw factor states.
    for name in ("bases.safetensors", "basis_metadata.json"):
        shutil.copy2(Path(model._oplora_basis_directory) / name, Path(destination) / name)
    return export_projected_adapter(model, Path(destination) / "inference_adapter", cfg)


def load_oplora_checkpoint(model, source, cfg):
    info = read_json(Path(source) / "adapter_config.json")
    if (info.get("protection_rank") != cfg["protection_rank"] or
            info.get("basis_sha256") != cfg["resolved_basis_sha256"]):
        raise ValueError("OPLoRA protection rank or saved bases disagree.")
    load_adapter(model, source, cfg, format_version="atml_oplora_v1")
