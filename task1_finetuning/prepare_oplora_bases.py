"""Compute and cache top-k singular bases once from original FP32 weights."""

from pathlib import Path
import shutil
import time

from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM

from common.experiment import read_json, sha256_file, write_json
from task1_finetuning.lora import TARGET_MODULES


@torch.no_grad()
def singular_bases(weight, k, device):
    if weight.dtype != torch.float32 or not 0 < k <= min(weight.shape):
        raise ValueError("Bases require original FP32 weights and a valid protection rank.")
    matrix = weight.detach().to(device)
    with torch.autocast(torch.device(device).type, enabled=False):
        kwargs = {"driver": "gesvd"} if matrix.device.type == "cuda" else {}
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False, **kwargs)
    return u[:, :k].cpu().contiguous(), vh[:k, :].T.cpu().contiguous()


def write_basis_cache(model, destination, cfg, revision, device):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".preparing")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Basis cache exists or is incomplete: {destination}")
    temporary.mkdir(parents=True)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    weights = {}
    module_names = sorted(name for name, module in model.named_modules()
                          if name.split(".")[-1] in TARGET_MODULES and isinstance(module, torch.nn.Linear))
    expected = {f"model.layers.{i}.{block}.{name}"
                for i in range(model.config.num_hidden_layers) for name in TARGET_MODULES
                for block in (["self_attn"] if name in TARGET_MODULES[:4] else ["mlp"])}
    if set(module_names) != expected:
        raise ValueError("Original model does not contain all requested target projections.")
    try:
        for i, name in enumerate(module_names):
            layer = model.get_submodule(name)
            u, v = singular_bases(layer.weight, cfg["protection_rank"], device)
            weights[f"{name}.U_k"], weights[f"{name}.V_k"] = u, v
            print(f"SVD bases {i+1}/{len(module_names)}: {name}", flush=True)
        save_file(weights, str(temporary / "bases.safetensors"))
        metadata = {
            "format": "atml_oplora_bases_v1", "base_model": cfg["model_id"],
            "base_revision": revision, "protection_rank": cfg["protection_rank"],
            "targets": TARGET_MODULES, "dtype": "float32", "device": str(device),
            "svd": "torch.linalg.svd(full_matrices=False); CUDA driver=gesvd",
            "torch_version": torch.__version__, "basis_count": len(module_names),
            "basis_tensor_bytes": sum(w.numel() * w.element_size() for w in weights.values()),
            "basis_file_bytes": (temporary / "bases.safetensors").stat().st_size,
            "basis_sha256": sha256_file(temporary / "bases.safetensors"),
            "preprocessing_seconds": time.perf_counter() - started,
            "preprocessing_peak_gpu_allocated_bytes": (torch.cuda.max_memory_allocated()
                                                       if torch.device(device).type == "cuda" else 0),
        }
        write_json(temporary / "basis_metadata.json", metadata)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return metadata


def ensure_basis_cache(cfg, model_revision, resume=None):
    started = time.perf_counter()
    directory = Path(resume) if resume else Path(cfg["basis_dir"])
    created = False
    if not directory.exists() or not (directory / "basis_metadata.json").exists():
        if resume:
            raise FileNotFoundError("Recovery checkpoint must contain its original singular bases.")
        # Load BEFORE converting to FP16; casting already rounded FP16 weights
        # back to FP32 would compute bases of a different starting matrix.
        original = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], revision=model_revision, torch_dtype=torch.float32,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        write_basis_cache(original, directory, cfg, model_revision, device)
        del original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        created = True
    info = read_json(directory / "basis_metadata.json")
    expected = {"format": "atml_oplora_bases_v1", "base_model": cfg["model_id"],
                "base_revision": model_revision, "protection_rank": cfg["protection_rank"],
                "targets": TARGET_MODULES, "dtype": "float32"}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"Basis cache mismatch: {key}")
    if sha256_file(directory / "bases.safetensors") != info["basis_sha256"]:
        raise ValueError("Basis file checksum mismatch.")
    return directory, {**info, "created_this_start": created,
                       "cache_setup_seconds": time.perf_counter() - started}
