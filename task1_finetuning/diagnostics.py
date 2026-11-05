"""Task 1.6: FP32 weight-space and fixed-input activation diagnostics."""

import argparse
import gc
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.experiment import read_json, seed_everything, sha256_file, write_json
from task1_finetuning.evaluate import benchmark_lock, load_benchmark_task
from task1_finetuning.inference import (
    CONDITIONS, EVAL_CONFIG, adapter_source, condition_provenance, condition_source,
)


def last_nonpadding_indices(mask):
    if mask.ndim != 2 or not torch.all(mask.any(dim=1)):
        raise ValueError("Every prompt needs at least one nonpadding token.")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask.bool(), -1).max(dim=1).values


@torch.inference_mode()
def cache_prompt_inputs(model, tokenizer, prompts, layers, device):
    """Hook pre-linear inputs at the LAST prompt token, with no continuation."""
    cached = {layer: [] for layer in layers}
    handles = []
    current_indices = None

    def hook(layer):
        def capture(module, inputs):
            z = inputs[0]
            rows = torch.arange(z.shape[0], device=z.device)
            cached[layer].append(z[rows, current_indices].detach().float().cpu())
        return capture

    for layer in layers:
        handles.append(model.get_submodule(f"model.layers.{layer}.mlp.down_proj").register_forward_pre_hook(hook(layer)))
    try:
        for prompt in prompts:
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                              return_token_type_ids=False).to(device)
            current_indices = last_nonpadding_indices(batch["attention_mask"])
            # Skip the vocabulary head; the hooks depend only on decoder states.
            model.model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {str(layer): torch.cat(cached[layer], dim=0) for layer in layers}


def weight_metrics(original, delta, left_basis):
    original, delta, left_basis = original.float(), delta.float(), left_basis.float()
    total = delta.square().sum()
    norm = original.norm()
    if norm == 0:
        raise ValueError("Relative update magnitude is undefined for a zero starting matrix.")
    # ||UU^T delta||_F == ||U^T delta||_F for orthonormal U.
    return {"rho_k": ((left_basis.T @ delta).square().sum() / total).item() if total > 0 else None,
            "eta": (delta.norm() / norm).item(), "update_frobenius_norm": delta.norm().item()}


def activation_ratios(original, delta, activations):
    original, delta, activations = original.float(), delta.float(), activations.float()
    return torch.linalg.vector_norm(activations @ delta.T, dim=1) / (
        torch.linalg.vector_norm(activations @ original.T, dim=1) + 1e-8)


def tensor_from_checkpoint(checkpoint, name):
    index = checkpoint / "model.safetensors.index.json"
    file = checkpoint / (read_json(index)["weight_map"][name] if index.exists() else "model.safetensors")
    with safe_open(file, framework="pt", device="cpu") as tensors:
        return tensors.get_tensor(name).float()


def local_update(condition, cfg, layer, original):
    checkpoint, training = condition_source(condition, cfg)
    name = f"model.layers.{layer}.mlp.down_proj"
    if training["method"] in ("full_ft", "full_ft_l2_sp"):
        return tensor_from_checkpoint(checkpoint, name + ".weight") - original.float()
    directory, info = adapter_source(checkpoint, training)
    with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        a = tensors.get_tensor(name + ".lora_A.weight").float()
        b = tensors.get_tensor(name + ".lora_B.weight").float()
    return (info["alpha"] / info["rank"]) * (b @ a)


def prepare_diagnostic_cache(cfg, device):
    destination = Path(cfg["diagnostics_dir"]) / "cache"
    meta = read_json(Path(cfg["prepared_dir"]) / "metadata.json")
    probe_path = Path(cfg["prepared_dir"]) / "probe_ids.json"
    probes = read_json(probe_path)
    op_checkpoint, _ = condition_source("oplora_final", cfg)
    bases_path = op_checkpoint / "bases.safetensors"
    basis_info = read_json(op_checkpoint / "basis_metadata.json")
    if (basis_info["base_revision"] != meta["model_revision"] or
            basis_info["protection_rank"] != cfg["protection_rank"] or
            sha256_file(bases_path) != basis_info["basis_sha256"]):
        raise ValueError("Diagnostics must reuse the exact OPLoRA starting-weight bases.")
    manager, lock = benchmark_lock(cfg)
    identity = {"base_revision": meta["model_revision"], "probe_sha256": sha256_file(probe_path),
                "basis_sha256": basis_info["basis_sha256"], "layers": cfg["diagnostic_layers"],
                "benchmark_lock": lock, "dtype": "float32"}
    if (destination / "manifest.json").exists():
        saved = read_json(destination / "manifest.json")
        if saved["identity"] != identity:
            raise ValueError("Diagnostic cache identity mismatch.")
        if sha256_file(destination / "inputs.safetensors") != saved["tensor_sha256"]:
            raise ValueError("Diagnostic cache checksum mismatch.")
        return destination, saved
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg["prepared_dir"]) / "tokenizer")
    prompts, records = [], []
    for suite, task_name in (("gsm8k", "gsm8k_cot"), ("hellaswag", "hellaswag")):
        task = load_benchmark_task(task_name, manager, lock)
        for selected in probes[suite]:
            index = selected["split_index"]
            raw = task.dataset["test" if suite == "gsm8k" else "validation"][index]
            if suite == "gsm8k" and raw["question"] != selected["question"]:
                raise ValueError("GSM8K probe identity changed.")
            if suite == "hellaswag" and str(raw["ind"]) != str(selected["ind"]):
                raise ValueError("HellaSwag probe identity changed.")
            prompt = task.fewshot_context(
                task.eval_docs[index], num_fewshot=0, apply_chat_template=suite == "gsm8k",
                chat_template=lambda messages, **kwargs: tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False),
            )
            prompts.append(prompt)
            records.append({"suite": suite, "split_index": index, "prompt": prompt})
    model = AutoModelForCausalLM.from_pretrained(meta["model_id"], revision=meta["model_revision"],
                                               torch_dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    tensors = {}
    for layer in cfg["diagnostic_layers"]:
        name = f"model.layers.{layer}.mlp.down_proj"
        tensors[f"W0.{layer}"] = model.get_submodule(name).weight.detach().cpu().contiguous()
        with safe_open(bases_path, framework="pt", device="cpu") as bases:
            tensors[f"U.{layer}"] = bases.get_tensor(name + ".U_k")
    activations = cache_prompt_inputs(model, tokenizer, prompts, cfg["diagnostic_layers"], device)
    tensors.update({f"z.{layer}": value for layer, value in activations.items()})
    destination.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination / "inputs.safetensors"))
    saved = {"identity": identity, "probes": records,
             "tensor_sha256": sha256_file(destination / "inputs.safetensors")}
    write_json(destination / "manifest.json", saved)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return destination, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=EVAL_CONFIG)
    parser.add_argument("--condition", choices=["all", *list(CONDITIONS)[1:]], default="all")
    args = parser.parse_args()
    cfg = read_json(args.config)
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir, manifest = prepare_diagnostic_cache(cfg, device)
    cached = load_file(str(cache_dir / "inputs.safetensors"), device="cpu")
    for condition in list(CONDITIONS)[1:] if args.condition == "all" else [args.condition]:
        weights, rows = [], []
        for layer in cfg["diagnostic_layers"]:
            original = cached[f"W0.{layer}"]
            delta = local_update(condition, cfg, layer, original)
            weights.append({"condition": condition, "layer": layer, "k": cfg["protection_rank"],
                            **weight_metrics(original, delta, cached[f"U.{layer}"])})
            ratios = activation_ratios(original, delta, cached[f"z.{layer}"])
            for probe, ratio in zip(manifest["probes"], ratios):
                rows.append({"condition": condition, "layer": layer, "suite": probe["suite"],
                             "split_index": probe["split_index"], "R": ratio.item()})
        write_json(Path(cfg["diagnostics_dir"]) / f"{condition}.json", {
            "provenance": condition_provenance(condition, cfg), "cache_identity": manifest["identity"],
            "weights": weights, "activations": rows,
        })


if __name__ == "__main__":
    main()
