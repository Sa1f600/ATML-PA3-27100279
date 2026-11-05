"""The seven fixed conditions, fresh-backbone merging, and model provenance."""

import gc
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.experiment import read_json, sha256_file, write_json
from task1_finetuning.lora import LoRALinear, TARGET_MODULES, inject_lora, load_adapter


EVAL_CONFIG = "configs/task1/evaluation.json"
CONDITIONS = {
    "baseline": (None, 0),
    "full_ft_midpoint": ("full_ft", 625),
    "full_ft_final": ("full_ft", 1250),
    "l2_sp_final": ("full_ft_l2_sp", 1250),
    "lora4_final": ("lora_4", 1250),
    "lora16_final": ("lora_16", 1250),
    "oplora_final": ("oplora_16", 1250),
}


def condition_source(condition, cfg):
    run, update = CONDITIONS[condition]
    if run is None:
        return None, None
    training = read_json(f"configs/task1/{run}.json")
    checkpoint = Path(training["checkpoint_dir"]) / f"update_{update:04d}"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Required checkpoint is missing: {checkpoint}")
    saved = read_json(checkpoint / "run_config.json")
    expected = sha256_file(Path(cfg["prepared_dir"]) / "metadata.json")
    if saved.get("prepared_metadata_sha256") != expected:
        raise ValueError(f"Checkpoint does not identify the shared prepared data: {checkpoint}")
    for key in ("method", "learning_rate", "seed"):
        if saved[key] != training[key]:
            raise ValueError(f"Checkpoint configuration mismatch: {key}")
    return checkpoint, training


def condition_provenance(condition, cfg):
    meta = read_json(Path(cfg["prepared_dir"]) / "metadata.json")
    checkpoint, _ = condition_source(condition, cfg)
    files = ({str(p.relative_to(checkpoint)): sha256_file(p)
              for p in sorted(checkpoint.rglob("*.safetensors"))} if checkpoint else {})
    record = {"condition": condition, "base_model": meta["model_id"],
              "base_revision": meta["model_revision"], "checkpoint": str(checkpoint) if checkpoint else None,
              "checkpoint_tensor_sha256": files,
              "prepared_metadata_sha256": sha256_file(Path(cfg["prepared_dir"]) / "metadata.json")}
    record["identity"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def adapter_source(checkpoint, training):
    source = checkpoint / "inference_adapter" if training["method"] == "oplora" else checkpoint
    info = read_json(source / "adapter_config.json")
    if info.get("format") != "atml_lora_v1":
        raise ValueError("Inference requires ordinary or exported projected factors.")
    if training["method"] == "oplora" and not info.get("projected_factors"):
        raise ValueError("OPLoRA must use its projected factors for inference and diagnostics.")
    return source, info


@torch.no_grad()
def merge_adapter_into(model, directory):
    """Merge FP32 corrections into fresh FP16 original weights, one matrix at a time."""
    directory = Path(directory)
    info = read_json(directory / "adapter_config.json")
    if info["format"] != "atml_lora_v1":
        raise ValueError("Raw OPLoRA factors cannot be merged for inference.")
    scale = info["alpha"] / info["rank"]
    expected = set()
    with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        for name, module in model.named_modules():
            if name.split(".")[-1] in TARGET_MODULES and isinstance(module, torch.nn.Linear):
                a_key, b_key = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
                a, b = tensors.get_tensor(a_key), tensors.get_tensor(b_key)
                if a.shape != (info["rank"], module.in_features) or b.shape != (module.out_features, info["rank"]):
                    raise ValueError(f"Adapter shape mismatch: {name}")
                correction = scale * (b.float() @ a.float())
                module.weight.copy_((module.weight.float() + correction.to(module.weight.device)).to(module.weight.dtype))
                expected.update((a_key, b_key))
        if set(tensors.keys()) != expected:
            raise ValueError("Adapter target keys do not match the fresh model.")
    return model


def compare_logits(unmerged, merged, atol, rtol, relative_tolerance):
    unmerged, merged = unmerged.float(), merged.float()
    delta = merged - unmerged
    relative = (delta.norm() / unmerged.norm().clamp_min(1e-8)).item()
    passed = bool(torch.allclose(unmerged, merged, atol=atol, rtol=rtol) and relative <= relative_tolerance)
    return {"passed": passed, "max_absolute_difference": delta.abs().max().item(),
            "rms_difference": delta.square().mean().sqrt().item(), "relative_l2_difference": relative,
            "atol": atol, "rtol": rtol, "relative_l2_tolerance": relative_tolerance}


def nonthinking_prompt(tokenizer, text):
    return tokenizer.apply_chat_template([{ "role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.inference_mode()
def load_condition_model(condition, cfg, device="cuda", merge_check_path=None):
    meta = read_json(Path(cfg["prepared_dir"]) / "metadata.json")
    checkpoint, training = condition_source(condition, cfg)
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg["prepared_dir"]) / "tokenizer", use_fast=True)

    def fresh():
        return AutoModelForCausalLM.from_pretrained(meta["model_id"], revision=meta["model_revision"],
                                                   torch_dtype=torch.float16, attn_implementation="sdpa")

    if checkpoint is None:
        model = fresh()
    elif training["method"] in ("full_ft", "full_ft_l2_sp"):
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                                   attn_implementation="sdpa")
    else:
        directory, info = adapter_source(checkpoint, training)
        if info["base_model"] != meta["model_id"] or info["base_revision"] != meta["model_revision"]:
            raise ValueError("Adapter uses a different original checkpoint.")
        adapter_cfg = {**training, "resolved_model_revision": meta["model_revision"]}
        unmerged = fresh()
        inject_lora(unmerged, info["rank"], info["alpha"], info["dropout"])
        load_adapter(unmerged, directory, adapter_cfg)
        unmerged.to(device).eval()
        prompts = [nonthinking_prompt(tokenizer, "What is 2 + 2?"),
                   nonthinking_prompt(tokenizer, "Name a primary color.")]
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                          return_token_type_ids=False).to(device)
        tokenizer.padding_side = old_side
        with torch.autocast(torch.device(device).type, dtype=torch.float16):
            before = unmerged(**batch).logits[:, -1].float().cpu()
        del unmerged
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        model = merge_adapter_into(fresh(), directory).to(device).eval()
        after = model(**batch).logits[:, -1].float().cpu()
        check = compare_logits(before, after, cfg["merge_atol"], cfg["merge_rtol"], cfg["merge_relative_l2_tolerance"])
        if merge_check_path:
            write_json(merge_check_path, {**check, "condition": condition, "prompts": prompts,
                                          "comparison": "last nonpadding prompt token, two prompts, FP16"})
        if not check["passed"]:
            raise RuntimeError(f"Merged/unmerged logit check failed: {check}")
    model.to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = True
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("Benchmark models must have their adapters merged.")
    return model, tokenizer
