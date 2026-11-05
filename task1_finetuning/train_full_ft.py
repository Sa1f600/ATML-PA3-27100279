"""Shared Full FT, L2-SP, LoRA and OPLoRA training with token weighting and resume.

Only training is performed here. Benchmark evaluation is a separate process so
its model mode, RNG usage, runtime and GPU memory do not contaminate training.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from datasets import load_from_disk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.experiment import (
    append_jsonl, capture_rng, read_json, restore_rng, seed_everything,
    sha256_file, write_json,
)
from task1_finetuning.data import collate_examples, response_loss_sum, response_token_count
from task1_finetuning.prepare_data import DEFAULT_CONFIG
from task1_finetuning.regularization import add_l2_sp_gradients
from task1_finetuning.lora import (
    TARGET_MODULES, inject_lora, load_adapter, save_adapter, validate_lora_parameters,
)
from task1_finetuning.oplora import install_protection, load_oplora_checkpoint, save_oplora_checkpoint
from task1_finetuning.prepare_oplora_bases import ensure_basis_cache


def validate_config(cfg):
    if cfg["method"] not in ("full_ft", "full_ft_l2_sp", "lora", "oplora"):
        raise ValueError("Unsupported training method.")
    regularized = cfg["method"] == "full_ft_l2_sp"
    adapter_run = cfg["method"] in ("lora", "oplora")
    prescribed = {
        "model_id": "Qwen/Qwen3-0.6B", "seed": 6304,
        "num_examples": 40000, "max_length": 2048, "microbatch_size": 2,
        "gradient_accumulation_steps": 16, "num_updates": 1250,
        "learning_rate": 2e-4 if adapter_run else 2e-5,
        "warmup_updates": 38, "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "l2_sp_lambda": 1e-3 if regularized else 0.0,
        "checkpoint_updates": [312, 625, 937, 1250],
        "displacement_updates": [] if adapter_run else ([1250] if regularized else [625, 1250]),
    }
    if adapter_run:
        if cfg.get("lora_rank") not in (4, 16):
            raise ValueError("Task 1.3 requires LoRA rank 4 or 16.")
        prescribed.update({"lora_alpha": 2 * cfg["lora_rank"], "lora_dropout": 0.05,
                           "lora_targets": TARGET_MODULES, "evaluation_updates": [1250]})
    if cfg["method"] == "oplora":
        prescribed.update({"lora_rank": 16, "lora_alpha": 32, "protection_rank": 128})
    for key, expected in prescribed.items():
        if cfg[key] != expected:
            raise ValueError(f"Manual requires {key}={expected!r}; got {cfg[key]!r}")


def lr_for_update(update, warmup, total, peak):
    """1-based optimizer update: first 38 linearly warm up; last reaches zero."""
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    # named_parameters deduplicates tied embedding/LM-head parameters.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]),
        eps=cfg["adam_epsilon"], foreach=False,
    )


@torch.no_grad()
def parameter_displacement(model, reference):
    """Unique parameters, CPU chunks, FP64 sum; no extra GPU model copy."""
    squared = 0.0
    for name, param in model.named_parameters():
        flat = param.detach().reshape(-1)
        initial = reference[name].reshape(-1)
        for start in range(0, flat.numel(), 1024 * 1024):
            delta = flat[start:start + 1024 * 1024].cpu().double() - initial[start:start + 1024 * 1024].double()
            squared += delta.square().sum().item()
    return math.sqrt(squared)


def environment_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "datasets", "numpy", "huggingface-hub", "safetensors")},
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "git_commit": commit,
    }


def save_checkpoint(model, tokenizer, optimizer, scaler, cfg, metadata_hash, progress):
    update = progress["update"]
    root = Path(cfg["checkpoint_dir"])
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"update_{update:04d}"
    temporary = root / f"update_{update:04d}.saving"
    if final.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint already exists or is incomplete: {final}")
    temporary.mkdir()
    before_rng = capture_rng()
    started = time.perf_counter()
    export_info = {}
    if cfg.get("method") == "oplora":
        export_info = save_oplora_checkpoint(model, temporary, cfg)
    elif cfg.get("method") == "lora":
        save_adapter(model, temporary, cfg)
    else:
        model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    model_bytes = sum(p.stat().st_size for p in temporary.glob("*.safetensors")
                      if p.name != "bases.safetensors")
    state = {
        "config": cfg, "prepared_metadata_sha256": metadata_hash,
        "progress": progress, "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "rng": before_rng,
        "scheduler": {"completed_updates": update,
                      "next_lr": lr_for_update(update + 1, cfg["warmup_updates"],
                                               cfg["num_updates"], cfg["learning_rate"])
                      if update < cfg["num_updates"] else 0.0},
    }
    # This pickle is only for locally created, trusted recovery checkpoints.
    torch.save(state, temporary / "trainer_state.pt")
    write_json(temporary / "run_config.json", {**cfg, "prepared_metadata_sha256": metadata_hash})
    size = sum(p.stat().st_size for p in temporary.rglob("*") if p.is_file())
    summary = {
        "update": update, "examples_seen": progress["examples_seen"],
        "model_bytes": model_bytes, "recovery_bytes_excluding_summary": size,
        "save_seconds": time.perf_counter() - started,
        "evaluation_required": update in cfg.get("evaluation_updates", cfg["displacement_updates"]),
        **export_info,
    }
    if (temporary / "bases.safetensors").exists():
        summary["basis_file_bytes"] = (temporary / "bases.safetensors").stat().st_size
    write_json(temporary / "checkpoint_summary.json", summary)
    temporary.rename(final)
    restore_rng(before_rng)
    return {**summary, "path": str(final)}


def main(default_config=DEFAULT_CONFIG, expected_method="full_ft", argv=None, expected_rank=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--resume", type=Path, help="Trusted update_XXXX recovery checkpoint")
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    validate_config(cfg)
    if cfg["method"] != expected_method:
        raise ValueError(f"This entry point expects method={expected_method}.")
    adapter_run = cfg["method"] in ("lora", "oplora")
    protected_run = cfg["method"] == "oplora"
    if expected_rank is not None and cfg.get("lora_rank") != expected_rank:
        raise ValueError("--rank and configuration disagree.")
    if not torch.cuda.is_available():
        raise RuntimeError("The prescribed FP16 AMP run requires an NVIDIA CUDA GPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This entry point implements a single-GPU effective batch of 32.")
    seed_everything(cfg["seed"])
    prepared = Path(cfg["prepared_dir"])
    meta = read_json(prepared / "metadata.json")
    if adapter_run:
        # Pin the backbone used by saved adapters and all recovery checkpoints.
        cfg["resolved_model_revision"] = meta["model_revision"]
    basis_info = {}
    if protected_run:
        basis_directory, basis_info = ensure_basis_cache(cfg, meta["model_revision"], args.resume)
        cfg["resolved_basis_sha256"] = basis_info["basis_sha256"]
    metadata_hash = sha256_file(prepared / "metadata.json")
    for relative, expected in meta["file_sha256"].items():
        if sha256_file(prepared / relative) != expected:
            raise ValueError(f"Prepared data/tokenizer changed: {relative}")
    for key in ("model_id", "num_examples", "max_length", "seed"):
        if meta[key] != cfg[key]:
            raise ValueError(f"Prepared data disagree with configuration: {key}")
    dataset = load_from_disk(str(prepared / "dataset"))
    if len(dataset) != cfg["num_examples"]:
        raise ValueError("Prepared dataset must have exactly 40,000 examples.")
    results = Path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    log_path = results / "training.jsonl"
    checkpoint_log = results / "checkpoints.jsonl"
    resume = None
    if args.resume:
        # Never load untrusted trainer_state.pt files.
        resume = torch.load(args.resume / "trainer_state.pt", map_location="cpu", weights_only=False)
        if resume["config"] != cfg or resume["prepared_metadata_sha256"] != metadata_hash:
            raise ValueError("Resume configuration or data differ from saved run.")
        saved_update = resume["progress"]["update"]
        if saved_update >= cfg["num_updates"]:
            raise ValueError("This run is already complete; no training remains to resume.")
        for existing in Path(cfg["checkpoint_dir"]).glob("update_*"):
            suffix = existing.name.removeprefix("update_")
            if suffix.isdigit() and int(suffix) > saved_update:
                raise ValueError("A newer checkpoint exists; resume the latest saved update.")
    elif log_path.exists() or (Path(cfg["checkpoint_dir"]).exists() and any(Path(cfg["checkpoint_dir"]).iterdir())):
        raise FileExistsError("Run already has outputs. Use --resume or new output directories.")
    tokenizer = AutoTokenizer.from_pretrained(prepared / "tokenizer", use_fast=True)
    # Always construct theta0 from the ORIGINAL revision, even when resuming.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=meta["model_revision"],
        torch_dtype=torch.float16 if adapter_run else torch.float32,
        attn_implementation=cfg["attention_implementation"],
    )
    reference_started = time.perf_counter()
    reference = {} if adapter_run else {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    reference_setup_seconds = time.perf_counter() - reference_started
    adapter_setup_started = time.perf_counter()
    if adapter_run:
        # An explicit seed makes LoRA-16 factors reproducible for OPLoRA later.
        torch.manual_seed(cfg["seed"])
        targets = inject_lora(model, cfg["lora_rank"], cfg["lora_alpha"],
                              cfg["lora_dropout"], cfg["lora_targets"])
        if protected_run:
            install_protection(model, basis_directory / "bases.safetensors")
            model._oplora_basis_directory = str(basis_directory)
        if resume:
            if protected_run:
                load_oplora_checkpoint(model, args.resume, cfg)
            else:
                load_adapter(model, args.resume, cfg)
    elif resume:
        from transformers.modeling_utils import load_sharded_checkpoint
        if (args.resume / "model.safetensors.index.json").exists():
            load_sharded_checkpoint(model, args.resume, strict=True, prefer_safe=True)
        else:
            from safetensors.torch import load_model
            load_model(model, str(args.resume / "model.safetensors"), strict=True)
    model.to("cuda")
    if not adapter_run:
        model.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if adapter_run:
        adapter_count = validate_lora_parameters(model)
        write_json(results / "adapter_parameters.json", {
            "rank": cfg["lora_rank"], "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"], "target_modules": targets,
            "target_count": len(targets), "adapter_parameters": adapter_count,
            "frozen_backbone_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "initialization": "Kaiming-uniform A, zero B; no adapter biases",
        })
    elif any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Full FT must keep FP32 master parameters.")
    adapter_setup_seconds = time.perf_counter() - adapter_setup_started if adapter_run else 0.0
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda")
    progress = {"update": 0, "examples_seen": 0, "skipped_attempts": 0,
                "training_seconds": 0.0, "diagnostic_seconds": 0.0,
                "checkpoint_seconds": 0.0, "peak_gpu_allocated_bytes": 0,
                "peak_gpu_reserved_bytes": 0}
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        progress = resume["progress"]
        # The checkpoint cannot include the duration of writing itself; its
        # companion summary supplies that duration when continuing the run.
        saved_summary = read_json(args.resume / "checkpoint_summary.json")
        progress["checkpoint_seconds"] += saved_summary["save_seconds"]
        progress["export_seconds"] = progress.get("export_seconds", 0) + saved_summary.get("export_seconds", 0)
        progress["export_bytes"] = progress.get("export_bytes", 0) + saved_summary.get("export_bytes", 0)
        if (progress["examples_seen"] != progress["update"] * 32 or
                resume["scheduler"]["completed_updates"] != progress["update"]):
            raise ValueError("Checkpoint data position/scheduler mismatch.")
        # Keep abandoned post-checkpoint records for audit, then continue a clean log.
        stamp = time.time_ns()
        for path in (log_path, checkpoint_log):
            if path.exists():
                records = [json.loads(line) for line in path.read_text().splitlines()]
                path.rename(path.with_name(f"{path.stem}.before_resume_{stamp}.jsonl"))
                for record in records:
                    if record["update"] <= progress["update"]:
                        append_jsonl(path, record)
        saved_checkpoint_records = ([json.loads(line) for line in checkpoint_log.read_text().splitlines()]
                                    if checkpoint_log.exists() else [])
        if not any(row["update"] == progress["update"] for row in saved_checkpoint_records):
            append_jsonl(checkpoint_log, {**saved_summary, "path": str(args.resume)})
        restore_rng(resume["rng"])
        del resume
    # setdefault also permits resuming an existing unregularized checkpoint.
    for key in ("regularization_seconds", "reference_transfer_seconds", "reference_transfer_bytes",
                "largest_reference_gpu_chunk_bytes", "reference_setup_seconds", "adapter_setup_seconds"):
        progress.setdefault(key, 0)
    progress["reference_setup_seconds"] += reference_setup_seconds
    progress["adapter_setup_seconds"] += adapter_setup_seconds
    for key in ("basis_cache_setup_seconds", "export_seconds", "export_bytes"):
        progress.setdefault(key, 0)
    if protected_run:
        progress["basis_cache_setup_seconds"] += basis_info["cache_setup_seconds"]
        progress["peak_gpu_allocated_bytes"] = max(
            progress["peak_gpu_allocated_bytes"], basis_info.get("preprocessing_peak_gpu_allocated_bytes", 0),
        )
        write_json(results / "basis_resources.json", basis_info)
    write_json(results / "run_config.json", {**cfg, "resolved_model_revision": meta["model_revision"],
                                            "prepared_metadata_sha256": metadata_hash})
    write_json(results / "environment.json", environment_info())
    torch.cuda.reset_peak_memory_stats()
    effective_batch = cfg["microbatch_size"] * cfg["gradient_accumulation_steps"]
    for update in range(progress["update"] + 1, cfg["num_updates"] + 1):
        start_index = progress["examples_seen"]
        rows = [dataset[i] for i in range(start_index, start_index + effective_batch)]
        denominator = response_token_count(rows)
        if denominator <= 0:
            raise ValueError("Effective batch has no supervised response tokens.")
        lr = lr_for_update(update, cfg["warmup_updates"], cfg["num_updates"], cfg["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        attempt_rng = capture_rng()
        for attempt in range(cfg["max_overflow_retries"] + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            torch.cuda.synchronize()
            started = time.perf_counter()
            for offset in range(0, effective_batch, cfg["microbatch_size"]):
                batch = collate_examples(rows[offset:offset + cfg["microbatch_size"]], tokenizer.pad_token_id)
                batch = {key: value.to("cuda") for key, value in batch.items()}
                labels = batch.pop("labels")
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(**batch)
                    numerator = response_loss_sum(output.logits, labels)
                    loss = numerator / denominator
                scaler.scale(loss).backward()
                loss_sum += numerator.detach().item()
                del output, numerator, loss, labels, batch
            regularization = None
            if cfg["l2_sp_lambda"]:
                # Add lambda*(theta-theta0) ONCE per effective batch in the
                # same scaled units as SFT, before unscale/overflow check/clip.
                regularization = add_l2_sp_gradients(
                    model, reference, cfg["l2_sp_lambda"], scaler.get_scale(),
                )
                for key in ("regularization_seconds", "reference_transfer_seconds", "reference_transfer_bytes"):
                    progress[key] += regularization[key]
                progress["largest_reference_gpu_chunk_bytes"] = max(
                    progress["largest_reference_gpu_chunk_bytes"],
                    regularization["largest_reference_gpu_chunk_bytes"],
                )
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            progress["training_seconds"] += elapsed
            if skipped:
                progress["skipped_attempts"] += 1
            record = {
                "update": update, "attempt": attempt + 1, "skipped": skipped,
                "examples_seen": start_index if skipped else start_index + effective_batch,
                "sft_loss": loss_sum / denominator if math.isfinite(loss_sum) else None,
                "learning_rate": lr, "response_tokens": denominator,
                "gradient_norm_before_clip": grad_norm.item() if torch.isfinite(grad_norm) else None,
                "scaler_before": previous_scale, "scaler_after": scaler.get_scale(),
                "attempt_seconds": elapsed, "training_seconds": progress["training_seconds"],
            }
            if regularization:
                total_loss = loss_sum / denominator + regularization["lambda_omega"]
                record.update(regularization)
                record["total_loss"] = total_loss if math.isfinite(total_loss) else None
                record["l2_sp_lambda"] = cfg["l2_sp_lambda"]
            append_jsonl(log_path, record)
            print(json.dumps(record), flush=True)
            if not skipped:
                break
            # Retry the same examples with a lower scale; do not advance data/LR.
            restore_rng(attempt_rng)
        else:
            raise RuntimeError("Too many FP16 overflow retries; inspect training.jsonl.")
        progress["update"] = update
        progress["examples_seen"] += effective_batch
        progress["peak_gpu_allocated_bytes"] = max(progress["peak_gpu_allocated_bytes"], torch.cuda.max_memory_allocated())
        progress["peak_gpu_reserved_bytes"] = max(progress["peak_gpu_reserved_bytes"], torch.cuda.max_memory_reserved())
        optimizer.zero_grad(set_to_none=True)
        if update in cfg["displacement_updates"]:
            started = time.perf_counter()
            displacement = parameter_displacement(model, reference)
            progress["diagnostic_seconds"] += time.perf_counter() - started
            write_json(results / f"displacement_{update:04d}.json", {
                "update": update, "examples_seen": progress["examples_seen"],
                "parameter_l2_displacement": displacement, "tied_parameters_counted_once": True,
            })
        if update in cfg["checkpoint_updates"]:
            summary = save_checkpoint(model, tokenizer, optimizer, scaler, cfg, metadata_hash, dict(progress))
            progress["checkpoint_seconds"] += summary["save_seconds"]
            progress["export_seconds"] += summary.get("export_seconds", 0)
            progress["export_bytes"] += summary.get("export_bytes", 0)
            append_jsonl(checkpoint_log, summary)
        # Include projection/export overhead even at the final checkpoint.
        progress["peak_gpu_allocated_bytes"] = max(progress["peak_gpu_allocated_bytes"], torch.cuda.max_memory_allocated())
        progress["peak_gpu_reserved_bytes"] = max(progress["peak_gpu_reserved_bytes"], torch.cuda.max_memory_reserved())
        write_json(results / "resources.json", {
            **progress, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "reference_cpu_bytes": sum(p.numel() * p.element_size() for p in reference.values()),
            "extra_example_presentations_from_overflow": progress["skipped_attempts"] * effective_batch,
            "status": "training_complete" if update == cfg["num_updates"] else "running",
            "evaluation_checkpoints": cfg.get("evaluation_updates", cfg["displacement_updates"]),
            "basis_tensor_bytes": basis_info.get("basis_tensor_bytes", 0),
            "basis_file_bytes": basis_info.get("basis_file_bytes", 0),
            "basis_preprocessing_seconds": basis_info.get("preprocessing_seconds", 0),
        })
    print("Training complete. Required benchmark evaluation remains a separate step.")


if __name__ == "__main__":
    main()
