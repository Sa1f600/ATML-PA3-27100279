"""Run two isolated real GPU updates before starting the Task 2 curriculum."""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.experiment import read_json, seed_everything, write_json
from task2_vlm.data import read_json as read_manifest
from task2_vlm.model import Connector, VLM, install_lora
from task2_vlm.train import CurriculumSampler, calibrate_connector, collate


def train_step(vlm, rows, tokenizer, cfg, optimizer, scaler, required_parameter=None):
    """Perform one production-shaped effective-batch update and validate it."""
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    for retry in range(9):
        before = vlm.connector.layers[0].weight.detach().clone()
        adapter_before = required_parameter.detach().clone() if required_parameter is not None else None
        optimizer.zero_grad(set_to_none=True)
        batch = collate(rows, tokenizer, cfg["feature_dir"], "cuda")
        with torch.autocast("cuda", torch.float16):
            loss = vlm(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in parameters)
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            new_scale = max(1.0, scaler.get_scale() / 2)
            scaler.update(new_scale=new_scale)
            print(f"  Gradient overflow; retrying this update with AMP scale {new_scale:.0f}...", flush=True)
            continue
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        connector_changed = not torch.equal(before, vlm.connector.layers[0].weight.detach())
        if not connector_changed:
            raise RuntimeError("The connector did not change after the smoke-test optimizer step.")
        if required_parameter is not None and torch.equal(adapter_before, required_parameter.detach()):
            raise RuntimeError("A LoRA adapter did not change after the smoke-test optimizer step.")
        return {"loss": float(loss.detach()), "gradient_norm": gradient_norm, "examples": len(rows), "overflow_retries": retry}
    raise RuntimeError("Smoke-test gradients remained non-finite after eight AMP-scale reductions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task2/vlm_196.json")
    parser.add_argument("--output", type=Path, required=True, help="Standalone JSON result; no training checkpoint is written.")
    args = parser.parse_args()
    cfg = read_json(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 smoke test requires a CUDA runtime.")

    seed_everything(cfg["seed"])
    manifest = read_manifest(Path(cfg["prepared_dir"]) / "manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading frozen Qwen for the isolated GPU smoke test...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float16).cuda()
    base.gradient_checkpointing_enable()
    base.requires_grad_(False)
    connector = Connector(cfg["feature_dim"], cfg["hidden_dim"]).cuda()
    vlm = VLM(base, connector).train()
    # The Qwen FP16 backward pass can overflow at PyTorch's default 65,536
    # initial scale on a T4.  This lower scale remains dynamic and matches the
    # production trainer.
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    calibration = calibrate_connector(connector, base, manifest, cfg)
    print(f"Calibrated connector output scale to {calibration['output_scale']:.6g}.", flush=True)

    # Stage 1: connector-only caption update.
    caption_sampler = CurriculumSampler(manifest, "caption", cfg["seed"])
    caption_rows = caption_sampler.next_examples(cfg["microbatch_size"])
    caption_optimizer = torch.optim.AdamW(connector.parameters(), lr=cfg["stages"]["caption"]["connector_lr"], weight_decay=cfg["weight_decay"])
    print("Running one connector-only caption update...", flush=True)
    caption_result = train_step(vlm, caption_rows, tokenizer, cfg, caption_optimizer, scaler)

    # Stage 2: install the specified LoRA adapters and run one grounding update.
    targets = install_lora(base, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"])
    lora_parameters = [parameter for parameter in base.parameters() if parameter.requires_grad]
    if not lora_parameters:
        raise RuntimeError("LoRA installation created no trainable adapter parameters.")
    # LoRA B starts at zero, so it is the adapter factor guaranteed to receive
    # a non-zero first-update gradient while LoRA A is still masked by B.
    lora_probe = next(parameter for name, parameter in base.named_parameters() if name.endswith(".b.weight"))
    grounding_sampler = CurriculumSampler(manifest, "grounding", cfg["seed"] + 1)
    grounding_rows = grounding_sampler.next_examples(cfg["microbatch_size"])
    grounding_optimizer = torch.optim.AdamW(
        [
            {"params": list(connector.parameters()), "lr": cfg["stages"]["grounding"]["connector_lr"]},
            {"params": lora_parameters, "lr": cfg["stages"]["grounding"]["lora_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    print(f"Running one connector + LoRA grounding update ({len(targets)} target modules)...", flush=True)
    grounding_result = train_step(vlm, grounding_rows, tokenizer, cfg, grounding_optimizer, scaler, lora_probe)

    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "caption_update": caption_result,
        "grounding_update": grounding_result,
        "connector_calibration": calibration,
        "lora_target_module_count": len(targets),
        "trainable_connector_parameters": sum(parameter.numel() for parameter in connector.parameters()),
        "trainable_lora_parameters": sum(parameter.numel() for parameter in lora_parameters),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.perf_counter() - started,
    }
    write_json(args.output, result)
    print(f"PASS Task 2 GPU smoke test; result saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
