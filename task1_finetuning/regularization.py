"""L2-SP with a fixed CPU reference and bounded GPU working memory."""

import math
import time

import torch


@torch.no_grad()
def add_l2_sp_gradients(model, reference, strength, gradient_scale, chunk_elements=1048576):
    """Add the exact gradient of strength/2 * sum((theta-theta0)**2).

    Call once per effective batch, AFTER SFT backward and BEFORE AMP unscale.
    Gradients here are scaled by the same factor as SFT, so GradScaler checks
    the combined gradient for overflow before the optimizer step. This avoids
    keeping a full-model penalty autograd graph or a second GPU model copy.

    Unique named_parameters include biases/norms and deduplicate tied weights.
    Neither the penalty nor its gradient is averaged over parameters, tokens,
    or gradient-accumulation steps. The CPU reference is never modified.
    """
    if strength <= 0 or not math.isfinite(strength):
        raise ValueError("L2-SP strength must be positive and finite.")
    if gradient_scale <= 0 or not math.isfinite(gradient_scale):
        raise ValueError("AMP scale must be positive and finite.")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive.")
    parameters = dict(model.named_parameters())
    if parameters.keys() != reference.keys():
        raise ValueError("Reference must contain exactly the unique model parameters.")
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    squared = torch.zeros((), device=device, dtype=torch.float64)
    transfer_events = []
    transfer_bytes = 0
    largest_reference_chunk = 0
    for name, parameter in parameters.items():
        initial = reference[name]
        if parameter.dtype != torch.float32 or initial.dtype != torch.float32:
            raise ValueError("L2-SP requires FP32 master parameters and reference.")
        if initial.device.type != "cpu" or initial.requires_grad or initial.shape != parameter.shape:
            raise ValueError(f"Invalid frozen CPU reference for {name}.")
        if not parameter.requires_grad:
            raise ValueError("Full FT + L2-SP must train every original parameter.")
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        flat = parameter.detach().view(-1)
        gradient = parameter.grad.view(-1)
        original = initial.view(-1)
        for offset in range(0, flat.numel(), chunk_elements):
            stop = offset + chunk_elements
            if cuda:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            reference_chunk = original[offset:stop].to(device=device)
            if cuda:
                end.record()
                transfer_events.append((begin, end))
                size = reference_chunk.numel() * reference_chunk.element_size()
                transfer_bytes += size
                largest_reference_chunk = max(largest_reference_chunk, size)
            delta = flat[offset:stop] - reference_chunk
            # FP32 differences; FP64 reduction provides a stable logged norm.
            squared.add_(delta.double().square().sum())
            gradient[offset:stop].add_(delta, alpha=strength * gradient_scale)
            del reference_chunk, delta
    if cuda:
        torch.cuda.synchronize(device)
    omega = 0.5 * squared.item()
    return {
        "omega": omega,
        "lambda_omega": strength * omega,
        "regularization_seconds": time.perf_counter() - started,
        "reference_transfer_seconds": sum(a.elapsed_time(b) for a, b in transfer_events) / 1000,
        "reference_transfer_bytes": transfer_bytes,
        "largest_reference_gpu_chunk_bytes": largest_reference_chunk,
    }
