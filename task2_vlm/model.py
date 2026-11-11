"""Frozen SigLIP-feature to frozen-Qwen interface with trainable connector/LoRA."""
import math
from torch import nn
import torch


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class Connector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.register_buffer("output_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, features):
        return self.output_scale * self.layers(features.float())


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=.05):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        base.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32, device=base.weight.device)
        self.b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32, device=base.weight.device)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)

    def forward(self, x):
        original = self.base(x)
        update = self.b(self.a(self.dropout(x.to(torch.float32))))
        return original + (self.scale * update).to(original.dtype)


def install_lora(model, rank=16, alpha=32, dropout=.05):
    model.requires_grad_(False)
    found = []
    for name, module in list(model.named_modules()):
        if name.split(".")[-1] in LORA_TARGETS and isinstance(module, nn.Linear):
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, LoRALinear(module, rank, alpha, dropout)); found.append(name)
    if not found:
        raise RuntimeError("No Qwen attention projections found for LoRA.")
    return found


class VLM(nn.Module):
    """Visual tokens are inserted immediately before the tokenized user question."""
    def __init__(self, language_model, connector):
        super().__init__(); self.language_model, self.connector = language_model, connector

    def forward(self, input_ids, attention_mask, position_ids, labels, visual_features=None, visual_positions=None):
        embeds = self.language_model.get_input_embeddings()(input_ids)
        if visual_features is None:
            return self.language_model(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
        rows, masks, positions, labels_out = [], [], [], []
        visual = self.connector(visual_features).to(embeds.dtype)
        for i in range(len(embeds)):
            at = int(visual_positions[i]); n = visual.shape[1]
            rows.append(torch.cat((embeds[i, :at], visual[i], embeds[i, at:]), 0))
            masks.append(torch.cat((attention_mask[i, :at], torch.ones(n, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask[i, at:]), 0))
            labels_out.append(torch.cat((labels[i, :at], torch.full((n,), -100, device=labels.device, dtype=labels.dtype), labels[i, at:]), 0))
            positions.append(torch.arange(rows[-1].shape[0], device=embeds.device))
        max_len = max(row.shape[0] for row in rows)
        def pad(items, value): return torch.stack([nn.functional.pad(x, (0, max_len - x.shape[0]), value=value) for x in items])
        return self.language_model(inputs_embeds=pad(rows, 0), attention_mask=pad(masks, 0), position_ids=pad(positions, 0), labels=pad(labels_out, -100))
