"""Validate L2-SP against direct autograd, including scaled accumulation."""

from copy import deepcopy
import json

import pytest
import torch

from task1_finetuning.data import collate_examples, response_loss_sum, response_token_count
from task1_finetuning.regularization import add_l2_sp_gradients
from task1_finetuning.train_full_ft import validate_config
from test_full_ft import tiny_model, examples, optimizer_config


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(6304)


def test_l2_sp_includes_bias_norm_and_tied_parameters_exactly_once():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            self.norm = torch.nn.LayerNorm(1)
            self.alias = self.linear  # same underlying weights registered twice

    model = Model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    reference = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[3.0, 4.0]]))
        model.linear.bias.fill_(12.0)
        model.norm.weight.fill_(5.0)
        model.norm.bias.fill_(2.0)
    stats = add_l2_sp_gradients(model, reference, 1e-3, 8.0, chunk_elements=1)
    assert stats["omega"] == pytest.approx((9 + 16 + 144 + 25 + 4) / 2)
    assert stats["lambda_omega"] == pytest.approx(0.099)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, 8 * 1e-3 * (p - reference[name]))
        assert torch.count_nonzero(reference[name]).item() == 0
        assert reference[name].grad is None and not reference[name].requires_grad
    assert stats["reference_transfer_bytes"] == 0  # CPU-only numerical test
    assert stats["reference_transfer_seconds"] == 0


def test_scaled_accumulation_plus_penalty_matches_direct_total_objective():
    model = tiny_model().train()
    reference = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    explicit = deepcopy(model)
    scale, strength = 16.0, 1e-3
    rows = examples()
    denominator = response_token_count(rows)
    batch = collate_examples(rows, 0)
    labels = batch.pop("labels")
    sft = response_loss_sum(explicit(**batch).logits, labels) / denominator
    omega = 0.5 * sum((p - reference[name]).square().sum() for name, p in explicit.named_parameters())
    (scale * (sft + strength * omega)).backward()
    for row in rows:
        batch = collate_examples([row], 0)
        labels = batch.pop("labels")
        (scale * response_loss_sum(model(**batch).logits, labels) / denominator).backward()
    stats = add_l2_sp_gradients(model, reference, strength, scale, chunk_elements=31)
    assert stats["omega"] == pytest.approx(omega.item(), rel=2e-7)
    # Dividing simulates AMP unscale. Clip the complete combined gradient.
    for p, q in zip(model.parameters(), explicit.parameters()):
        p.grad.div_(scale)
        q.grad.div_(scale)
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-7)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(explicit.parameters(), 1.0)
    from task1_finetuning.train_full_ft import make_optimizer
    make_optimizer(model, optimizer_config()).step()
    make_optimizer(explicit, optimizer_config()).step()
    for p, q in zip(model.parameters(), explicit.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)


def test_zero_starting_penalty_and_frozen_reference_after_optimizer_updates():
    model = torch.nn.Linear(2, 1)
    reference = {name: p.detach().clone() for name, p in model.named_parameters()}
    unchanged = deepcopy(reference)
    stats = add_l2_sp_gradients(model, reference, 1e-3, 1.0)
    assert stats["omega"] == 0
    assert all(torch.count_nonzero(p.grad).item() == 0 for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    stats = add_l2_sp_gradients(model, reference, 1e-3, 1.0)
    assert stats["omega"] > 0
    for name in reference:
        torch.testing.assert_close(reference[name], unchanged[name], rtol=0, atol=0)


def test_l2_sp_config_matches_full_ft_except_required_differences():
    with open("configs/task1/full_ft.json") as stream:
        vanilla = json.load(stream)
    with open("configs/task1/full_ft_l2_sp.json") as stream:
        regularized = json.load(stream)
    validate_config(regularized)
    differing = {key for key in set(vanilla) | set(regularized) if vanilla.get(key) != regularized.get(key)}
    assert differing == {"method", "l2_sp_lambda", "displacement_updates", "checkpoint_dir", "results_dir",
                         "checkpoint_retention"}
    assert regularized["checkpoint_retention"] == 1
    assert regularized["displacement_updates"] == [1250]
    regularized["l2_sp_lambda"] = 1e-4
    with pytest.raises(ValueError):
        validate_config(regularized)


def test_l2_sp_rejects_nonfrozen_reference():
    model = torch.nn.Linear(2, 1)
    invalid = {name: p.clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="frozen CPU reference"):
        add_l2_sp_gradients(model, invalid, 1e-3, 1.0)
