"""Subtask 6 verification: formulas, prompt-token hooks, and comparisons."""

import pytest
import torch

from task1_finetuning.analyze import PAIRS, contrast_table, score_table
from task1_finetuning.diagnostics import activation_ratios, cache_prompt_inputs, last_nonpadding_indices, weight_metrics
from task1_finetuning.evaluate import TASKS
from task1_finetuning.inference import CONDITIONS
from test_full_ft import tiny_model
from test_evaluation import toy_tokenizer


def test_weight_metrics_known_subspace_and_zero_update():
    original = torch.diag(torch.tensor([10.0, 2.0]))
    u = torch.tensor([[1.0], [0.0]])
    delta = torch.tensor([[3.0, 0.0], [4.0, 0.0]])
    result = weight_metrics(original, delta, u)
    assert result["rho_k"] == pytest.approx(9 / 25)
    assert result["eta"] == pytest.approx(5 / (104 ** 0.5))
    assert weight_metrics(original, torch.zeros_like(delta), u)["rho_k"] is None
    assert weight_metrics(original, torch.zeros_like(delta), u)["eta"] == 0


def test_activation_ratio_is_local_update_on_identical_inputs():
    original = torch.diag(torch.tensor([10.0, 2.0]))
    delta = torch.diag(torch.tensor([0.0, 2.0]))
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    torch.testing.assert_close(activation_ratios(original, delta, z), torch.tensor([0.0, 1.0, 0.0]))


def test_local_updates_use_full_fp32_weights_or_projected_oplora_factors(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from common.experiment import write_json
    from task1_finetuning.diagnostics import local_update
    original = torch.ones(2, 3)
    name = "model.layers.0.mlp.down_proj"
    save_file({name + ".weight": original + 7}, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr("task1_finetuning.diagnostics.condition_source", lambda *args: (tmp_path, {"method": "full_ft"}))
    torch.testing.assert_close(local_update("full_ft_final", {}, 0, original), torch.full((2, 3), 7.0))
    inference = tmp_path / "inference_adapter"
    inference.mkdir()
    a, b = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[4.0], [5.0]])
    save_file({name + ".lora_A.weight": a, name + ".lora_B.weight": b}, str(inference / "adapter_model.safetensors"))
    write_json(inference / "adapter_config.json", {"format": "atml_lora_v1", "projected_factors": True, "alpha": 2, "rank": 1})
    monkeypatch.setattr("task1_finetuning.diagnostics.condition_source", lambda *args: (tmp_path, {"method": "oplora"}))
    torch.testing.assert_close(local_update("oplora_final", {}, 0, original), 2 * b @ a)


def test_last_prompt_token_handles_left_and_right_padding():
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]])
    assert last_nonpadding_indices(mask).tolist() == [3, 1, 3]
    with pytest.raises(ValueError):
        last_nonpadding_indices(torch.zeros(1, 3))


def test_hooks_cache_decoder_inputs_without_generating_answers():
    torch.set_num_threads(1)
    model = tiny_model().eval()
    tokenizer = toy_tokenizer()
    prompts = ["Q", "Q A B"]
    actual = cache_prompt_inputs(model, tokenizer, prompts, [0], "cpu")["0"]
    expected = []
    handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
        lambda module, inputs: expected.append(inputs[0][:, -1].detach()))
    try:
        for prompt in prompts:
            model.model(**tokenizer(prompt, return_tensors="pt", add_special_tokens=False), use_cache=False)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    assert len(model.model.layers[0].mlp.down_proj._forward_pre_hooks) == 0


def test_percentage_points_and_all_controlled_comparisons():
    aggregates = {condition: {task: {"primary": 0.2 + 0.01 * i,
                                   "metrics": {"acc,none": 0.1, "acc_norm,none": 0.2}}
                              for task in TASKS} for i, condition in enumerate(CONDITIONS)}
    rows = score_table(aggregates)
    assert len(rows) == 28
    final = next(r for r in rows if r["condition"] == "full_ft_final" and r["benchmark"] == "gsm8k_cot")
    assert final["change_from_baseline_pp"] == pytest.approx(2.0)
    assert len(contrast_table(aggregates)) == len(PAIRS) * 4 == 20


def test_analysis_writes_tables_and_all_figures_from_saved_results(tmp_path, monkeypatch):
    import json
    from common.experiment import write_json
    from task1_finetuning.analyze import RUNS, main
    cfg = {"results_dir": str(tmp_path / "evaluation"), "figures_dir": str(tmp_path / "figures"),
           "diagnostics_dir": str(tmp_path / "diagnostics"), "diagnostic_layers": [0, 13, 27]}
    for condition in CONDITIONS:
        write_json(tmp_path / "evaluation" / condition / "aggregate.json", {
            task: {"primary": 0.4, "metrics": {"acc,none": 0.3, "acc_norm,none": 0.4}} for task in TASKS})
        if condition != "baseline":
            write_json(tmp_path / "diagnostics" / f"{condition}.json", {
                "weights": [{"condition": condition, "layer": layer, "rho_k": 0.1, "eta": 0.01} for layer in cfg["diagnostic_layers"]],
                "activations": [{"condition": condition, "layer": layer, "suite": suite, "R": 0.1, "split_index": index}
                                for layer in cfg["diagnostic_layers"] for suite in ("gsm8k", "hellaswag") for index in range(32)]})
    for run in RUNS:
        root, checkpoints = tmp_path / "training" / run, tmp_path / "checkpoints" / run
        write_json(tmp_path / "configs" / "task1" / f"{run}.json", {"results_dir": str(root), "checkpoint_dir": str(checkpoints)})
        write_json(root / "resources.json", {"trainable_parameters": 100, "training_seconds": 1,
                                             "peak_gpu_allocated_bytes": 1024, "checkpoint_seconds": 0.1, "reference_cpu_bytes": 0})
        write_json(checkpoints / "update_1250" / "checkpoint_summary.json", {"model_bytes": 100, "recovery_bytes_excluding_summary": 300})
        with (root / "training.jsonl").open("w") as stream:
            for update in range(1, 1251):
                stream.write(json.dumps({"update": update, "skipped": False, "sft_loss": 1 / update,
                                         "lambda_omega": 0.1, "total_loss": 1 / update + 0.1}) + "\n")
        for update in ([625, 1250] if run == "full_ft" else [1250]):
            write_json(root / f"displacement_{update:04d}.json", {"update": update, "parameter_l2_displacement": 0.1})
    write_json(tmp_path / "evaluation.json", cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["analyze", "--config", str(tmp_path / "evaluation.json")])
    main()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 4
    assert (tmp_path / "results/task1/analysis/benchmark_scores.csv").exists()
    assert (tmp_path / "results/task1/analysis/activation_summary.csv").exists()
