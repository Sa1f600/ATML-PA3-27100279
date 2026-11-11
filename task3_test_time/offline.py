"""Offline Task 3 analysis for the supplied 32-rollout exports.

Usage:
  python -m task3_test_time.offline --config configs/task3/offline.json

The input layout is configurable because the course data bundle is intentionally
not committed to Git.  Outputs are atomic JSON/CSV files and can be rerun safely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import Candidate, NO_ANSWER, adaptive_stop, attach_visconf, choose, is_correct, normalise_answer


KEY_ALIASES = {
    "sample_id": ("sample_id", "question_id", "id"),
    "rollout_id": ("rollout_id", "rollout", "candidate_id"),
    "answer": ("extracted_answer", "answer", "final_answer", "prediction"),
    "gold": ("gold_answer", "reference_answer", "target", "answer_gold"),
    "text": ("text", "generation", "response", "output"),
    "length": ("generated_token_count", "token_count", "length", "generated_tokens"),
    "logp": ("logp", "token_logp", "sampled_token_logp"),
    "kl": ("kl_u_p", "token_kl_u_p", "kl"),
    "image": ("image_attention", "attention_image", "image"),
    "prompt": ("prompt_attention", "attention_prompt", "prompt"),
    "generated": ("generated_attention", "attention_generated", "generated"),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    temporary.replace(path)


def find_column(frame: pd.DataFrame, key: str, required: bool = True) -> str | None:
    for column in KEY_ALIASES[key]:
        if column in frame.columns:
            return column
    if required:
        raise ValueError(f"No {key!r} column. Expected one of {KEY_ALIASES[key]}; found {list(frame.columns)}")
    return None


def mean_value(value: Any) -> float:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        numbers = [mean_value(part) for part in value]
        return float(np.mean(numbers)) if numbers else 0.0
    try:
        number = float(value)
        return number if np.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def aggregate_metrics(frame: pd.DataFrame, key_columns: list[str], source_column: str, output_column: str) -> pd.DataFrame:
    copied = frame[key_columns + [source_column]].copy()
    copied[output_column] = copied[source_column].map(mean_value)
    return copied.groupby(key_columns, as_index=False)[output_column].mean()


def load_source(spec: dict[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one LLM/VLM export and return normalized rollout rows."""
    resolve = lambda item: (root / item).resolve() if not Path(item).is_absolute() else Path(item)
    generations_path = resolve(spec["generations"])
    probability_path = resolve(spec["probability_metrics"])
    generations, probability = pd.read_parquet(generations_path), pd.read_parquet(probability_path)
    sample = find_column(generations, "sample_id")
    rollout = find_column(generations, "rollout_id")
    answer = find_column(generations, "answer")
    gold = find_column(generations, "gold")
    length = find_column(generations, "length", required=False)
    key_columns = [sample, rollout]
    psample, prollout = find_column(probability, "sample_id"), find_column(probability, "rollout_id")
    logp, kl = find_column(probability, "logp"), find_column(probability, "kl")
    probability = probability.rename(columns={psample: sample, prollout: rollout})
    metrics = aggregate_metrics(probability, key_columns, logp, "mean_logp").merge(
        aggregate_metrics(probability, key_columns, kl, "self_certainty"), on=key_columns, how="outer"
    )
    output = generations.merge(metrics, on=key_columns, how="left", validate="one_to_one")
    output = output.rename(columns={sample: "sample_id", rollout: "rollout_id", answer: "extracted_answer", gold: "gold"})
    output["token_count"] = output[length].map(mean_value).round().astype(int) if length else 0
    if not length or (output["token_count"] <= 0).all():
        text_column = find_column(generations, "text", required=False)
        output["token_count"] = output[text_column].fillna("").map(lambda text: max(1, len(str(text).split()))) if text_column else 0
    output["mean_logp"] = output["mean_logp"].fillna(0.0)
    output["self_certainty"] = output["self_certainty"].fillna(0.0)
    output["vision"] = 0.0
    attention_path = spec.get("attention_metrics")
    if attention_path:
        attention = pd.read_parquet(resolve(attention_path))
        asample, arollout = find_column(attention, "sample_id"), find_column(attention, "rollout_id")
        image, prompt, generated = (find_column(attention, name) for name in ("image", "prompt", "generated"))
        attention = attention.rename(columns={asample: "sample_id", arollout: "rollout_id"})
        image_values, prompt_values, generated_values = (attention[column].map(mean_value) for column in (image, prompt, generated))
        attention["vision"] = image_values / (image_values + prompt_values + generated_values).clip(lower=1e-8)
        attention = attention.groupby(["sample_id", "rollout_id"], as_index=False)["vision"].mean()
        output = output.drop(columns=["vision"]).merge(attention, on=["sample_id", "rollout_id"], how="left")
        output["vision"] = output["vision"].fillna(0.0)
    output["dataset"] = spec["dataset"]
    output["multiple_choice"] = bool(spec.get("multiple_choice", False))
    output["answer_group"] = [normalise_answer(a, multiple_choice=mc) for a, mc in zip(output.extracted_answer, output.multiple_choice)]
    if output.duplicated(["sample_id", "rollout_id"]).any():
        raise ValueError(f"Duplicate (sample_id, rollout_id) in {generations_path}")
    counts = output.groupby("sample_id").rollout_id.nunique()
    if not (counts == 32).all():
        bad = counts[counts != 32].head().to_dict()
        raise ValueError(f"Every question must have 32 rollouts; examples: {bad}")
    return output, {"name": spec["name"], "rows": len(output), "questions": int(counts.size)}


def build_manifest(rows: pd.DataFrame, output_dir: Path, seed: int, orders: int) -> dict[str, Any]:
    question_ids = sorted(rows.sample_id.astype(str).unique())
    if len(question_ids) != 1000:
        raise ValueError(f"Expected exactly 1,000 questions for {rows.dataset.iloc[0]}, found {len(question_ids)}")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(question_ids))
    dev, test = shuffled[:200], shuffled[200:]
    orders_map: dict[str, list[list[str]]] = {}
    for sample_id, group in rows.groupby(rows.sample_id.astype(str), sort=False):
        rollout_ids = sorted(group.rollout_id.astype(str).tolist())
        local_rng = np.random.default_rng(seed + sum(ord(character) for character in sample_id))
        orders_map[sample_id] = [list(local_rng.permutation(rollout_ids)) for _ in range(orders)]
    manifest = {"seed": seed, "dev_ids": dev, "test_ids": test, "orders": orders_map}
    write_json(output_dir / f"{rows.dataset.iloc[0]}_split_and_orders.json", manifest)
    return manifest


def candidates_for_order(group: pd.DataFrame, order: list[str]) -> list[Candidate]:
    indexed = group.set_index(group.rollout_id.astype(str))
    candidates = [
        Candidate(
            rollout_id=rollout_id,
            answer=str(indexed.loc[rollout_id, "answer_group"]),
            token_count=int(indexed.loc[rollout_id, "token_count"]),
            mean_logp=float(indexed.loc[rollout_id, "mean_logp"]),
            self_certainty=float(indexed.loc[rollout_id, "self_certainty"]),
            vision=float(indexed.loc[rollout_id, "vision"]),
        )
        for rollout_id in order
    ]
    return attach_visconf(candidates)


def method_specs(is_vlm: bool) -> list[tuple[str, str, str | None]]:
    specs = [
        ("firstans", "first", None),
        ("self_consistency", "self_consistency", None),
        ("mean_logp", "highest_score", "mean_logp"),
        ("self_certainty", "highest_score", "self_certainty"),
        ("self_certainty_rwc", "rwc", "self_certainty"),
    ]
    if is_vlm:
        specs.extend([
            ("vision_only", "highest_score", "vision"),
            ("vision_only_rwc", "rwc", "vision"),
            ("visconf", "highest_score", "visconf"),
            ("visconf_rwc", "rwc", "visconf"),
        ])
    return specs


def evaluate_fixed(rows: pd.DataFrame, manifest: dict[str, Any], is_vlm: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []
    for sample_id, group in rows.groupby(rows.sample_id.astype(str), sort=False):
        gold, multiple_choice = group.gold.iloc[0], bool(group.multiple_choice.iloc[0])
        for order_index, order in enumerate(manifest["orders"][sample_id]):
            candidates = candidates_for_order(group, order)
            for n in (1, 2, 4, 8, 16, 32):
                prefix = candidates[:n]
                oracle = any(is_correct(candidate.answer, gold, multiple_choice=multiple_choice) for candidate in prefix)
                distinct = len({candidate.answer for candidate in prefix})
                for name, aggregation, score in method_specs(is_vlm):
                    selected = choose(prefix, aggregation, score)
                    records.append({
                        "sample_id": sample_id, "order": order_index, "budget": n, "method": name,
                        "selected_answer": selected, "correct": is_correct(selected, gold, multiple_choice=multiple_choice),
                        "oracle": oracle, "token_cost": sum(candidate.token_count for candidate in (prefix[:1] if name == "firstans" else prefix)),
                        "distinct_answers": distinct, "correct_available_but_missed": oracle and not is_correct(selected, gold, multiple_choice=multiple_choice),
                    })
            if len(candidates) == 32:
                for candidate in candidates:
                    signal_records.append({"sample_id": sample_id, "order": order_index, "rollout_id": candidate.rollout_id,
                                           "length": candidate.token_count, "mean_logp": candidate.mean_logp,
                                           "self_certainty": candidate.self_certainty, "vision": candidate.vision,
                                           "visconf": candidate.visconf,
                                           "correct": is_correct(candidate.answer, gold, multiple_choice=multiple_choice)})
    return pd.DataFrame(records), pd.DataFrame(signal_records)


def evaluate_adaptive(rows: pd.DataFrame, manifest: dict[str, Any], is_vlm: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    score_name = "visconf" if is_vlm else "self_certainty"
    policies = [("agreement", tau, None) for tau in (.5, .625, .75, .875, 1.0)]
    policies += [("quality_aware", tau, gamma) for tau in (.5, .625, .75, .875, 1.0) for gamma in (0, .1, .25, .5)]
    policies.append(("fixed_32", 1.0, None))
    for sample_id, group in rows.groupby(rows.sample_id.astype(str), sort=False):
        gold, multiple_choice = group.gold.iloc[0], bool(group.multiple_choice.iloc[0])
        for order_index, order in enumerate(manifest["orders"][sample_id]):
            candidates = candidates_for_order(group, order)
            for family, tau, gamma in policies:
                consumed, selected = (32, choose(candidates, "self_consistency")) if family == "fixed_32" else adaptive_stop(
                    candidates, tau=tau, gamma=gamma, score_name=score_name if gamma is not None else None
                )
                records.append({"sample_id": sample_id, "order": order_index, "family": family, "tau": tau, "gamma": gamma,
                                "consumed": consumed, "token_cost": sum(candidate.token_count for candidate in candidates[:consumed]),
                                "selected_answer": selected, "correct": is_correct(selected, gold, multiple_choice=multiple_choice)})
    return pd.DataFrame(records)


def select_adaptive(dev: pd.DataFrame) -> pd.DataFrame:
    fixed_accuracy = dev[dev.family == "fixed_32"].correct.mean()
    options = dev[dev.family != "fixed_32"].groupby(["family", "tau", "gamma"], dropna=False).agg(
        accuracy=("correct", "mean"), token_cost=("token_cost", "mean"), rollouts=("consumed", "mean")
    ).reset_index()
    feasible = options[options.accuracy >= fixed_accuracy - .01]
    chosen = feasible.sort_values(["token_cost", "accuracy", "tau", "gamma"], ascending=[True, False, True, True]).groupby("family").head(1)
    return chosen


def summarize_fixed(records: pd.DataFrame, split_ids: set[str]) -> pd.DataFrame:
    subset = records[records.sample_id.isin(split_ids)]
    return subset.groupby(["budget", "method"], as_index=False).agg(
        accuracy=("correct", "mean"), oracle_coverage=("oracle", "mean"), token_cost=("token_cost", "mean"),
        distinct_answer_groups=("distinct_answers", "mean"), selection_misses=("correct_available_but_missed", "mean")
    )


def bootstrap_accuracy_differences(records: pd.DataFrame, split_ids: set[str], seed: int) -> pd.DataFrame:
    """Paired question-level 95% CIs against fixed-budget self-consistency."""
    heldout = records[records.sample_id.isin(split_ids)]
    per_question = heldout.groupby(["sample_id", "budget", "method"], as_index=False).correct.mean()
    baseline = per_question[per_question.method == "self_consistency"].rename(columns={"correct": "baseline"})
    rng = np.random.default_rng(seed)
    reports: list[dict[str, Any]] = []
    for (budget, method), current in per_question.groupby(["budget", "method"]):
        if method == "self_consistency":
            continue
        paired = current.merge(baseline[baseline.budget == budget][["sample_id", "baseline"]], on="sample_id", how="inner")
        delta = (paired.correct - paired.baseline).to_numpy(dtype=float)
        if not len(delta):
            continue
        draws = np.mean(rng.choice(delta, size=(1000, len(delta)), replace=True), axis=1)
        reports.append({"budget": budget, "method": method, "accuracy_difference_vs_self_consistency": float(delta.mean()),
                        "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
                        "questions": int(len(delta)), "bootstrap_resamples": 1000})
    return pd.DataFrame(reports)


def signal_summary(signals: pd.DataFrame) -> pd.DataFrame:
    """Within-question signal checks, including the required quartile contrasts."""
    reports: list[dict[str, Any]] = []
    for score in ("mean_logp", "self_certainty", "vision", "visconf"):
        if score not in signals:
            continue
        differences: list[float] = []
        for _, group in signals.groupby(["sample_id", "order"]):
            if len(group) < 4 or group[score].nunique() <= 1:
                continue
            ordered = group.sort_values(score)
            width = max(1, len(ordered) // 4)
            differences.append(float(ordered.tail(width).correct.mean() - ordered.head(width).correct.mean()))
        if differences:
            reports.append({"analysis": "top_minus_bottom_correctness_quartile", "signal": score,
                            "mean_difference": float(np.mean(differences)), "questions_orders": len(differences)})
    shortest = signals.loc[signals.groupby(["sample_id", "order"]).length.idxmin()]
    reports.append({"analysis": "shortest_rollout_control", "signal": "length", "mean_difference": float(shortest.correct.mean()),
                    "questions_orders": int(len(shortest))})
    return pd.DataFrame(reports)


def run(config: dict[str, Any]) -> None:
    root = Path(config["input_root"]).expanduser()
    output = Path(config.get("output_dir", "results/task3/offline")).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    all_summary: list[pd.DataFrame] = []
    for source in config["sources"]:
        print(f"Loading {source['name']} rollout tables...", flush=True)
        rows, details = load_source(source, root)
        print(f"  {details['questions']} questions, {details['rows']} rollouts", flush=True)
        dataset_output = output / source["name"]
        manifest = build_manifest(rows, dataset_output, int(config.get("seed", 6304)), int(config.get("orders", 10)))
        fixed, signals = evaluate_fixed(rows, manifest, bool(source.get("attention_metrics")))
        adaptive = evaluate_adaptive(rows, manifest, bool(source.get("attention_metrics")))
        fixed.to_parquet(dataset_output / "per_question_fixed.parquet", index=False)
        adaptive.to_parquet(dataset_output / "per_question_adaptive.parquet", index=False)
        signals.to_parquet(dataset_output / "signal_records.parquet", index=False)
        dev_ids, test_ids = set(manifest["dev_ids"]), set(manifest["test_ids"])
        dev_summary, test_summary = summarize_fixed(fixed, dev_ids), summarize_fixed(fixed, test_ids)
        dev_summary["split"], test_summary["split"] = "development", "heldout"
        selected = select_adaptive(adaptive[adaptive.sample_id.isin(dev_ids)])
        selected.to_csv(dataset_output / "adaptive_selected_on_development.csv", index=False)
        heldout_adaptive = adaptive[adaptive.sample_id.isin(test_ids)].merge(selected[["family", "tau", "gamma"]], on=["family", "tau", "gamma"], how="inner")
        adaptive_summary = heldout_adaptive.groupby("family", as_index=False).agg(accuracy=("correct", "mean"), token_cost=("token_cost", "mean"), rollouts=("consumed", "mean"))
        dev_summary.to_csv(dataset_output / "fixed_development_summary.csv", index=False)
        test_summary.to_csv(dataset_output / "fixed_heldout_summary.csv", index=False)
        adaptive_summary.to_csv(dataset_output / "adaptive_heldout_summary.csv", index=False)
        bootstrap_accuracy_differences(fixed, test_ids, int(config.get("seed", 6304))).to_csv(
            dataset_output / "fixed_heldout_bootstrap.csv", index=False
        )
        signal_summary(signals).to_csv(dataset_output / "signal_summary.csv", index=False)
        all_summary.extend([dev_summary.assign(source=source["name"]), test_summary.assign(source=source["name"])])
        write_json(dataset_output / "run_metadata.json", {**details, "normalizer": "conservative_builtin_v1", "seed": config.get("seed", 6304), "orders": config.get("orders", 10)})
        print(f"Saved Task 3 offline results: {dataset_output}", flush=True)
    pd.concat(all_summary, ignore_index=True).to_csv(output / "all_fixed_summaries.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3/offline.json")
    args = parser.parse_args()
    run(read_json(Path(args.config)))


if __name__ == "__main__":
    main()
