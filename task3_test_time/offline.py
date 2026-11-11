"""Offline Task 3 analysis of the supplied 32-rollout exports.

The loader retains generated-token arrays rather than averaging them on read. It
therefore verifies token alignment and recomputes VisConf from each revealed
prefix exactly as required by the assignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .core import (Candidate, SCORER_VERSION, adaptive_stop, attach_visconf,
                   choose, group_answers, answer_equivalent)


ALIASES = {
    "sample_id": ("sample_id", "question_id", "id"),
    "rollout_id": ("rollout_id", "rollout", "candidate_id"),
    "dataset": ("dataset",),
    "answer": ("extracted_answer", "answer", "final_answer", "prediction"),
    "gold": ("gold_answer", "reference_answer", "target", "answer_gold"),
    "length": ("generated_token_count", "token_count", "length", "generated_tokens"),
    "token_ids": ("generated_token_ids", "output_token_ids", "token_ids"),
    "logp": ("logp", "token_logp", "sampled_token_logp"),
    "kl": ("kl_u_p", "token_kl_u_p", "kl"),
    "image": ("image_attention", "attention_image", "image"),
    "prompt": ("prompt_attention", "attention_prompt", "prompt"),
    "generated": ("generated_attention", "attention_generated", "generated"),
    "answer_type": ("answer_type", "question_type", "type"),
    "precision": ("precision", "answer_precision", "decimal_places"),
    "choices": ("choices", "options", "answer_choices"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    temporary.replace(path)


def column(frame: pd.DataFrame, name: str, required: bool = True, configured: dict[str, str] | None = None) -> str | None:
    if configured and name in configured:
        if configured[name] not in frame.columns:
            raise ValueError(f"Configured metadata column {configured[name]!r} for {name!r} does not exist.")
        return configured[name]
    result = next((candidate for candidate in ALIASES[name] if candidate in frame.columns), None)
    if result is None and required:
        raise ValueError(f"Missing {name!r}; expected one of {ALIASES[name]}, found {list(frame.columns)}")
    return result


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [number for part in value for number in flatten_numbers(part)]
    try:
        number = float(value)
        return [number] if np.isfinite(number) else [0.0]
    except (TypeError, ValueError):
        return []


def sequence_table(frame: pd.DataFrame, keys: list[str], sequence_column: str, output_column: str) -> pd.DataFrame:
    table = frame[keys + [sequence_column]].copy()
    table[output_column] = table[sequence_column].map(flatten_numbers)
    return table.groupby(keys, as_index=False)[output_column].agg(lambda values: [item for value in values for item in value])


def _path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def metadata_for_source(spec: dict[str, Any], root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    metadata_path = spec.get("question_metadata")
    rows["multiple_choice"] = bool(spec.get("multiple_choice", False))
    rows["precision"] = None
    if not metadata_path:
        return rows
    path = _path(root, metadata_path)
    metadata = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path)
    mapping = spec.get("metadata_columns", {})
    source_id = column(metadata, "sample_id", configured=mapping)
    answer_type = column(metadata, "answer_type", required=False, configured=mapping)
    precision = column(metadata, "precision", required=False, configured=mapping)
    choices = column(metadata, "choices", required=False, configured=mapping)
    selected = pd.DataFrame({"sample_id": metadata[source_id].astype(str)})
    selected["multiple_choice"] = (metadata[answer_type].astype(str).str.lower().str.contains("multi|choice|option")
                                   if answer_type else bool(spec.get("multiple_choice", False)))
    selected["precision"] = pd.to_numeric(metadata[precision], errors="coerce") if precision else np.nan
    selected["choice_options"] = metadata[choices].map(parse_choices) if choices else [()] * len(metadata)
    rows = rows.drop(columns=["multiple_choice", "precision"]).merge(selected, on="sample_id", how="left", validate="many_to_one")
    rows["multiple_choice"] = rows["multiple_choice"].fillna(bool(spec.get("multiple_choice", False))).astype(bool)
    return rows


def parse_choices(value: Any) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return tuple(str(item) for item in decoded)
        except json.JSONDecodeError:
            pass
    return ()


def load_source(spec: dict[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if spec["dataset"].lower() == "math":
        try:
            import math_verify  # noqa: F401
        except ImportError as error:
            raise RuntimeError("MATH scoring requires math-verify. Run `pip install math-verify` before Task 3.") from error
    if spec["dataset"].lower() == "mathvista" and not spec.get("question_metadata"):
        raise ValueError("MathVista scoring requires its released question metadata for answer type/choice/precision rules.")
    generations = pd.read_parquet(_path(root, spec["generations"]))
    probability = pd.read_parquet(_path(root, spec["probability_metrics"]))
    sample, rollout = column(generations, "sample_id"), column(generations, "rollout_id")
    answer, gold = column(generations, "answer"), column(generations, "gold")
    generated_length, generated_ids = column(generations, "length", False), column(generations, "token_ids", False)
    keys = [sample, rollout]
    if column(generations, "dataset", False) and column(probability, "dataset", False):
        keys.insert(0, column(generations, "dataset"))
    p_sample, p_rollout = column(probability, "sample_id"), column(probability, "rollout_id")
    renames = {p_sample: sample, p_rollout: rollout}
    if len(keys) == 3:
        renames[column(probability, "dataset")] = keys[0]
    probability = probability.rename(columns=renames)
    logp, kl = column(probability, "logp"), column(probability, "kl")
    metrics = sequence_table(probability, keys, logp, "token_logp").merge(
        sequence_table(probability, keys, kl, "token_kls"), on=keys, validate="one_to_one"
    )
    rows = generations.merge(metrics, on=keys, how="left", validate="one_to_one")
    rows = rows.rename(columns={sample: "sample_id", rollout: "rollout_id", answer: "raw_answer", gold: "gold"})
    rows["sample_id"], rows["rollout_id"] = rows.sample_id.astype(str), rows.rollout_id.astype(str)
    rows["token_logp"] = rows.token_logp.map(lambda value: value if isinstance(value, list) else [])
    rows["token_kls"] = rows.token_kls.map(lambda value: value if isinstance(value, list) else [])
    rows["token_count"] = rows.token_logp.map(len)
    if generated_length:
        expected = pd.to_numeric(rows[generated_length], errors="coerce").fillna(0).astype(int)
    elif generated_ids:
        expected = rows[generated_ids].map(lambda value: len(value) if isinstance(value, (list, tuple, np.ndarray)) else 0)
    else:
        raise ValueError("The generation table needs generated-token IDs or an explicit generated-token length for alignment checks.")
    misaligned = rows[(rows.token_count != expected) | (rows.token_kls.map(len) != expected)]
    if not misaligned.empty:
        examples = misaligned[["sample_id", "rollout_id", "token_count"]].head(5).to_dict("records")
        raise ValueError(f"Token-list alignment failed for logp/KL/generated tokens; examples: {examples}")
    if generated_ids and generated_length:
        id_lengths = rows[generated_ids].map(lambda value: len(value) if isinstance(value, (list, tuple, np.ndarray)) else 0)
        # The course metrics exclude a terminal token; a generation token-ID list
        # may retain that one extra EOS token and is accepted only in that case.
        malformed = ~id_lengths.isin(expected) & ~id_lengths.isin(expected + 1)
        if malformed.any():
            raise ValueError("Generated token IDs do not agree with the supplied generated-token length.")
    rows["mean_logp"] = rows.token_logp.map(lambda values: float(np.mean(values)) if values else 0.0)
    rows["self_certainty"] = rows.token_kls.map(lambda values: float(np.mean(values)) if values else 0.0)
    rows["token_vision"] = rows.token_kls.map(lambda values: [0.0] * len(values))
    attention_path = spec.get("attention_metrics")
    if attention_path:
        attention = pd.read_parquet(_path(root, attention_path))
        a_sample, a_rollout = column(attention, "sample_id"), column(attention, "rollout_id")
        attention = attention.rename(columns={a_sample: "sample_id", a_rollout: "rollout_id"})
        a_keys = ["sample_id", "rollout_id"]
        image, prompt, generated = (column(attention, name) for name in ("image", "prompt", "generated"))
        image_values = sequence_table(attention, a_keys, image, "image_tokens")
        prompt_values = sequence_table(attention, a_keys, prompt, "prompt_tokens")
        generated_values = sequence_table(attention, a_keys, generated, "generated_tokens")
        attention = image_values.merge(prompt_values, on=a_keys, validate="one_to_one").merge(generated_values, on=a_keys, validate="one_to_one")
        def vision(row: pd.Series) -> list[float]:
            parts = [row.image_tokens, row.prompt_tokens, row.generated_tokens]
            if len({len(part) for part in parts}) != 1:
                raise ValueError(f"Attention-group token alignment failed for {row.sample_id}/{row.rollout_id}")
            return [image_value / max(image_value + prompt_value + generated_value, 1e-8)
                    for image_value, prompt_value, generated_value in zip(*parts)]
        attention["token_vision"] = attention.apply(vision, axis=1)
        rows = rows.drop(columns="token_vision").merge(attention[a_keys + ["token_vision"]], on=a_keys, how="left", validate="one_to_one")
        missing = rows.token_vision.isna()
        if missing.any():
            raise ValueError(f"Missing VLM attention records for {int(missing.sum())} rollouts.")
        bad = rows[rows.apply(lambda row: len(row.token_vision) != row.token_count, axis=1)]
        if not bad.empty:
            raise ValueError("Attention-token length does not match generated-token length.")
    rows["dataset"] = spec["dataset"]
    rows = metadata_for_source(spec, root, rows)
    groups, correctness = [], []
    for _, question in rows.groupby("sample_id", sort=False):
        precision = question.precision.iloc[0]
        precision = int(precision) if pd.notna(precision) else None
        multiple_choice = bool(question.multiple_choice.iloc[0])
        choices = tuple(question.choice_options.iloc[0]) if "choice_options" in question and question.choice_options.iloc[0] else None
        question_groups = group_answers(question.raw_answer, dataset=spec["dataset"], multiple_choice=multiple_choice, precision=precision, choices=choices)
        groups.extend(question_groups)
        correctness.extend(answer_equivalent(answer, question.gold.iloc[0], dataset=spec["dataset"], multiple_choice=multiple_choice, precision=precision, choices=choices)
                           for answer in question.raw_answer)
    rows["answer_group"], rows["correct"] = groups, correctness
    counts = rows.groupby("sample_id").rollout_id.nunique()
    if len(counts) != 1000 or not (counts == 32).all():
        raise ValueError(f"Expected 1,000 questions x 32 rollouts; got {len(counts)} questions and irregular examples {counts[counts != 32].head().to_dict()}")
    return rows, {"name": spec["name"], "questions": len(counts), "rows": len(rows), "token_alignment": "passed"}


def build_manifest(rows: pd.DataFrame, output: Path, seed: int, orders: int) -> dict[str, Any]:
    ids = sorted(rows.sample_id.unique())
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))
    orders_by_question: dict[str, list[list[str]]] = {}
    for offset, (sample_id, question) in enumerate(rows.groupby("sample_id", sort=True)):
        local = np.random.default_rng(seed + offset)
        rollout_ids = sorted(question.rollout_id.tolist())
        orders_by_question[str(sample_id)] = [list(local.permutation(rollout_ids)) for _ in range(orders)]
    manifest = {"seed": seed, "dev_ids": shuffled[:200], "test_ids": shuffled[200:], "orders": orders_by_question}
    write_json(output / f"{rows.dataset.iloc[0]}_split_and_orders.json", manifest)
    return manifest


def candidates(question: pd.DataFrame, order: list[str]) -> list[Candidate]:
    by_id = question.set_index("rollout_id")
    raw = [Candidate(rollout_id=item, answer=by_id.at[item, "answer_group"], token_count=int(by_id.at[item, "token_count"]),
                     mean_logp=float(by_id.at[item, "mean_logp"]), self_certainty=float(by_id.at[item, "self_certainty"]),
                     token_kls=tuple(by_id.at[item, "token_kls"]), token_vision=tuple(by_id.at[item, "token_vision"]),
                     correct=bool(by_id.at[item, "correct"])) for item in order]
    return attach_visconf(raw)


def methods(vlm: bool) -> list[tuple[str, str, str | None]]:
    out = [("firstans", "first", None), ("self_consistency", "self_consistency", None),
           ("mean_logp", "highest_score", "mean_logp"), ("self_certainty", "highest_score", "self_certainty"),
           ("self_certainty_rwc", "rwc", "self_certainty")]
    if vlm:
        out += [("vision_only", "highest_score", "vision"), ("vision_only_rwc", "rwc", "vision"),
                ("visconf", "highest_score", "visconf"), ("visconf_rwc", "rwc", "visconf"),
                ("visconf_token", "highest_score", "visconf_token"), ("visconf_token_rwc", "rwc", "visconf_token")]
    return out


def selection_correct(prefix: list[Candidate], answer: str) -> bool:
    return any(item.answer == answer and item.correct for item in prefix)


def fixed_records(rows: pd.DataFrame, manifest: dict[str, Any], vlm: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, signal = [], []
    for sample_id, question in rows.groupby("sample_id", sort=False):
        for order_index, order in enumerate(manifest["orders"][sample_id]):
            ordered = candidates(question, order)
            for n in (1, 2, 4, 8, 16, 32):
                prefix, oracle = ordered[:n], any(item.correct for item in ordered[:n])
                for name, aggregation, score in methods(vlm):
                    answer = choose(prefix, aggregation, score)
                    consumed = prefix[:1] if name == "firstans" else prefix
                    records.append({"sample_id": sample_id, "order": order_index, "budget": n, "method": name,
                                    "candidate_ids": json.dumps([item.rollout_id for item in consumed]), "selected_answer": answer,
                                    "correct": selection_correct(prefix, answer), "oracle": oracle,
                                    "token_cost": sum(item.token_count for item in consumed), "distinct_answer_groups": len({item.answer for item in prefix}),
                                    "correct_available_but_missed": oracle and not selection_correct(prefix, answer)})
            for item in ordered:
                signal.append({"sample_id": sample_id, "order": order_index, "rollout_id": item.rollout_id, "length": item.token_count,
                               "mean_logp": item.mean_logp, "self_certainty": item.self_certainty, "vision": item.vision,
                               "visconf": item.visconf, "visconf_token": item.tokenwise_visconf, "token_confidence": list(item.token_confidence),
                               "token_vision": list(item.token_vision), "correct": item.correct})
    return pd.DataFrame(records), pd.DataFrame(signal)


def adaptive_records(rows: pd.DataFrame, manifest: dict[str, Any], vlm: bool) -> pd.DataFrame:
    policies = [("agreement", tau, None) for tau in (.5, .625, .75, .875, 1.)] + [("quality_aware", tau, gamma)
        for tau in (.5, .625, .75, .875, 1.) for gamma in (0., .1, .25, .5)] + [("fixed_32", 1., None)]
    score = "visconf" if vlm else "self_certainty"
    records = []
    for sample_id, question in rows.groupby("sample_id", sort=False):
        for order_index, order in enumerate(manifest["orders"][sample_id]):
            ordered = candidates(question, order)
            fixed_answer = choose(ordered, "self_consistency")
            fixed_correct = selection_correct(ordered, fixed_answer)
            for family, tau, gamma in policies:
                consumed, answer, reason = (32, fixed_answer, "fixed_32") if family == "fixed_32" else adaptive_stop(
                    ordered, tau=tau, gamma=gamma, score_name=score if gamma is not None else None)
                selected_correct = selection_correct(ordered[:consumed], answer)
                records.append({"sample_id": sample_id, "order": order_index, "family": family, "tau": tau, "gamma": gamma,
                                "consumed": consumed, "stopping_reason": reason, "token_cost": sum(item.token_count for item in ordered[:consumed]),
                                "selected_answer": answer, "correct": selected_correct, "fixed32_correct": fixed_correct,
                                "incorrect_early_stop_recovered_at_32": consumed < 32 and not selected_correct and fixed_correct})
    return pd.DataFrame(records)


def summary(records: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    subset = records[records.sample_id.isin(ids)]
    return subset.groupby(["budget", "method"], as_index=False).agg(accuracy=("correct", "mean"), oracle_coverage=("oracle", "mean"),
        token_cost=("token_cost", "mean"), distinct_answer_groups=("distinct_answer_groups", "mean"),
        no_correct_candidate=("oracle", lambda values: 1 - values.mean()), selection_misses=("correct_available_but_missed", "mean"))


def bootstrap(records: pd.DataFrame, ids: set[str], seed: int) -> pd.DataFrame:
    grouped = records[records.sample_id.isin(ids)].groupby(["sample_id", "budget", "method"], as_index=False).correct.mean()
    base = grouped[grouped.method == "self_consistency"][["sample_id", "budget", "correct"]].rename(columns={"correct": "baseline"})
    rng, output = np.random.default_rng(seed), []
    for (budget, method), frame in grouped.groupby(["budget", "method"]):
        if method == "self_consistency":
            continue
        paired = frame.merge(base[base.budget == budget], on=["sample_id", "budget"])
        delta = paired.correct.astype(float).to_numpy() - paired.baseline.astype(float).to_numpy()
        draws = rng.choice(delta, size=(1000, len(delta)), replace=True).mean(axis=1)
        output.append({"budget": budget, "method": method, "accuracy_difference_vs_self_consistency": float(delta.mean()),
                       "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)), "bootstrap_resamples": 1000})
    return pd.DataFrame(output)


def signal_report(signal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def finite_mean(values: Iterable[float]) -> float:
        valid = [value for value in values if pd.notna(value)]
        return float(np.mean(valid)) if valid else float("nan")
    scores = ("mean_logp", "self_certainty", "vision", "visconf", "visconf_token")
    for score in scores:
        top_bottom, length_correlations, correctness_correlations = [], [], []
        for _, group in signal.groupby(["sample_id", "order"]):
            ordered, width = group.sort_values(score), max(1, len(group) // 4)
            top_bottom.append(ordered.tail(width)["correct"].mean() - ordered.head(width)["correct"].mean())
            length_correlations.append(group["length"].corr(group[score], method="spearman")
                                       if group["length"].nunique() > 1 and group[score].nunique() > 1 else float("nan"))
            correctness_correlations.append(group["correct"].astype(float).corr(group[score], method="spearman")
                                            if group["correct"].nunique() > 1 and group[score].nunique() > 1 else float("nan"))
        rows += [{"analysis": "top_minus_bottom_quartile_correctness", "signal": score, "value": finite_mean(top_bottom)},
                 {"analysis": "within_question_length_score_spearman", "signal": score, "value": finite_mean(length_correlations)},
                 {"analysis": "within_question_correctness_score_spearman", "signal": score, "value": finite_mean(correctness_correlations)}]
    shortest = signal.loc[signal.groupby(["sample_id", "order"])["length"].idxmin()]
    rows.append({"analysis": "shortest_rollout_accuracy", "signal": "length", "value": float(shortest["correct"].mean())})
    if signal["vision"].nunique() > 1:
        token_correlations = []
        for _, rollout in signal.groupby(["sample_id", "order", "rollout_id"]):
            confidence, vision = rollout.iloc[0]["token_confidence"], rollout.iloc[0]["token_vision"]
            if len(confidence) >= 2 and np.std(confidence) > 0 and np.std(vision) > 0:
                token_correlations.append(float(np.corrcoef(confidence, vision)[0, 1]))
        rows.append({"analysis": "within_rollout_token_confidence_vision_pearson", "signal": "confidence_vs_vision",
                     "value": finite_mean(token_correlations), "rollouts": len(token_correlations)})
        rows.append({"analysis": "mean_tokenwise_minus_trajectory_visconf", "signal": "visconf_token_minus_visconf",
                     "value": float((signal["visconf_token"] - signal["visconf"]).mean())})
    return pd.DataFrame(rows)


def select_policies(dev: pd.DataFrame) -> pd.DataFrame:
    fixed = dev[dev.family == "fixed_32"].correct.mean()
    options = dev[dev.family != "fixed_32"].groupby(["family", "tau", "gamma"], dropna=False).agg(
        accuracy=("correct", "mean"), token_cost=("token_cost", "mean"), mean_rollouts=("consumed", "mean")).reset_index()
    viable = options[options.accuracy >= fixed - .01]
    return viable.sort_values(["family", "token_cost", "accuracy", "tau", "gamma"], ascending=[True, True, False, True, True]).groupby("family").head(1)


def adaptive_summary(heldout: pd.DataFrame, policies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = heldout.merge(policies[["family", "tau", "gamma"]], on=["family", "tau", "gamma"], how="inner")
    aggregate = selected.groupby("family", as_index=False).agg(accuracy=("correct", "mean"), token_cost=("token_cost", "mean"),
        mean_rollouts=("consumed", "mean"), incorrect_early_stops_recovered_at_32=("incorrect_early_stop_recovered_at_32", "mean"))
    return aggregate, selected.groupby(["family", "consumed"], as_index=False).size().rename(columns={"size": "orders"})


def qualitative(records: pd.DataFrame, ids: set[str]) -> list[dict[str, Any]]:
    rows = records[(records.sample_id.isin(ids)) & (records.budget == 32)]
    selected = []
    for sample_id, question in rows.groupby("sample_id"):
        answers = question.groupby("method").selected_answer.first().to_dict()
        if len(set(answers.values())) > 1:
            selected.append({"sample_id": sample_id, "answers_by_method": answers, "selection_rule": "first held-out question with a method disagreement"})
        if len(selected) == 12:
            break
    return selected


def run(config: dict[str, Any]) -> None:
    root, output = Path(config["input_root"]).expanduser(), Path(config.get("output_dir", "results/task3/offline")).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for spec in config["sources"]:
        print(f"Loading and validating {spec['name']}...", flush=True)
        rows, metadata = load_source(spec, root)
        dataset_output = output / spec["name"]
        manifest = build_manifest(rows, dataset_output, int(config.get("seed", 6304)), int(config.get("orders", 10)))
        fixed, signals = fixed_records(rows, manifest, bool(spec.get("attention_metrics")))
        adaptive = adaptive_records(rows, manifest, bool(spec.get("attention_metrics")))
        for name, table in (("per_question_fixed.parquet", fixed), ("per_question_adaptive.parquet", adaptive), ("signal_records.parquet", signals)):
            dataset_output.mkdir(parents=True, exist_ok=True)
            table.to_parquet(dataset_output / name, index=False)
        dev_ids, test_ids = set(manifest["dev_ids"]), set(manifest["test_ids"])
        development, heldout = summary(fixed, dev_ids), summary(fixed, test_ids)
        development.assign(split="development").to_csv(dataset_output / "fixed_development_summary.csv", index=False)
        heldout.assign(split="heldout").to_csv(dataset_output / "fixed_heldout_summary.csv", index=False)
        bootstrap(fixed, test_ids, int(config.get("seed", 6304))).to_csv(dataset_output / "fixed_heldout_bootstrap.csv", index=False)
        signal_report(signals).to_csv(dataset_output / "signal_summary.csv", index=False)
        policies = select_policies(adaptive[adaptive.sample_id.isin(dev_ids)])
        policies.to_csv(dataset_output / "adaptive_selected_on_development.csv", index=False)
        aggregate, distribution = adaptive_summary(adaptive[adaptive.sample_id.isin(test_ids)], policies)
        aggregate.to_csv(dataset_output / "adaptive_heldout_summary.csv", index=False)
        distribution.to_csv(dataset_output / "adaptive_stopping_distribution.csv", index=False)
        write_json(dataset_output / "qualitative_selection_disagreements.json", qualitative(fixed, test_ids))
        write_json(dataset_output / "run_metadata.json", {**metadata, "scorer_version": SCORER_VERSION, "seed": config.get("seed", 6304), "orders": config.get("orders", 10)})
        all_summaries += [development.assign(source=spec["name"], split="development"), heldout.assign(source=spec["name"], split="heldout")]
        print(f"Saved {spec['name']} offline Task 3 results.", flush=True)
    pd.concat(all_summaries, ignore_index=True).to_csv(output / "all_fixed_summaries.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3/offline.json")
    args = parser.parse_args()
    run(read_json(Path(args.config)))


if __name__ == "__main__":
    main()
