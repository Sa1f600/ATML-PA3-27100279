"""Selection, equivalence, and stopping rules for Task 3.

No function in this module uses a gold answer while grouping, ranking, voting, or
stopping. Gold answers are consulted only by retrospective grading helpers.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
import math
import re
from typing import Any, Iterable

import numpy as np

NO_ANSWER = "__NO_ANSWER__"
SCORER_VERSION = "task3-equivalence-v2"


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("−", "-").replace("–", "-")).strip()


def normalise_answer(value: Any, *, multiple_choice: bool = False, precision: int | None = None,
                     choices: tuple[str, ...] | None = None) -> str:
    """Create a conservative canonical key; equivalence is handled separately."""
    if value is None or not _text(value):
        return NO_ANSWER
    text = _text(value)
    if multiple_choice:
        choice_letters = re.findall(r"\b([A-E])\b", text.upper())
        if choice_letters:
            return choice_letters[-1]
        if choices is not None:
            candidate = re.sub(r"^(?:the answer is|answer is|answer:|the)\s+", "", _text(text).casefold()).rstrip(". ")
            for index, option in enumerate(choices):
                if candidate == _text(option).casefold().rstrip(". "):
                    return chr(ord("A") + index)
        return NO_ANSWER
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text).replace(",", "").lower()
    if precision is not None:
        try:
            return f"decimal:{Decimal(text).quantize(Decimal(1).scaleb(-precision))}"
        except (InvalidOperation, ValueError):
            pass
    try:
        return f"num:{Fraction(text)}"
    except (ValueError, ZeroDivisionError):
        return text


@lru_cache(maxsize=200_000)
def _math_equivalent_cached(left: str, right: str) -> bool:
    """Use math-verify when installed, with a deterministic exact fallback."""
    if normalise_answer(left) == NO_ANSWER or normalise_answer(right) == NO_ANSWER:
        return False
    if normalise_answer(left) == normalise_answer(right):
        return True
    try:
        from math_verify import parse, verify  # type: ignore
        return bool(verify(parse(str(left)), parse(str(right))))
    except Exception:
        # The fallback is deliberately narrow. It never declares two different
        # symbolic strings equivalent merely to improve an experiment score.
        return False


def math_equivalent(left: Any, right: Any) -> bool:
    return _math_equivalent_cached(str(left), str(right))


def answer_equivalent(left: Any, right: Any, *, dataset: str, multiple_choice: bool = False,
                      precision: int | None = None, choices: tuple[str, ...] | None = None) -> bool:
    if multiple_choice:
        return (normalise_answer(left, multiple_choice=True, choices=choices) ==
                normalise_answer(right, multiple_choice=True, choices=choices) != NO_ANSWER)
    if dataset.lower() == "math":
        return math_equivalent(left, right)
    return normalise_answer(left, precision=precision) == normalise_answer(right, precision=precision) != NO_ANSWER


def group_answers(values: Iterable[Any], *, dataset: str, multiple_choice: bool = False,
                  precision: int | None = None, choices: tuple[str, ...] | None = None) -> list[str]:
    """Group predictions without looking at a gold answer."""
    representatives: list[tuple[str, Any]] = []
    groups: list[str] = []
    for value in values:
        canonical = normalise_answer(value, multiple_choice=multiple_choice, precision=precision, choices=choices)
        if canonical == NO_ANSWER:
            groups.append(NO_ANSWER)
            continue
        matched = next((key for key, representative in representatives if answer_equivalent(
            value, representative, dataset=dataset, multiple_choice=multiple_choice, precision=precision, choices=choices
        )), None)
        if matched is None:
            matched = canonical
            representatives.append((matched, value))
        groups.append(matched)
    return groups


@dataclass(frozen=True)
class Candidate:
    rollout_id: str
    answer: str
    token_count: int
    mean_logp: float
    self_certainty: float
    vision: float = 0.0
    visconf: float = 0.0
    tokenwise_visconf: float = 0.0
    token_kls: tuple[float, ...] = ()
    token_vision: tuple[float, ...] = ()
    token_confidence: tuple[float, ...] = ()
    correct: bool = False

    def score(self, name: str) -> float:
        values = {"mean_logp": self.mean_logp, "self_certainty": self.self_certainty,
                  "vision": self.vision, "visconf": self.visconf, "visconf_token": self.tokenwise_visconf}
        if name not in values:
            raise ValueError(f"Unknown quality score: {name}")
        return values[name]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def attach_visconf(candidates: list[Candidate]) -> list[Candidate]:
    """Recompute prefix-local VisConf from raw token KL and attention arrays."""
    pooled = [_finite(value) for candidate in candidates for value in candidate.token_kls]
    if not pooled:
        pooled = [0.0]
    q05, q95 = np.percentile(pooled, [5, 95], method="linear")
    denominator = max(float(q95 - q05), 1e-8)
    scored: list[Candidate] = []
    for candidate in candidates:
        confidence_tokens = tuple(min(1.0, max(0.0, (_finite(value) - q05) / denominator)) for value in candidate.token_kls)
        confidence = float(np.mean(confidence_tokens)) if confidence_tokens else 0.0
        vision = candidate.token_vision or tuple(0.0 for _ in confidence_tokens)
        if len(vision) != len(confidence_tokens):
            raise ValueError(f"Token alignment failure for rollout {candidate.rollout_id}: KL={len(confidence_tokens)}, vision={len(vision)}")
        visual_engagement = float(np.mean(vision)) if vision else 0.0
        tokenwise = float(np.mean([c * v for c, v in zip(confidence_tokens, vision)])) if vision else 0.0
        scored.append(replace(candidate, vision=visual_engagement, token_confidence=confidence_tokens, visconf=visual_engagement * confidence,
                              tokenwise_visconf=tokenwise))
    return scored


def vote_answer(candidates: Iterable[Candidate], *, weighted: dict[str, float] | None = None) -> str:
    totals: dict[str, float] = defaultdict(float)
    first: dict[str, int] = {}
    for position, candidate in enumerate(candidates):
        totals[candidate.answer] += 1.0 if weighted is None else weighted[candidate.rollout_id]
        first.setdefault(candidate.answer, position)
    if not totals:
        return NO_ANSWER
    return min(totals, key=lambda answer: (-totals[answer], first[answer], answer))


def rank_weights(candidates: list[Candidate], score_name: str) -> dict[str, float]:
    n = len(candidates)
    if n <= 1:
        return {candidate.rollout_id: 1.0 for candidate in candidates}
    groups: dict[float, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.score(score_name)].append(candidate)
    weights: dict[str, float] = {}
    start = 1
    for score in sorted(groups, reverse=True):
        members = groups[score]
        end = start + len(members) - 1
        weight = (n - ((start + end) / 2.0)) / (n - 1)
        for candidate in members:
            weights[candidate.rollout_id] = weight
        start = end + 1
    return weights


def choose(candidates: list[Candidate], method: str, score_name: str | None = None) -> str:
    if not candidates:
        return NO_ANSWER
    if method == "first":
        return candidates[0].answer
    if method == "self_consistency":
        return vote_answer(candidates)
    if method == "highest_score":
        if score_name is None:
            raise ValueError("highest_score needs score_name")
        return max(candidates, key=lambda candidate: candidate.score(score_name)).answer
    if method == "rwc":
        if score_name is None:
            raise ValueError("rwc needs score_name")
        return vote_answer(candidates, weighted=rank_weights(candidates, score_name))
    raise ValueError(f"Unknown aggregation method: {method}")


def rwc_margin(candidates: list[Candidate], score_name: str) -> float:
    weighted = rank_weights(candidates, score_name)
    totals: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        totals[candidate.answer] += weighted[candidate.rollout_id]
    values = sorted(totals.values(), reverse=True)
    return (values[0] - (values[1] if len(values) > 1 else 0.0)) / max(sum(values), 1e-8)


def adaptive_stop(candidates: list[Candidate], *, tau: float, gamma: float | None,
                  score_name: str | None) -> tuple[int, str, str]:
    for n in (4, 8, 16):
        prefix = candidates[:n]
        winner = choose(prefix, "self_consistency")
        counts: dict[str, int] = defaultdict(int)
        for candidate in prefix:
            counts[candidate.answer] += 1
        sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        unique_leader = len(sorted_counts) == 1 or sorted_counts[0][1] > sorted_counts[1][1]
        agreement = winner != NO_ANSWER and unique_leader and counts[winner] / n >= tau
        if gamma is None and agreement:
            return n, winner, "agreement"
        if gamma is not None:
            if score_name is None:
                raise ValueError("quality-aware stopping needs a score")
            if agreement and winner == choose(prefix, "rwc", score_name) and rwc_margin(prefix, score_name) >= gamma:
                return n, winner, "quality_aware"
    return min(32, len(candidates)), choose(candidates[:32], "self_consistency"), "fallback_32"
