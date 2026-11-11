"""Pure scoring utilities for Task 3.

These functions deliberately never use gold answers while selecting a candidate.
Gold answers are used only by ``is_correct`` after a selection has been made.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Any, Iterable

NO_ANSWER = "__NO_ANSWER__"


def normalise_answer(value: Any, *, multiple_choice: bool = False) -> str:
    """A deterministic conservative normaliser used for voting and grading.

    The supplied extraction is retained verbatim elsewhere; this only creates a
    grouping key.  Missing or empty extractions intentionally form one group.
    """
    if value is None:
        return NO_ANSWER
    text = str(value).strip()
    if not text:
        return NO_ANSWER
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()
    if multiple_choice:
        found = re.findall(r"\b([A-E])\b", text.upper())
        if found:
            return found[-1]
    text = text.lower()
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    text = text.replace(",", "")
    # Exact rational equality is useful for common MATH extractions, without
    # making unsafe symbolic-equivalence claims when math-verify is unavailable.
    try:
        return f"num:{Fraction(text)}"
    except (ValueError, ZeroDivisionError):
        return text


def is_correct(answer_group: str, gold: Any, *, multiple_choice: bool = False) -> bool:
    return answer_group != NO_ANSWER and answer_group == normalise_answer(gold, multiple_choice=multiple_choice)


@dataclass(frozen=True)
class Candidate:
    rollout_id: str
    answer: str
    token_count: int
    mean_logp: float
    self_certainty: float
    vision: float = 0.0
    visconf: float = 0.0

    def score(self, name: str) -> float:
        values = {
            "mean_logp": self.mean_logp,
            "self_certainty": self.self_certainty,
            "vision": self.vision,
            "visconf": self.visconf,
        }
        if name not in values:
            raise ValueError(f"Unknown quality score: {name}")
        return values[name]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def attach_visconf(candidates: list[Candidate], token_kls: list[list[float]] | None = None) -> list[Candidate]:
    """Compute prefix-local VisConf with linear 5th/95th percentiles.

    ``token_kls`` is optional because a source may only expose a rollout mean;
    in that case that mean is treated as its one-token sequence.
    """
    if not candidates:
        return []
    if token_kls is None:
        token_kls = [[candidate.self_certainty] for candidate in candidates]
    pooled = [_safe_float(v) for sequence in token_kls for v in sequence]
    if not pooled:
        pooled = [0.0]
    import numpy as np
    q05, q95 = np.percentile(pooled, [5, 95], method="linear")
    denom = max(float(q95 - q05), 1e-8)
    out: list[Candidate] = []
    for candidate, sequence in zip(candidates, token_kls):
        calibrated = [min(1.0, max(0.0, (_safe_float(v) - q05) / denom)) for v in sequence]
        confidence = sum(calibrated) / max(1, len(calibrated))
        out.append(Candidate(**{**candidate.__dict__, "visconf": candidate.vision * confidence}))
    return out


def vote_answer(candidates: Iterable[Candidate], *, weighted: dict[str, float] | None = None) -> str:
    """Choose answer by votes; ties go to the earliest supporting rollout."""
    totals: dict[str, float] = defaultdict(float)
    first: dict[str, int] = {}
    for position, candidate in enumerate(candidates):
        totals[candidate.answer] += 1.0 if weighted is None else weighted[candidate.rollout_id]
        first.setdefault(candidate.answer, position)
    if not totals:
        return NO_ANSWER
    return min(totals, key=lambda answer: (-totals[answer], first[answer], answer))


def rank_weights(candidates: list[Candidate], score_name: str) -> dict[str, float]:
    """Rank-weighted-consensus weights using average ranks for score ties."""
    n = len(candidates)
    if n <= 1:
        return {candidate.rollout_id: 1.0 for candidate in candidates}
    by_score: dict[float, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_score[candidate.score(score_name)].append(candidate)
    weights: dict[str, float] = {}
    start = 1
    for score in sorted(by_score, reverse=True):
        group = by_score[score]
        end = start + len(group) - 1
        average_rank = (start + end) / 2.0
        weight = (n - average_rank) / (n - 1)
        for candidate in group:
            weights[candidate.rollout_id] = weight
        start = end + 1
    return weights


def choose(candidates: list[Candidate], method: str, score_name: str | None = None) -> str:
    """Select an answer from a current prefix without seeing correctness."""
    if not candidates:
        return NO_ANSWER
    if method == "first":
        return candidates[0].answer
    if method == "self_consistency":
        return vote_answer(candidates)
    if method == "highest_score":
        if score_name is None:
            raise ValueError("highest_score needs score_name")
        # max preserves earliest candidate when scores tie.
        return max(candidates, key=lambda candidate: candidate.score(score_name)).answer
    if method == "rwc":
        if score_name is None:
            raise ValueError("rwc needs score_name")
        return vote_answer(candidates, weighted=rank_weights(candidates, score_name))
    raise ValueError(f"Unknown aggregation method: {method}")


def rwc_margin(candidates: list[Candidate], score_name: str) -> float:
    weights = rank_weights(candidates, score_name)
    totals: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        totals[candidate.answer] += weights[candidate.rollout_id]
    ordered = sorted(totals.values(), reverse=True)
    return (ordered[0] - ordered[1]) / max(sum(totals.values()), 1e-8) if len(ordered) > 1 else 1.0


def adaptive_stop(
    candidates: list[Candidate], *, tau: float, gamma: float | None, score_name: str | None
) -> tuple[int, str]:
    """Return (number consumed, self-consistency answer) for one saved order."""
    for n in (4, 8, 16):
        prefix = candidates[:n]
        winner = choose(prefix, "self_consistency")
        count = sum(candidate.answer == winner for candidate in prefix)
        unique_non_missing = winner != NO_ANSWER and sum(candidate.answer == winner for candidate in prefix) == max(
            sum(other.answer == answer for other in prefix) for answer in {other.answer for other in prefix}
        )
        agreement = unique_non_missing and count / n >= tau
        if gamma is None:
            if agreement:
                return n, winner
        else:
            if score_name is None:
                raise ValueError("quality-aware stopping needs a score")
            quality_winner = choose(prefix, "rwc", score_name)
            if agreement and winner == quality_winner and rwc_margin(prefix, score_name) >= gamma:
                return n, winner
    return min(32, len(candidates)), choose(candidates[:32], "self_consistency")
