"""Local VQA scoring, normalization, and complete-pair bootstrap utilities."""
import re
import string
import numpy as np


ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_answer(value):
    value = str(value).lower().replace("\n", " ").strip()
    value = re.sub(r"(?<!\d)[.,](?!\d)", "", value)
    words = value.translate(str.maketrans("", "", string.punctuation)).split()
    value = " ".join(NUMBER_WORDS.get(word, word) for word in words if word not in ARTICLES)
    return value


def vqa_soft_score(prediction, references):
    prediction = normalize_answer(prediction)
    answers = [normalize_answer(answer) for answer in references]
    return sum(min(1.0, sum(answer == prediction for j, answer in enumerate(answers) if j != i) / 3.0)
               for i in range(len(answers))) / len(answers)


def exact_score(prediction, target):
    return float(normalize_answer(prediction) == normalize_answer(target))


def bootstrap_pair_difference(correct, swapped, seed=6304, resamples=4000):
    """Paired percentile interval; each entry is one complete question pair."""
    correct, swapped = np.asarray(correct, dtype=float), np.asarray(swapped, dtype=float)
    if correct.shape != swapped.shape or not len(correct):
        raise ValueError("Expected aligned nonempty complete-pair measurements.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(correct), size=(resamples, len(correct)))
    samples = (correct[draws] - swapped[draws]).mean(axis=1)
    return {"difference": float((correct - swapped).mean()), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}
