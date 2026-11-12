from task3_test_time.core import Candidate, adaptive_stop, attach_visconf, choose, group_answers, normalise_answer, rank_weights
from task3_test_time.offline import load_source, signal_report
import pandas as pd
import pytest


def candidates():
    return [
        Candidate("a", "x", 5, -2.0, 1.0),
        Candidate("b", "y", 7, -1.0, 2.0),
        Candidate("c", "y", 9, -3.0, 3.0),
        Candidate("d", "x", 3, -4.0, 4.0),
    ]


def test_selection_and_earliest_ties():
    items = candidates()
    assert choose(items, "first") == "x"
    assert choose(items, "self_consistency") == "x"  # 2-vs-2; x appeared first.
    assert choose(items, "highest_score", "mean_logp") == "y"
    assert choose(items, "rwc", "self_certainty") == "x"  # equal RWC totals; x supported first.


def test_average_rank_ties():
    tied = [Candidate("a", "a", 1, 0, 1), Candidate("b", "b", 1, 0, 1)]
    weights = rank_weights(tied, "self_certainty")
    assert weights["a"] == weights["b"] == 0.5


def test_visconf_is_prefix_local_and_bounded():
    items = [Candidate(**{**item.__dict__, "token_kls": (float(index),), "token_vision": (.5,)})
             for index, item in enumerate(candidates())]
    scored = attach_visconf(items)
    assert all(0.0 <= item.visconf <= 1.0 for item in scored)
    assert all(0.0 <= item.tokenwise_visconf <= 1.0 for item in scored)


def test_adaptive_requires_non_missing_unique_leader():
    items = [Candidate(str(i), "yes", 1, 0, 1) for i in range(4)]
    consumed, answer, reason = adaptive_stop(items, tau=0.75, gamma=None, score_name=None)
    assert (consumed, answer, reason) == (4, "yes", "agreement")


def test_missing_answers_share_one_group_without_gold():
    assert group_answers([None, "", "2/4", "1/2"], dataset="MATH") == ["__NO_ANSWER__", "__NO_ANSWER__", "num:1/2", "num:1/2"]


def test_choice_text_maps_to_its_option_letter():
    assert normalise_answer("the blue square", multiple_choice=True, choices=("red circle", "blue square")) == "B"


def test_loader_preserves_and_validates_token_level_vlm_arrays(tmp_path):
    rows = [{"sample_id": f"q{i:04d}", "rollout_id": f"r{rollout}", "extracted_answer": "A", "gold_answer": "A",
             "generated_token_ids": [1, 2]} for i in range(1000) for rollout in range(32)]
    generations = pd.DataFrame(rows)
    probability = generations[["sample_id", "rollout_id"]].copy()
    probability["logp"], probability["kl_u_p"] = [[-.1, -.2]] * len(probability), [[.2, .4]] * len(probability)
    attention = generations[["sample_id", "rollout_id"]].copy()
    attention["image_attention"], attention["prompt_attention"], attention["generated_attention"] = [[1., 1.]] * len(attention), [[1., 1.]] * len(attention), [[2., 2.]] * len(attention)
    metadata = pd.DataFrame({"sample_id": [f"q{i:04d}" for i in range(1000)], "answer_type": "multiple choice", "precision": 0,
                             "choices": [["wrong", "A"]] * 1000})
    generations.to_parquet(tmp_path / "g.parquet")
    probability.to_parquet(tmp_path / "p.parquet")
    attention.to_parquet(tmp_path / "a.parquet")
    metadata.to_parquet(tmp_path / "m.parquet")
    loaded, details = load_source({"name": "toy", "dataset": "MathVista", "generations": "g.parquet", "probability_metrics": "p.parquet",
                                   "attention_metrics": "a.parquet", "question_metadata": "m.parquet"}, tmp_path)
    assert details["token_alignment"] == "passed"
    assert len(loaded) == 32000
    assert loaded.iloc[0].token_kls == [.2, .4]
    assert loaded.iloc[0].token_vision == [.25, .25]
    assert loaded.iloc[0].choice_options == ("wrong", "A")


def test_loader_accepts_released_mathvista_json_metadata(tmp_path):
    metadata = {
        "1": {"pid": "1", "question_type": "multi_choice", "answer_type": "text", "precision": 0,
              "choices": ["wrong", "right"]},
        "2": {"pid": "2", "question_type": "free_form", "answer_type": "float", "precision": 1,
              "choices": None},
    }
    (tmp_path / "metadata.json").write_text(__import__("json").dumps(metadata))
    rows = pd.DataFrame({"sample_id": ["1", "2"]})
    loaded = __import__("task3_test_time.offline", fromlist=["metadata_for_source"]).metadata_for_source(
        {"question_metadata": "metadata.json", "metadata_columns": {"sample_id": "pid", "answer_type": "question_type",
         "precision": "precision", "choices": "choices"}}, tmp_path, rows)
    assert loaded.multiple_choice.tolist() == [True, False]
    assert loaded.choice_options.iloc[0] == ("wrong", "right")


def test_metadata_loader_rejects_unmatched_rollout_id(tmp_path):
    (tmp_path / "metadata.json").write_text(__import__("json").dumps({
        "1": {"pid": "1", "question_type": "free_form", "answer_type": "text", "precision": 0, "choices": None}
    }))
    rows = pd.DataFrame({"sample_id": ["not-in-metadata"]})
    loader = __import__("task3_test_time.offline", fromlist=["metadata_for_source"]).metadata_for_source
    with pytest.raises(ValueError, match="does not cover"):
        loader({"question_metadata": "metadata.json", "metadata_columns": {"sample_id": "pid"}}, tmp_path, rows)


def test_smoke_config_requires_positive_question_count(tmp_path):
    from task3_test_time.offline import run
    with pytest.raises(ValueError, match="at least 1"):
        run({"input_root": str(tmp_path), "output_dir": str(tmp_path / "out"), "sources": [], "smoke_questions": 0})


def test_signal_report_uses_within_rollout_token_correlation():
    signal = pd.DataFrame([
        {"sample_id": "q", "order": 0, "rollout_id": "a", "length": 2, "mean_logp": -.1, "self_certainty": .5,
         "vision": .5, "visconf": .2, "visconf_token": .3, "token_confidence": [.1, .9], "token_vision": [.1, .9], "correct": True},
        {"sample_id": "q", "order": 0, "rollout_id": "b", "length": 2, "mean_logp": -.2, "self_certainty": .4,
         "vision": .4, "visconf": .1, "visconf_token": .2, "token_confidence": [.2, .8], "token_vision": [.2, .8], "correct": False},
    ])
    report = signal_report(signal)
    value = report.loc[report.analysis == "within_rollout_token_confidence_vision_pearson", "value"].iloc[0]
    assert value == pytest.approx(1.0)
