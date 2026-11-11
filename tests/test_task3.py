from task3_test_time.core import Candidate, adaptive_stop, attach_visconf, choose, rank_weights


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
    scored = attach_visconf(candidates(), [[0.0], [1.0], [2.0], [3.0]])
    assert all(0.0 <= item.visconf <= 1.0 for item in scored)


def test_adaptive_requires_non_missing_unique_leader():
    items = [Candidate(str(i), "yes", 1, 0, 1) for i in range(4)]
    consumed, answer = adaptive_stop(items, tau=0.75, gamma=None, score_name=None)
    assert (consumed, answer) == (4, "yes")
