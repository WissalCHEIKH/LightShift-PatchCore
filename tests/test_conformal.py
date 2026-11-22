import math

import numpy as np
import pytest

from app.core.conformal import (conformal_rank, conformal_threshold, is_anomaly,
                                min_calibration_size)


def test_rank_examples_from_guide():
    assert conformal_rank(20, 0.10) == 19
    assert conformal_rank(32, 0.05) == 32   # exemple du guide : seuil = max des scores
    assert conformal_rank(9, 0.10) == 9     # pas d'arrondi parasite à 10


def test_threshold_is_kth_smallest():
    scores = list(range(1, 21))             # 1..20
    assert conformal_threshold(scores, 0.10) == 19


def test_threshold_infinite_when_too_few_images():
    assert conformal_threshold([0.1, 0.2, 0.3, 0.4, 0.5], 0.10) == math.inf


@pytest.mark.parametrize("alpha,expected", [(0.10, 9), (0.05, 19), (0.20, 4)])
def test_min_calibration_size(alpha, expected):
    assert min_calibration_size(alpha) == expected
    assert math.isfinite(conformal_threshold(np.arange(expected), alpha))
    assert not math.isfinite(conformal_threshold(np.arange(expected - 1), alpha))


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_threshold([0.1, 0.2], alpha)


def test_strict_decision_rule():
    assert not is_anomaly(0.5, 0.5)
    assert is_anomaly(0.50001, 0.5)


@pytest.mark.parametrize("n,alpha", [(20, 0.10), (50, 0.05), (19, 0.05)])
def test_marginal_fpr_guarantee_on_simulated_scores(n, alpha):
    """Données simulées échangeables : le FPR moyen doit être <= alpha,
    et >= alpha - 1/(n+1) (le seuil conforme n'est pas trop conservateur)."""
    rng = np.random.default_rng(0)
    trials, n_test = 4000, 200
    fprs = []
    for _ in range(trials):
        cal = rng.normal(size=n)
        test = rng.normal(size=n_test)
        q = conformal_threshold(cal, alpha)
        fprs.append(np.mean(test > q))
    mean_fpr = float(np.mean(fprs))
    assert mean_fpr <= alpha + 0.005
    assert mean_fpr >= alpha - 1 / (n + 1) - 0.005
