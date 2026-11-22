"""Seuil conforme (split conformal) — même formule que la brique B3.

Soient n scores d'images normales de calibration, triés par ordre croissant
s(1) <= ... <= s(n). Pour un taux de fausses alertes visé alpha :

    k = ceil((n + 1) * (1 - alpha))
    q = s(k)        si k <= n
    q = +inf        si k > n   (pas assez d'images pour ce alpha)

Décision : anomalie si score > q.
Garantie (si calibration et test sont échangeables) : FPR marginal <= alpha.
"""

import math
from typing import Sequence

_EPS = 1e-12  # évite qu'une erreur d'arrondi (ex. 9.000000001) fasse monter k


def conformal_rank(n: int, alpha: float) -> int:
    """Rang k (indexé à partir de 1) du score retenu comme seuil."""
    _check_alpha(alpha)
    if n < 1:
        raise ValueError("Il faut au moins une image de calibration.")
    return math.ceil((n + 1) * (1 - alpha) - _EPS)


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """Seuil conforme ; renvoie math.inf si n est trop petit pour alpha."""
    n = len(scores)
    k = conformal_rank(n, alpha)
    if k > n:
        return math.inf
    return sorted(float(s) for s in scores)[k - 1]


def min_calibration_size(alpha: float) -> int:
    """Plus petit n donnant un seuil fini pour ce alpha."""
    _check_alpha(alpha)
    n = 1
    while conformal_rank(n, alpha) > n:
        n += 1
    return n


def is_anomaly(score: float, threshold: float) -> bool:
    return score > threshold


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement entre 0 et 1.")
