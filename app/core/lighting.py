"""Changements d'éclairage pour la démo.

Réutilise exactement apply_shift() du membre 1 (src/augmentation/lighting_shifts.py)
pour que la démo applique les mêmes transformations que les expériences.
"""

import sys

import cv2
import numpy as np

from app.config import REPO_ROOT

_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from augmentation.lighting_shifts import apply_shift as _apply_shift_bgr  # noqa: E402


def apply_lighting_shift(rgb: np.ndarray, shift: str, severity: int) -> np.ndarray:
    """apply_shift() travaille en BGR (OpenCV) : on convertit autour."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    shifted = _apply_shift_bgr(bgr, shift, severity)
    return cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
