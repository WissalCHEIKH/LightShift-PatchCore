"""Décodage d'images, carte de chaleur et encodage base64."""

import base64

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Octets PNG/JPG -> image RGB uint8 (H, W, 3)."""
    if not data:
        raise InvalidImageError("Fichier vide.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidImageError("Le fichier n'est pas une image lisible (PNG ou JPG attendu).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png_base64(rgb: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Encodage PNG impossible.")
    return base64.b64encode(png.tobytes()).decode("ascii")


def heatmap_overlay(rgb: np.ndarray, anomaly_map: np.ndarray, threshold: float = 0.5,
                    color=(200, 55, 45), max_opacity: float = 0.75) -> np.ndarray:
    """Teinte en rouge les zones suspectes, sur une échelle fixe liée à la limite.

    anomaly_map est sur la même échelle que le score : sous 70 % de la limite, rien n'est
    teinté ; à 130 % de la limite et au-delà, la teinte est maximale. Une pièce saine reste
    donc propre, au lieu d'être recolorée entièrement comme avec une normalisation min-max.
    """
    h, w = rgb.shape[:2]
    amap = anomaly_map.astype(np.float32)
    if amap.shape != (h, w):
        amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = 0.7 * threshold, 1.3 * threshold
    alpha = np.clip((amap - lo) / max(hi - lo, 1e-6), 0.0, 1.0)[..., None] * max_opacity
    tint = np.empty_like(rgb, dtype=np.float32)
    tint[:] = color
    out = rgb.astype(np.float32) * (1 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
