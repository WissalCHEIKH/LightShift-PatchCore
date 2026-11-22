"""Modèle de démonstration : un « mini-PatchCore » sans réseau de neurones.

⚠️ Ce n'est PAS le vrai PatchCore et ses scores n'ont aucune valeur scientifique.
Il reprend seulement le principe pour que l'application se comporte de façon réaliste :

  1. Mémoire : on découpe des pièces d'exemple SAINES (app/models/examples.py), prises sous
     la lumière normale, en petits patchs, et on garde leurs descripteurs.
  2. Test : chaque patch de l'image est comparé à son plus proche voisin dans la mémoire.
  3. Score image = distance du patch le plus éloigné (comme PatchCore).

Conséquences, comme avec le vrai modèle :
  - une pièce abîmée a des patchs inconnus -> score élevé ;
  - un changement de lumière éloigne TOUS les patchs -> fausses alertes ;
  - /calibrate (brique B3) corrige la limite pour la nouvelle lumière.
Méthode "lightshift" : chaque image est normalisée par canal avant comparaison (idée de B2).

Pensé pour les pièces d'exemple : une photo réelle (MVTec) sera jugée « inconnue ».
"""

import threading

import cv2
import numpy as np

from app.models.base import Prediction
from app.models.examples import draw_part

RES = 96          # résolution de travail
PATCH = 8         # taille d'un patch
STRIDE = 4
N_MEMORY = 32     # pièces saines dans la mémoire
FEAT = 4          # chaque patch est résumé en FEAT x FEAT x 3 valeurs


def _normalize(img: np.ndarray) -> np.ndarray:
    mean = img.mean(axis=(0, 1), keepdims=True)
    std = img.std(axis=(0, 1), keepdims=True) + 1e-6
    return (img - mean) / std * 0.18 + 0.5


def _patch_features(rgb: np.ndarray, normalize: bool):
    img = cv2.resize(rgb, (RES, RES), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if normalize:
        img = _normalize(img)
    n = (RES - PATCH) // STRIDE + 1
    windows = np.lib.stride_tricks.sliding_window_view(img, (PATCH, PATCH, 3))[::STRIDE, ::STRIDE, 0]
    windows = windows.reshape(n * n, PATCH, PATCH, 3)
    k = PATCH // FEAT
    feats = windows.reshape(n * n, FEAT, k, FEAT, k, 3).mean(axis=(2, 4)).reshape(n * n, -1)
    return feats, n


# Mémoires partagées entre toutes les instances : elles sont déterministes.
_MEMORY: dict = {}
_SCALE: dict = {}
_LOCK = threading.Lock()


class MockModel:
    name = "mock"
    version = "mini-patchcore-demo-0.3"

    def __init__(self):
        self._memory = _MEMORY
        self._scale = _SCALE
        self._lock = _LOCK

    def is_ready(self) -> bool:
        return True

    def warm_up(self, categories, methods) -> None:
        """Construit les mémoires à l'avance (appelé en arrière-plan au démarrage de l'API)."""
        for category in categories:
            for method in methods:
                self._bank(category, method == "lightshift")

    def default_threshold(self, category: str, method: str) -> float:
        return 0.5

    def _bank(self, category: str, normalize: bool):
        key = (category, normalize)
        with self._lock:
            if key not in self._memory:
                feats = [_patch_features(draw_part(category, False, 10_000 + i), normalize)[0]
                         for i in range(N_MEMORY)]
                bank = np.concatenate(feats).astype(np.float32)
                self._memory[key] = (bank, (bank ** 2).sum(1))
                # échelle : distance max des pièces saines (hors mémoire) -> score 0.5 un peu au-dessus
                held_out = [self._raw_distance(draw_part(category, False, 20_000 + i), category, normalize)[0]
                            for i in range(64)]
                self._scale[key] = float(np.max(held_out)) * 1.25
            return self._memory[key]

    def _raw_distance(self, rgb, category, normalize):
        bank, bank_sq = self._memory[(category, normalize)]
        feats, n = _patch_features(rgb, normalize)
        d2 = (feats ** 2).sum(1)[:, None] - 2 * feats @ bank.T + bank_sq[None, :]
        nn = np.sqrt(np.maximum(d2.min(axis=1), 0))
        return float(nn.max()), nn.reshape(n, n)

    def predict(self, image_rgb: np.ndarray, category: str, method: str) -> Prediction:
        normalize = method == "lightshift"
        self._bank(category, normalize)
        dist, grid = self._raw_distance(image_rgb, category, normalize)
        scale = self._scale[(category, normalize)]
        score = dist / (dist + scale)            # 0.5 quand dist == échelle des pièces saines
        h, w = image_rgb.shape[:2]
        grid = grid / (grid + scale)             # carte sur la même échelle que le score
        amap = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        amap = cv2.GaussianBlur(amap, (0, 0), max(1.0, w / 60))
        return Prediction(score=float(score), anomaly_map=amap)
