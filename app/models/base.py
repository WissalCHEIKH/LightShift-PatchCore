"""Interface commune à tous les modèles (contrat avec le membre 2).

Tout modèle branché dans l'application doit fournir :
    predict(image_rgb, category, method) -> Prediction
    default_threshold(category, method) -> float
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Prediction:
    score: float              # score d'anomalie image (plus haut = plus suspect)
    anomaly_map: np.ndarray   # carte (H, W) float, même taille que l'image ou redimensionnable
    map_threshold: Optional[float] = None  # limite pour colorer la carte (sinon : limite image)


class AnomalyModel(Protocol):
    name: str
    version: str

    def predict(self, image_rgb: np.ndarray, category: str, method: str) -> Prediction: ...

    def default_threshold(self, category: str, method: str) -> float: ...

    def is_ready(self) -> bool: ...
