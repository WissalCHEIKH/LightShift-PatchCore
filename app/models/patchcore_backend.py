"""Branchement des vrais modèles PatchCore (checkpoints anomalib 2.7.1).

Les checkpoints sont ceux produits par src/models/train_patchcore.py :
    results/Patchcore/MVTecAD/<category>/latest/weights/lightning/model.ckpt

⚠️ À VALIDER dès réception des checkpoints de Wissal : ce code suit l'API
anomalib 2.x (forward = pre_processor -> modèle -> post_processor), mais n'a
pas pu être exécuté sans les fichiers model.ckpt.

Scores BRUTS : par défaut, l'application utilise la sortie du modèle avant le post-processor
d'anomalib. Le post-processor normalise les scores dans [0, 1] et les coupe : sous un fort changement
de lumière, toutes les images valent alors 1.0 et la recalibration (B3) ne peut plus rien distinguer.
La limite par défaut est la limite brute apprise par anomalib (post_processor.image_threshold).
SCORE_MODE=normalized rétablit les scores normalisés d'anomalib (limite 0.5).

Méthode "lightshift" : en attente du membre 2. Quand ses briques seront prêtes,
brancher son code dans _predict_lightshift() sans changer le reste de l'API.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

from app.models.base import Prediction


class PatchcoreModel:
    name = "patchcore"
    version = "patchcore-wrn50-l2l3-cs0.1-k9"

    def __init__(self, checkpoint_root: Path, categories: list[str], score_mode: str = "raw"):
        self.score_mode = score_mode
        self.checkpoint_root = Path(checkpoint_root)
        self.categories = categories
        self._models = {}
        self._lock = threading.Lock()

    def checkpoint_path(self, category: str) -> Path:
        return (self.checkpoint_root / category / "latest" / "weights"
                / "lightning" / "model.ckpt")

    def available_categories(self) -> list[str]:
        return [c for c in self.categories if self.checkpoint_path(c).exists()]

    def is_ready(self) -> bool:
        # Prêt dès qu'au moins un checkpoint est présent ; une catégorie sans checkpoint
        # renvoie une erreur 503 explicite au moment de l'analyse.
        return bool(self.available_categories())

    def warm_up(self, categories, methods) -> None:
        """Charge les checkpoints disponibles à l'avance (appelé en arrière-plan au démarrage)."""
        for category in self.available_categories():
            try:
                self._load(category)
            except Exception:   # une erreur sera signalée proprement lors de la première analyse
                pass

    def default_threshold(self, category: str, method: str) -> float:
        # Rappel : cette limite a été apprise sur le jeu de test (guide de passation §5),
        # d'où l'intérêt de /calibrate.
        if self.score_mode == "normalized":
            return 0.5
        return float(self._load(category).post_processor.image_threshold)

    def predict(self, image_rgb: np.ndarray, category: str, method: str) -> Prediction:
        if method == "lightshift":
            return self._predict_lightshift(image_rgb, category)
        return self._predict_baseline(image_rgb, category)

    # --- baseline -------------------------------------------------------------
    def _predict_baseline(self, image_rgb: np.ndarray, category: str) -> Prediction:
        import torch

        model = self._load(category)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        with torch.no_grad():
            if self.score_mode == "normalized":
                output = model(tensor)                                   # pre -> modèle -> post
                map_threshold = 0.5
            else:
                batch = model.pre_processor(tensor) if model.pre_processor else tensor
                output = model.model(batch)                              # scores bruts, sans écrêtage
                map_threshold = float(model.post_processor.pixel_threshold)

        score = float(_first(output, "pred_score").reshape(-1)[0])
        amap = _first(output, "anomaly_map").squeeze().cpu().numpy().astype(np.float32)
        h, w = image_rgb.shape[:2]
        if amap.shape != (h, w):
            amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        return Prediction(score=score, anomaly_map=amap, map_threshold=map_threshold)

    # --- méthode améliorée (membre 2) -------------------------------------------
    def _predict_lightshift(self, image_rgb: np.ndarray, category: str) -> Prediction:
        raise NotImplementedError(
            "La méthode LightShift n'est pas encore branchée sur le vrai PatchCore : elle arrivera avec "
            "les briques B1/B2 du membre 2. En attendant, la recalibration (B3) corrige les fausses alertes."
        )

    def _load(self, category: str):
        with self._lock:
            if category not in self._models:
                path = self.checkpoint_path(category)
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint introuvable : {path}")
                from anomalib.models import Patchcore

                model = Patchcore.load_from_checkpoint(str(path), map_location="cpu")
                model.eval()
                self._models[category] = model
            return self._models[category]


def _first(output, name: str):
    value = getattr(output, name, None)
    if value is None and isinstance(output, dict):
        value = output.get(name)
    if value is None:
        raise RuntimeError(f"Sortie anomalib sans champ '{name}'.")
    return value
