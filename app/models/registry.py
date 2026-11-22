"""Choix du modèle selon MODEL_BACKEND (une seule ligne à changer pour passer au vrai)."""

from app import config


def build_model(backend: str | None = None):
    backend = (backend or config.MODEL_BACKEND).lower()
    if backend == "mock":
        from app.models.mock import MockModel
        return MockModel()
    if backend == "patchcore":
        from app.models.patchcore_backend import PatchcoreModel
        return PatchcoreModel(config.CHECKPOINT_ROOT, config.CATEGORIES, config.SCORE_MODE)
    raise ValueError(f"MODEL_BACKEND inconnu : {backend} (attendu : mock ou patchcore)")
