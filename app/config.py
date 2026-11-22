"""Configuration de l'application, lue depuis les variables d'environnement."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["metal_nut", "screw", "transistor"]
METHODS = ["baseline", "lightshift"]
SHIFTS = ["brightness", "contrast", "gamma", "color_temperature", "shading"]
SEVERITIES = [1, 2, 3]

# "mock" = faux modèle (aucune dépendance lourde), "patchcore" = vrais checkpoints anomalib
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "mock").lower()

# Dossier des checkpoints produits par src/models/train_patchcore.py
CHECKPOINT_ROOT = Path(
    os.getenv("CHECKPOINT_ROOT", REPO_ROOT / "results" / "Patchcore" / "MVTecAD")
)

# Vrai PatchCore : "raw" (scores bruts, recommandé pour la recalibration) ou "normalized" (0..1 d'anomalib)
SCORE_MODE = os.getenv("SCORE_MODE", "raw").lower()

# Vraies photos MVTec AD (téléchargées par chacun, non versionnées)
MVTEC_ROOT = Path(os.getenv("MVTEC_ROOT", REPO_ROOT / "data" / "raw" / "mvtec_anomaly_detection"))

# D'où viennent les pièces des boutons « Essayer » :
#   auto      -> photos MVTec si le vrai PatchCore est branché et le dataset présent, sinon pièces dessinées
#   mvtec     -> toujours MVTec (erreur si absent)
#   synthetic -> toujours les pièces dessinées
EXAMPLES_SOURCE = os.getenv("EXAMPLES_SOURCE", "auto").lower()

# Fichier où sont stockés les seuils recalibrés via /calibrate
STATE_DIR = Path(os.getenv("STATE_DIR", REPO_ROOT / "app_state"))

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 10 * 1024 * 1024))  # 10 Mo
MAX_CALIBRATION_IMAGES = int(os.getenv("MAX_CALIBRATION_IMAGES", 200))
