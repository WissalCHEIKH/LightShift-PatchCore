"""Accès aux vraies photos MVTec AD présentes sur la machine (data/raw/, non versionné).

Organisation attendue (celle du membre 1) :
    data/raw/mvtec_anomaly_detection/<category>/test/good/*.png
    data/raw/mvtec_anomaly_detection/<category>/test/<type_de_défaut>/*.png

Règle de séparation pour la démo :
  - les photos saines de test/good sont coupées en deux moitiés fixes et disjointes :
    indices pairs  -> réservées à la recalibration (/calibrate) ;
    indices impairs -> proposées par le bouton « Essayer une pièce saine ».
  - train/good n'est JAMAIS utilisé : ces photos ont servi à construire la mémoire de PatchCore.
Une même photo ne sert donc jamais à la fois à calibrer et à être jugée.
"""

from pathlib import Path
from typing import Optional

from app import config

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def category_root(category: str) -> Path:
    return config.MVTEC_ROOT / category / "test"


def is_available(category: str) -> bool:
    return bool(_images(category_root(category) / "good"))


def defect_types(category: str) -> list[str]:
    root = category_root(category)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != "good" and _images(d))


def demo_good(category: str) -> list[Path]:
    return _images(category_root(category) / "good")[1::2]


def calibration_good(category: str) -> list[Path]:
    return _images(category_root(category) / "good")[0::2]


def defect_images(category: str, defect: Optional[str] = None) -> list[Path]:
    types = [defect] if defect else defect_types(category)
    return [p for t in types for p in _images(category_root(category) / t)]


def pick(category: str, kind: str, seed: int, defect: Optional[str] = None) -> Path:
    """kind : "good" (essai), "bad" (défaut réel), "calibration" (pièce saine réservée)."""
    pool = {"good": lambda: demo_good(category),
            "calibration": lambda: calibration_good(category),
            "bad": lambda: defect_images(category, defect)}[kind]()
    if not pool:
        raise FileNotFoundError(f"Aucune photo MVTec '{kind}' pour {category} dans {category_root(category)}")
    if kind == "calibration" and seed >= len(pool):
        # Pas de reprise en boucle : une photo comptée deux fois fausserait la garantie de B3.
        raise FileNotFoundError(
            f"Seulement {len(pool)} photos saines MVTec sont réservées à la recalibration pour {category}.")
    return pool[seed % len(pool)]


def label(path: Path) -> str:
    """Ex. 'screw/test/scratch_head/012.png'."""
    try:
        return path.relative_to(config.MVTEC_ROOT).as_posix()
    except ValueError:
        return path.name


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
