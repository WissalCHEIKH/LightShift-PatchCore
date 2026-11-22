"""Stockage des seuils recalibrés (un par backend / méthode / catégorie)."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ThresholdStore:
    def __init__(self, state_dir: Path):
        self._path = Path(state_dir) / "thresholds.json"
        self._lock = threading.Lock()
        self._data = self._load()

    @staticmethod
    def key(backend: str, method: str, category: str) -> str:
        return f"{backend}:{method}:{category}"

    def get(self, backend: str, method: str, category: str) -> Optional[dict]:
        return self._data.get(self.key(backend, method, category))

    def set(self, backend: str, method: str, category: str, threshold: float,
            alpha: float, n_images: int) -> dict:
        entry = {
            "threshold": threshold,
            "alpha": alpha,
            "n_images": n_images,
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._data[self.key(backend, method, category)] = entry
            self._save()
        return entry

    def reset(self, backend: str, method: str, category: str) -> bool:
        with self._lock:
            removed = self._data.pop(self.key(backend, method, category), None) is not None
            self._save()
        return removed

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
