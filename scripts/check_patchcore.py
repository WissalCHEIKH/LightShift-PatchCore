"""Vérifie le branchement du vrai PatchCore dans l'application (checklist du membre 3).

À lancer depuis la racine du projet, une fois :
  - les checkpoints placés dans results/Patchcore/MVTecAD/<category>/latest/weights/lightning/model.ckpt
  - MVTec AD placé dans data/raw/mvtec_anomaly_detection/
  - requirements.txt installé (torch + anomalib)

    python scripts/check_patchcore.py
    python scripts/check_patchcore.py --category screw --n 10

Le script n'écrit rien dans le projet : il utilise un dossier temporaire pour les seuils.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_BACKEND", "patchcore")
os.environ.setdefault("WARM_UP", "0")
os.environ["STATE_DIR"] = tempfile.mkdtemp(prefix="lightshift-check-")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.api.main import create_app  # noqa: E402
from app.core import mvtec  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=config.CATEGORIES, nargs="*", default=config.CATEGORIES)
    parser.add_argument("--n", type=int, default=5, help="photos saines et défectueuses testées par catégorie")
    args = parser.parse_args()

    client = TestClient(create_app())
    health = client.get("/health").json()
    print(f"/health : {health['status']}  modèle={health['model_backend']}  ({health['model_version']})")
    ok = health["status"] == "ok" and health["model_backend"] == "patchcore"

    for cat in args.category:
        print(f"\n=== {cat}")
        if not mvtec.is_available(cat):
            print(f"  ✗ photos MVTec introuvables dans {mvtec.category_root(cat)}")
            ok = False
            continue
        scores = {"good": [], "bad": []}
        for kind in ("good", "bad"):
            pool = mvtec.demo_good(cat) if kind == "good" else mvtec.defect_images(cat)
            step = max(1, len(pool) // args.n)
            for path in pool[::step][: args.n]:
                t = time.perf_counter()
                r = client.post("/predict", data={"category": cat},
                                files={"image": (path.name, path.read_bytes(), "image/png")})
                ms = (time.perf_counter() - t) * 1000
                if r.status_code != 200:
                    print(f"  ✗ {mvtec.label(path)} -> {r.status_code} {r.json().get('detail')}")
                    ok = False
                    break
                j = r.json()
                scores[kind].append(j["score"])
                print(f"  {kind:4s} {mvtec.label(path):45s} score={j['score']:.3f} limite={j['threshold']:.3f} "
                      f"{j['decision']:8s} {ms:6.0f} ms")
        if scores["good"] and scores["bad"]:
            good_ok = sum(s <= j["threshold"] for s in scores["good"])
            bad_ok = sum(s > j["threshold"] for s in scores["bad"])
            mean_g, mean_b = sum(scores["good"]) / len(scores["good"]), sum(scores["bad"]) / len(scores["bad"])
            print(f"  -> saines jugées conformes : {good_ok}/{len(scores['good'])}, "
                  f"défauts détectés : {bad_ok}/{len(scores['bad'])}, "
                  f"score moyen saines {mean_g:.3f} < défauts {mean_b:.3f} : {'oui' if mean_g < mean_b else 'NON'}")
            ok = ok and mean_g < mean_b

    lat = client.get("/metrics").json()["latency_ms"]
    print(f"\nLatence CPU : p50 = {lat['p50']} ms, p95 = {lat['p95']} ms (sur {lat['window']} analyses)")
    print("\nRÉSULTAT :", "✓ branchement OK" if ok else "✗ à corriger (voir les lignes marquées ✗)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
