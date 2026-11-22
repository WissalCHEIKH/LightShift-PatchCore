"""Pièces d'exemple tirées des vraies photos MVTec AD (arborescence simulée)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture
def fake_mvtec(tmp_path, monkeypatch):
    root = tmp_path / "mvtec"
    for folder, n in [("good", 10), ("scratch_head", 3), ("thread_top", 2)]:
        d = root / "screw" / "test" / folder
        d.mkdir(parents=True)
        for i in range(n):
            img = np.full((64, 64, 3), 100 + i, np.uint8)
            cv2.imwrite(str(d / f"{i:03d}.png"), img)
    (root / "screw" / "train" / "good").mkdir(parents=True)   # ne doit jamais être utilisé
    monkeypatch.setattr("app.config.MVTEC_ROOT", root)
    return root


def names(client, kind, n, **params):
    return [client.get("/examples/screw", params={"kind": kind, "seed": i, **params}).headers["X-Example-Name"]
            for i in range(n)]


def test_auto_mode_keeps_drawn_parts_with_demo_model(fake_mvtec, tmp_path):
    c = TestClient(create_app(backend="mock", state_dir=tmp_path))
    assert c.get("/health").json()["examples_source"]["screw"] == "synthetic"
    assert c.get("/examples/screw").headers["X-Example-Source"] == "synthetic"


def test_forced_mvtec_mode_serves_real_photos(fake_mvtec, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.EXAMPLES_SOURCE", "mvtec")
    c = TestClient(create_app(backend="mock", state_dir=tmp_path))
    assert c.get("/health").json()["examples_source"]["screw"] == "mvtec"

    r = c.get("/examples/screw", params={"kind": "bad", "seed": 0})
    assert r.status_code == 200 and r.headers["X-Example-Source"] == "mvtec"
    assert r.headers["X-Example-Defect"] in {"scratch_head", "thread_top"}
    assert "/test/" in r.headers["X-Example-Name"]


def test_trial_and_calibration_photos_never_overlap(fake_mvtec, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.EXAMPLES_SOURCE", "mvtec")
    c = TestClient(create_app(backend="mock", state_dir=tmp_path))
    trial, calib = set(names(c, "good", 10)), set(names(c, "calibration", 5))
    assert len(trial) == 5 and len(calib) == 5          # 10 photos good -> deux moitiés
    assert trial.isdisjoint(calib)
    assert all("train" not in n for n in trial | calib)


def test_specific_defect_type(fake_mvtec, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.EXAMPLES_SOURCE", "mvtec")
    c = TestClient(create_app(backend="mock", state_dir=tmp_path))
    assert set(names(c, "bad", 4, defect="thread_top")) == {"screw/test/thread_top/000.png",
                                                            "screw/test/thread_top/001.png"}


def test_missing_category_photos_gives_clear_404(fake_mvtec, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.EXAMPLES_SOURCE", "mvtec")
    c = TestClient(create_app(backend="mock", state_dir=tmp_path))
    r = c.get("/examples/transistor", params={"kind": "good"})
    assert r.status_code == 404 and "transistor" in r.json()["detail"]


def test_calibration_pool_never_repeats_a_photo(fake_mvtec, tmp_path, monkeypatch):
    """10 photos good -> 5 réservées à la calibration : la 6e demande est refusée, pas recyclée."""
    monkeypatch.setattr("app.config.EXAMPLES_SOURCE", "mvtec")
    c = TestClient(create_app(backend="mock", state_dir=tmp_path))
    assert c.get("/health").json()["calibration_pool"]["screw"] == 5
    assert c.get("/examples/screw", params={"kind": "calibration", "seed": 4}).status_code == 200
    r = c.get("/examples/screw", params={"kind": "calibration", "seed": 5})
    assert r.status_code == 404 and "5 photos" in r.json()["detail"]
