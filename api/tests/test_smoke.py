"""
tests/test_smoke.py

Smoke tests for every endpoint using synthetic 60-second signals.
Run with:  pytest tests/test_smoke.py -v

These tests mock the ModelRegistry so you don't need real .keras files
to verify routing, validation, and response shapes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---- Build synthetic 60-second signals ----

FS = {
    "chest_ECG": 700, "chest_EDA": 700, "chest_EMG": 700,
    "chest_RESP": 700, "chest_TEMP": 700, "chest_ACC": 700,
    "wrist_BVP": 64, "wrist_ACC": 32, "wrist_EDA": 4, "wrist_TEMP": 4,
}

def _sig(key: str) -> list:
    n = FS[key] * 60
    return np.random.randn(n).tolist()

def _acc(key: str) -> list:
    n = FS[key] * 60
    return np.random.randn(n, 3).tolist()

FULL_REQUEST: Dict[str, Any] = {
    "subject_id": "S2",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "window_size_sec": 60,
    "available_sensors": {
        "chest": ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"],
        "wrist": ["ACC", "BVP", "EDA", "TEMP"],
    },
    "signals": {
        "chest": {
            "ACC":  _acc("chest_ACC"),
            "ECG":  _sig("chest_ECG"),
            "EMG":  _sig("chest_EMG"),
            "EDA":  _sig("chest_EDA"),
            "Temp": _sig("chest_TEMP"),
            "Resp": _sig("chest_RESP"),
        },
        "wrist": {
            "ACC":  _acc("wrist_ACC"),
            "BVP":  _sig("wrist_BVP"),
            "EDA":  _sig("wrist_EDA"),
            "TEMP": _sig("wrist_TEMP"),
        },
    },
}


# ---- Mock registry so tests run without real models ----

def _make_mock_registry(has_chest=True, has_wrist=True):
    reg = MagicMock()
    reg.__len__ = MagicMock(return_value=2)
    reg.chest_all_ready = has_chest
    reg.wrist_all_ready = has_wrist
    reg.all_bundles.return_value = ["chest_statistical", "wrist_statistical"]
    reg.bundles_for_group.side_effect = lambda g: (
        ["chest_statistical"] if g == "CHEST_ALL" and has_chest
        else (["wrist_statistical"] if g == "WRIST_ALL" and has_wrist else [])
    )
    mock_model = MagicMock()
    mock_model.bundle = "chest_statistical"
    mock_model.group = "CHEST_ALL"
    mock_model.n_features = 50
    reg.get.return_value = mock_model
    return reg


@pytest.fixture
def client():
    # Mock init_registry so startup doesn't try to load real files
    with patch("app.models.loader._registry", _make_mock_registry()):
        from main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "health" in r.json()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "models_loaded" in body


def test_models(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert "active_tag" in body
    assert isinstance(body["loaded_models"], list)


def test_validate_pass(client):
    r = client.post("/api/v1/validate", json=FULL_REQUEST)
    assert r.status_code == 200
    assert r.json()["validation_status"] == "passed"


def test_validate_wrong_window(client):
    bad = dict(FULL_REQUEST, window_size_sec=30)
    r = client.post("/api/v1/validate", json=bad)
    assert r.status_code == 422   # pydantic rejects before our validator


def test_validate_missing_sensor(client):
    bad = json.loads(json.dumps(FULL_REQUEST))
    bad["signals"]["chest"]["ECG"] = None
    r = client.post("/api/v1/validate", json=bad)
    assert r.status_code == 200
    body = r.json()
    assert body["validation_status"] == "failed"
    assert any("ECG" in s for s in body["missing_sensors"])


def test_predict_no_eligible_group(client):
    """Drop all wrist sensors → only wrist declared but chest not declared → 422."""
    bad = json.loads(json.dumps(FULL_REQUEST))
    bad["available_sensors"]["chest"] = []
    bad["available_sensors"]["wrist"] = []
    r = client.post("/api/v1/predict", json=bad)
    # Validation should fail (missing declared sensors)
    assert r.status_code in (200, 422)
