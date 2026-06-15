"""
app/inference/validator.py
Layer 1: structural and content validation of the incoming request.

Key change from original
------------------------
The original validator only checked that declared-available sensors had valid
data. That was correct. The problem was in engine.py requiring ALL sensors.
This validator is unchanged in logic but now includes a clearer check:
  - If a sensor is declared available AND data is provided → validate shape/NaN
  - If a sensor is declared available BUT data is missing → error (missing)
  - If a sensor is NOT declared available but data IS provided → error (inconsistent)
  - If a sensor is NOT declared and no data → fine, that sensor just won't be used

The validator does NOT require all possible sensors to be present.
Partial sensor sets are valid as long as they are self-consistent.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.core.settings import settings
from app.schemas.predict import PredictRequest, ValidationResponse

_FS  = settings.sampling_rates
_WIN = settings.inference.window_size_sec

# Expected sample counts per signal for a 60-second window
_EXPECTED_SAMPLES = {k: v * _WIN for k, v in _FS.items()}


def _check_signal(name: str, arr, errors: List[str]) -> None:
    """Validate shape, length, and NaN content of a single signal array."""
    if arr is None:
        return
    a = np.asarray(arr, dtype=np.float32)
    expected = _EXPECTED_SAMPLES.get(name)
    if expected is None:
        return

    if name in ("chest_ACC", "wrist_ACC"):
        if a.ndim != 2 or a.shape[1] != 3:
            errors.append(f"{name}: expected shape (N,3), got {a.shape}")
        elif a.shape[0] != expected:
            errors.append(f"{name}: expected {expected} rows, got {a.shape[0]}")
    else:
        if a.ndim != 1:
            errors.append(f"{name}: expected 1-D array, got shape {a.shape}")
        elif len(a) != expected:
            errors.append(f"{name}: expected {expected} samples, got {len(a)}")

    if np.isnan(a).any():
        errors.append(f"{name}: contains NaN values")


def validate_request(req: PredictRequest) -> Tuple[ValidationResponse, bool]:
    """
    Returns (ValidationResponse, ok: bool).
    ok=True means the request passed all checks.

    Rules
    -----
    1. At least ONE sensor must be declared and supplied (can't predict with nothing).
    2. For every sensor in available_sensors: data must be provided and valid.
    3. Data provided for a sensor NOT in available_sensors is flagged as inconsistent.
    4. Partial sensor sets (e.g. only ECG, no EMG) are perfectly fine — the engine
       will use whichever bundle models are satisfiable.
    """
    errors: List[str]  = []
    missing: List[str] = []
    invalid: List[str] = []

    avail_chest = set(req.available_sensors.chest)
    avail_wrist = set(req.available_sensors.wrist)

    cs = req.signals.chest
    ws = req.signals.wrist

    # ── Chest signal validation ───────────────────────────────────────────
    chest_map = {
        "ECG":  ("chest_ECG",  cs.ECG  if cs else None),
        "EDA":  ("chest_EDA",  cs.EDA  if cs else None),
        "EMG":  ("chest_EMG",  cs.EMG  if cs else None),
        "Resp": ("chest_RESP", cs.Resp if cs else None),
        "Temp": ("chest_TEMP", cs.Temp if cs else None),
        "ACC":  ("chest_ACC",  cs.ACC  if cs else None),
    }
    for sensor_key, (sig_name, data) in chest_map.items():
        if sensor_key in avail_chest:
            if data is None:
                missing.append(f"chest.{sensor_key}")
            else:
                _check_signal(sig_name, data, errors)
        elif data is not None:
            invalid.append(
                f"chest.{sensor_key} (data provided but not listed in available_sensors)"
            )

    # ── Wrist signal validation ───────────────────────────────────────────
    wrist_map = {
        "BVP":  ("wrist_BVP",  ws.BVP  if ws else None),
        "EDA":  ("wrist_EDA",  ws.EDA  if ws else None),
        "TEMP": ("wrist_TEMP", ws.TEMP if ws else None),
        "ACC":  ("wrist_ACC",  ws.ACC  if ws else None),
    }
    for sensor_key, (sig_name, data) in wrist_map.items():
        if sensor_key in avail_wrist:
            if data is None:
                missing.append(f"wrist.{sensor_key}")
            else:
                _check_signal(sig_name, data, errors)
        elif data is not None:
            invalid.append(
                f"wrist.{sensor_key} (data provided but not listed in available_sensors)"
            )

    # ── Must have at least something to work with ─────────────────────────
    total_declared = len(avail_chest) + len(avail_wrist)
    if total_declared == 0:
        errors.append(
            "available_sensors is empty for both chest and wrist. "
            "Declare at least one sensor."
        )

    all_errors = missing + invalid + errors
    ok = len(all_errors) == 0

    return (
        ValidationResponse(
            validation_status="passed" if ok else "failed",
            missing_sensors=missing,
            invalid_sensors=invalid,
            errors=errors,
        ),
        ok,
    )