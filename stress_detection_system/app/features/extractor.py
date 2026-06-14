"""
app/features/extractor.py

Thin orchestration layer over your existing feature modules.
Each `extract_*` function returns Dict[str, float].
Bundles are keyed exactly as they appear in config (chest_statistical, etc.)
so the model loader can align them directly.

Import paths assume your feature modules are importable from the Python path.
If they live elsewhere, adjust sys.path in main.py before this is imported.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.core.logging import get_logger
from app.core.settings import settings
from app.schemas.predict import PredictRequest

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for your feature modules.
# Point FEATURE_MODULE_DIR at wherever statistical_features.py etc. live.
# Override via env var WESAD_FEATURE_MODULE_DIR if needed.
# ---------------------------------------------------------------------------
import os

_FEATURE_MODULE_DIR = Path(
    os.environ.get(
        "WESAD_FEATURE_MODULE_DIR",
        "/home/rs/ml-projects/WDM_dataset"   # default — change if different
    )
)

if str(_FEATURE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURE_MODULE_DIR))


def _import_feature_modules():
    """Import feature modules lazily so startup doesn't fail if path wrong."""
    try:
        from statistical_features import extract_statistical_features_all_signals
        from heart_features import extract_heart_features
        from eda_features import extract_eda_features
        from acc_features import extract_acc_features
        from emg_features import extract_emg_features
        return (
            extract_statistical_features_all_signals,
            extract_heart_features,
            extract_eda_features,
            extract_acc_features,
            extract_emg_features,
        )
    except ImportError as e:
        logger.error(f"Could not import feature modules from {_FEATURE_MODULE_DIR}: {e}")
        raise


# ---------------------------------------------------------------------------
# Signal extraction helpers — pull numpy arrays from request
# ---------------------------------------------------------------------------

def _chest_signal_map(req: PredictRequest) -> Dict[str, Optional[np.ndarray]]:
    cs = req.signals.chest
    if cs is None:
        return {}
    return {
        "chest_ECG":  np.array(cs.ECG,  dtype=np.float32) if cs.ECG  else None,
        "chest_EDA":  np.array(cs.EDA,  dtype=np.float32) if cs.EDA  else None,
        "chest_EMG":  np.array(cs.EMG,  dtype=np.float32) if cs.EMG  else None,
        "chest_RESP": np.array(cs.Resp, dtype=np.float32) if cs.Resp else None,
        "chest_TEMP": np.array(cs.Temp, dtype=np.float32) if cs.Temp else None,
        "chest_ACC":  np.array(cs.ACC,  dtype=np.float32) if cs.ACC  else None,
    }


def _wrist_signal_map(req: PredictRequest) -> Dict[str, Optional[np.ndarray]]:
    ws = req.signals.wrist
    if ws is None:
        return {}
    return {
        "wrist_BVP":  np.array(ws.BVP,  dtype=np.float32) if ws.BVP  else None,
        "wrist_ACC":  np.array(ws.ACC,   dtype=np.float32) if ws.ACC  else None,
        "wrist_EDA":  np.array(ws.EDA,   dtype=np.float32) if ws.EDA  else None,
        "wrist_TEMP": np.array(ws.TEMP,  dtype=np.float32) if ws.TEMP else None,
    }


# ---------------------------------------------------------------------------
# Per-bundle extraction
# ---------------------------------------------------------------------------

def _extract_statistical(signals: Dict[str, Optional[np.ndarray]], device: str, fns) -> Dict[str, float]:
    (extract_statistical, *_) = fns
    present = {k: v for k, v in signals.items() if v is not None}
    sensor_fs = settings.sampling_rates
    return extract_statistical(present, sensor_fs)


def _extract_heart(signals: Dict[str, Optional[np.ndarray]], device: str, fns) -> Optional[Dict[str, float]]:
    (_, extract_heart, *_) = fns
    if device == "chest":
        sig = signals.get("chest_ECG")
        name, fs = "chest_ECG", settings.sampling_rates["chest_ECG"]
    else:
        sig = signals.get("wrist_BVP")
        name, fs = "wrist_BVP", settings.sampling_rates["wrist_BVP"]
    if sig is None:
        return None
    return extract_heart(sig, name, fs)


def _extract_eda(signals: Dict[str, Optional[np.ndarray]], device: str, fns) -> Optional[Dict[str, float]]:
    (*_, extract_eda, _e, _em) = fns
    if device == "chest":
        sig = signals.get("chest_EDA")
        name, fs = "chest_EDA", settings.sampling_rates["chest_EDA"]
    else:
        sig = signals.get("wrist_EDA")
        name, fs = "wrist_EDA", settings.sampling_rates["wrist_EDA"]
    if sig is None:
        return None
    return extract_eda(sig, name, fs)


def _extract_acc(signals: Dict[str, Optional[np.ndarray]], device: str, fns) -> Optional[Dict[str, float]]:
    (*_, extract_acc, _em) = fns
    if device == "chest":
        sig = signals.get("chest_ACC")
        name, fs = "chest_ACC", settings.sampling_rates["chest_ACC"]
    else:
        sig = signals.get("wrist_ACC")
        name, fs = "wrist_ACC", settings.sampling_rates["wrist_ACC"]
    if sig is None:
        return None
    return extract_acc(sig, name, fs)


def _extract_emg(signals: Dict[str, Optional[np.ndarray]], fns) -> Optional[Dict[str, float]]:
    (*_, extract_emg) = fns
    sig = signals.get("chest_EMG")
    if sig is None:
        return None
    return extract_emg(sig, "chest_EMG", settings.sampling_rates["chest_EMG"])


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_all_bundles(req: PredictRequest) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict keyed by bundle name:
        {
          "chest_statistical": {...},
          "chest_heart":       {...},
          ...
        }
    Only bundles for which all required signals are present are included.
    """
    fns = _import_feature_modules()

    chest_sigs = _chest_signal_map(req)
    wrist_sigs = _wrist_signal_map(req)

    bundles: Dict[str, Dict[str, float]] = {}

    # ---- Chest bundles ----
    if chest_sigs:
        try:
            bundles["chest_statistical"] = _extract_statistical(chest_sigs, "chest", fns)
        except Exception as e:
            logger.warning(f"chest_statistical extraction failed: {e}")

        result = _extract_heart(chest_sigs, "chest", fns)
        if result:
            bundles["chest_heart"] = result

        result = _extract_eda(chest_sigs, "chest", fns)
        if result:
            bundles["chest_eda"] = result

        result = _extract_acc(chest_sigs, "chest", fns)
        if result:
            bundles["chest_acc"] = result

        result = _extract_emg(chest_sigs, fns)
        if result:
            bundles["chest_emg"] = result

    # ---- Wrist bundles ----
    if wrist_sigs:
        try:
            bundles["wrist_statistical"] = _extract_statistical(wrist_sigs, "wrist", fns)
        except Exception as e:
            logger.warning(f"wrist_statistical extraction failed: {e}")

        result = _extract_heart(wrist_sigs, "wrist", fns)
        if result:
            bundles["wrist_heart"] = result

        result = _extract_eda(wrist_sigs, "wrist", fns)
        if result:
            bundles["wrist_eda"] = result

        result = _extract_acc(wrist_sigs, "wrist", fns)
        if result:
            bundles["wrist_acc"] = result

    logger.info(f"Extracted bundles: {list(bundles.keys())}")
    return bundles
