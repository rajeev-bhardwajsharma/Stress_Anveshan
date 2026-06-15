"""
app/inference/engine.py
Layers 3-6 of the pipeline:
  - Sensor availability mapping
  - Per-BUNDLE eligibility  ← key change: each bundle checked independently
  - Per-model inference (scale → predict)
  - Ensemble averaging across all eligible bundles

Design intent
-------------
The original design required ALL sensors in a group (chest or wrist) to be
present before ANY model could run. This defeated the purpose of training
per-bundle models — the whole point is that each bundle (chest_heart,
wrist_eda, etc.) is independently useful and can contribute a probability
even when other sensors are missing.

Fix: map each bundle → its required sensors, check availability per-bundle,
run every bundle whose sensors are present, then ensemble-average.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app.core.logging import get_logger
from app.core.settings import settings
from app.models.loader import LoadedModel, ModelRegistry
from app.schemas.predict import PredictRequest, PredictResponse

logger = get_logger(__name__)

_THRESHOLD = settings.inference.decision_threshold


# ---------------------------------------------------------------------------
# Bundle → required sensors mapping
# ---------------------------------------------------------------------------
# Each entry lists the sensors that MUST be present in available_sensors
# for that bundle's model to be usable.
# Add or adjust entries here if your bundle set changes.

_BUNDLE_REQUIRED_CHEST: Dict[str, List[str]] = {
    "chest_statistical": ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC"],
    "chest_heart":       ["ECG"],
    "chest_eda":         ["EDA"],
    "chest_emg":         ["EMG"],
    "chest_acc":         ["ACC"],
}

_BUNDLE_REQUIRED_WRIST: Dict[str, List[str]] = {
    "wrist_statistical": ["BVP", "EDA", "TEMP", "ACC"],
    "wrist_heart":       ["BVP"],
    "wrist_eda":         ["EDA"],
    "wrist_acc":         ["ACC"],
}


# ---------------------------------------------------------------------------
# Eligibility — per bundle, not per group
# ---------------------------------------------------------------------------

def _eligible_bundles(req: PredictRequest) -> List[str]:
    """
    Return every bundle name whose required sensors are all declared
    as available in the request.  Each bundle is checked independently.
    """
    avail_chest = set(req.available_sensors.chest)
    avail_wrist = set(req.available_sensors.wrist)

    eligible: List[str] = []

    for bundle, required in _BUNDLE_REQUIRED_CHEST.items():
        if all(s in avail_chest for s in required):
            eligible.append(bundle)

    for bundle, required in _BUNDLE_REQUIRED_WRIST.items():
        if all(s in avail_wrist for s in required):
            eligible.append(bundle)

    return eligible


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------

def _predict_one(
    model: LoadedModel,
    bundle_features: Dict[str, float],
) -> Tuple[float, float]:
    """
    Scale → reshape → predict with one Keras model.
    Returns (stress_prob, non_stress_prob).
    """
    scaler = model.scaler
    if hasattr(scaler, "feature_names_in_"):
        col_order = list(scaler.feature_names_in_)
    else:
        col_order = sorted(bundle_features.keys())

    x = np.array([bundle_features[c] for c in col_order], dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)

    input_shape = model.keras_model.input_shape
    if len(input_shape) == 3:
        x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[-1])

    preds = model.keras_model.predict(x_scaled, verbose=0)
    preds = np.squeeze(preds)

    if preds.ndim == 0:
        stress_p = float(preds)
        return stress_p, 1.0 - stress_p
    else:
        return float(preds[1]), float(preds[0])


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def run_inference(
    req: PredictRequest,
    bundles: Dict[str, Dict[str, float]],
    registry: ModelRegistry,
) -> PredictResponse:
    """
    Full inference pipeline.

    - Determines which bundles are satisfiable given available sensors.
    - Runs every loaded model whose bundle is satisfiable.
    - Ensemble-averages stress probabilities across all contributing models.
    - Raises ValueError if no bundle at all is satisfiable.
    - Raises RuntimeError if bundles were eligible but all models crashed.
    """
    eligible = _eligible_bundles(req)

    if not eligible:
        raise ValueError(
            "No eligible bundle. You must supply at least one sensor group:\n"
            "  Chest bundles: ECG → chest_heart | EDA → chest_eda | "
            "EMG → chest_emg | ACC → chest_acc\n"
            "  Wrist bundles: BVP → wrist_heart | EDA → wrist_eda | "
            "ACC → wrist_acc\n"
            "Provide the matching signals and declare them in available_sensors."
        )

    logger.info(f"Eligible bundles: {eligible}")

    stress_probs: List[float] = []
    bundles_used: List[str]   = []

    for bundle_name in eligible:
        # Features for this bundle must have been extracted
        if bundle_name not in bundles:
            logger.warning(
                f"Bundle '{bundle_name}' is sensor-eligible but has no extracted "
                "features — skipping (check your feature extraction step)."
            )
            continue

        loaded = registry.get(bundle_name)
        if loaded is None:
            logger.warning(
                f"Bundle '{bundle_name}' is eligible but no model is loaded for it "
                "— skipping (check Models/<tag>/ directory)."
            )
            continue

        try:
            stress_p, _ = _predict_one(loaded, bundles[bundle_name])
            stress_probs.append(stress_p)
            bundles_used.append(bundle_name)
            logger.debug(f"Bundle={bundle_name}  stress_p={stress_p:.4f}")
        except Exception as e:
            logger.error(f"Inference failed for bundle '{bundle_name}': {e}")

    if not stress_probs:
        raise RuntimeError(
            "All eligible models failed during inference — cannot produce prediction. "
            f"Eligible bundles were: {eligible}"
        )

    # Simple average ensemble
    avg_stress     = float(np.mean(stress_probs))
    avg_non_stress = 1.0 - avg_stress
    prediction     = "STRESSED" if avg_stress >= _THRESHOLD else "NOT_STRESSED"
    confidence     = avg_stress if prediction == "STRESSED" else avg_non_stress

    logger.info(
        f"Ensemble | bundles_used={bundles_used} | n_models={len(stress_probs)} | "
        f"stress_p={avg_stress:.4f} | prediction={prediction}"
    )

    return PredictResponse(
        subject_id=req.subject_id,
        timestamp=req.timestamp,
        models_used=bundles_used,        # now reports bundle names, not group names
        stress_probability=round(avg_stress, 4),
        non_stress_probability=round(avg_non_stress, 4),
        prediction=prediction,
        confidence_score=round(confidence, 4),
    )