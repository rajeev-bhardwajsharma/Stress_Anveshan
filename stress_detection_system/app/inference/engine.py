"""
app/inference/engine.py
Layers 3-6 of the pipeline:
  - Sensor availability mapping
  - Model group eligibility
  - Per-model inference (scale → predict)
  - Ensemble averaging
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

# Required sensors per group (must ALL be present for group to be eligible)
_CHEST_REQUIRED = set(settings.chest_all_sensors)   # {"ACC","ECG","EMG","EDA","Temp","Resp"}
_WRIST_REQUIRED = set(settings.wrist_all_sensors)   # {"ACC","BVP","EDA","TEMP"}


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def _eligible_groups(req: PredictRequest) -> List[str]:
    avail_chest = set(req.available_sensors.chest)
    avail_wrist = set(req.available_sensors.wrist)

    groups: List[str] = []
    if _CHEST_REQUIRED.issubset(avail_chest):
        groups.append("CHEST_ALL")
    if _WRIST_REQUIRED.issubset(avail_wrist):
        groups.append("WRIST_ALL")
    return groups


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
    # Build feature vector in the order the scaler saw during training.
    # scaler.feature_names_in_ is set when fitted on a DataFrame with column names.
    scaler = model.scaler
    if hasattr(scaler, "feature_names_in_"):
        col_order = list(scaler.feature_names_in_)
    else:
        # Fallback: sort keys (must match training-time sort)
        col_order = sorted(bundle_features.keys())

    x = np.array([bundle_features[c] for c in col_order], dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)

    # Keras model may expect 3-D input (batch, timesteps, features) for RNN/CNN
    input_shape = model.keras_model.input_shape
    if len(input_shape) == 3:
        x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[-1])

    preds = model.keras_model.predict(x_scaled, verbose=0)   # shape (1, 2) or (1, 1)
    preds = np.squeeze(preds)

    if preds.ndim == 0:
        # Single sigmoid output → stress probability
        stress_p = float(preds)
        return stress_p, 1.0 - stress_p
    else:
        # Softmax [non_stress, stress]
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
    Raises ValueError if no eligible model group found.
    """
    eligible = _eligible_groups(req)
    if not eligible:
        raise ValueError(
            "No eligible model group. "
            "Provide all chest sensors (ACC,ECG,EMG,EDA,Temp,Resp) "
            "and/or all wrist sensors (ACC,BVP,EDA,TEMP)."
        )

    stress_probs: List[float] = []
    models_used: List[str] = []

    for group in eligible:
        group_bundles = registry.bundles_for_group(group)
        if not group_bundles:
            logger.warning(f"Group {group} eligible but no models loaded — skipping")
            continue

        for bundle_name in group_bundles:
            if bundle_name not in bundles:
                logger.warning(f"Bundle '{bundle_name}' not in extracted features — skipping")
                continue

            loaded = registry.get(bundle_name)
            if loaded is None:
                continue

            try:
                stress_p, _ = _predict_one(loaded, bundles[bundle_name])
                stress_probs.append(stress_p)
                logger.debug(f"Bundle={bundle_name} stress_p={stress_p:.4f}")
            except Exception as e:
                logger.error(f"Inference failed for bundle '{bundle_name}': {e}")

        if group_bundles:
            models_used.append(group)

    if not stress_probs:
        raise RuntimeError("All models failed during inference — cannot produce prediction.")

    # Ensemble: simple average
    avg_stress = float(np.mean(stress_probs))
    avg_non_stress = 1.0 - avg_stress
    prediction = "STRESSED" if avg_stress >= _THRESHOLD else "NOT_STRESSED"
    confidence = avg_stress if prediction == "STRESSED" else avg_non_stress

    logger.info(
        f"Ensemble | models_used={models_used} | n_models={len(stress_probs)} | "
        f"stress_p={avg_stress:.4f} | prediction={prediction}"
    )

    return PredictResponse(
        subject_id=req.subject_id,
        timestamp=req.timestamp,
        models_used=models_used,
        stress_probability=round(avg_stress, 4),
        non_stress_probability=round(avg_non_stress, 4),
        prediction=prediction,
        confidence_score=round(confidence, 4),
    )
