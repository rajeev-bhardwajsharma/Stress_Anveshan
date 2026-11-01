"""
app/models/loader.py

Scans <models_dir>/<bundle_name>/ for:
    - final_model.keras  (or whatever model_filename is set to in config)
    - scaler.joblib      StandardScaler fit at training time
    - meta.json          training-time feature order (feature_names)
    - medians.npy        training-time per-feature medians for NaN imputation

Builds a ModelRegistry used by the inference engine.
Expected bundle subdirs (any subset may exist):
    chest_statistical, chest_heart, chest_eda, chest_acc, chest_emg,
    wrist_statistical, wrist_heart, wrist_eda, wrist_acc,
    chest_all, wrist_all, multimodal
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json

import joblib
import numpy as np

from app.core.logging import get_logger
from app.core.settings import settings

logger = get_logger(__name__)

# Groups that the inference engine cares about
CHEST_BUNDLES = {
    "chest_statistical", "chest_heart", "chest_eda", "chest_acc", "chest_emg"
}
WRIST_BUNDLES = {
    "wrist_statistical", "wrist_heart", "wrist_eda", "wrist_acc"
}

# Determine group membership
def _group_for(bundle: str) -> Optional[str]:
    if bundle in CHEST_BUNDLES or bundle == "chest_all":
        return "CHEST_ALL"
    if bundle in WRIST_BUNDLES or bundle == "wrist_all":
        return "WRIST_ALL"
    if bundle == "multimodal":
        return "MULTIMODAL"
    return None


@dataclass
class LoadedModel:
    bundle: str
    group: str                  # "CHEST_ALL" | "WRIST_ALL" | "MULTIMODAL"
    keras_model: object         # tf.keras.Model (typed as object to avoid tf import at top)
    scaler: object              # sklearn-compatible scaler
    n_features: int             # expected input width after scaling
    feature_names: Optional[List[str]] = None   # training column order (meta.json)
    medians: Optional[np.ndarray] = None        # training medians (medians.npy)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, LoadedModel] = {}   # bundle -> LoadedModel

    def register(self, lm: LoadedModel) -> None:
        self._models[lm.bundle] = lm
        logger.info(f"Registered model | bundle={lm.bundle} | group={lm.group} | n_features={lm.n_features}")

    def get(self, bundle: str) -> Optional[LoadedModel]:
        return self._models.get(bundle)

    def all_bundles(self) -> List[str]:
        return list(self._models.keys())

    def bundles_for_group(self, group: str) -> List[str]:
        return [b for b, m in self._models.items() if m.group == group]

    @property
    def chest_all_ready(self) -> bool:
        return len(self.bundles_for_group("CHEST_ALL")) > 0

    @property
    def wrist_all_ready(self) -> bool:
        return len(self.bundles_for_group("WRIST_ALL")) > 0

    def __len__(self) -> int:
        return len(self._models)


def _load_single_bundle(bundle_dir: Path, bundle_name: str) -> Optional[LoadedModel]:
    """Load one bundle directory. Returns None if files missing."""
    model_path = bundle_dir / settings.model.model_filename
    scaler_path = bundle_dir / settings.model.scaler_filename

    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    if not scaler_path.exists():
        logger.warning(f"Scaler file not found: {scaler_path}")
        return None

    group = _group_for(bundle_name)
    if group is None:
        logger.warning(f"Unknown bundle '{bundle_name}' — skipping")
        return None

    try:
        # Lazy import tensorflow to avoid slow startup if not needed at import time
        import  keras
        keras_model = keras.models.load_model(str(model_path), compile=False)
        scaler = joblib.load(scaler_path)

        # Training-time feature order and imputation medians. Without them the
        # engine cannot reproduce training preprocessing, so the bundle is skipped.
        meta_path, medians_path = bundle_dir / "meta.json", bundle_dir / "medians.npy"
        if not meta_path.exists() or not medians_path.exists():
            logger.warning(f"Bundle '{bundle_name}' lacks meta.json/medians.npy — skipping")
            return None
        feature_names = json.loads(meta_path.read_text())["feature_names"]
        medians = np.load(medians_path).astype(np.float32)
        if not (len(feature_names) == len(medians) == getattr(scaler, "n_features_in_", len(medians))):
            logger.error(f"Bundle '{bundle_name}': meta/medians/scaler widths disagree — skipping")
            return None

        # Infer input width from scaler (works for StandardScaler / MinMaxScaler)
        if hasattr(scaler, "n_features_in_"):
            n_features = scaler.n_features_in_
        elif hasattr(scaler, "scale_") and scaler.scale_ is not None:
            n_features = len(scaler.scale_)
        else:
            # Fall back: ask the model's first layer
            try:
                n_features = keras_model.input_shape[-1]
            except Exception:
                n_features = -1

        return LoadedModel(
            bundle=bundle_name,
            group=group,
            keras_model=keras_model,
            scaler=scaler,
            n_features=n_features,
            feature_names=feature_names,
            medians=medians,
        )
    except Exception as e:
        logger.error(f"Failed to load bundle '{bundle_name}': {e}")
        return None


def build_registry() -> ModelRegistry:
    """
    Scan active_models_dir for bundle subdirectories and load each.
    Returns a populated ModelRegistry.
    """
    registry = ModelRegistry()
    models_dir = settings.active_models_dir

    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return registry

    logger.info(f"Scanning models directory: {models_dir}")

    for bundle_dir in sorted(models_dir.iterdir()):
        if not bundle_dir.is_dir():
            continue
        bundle_name = bundle_dir.name
        lm = _load_single_bundle(bundle_dir, bundle_name)
        if lm is not None:
            registry.register(lm)

    logger.info(
        f"Registry built | total={len(registry)} | "
        f"chest_ready={registry.chest_all_ready} | wrist_ready={registry.wrist_all_ready}"
    )
    return registry


# Module-level singleton — populated in lifespan startup
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialised. Call init_registry() first.")
    return _registry


def init_registry() -> ModelRegistry:
    global _registry
    _registry = build_registry()
    return _registry
