"""
app/core/settings.py
Loads config.yaml once at import time. All other modules import `settings`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


_CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass
class InferenceConfig:
    decision_threshold: float
    window_size_sec: int


@dataclass
class ModelConfig:
    active_tag: str
    model_filename: str
    scaler_filename: str


@dataclass
class Settings:
    models_dir: Path
    features_dir: Path
    model: ModelConfig
    inference: InferenceConfig
    chest_all_sensors: List[str]
    wrist_all_sensors: List[str]
    sampling_rates: Dict[str, int]
    chest_bundles: List[str]
    wrist_bundles: List[str]
    log_level: str

    @property
    def active_models_dir(self) -> Path:
        return self.models_dir if not self.model.active_tag else self.models_dir / self.model.active_tag


def load_settings() -> Settings:
    cfg = _load_yaml()

    return Settings(
        models_dir=Path(cfg["paths"]["models_dir"]),
        features_dir=Path(cfg["paths"]["features_dir"]),
        model=ModelConfig(
            active_tag=cfg["model"]["active_tag"],
            model_filename=cfg["model"]["model_filename"],
            scaler_filename=cfg["model"]["scaler_filename"],
        ),
        inference=InferenceConfig(
            decision_threshold=cfg["inference"]["decision_threshold"],
            window_size_sec=cfg["inference"]["window_size_sec"],
        ),
        chest_all_sensors=cfg["sensors"]["chest_all"],
        wrist_all_sensors=cfg["sensors"]["wrist_all"],
        sampling_rates=cfg["sampling_rates"],
        chest_bundles=cfg["bundles"]["chest"],
        wrist_bundles=cfg["bundles"]["wrist"],
        log_level=cfg["logging"]["level"],
    )


# Singleton — import this everywhere
settings: Settings = load_settings()
