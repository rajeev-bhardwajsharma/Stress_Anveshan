"""
app/schemas/predict.py
All request and response Pydantic models for the inference API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-models for signal payloads
# ---------------------------------------------------------------------------

class ChestSignals(BaseModel):
    ACC:  Optional[List[List[float]]] = Field(None, description="(N,3) accelerometer")
    ECG:  Optional[List[float]] = None
    EMG:  Optional[List[float]] = None
    EDA:  Optional[List[float]] = None
    Temp: Optional[List[float]] = None
    Resp: Optional[List[float]] = None


class WristSignals(BaseModel):
    ACC:  Optional[List[List[float]]] = Field(None, description="(N,3) accelerometer")
    BVP:  Optional[List[float]] = None
    EDA:  Optional[List[float]] = None
    TEMP: Optional[List[float]] = None


class SignalPayload(BaseModel):
    chest: Optional[ChestSignals] = None
    wrist: Optional[WristSignals] = None


class SensorAvailability(BaseModel):
    chest: List[str] = Field(default_factory=list)
    wrist: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    subject_id: str = Field(..., examples=["S2"])
    timestamp: datetime
    window_size_sec: int = Field(..., ge=60, le=60, description="Must be 60")
    available_sensors: SensorAvailability
    signals: SignalPayload

    @model_validator(mode="after")
    def check_window_size(self) -> PredictRequest:
        if self.window_size_sec != 60:
            raise ValueError("window_size_sec must be exactly 60")
        return self


# ---------------------------------------------------------------------------
# Validation response
# ---------------------------------------------------------------------------

class ValidationResponse(BaseModel):
    validation_status: Literal["passed", "failed"]
    missing_sensors: List[str] = Field(default_factory=list)
    invalid_sensors: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature extraction response
# ---------------------------------------------------------------------------

class FeatureResponse(BaseModel):
    subject_id: str
    timestamp: datetime
    bundles_extracted: List[str]
    features: Dict[str, Dict[str, float]]   # bundle_name -> {feature_name -> value}


# ---------------------------------------------------------------------------
# Prediction response
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    subject_id: str
    timestamp: datetime
    models_used: List[str]
    stress_probability: float = Field(..., ge=0.0, le=1.0)
    non_stress_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: Literal["STRESSED", "NOT_STRESSED"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Models list response
# ---------------------------------------------------------------------------

class LoadedModelInfo(BaseModel):
    bundle: str
    group: Literal["CHEST_ALL", "WRIST_ALL"]
    input_features: int


class ModelsResponse(BaseModel):
    active_tag: str
    loaded_models: List[LoadedModelInfo]
    chest_all_ready: bool
    wrist_all_ready: bool


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: int
    chest_all_ready: bool
    wrist_all_ready: bool
    version: str
