"""
app/api/v1/routes.py
All five REST endpoints wired to the inference pipeline.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.core.settings import settings
from app.features.extractor import extract_all_bundles
from app.inference.engine import run_inference
from app.inference.validator import validate_request
from app.models.loader import get_registry
from app.schemas.predict import (
    FeatureResponse,
    HealthResponse,
    LoadedModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    ValidationResponse,
)

router = APIRouter(prefix="/api/v1")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Liveness + model readiness check."""
    try:
        registry = get_registry()
        return HealthResponse(
            status="ok",
            models_loaded=len(registry),
            chest_all_ready=registry.chest_all_ready,
            wrist_all_ready=registry.wrist_all_ready,
            version=settings.model.active_tag,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            models_loaded=0,
            chest_all_ready=False,
            wrist_all_ready=False,
            version=settings.model.active_tag,
        )


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelsResponse, tags=["System"])
def list_models() -> ModelsResponse:
    """List all loaded models and their metadata."""
    registry = get_registry()
    loaded = [
        LoadedModelInfo(
            bundle=b,
            group=registry.get(b).group,
            input_features=registry.get(b).n_features,
        )
        for b in registry.all_bundles()
    ]
    return ModelsResponse(
        active_tag=settings.model.active_tag,
        loaded_models=loaded,
        chest_all_ready=registry.chest_all_ready,
        wrist_all_ready=registry.wrist_all_ready,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/validate
# ---------------------------------------------------------------------------

@router.post("/validate", response_model=ValidationResponse, tags=["Inference"])
def validate(req: PredictRequest) -> ValidationResponse:
    """Validate incoming JSON without running feature extraction or inference."""
    result, _ = validate_request(req)
    return result


# ---------------------------------------------------------------------------
# POST /api/v1/features
# ---------------------------------------------------------------------------

@router.post("/features", response_model=FeatureResponse, tags=["Inference"])
def extract_features(req: PredictRequest) -> FeatureResponse:
    """Validate + extract features. Useful for debugging feature values."""
    val_result, ok = validate_request(req)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "validation": val_result.model_dump(),
                "message": "Validation failed — fix errors before extracting features.",
            },
        )

    try:
        bundles = extract_all_bundles(req)
    except Exception as e:
        logger.exception("Feature extraction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}",
        )

    return FeatureResponse(
        subject_id=req.subject_id,
        timestamp=req.timestamp,
        bundles_extracted=list(bundles.keys()),
        features=bundles,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/predict
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Full end-to-end inference: validate → extract → select models → ensemble."""
    # Layer 1: validate
    val_result, ok = validate_request(req)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "validation": val_result.model_dump(),
                "message": "Request validation failed.",
            },
        )

    # Layer 2: feature extraction
    try:
        bundles = extract_all_bundles(req)
    except Exception as e:
        logger.exception("Feature extraction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}",
        )

    # Layers 3-6: model selection + inference + ensemble
    registry = get_registry()
    try:
        response = run_inference(req, bundles, registry)
    except ValueError as e:
        # No eligible model group
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    return response
