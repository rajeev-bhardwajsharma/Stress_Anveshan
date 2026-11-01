from __future__ import annotations #always first line
#can remove it due to environment issue patch work
import keras

# Force Keras to safely drop the quantization_config attribute during load
_original_layer_init = keras.layers.Layer.__init__
def _patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_layer_init(self, *args, **kwargs)
keras.layers.Layer.__init__ = _patched_layer_init

#patch work
# --- Your original main.py code (FastAPI, etc.) continues below ---
"""
main.py — WESAD Stress Detection API
Run with:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""


from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router
from app.core.logging import get_logger, setup_logging
from app.core.settings import settings
from app.models.loader import init_registry

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all Keras models and scalers into memory."""
    logger.info("=" * 60)
    logger.info(f"Starting WESAD Inference API — tag: {settings.model.active_tag}")
    logger.info(f"Models dir: {settings.active_models_dir}")
    logger.info("=" * 60)

    registry = init_registry()

    if not registry.chest_all_ready and not registry.wrist_all_ready:
        logger.warning(
            "No models loaded! Check that Models/<tag>/ contains "
            f"'{settings.model.model_filename}' and '{settings.model.scaler_filename}'."
        )
    else:
        logger.info(
            f"Ready — {len(registry)} model(s) loaded | "
            f"chest={registry.chest_all_ready} | wrist={registry.wrist_all_ready}"
        )

    yield

    logger.info("Shutting down WESAD Inference API.")


app = FastAPI(
    title="WESAD Stress Detection API",
    description=(
        "Real-time physiological stress detection from WESAD sensor data. "
        "Accepts raw signals, extracts hand-engineered features, "
        "runs ensemble inference across trained Keras models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["System"])
def root():
    return {
        "service": "WESAD Stress Detection API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
