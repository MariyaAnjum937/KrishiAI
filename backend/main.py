"""
PlantCare AI — FastAPI Backend
================================
Main application entry point.

Architecture (based on research documents):
  - MobileNetV2 transfer learning model (96.6 % validation accuracy)
  - 38-class PlantVillage / New Plant Diseases Dataset
  - Treatment dictionary: chemical + organic pesticides, prevention, ETL
  - NPK-based fertilizer recommendation engine
  - Offline-first compatible (stateless REST API)

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Set MODEL_PATH env var to point to your trained .keras file:
  export MODEL_PATH=mobilenetv2_best.keras
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from files.models.schemas import HealthResponse
from routers import predict, fertilizer, history
from services.predictor import load_predictor, CLASS_NAMES

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "mobilenetv2_best.keras")
API_VERSION = "1.0.0"

# ── App lifespan (startup / shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("=" * 60)
    logger.info("  PlantCare AI Backend  v%s  starting up …", API_VERSION)
    logger.info("=" * 60)

    app.state.predictor = load_predictor(MODEL_PATH)
    app.state.history: list = []          # in-memory scan history

    logger.info("Predictor ready. Supported classes: %d", len(CLASS_NAMES))
    logger.info("API docs available at /docs  and  /redoc")
    logger.info("-" * 60)

    yield  # ← app runs

    # SHUTDOWN
    logger.info("PlantCare AI Backend shutting down. Goodbye!")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PlantCare AI — Crop Disease Detection API",
    description=(
        "AI-powered API for detecting plant diseases from leaf images using "
        "MobileNetV2 transfer learning. Returns disease classification, "
        "chemical & organic pesticide recommendations, prevention tips, and "
        "fertilizer guidance for all 38 PlantVillage disease categories.\n\n"
        "**Model**: MobileNetV2 fine-tuned on 87,000+ annotated images  \n"
        "**Validation accuracy**: 96.6 %  \n"
        "**Dataset**: New Plant Diseases Dataset (Kaggle)  \n"
        "**Supported crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, "
        "Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato"
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS (allow all origins for development; restrict in production) ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (uploaded images served for display) ─────────────────────────
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
(static_dir / "images").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(fertilizer.router)
app.include_router(history.router)


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An internal server error occurred. Please check server logs.",
            "detail": str(exc),
        },
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API health check",
)
async def health(request: Request):
    predictor = request.app.state.predictor
    model_loaded = predictor.__class__.__name__ == "KerasPredictor"
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        supported_classes=len(CLASS_NAMES),
        version=API_VERSION,
    )


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"], summary="API root / welcome")
async def root():
    return {
        "name": "PlantCare AI — Crop Disease Detection API",
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "health":              "GET  /health",
            "predict":             "POST /api/predict          (upload leaf image)",
            "classes":             "GET  /api/classes          (list all 38 classes)",
            "fertilizers":         "GET  /api/fertilizers      (fertilizer catalogue)",
            "fertilizer_recommend":"POST /api/fertilizers/recommend (NPK analysis)",
            "history":             "GET  /api/history          (scan history)",
            "clear_history":       "DEL  /api/history",
            "docs":                "GET  /docs                 (Swagger UI)",
            "redoc":               "GET  /redoc                (ReDoc UI)",
        },
        "model": "MobileNetV2 (transfer learning, 96.6% val accuracy)",
        "supported_crops": [
            "Apple", "Blueberry", "Cherry", "Corn/Maize", "Grape",
            "Orange", "Peach", "Bell Pepper", "Potato", "Raspberry",
            "Soybean", "Squash", "Strawberry", "Tomato",
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="8000", reload = True)