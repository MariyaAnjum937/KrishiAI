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

Usage (API server):
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  python main.py

Usage (live webcam):
  python main.py --webcam
  python main.py --webcam --camera 1 --fps 10

Set MODEL_PATH env var to point to your trained .keras file:
  export MODEL_PATH=mobilenetv2_best.keras
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from models.schemas import HealthResponse
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
MODEL_PATH  = os.environ.get("MODEL_PATH", "mobilenetv2_best.keras")
API_VERSION = "1.0.0"

# ── App lifespan (startup / shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  PlantCare AI Backend  v%s  starting up …", API_VERSION)
    logger.info("=" * 60)
    app.state.predictor = load_predictor(MODEL_PATH)
    app.state.history: list = []
    logger.info("Predictor ready. Supported classes: %d", len(CLASS_NAMES))
    logger.info("API docs available at /docs  and  /redoc")
    logger.info("-" * 60)
    yield
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

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
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
@app.get("/health", response_model=HealthResponse, tags=["System"], summary="API health check")
async def health(request: Request):
    predictor    = request.app.state.predictor
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
        "name":    "PlantCare AI — Crop Disease Detection API",
        "version": API_VERSION,
        "status":  "running",
        "endpoints": {
            "health":               "GET  /health",
            "predict":              "POST /api/predict",
            "classes":              "GET  /api/classes",
            "fertilizers":          "GET  /api/fertilizers",
            "fertilizer_recommend": "POST /api/fertilizers/recommend",
            "history":              "GET  /api/history",
            "clear_history":        "DEL  /api/history",
            "docs":                 "GET  /docs",
            "redoc":                "GET  /redoc",
        },
        "model": "MobileNetV2 (transfer learning, 96.6% val accuracy)",
        "supported_crops": [
            "Apple", "Blueberry", "Cherry", "Corn/Maize", "Grape",
            "Orange", "Peach", "Bell Pepper", "Potato", "Raspberry",
            "Soybean", "Squash", "Strawberry", "Tomato",
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBCAM / LOCAL INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_webcam_predictor = None   # lazy-loaded once per webcam session


def _get_webcam_predictor():
    global _webcam_predictor
    if _webcam_predictor is None:
        _webcam_predictor = load_predictor(MODEL_PATH)
    return _webcam_predictor


def predict_frame(frame: np.ndarray) -> tuple[str, float]:
    """
    Run MobileNetV2 inference on a single OpenCV BGR frame.

    Parameters
    ----------
    frame : np.ndarray  — BGR image array from cv2.VideoCapture

    Returns
    -------
    (class_name, confidence)  e.g. ('Tomato___Late_blight', 0.943)
    """
    import io
    # OpenCV gives BGR → convert to RGB before passing to model
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    buf     = io.BytesIO()
    pil_img.save(buf, format="JPEG")

    result = _get_webcam_predictor().predict(buf.getvalue())
    return result["class_name"], result["confidence"]


def run_local_webcam(camera_index: int = 0, target_fps: int = 15):
    """
    Continuous real-time webcam inference using OpenCV.
    Displays live overlay with predicted class + confidence bar.
    Press 'q' to quit.

    Parameters
    ----------
    camera_index : int  — webcam device index (0 = default built-in camera)
    target_fps   : int  — max inference FPS (15 is smooth without CPU overload)
    """
    if not CV2_AVAILABLE:
        print("❌  opencv-python is not installed.")
        print("    Run:  pip install opencv-python")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌  Cannot open camera {camera_index}.")
        print("    Try --camera 1 if you have multiple cameras.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_interval  = 1.0 / target_fps
    last_inference  = 0.0
    last_class      = "Initialising..."
    last_confidence = 0.0
    last_is_healthy = False
    fps_display     = 0
    fps_timer       = time.time()
    frame_count     = 0

    print("🎥  Webcam started — hold a plant leaf up to the camera  |  Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Frame capture failed — check camera connection.")
            break

        now = time.time()

        # ── Inference at controlled FPS ───────────────────────────────────
        if now - last_inference >= frame_interval:
            try:
                class_name, conf = predict_frame(frame)
                parts            = class_name.split("___")
                plant_name       = parts[0].replace("_", " ")
                condition        = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
                last_class       = f"{plant_name} — {condition}"
                last_confidence  = conf
                last_is_healthy  = "healthy" in condition.lower()
            except Exception as exc:
                logger.warning("Webcam inference error: %s", exc)
            last_inference = now

        # ── FPS counter ───────────────────────────────────────────────────
        frame_count += 1
        if now - fps_timer >= 1.0:
            fps_display = frame_count
            frame_count = 0
            fps_timer   = now

        # ── HUD overlay ───────────────────────────────────────────────────
        h, w       = frame.shape[:2]
        status_col = (0, 200, 80) if last_is_healthy else (50, 80, 220)

        # Coloured border
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), status_col, 3)

        # Semi-transparent header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Text
        badge = "HEALTHY ✔" if last_is_healthy else "DISEASE DETECTED ⚠"
        cv2.putText(frame, badge,
                    (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_col, 2)
        cv2.putText(frame, last_class,
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {last_confidence * 100:.1f}%",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps_display}",
                    (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Press 'q' to quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        # Confidence bar (bottom strip)
        bar_w = int(w * last_confidence)
        cv2.rectangle(frame, (0, h - 6), (w,     h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), status_col,   -1)

        cv2.imshow("PlantCare AI - Real-Time Disease Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n👋  Webcam closed by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import uvicorn

    if "--webcam" in sys.argv:
        # python main.py --webcam
        # python main.py --webcam --camera 1 --fps 10
        cam_idx    = int(sys.argv[sys.argv.index("--camera") + 1]) if "--camera" in sys.argv else 0
        target_fps = int(sys.argv[sys.argv.index("--fps")    + 1]) if "--fps"    in sys.argv else 15
        run_local_webcam(camera_index=cam_idx, target_fps=target_fps)
    else:
        # python main.py  →  starts the FastAPI server
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
