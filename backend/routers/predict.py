"""
Prediction Router
=================
POST /api/predict  — Upload a leaf image; get disease classification + treatment plan.
GET  /api/classes  — List all 38 supported disease classes.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from models.schemas import (
    PredictionResponse,
    PredictionResult,
    PesticideInfo,
    ClassesResponse,
)
from services.treatment_db import get_treatment, get_all_classes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Disease Detection"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB


def get_predictor(request: Request):
    """FastAPI dependency — retrieves the shared predictor from app state."""
    return request.app.state.predictor


def get_history(request: Request) -> list:
    """FastAPI dependency — retrieves the shared in-memory history list."""
    return request.app.state.history


# ── POST /api/predict ─────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Detect plant disease from a leaf image",
    description=(
        "Upload a JPEG/PNG image of a plant leaf. The MobileNetV2 model "
        "classifies the image into one of 38 disease/healthy categories and "
        "returns pesticide and organic treatment recommendations."
    ),
)
async def predict(
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG/WEBP, max 16 MB)"),
    predictor=Depends(get_predictor),
    history: list = Depends(get_history),
):
    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Allowed types: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    # ── Read & size-check ─────────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) / 1_048_576:.1f} MB). Max allowed: 16 MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Run model inference ───────────────────────────────────────────────────
    try:
        raw = predictor.predict(image_bytes)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    class_name: str = raw["class_name"]
    confidence: float = raw["confidence"]
    top5: list[dict] = raw.get("top5", [])

    # ── Look up treatment info ────────────────────────────────────────────────
    treatment = get_treatment(class_name)
    if treatment is None:
        raise HTTPException(
            status_code=500,
            detail=f"No treatment data found for class '{class_name}'.",
        )

    # ── Build response ────────────────────────────────────────────────────────
    pesticides = [PesticideInfo(**p) for p in treatment["pesticides"]]
    result = PredictionResult(
        class_name=class_name,
        plant=treatment["plant"],
        condition=treatment["condition"],
        is_healthy=treatment["is_healthy"],
        confidence=round(confidence, 4),
        confidence_pct=f"{confidence * 100:.1f}%",
        severity_risk=treatment["severity_risk"],
        description=treatment["description"],
        pesticides=pesticides,
        organic=treatment["organic"],
        prevention=treatment["prevention"],
        etl=treatment["etl"],
        fertilizer_note=treatment["fertilizer_note"],
        top5=top5,
    )

    # ── Persist to in-memory history ──────────────────────────────────────────
    history.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_name": class_name,
        "plant": treatment["plant"],
        "condition": treatment["condition"],
        "is_healthy": treatment["is_healthy"],
        "confidence": round(confidence, 4),
        "severity_risk": treatment["severity_risk"],
    })
    # Keep only the last 500 entries to prevent unbounded growth
    if len(history) > 500:
        del history[0]

    logger.info(
        "Prediction: %s | Confidence: %.2f%% | File: %s",
        class_name,
        confidence * 100,
        file.filename,
    )

    return PredictionResponse(
        success=True,
        message=f"Disease detection complete — {treatment['condition']} identified.",
        data=result,
    )


# ── GET /api/classes ──────────────────────────────────────────────────────────

@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="List all 38 supported plant disease classes",
)
async def get_classes():
    classes = get_all_classes()
    return ClassesResponse(success=True, count=len(classes), classes=classes)
