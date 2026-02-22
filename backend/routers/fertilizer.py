"""
Fertilizer Router
=================
GET  /api/fertilizers          — Full fertilizer catalogue.
POST /api/fertilizers/recommend — NPK-based fertilizer recommendation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import (
    FertilizerItem,
    FertilizerResponse,
    NPKInput,
    NPKRecommendation,
    NPKResponse,
)
from services.treatment_db import get_fertilizer_catalogue
from services.fertilizer_service import recommend_fertilizer

router = APIRouter(prefix="/api", tags=["Fertilizer"])


# ── GET /api/fertilizers ──────────────────────────────────────────────────────

@router.get(
    "/fertilizers",
    response_model=FertilizerResponse,
    summary="Get full fertilizer catalogue",
    description=(
        "Returns the government-approved fertilizer catalogue with NPK content, "
        "indicative prices (INR/MT), and subsidy schemes. "
        "Source: TNAU Agritech Portal & Ministry of Chemicals & Fertilizers."
    ),
)
async def get_fertilizers():
    catalogue = get_fertilizer_catalogue()
    items = [FertilizerItem(**f) for f in catalogue]
    return FertilizerResponse(success=True, count=len(items), data=items)


# ── POST /api/fertilizers/recommend ──────────────────────────────────────────

@router.post(
    "/fertilizers/recommend",
    response_model=NPKResponse,
    summary="Get crop-specific fertilizer recommendation from NPK soil data",
    description=(
        "Provide soil nitrogen, phosphorus, and potassium levels along with the "
        "crop name. The service compares values against agronomic thresholds and "
        "recommends a corrective fertilizer blend and application schedule. "
        "Optional weather parameters (temperature, humidity, rainfall) enable "
        "climate-adjusted advice."
    ),
)
async def fertilizer_recommend(payload: NPKInput):
    try:
        result = recommend_fertilizer(
            nitrogen=payload.nitrogen,
            phosphorus=payload.phosphorus,
            potassium=payload.potassium,
            crop=payload.crop,
            temperature=payload.temperature,
            humidity=payload.humidity,
            rainfall=payload.rainfall,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {exc}")

    recommendation = NPKRecommendation(**result)
    return NPKResponse(
        success=True,
        data=recommendation,
        message=(
            "Fertilizer recommendation generated. Always validate with a soil health card."
        ),
    )
