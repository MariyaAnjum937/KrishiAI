"""
Pydantic Schemas
================
Request and response models for the PlantCare AI FastAPI backend.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Pesticide ─────────────────────────────────────────────────────────────────

class PesticideInfo(BaseModel):
    name: str
    dosage: str
    frequency: str
    safety: str
    type: str  # "chemical" | "organic" | "N/A"


# ── Prediction Response ────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    class_name: str = Field(..., description="PlantVillage class label, e.g. 'Tomato___Late_blight'")
    plant: str
    condition: str
    is_healthy: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")
    confidence_pct: str = Field(..., description="Confidence as a percentage string")
    severity_risk: str
    description: str
    pesticides: list[PesticideInfo]
    organic: list[str]
    prevention: list[str]
    etl: str
    fertilizer_note: str
    top5: list[dict[str, Any]] = Field(default_factory=list, description="Top-5 predictions with labels and confidence")


class PredictionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[PredictionResult] = None


# ── Fertilizer ────────────────────────────────────────────────────────────────

class FertilizerItem(BaseModel):
    name: str
    npk: str
    price_inr_per_mt: int
    scheme: str
    best_for: str


class FertilizerResponse(BaseModel):
    success: bool
    count: int
    data: list[FertilizerItem]


# ── Fertilizer Recommendation (NPK-based) ─────────────────────────────────────

class NPKInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=100, description="Soil nitrogen level (kg/ha or ppm)")
    phosphorus: float = Field(..., ge=0, le=100, description="Soil phosphorus level")
    potassium: float = Field(..., ge=0, le=100, description="Soil potassium level")
    crop: str = Field(..., description="Crop name, e.g. 'Tomato'")
    temperature: Optional[float] = Field(None, description="Average temperature °C")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity %")
    rainfall: Optional[float] = Field(None, ge=0, description="Annual rainfall mm")

class NPKRecommendation(BaseModel):
    crop: str
    deficiencies: list[str]
    excesses: list[str]
    recommended_fertilizers: list[dict[str, str]]
    application_schedule: str
    notes: str

class NPKResponse(BaseModel):
    success: bool
    data: Optional[NPKRecommendation] = None
    message: str


# ── Disease History ────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    id: str
    timestamp: str
    class_name: str
    plant: str
    condition: str
    is_healthy: bool
    confidence: float
    severity_risk: str


class HistoryResponse(BaseModel):
    success: bool
    count: int
    data: list[HistoryEntry]


# ── Supported Classes ─────────────────────────────────────────────────────────

class ClassesResponse(BaseModel):
    success: bool
    count: int
    classes: list[str]


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    supported_classes: int
    version: str

# ── Chatbot ───────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str                        # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    model: Optional[str] = Field(
        default="llama3-8b-8192",
        description="Groq model ID. Options: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768"
    )


class ChatResponse(BaseModel):
    success: bool
    message: str
    model: str
    tokens_used: Optional[int] = None


class ChatHistoryResponse(BaseModel):
    success: bool
    count: int
    data: list[ChatMessage]
