"""
ML Inference Service
====================
Handles loading the trained MobileNetV2 Keras model and running predictions.
Falls back to a mock predictor if no model file is present (for development).

Architecture (from PlantCare AI doc):
  Input (224×224×3) → MobileNetV2 base → GAP → Dropout(0.35) → Dense(256)
                     → Dropout(0.25) → Dense(38, softmax)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── 38-Class Label List (must match training order) ───────────────────────────
CLASS_NAMES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

IMG_SIZE = (224, 224)


# ── Helper: preprocess image bytes ───────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a preprocessed numpy array ready for MobileNetV2.
    Steps: decode → RGB → resize to 224×224 → MobileNetV2 preprocess → add batch dim.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocessing: scale to [-1, 1]
    arr = (arr / 127.5) - 1.0

    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)


# ── Real Keras Model Predictor ────────────────────────────────────────────────

class KerasPredictor:
    """Loads the trained .keras model and runs inference."""

    def __init__(self, model_path: str):
        import tensorflow as tf  # deferred import — optional at server startup
        logger.info("Loading Keras model from %s …", model_path)
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully (%d parameters).",
                    self.model.count_params())

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        arr = preprocess_image(image_bytes)
        probs = self.model.predict(arr, verbose=0)[0]          # shape: (38,)
        top_idx = int(np.argmax(probs))
        top5_idx = np.argsort(probs)[::-1][:5]

        return {
            "class_name": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
            "top5": [
                {"class": CLASS_NAMES[i], "confidence": float(probs[i])}
                for i in top5_idx
            ],
        }


# ── Unavailable Predictor ─────────────────────────────────────────────────────

class ModelUnavailableError(RuntimeError):
    """Raised when a real prediction is requested without a loaded model."""


class UnavailablePredictor:
    def __init__(self, reason: str):
        self.reason = reason

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        raise ModelUnavailableError(self.reason)


# ── Factory ────────────────────────────────────────────────────────────────────

def load_predictor(model_path: str) -> KerasPredictor | UnavailablePredictor:
    """
    Load the real Keras model, or return an unavailable predictor with a
    clear reason instead of manufacturing disease predictions.
    """
    if not Path(model_path).exists():
        reason = (
            f"Trained disease model not found at '{model_path}'. "
            "Set MODEL_PATH to the trained .keras model file and restart the backend."
        )
        logger.warning(
            "%s",
            reason,
        )
        return UnavailablePredictor(reason)

    try:
        return KerasPredictor(model_path)
    except Exception as exc:
        reason = (
            "The trained disease model could not be loaded. Install the backend "
            "dependencies (including tensorflow-cpu) and restart the backend."
        )
        logger.exception("Failed to load Keras model from %s: %s", model_path, exc)
        return UnavailablePredictor(reason)
