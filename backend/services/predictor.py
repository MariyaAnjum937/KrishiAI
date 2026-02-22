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
import os
import random
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


# ── Mock Predictor (development / demo) ──────────────────────────────────────

class MockPredictor:
    """
    Returns deterministic mock predictions when no trained model is available.
    Cycles through a set of disease examples so the API is fully testable.
    """

    _DEMO_CLASSES = [
        "Tomato___Late_blight",
        "Corn_(maize)___Common_rust_",
        "Apple___Apple_scab",
        "Potato___Early_blight",
        "Grape___Black_rot",
        "Tomato___healthy",
        "Tomato___Bacterial_spot",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Pepper,_bell___Bacterial_spot",
    ]
    _counter = 0

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        # Use image byte checksum as seed so same image → same result
        seed = sum(image_bytes[:64]) % len(self._DEMO_CLASSES)
        primary = self._DEMO_CLASSES[seed]
        primary_conf = round(random.uniform(0.82, 0.97), 4)

        remaining = [c for c in CLASS_NAMES if c != primary]
        random.seed(seed)
        random.shuffle(remaining)
        others = remaining[:4]
        other_confs = sorted(
            [round(random.uniform(0.001, (1 - primary_conf) / 2), 4) for _ in others],
            reverse=True,
        )

        top5 = [{"class": primary, "confidence": primary_conf}] + [
            {"class": c, "confidence": conf} for c, conf in zip(others, other_confs)
        ]

        return {
            "class_name": primary,
            "confidence": primary_conf,
            "top5": top5,
        }


# ── Factory ────────────────────────────────────────────────────────────────────

def load_predictor(model_path: str) -> KerasPredictor | MockPredictor:
    """
    Try to load the real Keras model. Fall back to MockPredictor if:
      - The model file does not exist, or
      - TensorFlow is not installed.
    """
    if not Path(model_path).exists():
        logger.warning(
            "Model file '%s' not found — starting in MOCK / DEMO mode. "
            "Place your trained mobilenetv2_best.keras at that path to enable real inference.",
            model_path,
        )
        return MockPredictor()

    try:
        return KerasPredictor(model_path)
    except Exception as exc:
        logger.error("Failed to load Keras model: %s — falling back to mock predictor.", exc)
        return MockPredictor()
