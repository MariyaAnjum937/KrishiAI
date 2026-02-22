"""
Fertilizer Recommendation Service
===================================
Rule-based NPK analysis engine. Compares user-supplied soil nutrient
levels against crop-optimal thresholds and recommends corrective fertilizer
blends from the government-approved catalogue.

Source: Technical Framework doc — Fertilizer Logic and Nutrient Mapping section.
"""

from __future__ import annotations
from typing import Any

# ── Optimal NPK thresholds per crop (kg/ha or ppm equivalents) ────────────────
# Low / optimal / high bands based on agronomic literature
CROP_NPK_THRESHOLDS: dict[str, dict[str, dict[str, float]]] = {
    "tomato":   {"N": {"low": 80,  "high": 200}, "P": {"low": 40, "high": 80},  "K": {"low": 120, "high": 250}},
    "potato":   {"N": {"low": 100, "high": 220}, "P": {"low": 50, "high": 100}, "K": {"low": 150, "high": 280}},
    "corn":     {"N": {"low": 100, "high": 200}, "P": {"low": 40, "high": 90},  "K": {"low": 100, "high": 200}},
    "maize":    {"N": {"low": 100, "high": 200}, "P": {"low": 40, "high": 90},  "K": {"low": 100, "high": 200}},
    "apple":    {"N": {"low": 40,  "high": 120}, "P": {"low": 20, "high": 60},  "K": {"low": 80,  "high": 180}},
    "grape":    {"N": {"low": 40,  "high": 100}, "P": {"low": 20, "high": 50},  "K": {"low": 100, "high": 200}},
    "pepper":   {"N": {"low": 80,  "high": 180}, "P": {"low": 40, "high": 80},  "K": {"low": 120, "high": 220}},
    "soybean":  {"N": {"low": 20,  "high": 60},  "P": {"low": 40, "high": 80},  "K": {"low": 80,  "high": 160}},
    "strawberry": {"N": {"low": 40, "high": 100}, "P": {"low": 30, "high": 60}, "K": {"low": 80,  "high": 160}},
    "paddy":    {"N": {"low": 80,  "high": 160}, "P": {"low": 30, "high": 60},  "K": {"low": 60,  "high": 120}},
    "rice":     {"N": {"low": 80,  "high": 160}, "P": {"low": 30, "high": 60},  "K": {"low": 60,  "high": 120}},
    "wheat":    {"N": {"low": 80,  "high": 160}, "P": {"low": 40, "high": 80},  "K": {"low": 60,  "high": 120}},
    "default":  {"N": {"low": 60,  "high": 150}, "P": {"low": 30, "high": 70},  "K": {"low": 80,  "high": 160}},
}

# ── Fertilizer catalogue (from research doc) ──────────────────────────────────
FERTILIZER_DB = {
    "urea":  {"name": "Urea (46-0-0)",       "npk": "46-0-0", "nutrient": "N", "price_inr_per_mt": 5377,  "scheme": "Urea Subsidy Scheme"},
    "dap":   {"name": "DAP (18-46-0)",        "npk": "18-46-0","nutrient": "P", "price_inr_per_mt": 27000, "scheme": "Nutrient Based Subsidy"},
    "mop":   {"name": "MOP (0-0-60)",         "npk": "0-0-60", "nutrient": "K", "price_inr_per_mt": 31319, "scheme": "Nutrient Based Subsidy"},
    "npk_compound": {"name": "NPK 10-26-26",  "npk": "10-26-26","nutrient": "P+K","price_inr_per_mt": 29941,"scheme": "Nutrient Based Subsidy"},
    "ssp":   {"name": "SSP Granular (16% P)", "npk": "0-16-0", "nutrient": "P", "price_inr_per_mt": 10828, "scheme": "Nutrient Based Subsidy"},
}

# ── Application schedule templates ────────────────────────────────────────────
SCHEDULES = {
    "N_def": (
        "Apply 50 % of recommended nitrogen (Urea) as basal dose at planting. "
        "Split-apply remaining 50 % in 2 equal top-dressings at vegetative and "
        "flowering stages for efficient uptake."
    ),
    "P_def": (
        "Apply full phosphorus (DAP or SSP) as basal dose at or just before planting; "
        "phosphorus moves slowly in soil so early incorporation is essential."
    ),
    "K_def": (
        "Apply 50 % potassium (MOP) at planting; top-dress remaining 50 % at fruit "
        "development stage to improve fruit quality and disease resistance."
    ),
    "balanced": (
        "Soil levels appear balanced. Apply maintenance dose of NPK 10-26-26 "
        "at 200 kg/ha at planting as a preventive measure. Re-test soil after harvest."
    ),
    "excess_N": (
        "Nitrogen is excessive — reduce or skip nitrogen application this season. "
        "Excess N promotes lush tissue susceptible to fungal diseases. "
        "Focus on P and K applications only."
    ),
}


def recommend_fertilizer(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    crop: str,
    temperature: float | None = None,
    humidity: float | None = None,
    rainfall: float | None = None,
) -> dict[str, Any]:
    """
    Analyse soil NPK levels against crop-specific thresholds.
    Returns deficiencies, excesses, recommended fertilizers, and schedule.
    """
    crop_key = crop.lower().strip()
    thresholds = CROP_NPK_THRESHOLDS.get(crop_key, CROP_NPK_THRESHOLDS["default"])

    deficiencies: list[str] = []
    excesses: list[str] = []
    recommended: list[dict[str, str]] = []

    # ── Evaluate each nutrient ─────────────────────────────────────────────────
    for nutrient, value in [("N", nitrogen), ("P", phosphorus), ("K", potassium)]:
        low = thresholds[nutrient]["low"]
        high = thresholds[nutrient]["high"]

        if value < low:
            deficiencies.append(nutrient)
        elif value > high:
            excesses.append(nutrient)

    # ── Build recommendation list ──────────────────────────────────────────────
    if "N" in deficiencies:
        f = FERTILIZER_DB["urea"]
        recommended.append({
            "fertilizer": f["name"],
            "reason": "Soil nitrogen is below optimal range",
            "rate": "150–200 kg/ha Urea (adjust to soil test deficit)",
            "price_approx": f"INR {f['price_inr_per_mt']:,}/MT",
            "scheme": f["scheme"],
        })

    if "P" in deficiencies:
        # DAP is preferred; SSP is budget alternative
        f_dap = FERTILIZER_DB["dap"]
        f_ssp = FERTILIZER_DB["ssp"]
        recommended.append({
            "fertilizer": f"{f_dap['name']} (or {f_ssp['name']} as budget option)",
            "reason": "Soil phosphorus is below optimal range",
            "rate": "100 kg/ha DAP OR 250 kg/ha SSP as equivalent P dose",
            "price_approx": f"DAP: INR {f_dap['price_inr_per_mt']:,}/MT | SSP: INR {f_ssp['price_inr_per_mt']:,}/MT",
            "scheme": f_dap["scheme"],
        })

    if "K" in deficiencies:
        f = FERTILIZER_DB["mop"]
        recommended.append({
            "fertilizer": f["name"],
            "reason": "Soil potassium is below optimal range",
            "rate": "80–120 kg/ha MOP",
            "price_approx": f"INR {f['price_inr_per_mt']:,}/MT",
            "scheme": f["scheme"],
        })

    if not deficiencies and not excesses:
        f = FERTILIZER_DB["npk_compound"]
        recommended.append({
            "fertilizer": f["name"],
            "reason": "Maintenance dose — all nutrients in optimal range",
            "rate": "200 kg/ha",
            "price_approx": f"INR {f['price_inr_per_mt']:,}/MT",
            "scheme": f["scheme"],
        })

    # ── Pick schedule ──────────────────────────────────────────────────────────
    if "N" in excesses:
        schedule = SCHEDULES["excess_N"]
    elif deficiencies:
        primary_def = deficiencies[0]
        schedule = SCHEDULES.get(f"{primary_def}_def", SCHEDULES["balanced"])
    else:
        schedule = SCHEDULES["balanced"]

    # ── Weather-adjusted notes ────────────────────────────────────────────────
    notes_parts: list[str] = []
    if temperature is not None and temperature > 38:
        notes_parts.append(
            "High temperature detected (>38 °C): avoid urea application during peak heat "
            "to reduce volatilisation loss. Apply in early morning or evening."
        )
    if rainfall is not None and rainfall < 500:
        notes_parts.append(
            "Low annual rainfall zone: consider fertigation (drip fertilizer delivery) "
            "to improve nutrient use efficiency."
        )
    if humidity is not None and humidity > 85:
        notes_parts.append(
            "High humidity environment: reduce nitrogen application to limit lush tissue "
            "growth that is susceptible to fungal diseases."
        )
    if not notes_parts:
        notes_parts.append(
            "Always conduct a soil test before each season for precise recommendations. "
            "Soil Health Cards (Govt. of India) provide subsidised testing."
        )

    return {
        "crop": crop.title(),
        "deficiencies": deficiencies,
        "excesses": excesses,
        "recommended_fertilizers": recommended,
        "application_schedule": schedule,
        "notes": " | ".join(notes_parts),
    }
