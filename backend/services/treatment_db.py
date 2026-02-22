"""
Treatment Database
==================
Maps each of the 38 PlantVillage / New Plant Diseases Dataset class labels
to structured treatment data: chemical pesticides, organic alternatives,
prevention tips, ETL thresholds, and fertilizer guidance.

Source: PlantCare AI research doc + Technical Framework doc.
"""

from typing import Any

TREATMENT_DB: dict[str, dict[str, Any]] = {

    # ── APPLE ──────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "plant": "Apple", "condition": "Apple Scab",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Fungal disease caused by Venturia inaequalis producing dark, "
            "velvety, scab-like lesions on leaves and fruit surfaces."
        ),
        "pesticides": [
            {
                "name": "Captan 50 WP",
                "dosage": "2.5 g / L water",
                "frequency": "Every 10–14 days",
                "safety": "Wear gloves and mask. PHI: 0 days.",
                "type": "chemical",
            },
            {
                "name": "Myclobutanil (Rally 40 WP)",
                "dosage": "0.4 g / L water",
                "frequency": "Every 14 days",
                "safety": "Avoid spray drift. PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Sulfur 80 WG (Kumulus DF) — 3 g / L water",
            "Copper oxychloride 50 WP — 3 g / L water",
            "Neem oil 5 mL / L (preventive only)",
        ],
        "prevention": [
            "Prune canopy for airflow",
            "Remove and destroy fallen leaves in autumn",
            "Apply dormant copper sprays before budbreak",
            "Plant scab-resistant cultivars (e.g., Redfree, Liberty)",
        ],
        "etl": "5 % leaf area affected — spray threshold reached",
        "fertilizer_note": "Balanced NPK; avoid excess N which promotes tender tissue.",
    },

    "Apple___Black_rot": {
        "plant": "Apple", "condition": "Black Rot",
        "severity_risk": "High", "is_healthy": False,
        "description": (
            "Botryosphaeria obtusa causes frogeye leaf spots, fruit mummy rot, "
            "and cankers that can girdle branches."
        ),
        "pesticides": [
            {
                "name": "Thiophanate-methyl 70 WP",
                "dosage": "1.5 g / L water",
                "frequency": "Every 10 days from petal fall",
                "safety": "PHI: 3 days. Avoid eye contact.",
                "type": "chemical",
            },
            {
                "name": "Captan 50 WP",
                "dosage": "2.5 g / L water",
                "frequency": "Every 10–14 days",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper hydroxide 77 WP — 3 g / L (Kocide)",
            "Bordeaux mixture 1 % — preventive application",
        ],
        "prevention": [
            "Remove mummified fruit promptly",
            "Prune dead or cankered wood; disinfect tools",
            "Maintain tree vigor with balanced fertilization",
            "Avoid mechanical injuries to bark",
        ],
        "etl": "10 % fruit infection — begin protective sprays immediately",
        "fertilizer_note": "Correct any potassium deficiency to harden fruit skin.",
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple", "condition": "Cedar Apple Rust",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Gymnosporangium juniperi-virginianae produces bright orange-yellow "
            "leaf spots; requires both apple and juniper/cedar hosts."
        ),
        "pesticides": [
            {
                "name": "Myclobutanil (Rally 40 WP)",
                "dosage": "0.4 g / L water",
                "frequency": "Every 7–10 days during infection periods (petal fall to 2nd cover)",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Propiconazole 25 EC",
                "dosage": "1 mL / L water",
                "frequency": "Every 14 days",
                "safety": "PHI: 14 days. Full PPE required.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Sulfur dust 3 kg / acre at bud break",
            "Neem oil 5 mL / L (preventive, before spore release)",
        ],
        "prevention": [
            "Remove nearby juniper / cedar hosts where feasible",
            "Apply fungicides at pink-bud stage before infection periods",
            "Plant rust-resistant apple varieties",
        ],
        "etl": "Preventive — apply at first infection period regardless of visible symptoms",
        "fertilizer_note": "Avoid excessive nitrogen; balanced calcium improves resistance.",
    },

    "Apple___healthy": {
        "plant": "Apple", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The apple leaf appears healthy.",
        "pesticides": [],
        "organic": ["Continue monthly neem oil preventive spray (3 mL / L)"],
        "prevention": [
            "Maintain balanced N-P-K fertilization",
            "Regular irrigation; avoid waterlogging",
            "Weekly scouting for early symptom detection",
        ],
        "etl": "N/A",
        "fertilizer_note": "Soil test annually; apply lime if pH < 6.0.",
    },

    # ── BLUEBERRY ──────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "plant": "Blueberry", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The blueberry leaf appears healthy.",
        "pesticides": [],
        "organic": ["Apply compost mulch 5 cm deep", "Maintain soil pH 4.5–5.5 with sulfur"],
        "prevention": [
            "Avoid overhead irrigation (promotes fungal diseases)",
            "Ensure drainage; blueberries dislike wet feet",
            "Annual renewal pruning to remove old canes",
        ],
        "etl": "N/A",
        "fertilizer_note": "Use ammonium-based nitrogen (ammonium sulfate); avoid nitrates.",
    },

    # ── CHERRY ─────────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry (including sour)",
        "condition": "Powdery Mildew",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Podosphaeria clandestina causes white powdery fungal colonies on "
            "young leaves and shoots, causing leaf curl and defoliation."
        ),
        "pesticides": [
            {
                "name": "Trifloxystrobin (Flint 50 WG)",
                "dosage": "0.2 g / L water",
                "frequency": "Every 14 days; rotate fungicide groups",
                "safety": "PHI: 3 days.",
                "type": "chemical",
            },
            {
                "name": "Myclobutanil 40 WP",
                "dosage": "0.4 g / L water",
                "frequency": "Every 10–14 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Potassium bicarbonate 5 g / L",
            "Sulfur 80 WG 3 g / L (do not use above 30 °C)",
            "Neem oil 5 mL / L + 2 mL / L dish soap",
        ],
        "prevention": [
            "Improve canopy aeration by pruning",
            "Avoid excessive nitrogen fertilization",
            "Water at plant base; avoid wetting foliage",
        ],
        "etl": "First white colonies observed — begin spray programme",
        "fertilizer_note": "High N promotes succulent growth susceptible to mildew; moderate application.",
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry (including sour)", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The cherry leaf appears healthy.",
        "pesticides": [],
        "organic": ["Preventive sulfur spray at green tip and petal fall"],
        "prevention": [
            "Annual dormant pruning for open canopy",
            "Monitor for aphid infestations (vector of viruses)",
        ],
        "etl": "N/A",
        "fertilizer_note": "Apply balanced fertilizer post-harvest; avoid late-season N.",
    },

    # ── CORN / MAIZE ───────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)", "condition": "Gray Leaf Spot (Cercospora)",
        "severity_risk": "High", "is_healthy": False,
        "description": (
            "Cercospora zeae-maydis creates rectangular, gray-tan lesions "
            "bounded by leaf veins, causing premature senescence in warm humid conditions."
        ),
        "pesticides": [
            {
                "name": "Azoxystrobin + Propiconazole (Quilt Xcel)",
                "dosage": "1.5 L / ha",
                "frequency": "Single application at VT/R1 growth stage",
                "safety": "PHI: 14 days. Wear PPE.",
                "type": "chemical",
            },
            {
                "name": "Pyraclostrobin (Headline EC)",
                "dosage": "0.75 L / ha",
                "frequency": "Preventive at VT stage",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Bacillus subtilis WP (Serenade) — 3 g / L, weekly",
            "Trichoderma viride WP — soil drench at planting",
        ],
        "prevention": [
            "Plant resistant hybrid varieties",
            "Crop rotation: avoid maize-on-maize",
            "Minimum-tillage to speed residue decomposition",
            "Scout from V8 onwards in high-risk conditions",
        ],
        "etl": "5 % leaf area of ear leaf or higher at VT stage",
        "fertilizer_note": "Adequate potassium reduces disease severity; maintain K at 120 kg/ha.",
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)", "condition": "Common Rust",
        "severity_risk": "Medium-High", "is_healthy": False,
        "description": (
            "Puccinia sorghi produces cinnamon-brown oval pustules on both leaf "
            "surfaces; rapidly spreads in cool (16–23 °C) humid conditions."
        ),
        "pesticides": [
            {
                "name": "Propiconazole 25 EC",
                "dosage": "1 mL / L water",
                "frequency": "At first pustule detection",
                "safety": "PHI: 14 days.",
                "type": "chemical",
            },
            {
                "name": "Trifloxystrobin + Propiconazole (Stratego YLD)",
                "dosage": "1.5 L / ha",
                "frequency": "Single application at silking",
                "safety": "PHI: 14 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Sulfur 80 WP 3 g / L (preventive, before disease onset)",
            "Neem-based biopesticide spray",
        ],
        "prevention": [
            "Plant early-maturing rust-tolerant hybrids",
            "Avoid late planting into high-pressure windows",
            "Monitor bi-weekly from V6",
        ],
        "etl": "Pustules present on upper leaves before tasseling",
        "fertilizer_note": "Silicon supplementation (1 kg SiO2/ha) reduces rust severity.",
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)", "condition": "Northern Leaf Blight",
        "severity_risk": "High", "is_healthy": False,
        "description": (
            "Setosphaeria turcica produces long (5–15 cm) cigar-shaped "
            "gray-green lesions; yields can drop 30–50 % in severe cases."
        ),
        "pesticides": [
            {
                "name": "Azoxystrobin 23 SC",
                "dosage": "1 L / ha",
                "frequency": "At first lesion appearance on lower leaves",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2.5 kg / ha",
                "frequency": "Every 10–14 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper oxychloride 50 WP — 3 g / L",
            "Trichoderma viride WP (soil + foliar)",
        ],
        "prevention": [
            "Use resistant varieties with Ht1 or polygenic resistance",
            "Crop rotation with non-host crops",
            "Incorporate and bury crop residues after harvest",
            "Avoid overhead irrigation",
        ],
        "etl": "Any lesion on ear leaf or above at V14 growth stage",
        "fertilizer_note": "Adequate nitrogen improves canopy density but monitor closely.",
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The maize leaf appears healthy.",
        "pesticides": [],
        "organic": ["Apply Trichoderma to soil furrow at planting"],
        "prevention": [
            "Balanced NPK fertilization based on soil test",
            "Adequate plant spacing (60–75 cm rows)",
            "Effective weed management especially in first 6 weeks",
        ],
        "etl": "N/A",
        "fertilizer_note": "Top-dress with urea (46-0-0) at V6; monitor for zinc deficiency.",
    },

    # ── GRAPE ──────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "plant": "Grape", "condition": "Black Rot",
        "severity_risk": "High", "is_healthy": False,
        "description": (
            "Guignardia bidwellii produces circular brown leaf spots and "
            "shrivelled black mummified berries; spreads rapidly in warm wet weather."
        ),
        "pesticides": [
            {
                "name": "Myclobutanil (Rally 40 WP)",
                "dosage": "0.4 g / L water",
                "frequency": "Every 7–14 days from budbreak; critical 2–6 weeks post-bloom",
                "safety": "PHI: 14 days.",
                "type": "chemical",
            },
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2 g / L water",
                "frequency": "Every 7–10 days",
                "safety": "PHI: 66 days. Do not use within 66 days of harvest.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper hydroxide (Kocide) 2 g / L",
            "Bordeaux mixture 0.5 %",
        ],
        "prevention": [
            "Remove all mummified berries before budbreak",
            "Prune to open canopy for air circulation",
            "Apply fungicide before rain events",
            "Train vines to minimise berry contact",
        ],
        "etl": "First confirmed infection period post-budbreak — protective spray required",
        "fertilizer_note": "Adequate potassium (K) hardens berry skin and reduces infection.",
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape", "condition": "Esca (Black Measles)",
        "severity_risk": "High", "is_healthy": False,
        "description": (
            "A complex wood disease caused by Phaeomoniella, Phaeoacremonium, "
            "and Fomitiporia fungi causing tiger-striped leaves and vine apoplexy (sudden collapse)."
        ),
        "pesticides": [
            {
                "name": "Fosetyl-Al (Aliette WG) — Systemic",
                "dosage": "2.5 g / L water",
                "frequency": "2 applications post-pruning",
                "safety": "PHI: 21 days. No cure — protective only.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Trichoderma atroviride wound protectant after pruning",
            "Lac Balsam wound sealant on pruning cuts",
        ],
        "prevention": [
            "Prune only in dry weather",
            "Disinfect pruning tools with 70 % ethanol between vines",
            "Remove and burn all infected wood immediately",
            "Delay pruning until late in dormancy",
        ],
        "etl": "Any vine showing apoplexy — remove immediately to prevent spread",
        "fertilizer_note": "Avoid excessive vigor; balanced nutrition reduces wood stress.",
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape", "condition": "Leaf Blight (Isariopsis Leaf Spot)",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Isariopsis clavispora creates brown angular spots with dark borders "
            "on upper leaf surface; premature defoliation weakens vines."
        ),
        "pesticides": [
            {
                "name": "Copper oxychloride 50 WP",
                "dosage": "3 g / L water",
                "frequency": "Every 10–14 days",
                "safety": "PHI: 14 days.",
                "type": "chemical",
            },
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2 g / L water",
                "frequency": "Every 10 days in wet periods",
                "safety": "PHI: 66 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Bordeaux mixture 1 %",
            "Neem oil 5 mL / L",
        ],
        "prevention": [
            "Improve canopy management with shoot positioning",
            "Remove and compost infected leaves away from vineyard",
            "Avoid excessive nitrogen fertilization",
        ],
        "etl": "10 % leaf area infected — begin spray programme",
        "fertilizer_note": "Moderate nitrogen; excess promotes lush tissue susceptible to blight.",
    },

    "Grape___healthy": {
        "plant": "Grape", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The grape leaf appears healthy.",
        "pesticides": [],
        "organic": ["Preventive Bordeaux mixture at budbreak (1 %)"],
        "prevention": [
            "Regular canopy management and shoot positioning",
            "Balanced nutrition; soil test before season",
            "Under-vine weed control",
        ],
        "etl": "N/A",
        "fertilizer_note": "Apply potassium sulfate pre-veraison to improve berry quality.",
    },

    # ── ORANGE ─────────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange / Citrus", "condition": "Huanglongbing (Citrus Greening)",
        "severity_risk": "Critical", "is_healthy": False,
        "description": (
            "Caused by Candidatus Liberibacter spp., spread by Asian citrus psyllid "
            "(Diaphorina citri). Produces asymmetric leaf yellowing (blotchy mottle), "
            "small bitter lopsided fruit. INCURABLE — no effective curative treatment."
        ),
        "pesticides": [
            {
                "name": "Imidacloprid 200 SL (psyllid vector control)",
                "dosage": "0.5 mL / L water",
                "frequency": "Every 3 months; avoid during bloom to protect bees",
                "safety": "PHI: 7 days. Highly toxic to bees — do not apply while flowering.",
                "type": "chemical",
            },
            {
                "name": "Thiamethoxam 25 WG (soil drench)",
                "dosage": "0.2 g / L water (soil drench)",
                "frequency": "Quarterly",
                "safety": "PHI: 14 days. Bee hazard.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Kaolin clay (Surround WP) 30 g / L — deters psyllid",
            "Yellow sticky traps for psyllid monitoring",
            "Parasitic wasp (Tamarixia radiata) release for biocontrol",
        ],
        "prevention": [
            "IMMEDIATELY remove and destroy any confirmed infected trees",
            "Use only certified disease-free planting material",
            "Establish quarantine zones; restrict movement of plant material",
            "Monitor psyllid populations weekly with yellow traps",
            "Apply reflective mulch to reduce psyllid landing",
        ],
        "etl": "Any confirmed tree — immediate removal is strongly recommended",
        "fertilizer_note": "Thermotherapy + nutrient injection trials ongoing; not yet commercially viable.",
    },

    # ── PEACH ──────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "plant": "Peach", "condition": "Bacterial Spot",
        "severity_risk": "Medium-High", "is_healthy": False,
        "description": (
            "Xanthomonas arboricola pv. pruni produces water-soaked spots that turn "
            "angular and dark on leaves; causes fruit lesions and defoliation."
        ),
        "pesticides": [
            {
                "name": "Copper hydroxide 77 WP",
                "dosage": "2 g / L water",
                "frequency": "Every 7–10 days from green tip; avoid in hot dry conditions",
                "safety": "PHI: 1 day. Excess copper causes phytotoxicity.",
                "type": "chemical",
            },
            {
                "name": "Oxytetracycline HCl (Mycoshield)",
                "dosage": "As per label",
                "frequency": "At petal fall — 2 applications",
                "safety": "PHI: 21 days. Antibiotic — use judiciously; check local regulations.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper octanoate (Cueva) — 5 mL / L",
            "Bacillus subtilis (Serenade MAX) — 6 g / L",
        ],
        "prevention": [
            "Plant resistant varieties (e.g., Redhaven, Reliance)",
            "Use drip irrigation; avoid wetting foliage",
            "Prune for open canopy to improve airflow",
            "Remove and destroy infected leaves and fruit",
        ],
        "etl": "20 % fruit spotting or significant defoliation",
        "fertilizer_note": "Adequate calcium (Ca) foliar sprays improve fruit skin integrity.",
    },

    "Peach___healthy": {
        "plant": "Peach", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The peach leaf appears healthy.",
        "pesticides": [],
        "organic": ["Preventive copper spray at bud swell (dormant)"],
        "prevention": [
            "Annual dormant pruning to remove old fruiting wood",
            "Balanced N-P-K fertilization",
            "Monitor for peach leaf curl at bud swell",
        ],
        "etl": "N/A",
        "fertilizer_note": "Apply nitrogen in early spring; split applications are optimal.",
    },

    # ── BELL PEPPER ────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper", "condition": "Bacterial Spot",
        "severity_risk": "Medium-High", "is_healthy": False,
        "description": (
            "Xanthomonas euvesicatoria causes water-soaked spots turning "
            "necrotic on leaves and fruit; causes defoliation in severe cases."
        ),
        "pesticides": [
            {
                "name": "Copper hydroxide 77 WP",
                "dosage": "3 g / L water",
                "frequency": "Every 7 days; avoid in extreme heat (>35 °C)",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
            {
                "name": "Copper octanoate + Mancozeb tank mix",
                "dosage": "2 g / L + 2 g / L",
                "frequency": "Weekly during wet periods",
                "safety": "PHI: 5 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper octanoate (Cueva) — 5 mL / L",
            "Bacillus subtilis (Serenade) — 3 g / L",
        ],
        "prevention": [
            "Use certified disease-free / treated seeds",
            "Drip irrigation to avoid foliage wetting",
            "Remove and destroy infected plants promptly",
            "Disinfect tools and stakes between rows",
        ],
        "etl": "First lesions on 20 % of plants in the field",
        "fertilizer_note": "Adequate phosphorus and calcium promote stronger cell walls.",
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The bell pepper leaf appears healthy.",
        "pesticides": [],
        "organic": ["Apply compost at transplanting; neem cake in soil (200 kg/ha)"],
        "prevention": [
            "Crop rotation — avoid planting in same spot as Solanaceae",
            "Adequate plant spacing for airflow",
            "Weed management",
        ],
        "etl": "N/A",
        "fertilizer_note": "Use DAP (18-46-0) at transplanting; top-dress with urea at flowering.",
    },

    # ── POTATO ─────────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "plant": "Potato", "condition": "Early Blight",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Alternaria solani creates dark brown lesions with concentric "
            "rings (target-board pattern) on older leaves; defoliation reduces yield."
        ),
        "pesticides": [
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7–10 days from first symptom",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Chlorothalonil 75 WP",
                "dosage": "2 g / L water",
                "frequency": "Every 7–14 days",
                "safety": "PHI: 7 days. Wear respiratory protection.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper oxychloride 50 WP — 3 g / L",
            "Neem oil 5 mL / L + 1 mL / L soap emulsifier",
        ],
        "prevention": [
            "Destroy infected crop debris after harvest",
            "Use certified disease-free seed tubers",
            "Avoid overhead irrigation; use furrow or drip",
            "Maintain crop vigor with timely nitrogen application",
        ],
        "etl": "First lesions on lower leaves — begin protectant sprays",
        "fertilizer_note": "Potassium deficiency greatly increases early blight severity.",
    },

    "Potato___Late_blight": {
        "plant": "Potato", "condition": "Late Blight",
        "severity_risk": "Critical", "is_healthy": False,
        "description": (
            "Phytophthora infestans creates rapidly spreading water-soaked "
            "lesions that turn brown; can destroy an entire crop within days under "
            "cool humid conditions. The pathogen of the Irish Potato Famine."
        ),
        "pesticides": [
            {
                "name": "Metalaxyl-M + Mancozeb (Ridomil Gold MZ)",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7 days; use only when disease is active",
                "safety": "PHI: 7 days. Rotate with non-phenylamide fungicides.",
                "type": "chemical",
            },
            {
                "name": "Fluazinam (Shirlan 500 SC)",
                "dosage": "0.5 mL / L water",
                "frequency": "Every 7–10 days preventively",
                "safety": "PHI: 7 days. Irritant — wear PPE.",
                "type": "chemical",
            },
            {
                "name": "Cymoxanil + Mancozeb (Curzate)",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7 days — curative within 48 hrs of infection",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper oxychloride 50 WP — 3 g / L (preventive only)",
            "Copper hydroxide (Kocide) — 2 g / L",
        ],
        "prevention": [
            "Plant certified blight-resistant varieties (e.g., Sarpo Mira)",
            "Destroy volunteer potato plants",
            "Earth-up plants to protect tubers",
            "Harvest promptly; destroy haulm before harvest to avoid tuber infection",
            "Monitor using BlightCast / Smith Period forecasting models",
        ],
        "etl": "IMMEDIATE action at first sign — do not wait for threshold",
        "fertilizer_note": "Excess nitrogen creates soft tissue highly susceptible to late blight.",
    },

    "Potato___healthy": {
        "plant": "Potato", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The potato leaf appears healthy.",
        "pesticides": [],
        "organic": ["Preventive copper spray when BlightCast risk is moderate"],
        "prevention": [
            "Use certified seed tubers",
            "Rotate with non-Solanaceae crops",
            "Hill up soil around plants regularly",
        ],
        "etl": "N/A",
        "fertilizer_note": "Potatoes are heavy feeders; apply NPK 17-17-17 at planting.",
    },

    # ── RASPBERRY ──────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "plant": "Raspberry", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The raspberry leaf appears healthy.",
        "pesticides": [],
        "organic": ["Sulfur spray preventively in high humidity periods"],
        "prevention": [
            "Remove old floricanes after fruiting",
            "Maintain open row structure for airflow",
            "Mulch to prevent soil splash",
        ],
        "etl": "N/A",
        "fertilizer_note": "Apply ammonium nitrate (27-0-0) in early spring.",
    },

    # ── SOYBEAN ────────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "plant": "Soybean", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The soybean leaf appears healthy.",
        "pesticides": [],
        "organic": ["Apply Rhizobium inoculant at planting for nitrogen fixation"],
        "prevention": [
            "Crop rotation with non-legume crops",
            "Avoid compaction by minimizing field traffic",
            "Use certified disease-free seeds",
        ],
        "etl": "N/A",
        "fertilizer_note": "Inoculated soybeans need minimal nitrogen; focus on P and K.",
    },

    # ── SQUASH ─────────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "plant": "Squash", "condition": "Powdery Mildew",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Podosphaeria xanthii and Erysiphe cichoracearum create white powdery "
            "patches on leaves and stems; reduces photosynthesis and fruit quality."
        ),
        "pesticides": [
            {
                "name": "Myclobutanil 40 WP",
                "dosage": "0.4 g / L water",
                "frequency": "Every 10 days at first sign",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
            {
                "name": "Azoxystrobin 23 SC",
                "dosage": "1 mL / L water",
                "frequency": "Every 14 days; rotate with non-strobilurin",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Potassium bicarbonate 5 g / L (Milstop)",
            "Neem oil 5 mL / L + soap",
            "Sulfur 80 WG 3 g / L (avoid in heat)",
        ],
        "prevention": [
            "Plant resistant cultivars",
            "Ensure good spacing and airflow between plants",
            "Remove heavily infected leaves early",
        ],
        "etl": "First white colonies on any plant — begin spray",
        "fertilizer_note": "Avoid excess nitrogen; silica supplementation reduces mildew.",
    },

    # ── STRAWBERRY ─────────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry", "condition": "Leaf Scorch",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Diplocarpon earlianum creates dark purple to reddish irregular "
            "spots on leaves; in severe cases leaves appear scorched and die."
        ),
        "pesticides": [
            {
                "name": "Captan 50 WP",
                "dosage": "2 g / L water",
                "frequency": "Every 10–14 days",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
            {
                "name": "Myclobutanil (Rally 40 WP)",
                "dosage": "0.4 g / L water",
                "frequency": "Every 14 days",
                "safety": "PHI: 1 day.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper octanoate (Cueva) 5 mL / L",
            "Sulfur 80 WG 2 g / L",
        ],
        "prevention": [
            "Remove infected leaves and runners",
            "Avoid overhead irrigation; use drip",
            "Renovate beds by mowing after harvest",
            "Plant resistant cultivars (e.g., Tribute, Tristar)",
        ],
        "etl": "15 % defoliation or significant crown impact",
        "fertilizer_note": "Balanced nutrition; avoid excessive nitrogen which extends infection period.",
    },

    "Strawberry___healthy": {
        "plant": "Strawberry", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The strawberry leaf appears healthy.",
        "pesticides": [],
        "organic": ["Monthly neem oil spray (3 mL / L) as preventive"],
        "prevention": [
            "Replace planting every 3–4 years",
            "Renovate by mowing immediately after harvest",
            "Mulch with straw to prevent splash dispersal",
        ],
        "etl": "N/A",
        "fertilizer_note": "Apply balanced fertilizer (NPK 13-13-13) at planting and fruit set.",
    },

    # ── TOMATO ─────────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "plant": "Tomato", "condition": "Bacterial Spot",
        "severity_risk": "Medium-High", "is_healthy": False,
        "description": (
            "Xanthomonas spp. cause water-soaked spots that turn dark and "
            "scabby on leaves and fruit; defoliation exposes fruit to sunscald."
        ),
        "pesticides": [
            {
                "name": "Copper hydroxide 77 WP",
                "dosage": "3 g / L water",
                "frequency": "Every 7 days; especially after rain",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
            {
                "name": "Copper sulfate + Mancozeb (fixed copper)",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7–10 days",
                "safety": "PHI: 5 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Bacillus subtilis (Serenade) — 3 g / L",
            "Copper octanoate (Cueva) — 5 mL / L",
        ],
        "prevention": [
            "Use certified pathogen-free seed or hot-water treated seed",
            "Stake and train plants for airflow",
            "Drip irrigation; avoid overhead watering",
            "Rotate with non-Solanaceae crops for 2 years",
        ],
        "etl": "20 % plants with lesions or 10 % fruit infection",
        "fertilizer_note": "Calcium foliar spray (0.5 % CaCl2) reduces bacterial entry points.",
    },

    "Tomato___Early_blight": {
        "plant": "Tomato", "condition": "Early Blight",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Alternaria solani produces dark concentric-ring lesions on lower "
            "leaves; collar rot can occur at the stem base of seedlings."
        ),
        "pesticides": [
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7 days from first symptom",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Azoxystrobin 23 SC",
                "dosage": "1 mL / L water",
                "frequency": "Every 14 days; rotate with contact fungicide",
                "safety": "PHI: 0 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper oxychloride 50 WP — 3 g / L",
            "Neem oil 5 mL / L",
            "Bacillus subtilis (Serenade) — 3 g / L",
        ],
        "prevention": [
            "Avoid planting in same Solanaceae field two years running",
            "Remove and destroy lower infected leaves promptly",
            "Stake plants; ensure airflow",
            "Apply mulch to reduce soil splash",
        ],
        "etl": "First lesions on bottom leaves — begin sprays",
        "fertilizer_note": "Nitrogen deficiency weakens plants; maintain adequate N.",
    },

    "Tomato___Late_blight": {
        "plant": "Tomato", "condition": "Late Blight",
        "severity_risk": "Critical", "is_healthy": False,
        "description": (
            "Phytophthora infestans causes rapidly advancing water-soaked lesions "
            "on leaves and fruit; can destroy an entire crop in 7–10 days in wet conditions."
        ),
        "pesticides": [
            {
                "name": "Metalaxyl-M + Mancozeb (Ridomil Gold MZ)",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7 days when disease active",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Fluazinam 500 SC",
                "dosage": "0.5 mL / L water",
                "frequency": "Every 7–10 days preventively",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper hydroxide (Kocide) — 2 g / L (preventive only)",
        ],
        "prevention": [
            "Grow resistant varieties (e.g., Mountain Merit, Defiant)",
            "Remove volunteer tomato and potato plants",
            "Avoid overhead irrigation",
            "Scout twice weekly in humid weather",
        ],
        "etl": "IMMEDIATE spray at first lesion; do not wait",
        "fertilizer_note": "High nitrogen softens tissue; balance with adequate potassium.",
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato", "condition": "Leaf Mold",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Passalora fulva (Cladosporium fulvum) causes pale green-yellow "
            "patches on upper leaf surface with olive-green mold on the underside. "
            "Primarily a problem in greenhouse / high-humidity conditions."
        ),
        "pesticides": [
            {
                "name": "Chlorothalonil 75 WP",
                "dosage": "2 g / L water",
                "frequency": "Every 7 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2.5 g / L",
                "frequency": "Every 7–10 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper oxychloride — 3 g / L",
            "Bacillus subtilis (Serenade) — 3 g / L",
        ],
        "prevention": [
            "Reduce greenhouse humidity below 85 % (ventilation, heating)",
            "Increase plant spacing",
            "Remove lower leaves to improve airflow",
            "Avoid wetting foliage when watering",
        ],
        "etl": "Patches on > 10 % of leaf area",
        "fertilizer_note": "No specific nutritional link; maintain general crop health.",
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato", "condition": "Septoria Leaf Spot",
        "severity_risk": "Medium-High", "is_healthy": False,
        "description": (
            "Septoria lycopersici produces numerous small circular spots with "
            "dark borders and lighter centres on lower leaves; causes rapid defoliation."
        ),
        "pesticides": [
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2.5 g / L water",
                "frequency": "Every 7–10 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
            {
                "name": "Chlorothalonil 75 WP",
                "dosage": "2 g / L",
                "frequency": "Every 7 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper hydroxide 77 WP — 2 g / L",
            "Bacillus subtilis (Serenade) — 3 g / L",
        ],
        "prevention": [
            "Remove lower leaves that show first lesions",
            "Avoid working in field when wet",
            "Mulch to prevent soil splash (primary infection source)",
            "Crop rotation 2-year minimum",
        ],
        "etl": "First spots on lower leaves — begin spray",
        "fertilizer_note": "Stressed (N-deficient) plants are more susceptible.",
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato", "condition": "Two-Spotted Spider Mite",
        "severity_risk": "Medium-High", "is_healthy": False,
        "description": (
            "Tetranychus urticae causes stippled yellowing on upper leaf surface; "
            "fine webbing underneath; leaves turn bronze then die. Worst in hot, dry conditions."
        ),
        "pesticides": [
            {
                "name": "Abamectin 1.8 EC (Vertimec)",
                "dosage": "0.5 mL / L water",
                "frequency": "Every 7 days; max 2 consecutive applications",
                "safety": "PHI: 3 days. Avoid in high temperatures.",
                "type": "chemical",
            },
            {
                "name": "Bifenazate (Floramite SC)",
                "dosage": "1 mL / L water",
                "frequency": "Single application; rotate",
                "safety": "PHI: 3 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Neem oil 5 mL / L — suffocates eggs and nymphs",
            "Predatory mites (Phytoseiulus persimilis) release",
            "Insecticidal soap 10 mL / L",
            "Strong water jet spray to dislodge mites",
        ],
        "prevention": [
            "Maintain adequate irrigation (dry conditions favour mites)",
            "Avoid dusty field conditions",
            "Monitor with hand lens; check leaf undersides weekly",
            "Conserve natural predators (avoid broad-spectrum insecticides)",
        ],
        "etl": "Average 5 motile mites per leaflet across 10 leaflets",
        "fertilizer_note": "Silica supplementation (potassium silicate 1 g / L) deters mites.",
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato", "condition": "Target Spot",
        "severity_risk": "Medium", "is_healthy": False,
        "description": (
            "Corynespora cassiicola creates brown circular lesions with "
            "concentric rings on leaves, stems, and fruit in warm humid conditions."
        ),
        "pesticides": [
            {
                "name": "Azoxystrobin + Difenoconazole (Amistar Top)",
                "dosage": "1 mL / L water",
                "frequency": "Every 14 days",
                "safety": "PHI: 3 days.",
                "type": "chemical",
            },
            {
                "name": "Mancozeb 75 WP",
                "dosage": "2.5 g / L",
                "frequency": "Every 7 days",
                "safety": "PHI: 7 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Copper oxychloride 50 WP — 3 g / L",
            "Trichoderma viride WP — 5 g / L",
        ],
        "prevention": [
            "Prune lower leaves for airflow",
            "Avoid wetting leaves when irrigating",
            "Remove crop debris promptly after season",
        ],
        "etl": "10 % leaf area infected",
        "fertilizer_note": "Maintain potassium to support plant immune response.",
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato", "condition": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "severity_risk": "Critical", "is_healthy": False,
        "description": (
            "TYLCV (Begomovirus, transmitted by silverleaf whitefly Bemisia tabaci) "
            "causes upward leaf curling, yellowing, stunting, and flower drop. No cure."
        ),
        "pesticides": [
            {
                "name": "Imidacloprid 70 WG (whitefly control)",
                "dosage": "0.2 g / L water",
                "frequency": "Every 14 days; avoid during bloom",
                "safety": "PHI: 7 days. Harmful to bees.",
                "type": "chemical",
            },
            {
                "name": "Thiamethoxam 25 WG",
                "dosage": "0.2 g / L",
                "frequency": "Soil drench at transplanting",
                "safety": "PHI: 14 days.",
                "type": "chemical",
            },
        ],
        "organic": [
            "Reflective silver mulch to deter whitefly",
            "Yellow sticky traps (4 per 100 m²) for monitoring and mass trapping",
            "Neem oil 5 mL / L — repels whitefly",
            "Insecticidal soap 10 mL / L",
        ],
        "prevention": [
            "Remove and destroy infected plants immediately",
            "Use TYLCV-resistant varieties where available",
            "Insect-proof netting on young nursery plants",
            "Avoid planting adjacent to other Solanaceae",
            "Whitefly populations must be managed from seedling stage",
        ],
        "etl": "Any confirmed infected plant — remove immediately",
        "fertilizer_note": "No nutritional cure; focus on vector management.",
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato", "condition": "Tomato Mosaic Virus (ToMV)",
        "severity_risk": "High", "is_healthy": False,
        "description": (
            "ToMV (Tobamovirus) causes mosaic light/dark green mottling, "
            "leaf distortion, and fruit blemishes. Mechanically transmitted; extremely stable."
        ),
        "pesticides": [
            {
                "name": "No effective chemical treatment — virus management only",
                "dosage": "N/A",
                "frequency": "N/A",
                "safety": "No curative pesticide exists for ToMV.",
                "type": "N/A",
            },
        ],
        "organic": [
            "Milk spray (10 % whole milk) — inactivates virus on surfaces",
            "Skim milk as tool disinfectant",
        ],
        "prevention": [
            "Use ToMV-resistant varieties (Tm-2a gene)",
            "Disinfect ALL tools with 10 % bleach or 70 % ethanol",
            "Wash hands thoroughly before handling plants",
            "Never smoke near tomato plants (tobacco carries TMV)",
            "Remove and destroy all infected plants",
            "Avoid re-using soil from infected plots",
        ],
        "etl": "Any confirmed plant — remove to prevent mechanical spread",
        "fertilizer_note": "No nutritional interventions are effective against virus.",
    },

    "Tomato___healthy": {
        "plant": "Tomato", "condition": "Healthy",
        "severity_risk": "None", "is_healthy": True,
        "description": "No disease detected. The tomato leaf appears healthy.",
        "pesticides": [],
        "organic": ["Preventive copper spray (1.5 g / L) in humid periods"],
        "prevention": [
            "Stake and train plants to maximise airflow",
            "Drip irrigation; mulch soil surface",
            "Crop rotation — 3-year break from Solanaceae",
        ],
        "etl": "N/A",
        "fertilizer_note": "Use calcium-rich fertilizer at fruit set to prevent blossom-end rot.",
    },
}

# ── Fertilizer catalogue (from research doc) ──────────────────────────────────
FERTILIZER_CATALOGUE = [
    {
        "name": "Urea",
        "npk": "46-0-0",
        "price_inr_per_mt": 5377,
        "scheme": "Urea Subsidy Scheme",
        "best_for": "Nitrogen top-dressing; leafy growth stages",
    },
    {
        "name": "DAP (Di-Ammonium Phosphate)",
        "npk": "18-46-0",
        "price_inr_per_mt": 27000,
        "scheme": "Nutrient Based Subsidy",
        "best_for": "Planting / transplanting; phosphorus boost for roots",
    },
    {
        "name": "MOP (Muriate of Potash)",
        "npk": "0-0-60",
        "price_inr_per_mt": 31319,
        "scheme": "Nutrient Based Subsidy",
        "best_for": "Potassium supplement; fruit and disease resistance",
    },
    {
        "name": "NPK 10-26-26",
        "npk": "10-26-26",
        "price_inr_per_mt": 29941,
        "scheme": "Nutrient Based Subsidy",
        "best_for": "Balanced fertilizer for fruiting / flowering crops",
    },
    {
        "name": "SSP (Single Super Phosphate — Granular)",
        "npk": "0-16-0",
        "price_inr_per_mt": 10828,
        "scheme": "Nutrient Based Subsidy",
        "best_for": "Low-cost phosphorus; also supplies sulfur",
    },
]


def get_treatment(class_name: str) -> dict[str, Any] | None:
    """Return treatment data for a given PlantVillage class label."""
    return TREATMENT_DB.get(class_name)


def get_all_classes() -> list[str]:
    """Return all supported disease class labels."""
    return list(TREATMENT_DB.keys())


def get_fertilizer_catalogue() -> list[dict[str, Any]]:
    """Return the complete fertilizer catalogue."""
    return FERTILIZER_CATALOGUE
