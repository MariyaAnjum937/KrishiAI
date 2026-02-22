// ============================================================
// OFFLINE FAQ FALLBACK
// ============================================================
const FAQ = {
    en: [
        {
            patterns: ["hello", "hi", "hey", "help"],
            reply: "Hello farmer! I'm KrishiAI assistant. Tell me about your crop, or say 'scan' to check for diseases.",
        },
        {
            patterns: ["weather", "rain", "humidity", "temperature"],
            reply: "High humidity can promote fungal diseases. Keep monitoring your crops and ensure proper drainage.",
        },
        {
            patterns: ["fertilizer", "npk", "urea", "manure"],
            reply: "For best results, follow soil test recommendations. Common advice: NPK 19:19:19 for general growth.",
        },
        {
            patterns: ["pesticide", "spray", "insect", "pest", "bug"],
            reply: "Use recommended pesticides at correct dosage. Spray during early morning or evening. Avoid during rain.",
        },
        {
            patterns: ["scan", "detect", "diagnose", "disease", "sick", "spots", "yellow", "damaged", "blight"],
            reply: "Please scan your crop for accurate diagnosis.",
            redirect_to_scan: true,
        },
    ],
    te: [
        {
            patterns: ["హలో", "హాయ్", "నమస్కారం", "సహాయం"],
            reply: "నమస్కారం రైతు! నేను KrishiAI సహాయకుడిని. మీ పంట గురించి చెప్పండి లేదా వ్యాధి తనిఖీ కోసం 'స్కాన్' అని చెప్పండి.",
        },
        {
            patterns: ["మచ్చలు", "పసుపు", "చెడు", "జబ్బు", "వ్యాధి", "స్కాన్"],
            reply: "ఖచ్చితమైన నిర్ధారణ కోసం మీ పంటను స్కాన్ చేయండి.",
            redirect_to_scan: true,
        },
    ],
    hi: [
        {
            patterns: ["नमस्ते", "हैलो", "हाय", "मदद"],
            reply: "नमस्ते किसान! मैं KrishiAI सहायक हूँ। अपनी फसल के बारे में बताएं या बीमारी जांच के लिए 'स्कैन' कहें।",
        },
        {
            patterns: ["धब्बे", "पीला", "खराब", "बीमार", "रोग", "स्कैन"],
            reply: "सटीक निदान के लिए कृपया अपनी फसल को स्कैन करें।",
            redirect_to_scan: true,
        },
    ],
};

// Keyword patterns that always trigger scan redirect
const SCAN_TRIGGERS = [
    "spot", "spots", "yellow", "damage", "damaged", "sick", "blight", "rot",
    "disease", "leaf", "leaves", "crop", "plant", "infected", "wilting", "wilt",
    "మచ్చలు", "పసుపు", "జబ్బు", "వ్యాధి",
    "धब्बे", "पीला", "बीमार", "रोग",
];

function detectLanguage(text) {
    if (/[\u0C00-\u0C7F]/.test(text)) return "te";
    if (/[\u0900-\u0977]/.test(text)) return "hi";
    return "en";
}

function getOfflineFallback(message, language) {
    const lang = language || detectLanguage(message);
    const lower = message.toLowerCase();

    // Check scan triggers first
    const isScanIntent = SCAN_TRIGGERS.some((kw) => lower.includes(kw));
    if (isScanIntent) {
        const replies = {
            en: "Please scan your crop for accurate diagnosis.",
            te: "ఖచ్చితమైన నిర్ధారణ కోసం మీ పంటను స్కాన్ చేయండి.",
            hi: "सटीक निदान के लिए कृपया अपनी फसल को स्कैन करें।",
        };
        return { reply: replies[lang] || replies.en, redirect_to_scan: true };
    }

    const faqs = FAQ[lang] || FAQ.en;
    for (const faq of faqs) {
        if (faq.patterns.some((p) => lower.includes(p))) {
            return { reply: faq.reply, redirect_to_scan: !!faq.redirect_to_scan };
        }
    }

    const defaults = {
        en: "I'm here to help! Ask me about crop care, or scan your crop for disease detection.",
        te: "నేను సహాయపడేందుకు ఇక్కడ ఉన్నాను! పంట సంరక్షణ గురించి అడగండి లేదా వ్యాధి కోసం స్కాన్ చేయండి.",
        hi: "मैं मदद के लिए यहाँ हूँ! फसल देखभाल के बारे में पूछें, या रोग पहचान के लिए स्कैन करें।",
    };
    return { reply: defaults[lang] || defaults.en, redirect_to_scan: false };
}

// ============================================================
// API CALL
// ============================================================
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function sendChatMessage({ message, language, mode = "text" }) {
    try {
        const res = await fetch(`${BASE_URL}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, language, mode }),
            signal: AbortSignal.timeout(5000),
        });

        if (!res.ok) throw new Error("API error");
        const data = await res.json();
        return {
            reply: data.message || "Sorry, I didn't understand that.",
            redirect_to_scan: !!data.redirect_to_scan,
            offline: false,
        };
    } catch {
        // Graceful offline fallback
        return { ...getOfflineFallback(message, language), offline: true };
    }
}

export { detectLanguage };
