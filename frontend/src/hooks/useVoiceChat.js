import { useState, useRef, useCallback, useEffect } from "react";
import { detectLanguage } from "../services/chatApi";

const LANG_MAP = {
    en: "en-IN",
    te: "te-IN",
    hi: "hi-IN",
};

export function useVoiceChat(currentLang = "en") {
    const [isListening, setIsListening] = useState(false);
    const [transcript, setTranscript] = useState("");
    const [detectedLang, setDetectedLang] = useState(currentLang);
    const recognitionRef = useRef(null);
    const continuousRef = useRef(false);

    // Sync detectedLang when parent lang changes
    useEffect(() => {
        setDetectedLang(currentLang);
    }, [currentLang]);

    const startListening = useCallback(
        (onResult, { continuous = false } = {}) => {
            if (
                !("webkitSpeechRecognition" in window) &&
                !("SpeechRecognition" in window)
            ) {
                console.warn("Speech recognition not supported in this browser.");
                return;
            }

            // Stop any existing instance
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }

            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SR();
            continuousRef.current = continuous;

            recognition.lang = LANG_MAP[detectedLang] || "en-IN";
            recognition.interimResults = true;
            recognition.continuous = continuous;
            recognition.maxAlternatives = 1;

            recognition.onstart = () => {
                setIsListening(true);
                setTranscript("");
            };

            recognition.onresult = (e) => {
                let interim = "";
                let final = "";
                for (let i = e.resultIndex; i < e.results.length; i++) {
                    const text = e.results[i][0].transcript;
                    if (e.results[i].isFinal) {
                        final += text;
                    } else {
                        interim += text;
                    }
                }

                const display = final || interim;
                setTranscript(display);

                if (final) {
                    // Auto-detect language from spoken text
                    const lang = detectLanguage(final);
                    setDetectedLang(lang);
                    onResult(final.trim(), lang);
                    if (!continuous) setTranscript("");
                }
            };

            recognition.onerror = (e) => {
                if (e.error !== "no-speech") {
                    console.error("Speech recognition error:", e.error);
                }
                setIsListening(false);
            };

            recognition.onend = () => {
                setIsListening(false);
                // Auto-restart if continuous mode
                if (continuousRef.current) {
                    try {
                        recognition.start();
                        setIsListening(true);
                    } catch {
                        setIsListening(false);
                    }
                }
            };

            recognitionRef.current = recognition;
            try {
                recognition.start();
            } catch {
                setIsListening(false);
            }
        },
        [detectedLang]
    );

    const stopListening = useCallback(() => {
        continuousRef.current = false;
        if (recognitionRef.current) {
            recognitionRef.current.stop();
            recognitionRef.current = null;
        }
        setIsListening(false);
        setTranscript("");
    }, []);

    const speak = useCallback(
        (text, lang) => {
            if (!("speechSynthesis" in window)) return;
            window.speechSynthesis.cancel();
            const utt = new SpeechSynthesisUtterance(text);
            utt.lang = LANG_MAP[lang || detectedLang] || "en-IN";
            utt.rate = 0.95;
            utt.pitch = 1;
            window.speechSynthesis.speak(utt);
        },
        [detectedLang]
    );

    const cancelSpeech = useCallback(() => {
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }
    }, []);

    return {
        isListening,
        transcript,
        detectedLang,
        startListening,
        stopListening,
        speak,
        cancelSpeech,
        supported:
            "webkitSpeechRecognition" in window || "SpeechRecognition" in window,
    };
}
