import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { sendChatMessage } from "../services/chatApi";
import { useVoiceChat } from "../hooks/useVoiceChat";

// ============================================================
// INLINE ICONS
// ============================================================
const Icons = {
    Bot: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="11" width="18" height="10" rx="2" />
            <circle cx="12" cy="5" r="2" />
            <path d="M12 7v4" />
            <line x1="8" y1="16" x2="8" y2="16" strokeWidth="3" strokeLinecap="round" />
            <line x1="12" y1="16" x2="12" y2="16" strokeWidth="3" strokeLinecap="round" />
            <line x1="16" y1="16" x2="16" y2="16" strokeWidth="3" strokeLinecap="round" />
        </svg>
    ),
    Close: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
    ),
    Send: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
        </svg>
    ),
    Mic: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" y1="19" x2="12" y2="23" />
            <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
    ),
    MicOff: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="1" y1="1" x2="23" y2="23" />
            <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
            <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
            <line x1="12" y1="19" x2="12" y2="23" />
            <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
    ),
    Volume: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
        </svg>
    ),
    Leaf: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z" />
            <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
        </svg>
    ),
    Wifi: (p) => (
        <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="1" y1="1" x2="23" y2="23" />
            <path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55" />
            <path d="M5 12.55a11 11 0 0 1 5.17-2.39" />
        </svg>
    ),
};

// ============================================================
// CONSTANTS
// ============================================================
const LANG_LABELS = { en: "EN", te: "TE", hi: "HI" };

const GREETING = {
    en: "👋 Hi! I'm KrishiAI assistant. Ask me about crop care, or describe your crop's condition.",
    te: "👋 నమస్కారం! నేను KrishiAI సహాయకుడిని. పంట సంరక్షణ గురించి అడగండి.",
    hi: "👋 नमस्ते! मैं KrishiAI सहायक हूँ। फसल देखभाल के बारे में पूछें।",
};

const SCAN_PROMPT = {
    en: "📸 Please scan your crop for accurate diagnosis.",
    te: "📸 ఖచ్చితమైన నిర్ధారణ కోసం మీ పంటను స్కాన్ చేయండి.",
    hi: "📸 सटीक निदान के लिए कृपया अपनी फसल को स्कैन करें।",
};

// ============================================================
// MESSAGE BUBBLE
// ============================================================
function MessageBubble({ msg, onSpeak }) {
    const isBot = msg.role === "bot";
    return (
        <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.25 }}
            className={`flex ${isBot ? "justify-start" : "justify-end"} mb-2`}
        >
            {isBot && (
                <div className="w-7 h-7 rounded-full flex items-center justify-center mr-2 flex-shrink-0"
                    style={{ background: "rgba(74,163,82,0.25)", border: "1px solid rgba(74,163,82,0.4)" }}>
                    <Icons.Bot style={{ width: 14, height: 14, color: "#4ade80" }} />
                </div>
            )}
            <div style={{
                maxWidth: "78%",
                padding: "9px 13px",
                borderRadius: isBot ? "4px 16px 16px 16px" : "16px 4px 16px 16px",
                background: isBot
                    ? "rgba(255,255,255,0.08)"
                    : "linear-gradient(135deg,#16a34a,#22c55e)",
                border: isBot ? "1px solid rgba(255,255,255,0.1)" : "none",
                backdropFilter: "blur(10px)",
                fontSize: 13,
                lineHeight: 1.5,
                color: "#fff",
                wordBreak: "break-word",
            }}>
                {msg.text}
                {msg.offline && (
                    <span style={{ display: "block", fontSize: 10, color: "rgba(255,255,255,0.45)", marginTop: 3 }}>
                        ⚡ offline mode
                    </span>
                )}
                {msg.redirect && (
                    <span style={{ display: "block", fontSize: 11, color: "#4ade80", marginTop: 4, fontWeight: 700 }}>
                        🌿 Redirecting to Scan...
                    </span>
                )}
            </div>
            {isBot && (
                <button
                    onClick={() => onSpeak(msg.text, msg.lang)}
                    style={{ marginLeft: 6, background: "none", border: "none", cursor: "pointer", padding: 4, flexShrink: 0 }}
                    title="Speak"
                >
                    <Icons.Volume style={{ width: 14, height: 14, color: "rgba(255,255,255,0.35)" }} />
                </button>
            )}
        </motion.div>
    );
}

// ============================================================
// TYPING INDICATOR
// ============================================================
function TypingIndicator() {
    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-1 mb-2 ml-9">
            {[0, 1, 2].map((i) => (
                <motion.div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ade80" }}
                    animate={{ y: [0, -5, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                />
            ))}
        </motion.div>
    );
}

// ============================================================
// MAIN CHATBOT COMPONENT
// ============================================================
export default function ChatBot({ onNavigate }) {
    const [open, setOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [chatLang, setChatLang] = useState("en");
    const [continuous, setContinuous] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const { isListening, transcript, detectedLang, startListening, stopListening, speak, supported } =
        useVoiceChat(chatLang);

    // Sync detected language back to chatLang
    useEffect(() => {
        if (detectedLang && detectedLang !== chatLang) {
            setChatLang(detectedLang);
        }
    }, [detectedLang]);

    // Show greeting on first open
    useEffect(() => {
        if (open && messages.length === 0) {
            setMessages([{ id: Date.now(), role: "bot", text: GREETING[chatLang] || GREETING.en, lang: chatLang }]);
        }
    }, [open]);

    // Auto-scroll
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    // Update transcript in input box while speaking
    useEffect(() => {
        if (transcript) setInput(transcript);
    }, [transcript]);

    const addMessage = useCallback((role, text, extras = {}) => {
        setMessages((prev) => [...prev, { id: Date.now() + Math.random(), role, text, lang: chatLang, ...extras }]);
    }, [chatLang]);

    const sendMessage = useCallback(async (text, lang) => {
        const msg = (text || input).trim();
        if (!msg) return;
        setInput("");
        addMessage("user", msg);
        setLoading(true);

        const activeLang = lang || chatLang;

        try {
            const res = await sendChatMessage({ message: msg, language: activeLang, mode: "text" });
            const replyText = res.redirect_to_scan
                ? (SCAN_PROMPT[activeLang] || SCAN_PROMPT.en)
                : res.reply;

            addMessage("bot", replyText, { offline: res.offline, redirect: res.redirect_to_scan });

            // Speak the reply
            speak(replyText, activeLang);

            // Redirect to scan page if needed
            if (res.redirect_to_scan && onNavigate) {
                setTimeout(() => {
                    onNavigate("scan");
                    setOpen(false);
                }, 2000);
            }
        } catch {
            addMessage("bot", "Sorry, something went wrong. Please try again.", { offline: true });
        } finally {
            setLoading(false);
        }
    }, [input, chatLang, addMessage, speak, onNavigate]);

    const handleVoiceToggle = () => {
        if (isListening) {
            stopListening();
        } else {
            startListening((text, lang) => {
                sendMessage(text, lang);
            }, { continuous });
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    // ============================================================
    // RENDER
    // ============================================================
    return (
        <>
            {/* Floating Button + Label */}
            <div style={{ position: "fixed", bottom: 92, right: 16, zIndex: 9999, display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>


                {/* Main button */}
                <motion.button
                    onClick={() => setOpen((o) => !o)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    animate={open
                        ? { scale: 1 }
                        : { y: [0, -6, 0, -4, 0], scale: [1, 1.04, 1, 1.02, 1] }}
                    transition={open ? {} : { duration: 2.5, repeat: Infinity, repeatDelay: 2.5, ease: "easeInOut" }}
                    style={{
                        width: 68,
                        height: 68,
                        borderRadius: "50%",
                        border: "none",
                        cursor: "pointer",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 2,
                        background: open
                            ? "linear-gradient(135deg,#15803d,#166534)"
                            : "linear-gradient(135deg,#22c55e,#16a34a,#15803d)",
                        boxShadow: open
                            ? "0 0 0 4px rgba(34,197,94,0.3), 0 8px 28px rgba(0,0,0,0.5)"
                            : "0 0 32px rgba(34,197,94,0.65), 0 0 64px rgba(46,125,50,0.3), 0 8px 28px rgba(0,0,0,0.5)",
                    }}
                    aria-label="Open KrishiAI Chatbot"
                    title="Talk to KrishiAI"
                >
                    <AnimatePresence mode="wait">
                        {open ? (
                            <motion.span key="close" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
                                <Icons.Close style={{ width: 26, height: 26, color: "#fff" }} />
                            </motion.span>
                        ) : (
                            <motion.span key="bot" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.2 }}>
                                <Icons.Bot style={{ width: 28, height: 28, color: "#fff" }} />
                            </motion.span>
                        )}
                    </AnimatePresence>

                    {/* Outer pulse ring */}
                    {!open && (
                        <motion.span
                            style={{
                                position: "absolute", inset: -6, borderRadius: "50%",
                                border: "2.5px solid rgba(34,197,94,0.5)", pointerEvents: "none",
                            }}
                            animate={{ scale: [1, 1.5, 1], opacity: [0.7, 0, 0.7] }}
                            transition={{ duration: 2.2, repeat: Infinity }}
                        />
                    )}
                    {/* Inner pulse ring */}
                    {!open && (
                        <motion.span
                            style={{
                                position: "absolute", inset: -2, borderRadius: "50%",
                                border: "2px solid rgba(6,182,212,0.4)", pointerEvents: "none",
                            }}
                            animate={{ scale: [1, 1.25, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2.2, repeat: Infinity, delay: 0.5 }}
                        />
                    )}
                </motion.button>

                {/* "Ask AI" label below button */}
                {!open && (
                    <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        style={{
                            background: "linear-gradient(135deg,#16a34a,#0d9488)",
                            borderRadius: 20,
                            padding: "3px 12px",
                            fontSize: 11,
                            fontWeight: 800,
                            color: "#fff",
                            letterSpacing: "0.05em",
                            boxShadow: "0 2px 12px rgba(22,163,74,0.45)",
                            userSelect: "none",
                            pointerEvents: "none",
                        }}>
                        Ask Krishi AI
                    </motion.div>
                )}
            </div>

            {/* Chat Panel */}
            <AnimatePresence>
                {open && (
                    <motion.div
                        key="chatpanel"
                        initial={{ opacity: 0, y: 30, scale: 0.92 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 30, scale: 0.92 }}
                        transition={{ type: "spring", stiffness: 280, damping: 28 }}
                        style={{
                            position: "fixed",
                            bottom: 155,
                            right: 16,
                            zIndex: 9998,
                            width: "min(380px, calc(100vw - 32px))",
                            height: "min(540px, calc(100dvh - 200px))",
                            borderRadius: 20,
                            display: "flex",
                            flexDirection: "column",
                            overflow: "hidden",
                            background: "rgba(15,45,26,0.92)",
                            backdropFilter: "blur(24px)",
                            border: "1px solid rgba(74,163,82,0.25)",
                            boxShadow: "0 24px 64px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06)",
                        }}
                    >
                        {/* Header */}
                        <div style={{
                            padding: "14px 16px",
                            background: "rgba(22,163,74,0.15)",
                            borderBottom: "1px solid rgba(74,163,82,0.2)",
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            flexShrink: 0,
                        }}>
                            <div style={{
                                width: 36, height: 36, borderRadius: "50%", display: "flex",
                                alignItems: "center", justifyContent: "center",
                                background: "linear-gradient(135deg,#16a34a,#22c55e)",
                            }}>
                                <Icons.Bot style={{ width: 18, height: 18, color: "#fff" }} />
                            </div>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontWeight: 800, fontSize: 14, color: "#fff" }}>KrishiAI Assistant</div>
                                <div style={{ fontSize: 11, color: "#4ade80", display: "flex", alignItems: "center", gap: 4 }}>
                                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ade80", display: "inline-block" }} />
                                    Online · Multilingual
                                </div>
                            </div>
                            {/* Language toggle */}
                            <div style={{ display: "flex", gap: 3, borderRadius: 8, overflow: "hidden", border: "1px solid rgba(255,255,255,0.15)" }}>
                                {["en", "te", "hi"].map((l) => (
                                    <button key={l} onClick={() => setChatLang(l)}
                                        style={{
                                            padding: "3px 7px", fontSize: 10, fontWeight: 700,
                                            border: "none", cursor: "pointer",
                                            background: chatLang === l ? "#16a34a" : "transparent",
                                            color: chatLang === l ? "#fff" : "rgba(255,255,255,0.5)",
                                            transition: "all 0.2s",
                                        }}>
                                        {LANG_LABELS[l]}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Messages */}
                        <div style={{
                            flex: 1, overflowY: "auto", padding: "14px 12px 8px", scrollbarWidth: "thin",
                            scrollbarColor: "rgba(74,163,82,0.3) transparent"
                        }}>
                            {messages.map((msg) => (
                                <MessageBubble key={msg.id} msg={msg} onSpeak={speak} />
                            ))}
                            {loading && <TypingIndicator />}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Voice transcript preview */}
                        <AnimatePresence>
                            {isListening && transcript && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                                    style={{ padding: "6px 14px", background: "rgba(74,163,82,0.08)", borderTop: "1px solid rgba(74,163,82,0.15)" }}>
                                    <span style={{ fontSize: 12, color: "#86efac", fontStyle: "italic" }}>🎤 {transcript}</span>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Continuous mode toggle */}
                        <div style={{ padding: "4px 14px", display: "flex", alignItems: "center", gap: 6 }}>
                            <label style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer", fontSize: 11, color: "rgba(255,255,255,0.45)" }}>
                                <div
                                    onClick={() => setContinuous((c) => !c)}
                                    style={{
                                        width: 28, height: 16, borderRadius: 8, position: "relative", cursor: "pointer",
                                        background: continuous ? "#16a34a" : "rgba(255,255,255,0.15)",
                                        transition: "background 0.25s",
                                    }}>
                                    <motion.div animate={{ x: continuous ? 13 : 1 }} transition={{ type: "spring", stiffness: 400, damping: 25 }}
                                        style={{ position: "absolute", top: 2, width: 12, height: 12, borderRadius: "50%", background: "#fff" }} />
                                </div>
                                Continuous listening
                            </label>
                        </div>

                        {/* Input Row */}
                        <div style={{
                            padding: "10px 12px",
                            borderTop: "1px solid rgba(255,255,255,0.08)",
                            display: "flex",
                            gap: 8,
                            alignItems: "center",
                            flexShrink: 0,
                            background: "rgba(0,0,0,0.2)",
                        }}>
                            {/* Voice button */}
                            {supported && (
                                <motion.button
                                    onClick={handleVoiceToggle}
                                    whileHover={{ scale: 1.1 }}
                                    whileTap={{ scale: 0.9 }}
                                    style={{
                                        width: 38, height: 38, borderRadius: "50%", border: "none", cursor: "pointer",
                                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                        background: isListening ? "#dc2626" : "rgba(74,163,82,0.2)",
                                        boxShadow: isListening ? "0 0 12px rgba(220,38,38,0.5)" : "none",
                                        transition: "all 0.25s",
                                    }}>
                                    {isListening
                                        ? <Icons.MicOff style={{ width: 16, height: 16, color: "#fff" }} />
                                        : <Icons.Mic style={{ width: 16, height: 16, color: "#4ade80" }} />}
                                </motion.button>
                            )}

                            {/* Text input */}
                            <input
                                ref={inputRef}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isListening ? "Listening..." : "Type or speak..."}
                                disabled={loading}
                                style={{
                                    flex: 1, borderRadius: 12, border: "1px solid rgba(255,255,255,0.1)",
                                    background: "rgba(255,255,255,0.06)", color: "#fff", padding: "9px 12px",
                                    fontSize: 13, outline: "none", fontFamily: "inherit",
                                    opacity: loading ? 0.5 : 1,
                                }}
                            />

                            {/* Send button */}
                            <motion.button
                                onClick={() => sendMessage()}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                disabled={!input.trim() || loading}
                                style={{
                                    width: 38, height: 38, borderRadius: "50%", border: "none", cursor: "pointer",
                                    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                    background: input.trim() && !loading ? "linear-gradient(135deg,#16a34a,#22c55e)" : "rgba(255,255,255,0.1)",
                                    transition: "all 0.25s",
                                }}>
                                <Icons.Send style={{ width: 15, height: 15, color: "#fff" }} />
                            </motion.button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
