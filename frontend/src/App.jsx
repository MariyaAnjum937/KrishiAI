
import { useState, useEffect, useRef, useCallback, createContext, useContext } from "react";
import { motion, AnimatePresence, useAnimation } from "framer-motion";
import ChatBot from "./components/ChatBot";

// ============================================================
// CONTEXT
// ============================================================
const AppContext = createContext(null);

const translations = {
  en: {
    appName: "KrishiAI",
    tagline: "AI-Powered Crop Disease Detection",
    heroTitle: "Protect Your Crops with AI",
    heroSub: "Instant disease detection, treatment recommendations & cost estimates — in your language.",
    scanCTA: "Scan Your Crop",
    viewHistory: "View History",
    uploadLeaf: "Upload Leaf Image",
    capturePhoto: "Capture Photo",
    orDrag: "or drag & drop here",
    analyzeBtn: "Analyze Disease",
    voiceInput: "Speak Symptoms",
    listening: "Listening...",
    language: "Language",
    diseaseDetected: "Disease Detected",
    severity: "Severity",
    healthScore: "Crop Health Score",
    treatment: "Treatment Plan",
    fertilizer: "Fertilizer",
    pesticide: "Pesticide",
    dosage: "Dosage",
    instructions: "Usage Instructions",
    costCalc: "Treatment Cost",
    perAcre: "per acre",
    weatherAlert: "Weather Risk Alert",
    weatherWarning: "High humidity detected – disease may spread rapidly. Take immediate action.",
    shopsNearby: "Nearby Agro-Shops",
    history: "Scan History",
    noHistory: "No scans yet. Start by scanning a crop leaf!",
    offline: "Offline Mode – Showing Cached Data",
    loading: "Analyzing your crop...",
    demo: "Running in Demo Mode",
    low: "Low", medium: "Medium", high: "High",
    healthy: "Healthy",
    back: "Back",
    nav: { home: "Home", scan: "Scan", results: "Results", history: "History" }
  },
  te: {
    appName: "కృషిAI",
    tagline: "AI ఆధారిత పంట వ్యాధి నిర్ధారణ",
    heroTitle: "AI తో మీ పంటను రక్షించండి",
    heroSub: "తక్షణ వ్యాధి గుర్తింపు, చికిత్స సిఫార్సులు & ఖర్చు అంచనాలు — మీ భాషలో.",
    scanCTA: "మీ పంటను స్కాన్ చేయండి",
    viewHistory: "చరిత్ర చూడండి",
    uploadLeaf: "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
    capturePhoto: "ఫోటో తీయండి",
    orDrag: "లేదా ఇక్కడ డ్రాగ్ & డ్రాప్ చేయండి",
    analyzeBtn: "వ్యాధిని విశ్లేషించండి",
    voiceInput: "లక్షణాలు చెప్పండి",
    listening: "వింటున్నాను...",
    language: "భాష",
    diseaseDetected: "వ్యాధి గుర్తించబడింది",
    severity: "తీవ్రత",
    healthScore: "పంట ఆరోగ్య స్కోరు",
    treatment: "చికిత్స ప్రణాళిక",
    fertilizer: "ఎరువు",
    pesticide: "పురుగుమందు",
    dosage: "మోతాదు",
    instructions: "వాడుక సూచనలు",
    costCalc: "చికిత్స వ్యయం",
    perAcre: "ఎకరానికి",
    weatherAlert: "వాతావరణ ప్రమాద హెచ్చరిక",
    weatherWarning: "అధిక తేమ గుర్తించబడింది – వ్యాధి వేగంగా వ్యాపించవచ్చు. వెంటనే చర్య తీసుకోండి.",
    shopsNearby: "సమీప వ్యవసాయ దుకాణాలు",
    history: "స్కాన్ చరిత్ర",
    noHistory: "ఇంకా స్కాన్‌లు లేవు. పంట ఆకును స్కాన్ చేయడం ద్వారా ప్రారంభించండి!",
    offline: "ఆఫ్‌లైన్ మోడ్ – కాష్ చేసిన డేటా చూపబడుతోంది",
    loading: "మీ పంటను విశ్లేషిస్తోంది...",
    demo: "డెమో మోడ్‌లో నడుస్తోంది",
    low: "తక్కువ", medium: "మధ్యమ", high: "అధిక",
    healthy: "ఆరోగ్యకరమైన",
    back: "వెనుకకు",
    nav: { home: "హోమ్", scan: "స్కాన్", results: "ఫలితాలు", history: "చరిత్ర" }
  },
  hi: {
    appName: "कृषिAI",
    tagline: "AI आधारित फसल रोग पहचान",
    heroTitle: "AI से अपनी फसल की रक्षा करें",
    heroSub: "तत्काल रोग पहचान, उपचार सिफारिशें और लागत अनुमान — आपकी भाषा में।",
    scanCTA: "अपनी फसल स्कैन करें",
    viewHistory: "इतिहास देखें",
    uploadLeaf: "पत्ती की छवि अपलोड करें",
    capturePhoto: "फोटो कैप्चर करें",
    orDrag: "या यहाँ खींचें और छोड़ें",
    analyzeBtn: "रोग का विश्लेषण करें",
    voiceInput: "लक्षण बोलें",
    listening: "सुन रहा हूं...",
    language: "भाषा",
    diseaseDetected: "रोग पहचाना गया",
    severity: "गंभीरता",
    healthScore: "फसल स्वास्थ्य स्कोर",
    treatment: "उपचार योजना",
    fertilizer: "उर्वरक",
    pesticide: "कीटनाशक",
    dosage: "खुराक",
    instructions: "उपयोग निर्देश",
    costCalc: "उपचार लागत",
    perAcre: "प्रति एकड़",
    weatherAlert: "मौसम जोखिम चेतावनी",
    weatherWarning: "उच्च आर्द्रता पाई गई – रोग तेजी से फैल सकता है। तत्काल कार्रवाई करें।",
    shopsNearby: "नजदीकी कृषि दुकानें",
    history: "स्कैन इतिहास",
    noHistory: "अभी तक कोई स्कैन नहीं। एक फसल पत्ती स्कैन करके शुरू करें!",
    offline: "ऑफलाइन मोड – कैश डेटा दिखाया जा रहा है",
    loading: "आपकी फसल का विश्लेषण हो रहा है...",
    demo: "डेमो मोड में चल रहा है",
    low: "कम", medium: "मध्यम", high: "उच्च",
    healthy: "स्वस्थ",
    back: "वापस",
    nav: { home: "होम", scan: "स्कैन", results: "परिणाम", history: "इतिहास" }
  }
};

const DEMO_RESULT = {
  diseaseName: "Leaf Blight (Alternaria Solani)",
  severity: "medium",
  healthScore: 54,
  treatment: {
    fertilizer: "NPK 19:19:19",
    pesticide: "Mancozeb 75% WP",
    dosage: "2.5g per litre of water",
    instructions: "Spray evenly on both sides of leaves in the early morning or evening. Repeat after 10 days if symptoms persist. Avoid spraying during rain or intense sunlight."
  },
  costPerAcre: 1850,
  weatherRisk: true,
  shops: [
    { name: "Raju Agro Center", distance: "1.2 km", phone: "9876543210", rating: 4.5 },
    { name: "Krishnamurthy Seeds & Pesticides", distance: "2.8 km", phone: "9123456789", rating: 4.2 },
    { name: "Farmers Friend Store", distance: "3.5 km", phone: "9988776655", rating: 4.7 },
  ]
};

// ============================================================
// UTILITY HOOKS
// ============================================================
function useSpeech(lang) {
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  const langMap = { en: "en-IN", te: "te-IN", hi: "hi-IN" };

  const startListening = useCallback((onResult) => {
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognitionRef.current = new SR();
    recognitionRef.current.lang = langMap[lang] || "en-IN";
    recognitionRef.current.onresult = (e) => onResult(e.results[0][0].transcript);
    recognitionRef.current.onend = () => setIsListening(false);
    recognitionRef.current.start();
    setIsListening(true);
  }, [lang]);

  const speak = useCallback((text) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = langMap[lang] || "en-IN";
    window.speechSynthesis.speak(utt);
  }, [lang]);

  return { isListening, startListening, speak };
}

// ============================================================
// ICON COMPONENTS (inline SVGs to avoid import issues)
// ============================================================
const Icon = {
  Leaf: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z" /><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" /></svg>,
  Camera: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" /><circle cx="12" cy="13" r="4" /></svg>,
  Upload: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="16 16 12 12 8 16" /><line x1="12" y1="12" x2="12" y2="21" /><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" /></svg>,
  Mic: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" /></svg>,
  AlertTriangle: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>,
  MapPin: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" /><circle cx="12" cy="10" r="3" /></svg>,
  Clock: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>,
  Star: (p) => <svg {...p} viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" /></svg>,
  TrendingUp: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>,
  Pill: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.5 20H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H20a2 2 0 0 1 2 2v3" /><circle cx="18" cy="18" r="3" /><path d="m22 22-1.5-1.5" /></svg>,
  Droplets: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M7 16.3c2.2 0 4-1.83 4-4.05 0-1.16-.57-2.26-1.71-3.19S7.29 6.75 7 5.3c-.29 1.45-1.14 2.84-2.29 3.76S3 11.1 3 12.25c0 2.22 1.8 4.05 4 4.05z" /><path d="M12.56 6.6A10.97 10.97 0 0 0 14 3.02c.5 2.5 2 4.9 4 6.5s3 3.5 3 5.5a6.98 6.98 0 0 1-11.91 4.97" /></svg>,
  IndianRupee: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M6 3h12" /><path d="M6 8h12" /><path d="m6 13 8.5 8" /><path d="M6 13h3" /><path d="M9 13c6.667 0 6.667-10 0-10" /></svg>,
  Volume2: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" /></svg>,
  Home: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>,
  Scan: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 7V5a2 2 0 0 1 2-2h2" /><path d="M17 3h2a2 2 0 0 1 2 2v2" /><path d="M21 17v2a2 2 0 0 1-2 2h-2" /><path d="M7 21H5a2 2 0 0 1-2-2v-2" /><line x1="7" y1="12" x2="17" y2="12" /></svg>,
  History: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l4 2" /></svg>,
  CheckCircle: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>,
  Wifi: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12.55a11 11 0 0 1 14.08 0" /><path d="M1.42 9a16 16 0 0 1 21.16 0" /><path d="M8.53 16.11a6 6 0 0 1 6.95 0" /><circle cx="12" cy="20" r="1" fill="currentColor" /></svg>,
  WifiOff: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="1" y1="1" x2="23" y2="23" /><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55" /><path d="M5 12.55a11 11 0 0 1 5.17-2.39" /><path d="M10.71 5.05A16 16 0 0 1 22.56 9" /><path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88" /><path d="M8.53 16.11a6 6 0 0 1 6.95 0" /><circle cx="12" cy="20" r="1" fill="currentColor" /></svg>,
  Phone: (p) => <svg {...p} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.69 12a19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 3.56 1h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 8a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 14.92z" /></svg>,
};

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [lang, setLang] = useState("en");
  const [page, setPage] = useState("home");
  const [scanResult, setScanResult] = useState(null);
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem("krishi_history") || "[]"); } catch { return []; }
  });
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const t = translations[lang];

  useEffect(() => {
    const on = () => setIsOnline(true);
    const off = () => setIsOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => { window.removeEventListener("online", on); window.removeEventListener("offline", off); };
  }, []);

  const addHistory = (result, imageUrl) => {
    const entry = { id: Date.now(), result, imageUrl, date: new Date().toISOString(), lang };
    const updated = [entry, ...history].slice(0, 20);
    setHistory(updated);
    try { localStorage.setItem("krishi_history", JSON.stringify(updated)); } catch { }
  };

  const ctx = { lang, setLang, t, page, setPage, scanResult, setScanResult, history, addHistory, isOnline };

  return (
    <AppContext.Provider value={ctx}>
      <div className="min-h-screen" style={{ fontFamily: "'Nunito', 'Segoe UI', sans-serif", background: "linear-gradient(160deg, #091f17 0%, #0f2d1e 35%, #152b1a 65%, #0d2015 100%)" }}>
        {/* Google Font */}
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap');
          .glass { background: rgba(255,255,255,0.05); backdrop-filter: blur(24px) saturate(160%); border: 1px solid rgba(255,255,255,0.10); box-shadow: 0 4px 32px rgba(0,0,0,0.25); }
          .glass-dark { background: rgba(5,14,26,0.55); backdrop-filter: blur(24px) saturate(160%); border: 1px solid rgba(255,255,255,0.07); }
          .glass-green { background: rgba(6,182,212,0.08); backdrop-filter: blur(24px); border: 1px solid rgba(6,182,212,0.25); }
          .glass-amber { background: rgba(245,158,11,0.08); backdrop-filter: blur(20px); border: 1px solid rgba(245,158,11,0.2); }
          .glow-green { box-shadow: 0 0 40px rgba(6,182,212,0.35), 0 0 80px rgba(16,185,129,0.15); }
          .glow-amber { box-shadow: 0 0 30px rgba(245,158,11,0.4); }
          .text-shadow { text-shadow: 0 2px 32px rgba(0,0,0,0.7); }
          .text-gradient { background: linear-gradient(135deg, #34d399, #06b6d4, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
          .text-gradient-amber { background: linear-gradient(135deg, #fbbf24, #f59e0b, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
          .btn-glow:hover { box-shadow: 0 0 32px rgba(6,182,212,0.55), 0 8px 24px rgba(0,0,0,0.4); transform: translateY(-3px) scale(1.02); }
          .btn-glow { transition: all 0.35s cubic-bezier(0.34,1.56,0.64,1); }
          .btn-amber-glow:hover { box-shadow: 0 0 32px rgba(245,158,11,0.55), 0 8px 24px rgba(0,0,0,0.4); transform: translateY(-3px) scale(1.02); }
          .btn-amber-glow { transition: all 0.35s cubic-bezier(0.34,1.56,0.64,1); }
          .card-hover { transition: transform 0.3s ease, box-shadow 0.3s ease; }
          .card-hover:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 30px rgba(6,182,212,0.15); }
          .no-scrollbar::-webkit-scrollbar { display: none; }
          ::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); } ::-webkit-scrollbar-thumb { background: linear-gradient(#06b6d4,#10b981); border-radius: 4px; }
          .noise-overlay::before { content:''; position:fixed; inset:0; background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E"); pointer-events:none; z-index:0; opacity:0.4; }
          @keyframes shimmer { 0%{background-position:-200% center} 100%{background-position:200% center} }
          .shimmer-text { background: linear-gradient(90deg,#34d399,#06b6d4,#a78bfa,#34d399); background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; animation:shimmer 4s linear infinite; }
          .aurora { background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(46,125,50,0.25), transparent), radial-gradient(ellipse 50% 40% at 80% 80%, rgba(27,94,67,0.2), transparent), radial-gradient(ellipse 40% 30% at 20% 60%, rgba(76,175,80,0.1), transparent); }
        `}</style>

        {/* Offline Banner */}
        {!isOnline && (
          <motion.div initial={{ y: -40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="fixed top-0 left-0 right-0 z-50 flex items-center justify-center gap-2 py-2.5 text-sm font-bold text-white backdrop-blur-xl" style={{ background: "linear-gradient(90deg, rgba(245,158,11,0.9), rgba(234,88,12,0.9))" }}>
            <Icon.WifiOff className="w-4 h-4" /> {t.offline}
          </motion.div>
        )}

        {/* Nav */}
        <nav className="fixed bottom-0 left-0 right-0 z-40 flex justify-around items-center py-3 px-4 md:top-0 md:bottom-auto md:px-8 md:py-4" style={{ background: "rgba(5,14,26,0.75)", backdropFilter: "blur(28px) saturate(180%)", borderTop: "1px solid rgba(6,182,212,0.12)", boxShadow: "0 -4px 40px rgba(0,0,0,0.35)" }}>
          {[
            { key: "home", label: t.nav.home, icon: Icon.Home },
            { key: "scan", label: t.nav.scan, icon: Icon.Scan },
            { key: "results", label: t.nav.results, icon: Icon.Leaf },
            { key: "history", label: t.nav.history, icon: Icon.History },
          ].map(({ key, label, icon: Ic }) => (
            <motion.button key={key} onClick={() => setPage(key)} whileTap={{ scale: 0.88 }} whileHover={{ y: -2 }}
              className={`flex flex-col items-center gap-1 px-3 py-1 rounded-xl transition-all duration-200 ${page === key ? "text-cyan-400" : "text-gray-500 hover:text-gray-200"}`}>
              <Ic className={`w-5 h-5 ${page === key ? "drop-shadow-[0_0_10px_rgba(6,182,212,0.9)]" : ""}`} />
              <span className="text-xs font-bold tracking-wide">{label}</span>
              {page === key && <motion.div layoutId="navActive" className="h-0.5 w-5 rounded-full" style={{ background: "linear-gradient(90deg,#06b6d4,#34d399)" }} />}
            </motion.button>
          ))}
          {/* Language Toggle */}
          <div className="flex flex-col items-center gap-1">
            <div className="flex rounded-lg overflow-hidden" style={{ border: "1px solid rgba(6,182,212,0.25)", background: "rgba(5,14,26,0.6)" }}>
              {["en", "te", "hi"].map(l => (
                <motion.button key={l} onClick={() => setLang(l)} whileTap={{ scale: 0.9 }}
                  className={`px-2 py-1 text-xs font-bold transition-all duration-200 ${lang === l ? "text-white" : "text-gray-500 hover:text-gray-300"}`}
                  style={lang === l ? { background: "linear-gradient(135deg,#0891b2,#059669)" } : {}}>
                  {l.toUpperCase()}
                </motion.button>
              ))}
            </div>
          </div>
        </nav>

        {/* Pages */}
        <div className={`${!isOnline ? "pt-8" : ""} pb-24 md:pt-20 md:pb-8`}>
          <AnimatePresence mode="wait">
            {page === "home" && <HomePage key="home" />}
            {page === "scan" && <ScanPage key="scan" />}
            {page === "results" && <ResultsPage key="results" />}
            {page === "history" && <HistoryPage key="history" />}
          </AnimatePresence>
        </div>
        {/* AI Chatbot */}
        <ChatBot onNavigate={setPage} />
      </div>
    </AppContext.Provider>
  );
}

// ============================================================
// PAGE TRANSITION WRAPPER
// ============================================================
function PageWrap({ children }) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.35 }}>
      {children}
    </motion.div>
  );
}

// ============================================================
// HOME PAGE
// ============================================================
function HomePage() {
  const { t, setPage } = useContext(AppContext);
  const particles = Array.from({ length: 20 }, (_, i) => i);

  return (
    <PageWrap>
      <div className="relative flex flex-col overflow-hidden aurora" style={{ minHeight: "100dvh", height: "100dvh" }}>
        {/* Animated BG particles */}
        {particles.map(i => (
          <motion.div key={i} className="absolute rounded-full pointer-events-none"
            style={{
              width: Math.random() * 80 + 8,
              height: Math.random() * 80 + 8,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: [`radial-gradient(circle,rgba(6,182,212,0.4),transparent)`, `radial-gradient(circle,rgba(52,211,153,0.3),transparent)`, `radial-gradient(circle,rgba(167,139,250,0.3),transparent)`][i % 3]
            }}
            animate={{ y: [0, -(Math.random() * 40 + 20), 0], scale: [1, 1.3, 1], opacity: [0.04, 0.18, 0.04] }}
            transition={{ duration: Math.random() * 6 + 4, repeat: Infinity, delay: Math.random() * 4 }} />
        ))}

        {/* ===== FULL-SCREEN TWO-COLUMN HERO ===== */}
        <div className="relative z-10 w-full flex-1 grid grid-cols-1 md:grid-cols-2" style={{ minHeight: 0, height: "100%" }}>

          {/* LEFT — text + buttons */}
          <motion.div className="flex flex-col justify-center px-6 md:px-12 lg:px-16 py-12 md:py-20"
            initial={{ opacity: 0, x: -40 }} animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}>

            {/* Radial glow behind logo */}
            <div className="absolute" style={{ width: 320, height: 320, top: "10%", left: "5%", background: "radial-gradient(circle,rgba(6,182,212,0.12),transparent 70%)", pointerEvents: "none", borderRadius: "50%" }} />

            {/* Icon badge */}
            <motion.div className="flex mb-6"
              animate={{ rotate: [0, 4, -4, 0], y: [0, -6, 0] }}
              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}>
              <div className="relative">
                <div className="absolute inset-0 rounded-3xl blur-xl opacity-60" style={{ background: "linear-gradient(135deg,#06b6d4,#10b981)" }} />
                <div className="relative w-20 h-20 rounded-3xl flex items-center justify-center" style={{ background: "linear-gradient(135deg,rgba(6,182,212,0.2),rgba(16,185,129,0.15))", border: "1px solid rgba(6,182,212,0.35)", backdropFilter: "blur(20px)" }}>
                  <Icon.Leaf className="w-10 h-10" style={{ color: "#34d399" }} />
                </div>
              </div>
            </motion.div>

            <motion.h1 className="font-black text-shadow mb-3 leading-tight"
              style={{ fontFamily: "'Playfair Display', serif", fontSize: "clamp(2.8rem, 5.5vw, 5rem)", lineHeight: 1.08 }}
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }}>
              {translations.en.appName === "KrishiAI"
                ? <><span className="shimmer-text">Krishi</span><span className="text-white">AI</span></>
                : t.appName}
            </motion.h1>

            <motion.p className="font-bold mb-3"
              style={{ color: "#67e8f9", fontSize: "clamp(1.05rem, 2.2vw, 1.3rem)", lineHeight: 1.5 }}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.22 }}>
              {t.tagline}
            </motion.p>

            <motion.p className="mb-8 leading-relaxed"
              style={{ color: "rgba(200,215,225,0.85)", fontSize: "clamp(0.95rem, 1.8vw, 1.1rem)", lineHeight: 1.8 }}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.32 }}>
              {t.heroSub}
            </motion.p>

            <motion.div className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.42 }}>
              <div className="flex flex-col gap-2">
                <motion.button onClick={() => setPage("scan")} whileHover={{ scale: 1.07 }} whileTap={{ scale: 0.93 }}
                  className="btn-glow rounded-2xl text-white font-black flex items-center gap-3 justify-center"
                  style={{ background: "linear-gradient(135deg,#22c55e,#16a34a,#15803d)", boxShadow: "0 6px 32px rgba(34,197,94,0.5)", minHeight: 60, padding: "0 36px", fontSize: "1.15rem" }}>
                  <Icon.Camera className="w-7 h-7" /> {t.scanCTA}
                </motion.button>
                <p style={{ color: "rgba(187,247,208,0.75)", fontSize: "0.8rem", textAlign: "center", fontStyle: "italic" }}>
                  📷 Take photo of leaf for instant help
                </p>
              </div>
              <motion.button onClick={() => setPage("history")} whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.93 }}
                className="btn-glow rounded-2xl font-bold flex items-center gap-3 justify-center"
                style={{ background: "rgba(255,255,255,0.08)", border: "1.5px solid rgba(134,239,172,0.35)", backdropFilter: "blur(16px)", color: "#86efac", minHeight: 60, padding: "0 28px", fontSize: "1rem" }}>
                <Icon.History className="w-6 h-6" /> {t.viewHistory}
              </motion.button>
            </motion.div>

            {/* Trust bar */}
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.65 }}
              className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-1"
              style={{ fontSize: "0.78rem", color: "rgba(187,247,208,0.7)", fontWeight: 600 }}>
              <span>✅ Trusted by Farmers</span>
              <span style={{ opacity: 0.4 }}>•</span>
              <span>🏛️ सरकारी सहायता समर्थित</span>
              <span style={{ opacity: 0.4 }}>•</span>
              <span>📶 Offline Ready</span>
            </motion.div>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}
              className="mt-4 inline-flex items-center gap-2 rounded-full px-4 py-2"
              style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(134,239,172,0.25)" }}>
              <motion.div className="w-2.5 h-2.5 rounded-full" style={{ background: "#4ade80" }}
                animate={{ opacity: [1, 0.3, 1], scale: [1, 1.4, 1] }} transition={{ duration: 1.8, repeat: Infinity }} />
              <span className="font-semibold" style={{ fontSize: "0.88rem", color: "#86efac" }}>{t.demo}</span>
            </motion.div>
          </motion.div>

          {/* RIGHT — farmer image, contained size */}
          <motion.div className="relative hidden md:flex items-center justify-center px-6 py-10"
            initial={{ opacity: 0, x: 60 }} animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.9, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}>
            <motion.div
              whileHover={{ scale: 1.025, y: -4 }}
              transition={{ type: "spring", stiffness: 140, damping: 22 }}
              className="relative overflow-hidden rounded-3xl w-full"
              style={{
                maxHeight: 420,
                maxWidth: "88%",
                border: "1.5px solid rgba(134,239,172,0.28)",
                boxShadow: "0 8px 60px rgba(0,0,0,0.55), 0 0 40px rgba(46,125,50,0.18)",
              }}>
              <img
                src="/farmer.png"
                alt="Indian farmer plowing field at sunset"
                className="w-full object-cover block"
                style={{ height: 420, filter: "saturate(0.88) brightness(0.92)" }}
              />
              {/* Overlays */}
              <div className="absolute inset-0" style={{ background: "rgba(10,30,15,0.18)", mixBlendMode: "multiply" }} />
              <div className="absolute inset-x-0 bottom-0 h-1/3" style={{ background: "linear-gradient(to top,rgba(15,61,46,0.7),transparent)" }} />
              {/* Caption badge */}
              <div className="absolute bottom-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full"
                style={{ background: "rgba(10,30,15,0.85)", backdropFilter: "blur(14px)", border: "1px solid rgba(134,239,172,0.3)" }}>
                <Icon.Leaf style={{ width: 13, height: 13, color: "#4ade80" }} />
                <span style={{ color: "#bbf7d0", fontSize: 11, fontWeight: 700 }}>AI-Powered Protection</span>
              </div>
            </motion.div>
          </motion.div>
          {/* Mobile: image below text */}
          <div className="block md:hidden w-full overflow-hidden rounded-2xl" style={{ minHeight: 240 }}>
            <img src="/farmer.png" alt="Farmer" className="w-full object-cover" style={{ height: 240, filter: "saturate(0.88) brightness(0.92)" }} />
          </div>
        </div>

        {/* Feature Cards — full-width below hero */}
        <div className="relative z-10 w-full grid grid-cols-2 md:grid-cols-4 gap-4 px-6 md:px-12 pb-8 md:pb-10">
          {[
            { icon: Icon.Scan, label: "AI Detection", color: "#06b6d4", glow: "rgba(6,182,212,0.3)" },
            { icon: Icon.Droplets, label: "Treatment Plan", color: "#34d399", glow: "rgba(52,211,153,0.3)" },
            { icon: Icon.IndianRupee, label: "Cost Estimate", color: "#fbbf24", glow: "rgba(251,191,36,0.3)" },
            { icon: Icon.MapPin, label: "Shop Finder", color: "#c084fc", glow: "rgba(192,132,252,0.3)" },
          ].map(({ icon: Ic, label, color, glow }, i) => (
            <motion.div key={label}
              initial={{ opacity: 0, y: 32, scale: 0.92 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: i * 0.08 + 0.55, ease: [0.22, 1, 0.36, 1] }}
              whileHover={{ y: -6, scale: 1.04 }}
              className="card-hover rounded-2xl p-4 text-center cursor-pointer"
              style={{ background: "rgba(255,255,255,0.04)", backdropFilter: "blur(20px)", border: `1px solid ${color}22` }}
              onClick={() => setPage("scan")}>
              <div className="w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center" style={{ background: glow, boxShadow: `0 0 20px ${glow}` }}>
                <Ic style={{ width: 24, height: 24, color }} />
              </div>
              <p className="text-white font-bold" style={{ fontSize: "0.95rem" }}>{label}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </PageWrap>
  );
}

// ============================================================
// SCAN PAGE
// ============================================================
function ScanPage() {
  const { t, lang, setPage, setScanResult, addHistory } = useContext(AppContext);
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [symptomText, setSymptomText] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  // ── Webcam state ─────────────────────────────────────────
  const [showCamera, setShowCamera] = useState(false);
  const [camError, setCamError] = useState("");
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const { isListening, startListening, speak } = useSpeech(lang);

const handleVoice = () => {
  startListening((transcript) => {
    setSymptomText(transcript);
    speak(`Got it. You said: ${transcript}`);
  });
};
  const fileRef = useRef();

  // ── Start webcam ─────────────────────────────────────────
  const startCamera = async () => {
    setCamError("");
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      // Wait for the video element to mount, then attach stream
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      }, 100);
    } catch (err) {
      console.error("Camera error:", err);
      if (err.name === "NotAllowedError") {
        setCamError("Camera permission denied. Please allow camera access in your browser settings.");
      } else if (err.name === "NotFoundError") {
        setCamError("No camera found on this device.");
      } else {
        setCamError(`Camera error: ${err.message}`);
      }
    }
  };

  // ── Stop webcam ──────────────────────────────────────────
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setShowCamera(false);
    setCamError("");
  };

  // ── Capture frame from video ─────────────────────────────
  const capturePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    setImage(dataUrl);

    // Convert dataURL → File for backend upload
    canvas.toBlob((blob) => {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      setImageFile(file);
    }, "image/jpeg", 0.92);

    stopCamera();
  };

  // ── File upload handler ──────────────────────────────────
  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImage(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // ── Analyze ──────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!image && !symptomText) return;
    setLoading(true);
    try {
      const formData = new FormData();
      if (imageFile) {
        formData.append("file", imageFile);
      }

      const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${BASE_URL}/api/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();

      // Map backend response to the shape your ResultsPage expects
      const result = {
        diseaseName: data.data.condition,
        severity: data.data.severity_risk.toLowerCase().includes("critical") ? "high"
                : data.data.severity_risk.toLowerCase().includes("medium")   ? "medium"
                : "low",
        healthScore: data.data.is_healthy ? 92 : Math.round((1 - data.data.confidence) * 100),
        treatment: {
          fertilizer: data.data.fertilizer_note || "Follow soil test recommendations",
          pesticide:  data.data.pesticides?.[0]?.name || data.data.organic?.[0] || "See treatment plan",
          dosage:     data.data.pesticides?.[0]?.dosage || "As per label",
          instructions: data.data.prevention?.join(". ") || "Follow standard crop care practices",
        },
        costPerAcre: 1850,
        weatherRisk: data.data.severity_risk === "Critical" || data.data.severity_risk === "High",
        shops: DEMO_RESULT.shops,  // keep demo shops for now
      };

      setScanResult(result);
      addHistory(result, image);
      setPage("results");

    } catch (err) {
      console.error(err);
      // Fallback to demo if backend is offline
      setScanResult(DEMO_RESULT);
      addHistory(DEMO_RESULT, image);
      setPage("results");
    } finally {
      setLoading(false);
    }
};

  // ── Render ───────────────────────────────────────────────
  return (
    <PageWrap>
      <div className="max-w-2xl mx-auto px-4 py-8">
        <motion.h2
          className="text-3xl font-black text-white mb-2"
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
        >
          {t.uploadLeaf}
        </motion.h2>
        <p className="text-gray-400 mb-8">{t.orDrag}</p>

        {/* ── Webcam Modal ───────────────────────────────── */}
        {showCamera && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-0 z-50 flex flex-col items-center justify-center"
            style={{ background: "rgba(0,0,0,0.92)", backdropFilter: "blur(12px)" }}
          >
            <div
              className="relative w-full max-w-lg rounded-3xl overflow-hidden"
              style={{ border: "1.5px solid rgba(52,211,153,0.3)", boxShadow: "0 24px 80px rgba(0,0,0,0.7)" }}
            >
              {/* Header */}
              <div
                className="flex items-center justify-between px-5 py-4"
                style={{ background: "rgba(15,45,26,0.95)", borderBottom: "1px solid rgba(52,211,153,0.15)" }}
              >
                <span className="text-white font-black text-lg flex items-center gap-2">
                  <Icon.Camera className="w-5 h-5 text-emerald-400" /> Live Camera
                </span>
                <button
                  onClick={stopCamera}
                  className="text-gray-400 hover:text-white transition-colors text-2xl leading-none"
                >
                  ✕
                </button>
              </div>

              {/* Video feed or error */}
              <div
                className="relative flex items-center justify-center"
                style={{ background: "#000", minHeight: 320 }}
              >
                {camError ? (
                  <div className="text-center p-8">
                    <p className="text-red-400 font-semibold mb-2">⚠️ {camError}</p>
                    <p className="text-gray-400 text-sm">
                      Try uploading an image instead using the file picker below.
                    </p>
                  </div>
                ) : (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full rounded-b-none"
                      style={{ maxHeight: 400, objectFit: "cover" }}
                    />
                    {/* Green crosshair overlay */}
                    <div
                      className="absolute inset-0 pointer-events-none flex items-center justify-center"
                    >
                      <div
                        style={{
                          width: 200, height: 200,
                          border: "2px solid rgba(52,211,153,0.7)",
                          borderRadius: 16,
                          boxShadow: "0 0 0 9999px rgba(0,0,0,0.35)",
                        }}
                      />
                    </div>
                    <p
                      className="absolute bottom-3 left-0 right-0 text-center text-xs font-semibold"
                      style={{ color: "rgba(187,247,208,0.8)" }}
                    >
                      Position leaf inside the frame
                    </p>
                  </>
                )}
              </div>

              {/* Capture button */}
              {!camError && (
                <div
                  className="flex gap-3 px-5 py-4"
                  style={{ background: "rgba(15,45,26,0.95)" }}
                >
                  <button
                    onClick={stopCamera}
                    className="flex-1 rounded-2xl py-3 font-bold text-gray-300 hover:text-white transition-all"
                    style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.12)" }}
                  >
                    Cancel
                  </button>
                  <motion.button
                    onClick={capturePhoto}
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.96 }}
                    className="flex-1 rounded-2xl py-3 font-black text-white flex items-center justify-center gap-2"
                    style={{ background: "linear-gradient(135deg,#22c55e,#16a34a)", boxShadow: "0 4px 20px rgba(34,197,94,0.45)" }}
                  >
                    <Icon.Camera className="w-5 h-5" /> Capture
                  </motion.button>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* ── Upload Zone ────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`relative rounded-3xl border-2 border-dashed transition-all duration-300 ${
            dragging ? "border-emerald-400 bg-emerald-500/10" : "border-white/20 glass"
          } min-h-[220px] flex flex-col items-center justify-center gap-4 p-6 cursor-pointer overflow-hidden`}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {image ? (
            <motion.img
              src={image}
              alt="leaf"
              className="max-h-48 rounded-2xl object-contain shadow-xl"
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
            />
          ) : (
            <>
              <div className="w-16 h-16 rounded-2xl glass-green flex items-center justify-center">
                <Icon.Upload className="w-8 h-8 text-emerald-400" />
              </div>
              <p className="text-white font-semibold text-center">{t.uploadLeaf}</p>
              <p className="text-gray-400 text-sm text-center">{t.orDrag}</p>
            </>
          )}
        </motion.div>

        {/* ── Camera Button ──────────────────────────────── */}
        <motion.div
          className="mt-4 flex gap-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <button
            className="flex-1 glass rounded-2xl py-3 text-white font-semibold flex items-center justify-center gap-2 hover:bg-white/10 transition-all"
            onClick={startCamera}   // ← now calls getUserMedia instead of file input
          >
            <Icon.Camera className="w-5 h-5 text-emerald-400" /> {t.capturePhoto}
          </button>
        </motion.div>

        {/* ── Voice Input ────────────────────────────────── */}
        <motion.div className="mt-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <label className="text-gray-300 text-sm font-semibold mb-2 block">{t.voiceInput}</label>
          <div className="flex gap-3">
            <input
              value={symptomText}
              onChange={(e) => setSymptomText(e.target.value)}
              placeholder={isListening ? t.listening : "Describe symptoms..."}
              className="flex-1 glass rounded-xl px-4 py-3 text-white placeholder-gray-400 outline-none focus:border-emerald-400 border border-transparent transition-all"
            />
            <motion.button
              onClick={handleVoice}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
                isListening ? "bg-red-500 animate-pulse" : "bg-emerald-500 hover:bg-emerald-400"
              }`}
            >
              <Icon.Mic className="w-5 h-5 text-white" />
            </motion.button>
          </div>
          {isListening && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-emerald-400 text-sm mt-2 flex items-center gap-2"
            >
              <span className="w-2 h-2 bg-red-400 rounded-full animate-pulse inline-block" />
              {t.listening}
            </motion.p>
          )}
        </motion.div>

        {/* ── Analyze Button ─────────────────────────────── */}
        <motion.button
          onClick={handleAnalyze}
          disabled={loading || (!image && !symptomText)}
          whileHover={!loading ? { scale: 1.02 } : {}}
          whileTap={!loading ? { scale: 0.98 } : {}}
          className="mt-8 w-full py-4 rounded-2xl bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold text-lg flex items-center justify-center gap-3 btn-glow"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          {loading ? (
            <>
              <motion.div
                className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              {t.loading}
            </>
          ) : (
            <>
              <Icon.Scan className="w-6 h-6" /> {t.analyzeBtn}
            </>
          )}
        </motion.button>
      </div>
    </PageWrap>
  );
}

// ============================================================
// RESULTS PAGE
// ============================================================
function ResultsPage() {
  const { t, scanResult, setPage } = useContext(AppContext);
  const { speak } = useSpeech(useContext(AppContext).lang);

  if (!scanResult) {
    return (
      <PageWrap>
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <Icon.Leaf className="w-16 h-16 text-emerald-400 opacity-50" />
          <p className="text-gray-400 text-lg">{t.noHistory}</p>
          <button onClick={() => setPage("scan")} className="px-6 py-3 bg-emerald-500 text-white rounded-2xl font-bold btn-glow">{t.scanCTA}</button>
        </div>
      </PageWrap>
    );
  }

  const r = scanResult;
  const severityColor = { low: "text-green-400 bg-green-400/10", medium: "text-yellow-400 bg-yellow-400/10", high: "text-red-400 bg-red-400/10" }[r.severity] || "text-gray-400";
  const healthColor = r.healthScore > 70 ? "#4ade80" : r.healthScore > 40 ? "#facc15" : "#f87171";

  const speakResult = () => {
    speak(`Disease detected: ${r.diseaseName}. Severity: ${r.severity}. Health score: ${r.healthScore} out of 100. Use ${r.treatment.pesticide} with dosage ${r.treatment.dosage}.`);
  };

  const cards = [
    { key: "disease", delay: 0 },
    { key: "treatment", delay: 0.1 },
    { key: "cost", delay: 0.2 },
    { key: "weather", delay: 0.3 },
    { key: "shops", delay: 0.4 },
  ];

  return (
    <PageWrap>
      <div className="max-w-2xl mx-auto px-4 py-8 space-y-5">
        <div className="flex items-center justify-between mb-2">
          <motion.h2 className="text-3xl font-black text-white" initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }}>
            {t.diseaseDetected}
          </motion.h2>
          <motion.button onClick={speakResult} whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
            className="glass w-10 h-10 rounded-xl flex items-center justify-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Icon.Volume2 className="w-5 h-5 text-emerald-400" />
          </motion.button>
        </div>

        {/* ===== DISEASE HERO BANNER ===== */}
        <motion.div
          className="rounded-3xl overflow-hidden"
          initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0, ease: [0.22, 1, 0.36, 1] }}
          style={{
            background: r.severity === "high"
              ? "linear-gradient(135deg,rgba(239,68,68,0.18),rgba(220,38,38,0.08))"
              : r.severity === "medium"
                ? "linear-gradient(135deg,rgba(234,179,8,0.18),rgba(202,138,4,0.08))"
                : "linear-gradient(135deg,rgba(34,197,94,0.18),rgba(22,163,74,0.08))",
            border: `1px solid ${r.severity === "high" ? "rgba(239,68,68,0.4)" : r.severity === "medium" ? "rgba(234,179,8,0.4)" : "rgba(34,197,94,0.4)"}`,
            boxShadow: "0 12px 60px rgba(0,0,0,0.35)",
          }}>
          {/* Top strip */}
          <div className="px-6 pt-5 pb-4 flex items-center justify-between"
            style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div>
              <p className="text-xs font-bold uppercase tracking-widest mb-1"
                style={{ color: r.severity === "high" ? "#fca5a5" : r.severity === "medium" ? "#fde047" : "#86efac" }}>
                {t.diseaseDetected}
              </p>
              <h3 className="text-white font-black leading-tight" style={{ fontSize: "clamp(1.4rem,4vw,1.9rem)" }}>
                {r.diseaseName}
              </h3>
            </div>
            <span className={`px-4 py-2 rounded-full text-sm font-black ${severityColor}`}
              style={{ fontSize: "0.95rem" }}>
              {t[r.severity]}
            </span>
          </div>

          {/* Health Score + quick stats */}
          <div className="px-6 py-5 flex items-center gap-6">
            {/* Arc */}
            <div className="relative flex-shrink-0" style={{ width: 108, height: 108 }}>
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="9" />
                <motion.circle cx="50" cy="50" r="40" fill="none" stroke={healthColor} strokeWidth="9"
                  strokeLinecap="round" strokeDasharray={`${2 * Math.PI * 40}`}
                  initial={{ strokeDashoffset: 2 * Math.PI * 40 }}
                  animate={{ strokeDashoffset: 2 * Math.PI * 40 * (1 - r.healthScore / 100) }}
                  transition={{ duration: 1.5, ease: "easeOut" }} />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-white font-black" style={{ fontSize: "1.7rem", lineHeight: 1 }}>{r.healthScore}</span>
                <span className="text-gray-400" style={{ fontSize: "0.7rem" }}>/100</span>
              </div>
            </div>
            <div className="flex-1">
              <p className="text-gray-400 text-sm mb-1">{t.healthScore}</p>
              <p className="text-white font-black text-xl mb-1">
                {r.healthScore > 70 ? "🟢 " + t.healthy : r.healthScore > 40 ? "🟡 Moderate Risk" : "🔴 " + t.high + " Risk"}
              </p>
              <p style={{ color: "rgba(200,220,210,0.7)", fontSize: "0.82rem" }}>
                {t.severity}: <span className={`font-bold ${severityColor.split(" ")[0]}`}>{t[r.severity]}</span>
              </p>
            </div>
          </div>
        </motion.div>

        {/* ===== WHAT TO DO ===== */}
        <motion.div className="rounded-3xl p-6 space-y-4"
          initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
          style={{ background: "rgba(15,55,40,0.7)", backdropFilter: "blur(24px)", border: "1px solid rgba(52,211,153,0.3)", boxShadow: "0 8px 40px rgba(0,0,0,0.3)" }}>
          <h3 className="text-white font-black flex items-center gap-2" style={{ fontSize: "1.2rem" }}>
            <span style={{ fontSize: "1.4rem" }}>💡</span> What To Do
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { icon: "🌿", label: t.fertilizer, val: r.treatment.fertilizer, color: "rgba(34,197,94,0.15)", border: "rgba(34,197,94,0.3)" },
              { icon: "💚", label: t.pesticide, val: r.treatment.pesticide, color: "rgba(59,130,246,0.15)", border: "rgba(59,130,246,0.3)" },
              { icon: "⚖️", label: t.dosage, val: r.treatment.dosage, color: "rgba(234,179,8,0.15)", border: "rgba(234,179,8,0.3)" },
            ].map(({ icon, label, val, color, border }, i) => (
              <motion.div key={label}
                className="rounded-2xl p-4 text-center"
                style={{ background: color, border: `1px solid ${border}` }}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.12 + i * 0.08 }}>
                <div style={{ fontSize: "1.6rem", lineHeight: 1, marginBottom: 6 }}>{icon}</div>
                <p className="text-gray-300 text-xs font-semibold mb-1 uppercase tracking-wide">{label}</p>
                <p className="text-white font-black text-sm">{val}</p>
              </motion.div>
            ))}
          </div>
          {/* Instructions as bullet points */}
          <div className="rounded-2xl p-4" style={{ background: "rgba(0,0,0,0.2)" }}>
            <p className="text-emerald-300 text-xs font-bold uppercase tracking-wide mb-3">📝 {t.instructions}</p>
            <ul className="space-y-2">
              {(r.treatment.instructions || "").split(".").filter(s => s.trim()).map((step, i) => (
                <li key={i} className="flex items-start gap-2" style={{ color: "rgba(210,240,225,0.9)", fontSize: "0.9rem", lineHeight: 1.5 }}>
                  <span className="text-emerald-400 font-black flex-shrink-0" style={{ fontSize: "0.8rem", marginTop: 2 }}>{i + 1}.</span>
                  <span>{step.trim()}.</span>
                </li>
              ))}
            </ul>
          </div>
        </motion.div>

        {/* ===== COST CARD ===== */}
        <motion.div className="rounded-3xl p-6"
          initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          style={{ background: "linear-gradient(135deg,rgba(251,191,36,0.12),rgba(245,158,11,0.06))", backdropFilter: "blur(24px)", border: "1px solid rgba(251,191,36,0.35)", boxShadow: "0 8px 40px rgba(0,0,0,0.3)" }}>
          <h3 className="text-white font-black flex items-center gap-2 mb-4" style={{ fontSize: "1.2rem" }}>
            <Icon.IndianRupee className="w-5 h-5 text-yellow-400" /> {t.costCalc}
          </h3>
          <div className="flex items-center gap-4 mb-4">
            <div>
              <p className="text-yellow-200 text-sm font-semibold mb-1">{t.perAcre}</p>
              <motion.p className="font-black text-yellow-400" style={{ fontSize: "clamp(2.2rem,6vw,3rem)", lineHeight: 1 }}
                initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.4, type: "spring" }}>
                ₹{r.costPerAcre.toLocaleString()}
              </motion.p>
            </div>
          </div>
          <div className="rounded-2xl p-4 mb-3" style={{ background: "rgba(0,0,0,0.25)" }}>
            <p className="text-yellow-300 text-xs font-bold mb-2">Quick Estimate</p>
            <div className="grid grid-cols-4 gap-2">
              {[1, 2, 5, 10].map(acres => (
                <div key={acres} className="rounded-xl p-2 text-center" style={{ background: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.2)" }}>
                  <p className="text-yellow-400 font-black text-sm">₹{(r.costPerAcre * acres).toLocaleString()}</p>
                  <p className="text-gray-400 text-xs">{acres} ac</p>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* ===== WEATHER ALERT ===== */}
        {r.weatherRisk && (
          <motion.div className="rounded-3xl p-6"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            style={{ background: "linear-gradient(135deg,rgba(251,146,60,0.2),rgba(239,68,68,0.1))", border: "2px solid rgba(251,146,60,0.5)", boxShadow: "0 0 40px rgba(251,146,60,0.15)" }}>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: "rgba(251,146,60,0.3)" }}>
                <Icon.AlertTriangle className="w-5 h-5 text-orange-300" />
              </div>
              <h3 className="text-orange-200 font-black" style={{ fontSize: "1.15rem" }}>{t.weatherAlert}</h3>
            </div>
            <p className="text-orange-100 mb-4" style={{ fontSize: "0.95rem", lineHeight: 1.6 }}>{t.weatherWarning}</p>
            <div className="grid grid-cols-3 gap-3">
              {[{ icon: "💧", label: "Humidity", val: "87%" }, { icon: "🌡️", label: "Temp", val: "31°C" }, { icon: "💨", label: "Wind", val: "12 km/h" }].map(({ icon, label, val }) => (
                <div key={label} className="rounded-xl px-3 py-3 text-center" style={{ background: "rgba(0,0,0,0.25)" }}>
                  <p style={{ fontSize: "1.2rem" }}>{icon}</p>
                  <p className="text-orange-300 font-black text-base">{val}</p>
                  <p className="text-gray-400 text-xs">{label}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* ===== AGRO SHOPS ===== */}
        <motion.div className="rounded-3xl p-6"
          initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
          style={{ background: "rgba(15,30,50,0.7)", backdropFilter: "blur(24px)", border: "1px solid rgba(192,132,252,0.25)", boxShadow: "0 8px 40px rgba(0,0,0,0.3)" }}>
          <h3 className="text-white font-black flex items-center gap-2 mb-4" style={{ fontSize: "1.2rem" }}>
            <span style={{ fontSize: "1.3rem" }}>🏪</span> {t.shopsNearby}
          </h3>
          <div className="space-y-3">
            {r.shops.map((shop, i) => (
              <motion.div key={shop.name}
                className="rounded-2xl p-4"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(192,132,252,0.15)" }}
                initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.45 + i * 0.08 }}>
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <p className="text-white font-black" style={{ fontSize: "1rem" }}>{shop.name}</p>
                    <p className="text-gray-400 text-sm flex items-center gap-1 mt-0.5">
                      <Icon.MapPin className="w-3.5 h-3.5" /> {shop.distance}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 px-2 py-1 rounded-lg" style={{ background: "rgba(251,191,36,0.15)" }}>
                    <Icon.Star className="w-3.5 h-3.5 text-yellow-400" />
                    <span className="text-yellow-400 font-bold text-sm">{shop.rating}</span>
                  </div>
                </div>
                {/* Action buttons */}
                <div className="flex gap-2">
                  <a href={`tel:${shop.phone}`}
                    className="flex-1 flex items-center justify-center gap-2 rounded-xl py-2.5 font-bold text-sm"
                    style={{ background: "linear-gradient(135deg,#16a34a,#15803d)", color: "#fff", boxShadow: "0 4px 16px rgba(22,163,74,0.4)" }}>
                    <Icon.Phone className="w-4 h-4" /> Call
                  </a>
                  <a href={`https://maps.google.com/?q=${encodeURIComponent(shop.name)}`} target="_blank" rel="noopener noreferrer"
                    className="flex-1 flex items-center justify-center gap-2 rounded-xl py-2.5 font-bold text-sm"
                    style={{ background: "rgba(59,130,246,0.18)", color: "#93c5fd", border: "1px solid rgba(59,130,246,0.35)" }}>
                    <Icon.MapPin className="w-4 h-4" /> Map
                  </a>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Scan Again */}
        <motion.button onClick={() => setPage("scan")} whileHover={{ scale: 1.03, y: -2 }} whileTap={{ scale: 0.97 }}
          className="w-full py-4 rounded-2xl font-bold text-lg flex items-center justify-center gap-2 transition-all btn-glow"
          style={{ background: "rgba(6,182,212,0.08)", border: "1px solid rgba(6,182,212,0.25)", color: "#67e8f9", backdropFilter: "blur(16px)" }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
          <Icon.Camera className="w-5 h-5 text-emerald-400" /> {t.scanCTA}
        </motion.button>
      </div>
    </PageWrap>
  );
}

// ============================================================
// HISTORY PAGE
// ============================================================
function HistoryPage() {
  const { t, history, setPage, setScanResult } = useContext(AppContext);

  return (
    <PageWrap>
      <div className="max-w-2xl mx-auto px-4 py-8">
        <motion.h2 className="text-3xl font-black text-white mb-2" initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }}>
          {t.history}
        </motion.h2>

        {history.length === 0 ? (
          <motion.div className="flex flex-col items-center justify-center min-h-[50vh] gap-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Icon.History className="w-16 h-16 text-emerald-400 opacity-30" />
            <p className="text-gray-400 text-center">{t.noHistory}</p>
            <button onClick={() => setPage("scan")} className="px-6 py-3 bg-emerald-500 text-white rounded-2xl font-bold btn-glow">{t.scanCTA}</button>
          </motion.div>
        ) : (
          <>
            {/* Mini Progress Graph */}
            <motion.div className="rounded-3xl p-5 mb-6" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, ease: [0.22, 1, 0.36, 1] }} style={{ background: "rgba(6,182,212,0.06)", backdropFilter: "blur(24px)", border: "1px solid rgba(6,182,212,0.15)", boxShadow: "0 8px 40px rgba(0,0,0,0.25)" }}>
              <h3 className="text-white font-bold mb-3 flex items-center gap-2"><Icon.TrendingUp className="w-4 h-4 text-emerald-400" /> Health Trend</h3>
              <div className="flex items-end gap-2 h-20">
                {history.slice(0, 10).reverse().map((h, i) => {
                  const score = h.result?.healthScore || 0;
                  const barH = (score / 100) * 100;
                  const color = score > 70 ? "#4ade80" : score > 40 ? "#facc15" : "#f87171";
                  return (
                    <motion.div key={h.id} className="flex-1 rounded-t-lg relative group" style={{ background: color, height: `${barH}%`, minHeight: "4px" }}
                      initial={{ height: 0 }} animate={{ height: `${barH}%` }} transition={{ delay: i * 0.05, duration: 0.5 }}>
                      <div className="absolute -top-7 left-1/2 -translate-x-1/2 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">{score}%</div>
                    </motion.div>
                  );
                })}
              </div>
              <p className="text-gray-400 text-xs mt-2">Last {Math.min(history.length, 10)} scans</p>
            </motion.div>

            {/* Scan Cards */}
            <div className="space-y-4">
              {history.map((entry, i) => {
                const score = entry.result?.healthScore || 0;
                const scoreColor = score > 70 ? "#4ade80" : score > 40 ? "#facc15" : "#f87171";
                const severity = entry.result?.severity || "low";
                const sevBg = { low: "rgba(34,197,94,0.15)", medium: "rgba(234,179,8,0.15)", high: "rgba(239,68,68,0.15)" }[severity];
                const sevText = { low: "text-green-400", medium: "text-yellow-400", high: "text-red-400" }[severity];
                return (
                  <motion.div key={entry.id}
                    className="rounded-2xl overflow-hidden cursor-pointer"
                    style={{ background: "rgba(255,255,255,0.04)", backdropFilter: "blur(20px)", border: "1px solid rgba(255,255,255,0.08)" }}
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.06, ease: [0.22, 1, 0.36, 1] }}
                    whileHover={{ y: -3, borderColor: "rgba(52,211,153,0.3)" }}
                    onClick={() => { setScanResult(entry.result); setPage("results"); }}>
                    <div className="flex items-stretch gap-0">
                      {/* Thumbnail */}
                      <div className="flex-shrink-0" style={{ width: 88, minHeight: 88, background: "rgba(15,50,30,0.6)" }}>
                        {entry.imageUrl ? (
                          <img src={entry.imageUrl} alt="" className="w-full h-full object-cover" style={{ minHeight: 88 }} />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center" style={{ minHeight: 88 }}>
                            <Icon.Leaf className="w-8 h-8 text-emerald-400 opacity-60" />
                          </div>
                        )}
                      </div>
                      {/* Content */}
                      <div className="flex-1 p-4">
                        <div className="flex items-start justify-between mb-1">
                          <p className="text-white font-black" style={{ fontSize: "1rem", lineHeight: 1.3 }}>
                            {entry.result?.diseaseName || "Unknown"}
                          </p>
                          <span className={`px-2 py-0.5 rounded-full text-xs font-bold ml-2 flex-shrink-0 ${sevText}`}
                            style={{ background: sevBg }}>
                            {severity}
                          </span>
                        </div>
                        <p className="text-gray-400 text-xs flex items-center gap-1 mb-3">
                          <Icon.Clock className="w-3 h-3" /> {new Date(entry.date).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ background: scoreColor }} />
                            <span className="font-black" style={{ color: scoreColor, fontSize: "1.1rem" }}>{score}</span>
                            <span className="text-gray-500 text-xs">/ 100</span>
                          </div>
                          <button
                            onClick={e => { e.stopPropagation(); }}
                            className="flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-bold"
                            style={{ background: "rgba(52,211,153,0.12)", border: "1px solid rgba(52,211,153,0.3)", color: "#6ee7b7" }}>
                            <Icon.Volume2 className="w-3 h-3" /> Replay Voice
                          </button>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </PageWrap>
  );
}
