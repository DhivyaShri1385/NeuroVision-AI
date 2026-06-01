import { useState, useEffect } from "react";
import { analyze, getHealth }  from "./api/neurovision";
import { DEMO_RESULT }         from "./demoData";
import StatusBar               from "./components/StatusBar";
import UploadPanel             from "./components/UploadPanel";
import AnalysisResult          from "./components/AnalysisResult";
import HeroSection             from "./components/HeroSection";
import {
  Brain, Loader2, AlertCircle,
  Sparkles, GitFork, FileText, Play,
} from "lucide-react";
import "./index.css";

export default function App() {
  const [file,       setFile]       = useState(null);
  const [method,     setMethod]     = useState("gradcam");
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);
  const [apiOnline,  setApiOnline]  = useState(false);
  const [demoMode,   setDemoMode]   = useState(false);

  // Poll API status
  useEffect(() => {
    getHealth().then(() => setApiOnline(true)).catch(() => setApiOnline(false));
  }, []);

  const run = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null); setDemoMode(false);
    try {
      const data = await analyze(file, method);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const loadDemo = () => {
    setDemoMode(true);
    setResult(DEMO_RESULT);
    setError(null);
  };

  return (
    <div style={S.app}>
      {/* ── Nav ───────────────────────────────────────────── */}
      <header style={S.nav}>
        <div style={S.navLeft}>
          <div style={S.logoIcon}><Brain size={20} color="#60a5fa"/></div>
          <div>
            <span style={S.logoText} className="glow-text">NeuroVision AI</span>
            <span style={S.logoSub}> · v1.0.0</span>
          </div>
        </div>
        <div style={S.navRight}>
          <a href="https://github.com/DhivyaShri1385/NeuroVision-AI" target="_blank"
             rel="noreferrer" style={S.navLink}>
            <GitFork size={14}/> GitHub
          </a>
          <a href={`${import.meta.env.VITE_API_URL||"http://localhost:8000"}/docs`}
             target="_blank" rel="noreferrer" style={S.navLink}>
            <FileText size={14}/> API Docs
          </a>
        </div>
      </header>

      <StatusBar demoMode={demoMode} />

      {/* ── Main ──────────────────────────────────────────── */}
      <main style={S.main}>

        {/* LEFT sidebar */}
        <aside style={S.sidebar}>

          {/* Upload */}
          <section style={S.section}>
            <div style={S.sectionLabel}>
              <span style={S.dot}/> MRI Image
            </div>
            <UploadPanel onFile={setFile} file={file} />
          </section>

          {/* Controls */}
          <section style={S.section}>
            <div style={S.sectionLabel}><span style={S.dot}/> XAI Method</div>
            <div style={S.methodPicker}>
              {["gradcam", "smoothgrad"].map(m => (
                <button key={m}
                  style={{ ...S.methodBtn, ...(method===m ? S.methodActive : {}) }}
                  onClick={() => setMethod(m)}>
                  <Sparkles size={12}/>
                  {m === "gradcam" ? "Grad-CAM" : "SmoothGrad"}
                  {m === "gradcam" && <span style={S.methodTag}>fast</span>}
                  {m === "smoothgrad" && <span style={S.methodTag}>precise</span>}
                </button>
              ))}
            </div>
          </section>

          {/* Analyse button */}
          <button style={S.analyseBtn(loading || !file)} onClick={run} disabled={loading || !file}>
            {loading
              ? <><Loader2 size={16} style={S.spin}/> Analysing…</>
              : <><Brain size={16}/> Analyse MRI</>}
          </button>

          {/* Demo button */}
          {!apiOnline && !result && (
            <button style={S.demoBtn} onClick={loadDemo}>
              <Play size={14}/> Load Demo Results
            </button>
          )}

          {/* Error */}
          {error && (
            <div style={S.errorBox}>
              <AlertCircle size={14} style={{flexShrink:0}}/>
              <span>{error}</span>
            </div>
          )}

          {/* Demo notice */}
          {demoMode && (
            <div style={S.demoNotice}>
              ⚡ Demo mode — mock results. Start the backend to analyse real images.
            </div>
          )}
        </aside>

        {/* RIGHT content */}
        <div style={S.content}>
          {result
            ? <AnalysisResult result={result} />
            : <HeroSection />}
        </div>
      </main>

      {/* Footer */}
      <footer style={S.footer}>
        <span>Built with ❤️ · NeuroVision AI Research Platform</span>
        <span>Phase 1–7 · TF 2.20 · EfficientNetB3 · Attention U-Net</span>
      </footer>

      {/* Spin keyframe */}
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

const S = {
  app: { minHeight:"100vh", display:"flex", flexDirection:"column" },

  /* Nav */
  nav: {
    display:"flex", alignItems:"center", justifyContent:"space-between",
    padding:"0.75rem 1.5rem",
    background:"rgba(10,15,30,0.85)",
    borderBottom:"1px solid var(--border)",
    backdropFilter:"blur(20px)",
    position:"sticky", top:0, zIndex:100,
  },
  navLeft: { display:"flex", alignItems:"center", gap:"0.6rem" },
  logoIcon: {
    width:36, height:36, borderRadius:10,
    background:"linear-gradient(135deg,rgba(59,130,246,0.25),rgba(139,92,246,0.25))",
    border:"1px solid rgba(96,165,250,0.3)",
    display:"flex", alignItems:"center", justifyContent:"center",
  },
  logoText: { fontSize:17, fontWeight:800, letterSpacing:"-0.03em" },
  logoSub:  { fontSize:11, color:"var(--text-dim)" },
  navRight: { display:"flex", alignItems:"center", gap:"0.5rem" },
  navLink: {
    display:"flex", alignItems:"center", gap:"0.35rem",
    padding:"0.3rem 0.7rem", borderRadius:"var(--r-sm)",
    background:"var(--glass)", border:"1px solid var(--border)",
    color:"var(--text-dim)", fontSize:12, fontWeight:500,
    transition:"all 0.15s",
    textDecoration:"none",
  },

  /* Main layout */
  main: {
    flex:1, maxWidth:1100, width:"100%", margin:"0 auto",
    padding:"1.5rem", display:"grid",
    gridTemplateColumns:"320px 1fr", gap:"1.25rem", alignItems:"start",
  },
  sidebar: { display:"flex", flexDirection:"column", gap:"0.9rem", position:"sticky", top:80 },
  content: { minWidth:0 },

  /* Section */
  section: {
    background:"var(--glass)", border:"1px solid var(--border)",
    borderRadius:"var(--r-lg)", padding:"1rem",
    backdropFilter:"blur(16px)",
  },
  sectionLabel: {
    display:"flex", alignItems:"center", gap:"0.4rem",
    fontSize:11, fontWeight:600, color:"var(--text-dim)",
    textTransform:"uppercase", letterSpacing:"0.08em",
    marginBottom:"0.75rem",
  },
  dot: {
    width:6, height:6, borderRadius:"50%",
    background:"var(--accent)", display:"inline-block",
    boxShadow:"0 0 8px var(--accent)",
  },

  /* Method picker */
  methodPicker: { display:"flex", flexDirection:"column", gap:"0.4rem" },
  methodBtn: {
    display:"flex", alignItems:"center", gap:"0.4rem",
    padding:"0.5rem 0.75rem", borderRadius:"var(--r-md)",
    border:"1px solid var(--border)", background:"transparent",
    color:"var(--text-dim)", cursor:"pointer", fontSize:13, fontWeight:500,
    transition:"all 0.2s",
  },
  methodActive: {
    background:"linear-gradient(135deg,rgba(59,130,246,0.15),rgba(139,92,246,0.15))",
    borderColor:"rgba(96,165,250,0.35)", color:"var(--text-bright)",
    boxShadow:"0 0 16px rgba(96,165,250,0.08)",
  },
  methodTag: {
    marginLeft:"auto", fontSize:9, padding:"1px 6px", borderRadius:99,
    background:"rgba(255,255,255,0.08)", color:"var(--text-dim)",
    fontWeight:600, letterSpacing:"0.05em",
  },

  /* Analyse button */
  analyseBtn: (disabled) => ({
    width:"100%", padding:"0.75rem",
    background: disabled ? "rgba(255,255,255,0.04)" : "linear-gradient(135deg,#3b82f6,#8b5cf6)",
    color: disabled ? "var(--text-dim)" : "#fff",
    border: disabled ? "1px solid var(--border)" : "none",
    borderRadius:"var(--r-md)", fontWeight:700, fontSize:14,
    cursor: disabled ? "not-allowed" : "pointer",
    display:"flex", alignItems:"center", justifyContent:"center", gap:"0.5rem",
    transition:"all 0.2s",
    boxShadow: disabled ? "none" : "0 4px 24px rgba(96,165,250,0.25)",
    letterSpacing:"0.02em",
  }),
  spin: { animation:"spin 0.9s linear infinite" },

  /* Demo button */
  demoBtn: {
    width:"100%", padding:"0.55rem",
    background:"rgba(251,191,36,0.08)",
    border:"1px dashed rgba(251,191,36,0.3)",
    borderRadius:"var(--r-md)", color:"var(--yellow)",
    cursor:"pointer", fontSize:13, fontWeight:600,
    display:"flex", alignItems:"center", justifyContent:"center", gap:"0.4rem",
    transition:"all 0.2s",
  },

  /* Error */
  errorBox: {
    display:"flex", alignItems:"flex-start", gap:"0.5rem",
    padding:"0.75rem", borderRadius:"var(--r-md)",
    background:"rgba(248,113,113,0.08)",
    border:"1px solid rgba(248,113,113,0.25)",
    color:"var(--red)", fontSize:12,
  },

  /* Demo notice */
  demoNotice: {
    padding:"0.6rem 0.8rem", borderRadius:"var(--r-md)",
    background:"rgba(251,191,36,0.07)",
    border:"1px solid rgba(251,191,36,0.2)",
    color:"var(--yellow)", fontSize:11, lineHeight:1.5,
  },

  /* Footer */
  footer: {
    display:"flex", justifyContent:"space-between", flexWrap:"wrap",
    padding:"0.75rem 1.5rem",
    borderTop:"1px solid var(--border)",
    fontSize:11, color:"var(--text-dim)",
    background:"rgba(10,15,30,0.5)",
  },
};
