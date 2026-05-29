import { useState } from "react";
import { analyze }         from "./api/neurovision";
import StatusBar           from "./components/StatusBar";
import UploadPanel         from "./components/UploadPanel";
import AnalysisResult      from "./components/AnalysisResult";
import { Brain, Loader2, AlertCircle } from "lucide-react";
import "./index.css";

/* ─── layout styles ─────────────────────────────────────────────── */
const S = {
  app: { minHeight: "100vh", display: "flex", flexDirection: "column" },

  header: {
    display: "flex", alignItems: "center", gap: "0.75rem",
    padding: "1rem 1.5rem",
    borderBottom: "1px solid var(--border)",
    background: "var(--surface)",
  },
  logo: { color: "var(--accent)" },
  title: { fontSize: 18, fontWeight: 700, color: "var(--text)" },
  subtitle: { fontSize: 12, color: "var(--muted)", marginTop: 1 },

  main: {
    flex: 1, maxWidth: 960, width: "100%",
    margin: "0 auto", padding: "1.5rem",
    display: "grid", gap: "1.25rem",
    gridTemplateColumns: "340px 1fr",
    alignItems: "start",
  },

  leftCol: { display: "flex", flexDirection: "column", gap: "1rem" },

  card: {
    background: "var(--surface)", border: "1px solid var(--border)",
    borderRadius: "var(--radius)", padding: "1rem",
  },
  cardTitle: { fontWeight: 600, marginBottom: "0.75rem", fontSize: 13, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em" },

  select: {
    width: "100%", padding: "0.4rem 0.6rem",
    background: "var(--bg)", border: "1px solid var(--border)",
    borderRadius: "var(--radius)", color: "var(--text)", fontSize: 13,
  },

  btn: (loading) => ({
    width: "100%", padding: "0.6rem",
    background: loading ? "var(--border)" : "var(--accent)",
    color: loading ? "var(--muted)" : "#0d1117",
    border: "none", borderRadius: "var(--radius)",
    fontWeight: 700, fontSize: 14, cursor: loading ? "not-allowed" : "pointer",
    display: "flex", alignItems: "center", justifyContent: "center", gap: "0.4rem",
    transition: "background 0.2s",
  }),

  error: {
    display: "flex", alignItems: "flex-start", gap: "0.5rem",
    background: "rgba(248,81,73,0.1)", border: "1px solid rgba(248,81,73,0.3)",
    borderRadius: "var(--radius)", padding: "0.75rem", fontSize: 13,
    color: "var(--red)",
  },

  placeholder: {
    gridColumn: 2, display: "flex", alignItems: "center", justifyContent: "center",
    background: "var(--surface)", border: "1px solid var(--border)",
    borderRadius: "var(--radius)", minHeight: 300,
    color: "var(--muted)", fontSize: 14, gap: "0.5rem",
  },
};

export default function App() {
  const [file,    setFile]    = useState(null);
  const [method,  setMethod]  = useState("gradcam");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyze(file, method);
      setResult(data);
    } catch (e) {
      const msg = e.response?.data?.detail ?? e.message ?? "Unknown error";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={S.app}>
      {/* ── header ── */}
      <header style={S.header}>
        <Brain size={24} style={S.logo} />
        <div>
          <div style={S.title}>NeuroVision AI</div>
          <div style={S.subtitle}>Explainable Brain Tumour Diagnosis & Segmentation</div>
        </div>
      </header>

      <StatusBar />

      {/* ── main grid ── */}
      <main style={S.main}>

        {/* LEFT: upload + controls */}
        <div style={S.leftCol}>

          <div style={S.card}>
            <p style={S.cardTitle}>Upload MRI</p>
            <UploadPanel onFile={setFile} file={file} />
          </div>

          <div style={S.card}>
            <p style={S.cardTitle}>XAI Method</p>
            <select
              style={S.select}
              value={method}
              onChange={(e) => setMethod(e.target.value)}
            >
              <option value="gradcam">Grad-CAM (fast)</option>
              <option value="smoothgrad">SmoothGrad (sharper)</option>
            </select>
          </div>

          <button
            style={S.btn(loading || !file)}
            onClick={run}
            disabled={loading || !file}
          >
            {loading
              ? <><Loader2 size={15} className="spin" /> Analysing…</>
              : <><Brain size={15} /> Analyse</>
            }
          </button>

          {error && (
            <div style={S.error}>
              <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* RIGHT: results */}
        {result
          ? <AnalysisResult result={result} />
          : (
            <div style={S.placeholder}>
              <Brain size={20} />
              Upload an MRI and click Analyse
            </div>
          )
        }
      </main>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 0.9s linear infinite; }
      `}</style>
    </div>
  );
}
