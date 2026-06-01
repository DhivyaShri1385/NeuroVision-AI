import { useState } from "react";
import ClassificationCard from "./ClassificationCard";
import ImageViewer from "./ImageViewer";
import { Brain, Layers, Zap, Clock } from "lucide-react";

const TABS = [
  { id: "classify", label: "Classification", icon: <Brain size={14}/> },
  { id: "segment",  label: "Segmentation",   icon: <Layers size={14}/> },
  { id: "explain",  label: "Explanation",     icon: <Zap size={14}/> },
];

const CLASS_GRADS = {
  glioma: "var(--grad-glioma)", meningioma: "var(--grad-mening)",
  no_tumor: "var(--grad-none)", pituitary: "var(--grad-pituit)",
};

export default function AnalysisResult({ result }) {
  const [tab, setTab] = useState("classify");
  if (!result) return null;
  const { classification: clf, segmentation: seg, explanation: xai, total_inference_time_ms } = result;

  return (
    <div style={S.wrap} className="fade-up">

      {/* Timing ribbon */}
      <div style={S.ribbon}>
        {[
          ["Total", total_inference_time_ms],
          ["Classify", clf?.inference_time_ms],
          ["Segment", seg?.inference_time_ms || "—"],
          ["XAI", xai?.inference_time_ms],
        ].map(([k, v]) => (
          <div key={k} style={S.ribbonItem}>
            <Clock size={10}/> <b>{v}{typeof v === "number" ? " ms" : ""}</b> <span style={{opacity:0.6}}>{k}</span>
          </div>
        ))}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ ...S.chip, background: CLASS_GRADS[clf?.predicted_class] ?? "var(--grad-accent)" }}>
            {clf?.predicted_class?.replace("_"," ")}
          </span>
          <span style={S.confChip}>{(clf?.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={S.tabs}>
        {TABS.map(({ id, label, icon }) => (
          <button key={id} style={{ ...S.tab, ...(tab===id ? S.tabActive : {}) }} onClick={() => setTab(id)}>
            {icon} {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={S.content}>
        {tab === "classify" && <ClassificationCard result={clf} />}

        {tab === "segment" && (
          seg?.overlay_b64
            ? <div style={S.grid2}>
                <ImageViewer title="Segmentation Mask" badge="binary" b64={seg.mask_b64}
                  footer={`Tumour area: ${(seg.foreground_ratio*100).toFixed(2)}%`} />
                <ImageViewer title="Mask Overlay" badge={seg.model_name} b64={seg.overlay_b64}
                  footer={`${seg.inference_time_ms} ms`} badgeColor="rgba(52,211,153,0.3)" />
              </div>
            : <Unavailable name="Segmentation" cmd="python scripts/run_phase3.py" />
        )}

        {tab === "explain" && (
          xai?.overlay_b64
            ? <div style={S.grid2}>
                <ImageViewer title="Heatmap" badge={xai.method?.toUpperCase()} b64={xai.heatmap_b64}
                  footer={`Class: ${xai.predicted_class} · ${(xai.confidence*100).toFixed(1)}%`}
                  badgeColor="rgba(96,165,250,0.3)" />
                <ImageViewer title="Overlay" badge={xai.method?.toUpperCase()} b64={xai.overlay_b64}
                  footer={`Inference: ${xai.inference_time_ms} ms`}
                  badgeColor="rgba(96,165,250,0.3)" />
              </div>
            : <Unavailable name="XAI" cmd="python scripts/run_phase2.py" />
        )}
      </div>
    </div>
  );
}

function Unavailable({ name, cmd }) {
  return (
    <div style={S.unavail}>
      <div style={S.unavailIcon}>⚠</div>
      <p style={S.unavailTitle}>{name} model not loaded</p>
      <code style={S.unavailCmd}>{cmd}</code>
    </div>
  );
}

const S = {
  wrap: { display: "flex", flexDirection: "column", gap: "0.75rem" },
  ribbon: {
    display: "flex", alignItems: "center", flexWrap: "wrap", gap: "0.75rem",
    padding: "0.6rem 1rem",
    background: "var(--glass)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-md)",
    fontSize: 11, color: "var(--text-dim)",
    backdropFilter: "blur(12px)",
  },
  ribbonItem: { display: "flex", alignItems: "center", gap: "0.25rem" },
  chip: {
    padding: "2px 10px", borderRadius: 99,
    fontSize: 11, fontWeight: 700, color: "#fff",
    textTransform: "capitalize",
  },
  confChip: {
    padding: "2px 8px", borderRadius: 99, fontSize: 11,
    background: "rgba(96,165,250,0.15)",
    border: "1px solid rgba(96,165,250,0.3)",
    color: "var(--accent)", fontWeight: 600,
  },
  tabs: {
    display: "flex", gap: "0.25rem",
    padding: "0.25rem",
    background: "var(--glass)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-md)",
    backdropFilter: "blur(12px)",
  },
  tab: {
    flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "0.4rem",
    padding: "0.5rem 0.75rem",
    border: "none", borderRadius: "var(--r-sm)",
    background: "transparent", color: "var(--text-dim)",
    cursor: "pointer", fontSize: 13, fontWeight: 500,
    transition: "all 0.2s",
  },
  tabActive: {
    background: "linear-gradient(135deg,rgba(59,130,246,0.2),rgba(139,92,246,0.2))",
    color: "var(--text-bright)",
    border: "1px solid rgba(96,165,250,0.25)",
    boxShadow: "0 2px 12px rgba(96,165,250,0.1)",
  },
  content: { minHeight: 300 },
  grid2: {
    display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem",
  },
  unavail: {
    display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", gap: "0.5rem",
    padding: "3rem",
    background: "var(--glass)",
    border: "1px dashed var(--border)",
    borderRadius: "var(--r-lg)",
    textAlign: "center",
  },
  unavailIcon: { fontSize: 28, marginBottom: 4 },
  unavailTitle: { fontSize: 14, color: "var(--text-dim)" },
  unavailCmd: {
    fontSize: 11, padding: "4px 12px", borderRadius: "var(--r-sm)",
    background: "rgba(0,0,0,0.4)",
    border: "1px solid var(--border)",
    color: "var(--accent)",
    fontFamily: "'JetBrains Mono',monospace",
  },
};
