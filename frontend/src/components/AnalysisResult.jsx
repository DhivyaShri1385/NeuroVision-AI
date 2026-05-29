/**
 * Full analysis results dashboard.
 * Shows: ClassificationCard | Seg overlay | Heatmap overlay | Method tabs
 */
import ClassificationCard from "./ClassificationCard";
import ImageViewer        from "./ImageViewer";
import { useState }       from "react";
import { Brain, Layers, Zap } from "lucide-react";

const styles = {
  wrapper: { display: "flex", flexDirection: "column", gap: "1rem" },
  tabs: {
    display: "flex",
    gap: "0.5rem",
    borderBottom: "1px solid var(--border)",
    marginBottom: "0.75rem",
  },
  tab: (active) => ({
    padding: "0.4rem 0.9rem",
    border: "none",
    borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
    background: "none",
    color: active ? "var(--accent)" : "var(--muted)",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: active ? 600 : 400,
    display: "flex",
    alignItems: "center",
    gap: "0.35rem",
  }),
  grid2: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "0.75rem",
  },
  timing: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "0.75rem 1rem",
    display: "flex",
    gap: "1.5rem",
    fontSize: 12,
    color: "var(--muted)",
    flexWrap: "wrap",
  },
  timingItem: { display: "flex", flexDirection: "column", alignItems: "center" },
  timingVal: { fontSize: 18, fontWeight: 700, color: "var(--text)" },
};

const TABS = [
  { id: "classify", label: "Classification", icon: <Brain size={13} /> },
  { id: "segment",  label: "Segmentation",   icon: <Layers size={13} /> },
  { id: "explain",  label: "Explanation",    icon: <Zap size={13} /> },
];

export default function AnalysisResult({ result }) {
  const [tab, setTab] = useState("classify");
  if (!result) return null;

  const { classification, segmentation, explanation, total_inference_time_ms } = result;

  return (
    <div style={styles.wrapper}>
      {/* Timing summary */}
      <div style={styles.timing}>
        <div style={styles.timingItem}>
          <span style={styles.timingVal}>{total_inference_time_ms} ms</span>
          <span>Total</span>
        </div>
        <div style={styles.timingItem}>
          <span style={styles.timingVal}>{classification?.inference_time_ms} ms</span>
          <span>Classify</span>
        </div>
        <div style={styles.timingItem}>
          <span style={styles.timingVal}>{segmentation?.inference_time_ms || "—"} ms</span>
          <span>Segment</span>
        </div>
        <div style={styles.timingItem}>
          <span style={styles.timingVal}>{explanation?.inference_time_ms} ms</span>
          <span>XAI</span>
        </div>
      </div>

      {/* Tab bar */}
      <div style={styles.tabs}>
        {TABS.map(({ id, label, icon }) => (
          <button key={id} style={styles.tab(tab === id)} onClick={() => setTab(id)}>
            {icon} {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "classify" && (
        <ClassificationCard result={classification} />
      )}

      {tab === "segment" && (
        <>
          {segmentation?.overlay_b64 ? (
            <div style={styles.grid2}>
              <ImageViewer
                title="Segmentation Mask"
                badge="binary"
                b64={segmentation.mask_b64}
                footer={`Tumour area: ${(segmentation.foreground_ratio * 100).toFixed(2)}%`}
              />
              <ImageViewer
                title="Mask Overlay"
                badge={segmentation.model_name}
                b64={segmentation.overlay_b64}
                footer={`Inference: ${segmentation.inference_time_ms} ms`}
              />
            </div>
          ) : (
            <NoModel name="Segmentation" script="run_phase3.py" />
          )}
        </>
      )}

      {tab === "explain" && (
        <>
          {explanation?.overlay_b64 ? (
            <div style={styles.grid2}>
              <ImageViewer
                title="Heatmap"
                badge={explanation.method}
                b64={explanation.heatmap_b64}
                footer={`Class: ${explanation.predicted_class} (${(explanation.confidence * 100).toFixed(1)}%)`}
              />
              <ImageViewer
                title="Overlay"
                badge={explanation.method}
                b64={explanation.overlay_b64}
                footer={`Inference: ${explanation.inference_time_ms} ms`}
              />
            </div>
          ) : (
            <NoModel name="XAI" script="run_phase2.py" />
          )}
        </>
      )}
    </div>
  );
}

function NoModel({ name, script }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: "var(--radius)", padding: "2rem", textAlign: "center",
      color: "var(--muted)",
    }}>
      <p style={{ marginBottom: "0.5rem" }}>⚠ {name} model not loaded.</p>
      <code style={{ fontSize: 12, background: "var(--bg)", padding: "2px 8px", borderRadius: 4 }}>
        python scripts/{script}
      </code>
    </div>
  );
}
