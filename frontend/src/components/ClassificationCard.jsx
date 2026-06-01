const CLASS_META = {
  glioma:     { grad: "var(--grad-glioma)",  emoji: "🔴", risk: "High" },
  meningioma: { grad: "var(--grad-mening)",  emoji: "🟠", risk: "Moderate" },
  no_tumor:   { grad: "var(--grad-none)",    emoji: "🟢", risk: "None" },
  pituitary:  { grad: "var(--grad-pituit)",  emoji: "🟣", risk: "Moderate" },
};

export default function ClassificationCard({ result }) {
  if (!result) return null;
  const { predicted_class, confidence, probabilities, model_name, inference_time_ms } = result;
  const meta = CLASS_META[predicted_class] ?? { grad: "var(--grad-accent)", emoji:"🧠", risk:"—" };

  return (
    <div style={S.card} className="fade-up">
      {/* Top accent bar */}
      <div style={{ ...S.accentBar, background: meta.grad }} />

      {/* Prediction hero */}
      <div style={S.hero}>
        <div style={{ ...S.badge, background: meta.grad }}>
          {meta.emoji} {predicted_class.replace("_", " ")}
        </div>
        <div style={S.confWrap}>
          <span style={S.confNum}>{(confidence * 100).toFixed(1)}%</span>
          <span style={S.confLabel}>confidence</span>
        </div>
      </div>

      {/* Risk label */}
      <div style={S.risk}>
        Risk level: <span style={{ color: meta.grad.includes("none") ? "var(--green)" : "var(--orange)" }}>
          {meta.risk}
        </span>
      </div>

      <div style={S.divider} />

      {/* Probability bars */}
      <div style={S.barsTitle}>Class probabilities</div>
      {Object.entries(probabilities)
        .sort(([,a],[,b]) => b - a)
        .map(([cls, prob]) => {
          const cm = CLASS_META[cls] ?? { grad: "var(--grad-accent)" };
          const pct = (prob * 100).toFixed(1);
          const isTop = cls === predicted_class;
          return (
            <div key={cls} style={S.barRow}>
              <div style={S.barLabel}>{cls.replace("_"," ")}</div>
              <div style={S.barTrack}>
                <div style={{
                  ...S.barFill,
                  width: `${pct}%`,
                  background: cm.grad,
                  opacity: isTop ? 1 : 0.45,
                  boxShadow: isTop ? `0 0 12px rgba(96,165,250,0.3)` : "none",
                }} />
              </div>
              <div style={{ ...S.barPct, opacity: isTop ? 1 : 0.6 }}>{pct}%</div>
            </div>
          );
        })}

      <div style={S.footer}>
        <span>🤖 {model_name}</span>
        <span>⏱ {inference_time_ms} ms</span>
      </div>
    </div>
  );
}

const S = {
  card: {
    position: "relative",
    background: "var(--glass)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-lg)",
    overflow: "hidden",
    backdropFilter: "blur(16px)",
    boxShadow: "var(--shadow-card)",
  },
  accentBar: { height: 3, width: "100%" },
  hero: {
    padding: "1.25rem 1.25rem 0.75rem",
    display: "flex", alignItems: "center",
    justifyContent: "space-between",
  },
  badge: {
    padding: "0.4rem 1rem",
    borderRadius: 99,
    fontSize: 14, fontWeight: 700,
    color: "#fff",
    textTransform: "capitalize",
    letterSpacing: "0.02em",
  },
  confWrap: { textAlign: "right" },
  confNum: {
    fontSize: 32, fontWeight: 800,
    background: "var(--grad-accent)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
    backgroundClip: "text",
    display: "block", lineHeight: 1,
  },
  confLabel: { fontSize: 11, color: "var(--text-dim)", display: "block" },
  risk: {
    padding: "0 1.25rem 0.75rem",
    fontSize: 12, color: "var(--text-dim)",
  },
  divider: { height: 1, background: "var(--border)", margin: "0 1.25rem" },
  barsTitle: {
    padding: "0.75rem 1.25rem 0.4rem",
    fontSize: 11, fontWeight: 600,
    color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.08em",
  },
  barRow: {
    display: "flex", alignItems: "center", gap: "0.5rem",
    padding: "0.25rem 1.25rem",
  },
  barLabel: { width: 90, fontSize: 12, color: "var(--text-dim)", textTransform: "capitalize" },
  barTrack: {
    flex: 1, height: 6, borderRadius: 99,
    background: "rgba(255,255,255,0.05)", overflow: "hidden",
  },
  barFill: {
    height: "100%", borderRadius: 99,
    transition: "width 0.8s cubic-bezier(0.16,1,0.3,1)",
  },
  barPct: { width: 40, fontSize: 11, color: "var(--text-dim)", textAlign: "right" },
  footer: {
    display: "flex", justifyContent: "space-between",
    padding: "0.75rem 1.25rem",
    fontSize: 11, color: "var(--text-dim)",
    borderTop: "1px solid var(--border)",
    marginTop: "0.5rem",
  },
};
