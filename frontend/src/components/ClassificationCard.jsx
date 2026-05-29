import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { CheckCircle, AlertCircle } from "lucide-react";

const CLASS_COLORS = {
  glioma:      "#f85149",
  meningioma:  "#d29922",
  no_tumor:    "#3fb950",
  pituitary:   "#bc8cff",
};

const styles = {
  card: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "1.25rem",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    marginBottom: "1rem",
    fontWeight: 600,
    fontSize: 15,
    color: "var(--text)",
  },
  prediction: {
    textAlign: "center",
    marginBottom: "1rem",
  },
  label: {
    fontSize: 22,
    fontWeight: 700,
    textTransform: "capitalize",
  },
  conf: { color: "var(--muted)", fontSize: 13, marginTop: 2 },
  barRow: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    marginBottom: "0.4rem",
  },
  barName: { width: 90, color: "var(--muted)", fontSize: 12, textAlign: "right" },
  barOuter: {
    flex: 1,
    height: 8,
    background: "var(--border)",
    borderRadius: 4,
    overflow: "hidden",
  },
  barInner: { height: "100%", borderRadius: 4, transition: "width 0.5s ease" },
  barVal: { width: 38, fontSize: 12, color: "var(--muted)", textAlign: "right" },
  ms: { color: "var(--muted)", fontSize: 11, textAlign: "right", marginTop: "0.75rem" },
};

export default function ClassificationCard({ result }) {
  if (!result) return null;
  const { predicted_class, confidence, probabilities, inference_time_ms } = result;
  const color = CLASS_COLORS[predicted_class] ?? "var(--accent)";

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <CheckCircle size={16} color="var(--green)" />
        Classification
      </div>

      <div style={styles.prediction}>
        <div style={{ ...styles.label, color }}>{predicted_class}</div>
        <div style={styles.conf}>{(confidence * 100).toFixed(1)}% confidence</div>
      </div>

      {/* Probability bars */}
      {Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a)
        .map(([cls, prob]) => (
          <div key={cls} style={styles.barRow}>
            <span style={styles.barName}>{cls}</span>
            <div style={styles.barOuter}>
              <div
                style={{
                  ...styles.barInner,
                  width: `${(prob * 100).toFixed(1)}%`,
                  background: CLASS_COLORS[cls] ?? "var(--accent)",
                  opacity: cls === predicted_class ? 1 : 0.45,
                }}
              />
            </div>
            <span style={styles.barVal}>{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}

      <div style={styles.ms}>⏱ {inference_time_ms} ms</div>
    </div>
  );
}
