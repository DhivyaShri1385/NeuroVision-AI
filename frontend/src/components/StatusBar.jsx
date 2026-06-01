import { useState, useEffect } from "react";
import { getHealth } from "../api/neurovision";
import { Wifi, WifiOff, Circle } from "lucide-react";

export default function StatusBar({ demoMode }) {
  const [health, setHealth] = useState(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    const check = () =>
      getHealth().then(h => { setHealth(h); setErr(false); }).catch(() => setErr(true));
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  const ok = !err && health?.status === "ok";

  return (
    <div style={S.bar}>
      {/* Status dot */}
      <div style={S.statusGroup}>
        <span style={{ ...S.dot, background: demoMode ? "var(--yellow)" : ok ? "var(--green)" : "var(--red)" }} />
        <span style={S.statusText}>
          {demoMode ? "Demo mode" : ok ? "API connected" : "API offline"}
        </span>
        {!ok && !demoMode && (
          <span style={S.hint}>· run python scripts/run_phase5.py</span>
        )}
      </div>

      {/* Model chips */}
      {ok && health?.models_loaded?.map(m => (
        <span key={m} style={S.chip}>{m}</span>
      ))}

      <div style={S.right}>
        {ok ? <Wifi size={12} color="var(--green)" /> : <WifiOff size={12} color="var(--text-dim)" />}
        <span style={S.ver}>v{health?.version ?? "—"}</span>
      </div>
    </div>
  );
}

const S = {
  bar: {
    display: "flex", alignItems: "center", flexWrap: "wrap", gap: "0.5rem",
    padding: "0.45rem 1.5rem",
    background: "rgba(10,15,30,0.7)",
    borderBottom: "1px solid var(--border)",
    backdropFilter: "blur(10px)",
    fontSize: 11, color: "var(--text-dim)",
  },
  statusGroup: { display: "flex", alignItems: "center", gap: "0.4rem" },
  dot: {
    width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
    boxShadow: "0 0 6px currentColor",
  },
  statusText: { color: "var(--text)", fontWeight: 500 },
  hint: { color: "var(--text-dim)", fontFamily: "'JetBrains Mono',monospace", fontSize: 10 },
  chip: {
    padding: "1px 8px", borderRadius: 99, fontSize: 10,
    background: "rgba(96,165,250,0.1)",
    border: "1px solid rgba(96,165,250,0.2)",
    color: "var(--accent)", fontWeight: 500,
  },
  right: { marginLeft: "auto", display: "flex", alignItems: "center", gap: "0.4rem" },
  ver:  { fontFamily: "'JetBrains Mono',monospace", fontSize: 10 },
};
