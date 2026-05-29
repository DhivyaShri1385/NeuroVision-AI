/** Top status bar showing API health + loaded models. */
import { useState, useEffect } from "react";
import { getHealth } from "../api/neurovision";
import { Activity, Wifi, WifiOff } from "lucide-react";

const styles = {
  bar: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.5rem 1.25rem",
    background: "var(--surface)",
    borderBottom: "1px solid var(--border)",
    fontSize: 12,
    color: "var(--muted)",
  },
  dot: (ok) => ({
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: ok ? "var(--green)" : "var(--red)",
    flexShrink: 0,
  }),
  model: {
    padding: "1px 7px",
    borderRadius: 10,
    background: "rgba(88,166,255,0.12)",
    color: "var(--accent)",
    fontSize: 11,
  },
  right: { marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 },
};

export default function StatusBar() {
  const [health, setHealth] = useState(null);
  const [error, setError]   = useState(false);

  useEffect(() => {
    const check = () =>
      getHealth()
        .then((h) => { setHealth(h); setError(false); })
        .catch(() => setError(true));
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  const ok = !error && health?.status === "ok";

  return (
    <div style={styles.bar}>
      <div style={styles.dot(ok)} />
      <span>{ok ? "API online" : "API offline"}</span>

      {ok && health.models_loaded.map((m) => (
        <span key={m} style={styles.model}>{m}</span>
      ))}

      <div style={styles.right}>
        <Activity size={13} />
        <span>v{health?.version ?? "—"}</span>
      </div>
    </div>
  );
}
