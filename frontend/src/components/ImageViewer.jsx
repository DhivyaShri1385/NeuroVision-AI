/** Reusable panel that shows a base64 PNG with a title and optional badge. */
const styles = {
  card: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0.75rem 1rem",
    borderBottom: "1px solid var(--border)",
    fontWeight: 600,
    fontSize: 14,
  },
  badge: {
    fontSize: 11,
    padding: "2px 8px",
    borderRadius: 20,
    background: "rgba(88,166,255,0.15)",
    color: "var(--accent)",
  },
  img: { width: "100%", display: "block" },
  footer: {
    padding: "0.4rem 1rem",
    fontSize: 11,
    color: "var(--muted)",
    borderTop: "1px solid var(--border)",
  },
};

export default function ImageViewer({ title, badge, b64, footer }) {
  if (!b64) return null;
  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span>{title}</span>
        {badge && <span style={styles.badge}>{badge}</span>}
      </div>
      <img
        src={`data:image/png;base64,${b64}`}
        alt={title}
        style={styles.img}
      />
      {footer && <div style={styles.footer}>{footer}</div>}
    </div>
  );
}
