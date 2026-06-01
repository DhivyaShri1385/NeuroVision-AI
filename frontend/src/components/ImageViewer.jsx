import { useState } from "react";
import { ZoomIn, ZoomOut, Download } from "lucide-react";

export default function ImageViewer({ title, badge, b64, footer, badgeColor }) {
  const [zoom, setZoom] = useState(false);
  if (!b64) return null;

  const src = `data:image/png;base64,${b64}`;

  const download = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = `${title.toLowerCase().replace(/\s/g,"-")}.png`;
    a.click();
  };

  return (
    <>
      <div style={S.card} className="fade-up">
        <div style={S.header}>
          <span style={S.title}>{title}</span>
          <div style={S.actions}>
            {badge && (
              <span style={{ ...S.badge, background: badgeColor ?? "rgba(96,165,250,0.15)", color: badgeColor ? "#fff" : "var(--accent)" }}>
                {badge}
              </span>
            )}
            <button style={S.btn} onClick={() => setZoom(true)} title="Full screen">
              <ZoomIn size={13}/>
            </button>
            <button style={S.btn} onClick={download} title="Download">
              <Download size={13}/>
            </button>
          </div>
        </div>
        <div style={S.imgWrap}>
          <img src={src} alt={title} style={S.img} />
          {/* Scan overlay animation */}
          <div style={S.scanOverlay} />
        </div>
        {footer && <div style={S.footer}>{footer}</div>}
      </div>

      {/* Lightbox */}
      {zoom && (
        <div style={S.lightbox} onClick={() => setZoom(false)}>
          <div style={S.lightboxInner}>
            <img src={src} alt={title} style={S.lightboxImg} />
            <button style={S.closeBtn} onClick={() => setZoom(false)}>
              <ZoomOut size={18}/>
            </button>
          </div>
        </div>
      )}
    </>
  );
}

const S = {
  card: {
    background: "var(--glass)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-lg)",
    overflow: "hidden",
    backdropFilter: "blur(12px)",
    boxShadow: "var(--shadow-card)",
  },
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0.65rem 0.9rem",
    borderBottom: "1px solid var(--border)",
  },
  title: { fontSize: 12, fontWeight: 600, color: "var(--text)" },
  actions: { display: "flex", alignItems: "center", gap: "0.4rem" },
  badge: {
    fontSize: 10, padding: "2px 8px", borderRadius: 99,
    fontWeight: 600, letterSpacing: "0.04em",
  },
  btn: {
    background: "var(--glass)", border: "1px solid var(--border)",
    borderRadius: "var(--r-sm)", padding: "3px 6px",
    cursor: "pointer", color: "var(--text-dim)",
    display: "flex", alignItems: "center",
    transition: "all 0.15s",
  },
  imgWrap: { position: "relative", overflow: "hidden" },
  img: { width: "100%", display: "block" },
  scanOverlay: {
    position: "absolute", inset: 0,
    background: "linear-gradient(to bottom,transparent 40%,rgba(15,23,42,0.2) 100%)",
    pointerEvents: "none",
  },
  footer: {
    padding: "0.5rem 0.9rem", fontSize: 11, color: "var(--text-dim)",
    borderTop: "1px solid var(--border)",
    background: "rgba(0,0,0,0.1)",
  },
  lightbox: {
    position: "fixed", inset: 0, zIndex: 1000,
    background: "rgba(2,8,23,0.92)", backdropFilter: "blur(12px)",
    display: "flex", alignItems: "center", justifyContent: "center",
    cursor: "zoom-out",
  },
  lightboxInner: { position: "relative", maxWidth: "90vw", maxHeight: "90vh" },
  lightboxImg: {
    maxWidth: "90vw", maxHeight: "90vh",
    borderRadius: "var(--r-lg)", border: "1px solid var(--border)",
    boxShadow: "0 20px 80px rgba(0,0,0,0.8)",
  },
  closeBtn: {
    position: "absolute", top: -16, right: -16,
    background: "var(--surface-2)", border: "1px solid var(--border)",
    borderRadius: "50%", width: 36, height: 36,
    cursor: "pointer", color: "var(--text)",
    display: "flex", alignItems: "center", justifyContent: "center",
  },
};
