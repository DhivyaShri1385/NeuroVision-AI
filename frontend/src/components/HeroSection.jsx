import { Brain, Zap, Shield, Activity } from "lucide-react";

const features = [
  { icon: <Brain size={18}/>,    label: "AI Diagnosis",      desc: "4-class tumour classification" },
  { icon: <Activity size={18}/>, label: "Segmentation",      desc: "Pixel-level tumour masking" },
  { icon: <Zap size={18}/>,      label: "Explainability",    desc: "Grad-CAM + SmoothGrad XAI" },
  { icon: <Shield size={18}/>,   label: "Clinical Reports",  desc: "One-click PDF generation" },
];

export default function HeroSection() {
  return (
    <div style={S.hero}>
      {/* Ambient glow orbs */}
      <div style={S.orb1} />
      <div style={S.orb2} />

      {/* Brain animation */}
      <div style={S.brainWrap} className="float">
        <div style={S.brainRing} />
        <div style={S.brainRing2} />
        <div style={S.brainIcon}>
          <Brain size={52} color="#60a5fa" strokeWidth={1.5} />
        </div>
        {/* Scan line */}
        <div style={S.scanLine} />
      </div>

      <h1 style={S.title}>
        <span className="glow-text">NeuroVision AI</span>
      </h1>
      <p style={S.subtitle}>
        Explainable Brain Tumour Diagnosis &amp; Segmentation Platform
      </p>

      {/* Feature pills */}
      <div style={S.pills}>
        {features.map(({ icon, label, desc }) => (
          <div key={label} style={S.pill}>
            <span style={S.pillIcon}>{icon}</span>
            <div>
              <div style={S.pillLabel}>{label}</div>
              <div style={S.pillDesc}>{desc}</div>
            </div>
          </div>
        ))}
      </div>

      <p style={S.cta}>↑ Upload an MRI image above to begin</p>
    </div>
  );
}

const S = {
  hero: {
    position: "relative",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "3rem 2rem",
    overflow: "hidden",
    borderRadius: "var(--r-xl)",
    background: "linear-gradient(160deg,rgba(30,27,75,0.7) 0%,rgba(15,23,42,0.5) 100%)",
    border: "1px solid var(--border)",
    backdropFilter: "blur(12px)",
    minHeight: 420,
  },
  orb1: {
    position: "absolute", top: -80, left: -80,
    width: 320, height: 320, borderRadius: "50%",
    background: "radial-gradient(circle,rgba(96,165,250,0.12) 0%,transparent 70%)",
    pointerEvents: "none",
  },
  orb2: {
    position: "absolute", bottom: -60, right: -60,
    width: 280, height: 280, borderRadius: "50%",
    background: "radial-gradient(circle,rgba(139,92,246,0.10) 0%,transparent 70%)",
    pointerEvents: "none",
  },
  brainWrap: {
    position: "relative", width: 120, height: 120,
    display: "flex", alignItems: "center", justifyContent: "center",
    marginBottom: "1.5rem",
  },
  brainRing: {
    position: "absolute", inset: -10,
    border: "1px solid rgba(96,165,250,0.25)",
    borderRadius: "50%",
    animation: "pulse-ring 3s ease-in-out infinite",
  },
  brainRing2: {
    position: "absolute", inset: -24,
    border: "1px solid rgba(96,165,250,0.12)",
    borderRadius: "50%",
    animation: "pulse-ring 3s ease-in-out infinite 0.8s",
  },
  brainIcon: {
    width: 96, height: 96, borderRadius: "50%",
    background: "linear-gradient(135deg,rgba(59,130,246,0.2),rgba(139,92,246,0.2))",
    border: "1px solid rgba(96,165,250,0.3)",
    display: "flex", alignItems: "center", justifyContent: "center",
    boxShadow: "0 0 40px rgba(96,165,250,0.15)",
  },
  scanLine: {
    position: "absolute", left: 10, right: 10, height: 2,
    background: "linear-gradient(90deg,transparent,rgba(96,165,250,0.7),transparent)",
    animation: "scan-line 3s ease-in-out infinite",
    borderRadius: 99,
  },
  title: {
    fontSize: "clamp(2rem, 5vw, 3rem)",
    fontWeight: 800, letterSpacing: "-0.04em",
    marginBottom: "0.5rem", textAlign: "center",
  },
  subtitle: {
    fontSize: 15, color: "var(--text-dim)",
    textAlign: "center", maxWidth: 400,
    marginBottom: "2rem", lineHeight: 1.5,
  },
  pills: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: "0.6rem", width: "100%", maxWidth: 480,
    marginBottom: "2rem",
  },
  pill: {
    display: "flex", alignItems: "center", gap: "0.6rem",
    padding: "0.6rem 0.8rem",
    background: "var(--glass)",
    border: "1px solid var(--border)",
    borderRadius: "var(--r-md)",
    backdropFilter: "blur(8px)",
  },
  pillIcon: { color: "var(--accent)", flexShrink: 0 },
  pillLabel: { fontSize: 12, fontWeight: 600, color: "var(--text-bright)" },
  pillDesc:  { fontSize: 11, color: "var(--text-dim)" },
  cta: {
    fontSize: 12, color: "var(--accent)",
    opacity: 0.7, animation: "fade-up 1s ease 0.5s both",
  },
};
