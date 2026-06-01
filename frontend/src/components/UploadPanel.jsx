import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon, X, CheckCircle } from "lucide-react";

export default function UploadPanel({ onFile, file }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((accepted) => {
    const f = accepted[0];
    if (!f) return;
    setPreview(URL.createObjectURL(f));
    onFile(f);
  }, [onFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg",".jpeg",".png",".bmp",".tiff"] },
    maxFiles: 1,
  });

  const clear = (e) => { e.stopPropagation(); setPreview(null); onFile(null); };

  return (
    <div {...getRootProps()} style={{ ...S.zone, ...(isDragActive ? S.active : {}), ...(preview ? S.filled : {}) }}>
      <input {...getInputProps()} />

      {/* Animated corner accents */}
      <span style={{...S.corner, top:0, left:0,   borderTop:"2px solid", borderLeft:"2px solid"}} />
      <span style={{...S.corner, top:0, right:0,  borderTop:"2px solid", borderRight:"2px solid"}} />
      <span style={{...S.corner, bottom:0, left:0, borderBottom:"2px solid", borderLeft:"2px solid"}} />
      <span style={{...S.corner, bottom:0, right:0,borderBottom:"2px solid", borderRight:"2px solid"}} />

      {preview ? (
        <div style={S.previewWrap}>
          <img src={preview} alt="MRI preview" style={S.img} />
          <div style={S.imgOverlay}>
            <CheckCircle size={20} color="var(--green)" />
          </div>
          <button style={S.clear} onClick={clear}><X size={12}/></button>
          <div style={S.meta}>
            <ImageIcon size={11}/> {file?.name} · {(file?.size/1024).toFixed(1)} KB
          </div>
        </div>
      ) : (
        <div style={S.empty}>
          <div style={{...S.iconWrap, ...(isDragActive ? S.iconActive : {})}}>
            <Upload size={28} color={isDragActive ? "var(--accent)" : "var(--text-dim)"} />
          </div>
          <p style={S.title}>{isDragActive ? "Release to upload" : "Drop MRI here"}</p>
          <p style={S.sub}>or <span style={S.browse}>click to browse</span></p>
          <p style={S.formats}>JPEG · PNG · BMP · TIFF</p>
        </div>
      )}
    </div>
  );
}

const S = {
  zone: {
    position: "relative", cursor: "pointer",
    borderRadius: "var(--r-lg)", padding: "1.5rem",
    background: "var(--glass)", border: "1.5px dashed var(--border)",
    transition: "all 0.25s ease",
    minHeight: 180,
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  active: {
    borderColor: "var(--accent)",
    background: "rgba(96,165,250,0.06)",
    boxShadow: "0 0 0 4px rgba(96,165,250,0.08)",
  },
  filled: {
    borderStyle: "solid", borderColor: "rgba(52,211,153,0.4)",
    background: "rgba(52,211,153,0.03)",
  },
  corner: {
    position: "absolute", width: 12, height: 12,
    borderColor: "var(--accent)", opacity: 0.6,
    transition: "opacity 0.2s",
  },
  empty: { textAlign: "center" },
  iconWrap: {
    width: 60, height: 60, borderRadius: "50%",
    background: "var(--glass)", border: "1px solid var(--border)",
    display: "flex", alignItems: "center", justifyContent: "center",
    margin: "0 auto 0.75rem",
    transition: "all 0.2s",
  },
  iconActive: {
    background: "rgba(96,165,250,0.1)",
    borderColor: "var(--accent)",
    boxShadow: "0 0 20px rgba(96,165,250,0.2)",
  },
  title: { fontWeight: 600, fontSize: 14, color: "var(--text-bright)", marginBottom: 4 },
  sub:   { fontSize: 12, color: "var(--text-dim)" },
  browse:{ color: "var(--accent)", cursor: "pointer" },
  formats:{ fontSize: 10, color: "var(--text-dim)", marginTop: 8, letterSpacing: "0.05em" },
  previewWrap: { position: "relative", width: "100%", textAlign: "center" },
  img: {
    maxHeight: 200, maxWidth: "100%", borderRadius: "var(--r-md)",
    border: "1px solid var(--border)", display: "inline-block",
  },
  imgOverlay: {
    position: "absolute", top: 8, left: 8,
    background: "rgba(0,0,0,0.7)", borderRadius: "50%",
    padding: 4, display: "flex",
  },
  clear: {
    position: "absolute", top: 8, right: 8,
    background: "var(--red)", border: "none", borderRadius: "50%",
    width: 22, height: 22, cursor: "pointer", color: "#fff",
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  meta: {
    display: "flex", alignItems: "center", justifyContent: "center",
    gap: 4, marginTop: 8, fontSize: 11, color: "var(--text-dim)",
  },
};
