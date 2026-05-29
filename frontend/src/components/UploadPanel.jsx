import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon, X } from "lucide-react";

const styles = {
  wrapper: {
    background: "var(--surface)",
    border: "2px dashed var(--border)",
    borderRadius: "var(--radius)",
    padding: "2rem",
    textAlign: "center",
    cursor: "pointer",
    transition: "border-color 0.2s, background 0.2s",
  },
  wrapperActive: {
    borderColor: "var(--accent)",
    background: "rgba(88,166,255,0.05)",
  },
  icon: { color: "var(--muted)", marginBottom: "0.75rem" },
  title: { color: "var(--text)", fontWeight: 600, marginBottom: "0.25rem" },
  sub: { color: "var(--muted)", fontSize: 12 },
  preview: {
    marginTop: "1.25rem",
    position: "relative",
    display: "inline-block",
  },
  img: {
    maxHeight: 220,
    maxWidth: "100%",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    display: "block",
  },
  clearBtn: {
    position: "absolute",
    top: -8,
    right: -8,
    background: "var(--red)",
    border: "none",
    borderRadius: "50%",
    width: 22,
    height: 22,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    color: "#fff",
  },
  meta: { marginTop: "0.5rem", color: "var(--muted)", fontSize: 12 },
};

export default function UploadPanel({ onFile, file }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback(
    (accepted) => {
      const f = accepted[0];
      if (!f) return;
      setPreview(URL.createObjectURL(f));
      onFile(f);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] },
    maxFiles: 1,
  });

  const clear = (e) => {
    e.stopPropagation();
    setPreview(null);
    onFile(null);
  };

  return (
    <div
      {...getRootProps()}
      style={{
        ...styles.wrapper,
        ...(isDragActive ? styles.wrapperActive : {}),
      }}
    >
      <input {...getInputProps()} />

      {!preview ? (
        <>
          <div style={styles.icon}>
            <Upload size={36} />
          </div>
          <p style={styles.title}>
            {isDragActive ? "Drop it here…" : "Drag & drop an MRI image"}
          </p>
          <p style={styles.sub}>or click to browse · JPEG, PNG, BMP, TIFF</p>
        </>
      ) : (
        <div style={styles.preview}>
          <img src={preview} alt="preview" style={styles.img} />
          <button style={styles.clearBtn} onClick={clear}>
            <X size={12} />
          </button>
          <p style={styles.meta}>
            <ImageIcon
              size={12}
              style={{ verticalAlign: "middle", marginRight: 4 }}
            />
            {file?.name} · {(file?.size / 1024).toFixed(1)} KB
          </p>
        </div>
      )}
    </div>
  );
}
