/**
 * NeuroVision AI — API client
 * All calls go to the Phase 5 FastAPI backend.
 */
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE_URL, timeout: 60000 });

/** GET /health */
export const getHealth = () => client.get("/health").then((r) => r.data);

/** GET /classes */
export const getClasses = () => client.get("/classes").then((r) => r.data);

/** POST /classify  — file: File object */
export const classify = (file) => {
  const fd = new FormData();
  fd.append("file", file);
  return client.post("/classify", fd).then((r) => r.data);
};

/** POST /segment */
export const segment = (file) => {
  const fd = new FormData();
  fd.append("file", file);
  return client.post("/segment", fd).then((r) => r.data);
};

/** POST /explain  — method: "gradcam" | "smoothgrad" */
export const explain = (file, method = "gradcam") => {
  const fd = new FormData();
  fd.append("file", file);
  return client.post(`/explain?method=${method}`, fd).then((r) => r.data);
};

/** POST /analyze  — full pipeline */
export const analyze = (file, xaiMethod = "gradcam") => {
  const fd = new FormData();
  fd.append("file", file);
  return client
    .post(`/analyze?xai_method=${xaiMethod}`, fd)
    .then((r) => r.data);
};
