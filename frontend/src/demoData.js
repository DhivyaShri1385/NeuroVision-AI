/**
 * Pre-computed demo result for GitHub Pages / offline showcase.
 * Shows what the full analysis looks like when the API is running.
 */
export const DEMO_RESULT = {
  classification: {
    predicted_class:   "glioma",
    class_index:       0,
    confidence:        0.874,
    probabilities: {
      glioma:     0.874,
      meningioma: 0.072,
      no_tumor:   0.031,
      pituitary:  0.023,
    },
    model_name:        "EfficientNetB3",
    inference_time_ms: 487,
  },
  segmentation: {
    mask_b64:         "",   // empty — no real image in demo
    overlay_b64:      "",
    foreground_ratio: 0.187,
    model_name:       "AttentionUNet",
    inference_time_ms: 1243,
  },
  explanation: {
    method:           "gradcam",
    heatmap_b64:      "",
    overlay_b64:      "",
    predicted_class:  "glioma",
    confidence:       0.874,
    inference_time_ms: 1102,
  },
  total_inference_time_ms: 2832,
};
