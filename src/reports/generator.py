"""
PDF Report Generator — Phase 7

Generates a clinical-quality PDF analysis report using ReportLab.

Report sections:
  1. Header       — logo, title, timestamp, session ID
  2. MRI Input    — original uploaded image
  3. Diagnosis    — predicted class, confidence bar, full probability table
  4. Segmentation — mask overlay image + tumour area percentage
  5. Explanation  — Grad-CAM / SmoothGrad heatmap overlay
  6. Technical    — model names, inference times, disclaimer

Usage:
    from src.reports.generator import ReportGenerator
    gen = ReportGenerator(config)
    pdf_bytes = gen.generate(
        image_bytes    = raw_mri_bytes,
        classification = clf_response_dict,
        segmentation   = seg_response_dict,   # or None
        explanation    = xai_response_dict,
        session_id     = "NV-20260529-001",
    )
    with open("report.pdf", "wb") as f:
        f.write(pdf_bytes)
"""

from __future__ import annotations

import base64
import io
import uuid
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

from src.utils.config import AppConfig
from src.utils.logger import logger

# ── Colour palette (dark theme adapted for print) ──────────────────
_C_BG      = colors.HexColor("#0d1117")
_C_SURFACE = colors.HexColor("#161b22")
_C_BORDER  = colors.HexColor("#30363d")
_C_TEXT    = colors.HexColor("#c9d1d9")
_C_MUTED   = colors.HexColor("#8b949e")
_C_ACCENT  = colors.HexColor("#58a6ff")
_C_GREEN   = colors.HexColor("#3fb950")
_C_ORANGE  = colors.HexColor("#d29922")
_C_RED     = colors.HexColor("#f85149")
_C_PURPLE  = colors.HexColor("#bc8cff")
_C_WHITE   = colors.white

_CLASS_COLORS = {
    "glioma":     _C_RED,
    "meningioma": _C_ORANGE,
    "no_tumor":   _C_GREEN,
    "pituitary":  _C_PURPLE,
}

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _make_styles() -> dict:
    base = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "title": S("nv_title",
            fontSize=22, textColor=_C_ACCENT, fontName="Helvetica-Bold",
            alignment=TA_LEFT, spaceAfter=2),
        "subtitle": S("nv_sub",
            fontSize=10, textColor=_C_MUTED, fontName="Helvetica",
            alignment=TA_LEFT, spaceAfter=6),
        "section": S("nv_section",
            fontSize=12, textColor=_C_ACCENT, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=6),
        "body": S("nv_body",
            fontSize=9, textColor=_C_TEXT, fontName="Helvetica",
            leading=14, spaceAfter=4),
        "small": S("nv_small",
            fontSize=8, textColor=_C_MUTED, fontName="Helvetica",
            leading=11),
        "label": S("nv_label",
            fontSize=8, textColor=_C_MUTED, fontName="Helvetica-Bold",
            alignment=TA_RIGHT),
        "value": S("nv_value",
            fontSize=9, textColor=_C_TEXT, fontName="Helvetica"),
        "pred_class": S("nv_pred",
            fontSize=18, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "disclaimer": S("nv_disc",
            fontSize=7, textColor=_C_MUTED, fontName="Helvetica-Oblique",
            leading=10),
        "footer": S("nv_footer",
            fontSize=7, textColor=_C_MUTED, fontName="Helvetica",
            alignment=TA_CENTER),
    }


# ---------------------------------------------------------------------------
# Page template (dark background + header stripe)
# ---------------------------------------------------------------------------

class _NVDocTemplate(BaseDocTemplate):
    def __init__(self, buf, **kw):
        super().__init__(buf, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN + 1.2*cm, bottomMargin=MARGIN,
                         **kw)
        frame = Frame(MARGIN, MARGIN,
                      PAGE_W - 2*MARGIN, PAGE_H - 2*MARGIN - 1.2*cm,
                      id="main")
        self.addPageTemplates([
            PageTemplate(id="main", frames=[frame],
                         onPage=_draw_page_chrome)
        ])


def _draw_page_chrome(canvas, doc):
    """Draw dark background, header stripe and page number on every page."""
    canvas.saveState()

    # Dark background
    canvas.setFillColor(_C_BG)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Header stripe
    stripe_h = 1.2 * cm
    canvas.setFillColor(_C_SURFACE)
    canvas.rect(0, PAGE_H - stripe_h, PAGE_W, stripe_h, fill=1, stroke=0)

    # Header text
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(_C_ACCENT)
    canvas.drawString(MARGIN, PAGE_H - stripe_h + 4*mm, "NeuroVision AI")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_C_MUTED)
    canvas.drawRightString(
        PAGE_W - MARGIN, PAGE_H - stripe_h + 4*mm,
        f"Confidential — Page {doc.page}"
    )

    # Bottom border line
    canvas.setStrokeColor(_C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, MARGIN - 2*mm, PAGE_W - MARGIN, MARGIN - 2*mm)

    canvas.restoreState()


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Produces a PDF analysis report from NeuroVision AI results.

    Args:
        config: AppConfig.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.styles = _make_styles()

    def generate(
        self,
        image_bytes:    bytes,
        classification: dict,
        segmentation:   dict | None,
        explanation:    dict | None,
        session_id:     str | None = None,
    ) -> bytes:
        """Build the PDF and return it as bytes.

        Args:
            image_bytes:    Raw bytes of the uploaded MRI image.
            classification: ClassificationResponse dict from API.
            segmentation:   SegmentationResponse dict (or None if unavailable).
            explanation:    ExplanationResponse dict (or None if unavailable).
            session_id:     Optional session identifier string.

        Returns:
            PDF file as bytes.
        """
        if session_id is None:
            session_id = f"NV-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"

        buf   = io.BytesIO()
        doc   = _NVDocTemplate(buf)
        story = []
        S     = self.styles

        # ── 1. Title block ────────────────────────────────────────────
        story += self._title_block(session_id, classification)

        # ── 2. MRI image ──────────────────────────────────────────────
        story += self._section("Brain MRI — Input Image")
        story += self._mri_block(image_bytes)

        # ── 3. Diagnosis ──────────────────────────────────────────────
        story += self._section("Diagnosis")
        story += self._diagnosis_block(classification)

        # ── 4. Segmentation ───────────────────────────────────────────
        story += self._section("Tumour Segmentation")
        story += self._segmentation_block(image_bytes, segmentation)

        # ── 5. XAI explanation ────────────────────────────────────────
        story += self._section("Explainability (Grad-CAM)")
        story += self._xai_block(image_bytes, explanation)

        # ── 6. Technical footer ───────────────────────────────────────
        story += self._technical_block(classification, segmentation, explanation)
        story += self._disclaimer_block()

        doc.build(story)
        pdf_bytes = buf.getvalue()
        logger.info(f"PDF report generated | session={session_id} | size={len(pdf_bytes)//1024} KB")
        return pdf_bytes

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _title_block(self, session_id: str, clf: dict) -> list:
        S  = self.styles
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S UTC")
        pred  = clf.get("predicted_class", "unknown")
        color = _CLASS_COLORS.get(pred, _C_ACCENT)

        return [
            Paragraph("NeuroVision AI", S["title"]),
            Paragraph("Explainable Brain Tumour Diagnosis Report", S["subtitle"]),
            HRFlowable(width="100%", thickness=1, color=_C_BORDER, spaceAfter=6),
            self._kv_table([
                ("Session ID", session_id),
                ("Generated",  ts),
                ("Diagnosis",  pred.upper()),
                ("Confidence", f"{clf.get('confidence', 0)*100:.1f}%"),
                ("Model",      clf.get("model_name", "EfficientNetB3")),
            ]),
            Spacer(1, 6),
        ]

    def _mri_block(self, image_bytes: bytes) -> list:
        try:
            img_io = io.BytesIO(image_bytes)
            img = Image(img_io, width=6*cm, height=6*cm, kind="proportional")
            return [img, Spacer(1, 4)]
        except Exception as e:
            return [Paragraph(f"[Image unavailable: {e}]", self.styles["small"])]

    def _diagnosis_block(self, clf: dict) -> list:
        S      = self.styles
        pred   = clf.get("predicted_class", "unknown")
        conf   = clf.get("confidence", 0.0)
        probs  = clf.get("probabilities", {})
        color  = _CLASS_COLORS.get(pred, _C_ACCENT)

        elements = []

        # Big prediction label
        elements.append(
            Paragraph(f'<font color="{color.hexval()}">{pred.upper()}</font>', S["pred_class"])
        )
        elements.append(Spacer(1, 4))

        # Probability table
        headers = ["Class", "Probability", "Bar"]
        rows    = [headers]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for cls, p in sorted_probs:
            bar_width = max(int(p * 80), 1)
            bar_color = _CLASS_COLORS.get(cls, _C_ACCENT).hexval()
            bar_cell  = f'<font color="{bar_color}">{"█" * bar_width}</font>'
            rows.append([
                Paragraph(cls, S["value"]),
                Paragraph(f"{p*100:.1f}%", S["value"]),
                Paragraph(bar_cell, S["small"]),
            ])

        tbl = Table(rows, colWidths=[3.5*cm, 2.5*cm, 9*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), _C_SURFACE),
            ("TEXTCOLOR",   (0,0), (-1,0), _C_MUTED),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,0), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [_C_BG, _C_SURFACE]),
            ("GRID",        (0,0), (-1,-1), 0.3, _C_BORDER),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0),(-1,-1), 6),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 6))
        return elements

    def _segmentation_block(self, image_bytes: bytes, seg: dict | None) -> list:
        S = self.styles
        if not seg or not seg.get("overlay_b64"):
            return [Paragraph("Segmentation model not available.", S["small"]), Spacer(1, 4)]

        elements = []
        ratio = seg.get("foreground_ratio", 0)

        # Side-by-side: original | overlay
        imgs = []
        imgs.append(self._b64_image(seg.get("mask_b64", ""), 7*cm))
        imgs.append(self._b64_image(seg.get("overlay_b64", ""), 7*cm))

        imgs_ok = [i for i in imgs if i]
        if imgs_ok:
            tbl = Table([imgs_ok], colWidths=[7.5*cm] * len(imgs_ok))
            tbl.setStyle(TableStyle([
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            elements.append(tbl)

        elements.append(Spacer(1, 4))
        elements.append(
            Paragraph(f"Tumour area (pixel fraction): <b>{ratio*100:.2f}%</b>", S["body"])
        )
        elements.append(
            Paragraph(f"Model: {seg.get('model_name', 'AttentionUNet')}  |  "
                      f"Inference: {seg.get('inference_time_ms', 0)} ms", S["small"])
        )
        return elements

    def _xai_block(self, image_bytes: bytes, xai: dict | None) -> list:
        S = self.styles
        if not xai or not xai.get("overlay_b64"):
            return [Paragraph("XAI model not available.", S["small"]), Spacer(1, 4)]

        elements = []
        imgs = []
        imgs.append(self._b64_image(xai.get("heatmap_b64", ""), 7*cm))
        imgs.append(self._b64_image(xai.get("overlay_b64", ""), 7*cm))

        imgs_ok = [i for i in imgs if i]
        if imgs_ok:
            tbl = Table([imgs_ok], colWidths=[7.5*cm] * len(imgs_ok))
            tbl.setStyle(TableStyle([
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            elements.append(tbl)

        elements.append(Spacer(1, 4))
        elements.append(
            Paragraph(
                f"Method: {xai.get('method', 'GradCAM').upper()}  |  "
                f"Target class: {xai.get('predicted_class', '—')}  |  "
                f"Inference: {xai.get('inference_time_ms', 0)} ms",
                S["small"]
            )
        )
        return elements

    def _technical_block(self, clf, seg, xai) -> list:
        rows = [
            ("Classifier",        clf.get("model_name", "EfficientNetB3")),
            ("Seg. architecture", seg.get("model_name", "N/A") if seg else "N/A"),
            ("XAI method",        xai.get("method", "N/A") if xai else "N/A"),
            ("Classify time",     f"{clf.get('inference_time_ms', 0)} ms"),
        ]
        return [
            *self._section("Technical Details"),   # unpack so no nested list
            self._kv_table(rows),
            Spacer(1, 6),
        ]

    def _disclaimer_block(self) -> list:
        S = self.styles
        text = (
            "<b>DISCLAIMER:</b> This report is generated by an AI research prototype "
            "(NeuroVision AI) and is intended for research and educational purposes ONLY. "
            "It must NOT be used as a substitute for professional medical diagnosis or "
            "clinical decision-making. Always consult a qualified medical professional. "
            "The predictions are based on a machine learning model trained on limited data "
            "and may contain errors."
        )
        return [
            HRFlowable(width="100%", thickness=0.5, color=_C_BORDER, spaceAfter=4),
            Paragraph(text, S["disclaimer"]),
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _section(self, title: str) -> list:
        return [
            Paragraph(title, self.styles["section"]),
            HRFlowable(width="100%", thickness=0.5, color=_C_BORDER, spaceAfter=4),
        ]

    def _kv_table(self, rows: list[tuple[str, str]]) -> Table:
        S = self.styles
        data = [[Paragraph(k, S["label"]), Paragraph(str(v), S["value"])]
                for k, v in rows]
        tbl  = Table(data, colWidths=[3.5*cm, None])
        tbl.setStyle(TableStyle([
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
            ("TOPPADDING",  (0,0),(-1,-1), 3),
        ]))
        return tbl

    @staticmethod
    def _b64_image(b64: str, max_size: float) -> Image | None:
        if not b64:
            return None
        try:
            raw = base64.b64decode(b64)
            return Image(io.BytesIO(raw), width=max_size, height=max_size, kind="proportional")
        except Exception:
            return None
