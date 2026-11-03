"""PDF report generation for a persisted inspection and its safe assets."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


CYAN = colors.HexColor("#00AEB8")
DARK = colors.HexColor("#142126")
MUTED = colors.HexColor("#56666D")
PALE = colors.HexColor("#EAF3F4")


def build_inspection_pdf(record: dict[str, Any], assets: dict[str, Path | None]) -> bytes:
    """Build a compact downloadable report with metadata and available visuals."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"DefectZero Inspection {record['inspection_id']}",
        author="DefectZero AI",
    )
    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DefectZeroTitle",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        alignment=TA_CENTER,
        textColor=DARK,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DefectZeroSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "DefectZeroSection",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=DARK,
        spaceBefore=8,
        spaceAfter=7,
    )
    label_style = ParagraphStyle(
        "DefectZeroLabel",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=MUTED,
    )
    value_style = ParagraphStyle(
        "DefectZeroValue",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=DARK,
        wordWrap="CJK",
    )
    note_style = ParagraphStyle(
        "DefectZeroNote",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=MUTED,
    )

    timestamp = _format_timestamp(str(record["timestamp_utc"]))
    category = escape(str(record["category"]).replace("_", " "))
    filename = escape(str(record["filename"]))
    inspection_id = escape(str(record["inspection_id"]))

    story: list[Any] = [
        Paragraph("DefectZero AI", title_style),
        Paragraph("Zero-Shot Industrial Inspection Report", subtitle_style),
        _section_rule("INSPECTION SUMMARY", section_style),
    ]

    rows = [
        [Paragraph("Category", label_style), Paragraph(category, value_style), Paragraph("Filename", label_style), Paragraph(filename, value_style)],
        [Paragraph("Predicted label", label_style), Paragraph(escape(str(record["predicted_label"]).title()), value_style), Paragraph("Timestamp (UTC)", label_style), Paragraph(escape(timestamp), value_style)],
        [Paragraph("Anomaly score", label_style), Paragraph(f"{float(record['anomaly_score']):.4f}", value_style), Paragraph("Threshold", label_style), Paragraph(f"{float(record['threshold']):.4f}", value_style)],
        [Paragraph("Inference time", label_style), Paragraph(f"{float(record['inference_time_ms']) / 1000:.2f} s", value_style), Paragraph("Inspection ID", label_style), Paragraph(inspection_id, value_style)],
    ]
    summary = Table(rows, colWidths=[28 * mm, 47 * mm, 30 * mm, 68 * mm], hAlign="LEFT")
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D4E1E3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4E1E3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        summary,
        Spacer(1, 8),
        Paragraph(
            "Classification uses a score threshold. The anomaly score is an inspection signal, not a probability. "
            "This report describes pretrained WinCLIP zero-shot inference (k-shot = 0).",
            note_style,
        ),
        _section_rule("INSPECTION VISUALS", section_style),
    ])

    visual_specs = [
        ("Input image", assets.get("image")),
        ("Anomaly heatmap", assets.get("heatmap")),
        ("Ground-truth mask", assets.get("ground_truth")),
    ]
    visual_specs = [(label, path) for label, path in visual_specs if path is not None]
    max_width = (document.width - 8 * mm) / 2
    max_height = 70 * mm
    visual_cells = []
    for label, path in visual_specs:
        image = _fit_image(path, max_width, max_height)
        cell = [Paragraph(label, ParagraphStyle(
            f"VisualLabel{len(visual_cells)}",
            parent=label_style,
            textColor=DARK,
            fontSize=9,
            spaceAfter=4,
        )), image]
        visual_cells.append(cell)

    for index in range(0, len(visual_cells), 2):
        first = visual_cells[index]
        second = visual_cells[index + 1] if index + 1 < len(visual_cells) else []
        visual_table = Table([[first, second]], colWidths=[max_width + 4 * mm, max_width + 4 * mm], hAlign="LEFT")
        visual_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(visual_table)

    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output.getvalue()


def build_batch_summary_pdf(records: list[dict[str, Any]]) -> bytes:
    """Build a compact batch summary for already persisted inspection records."""
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=17 * mm, bottomMargin=17 * mm, title="DefectZero Batch Inspection Summary", author="DefectZero AI")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("BatchTitle", parent=styles["Title"], fontName="Helvetica-Bold", textColor=DARK, fontSize=20, leading=24, alignment=TA_CENTER)
    note_style = ParagraphStyle("BatchNote", parent=styles["Normal"], fontName="Helvetica-Oblique", textColor=MUTED, fontSize=8, leading=11)
    heading_style = ParagraphStyle("BatchHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", textColor=DARK, fontSize=11, leading=14, spaceBefore=10, spaceAfter=7)
    rows: list[list[Any]] = [["Filename", "Category", "Prediction", "Score", "Inference", "Timestamp (UTC)"]]
    for record in records:
        rows.append([
            Paragraph(escape(str(record["filename"])), styles["BodyText"]),
            escape(str(record["category"]).replace("_", " ")),
            str(record["predicted_label"]).title(),
            f"{float(record['anomaly_score']):.4f}",
            f"{float(record['inference_time_ms']) / 1000:.2f} s",
            _format_timestamp(str(record["timestamp_utc"])),
        ])
    table = Table(rows, repeatRows=1, colWidths=[43 * mm, 24 * mm, 24 * mm, 20 * mm, 23 * mm, 45 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 1), (-1, -1), PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4E1E3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story = [Paragraph("DefectZero AI", title_style), Paragraph("Batch Inspection Summary · Pretrained WinCLIP · Zero-Shot", note_style), Paragraph(f"{len(records)} saved inspection(s)", heading_style), table, Spacer(1, 8), Paragraph("Each image was submitted through the standard single-image inspection workflow. Anomaly scores are inspection signals, not probabilities.", note_style)]
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output.getvalue()


def _section_rule(text: str, style: ParagraphStyle) -> Table:
    rule = Table([[Paragraph(text, style)]], colWidths=[176 * mm])
    rule.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1, CYAN),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return rule


def _fit_image(path: Path, max_width: float, max_height: float) -> Image:
    width, height = ImageReader(str(path)).getSize()
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale, kind="proportional")


def _format_timestamp(value: str) -> str:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _draw_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D4E1E3"))
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, 12 * mm, A4[0] - document.rightMargin, 12 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(document.leftMargin, 8 * mm, "DefectZero AI · Pretrained WinCLIP · Zero-Shot")
    canvas.drawRightString(A4[0] - document.rightMargin, 8 * mm, f"Page {document.page}")
    canvas.restoreState()
