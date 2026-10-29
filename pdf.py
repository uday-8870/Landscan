"""One page PDF due diligence report."""
import io
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

COL = {"HIGH": "#B3261E", "MEDIUM": "#B26A00", "LOW": "#1B7F3B"}
LVL = {"high": "#B3261E", "medium": "#B26A00", "info": "#37474F"}


def make(rep, label=""):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="Land Due Diligence Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    p = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.5, leading=13)
    sm = ParagraphStyle("sm", parent=p, fontSize=8, textColor=colors.HexColor("#555555"))

    safe_label = escape(label.strip()) if label else ""
    s = [Paragraph("Land Due Diligence Report", h1),
         Paragraph(f"{safe_label + ' | ' if safe_label else ''}Lat {rep['lat']:.5f}, Lon {rep['lon']:.5f} | {date.today():%d %b %Y}", sm)]
    if rep["demo"]:
        s.append(Paragraph("<font color='#B3261E'><b>DEMO DATA, not a real analysis.</b></font>", p))

    vc = COL[rep["verdict"]]
    s += [Spacer(1, 6), Paragraph(
        f"<font size=16 color='{vc}'><b>Screening risk: {rep['verdict']} ({rep['risk']}/100)</b></font>"
        f"&nbsp;&nbsp; Area growth: <b>{escape(str(rep['growth']))}</b>", p)]

    s.append(Paragraph("Flags", h2))
    for f in rep["flags"]:
        s.append(Paragraph(f"<font color='{LVL[f['level']]}'><b>{escape(f['level'].upper())}</b></font> "
                           f"<b>{escape(f['title'])}</b><br/>{escape(f['detail'])}", p))
        s.append(Spacer(1, 3))

    s.append(Paragraph("Land cover, 2017 to 2024", h2))
    rows = [["Year"] + [str(t["year"]) for t in rep["timeline"]],
            ["Cover"] + [t["cover"] for t in rep["timeline"]]]
    t = Table(rows)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF1"))]))
    s.append(t)

    s.append(Paragraph("Year over year change (AlphaEarth embeddings)", h2))
    rows = [["Period"] + [f"{x['from']}-{str(x['to'])[2:]}" for x in rep["steps"]],
            ["Change"] + [("-" if x["change"] is None else f"{x['change']:.2f}") + ("*" if x["big"] else "")
                          for x in rep["steps"]]]
    t = Table(rows)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF1"))]))
    s += [t, Paragraph("* unusually large change versus this plot's normal year to year drift", sm)]

    look = rep.get("lookalike")
    if look:
        s.append(Paragraph("Lookalike scores", h2))
        s.append(Paragraph(", ".join(f"{escape(str(k))}: {v:.2f}" for k, v in look["scores"].items()), p))

    s += [Spacer(1, 10), Paragraph(
        "Screening tool built on satellite data at 10 m resolution (Google DeepMind AlphaEarth Foundations "
        "embeddings, Dynamic World, JRC Global Surface Water, SRTM). It does not replace a title search, "
        "survey and legal verification, or a site visit. Small plots can be noisy.", sm)]
    doc.build(s)
    return buf.getvalue()
