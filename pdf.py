"""Due diligence PDF report with before / after satellite thumbnails."""
import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

COL = {"HIGH": "#B3261E", "MEDIUM": "#B26A00", "LOW": "#1B7F3B"}
LVL = {"high": "#B3261E", "medium": "#B26A00", "info": "#37474F"}
GRID = colors.HexColor("#B0BEC5")
HEAD = colors.HexColor("#ECEFF1")


def _f(v, unit="", nd=0):
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def _table(rows, widths=None, size=8):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), size), ("GRID", (0, 0), (-1, -1), 0.25, GRID),
                           ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def make(rep, label="", thumbs=None):
    thumbs = thumbs or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title="Land Due Diligence Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=4)
    p = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.5, leading=13)
    sm = ParagraphStyle("sm", parent=p, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))

    kind = "Drawn boundary" if rep.get("polygon") else f"Pin, {rep.get('radius') or 100:.0f} m radius"
    s = [Paragraph("Land Due Diligence Report", h1),
         Paragraph(f"{label + ' | ' if label else ''}Lat {rep['lat']:.5f}, Lon {rep['lon']:.5f} | {kind} | "
                   f"{rep['area_m2']:,.0f} m2 ({rep['area_acres']:.2f} acre) | {date.today():%d %b %Y}", sm)]
    if rep["demo"]:
        s.append(Paragraph("<font color='#B3261E'><b>DEMO DATA, not a real analysis.</b></font>", p))

    vc = COL[rep["verdict"]]
    s += [Spacer(1, 6), Paragraph(
        f"<font size=16 color='{vc}'><b>Screening risk: {rep['verdict']} ({rep['risk']}/100)</b></font>"
        f"&nbsp;&nbsp; Area growth: <b>{rep['growth']}</b>", p)]

    # before / after thumbnails
    ys = sorted(thumbs)
    if ys:
        cells, caps = [], []
        for y in ys:
            cells.append(Image(io.BytesIO(thumbs[y]), width=82 * mm, height=82 * mm))
            caps.append(Paragraph(f"<b>{y}</b> Sentinel-2, plot outlined in red", sm))
        t = Table([cells, caps], colWidths=[87 * mm] * len(ys))
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        s += [Paragraph("Then and now", h2), t]

    s.append(Paragraph("Flags", h2))
    for f in rep["flags"]:
        s.append(KeepTogether([Paragraph(f"<font color='{LVL[f['level']]}'><b>{f['level'].upper()}</b></font> "
                                         f"<b>{f['title']}</b><br/>{f['detail']}", p), Spacer(1, 3)]))

    ev = rep.get("evidence") or {}
    w = ev.get("water") or {}
    sar = ev.get("sar") or {}
    rows = [["Evidence", "Value"],
            ["Plot size", f"{rep['area_m2']:,.0f} m2, about {ev.get('pixels', 0)} satellite pixels (10 m)"],
            ["Surface water seen (% of time): on plot / 30 m / 300 m",
             f"{_f(w.get('on'), '%')} / {_f(w.get('m30'), '%')} / {_f(w.get('m300'), '%')}"],
            ["Height above nearest drainage (HAND)", _f(ev.get("hand"), " m", 1)],
            ["Radar wet share: monsoon / dry season",
             f"{_f(sar.get('mon') and sar['mon'] * 100, '%')} / {_f(sar.get('dry') and sar['dry'] * 100, '%')}"
             if sar else "n/a"],
            ["Slope", _f(ev.get("slope"), " deg", 1)],
            ["Below / above 1 km average", _f(ev.get("below_surroundings_m"), " m below", 1)]]
    s.append(KeepTogether([Paragraph("Evidence", h2), _table(rows, [85 * mm, 89 * mm], 8.5)]))

    s.append(KeepTogether([Paragraph("Land cover, 2017 to 2024", h2),
                           _table([["Year"] + [str(t["year"]) for t in rep["timeline"]],
                                   ["Cover"] + [t["cover"] for t in rep["timeline"]]], size=7.5)]))

    s.append(KeepTogether([
        Paragraph("Year over year change (AlphaEarth embeddings)", h2),
        _table([["Period"] + [f"{x['from']}-{str(x['to'])[2:]}" for x in rep["steps"]],
                ["Change"] + [("-" if x["change"] is None else f"{x['change']:.2f}") + ("*" if x["big"] else "")
                              for x in rep["steps"]]], size=7.5),
        Paragraph("* unusually large change versus this plot's normal year to year drift", sm)]))

    look = rep.get("lookalike")
    if look:
        s += [Paragraph("Lookalike scores", h2),
              Paragraph(", ".join(f"{k}: {v:.2f}" for k, v in look["scores"].items()) +
                        f" (flag threshold {look['thresholds']['sim']:.2f}"
                        f"{', calibrated' if look['thresholds']['calibrated'] else ', default'})", p)]

    s += [Spacer(1, 10), Paragraph(
        "Screening tool built on satellite data at 10 m resolution (Google DeepMind AlphaEarth Foundations "
        "embeddings, Dynamic World, JRC Global Surface Water, Sentinel-1, MERIT Hydro, SRTM, Sentinel-2). "
        "It does not replace a title search, survey, legal verification, or a site visit. "
        "Small plots can be noisy and water history is not a legal boundary.", sm)]
    doc.build(s)
    return buf.getvalue()
