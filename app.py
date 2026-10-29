"""FastAPI server. Run: uvicorn app:app --reload"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ee_data
import logic
import model
import pdf
from ee_batch import emb_points, feature_vec

app = FastAPI(title="LandScan")
HERE = os.path.dirname(__file__)


class LabelIn(BaseModel):
    lat: float
    lon: float
    label: str


def make_report(lat, lon):
    raw = ee_data.fetch(lat, lon)
    rep = logic.build(raw)
    fv = feature_vec(raw["emb"])
    return model.apply(rep, model.predict(fv))


@app.get("/api/report")
def report(lat: float, lon: float):
    try:
        return make_report(lat, lon)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/report.pdf")
def report_pdf(lat: float, lon: float, label: str = ""):
    data = pdf.make(make_report(lat, lon), label)
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="landscan_{lat:.4f}_{lon:.4f}.pdf"'})


@app.get("/api/labels")
def labels():
    return {"counts": model.counts(), "classes": {k: v[0] for k, v in model.CLASSES.items()},
            "points": [{k: r[k] for k in ("lat", "lon", "label")} for r in model.load()],
            "ee": ee_data.ee_ready()}


@app.post("/api/label")
def label(b: LabelIn):
    try:
        return {"counts": model.add_label(b.lat, b.lon, b.label)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
