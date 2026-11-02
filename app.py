"""FastAPI server. Run: uvicorn app:app --reload"""
import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ee_data
import logic
import model
import pdf
from ee_batch import feature_from_raw

app = FastAPI(title="LandScan")
HERE = os.path.dirname(__file__)


class TTL:
    """Tiny thread safe LRU with expiry. Used to avoid repeat Earth Engine round trips."""

    def __init__(self, size=256, ttl=3600):
        self.d, self.size, self.ttl, self.lock = OrderedDict(), size, ttl, threading.Lock()

    def get(self, k):
        with self.lock:
            v = self.d.get(k)
            if not v or time.time() - v[0] > self.ttl:
                self.d.pop(k, None)
                return None
            self.d.move_to_end(k)
            return v[1]

    def put(self, k, val):
        with self.lock:
            self.d[k] = (time.time(), val)
            self.d.move_to_end(k)
            while len(self.d) > self.size:
                self.d.popitem(last=False)


RAW, THUMBS = TTL(256, 3600), TTL(64, 3600)


class PlotIn(BaseModel):
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    radius: float = Field(100, ge=20, le=1000)
    polygon: Optional[List[List[float]]] = None  # [[lon, lat], ...]


class LabelIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    label: str


class LabelUpdate(BaseModel):
    label: str


def spec_of(p: PlotIn):
    try:
        return ee_data.norm_spec(p.lat, p.lon, p.radius, p.polygon)
    except (ValueError, TypeError, IndexError) as e:
        raise HTTPException(422, str(e))


def rid_of(spec):
    return hashlib.sha1(ee_data.spec_key(spec).encode()).hexdigest()[:16]


def get_raw(spec):
    """Earth Engine result for this plot, cached so /report and /report.pdf share ONE fetch."""
    rid = rid_of(spec)
    hit = RAW.get(rid)
    if hit is None:
        hit = {"spec": spec, "raw": ee_data.fetch(spec)}
        RAW.put(rid, hit)
    return rid, hit


def make_report(spec):
    rid, hit = get_raw(spec)
    raw = hit["raw"]
    rep = logic.build(raw)
    rep = model.apply(rep, model.predict(feature_from_raw(raw)))
    rep["rid"] = rid
    return rep


def safe(fn, *a):
    try:
        return fn(*a)
    except HTTPException:
        raise
    except Exception as e:  # noqa
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "ee_ready": ee_data.ee_ready()}


@app.get("/api/report")
def report_get(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
               radius: float = Query(100, ge=20, le=1000)):
    return safe(make_report, spec_of(PlotIn(lat=lat, lon=lon, radius=radius)))


@app.post("/api/report")
def report_post(p: PlotIn):
    return safe(make_report, spec_of(p))


@app.get("/api/report.pdf")
def report_pdf(rid: Optional[str] = None, lat: Optional[float] = Query(None, ge=-90, le=90),
               lon: Optional[float] = Query(None, ge=-180, le=180), radius: float = Query(100, ge=20, le=1000),
               label: str = ""):
    hit = RAW.get(rid) if rid else None
    if hit:
        spec = hit["spec"]
    elif lat is not None and lon is not None:
        spec = spec_of(PlotIn(lat=lat, lon=lon, radius=radius))
    else:
        raise HTTPException(410, "Report expired. Run the check again, then download.")

    def build():
        rep = make_report(spec)  # reuses cached EE data when fresh
        key = rid_of(spec)
        th = THUMBS.get(key)
        if th is None:
            th = ee_data.thumbs(spec)
            THUMBS.put(key, th)
        return pdf.make(rep, label, th)

    data = safe(build)
    return Response(data, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="landscan_{spec["lat"]:.4f}_{spec["lon"]:.4f}.pdf"'})


# ---------- labels ----------

@app.get("/api/labels")
def labels():
    rows = model.all_rows()
    return {"counts": model.counts(), "classes": {k: v[0] for k, v in model.CLASSES.items()},
            "points": [{k: r[k] for k in ("id", "lat", "lon", "label")} for r in rows],
            "stale": model.stale_count(), "ee": ee_data.ee_ready()}


@app.post("/api/label")
def label_add(b: LabelIn):
    try:
        lid = model.add_label(b.lat, b.lon, b.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": lid, "counts": model.counts()}


@app.put("/api/label/{lid}")
def label_update(lid: int, b: LabelUpdate):
    try:
        ok = model.update_label(lid, b.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "no such label")
    return {"counts": model.counts()}


@app.delete("/api/label/{lid}")
def label_delete(lid: int):
    if not model.delete_label(lid):
        raise HTTPException(404, "no such label")
    return {"counts": model.counts()}


@app.get("/api/model-metrics")
def model_metrics():
    return model.metrics()


@app.post("/api/reindex")
def reindex():
    return safe(model.reindex)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
