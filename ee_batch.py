"""Batch AlphaEarth embedding fetch for many points (used for labeled examples and area scans)."""
import numpy as np

from ee_data import EMB_BANDS, YEARS, ee_ready, demo


def _vec(props):
    if not props:
        return None
    v = [props.get(b) for b in EMB_BANDS]
    return None if any(x is None for x in v) else v


def emb_points(points, years=(YEARS[0], YEARS[-1]), radius=100):
    """points: list of (lat, lon). Returns list of {year: 64-vector or None} per point.
    One Earth Engine call per year for the whole batch."""
    if not ee_ready():
        return [{y: demo(la, lo)["emb"][y] for y in years} for la, lo in points]
    import ee

    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lo, la]).buffer(radius), {"i": i}) for i, (la, lo) in enumerate(points)
    ])
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    out = [{y: None for y in years} for _ in points]
    for y in years:
        img = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().select(EMB_BANDS)
        feats = img.reduceRegions(fc, ee.Reducer.mean(), 10).getInfo()["features"]
        for f in feats:
            out[f["properties"]["i"]][y] = _vec(f["properties"])
    return out


def feature_vec(e):
    """Model features from first and last year embeddings: [first, last, last-first]."""
    if not e:
        return None
    a, b = e.get(YEARS[0]), e.get(YEARS[-1])
    if a is None or b is None or any(x is None for x in a) or any(x is None for x in b):
        return None
    a, b = np.array(a, float), np.array(b, float)
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    return np.concatenate([a, b, (b - a)]).tolist()
