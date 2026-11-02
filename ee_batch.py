"""Batch AlphaEarth fetch for many points (labeled examples) and the feature vector used by the lookalike model."""
import numpy as np

from ee_data import DW_CLASSES, EMB_BANDS, YEARS, demo, ee_ready, norm_spec

FEAT_VERSION = 2
W_SCAL = 3.0   # weight of the trajectory scalars block
W_LC = 0.5     # weight of the land cover block
FEAT_DIM = 64 * 4 + 4 + 18


def _u(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


def lc_to_vec(d):
    """{class_name: fraction} -> 9 vector in DW_CLASSES order."""
    d = d or {}
    return np.array([d.get(c, 0.0) for c in DW_CLASSES], float)


def _vec(props):
    v = [props.get(b) for b in EMB_BANDS]
    return None if any(x is None for x in v) else v


def _fc(points, radius):
    import ee
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lo, la]).buffer(radius), {"i": i}) for i, (la, lo) in enumerate(points)
    ])


def emb_points(points, years=YEARS, radius=50):
    """points: [(lat, lon)]. Returns [{year: 64-vector or None}] per point. One EE call per year."""
    if not ee_ready():
        return [{y: demo(norm_spec(la, lo, radius))["emb"][y] for y in years} for la, lo in points]
    import ee
    fc = _fc(points, radius)
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    out = [dict() for _ in points]
    for y in years:
        img = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().select(EMB_BANDS)
        for f in img.reduceRegions(fc, ee.Reducer.mean(), 10).getInfo()["features"]:
            out[f["properties"]["i"]][y] = _vec(f["properties"])
    return out


def lc_points(points, years=(YEARS[0], YEARS[-1]), radius=50):
    """Dynamic World class shares for first and last year per point: [{year: 9 vector}]."""
    if not ee_ready():
        return [{y: lc_to_vec(demo(norm_spec(la, lo, radius))["lc"][y]) for y in years} for la, lo in points]
    import ee
    fc = _fc(points, radius)
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    out = [dict() for _ in points]
    for y in years:
        mode = dw.filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("label").mode()
        for f in mode.reduceRegions(fc, ee.Reducer.frequencyHistogram(), 10).getInfo()["features"]:
            p = f["properties"]
            h = p.get("histogram") or p.get("label") or {}
            parsed = {}
            for k, v in h.items():
                try:
                    idx = int(float(k))
                    if 0 <= idx < len(DW_CLASSES):
                        parsed[DW_CLASSES[idx]] = parsed.get(DW_CLASSES[idx], 0.0) + float(v)
                except (ValueError, TypeError):
                    continue
            tot = sum(parsed.values()) or 1.0
            out[p["i"]][y] = lc_to_vec({k: v / tot for k, v in parsed.items()})
    return out


def feature_vec(emb, lc=None):
    """278 dim model feature from all 8 yearly embeddings (+ optional first/last land cover shares).

    Blocks are unit normalised separately so the big change vector cannot drown the baseline:
    [2017 state | 2024 state | change | mid decade state | trajectory scalars | land cover]
    The scalars (peak year over year jump, when it happened, variance, path length) catch plots that
    flooded or were cleared mid decade and then stabilised."""
    rows = [emb.get(y) for y in YEARS]
    if any(r is None or any(x is None for x in r) for r in rows):
        return None
    M = np.array(rows, float)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    a, b = M[0], M[-1]
    vel = np.array([1 - M[i] @ M[i + 1] for i in range(len(M) - 1)])
    scal = np.array([vel.max(), vel.argmax() / (len(vel) - 1), M.var(axis=0).sum(), vel.sum()])
    mid = M[2:6].mean(axis=0)
    lcv = np.zeros(18) if lc is None else np.concatenate([lc[0], lc[1]])
    return np.concatenate([a, b, _u(b - a), _u(mid), scal * W_SCAL, lcv * W_LC]).tolist()


def feature_from_raw(raw):
    lc = (lc_to_vec(raw["lc"].get(YEARS[0])), lc_to_vec(raw["lc"].get(YEARS[-1])))
    return feature_vec(raw["emb"], lc)


def feature_from_points(points):
    """Features for [(lat, lon)] spots, same recipe as feature_from_raw. List of vectors or None."""
    embs, lcs = emb_points(points), lc_points(points)
    return [feature_vec(e, (l[YEARS[0]], l[YEARS[-1]]) if len(l) == 2 else None) for e, l in zip(embs, lcs)]
