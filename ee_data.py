"""Earth Engine data fetch for one plot. Falls back to demo data if EE is not set up."""
import os
import random

YEARS = list(range(2017, 2025))
DW_CLASSES = ["water", "trees", "grass", "flooded_veg", "crops", "shrub", "built", "bare", "snow"]
EMB_BANDS = [f"A{i:02d}" for i in range(64)]

_ready = None


def ee_ready():
    """Init Earth Engine once. Needs `earthengine authenticate` and EE_PROJECT env var."""
    global _ready
    if _ready is not None:
        return _ready
    try:
        import ee
        ee.Initialize(project=os.environ.get("EE_PROJECT"))
        _ready = True
    except Exception as e:  # noqa
        print("Earth Engine not available, using DEMO data:", e)
        _ready = False
    return _ready


def _vec(props):
    if not props:
        return None
    v = [props.get(b) for b in EMB_BANDS]
    return None if any(x is None for x in v) else v


def _parse_hist(h):
    """Safely parse Earth Engine frequencyHistogram keys (handles '6' or '6.0')."""
    out = {}
    if not h:
        return out
    for k, v in h.items():
        try:
            idx = int(float(k))
            if 0 <= idx < len(DW_CLASSES):
                out[idx] = out.get(idx, 0.0) + float(v)
        except (ValueError, TypeError):
            continue
    return out


def fetch(lat, lon, radius=100, ring=1000):
    """Return raw data dict for a plot. Uses real EE data when available."""
    if not ee_ready():
        return demo(lat, lon)
    import ee

    pt = ee.Geometry.Point([lon, lat])
    plot = pt.buffer(radius)
    around = pt.buffer(ring)

    emb_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    dw_col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")

    def _per_year(y):
        y = ee.Number(y)
        s = ee.Date.fromYMD(y, 1, 1)
        e = s.advance(1, "year")
        img = emb_col.filterDate(s, e).filterBounds(pt).mosaic().select(EMB_BANDS)
        v = img.reduceRegion(ee.Reducer.mean(), plot, 10, maxPixels=1e8)

        mode = dw_col.filterDate(s, e).filterBounds(pt).select("label").mode()
        hist = mode.reduceRegion(ee.Reducer.frequencyHistogram(), plot, 10, maxPixels=1e8)
        h2 = mode.reduceRegion(ee.Reducer.frequencyHistogram(), around, 30, maxPixels=1e8)
        return ee.Feature(None, {
            "y": y,
            "emb": v,
            "hist": hist.get("label"),
            "h2": h2.get("label"),
        })

    yearly_fc = ee.FeatureCollection(ee.List(YEARS).map(_per_year))

    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    near_water = gsw.reduceRegion(ee.Reducer.max(), pt.buffer(300), 30, maxPixels=1e8).get("occurrence")
    dem = ee.Image("USGS/SRTMGL1_003")
    elev = dem.reduceRegion(ee.Reducer.mean(), plot, 30).get("elevation")
    elev_ring = dem.reduceRegion(ee.Reducer.mean(), around, 30, maxPixels=1e8).get("elevation")
    slope = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), plot, 30).get("slope")

    bundle = ee.Dictionary({
        "years": yearly_fc.toList(len(YEARS)),
        "water_occ": near_water,
        "elev": elev,
        "elev_ring": elev_ring,
        "slope": slope,
    }).getInfo()

    emb, lc, nb = {}, {}, {}
    for feat in bundle.get("years", []):
        props = feat.get("properties", {})
        y = int(props["y"])
        emb[y] = _vec(props.get("emb") or {})

        h_parsed = _parse_hist(props.get("hist") or {})
        tot = sum(h_parsed.values()) or 1.0
        lc[y] = {DW_CLASSES[idx]: val / tot for idx, val in h_parsed.items()}

        h2_parsed = _parse_hist(props.get("h2") or {})
        t2 = sum(h2_parsed.values()) or 1.0
        nb[y] = h2_parsed.get(6, 0.0) / t2

    return {
        "demo": False, "lat": lat, "lon": lon, "radius": radius,
        "emb": emb, "lc": lc, "nb_built": nb,
        "water_occ": bundle.get("water_occ") or 0,
        "elev": bundle.get("elev"),
        "elev_ring": bundle.get("elev_ring"),
        "slope": bundle.get("slope"),
    }


def demo(lat, lon):
    """Deterministic fake data so the UI and PDF can be tested without EE."""
    rnd = random.Random(int(abs(lat * 1000) + abs(lon * 1000)))
    base = [rnd.gauss(0, 1) for _ in range(64)]
    emb, lc, nb = {}, {}, {}
    switch = rnd.choice([2020, 2021, 2022])
    for y in YEARS:
        shift = 0 if y < switch else 1.4
        emb[y] = [b + shift * (1 if i % 3 == 0 else 0.2) + rnd.gauss(0, 0.05) for i, b in enumerate(base)]
        lc[y] = {"crops": 0.9, "grass": 0.1} if y < switch else {"built": 0.7, "bare": 0.2, "grass": 0.1}
        nb[y] = min(0.6, 0.08 + 0.06 * (y - 2017))
    return {
        "demo": True, "lat": lat, "lon": lon, "radius": 100,
        "emb": emb, "lc": lc, "nb_built": nb,
        "water_occ": rnd.choice([0, 0, 12, 55]), "elev": 540.0, "elev_ring": rnd.choice([548.0, 541.0]),
        "slope": 2.1,
    }
