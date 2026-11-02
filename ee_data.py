"""Earth Engine data fetch for one plot (circle or polygon). Falls back to DEMO data if EE is not set up."""
import json
import math
import os
import random
import urllib.request

YEARS = list(range(2017, 2025))
DW_CLASSES = ["water", "trees", "grass", "flooded_veg", "crops", "shrub", "built", "bare", "snow"]
EMB_BANDS = [f"A{i:02d}" for i in range(64)]
SAR_WET_DB = -18.0          # Sentinel-1 VV below this = open water / standing water
MAX_AREA_M2 = 5_000_000     # reject giant polygons (5 km2)

_ready = None


def ee_ready():
    """Init Earth Engine once. Supports cloud Service Account JSON, local key file, local auth, or DEMO fallback."""
    global _ready
    if _ready is not None:
        return _ready
    try:
        import base64
        import glob
        import ee

        sa_raw = os.environ.get("EE_SERVICE_ACCOUNT_JSON", "").strip()
        project = os.environ.get("EE_PROJECT", "landscan-513009").strip() or "landscan-513009"

        if not sa_raw:
            here = os.path.dirname(__file__)
            matches = glob.glob(os.path.join(here, "landscan-513009-*.json")) + glob.glob(os.path.join(here, "ee-key.json"))
            if matches:
                with open(matches[0]) as f:
                    sa_raw = f.read().strip()

        if sa_raw:
            if not sa_raw.startswith("{"):
                sa_raw = base64.b64decode(sa_raw).decode("utf-8")
            info = json.loads(sa_raw)
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=sa_raw)
            ee.Initialize(credentials=creds, project=project or info.get("project_id"))
        else:
            ee.Initialize(project=project)
        _ready = True
    except Exception as e:  # noqa
        print("Earth Engine not available, using DEMO data:", e)
        _ready = False
    return _ready


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


# ---------- plot spec (circle or polygon) ----------

def area_m2(spec):
    if not spec.get("polygon"):
        return math.pi * spec["radius"] ** 2
    ring = spec["polygon"]
    lat0, lon0 = spec["lat"], spec["lon"]
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540
    pts = [((x - lon0) * kx, (y - lat0) * ky) for x, y in ring]
    s = sum(pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1] for i in range(len(pts) - 1))
    return abs(s) / 2


def norm_spec(lat=None, lon=None, radius=100, polygon=None):
    """Validate and normalise a plot. polygon is [[lon, lat], ...]."""
    if polygon:
        ring = [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in polygon]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            raise ValueError("polygon needs at least 3 points")
        for x, y in ring:
            if not (-180 <= x <= 180 and -90 <= y <= 90):
                raise ValueError("polygon coordinates must be [lon, lat]")
        pts = ring[:-1]
        spec = {"polygon": ring, "radius": None,
                "lon": round(sum(p[0] for p in pts) / len(pts), 6),
                "lat": round(sum(p[1] for p in pts) / len(pts), 6)}
    else:
        if lat is None or lon is None:
            raise ValueError("lat/lon or polygon required")
        spec = {"polygon": None, "lat": round(float(lat), 5), "lon": round(float(lon), 5), "radius": float(radius)}
    spec["area_m2"] = area_m2(spec)
    if spec["area_m2"] > MAX_AREA_M2:
        raise ValueError(f"plot too large ({spec['area_m2'] / 1e6:.1f} km2, max {MAX_AREA_M2 / 1e6:.0f} km2)")
    if spec["area_m2"] < 1:
        raise ValueError("plot has no area")
    return spec


def spec_key(spec):
    return json.dumps({k: spec[k] for k in ("polygon", "lat", "lon", "radius")}, sort_keys=True)


def plot_geom(spec):
    import ee
    if spec.get("polygon"):
        return ee.Geometry.Polygon([spec["polygon"]])
    return ee.Geometry.Point([spec["lon"], spec["lat"]]).buffer(spec["radius"])


# ---------- main fetch ----------

def fetch(spec):
    """Return raw data dict for a plot. Uses real EE data when available."""
    if not ee_ready():
        return demo(spec)
    import ee

    plot = plot_geom(spec)
    pt = ee.Geometry.Point([spec["lon"], spec["lat"]])
    around = pt.buffer(1000)
    emb_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    dw_col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    mean, mx = ee.Reducer.mean(), ee.Reducer.max()

    # everything in ONE server-side dictionary => one round trip
    q = {"area": plot.area(1)}
    for y in YEARS:
        s, e = f"{y}-01-01", f"{y + 1}-01-01"
        img = emb_col.filterDate(s, e).filterBounds(pt).mosaic().select(EMB_BANDS)
        q[f"emb{y}"] = img.reduceRegion(mean, plot, 10, maxPixels=1e8)
        mode = dw_col.filterDate(s, e).filterBounds(pt).select("label").mode()
        q[f"lc{y}"] = mode.reduceRegion(ee.Reducer.frequencyHistogram(), plot, 10, maxPixels=1e8)
        q[f"nb{y}"] = mode.reduceRegion(ee.Reducer.frequencyHistogram(), around, 30, maxPixels=1e8)

    # water occurrence (%) in rings: on plot, 30 m (buffer zone), 300 m (FTL proximity)
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    q["w_on"] = gsw.reduceRegion(mx, plot, 30, maxPixels=1e8)
    q["w_30"] = gsw.reduceRegion(mx, plot.buffer(30), 30, maxPixels=1e8)
    q["w_300"] = gsw.reduceRegion(mx, plot.buffer(300), 30, maxPixels=1e8)

    dem = ee.Image("USGS/SRTMGL1_003")
    q["elev"] = dem.reduceRegion(mean, plot, 30, maxPixels=1e8)
    q["elev_ring"] = dem.reduceRegion(mean, around, 30, maxPixels=1e8)
    q["slope"] = ee.Terrain.slope(dem).reduceRegion(mean, plot, 30, maxPixels=1e8)

    r = ee.Dictionary(q).getInfo()
    g = lambda d, k: (r.get(d) or {}).get(k)  # noqa

    emb = {y: [(r[f"emb{y}"] or {}).get(b) for b in EMB_BANDS] for y in YEARS}
    lc, nb = {}, {}
    for y in YEARS:
        h_parsed = _parse_hist((r[f"lc{y}"] or {}).get("label") or {})
        tot = sum(h_parsed.values()) or 1.0
        lc[y] = {DW_CLASSES[idx]: val / tot for idx, val in h_parsed.items()}
        h2_parsed = _parse_hist((r[f"nb{y}"] or {}).get("label") or {})
        nb[y] = h2_parsed.get(6, 0.0) / (sum(h2_parsed.values()) or 1.0)

    notes = []
    sar = hand = None
    try:  # optional layers: never let them break the core report
        sar, hand = _extras(plot)
    except Exception as e:  # noqa
        notes.append(f"Radar / HAND flood layers unavailable ({type(e).__name__}).")

    return {
        "demo": False, "lat": spec["lat"], "lon": spec["lon"], "radius": spec["radius"],
        "polygon": spec["polygon"], "area_m2": r.get("area") or spec["area_m2"],
        "emb": emb, "lc": lc, "nb_built": nb,
        "water": {"on": g("w_on", "occurrence"), "m30": g("w_30", "occurrence"), "m300": g("w_300", "occurrence")},
        "elev": g("elev", "elevation"), "elev_ring": g("elev_ring", "elevation"), "slope": g("slope", "slope"),
        "sar": sar, "hand": hand, "notes": notes,
    }


def _extras(plot):
    """Sentinel-1 SAR monsoon vs dry season wetness, and MERIT Hydro HAND (height above nearest drainage)."""
    import ee
    year = YEARS[-1]
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filterBounds(plot).select("VV"))

    def wet_share(s, e):  # share of acquisitions in which each pixel was wet, averaged over the plot
        col = s1.filterDate(s, e).map(lambda i: i.lt(SAR_WET_DB))
        return col.mean().reduceRegion(ee.Reducer.mean(), plot, 10, maxPixels=1e8)

    d = ee.Dictionary({
        "mon": wet_share(f"{year}-07-01", f"{year}-10-01"),
        "dry": wet_share(f"{year}-02-01", f"{year}-04-01"),
        "hand": ee.Image("MERIT/Hydro/v1_0_1").select("hnd").reduceRegion(ee.Reducer.mean(), plot, 90, maxPixels=1e8),
    }).getInfo()
    sar = {"mon": (d["mon"] or {}).get("VV"), "dry": (d["dry"] or {}).get("VV")}
    if sar["mon"] is None and sar["dry"] is None:
        sar = None
    return sar, (d["hand"] or {}).get("hnd")


# ---------- thumbnails for the PDF ----------

def thumbs(spec, years=(YEARS[0], YEARS[-1]), px=420):
    """Before / after Sentinel-2 true colour thumbnails with the plot outlined. {} if unavailable."""
    if not ee_ready():
        return {}
    import ee
    plot = plot_geom(spec)
    size = max(250, math.sqrt(spec["area_m2"]) * 1.5)
    view = plot.buffer(size).bounds()
    out = {}
    for y in years:
        try:
            col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                   .filterDate(f"{y}-01-01", f"{y}-05-01").filterBounds(plot)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
            img = col.median().visualize(bands=["B4", "B3", "B2"], min=0, max=3000)
            edge = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(plot)]), 1, 3).visualize(
                palette=["ff2d2d"], min=0, max=1)
            url = img.blend(edge).getThumbURL({"region": view, "dimensions": px, "format": "png"})
            with urllib.request.urlopen(url, timeout=60) as resp:
                out[y] = resp.read()
        except Exception as e:  # noqa
            print("thumbnail failed", y, e)
    return out


# ---------- demo data ----------

def demo(spec):
    """Deterministic fake data so the UI and PDF can be tested without EE."""
    rnd = random.Random(int(abs(spec["lat"] * 1000) + abs(spec["lon"] * 1000)))
    base = [rnd.gauss(0, 1) for _ in range(64)]
    emb, lc, nb = {}, {}, {}
    switch = rnd.choice([2020, 2021, 2022])
    for y in YEARS:
        shift = 0 if y < switch else 1.4
        emb[y] = [b + shift * (1 if i % 3 == 0 else 0.2) + rnd.gauss(0, 0.05) for i, b in enumerate(base)]
        lc[y] = {"crops": 0.9, "grass": 0.1} if y < switch else {"built": 0.7, "bare": 0.2, "grass": 0.1}
        nb[y] = min(0.6, 0.08 + 0.06 * (y - 2017))
    water = rnd.choice([{"on": 0, "m30": 0, "m300": 0}, {"on": 0, "m30": 0, "m300": 70},
                        {"on": 0, "m30": 62, "m300": 90}, {"on": 15, "m30": 20, "m300": 20}])
    return {
        "demo": True, "lat": spec["lat"], "lon": spec["lon"], "radius": spec["radius"], "polygon": spec["polygon"],
        "area_m2": spec["area_m2"], "emb": emb, "lc": lc, "nb_built": nb, "water": water,
        "elev": 540.0, "elev_ring": rnd.choice([548.0, 541.0]), "slope": rnd.choice([1.5, 2.1, 11.0, 18.0]),
        "sar": rnd.choice([{"mon": 0.02, "dry": 0.01}, {"mon": 0.45, "dry": 0.03}]),
        "hand": rnd.choice([0.8, 3.5, 12.0]), "notes": [],
    }
