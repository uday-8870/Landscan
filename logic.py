"""Turn raw data into a plain-language due diligence report with flags and a risk score."""
import numpy as np

from ee_data import YEARS

ACRE = 4046.86


def _cos_dist(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top_class(d):
    return max(d, key=d.get) if d else "unknown"


def build(raw):
    emb, lc, nb = raw["emb"], raw["lc"], raw["nb_built"]
    area = raw.get("area_m2") or 0

    # 1. year over year change from AlphaEarth embeddings
    steps = []
    for y0, y1 in zip(YEARS, YEARS[1:]):
        steps.append({"from": y0, "to": y1, "change": _cos_dist(emb[y0], emb[y1])})
    vals = [s["change"] for s in steps if s["change"] is not None]
    med = float(np.median(vals)) if vals else 0
    big = None
    for s in steps:
        c = s["change"]
        s["big"] = bool(c is not None and c > max(0.15, 3 * med))
        if s["big"] and (big is None or c > big["change"]):
            big = s

    # 2. land cover timeline
    timeline = [{"year": y, "cover": top_class(lc[y]), "share": round(max(lc[y].values()) if lc[y] else 0, 2)}
                for y in YEARS]
    first, last = timeline[0]["cover"], timeline[-1]["cover"]

    flags, risk = [], 0

    def add(level, title, detail, pts):
        nonlocal risk
        flags.append({"level": level, "title": title, "detail": detail})
        risk += pts

    # land use change
    if "unknown" not in (first, last) and first != last:
        when = f" (largest shift around {big['to']})" if big else ""
        add("info" if last != "water" else "high", f"Land cover changed: {first} to {last}",
            f"Satellite data shows this plot moved from {first} to {last}{when}.", 10)
        if first == "crops" and last == "built":
            add("medium", "Agricultural land converted to built-up",
                "Check for proper land-use conversion (non-agricultural) approval for this survey number.", 20)

    # water rings: on plot / 30 m buffer / 300 m. Values are % of time with surface water (JRC, 1984 to 2021)
    w = raw.get("water") or {}
    on, m30, m300 = (w.get(k) or 0 for k in ("on", "m30", "m300"))
    norms = "Buffer and FTL rules differ by authority, so confirm the exact distance with the local lake authority."
    if on >= 50:
        add("high", "Plot overlaps permanent water", f"Surface water seen {on:.0f}% of the time on the plot. "
            "Likely lake bed or water body land.", 40)
    elif m30 >= 50:
        add("high", "Within 30 m of permanent water", f"Permanent water ({m30:.0f}% of the time) touches the 30 m ring. "
            f"Typical buffer zone territory. {norms}", 35)
    elif on >= 10 or m30 >= 10:
        add("medium", "Seasonal water on or touching the plot",
            f"Water seen up to {max(on, m30):.0f}% of the time on or within 30 m. Check flood history.", 25)
    elif m300 >= 50:
        add("medium", "Permanent water within 300 m", f"Water body within 300 m. May fall inside FTL or buffer. {norms}", 20)
    elif m300 >= 10:
        add("info", "Seasonal water within 300 m", f"Water seen up to {m300:.0f}% of the time within 300 m.", 5)

    # HAND: height above nearest drainage (physical flood exposure, 90 m)
    hand = raw.get("hand")
    if hand is not None:
        if hand < 2:
            add("medium", f"Very low above nearest drainage ({hand:.1f} m)",
                "Plot sits almost at drainage level. High flood and waterlogging exposure in heavy rain.", 25)
        elif hand < 5:
            add("medium", f"Low above nearest drainage ({hand:.1f} m)",
                "Moderate flood exposure in extreme rain. Check drainage and past flood records.", 12)

    # Sentinel-1 radar: standing water in monsoon but not in dry season
    sar = raw.get("sar")
    if sar and sar.get("mon") is not None:
        mon, dry = sar["mon"], sar.get("dry") or 0
        if mon >= 0.25 and dry <= 0.10:
            add("medium", "Radar shows monsoon standing water",
                f"Sentinel-1 saw water on {mon * 100:.0f}% of monsoon passes versus {dry * 100:.0f}% in dry season. "
                "Likely waterlogging.", 20)

    # terrain
    elev, ring = raw.get("elev"), raw.get("elev_ring")
    low = None
    if elev is not None and ring is not None:
        low = ring - elev
        if low > 3:
            add("medium", "Low-lying compared to surroundings",
                f"Plot sits about {low:.1f} m below the 1 km average. Higher waterlogging exposure.", 15)
    slope = raw.get("slope")
    if slope is not None:
        if slope > 15:
            add("medium", f"Steep terrain ({slope:.0f} degrees)",
                "Hillside plot. Expect heavy cut and fill, retaining walls, and extra approvals.", 15)
        elif slope > 8:
            add("info", f"Sloping plot ({slope:.0f} degrees)", "Moderate slope. Factor in levelling cost.", 5)

    # neighbourhood growth
    g = nb[YEARS[-1]] - nb[YEARS[0]]
    if g > 0.15:
        add("info", "Fast growing area",
            f"Built-up share within 1 km rose by {g * 100:.0f} points since {YEARS[0]}. Demand likely high.", 0)
        growth = "fast"
    elif g > 0.05:
        growth = "moderate"
    else:
        growth = "slow"

    # data resolution: AlphaEarth is 10 m, so tiny plots are only a few pixels
    px = area / 100.0
    if px < 25:
        flags.append({"level": "medium" if px < 9 else "info", "title": f"Small plot: only about {px:.0f} satellite pixels",
                      "detail": "Results at this size are noisy and can be dominated by neighbouring land. "
                                "Treat flags as hints and prefer a drawn boundary over a pin."})
    for n in raw.get("notes") or []:
        flags.append({"level": "info", "title": "Data note", "detail": n})

    risk = min(100, risk)
    return {
        "lat": raw["lat"], "lon": raw["lon"], "demo": raw["demo"], "area_m2": round(area, 1),
        "area_acres": round(area / ACRE, 3), "polygon": raw.get("polygon"), "radius": raw.get("radius"),
        "risk": risk, "verdict": verdict_for(risk), "growth": growth,
        "steps": steps, "timeline": timeline,
        "neighbour_built": {str(y): round(v, 3) for y, v in nb.items()},
        "evidence": {"water": {"on": on, "m30": m30, "m300": m300}, "hand": hand, "sar": sar,
                     "slope": slope, "below_surroundings_m": low, "pixels": round(px)},
        "flags": flags,
    }


def verdict_for(risk):
    return "LOW" if risk < 25 else "MEDIUM" if risk < 55 else "HIGH"
