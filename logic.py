"""Turn raw data into a plain-language due diligence report with flags and a risk score."""
import numpy as np

from ee_data import YEARS


def _cos_dist(a, b):
    if a is None or b is None or any(x is None for x in a) or any(x is None for x in b):
        return None
    a, b = np.array(a, float), np.array(b, float)
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top_class(d):
    return max(d, key=d.get) if d else "unknown"


def build(raw):
    emb, lc, nb = raw["emb"], raw["lc"], raw["nb_built"]

    # 1. year over year change from AlphaEarth embeddings
    steps = []
    for y0, y1 in zip(YEARS, YEARS[1:]):
        d = _cos_dist(emb.get(y0), emb.get(y1))
        steps.append({"from": y0, "to": y1, "change": d})
    vals = [s["change"] for s in steps if s["change"] is not None]
    med = float(np.median(vals)) if vals else 0
    big = None
    for s in steps:
        c = s["change"]
        s["big"] = bool(c is not None and c > max(0.15, 3 * med))
        if s["big"] and (big is None or c > big["change"]):
            big = s

    # 2. land cover timeline
    timeline = [{"year": y, "cover": top_class(lc.get(y)), "share": round(max(lc[y].values()) if lc.get(y) else 0, 2)} for y in YEARS]
    first, last = timeline[0]["cover"], timeline[-1]["cover"]

    flags = []
    risk = 0

    def add(level, title, detail, pts):
        nonlocal risk
        flags.append({"level": level, "title": title, "detail": detail})
        risk += pts

    # converted land
    if first != last and first != "unknown" and last != "unknown":
        yr = big["to"] if big else None
        when = f" (largest shift around {yr})" if yr else ""
        add("info" if last != "water" else "high", f"Land cover changed: {first} to {last}",
            f"Satellite data shows this plot moved from {first} to {last}{when}.", 10)
    if first == "crops" and last == "built":
        add("medium", "Agricultural land converted to built-up",
            "Check for proper land-use conversion (non-agricultural) approval for this survey number.", 20)

    # water
    w = raw["water_occ"] or 0
    if w >= 50:
        add("high", "Plot is on or next to permanent water",
            f"Surface water seen {w:.0f}% of the time within 300 m. Possible lake bed or buffer zone.", 40)
    elif w >= 10:
        add("medium", "Seasonal water nearby",
            f"Surface water seen {w:.0f}% of the time within 300 m. Check FTL/buffer zone and flood history.", 25)

    # terrain
    if raw["elev"] is not None and raw["elev_ring"] is not None:
        low = raw["elev_ring"] - raw["elev"]
        if low > 3:
            add("medium", "Low-lying compared to surroundings",
                f"Plot sits about {low:.1f} m below the 1 km average. Higher waterlogging and flood exposure.", 15)

    # neighbourhood growth
    g = nb.get(YEARS[-1], 0.0) - nb.get(YEARS[0], 0.0)
    if g > 0.15:
        add("info", "Fast growing area",
            f"Built-up share within 1 km rose by {g * 100:.0f} points since {YEARS[0]}. Demand and price momentum likely high.", 0)
        growth = "fast"
    elif g > 0.05:
        growth = "moderate"
    else:
        growth = "slow"

    # data quality note
    if raw["radius"] < 150:
        pass

    risk = min(100, risk)
    verdict = "LOW" if risk < 25 else "MEDIUM" if risk < 55 else "HIGH"
    return {
        "lat": raw["lat"], "lon": raw["lon"], "demo": raw["demo"],
        "risk": risk, "verdict": verdict, "growth": growth,
        "steps": steps, "timeline": timeline,
        "neighbour_built": {str(y): round(v, 3) for y, v in nb.items()},
        "water_occ": w, "flags": flags,
    }
