"""Lookalike model on AlphaEarth embeddings. Learns from spots the user labels on the map."""
import json
import os

import numpy as np

from ee_batch import emb_points, feature_vec

LABELS_PATH = os.environ.get("LABELS_PATH", os.path.join(os.path.dirname(__file__), "labels.json"))

# label -> (display name, risk points if this plot looks like it, is_risky)
CLASSES = {
    "lake_bed": ("known lake bed / water body land", 45, True),
    "encroachment": ("known encroachment (buffer / FTL / govt land)", 40, True),
    "unapproved_layout": ("known unapproved layout", 30, True),
    "clean": ("clean, problem free plot", 0, False),
    "hotspot": ("area that later boomed", 0, False),
}


def load():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return []


def save(rows):
    with open(LABELS_PATH, "w") as f:
        json.dump(rows, f)


def add_label(lat, lon, label):
    if label not in CLASSES:
        raise ValueError(f"unknown label {label}")
    emb = emb_points([(lat, lon)])[0]
    fv = feature_vec(emb)
    if fv is None:
        raise ValueError("no embedding data at this location")
    rows = load()
    rows.append({"lat": lat, "lon": lon, "label": label, "feat": fv})
    save(rows)
    return counts(rows)


def counts(rows=None):
    rows = load() if rows is None else rows
    c = {k: 0 for k in CLASSES}
    for r in rows:
        c[r["label"]] = c.get(r["label"], 0) + 1
    return c


def _unit(m):
    m = np.atleast_2d(np.array(m, float))
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def predict(feat, k=3, min_per_class=2):
    """Cosine kNN per class. Returns {label: similarity 0..1} and top neighbours."""
    rows = load()
    if not rows or feat is None:
        return None
    X = _unit([r["feat"] for r in rows])
    q = _unit(feat)[0]
    sims = X @ q
    out, nbrs = {}, []
    for lab in CLASSES:
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        if len(idx) < min_per_class:
            continue
        s = np.sort(sims[idx])[::-1][:k]
        out[lab] = float(s.mean())
    for i in np.argsort(-sims)[:3]:
        nbrs.append({"label": rows[i]["label"], "sim": round(float(sims[i]), 3),
                     "lat": rows[i]["lat"], "lon": rows[i]["lon"]})
    if not out:
        return None
    best = max(out, key=out.get)
    # confidence: gap between best class and the runner up
    vals = sorted(out.values(), reverse=True)
    gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]
    return {"scores": {k: round(v, 3) for k, v in out.items()}, "best": best,
            "best_sim": round(out[best], 3), "gap": round(float(gap), 3), "neighbours": nbrs,
            "n_labels": len(rows)}


def apply(report, look):
    """Merge lookalike result into the report: flags, risk points, verdict."""
    report["lookalike"] = look
    if not look:
        report["flags"].append({"level": "info", "title": "Lookalike model not trained yet",
                                "detail": "Label at least 2 spots per class on the map to enable AI lookalike risk."})
        return report
    name, pts, risky = CLASSES[look["best"]]
    strong = look["best_sim"] >= 0.75 and look["gap"] >= 0.03
    if risky and strong:
        level = "high" if pts >= 40 else "medium"
        report["flags"].insert(0, {
            "level": level, "title": f"AlphaEarth lookalike: resembles {name}",
            "detail": f"Embedding trajectory 2017 to 2024 is {look['best_sim'] * 100:.0f}% similar to labeled "
                      f"examples of this type ({look['n_labels']} labeled spots). A screening signal, not proof."})
        report["risk"] = min(100, report["risk"] + pts)
    elif look["best"] == "clean" and strong:
        report["flags"].append({"level": "info", "title": "AlphaEarth lookalike: resembles clean plots",
                                "detail": f"{look['best_sim'] * 100:.0f}% similar to labeled problem free plots."})
        report["risk"] = max(0, report["risk"] - 10)
    elif look["best"] == "hotspot" and strong:
        report["growth"] = "fast"
        report["flags"].append({"level": "info", "title": "AlphaEarth lookalike: resembles an area that boomed",
                                "detail": f"{look['best_sim'] * 100:.0f}% similar to labeled hotspot areas."})
    r = report["risk"]
    report["verdict"] = "LOW" if r < 25 else "MEDIUM" if r < 55 else "HIGH"
    return report
