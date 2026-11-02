"""Lookalike model on AlphaEarth embeddings. Learns from spots the user labels on the map.

Storage is SQLite (safe for concurrent requests). Prediction is a similarity weighted cosine kNN per class.
Leave one out cross validation gives honest accuracy numbers and calibrates the confidence thresholds."""
import json
import os
import sqlite3
import threading

import numpy as np

from ee_batch import FEAT_DIM, FEAT_VERSION, feature_from_points
from logic import verdict_for

HERE = os.path.dirname(__file__)
DB_PATH = os.environ.get("LABELS_DB", os.path.join(HERE, "landscan.db"))
OLD_JSON = os.environ.get("LABELS_PATH", os.path.join(HERE, "labels.json"))

# label -> (display name, risk points if this plot looks like it, is_risky)
CLASSES = {
    "lake_bed": ("known lake bed / water body land", 45, True),
    "encroachment": ("known encroachment (buffer / FTL / govt land)", 40, True),
    "unapproved_layout": ("known unapproved layout", 30, True),
    "clean": ("clean, problem free plot", 0, False),
    "hotspot": ("area that later boomed", 0, False),
}
DEFAULT_SIM, DEFAULT_GAP = 0.75, 0.03

_lock = threading.Lock()
_cache = {}


def _db():
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS labels(
        id INTEGER PRIMARY KEY AUTOINCREMENT, lat REAL NOT NULL, lon REAL NOT NULL, label TEXT NOT NULL,
        feat TEXT NOT NULL, ver INTEGER NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)""")
    return c


def _migrate_json():
    """One time import of v1 labels.json. Old rows have old feature versions and need reindex()."""
    if not os.path.exists(OLD_JSON):
        return
    with _lock, _db() as c:
        if c.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0:
            for r in json.load(open(OLD_JSON)):
                c.execute("INSERT INTO labels(lat,lon,label,feat,ver) VALUES(?,?,?,?,?)",
                          (r["lat"], r["lon"], r["label"], json.dumps(r.get("feat", [])), 1))
    os.replace(OLD_JSON, OLD_JSON + ".migrated")


_migrate_json()


def _touch():
    _cache.clear()


def all_rows():
    with _db() as c:
        return [dict(r) for r in c.execute("SELECT id,lat,lon,label,ver FROM labels ORDER BY id")]


def load_usable():
    with _db() as c:
        rows = c.execute("SELECT id,lat,lon,label,feat FROM labels WHERE ver=? ORDER BY id", (FEAT_VERSION,)).fetchall()
    return [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "label": r["label"], "feat": json.loads(r["feat"])}
            for r in rows]


def stale_count():
    with _db() as c:
        return c.execute("SELECT COUNT(*) FROM labels WHERE ver<>?", (FEAT_VERSION,)).fetchone()[0]


def counts():
    out = {k: 0 for k in CLASSES}
    for r in all_rows():
        out[r["label"]] = out.get(r["label"], 0) + 1
    return out


def _insert(lat, lon, label, feat, ver=FEAT_VERSION):
    with _lock, _db() as c:
        cur = c.execute("INSERT INTO labels(lat,lon,label,feat,ver) VALUES(?,?,?,?,?)",
                        (lat, lon, label, json.dumps(feat), ver))
    _touch()
    return cur.lastrowid


def add_label(lat, lon, label):
    if label not in CLASSES:
        raise ValueError(f"unknown label {label}")
    fv = feature_from_points([(lat, lon)])[0]
    if fv is None:
        raise ValueError("no embedding data at this location")
    return _insert(lat, lon, label, fv)


def update_label(lid, label):
    if label not in CLASSES:
        raise ValueError(f"unknown label {label}")
    with _lock, _db() as c:
        n = c.execute("UPDATE labels SET label=? WHERE id=?", (label, lid)).rowcount
    _touch()
    return n > 0


def delete_label(lid):
    with _lock, _db() as c:
        n = c.execute("DELETE FROM labels WHERE id=?", (lid,)).rowcount
    _touch()
    return n > 0


def reindex():
    """Recompute features for labels made with an older feature recipe (needs Earth Engine for real data)."""
    with _db() as c:
        stale = c.execute("SELECT id,lat,lon FROM labels WHERE ver<>?", (FEAT_VERSION,)).fetchall()
    done = 0
    for i in range(0, len(stale), 200):
        chunk = stale[i:i + 200]
        feats = feature_from_points([(r["lat"], r["lon"]) for r in chunk])
        with _lock, _db() as c:
            for r, f in zip(chunk, feats):
                if f is not None:
                    c.execute("UPDATE labels SET feat=?, ver=? WHERE id=?", (json.dumps(f), FEAT_VERSION, r["id"]))
                    done += 1
    _touch()
    return {"reindexed": done, "failed": len(stale) - done}


# ---------- prediction ----------

def _unit(m):
    m = np.atleast_2d(np.array(m, float))
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def _scores(X, labs, q, skip=None, k=3, min_per_class=2):
    """Per class score = similarity weighted mean of the top k cosine similarities (sum s^2 / sum s)."""
    sims = X @ q
    out = {}
    for lab in CLASSES:
        idx = [i for i, l in enumerate(labs) if l == lab and i != skip]
        if len(idx) < min_per_class:
            continue
        s = np.clip(np.sort(sims[idx])[::-1][:k], 1e-6, None)
        out[lab] = float((s * s).sum() / s.sum())
    return out, sims


def _best(out):
    best = max(out, key=out.get)
    vals = sorted(out.values(), reverse=True)
    return best, (vals[0] - vals[1]) if len(vals) > 1 else vals[0]


def metrics():
    """Leave one out cross validation + calibrated confidence thresholds."""
    rows = load_usable()
    key = ("m", len(rows), rows[-1]["id"] if rows else 0, tuple(r["label"] for r in rows))
    if key in _cache:
        return _cache[key]
    labs = [r["label"] for r in rows]
    res = {"n": len(rows), "evaluated": 0, "accuracy": None, "per_class": {}, "confusion": {},
           "thresholds": {"sim": DEFAULT_SIM, "gap": DEFAULT_GAP, "calibrated": False}, "ready": False,
           "stale": stale_count()}
    if len(rows) >= 6:
        X = _unit([r["feat"] for r in rows])
        ok_sims, ok_gaps, hit, ev = [], [], 0, 0
        conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
        per = {c: {"n": labs.count(c), "correct": 0, "tested": 0} for c in CLASSES}
        for i in range(len(rows)):
            out, _ = _scores(X, labs, X[i], skip=i)
            per[labs[i]]["tested"] += 1
            if not out:
                continue
            best, gap = _best(out)
            ev += 1
            conf[labs[i]][best] += 1
            if best == labs[i]:
                hit += 1
                per[labs[i]]["correct"] += 1
                ok_sims.append(out[best])
                ok_gaps.append(gap)
        res.update(evaluated=ev, accuracy=round(hit / ev, 3) if ev else None, confusion=conf)
        res["per_class"] = {c: {"n": v["n"], "recall": round(v["correct"] / v["tested"], 3) if v["tested"] else None}
                            for c, v in per.items() if v["n"]}
        res["ready"] = ev >= 10
        if ev >= 12 and hit / ev >= 0.6:
            res["thresholds"] = {"sim": round(float(np.clip(np.percentile(ok_sims, 10), 0.5, 0.95)), 3),
                                 "gap": round(float(np.clip(np.percentile(ok_gaps, 10), 0.01, 0.2)), 3),
                                 "calibrated": True}
    _cache[key] = res
    return res


def predict(feat):
    rows = load_usable()
    if not rows or feat is None:
        return None
    labs = [r["label"] for r in rows]
    X = _unit([r["feat"] for r in rows])
    q = _unit(feat)[0]
    out, sims = _scores(X, labs, q)
    if not out:
        return None
    best, gap = _best(out)
    thr = metrics()["thresholds"]
    nbrs = [{"id": rows[i]["id"], "label": rows[i]["label"], "sim": round(float(sims[i]), 3),
             "lat": rows[i]["lat"], "lon": rows[i]["lon"]} for i in np.argsort(-sims)[:3]]
    return {"scores": {k: round(v, 3) for k, v in out.items()}, "best": best, "best_sim": round(out[best], 3),
            "gap": round(float(gap), 3), "neighbours": nbrs, "n_labels": len(rows), "thresholds": thr,
            "strong": bool(out[best] >= thr["sim"] and gap >= thr["gap"])}


def apply(report, look):
    """Merge lookalike result into the report: flags, risk points, verdict."""
    report["lookalike"] = look
    stale = stale_count()
    if stale:
        report["flags"].append({"level": "info", "title": f"{stale} labeled spots use an old feature recipe",
                                "detail": "Open Teach the AI and press Reindex labels so they are used again."})
    if not look:
        report["flags"].append({"level": "info", "title": "Lookalike model not trained yet",
                                "detail": "Label at least 2 spots per class on the map to enable AI lookalike risk."})
        return report
    name, pts, risky = CLASSES[look["best"]]
    pct = f"{look['best_sim'] * 100:.0f}%"
    if risky and look["strong"]:
        report["flags"].insert(0, {
            "level": "high" if pts >= 40 else "medium", "title": f"AlphaEarth lookalike: resembles {name}",
            "detail": f"Embedding trajectory 2017 to 2024 is {pct} similar to labeled examples of this type "
                      f"({look['n_labels']} labeled spots). A screening signal, not proof."})
        report["risk"] = min(100, report["risk"] + pts)
    elif look["best"] == "clean" and look["strong"]:
        report["flags"].append({"level": "info", "title": "AlphaEarth lookalike: resembles clean plots",
                                "detail": f"{pct} similar to labeled problem free plots."})
        report["risk"] = max(0, report["risk"] - 10)
    elif look["best"] == "hotspot" and look["strong"]:
        report["growth"] = "fast"
        report["flags"].append({"level": "info", "title": "AlphaEarth lookalike: resembles an area that boomed",
                                "detail": f"{pct} similar to labeled hotspot areas."})
    report["verdict"] = verdict_for(report["risk"])
    return report
