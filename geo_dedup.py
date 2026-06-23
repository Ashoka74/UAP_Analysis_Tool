"""Geo-enrichment + spatiotemporal proximity for the timeline dedupe.

Adds the location axis the exact-string block was missing:
  1. forward-geocode `location.name` (state/country-hinted, confidence-gated,
     cached to geocode_cache.csv — one-time, polite to Nominatim);
  2. per candidate pair, compute geo_km (haversine), date_gap_days
     (uncertainty-scaled interval overlap), and a fused proximity_score;
  3. a calibration harness (band-ratio + threshold sweep) that answers
     "how do I validate the radius / window / τ?" against labels.

proximity_score is a BLOCKING + ranking signal, never the merge verdict —
narrative similarity (the confirmer) stays a separate column.
"""
import os
import re
import math
import time
import json
import datetime
import urllib.parse
import urllib.request

import pandas as pd
import make_timeline as mt          # reuse SRC, helpers, _NARR, _narrative_sim

CACHE = "geocode_cache.csv"
UA = "uap-timeline-dedup/0.1 (research; contact: maintainer)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Confidence gate: only town-or-finer, non-trivial-importance geocodes are
# trusted for *tight* proximity blocking. Coarser hits (state/country centroids)
# are treated as "no usable coordinate" so they never collapse to one point.
MIN_PLACE_RANK = 12          # Nominatim: ~16 town, ~12 suburb, ~8 state, ~4 country
MIN_IMPORTANCE = 0.15

# proximity_score = exp(-km/D0) * exp(-gap/T0): closeness scales, not a hard cliff.
D0_KM, T0_DAYS = 30.0, 3.0

# Uncertainty → half-width (days) of a row's plausible date interval.
DATE_SPAN = {"exact": 0, "approximate_day": 2, "approximate_month": 15,
             "approximate_year": 182, "unknown": 3650}


# ── geocoding ───────────────────────────────────────────────────────────────
def _clean_name(name):
    """Strip parentheticals and locative noise; for 'X and Y' / 'between X and Y'
    keep the first place token."""
    s = re.sub(r"\([^)]*\)", " ", str(name))
    s = re.sub(r"^\s*(near|vicinity of|over|around|outside|close to)\s+", "", s, flags=re.I)
    s = re.sub(r"^\s*between\s+", "", s, flags=re.I)
    s = re.split(r"\s+(?:and|to)\s+", s, maxsplit=1)[0]
    return s.strip(" ,;")


def _query(row):
    """Structured query when a clean city exists, else free-form name + country."""
    iso = row.get("location_country_iso")
    iso = str(iso).lower() if pd.notna(iso) and len(str(iso)) == 2 else None
    city = row.get("location.city")
    state = row.get("location.state")
    if state is None or (isinstance(state, float) and pd.isna(state)):
        state = row.get("location_state_norm")
    p = {"format": "json", "limit": 1, "addressdetails": 0}
    if pd.notna(city) and str(city).strip():
        p["city"] = str(city).strip()
        if pd.notna(state) and str(state).strip():
            p["state"] = str(state).strip()
    else:
        name = row.get("location.name") or row.get("sightingDetails.location.name")
        if not (pd.notna(name) and str(name).strip()):
            return None
        p["q"] = _clean_name(name)
    if iso:
        p["countrycodes"] = iso
    return p


def _fetch(params):
    url = NOMINATIM + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    if not data:
        return None
    d = data[0]
    return {
        "lat": float(d["lat"]), "lon": float(d["lon"]),
        "importance": float(d.get("importance", 0.0)),
        "place_rank": int(d.get("place_rank", 0)),
        "addresstype": d.get("addresstype", ""),
    }


def _cache_key(params):
    return json.dumps(params, sort_keys=True)


def geocode_rows(rows, polite=1.1):
    """Return {row_index: geocode_dict or None}, persisting a sidecar cache.
    Native lat/lon (when present) wins and skips the network entirely."""
    cache = {}
    if os.path.exists(CACHE):
        cdf = pd.read_csv(CACHE)
        for _, c in cdf.iterrows():
            cache[c["key"]] = None if pd.isna(c["lat"]) else {
                "lat": c["lat"], "lon": c["lon"], "importance": c["importance"],
                "place_rank": int(c["place_rank"]), "addresstype": c.get("addresstype", ""),
            }
    out, new = {}, False
    for i, row in enumerate(rows):
        # native coordinate takes priority
        for la, lo in (("location.latitude", "location.longitude"),
                       ("sightingDetails.location.latitude", "sightingDetails.location.longitude")):
            if pd.notna(row.get(la)) and pd.notna(row.get(lo)):
                out[i] = {"lat": float(row[la]), "lon": float(row[lo]),
                          "importance": 1.0, "place_rank": 30, "addresstype": "native"}
                break
        if i in out:
            continue
        params = _query(row)
        if params is None:
            out[i] = None
            continue
        key = _cache_key(params)
        if key not in cache:
            cache[key] = _fetch(params)
            new = True
            time.sleep(polite)        # Nominatim ≤ 1 req/s
        out[i] = cache[key]
    if new:
        pd.DataFrame([{"key": k, **(v or {"lat": None, "lon": None, "importance": None,
                                           "place_rank": None, "addresstype": None})}
                      for k, v in cache.items()]).to_csv(CACHE, index=False)
    return out


def trusted_coord(geo):
    """A geocode usable for tight proximity blocking (town-or-finer)."""
    if not geo:
        return None
    if geo["place_rank"] >= MIN_PLACE_RANK and geo["importance"] >= MIN_IMPORTANCE:
        return (geo["lat"], geo["lon"])
    return None


# ── proximity ────────────────────────────────────────────────────────────────
def haversine(a, b):
    (la1, lo1), (la2, lo2) = a, b
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _center_span(row):
    y = mt._int_or_none(row.get("date_time.year"))
    if y is None:
        return None
    m = mt._int_or_none(row.get("date_time.month"))
    d = mt._int_or_none(row.get("date_time.day"))
    try:
        center = datetime.date(y, min(max(m or 1, 1), 12), min(max(d or 1, 1), 28)).toordinal()
    except ValueError:
        center = datetime.date(y, 1, 1).toordinal()
    span = DATE_SPAN.get(row.get("date_time.dateUncertainty"), 30)
    if m is None:
        span = max(span, 182)
    elif d is None:
        span = max(span, 15)
    return center, span


def date_gap_days(a, b):
    """Gap between plausible date intervals (0 if they overlap)."""
    ca, cb = _center_span(a), _center_span(b)
    if ca is None or cb is None:
        return None
    (c1, s1), (c2, s2) = ca, cb
    return max(0, abs(c1 - c2) - (s1 + s2))


def proximity_score(geo_km, gap_days):
    if geo_km is None or gap_days is None:
        return None
    return math.exp(-geo_km / D0_KM) * math.exp(-gap_days / T0_DAYS)


# ── candidate-pair table (compute once; sweep thresholds cheaply) ────────────
def build_pairs(rows, geos):
    pairs = []
    n = len(rows)
    for i in range(n):
        ci = trusted_coord(geos.get(i))
        for j in range(i + 1, n):
            cj = trusted_coord(geos.get(j))
            gap = date_gap_days(rows[i], rows[j])
            geo_km = haversine(ci, cj) if (ci and cj) else None
            pairs.append({
                "i": i, "j": j,
                "geo_km": geo_km,
                "date_gap_days": gap,
                "proximity_score": proximity_score(geo_km, gap),
                "narrative_sim": mt._narrative_sim(str(rows[i].get(mt._NARR, "")),
                                                   str(rows[j].get(mt._NARR, ""))),
                "same_str_block": mt._block_key(rows[i]) is not None
                                  and mt._block_key(rows[i]) == mt._block_key(rows[j]),
            })
    return pd.DataFrame(pairs)


# ── validation: silver labels + band-ratio + threshold sweep ─────────────────
def silver_label(p, rows):
    """Auto-label for calibration ONLY — a placeholder for an LLM judge / human
    gold set. Derived from signals ORTHOGONAL to what we validate where possible:
      POS: same exact (date, city) AND same craft shape (structural agreement);
      NEG: date intervals > 30 days apart, OR > 250 km apart;
      else None (the borderline band the LLM/gold set must resolve)."""
    ri, rj = rows[p["i"]], rows[p["j"]]
    if p["same_str_block"]:
        si, sj = ri.get("craft_primary_shape_norm"), rj.get("craft_primary_shape_norm")
        if pd.notna(si) and si == sj:
            return 1
    if (p["date_gap_days"] is not None and p["date_gap_days"] > 30) or \
       (p["geo_km"] is not None and p["geo_km"] > 250):
        return 0
    return None


def band_ratio(labeled, col, edges):
    print(f"\n  dup-rate by {col} band (labeled pairs only):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = labeled[(labeled[col] >= lo) & (labeled[col] < hi)]
        if len(b):
            print(f"    [{lo:>6.1f}, {hi:>6.1f}): {b['label'].mean():5.0%}  (n={len(b)})")


def sweep(labeled, prox_grid, sim_grid):
    """Precision/recall over a (proximity_score, narrative_sim) grid. Pick the
    operating point by a PRECISION FLOOR (FP ≫ FN cost), not F1."""
    print("\n  threshold sweep — auto-merge iff proximity ≥ p AND sim ≥ τ:")
    print("     p_min   τ     precision   recall   (TP/FP/FN)")
    pos_total = int((labeled["label"] == 1).sum())
    for p in prox_grid:
        for t in sim_grid:
            sel = labeled[(labeled["proximity_score"].fillna(0) >= p) &
                          (labeled["narrative_sim"] >= t)]
            tp = int((sel["label"] == 1).sum())
            fp = int((sel["label"] == 0).sum())
            fn = pos_total - tp
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / pos_total if pos_total else float("nan")
            print(f"    {p:5.2f}  {t:4.2f}    {prec:7.2%}   {rec:6.0%}   "
                  f"({tp}/{fp}/{fn})")


# ── flag-similar-events: tunable full-row predictor (autoresearch loop edits this) ──
# The decision rule the loop optimizes toward agreement with the LLM gold set.
# Conjunctive gates (per the design): similarity AND date AND location AND fields.
PREDICT_PARAMS = {
    "tau": 0.10,             # narrative-similarity floor (Tier-2 confirmer)
    "use_date_gate": True,   # require date intervals within t_max_days
    "t_max_days": 0,
    "use_loc_gate": False,   # require same normalized location / shared state
    "use_field_score": True,
    "field_min": 0.5,        # min fraction of key structured fields that must agree
    "use_embed": False,      # use MiniLM embedding cosine as the confirmer (vs lexical)
    "embed_tau": 0.78,       # embedding-cosine floor when use_embed
}

_FIELD_COLS = ["location_country_iso", "location_state_norm", "craft_primary_shape_norm",
               "craft_size_band", "witness_primary_role", "date_time.year"]


def _loc_match(a, b):
    ka, kb = mt._loc_key(a), mt._loc_key(b)
    if ka and ka == kb:
        return True
    sa, sb = a.get("location_state_norm"), b.get("location_state_norm")
    ca, cb = a.get("location_country_iso"), b.get("location_country_iso")
    if pd.notna(sa) and sa == sb and ca == cb:
        return True
    return False


def _field_agreement(a, b):
    seen = agree = 0
    for c in _FIELD_COLS:
        va, vb = a.get(c), b.get(c)
        if pd.isna(va) or pd.isna(vb):
            continue
        seen += 1
        if va == vb:
            agree += 1
    return agree / seen if seen else 0.0


def predict_same_event(a, b, narrative_sim, embed_sim=None, P=None):
    P = P or PREDICT_PARAMS
    if P.get("use_embed") and embed_sim is not None:
        if embed_sim < P["embed_tau"]:
            return False
    elif narrative_sim < P["tau"]:
        return False
    if P["use_date_gate"]:
        g = date_gap_days(a, b)
        if g is None or g > P["t_max_days"]:
            return False
    if P["use_loc_gate"] and not _loc_match(a, b):
        return False
    if P["use_field_score"] and _field_agreement(a, b) < P["field_min"]:
        return False
    return True


def verify_f1(gold="gold_pairs.csv"):
    """Metric: F1 of predict_same_event vs the LLM gold labels. Prints the number."""
    g = pd.read_csv(gold)
    df = pd.read_csv(mt.SRC)
    tp = fp = fn = 0
    for _, r in g.iterrows():
        a, b = df.loc[int(r["i_id"])], df.loc[int(r["j_id"])]
        es = float(r["embed_sim"]) if "embed_sim" in r and pd.notna(r["embed_sim"]) else None
        pred = predict_same_event(a, b, float(r["narrative_sim"]), embed_sim=es)
        lab = bool(int(r["llm_same_event"]))
        if pred and lab:
            tp += 1
        elif pred and not lab:
            fp += 1
        elif (not pred) and lab:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"# tp={tp} fp={fp} fn={fn} precision={prec:.3f} recall={rec:.3f}")
    print(f"{f1:.4f}")
    return f1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_f1()
        sys.exit(0)
    df = pd.read_csv(mt.SRC)
    df = df[df["date_time.year"].notna() & (df["date_time.year"] > 1000)].copy()
    df["_y"] = df["date_time.year"]; df["_m"] = df["date_time.month"].fillna(13)
    df["_d"] = df["date_time.day"].fillna(32); df["_t"] = df["date_time.local_time"].fillna("99:99").astype(str)
    df = df.sort_values(["_y", "_m", "_d", "_t"]).head(40).reset_index(drop=True)
    rows = [r for _, r in df.iterrows()]

    print("Geocoding 40 rows (cached)…")
    geos = geocode_rows(rows)
    ok = sum(1 for g in geos.values() if trusted_coord(g))
    print(f"  trusted coordinates: {ok}/40 "
          f"({sum(1 for g in geos.values() if g)} raw hits)\n")

    pairs = build_pairs(rows, geos)

    print("proximity_score on the three diagnostic pairs:")
    def show(y, m, d, label):
        idx = [k for k, r in enumerate(rows)
               if mt._int_or_none(r.get("date_time.year")) == y
               and mt._int_or_none(r.get("date_time.month")) == m
               and mt._int_or_none(r.get("date_time.day")) == d]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                row = pairs[(pairs["i"] == idx[a]) & (pairs["j"] == idx[b])]
                if len(row):
                    r = row.iloc[0]
                    km = "n/a" if pd.isna(r["geo_km"]) else f"{r['geo_km']:.0f}km"
                    pr = "n/a" if pd.isna(r["proximity_score"]) else f"{r['proximity_score']:.2f}"
                    print(f"  {label}: geo={km:>7} gap={r['date_gap_days']:.0f}d "
                          f"prox={pr:>4} sim={r['narrative_sim']:.2f}")
    show(1833, 12, 8, "1833 Las Vegas NM (true dup)")
    show(1940, 5, 7, "1940 Camp Hood   (likely dup)")
    show(1935, 12, 4, "1935 ABQ↔LosAlamos (distinct?)")

    pairs["label"] = pairs.apply(lambda p: silver_label(p, rows), axis=1)
    labeled = pairs[pairs["label"].notna()].copy()
    print(f"\nValidation set: {int((labeled['label']==1).sum())} silver-POS, "
          f"{int((labeled['label']==0).sum())} silver-NEG, "
          f"{int(pairs['label'].isna().sum())} UNKNOWN (→ LLM/gold).")
    geo_lab = labeled[labeled["geo_km"].notna()]
    if len(geo_lab):
        band_ratio(geo_lab, "geo_km", [0, 10, 25, 50, 100, 250, 1e4])
    band_ratio(labeled, "date_gap_days", [0, 1, 3, 7, 30, 1e5])
    sweep(labeled, prox_grid=[0.0, 0.3, 0.6], sim_grid=[0.45, 0.55, 0.65])
