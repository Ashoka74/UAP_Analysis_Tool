import os
import sys
import math
from datetime import datetime
from difflib import SequenceMatcher
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from . import us_gazetteer

logger = logging.getLogger(__name__)

# Selectable embedding models — mirrors the picker in rag_search.py so Dedupe
# Studio and Smart-Search offer the same three options from one place.
HARRIER_MODEL = "microsoft/harrier-oss-v1-270m"
DEFAULT_EMBED_COLS = ["witness.notes", "description", "case_text.text", "narrative", "text"]

# Opportunistically surfaced in the cluster preview (Stage B) alongside
# whatever embedding columns the caller picked — free-text craft
# color/shape descriptors are the fastest way for a human to visually
# corroborate "this cluster really is one event, multiple witnesses" (colors
# agree) vs. "this might be a bad merge" (colors disagree), independent of
# whichever columns were used for the text-similarity comparison itself.
QUICK_GLANCE_COLS = [
    "craft.colour", "sightingDetails.uapCharacteristics.color", "sightingDetails.craft.colour",
    "craft.primary_shape", "sightingDetails.uapCharacteristics.shape", "sightingDetails.craft.primary_shape",
    # Witness count/role — a cluster with radically different witness counts
    # per member (e.g. "1" vs "12") is a signal worth seeing at a glance
    # even when witness.notes itself wasn't picked as an embedding column.
    "witness.count", "witness_count_num", "sightingDetails.observerDetails.numberOfWitnesses",
    "witness.roles", "witness_primary_role",
    # Set by the React app's Database Pool (Cross-DB) run when pooling
    # multiple source databases into one comparison — surfacing it here
    # means a cluster spanning several databases visibly shows which ones,
    # not just "5 similar reports" with no indication they're cross-source.
    "_source_dataset",
]
EMBEDDING_MODEL_OPTIONS = [
    HARRIER_MODEL,
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
_MODEL_NAME = HARRIER_MODEL  # backward-compat default surfaced in API responses

_model_cache: Dict[str, Any] = {}

def _get_model(model_name: str = HARRIER_MODEL):
    """Return a cached embedder for model_name, loading it on first use.

    Harrier uses the shared uap_analyzer loader (asymmetric encode_document/
    encode_query API, same instance the rest of the app uses — no duplicate
    weights in memory). Other models load as plain sentence-transformers,
    cached per model_name so switching models mid-session doesn't reload.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        if model_name == HARRIER_MODEL:
            from uap_analyzer import get_embed_model
            m = get_embed_model()
        else:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            m = SentenceTransformer(model_name, device=device)
        _model_cache[model_name] = m
        return m
    except Exception as e:
        logger.warning(f"Failed to load embedder '{model_name}': {e}")
        return None

def _encode_texts(texts: List[str], model_name: str = HARRIER_MODEL) -> np.ndarray:
    """Batch-encode many texts in one forward pass (used by the cross-DB
    matrix pipeline — encoding row-by-row does not scale past a few hundred
    rows)."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    model = _get_model(model_name)
    if model is None:
        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(len(texts), 1024)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    if model_name == HARRIER_MODEL and hasattr(model, "encode_document"):
        vecs = model.encode_document(texts, batch_size=256, show_progress_bar=False)
    else:
        vecs = model.encode(texts, batch_size=256, normalize_embeddings=True,
                            show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)

def _encode_text(text: str, model_name: str = HARRIER_MODEL) -> np.ndarray:
    return _encode_texts([text], model_name)[0]

def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def _hdbscan_block_labels(vecs: np.ndarray, min_cluster_size: int = 15,
                           n_components: int = 10, seed: int = 42) -> np.ndarray:
    """UMAP + HDBSCAN cluster labels for embedding-similarity BLOCKING —
    same technique as the Analysis tab's cluster pipeline (uap_analyzer.py's
    reduce_for_clustering/cluster_data), but standalone here since we already
    have the raw embedding vectors and don't need the DataFrame/plotting/
    naming machinery UAPAnalyzer carries.

    Rows sharing a label only get compared against each other in the blocked
    candidate-pair search below, instead of every row against every other
    row. This turns the O(n*m) dense similarity matrix into a sum of much
    smaller per-block matrices — the difference between a matrix that fits
    in memory and one that doesn't once a dataset reaches tens of thousands
    of rows.

    Label -1 (HDBSCAN "noise", i.e. no dense neighborhood) is returned as-is
    and later treated as one shared block, not dropped — so noise-labeled
    rows still get compared against each other, just not against clustered
    rows. On failure (missing deps, degenerate input) returns all-zeros,
    i.e. one block containing every row — equivalent to no blocking.
    """
    n = vecs.shape[0]
    if n < max(4, min_cluster_size):
        return np.zeros(n, dtype=int)
    try:
        import umap
        import fast_hdbscan as hdbscan
        n_comp = min(n_components, max(2, n - 2))
        reduced = umap.UMAP(n_components=n_comp, min_dist=0.0, random_state=seed).fit_transform(vecs)
        labels = hdbscan.HDBSCAN(min_cluster_size=min(min_cluster_size, n)).fit(reduced).labels_
        return np.asarray(labels)
    except Exception as e:
        logger.warning(f"Cluster-blocking failed ({e}); falling back to a single block (no blocking).")
        return np.zeros(n, dtype=int)

def _compute_block_labels(vecs_a: np.ndarray, vecs_b: np.ndarray, mode: str,
                           min_cluster_size: int) -> tuple:
    """Block labels for A and B rows on a COMPARABLE label space.

    within_dataset: vecs_b is vecs_a, so one clustering pass labels both.
    between_datasets: A and B are clustered TOGETHER (concatenated) so a
    label means the same semantic cluster on both sides — clustering them
    separately would make label ids meaningless across datasets.
    """
    if mode == "within_dataset":
        labels = _hdbscan_block_labels(vecs_a, min_cluster_size)
        return labels, labels
    combined = np.vstack([vecs_a, vecs_b])
    labels = _hdbscan_block_labels(combined, min_cluster_size)
    return labels[:len(vecs_a)], labels[len(vecs_a):]

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _parse_date(d_str: Any) -> Optional[datetime]:
    if not d_str or pd.isna(d_str):
        return None
    s = str(d_str).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%d/%m/%Y", "%Y"):
        try:
            return datetime.strptime(s[:10] if len(s) >= 10 else s, fmt)
        except Exception:
            continue
    return None

def _resolve_date(rec: Dict[str, Any], date_col: Optional[str]) -> Optional[datetime]:
    """Explicit column when given (from the UI's date-column selector),
    otherwise the historical common-name guesses."""
    if date_col:
        return _parse_date(rec.get(date_col))
    return _parse_date(rec.get("date_time") or rec.get("date"))

def _resolve_coord(rec: Dict[str, Any], lat_col: Optional[str], lon_col: Optional[str]) -> Optional[tuple]:
    """Explicit lat/lon columns when given, otherwise the historical
    common-name guesses. Returns None if not resolvable to two floats."""
    try:
        if lat_col and lon_col:
            return float(rec.get(lat_col)), float(rec.get(lon_col))
        return float(rec.get("latitude") or rec.get("lat")), float(rec.get("longitude") or rec.get("lon"))
    except (TypeError, ValueError):
        return None

def _location_name_similarity(a: Any, b: Any) -> Optional[float]:
    """Lexical fallback for datasets with no numeric coordinates and no
    gazetteer match — a normalized SequenceMatcher ratio over the
    location-name text. Coarser than haversine distance (no real "how far
    apart" answer, and sensitive to how consistently the names are
    written), but still better than a gate that silently never fires."""
    sa = str(a or "").strip().lower()
    sb = str(b or "").strip().lower()
    if not sa or not sb or sa in ("nan", "none", "null") or sb in ("nan", "none", "null"):
        return None
    return SequenceMatcher(None, sa, sb).ratio()

def _resolve_effective_coord(
    rec: Dict[str, Any],
    lat_col: Optional[str],
    lon_col: Optional[str],
    location_col: Optional[str],
    state_col: Optional[str],
    use_gazetteer: bool = False,
    city_col: Optional[str] = None,
):
    """(lat, lon) for a row plus how it was obtained: numeric lat/lon
    columns first, then — only when use_gazetteer=True — an offline US
    Census Gazetteer lookup. When a dedicated city_col is given, it's tried
    first against state_col: a clean, already-separated city name is a far
    more reliable gazetteer key than anything pulled out of a free-text
    location_col (which is often a full sentence — "Parking lot back of
    police station, Portland, Oregon" — that a naive comma-split mis-parses).
    Falls back to location_col (+ state_col, or a parsed "City, ST" in
    location_col itself when there's no separate state column) when city_col
    isn't given or doesn't resolve. Returns ((lat, lon), 'coords' |
    'gazetteer') or None if nothing resolves — at which point the caller
    falls back further to text similarity. Off by default: it's US-only and
    a place name can collide across states/countries, so it's opt-in like
    cluster-blocking rather than a silent default for every location-name
    dataset."""
    coord = _resolve_coord(rec, lat_col, lon_col)
    if coord:
        return coord, "coords"
    if not use_gazetteer:
        return None
    state_val = rec.get(state_col) if state_col else None
    if city_col:
        geo = us_gazetteer.geocode_us_place(rec.get(city_col), state_val)
        if geo:
            return geo, "gazetteer"
    if location_col:
        raw_loc = rec.get(location_col)
        if state_val:
            geo = us_gazetteer.geocode_us_place(raw_loc, state_val)
        else:
            parsed = us_gazetteer.parse_city_state(raw_loc)
            geo = us_gazetteer.geocode_us_place(*parsed) if parsed else None
        if geo:
            return geo, "gazetteer"
    return None

def check_similarity(text_a: str, text_b: str, model_name: str = HARRIER_MODEL) -> Dict[str, Any]:
    """
    Simple check: computes embedding cosine similarity between two narratives
    using the selected model (Harrier by default).
    """
    if not text_a or not text_b:
        return {
            "similarity_score": 0.0,
            "is_similar": False,
            "model": model_name,
            "reason": "One or both text inputs are empty."
        }

    vec_a = _encode_text(text_a, model_name)
    vec_b = _encode_text(text_b, model_name)
    sim = _cosine_sim(vec_a, vec_b)

    return {
        "similarity_score": round(sim, 4),
        "is_similar": sim >= 0.80,
        "model": model_name,
        "reason": f"Cosine similarity {sim:.4f} is {'above' if sim >= 0.80 else 'below'} the 0.80 threshold."
    }

def check_duplicate(record_a: Dict[str, Any], record_b: Dict[str, Any],
                    model_name: str = HARRIER_MODEL) -> Dict[str, Any]:
    """
    Screening check: evaluates whether two records describe the same sighting/event
    by combining narrative similarity (selected model), haversine geographic
    distance, and temporal difference.
    """
    narr_a = str(record_a.get("narrative") or record_a.get("text") or record_a.get("description") or "")
    narr_b = str(record_b.get("narrative") or record_b.get("text") or record_b.get("description") or "")

    sim_result = check_similarity(narr_a, narr_b, model_name)
    sim = sim_result["similarity_score"]
    
    # Distance calculation
    km_diff = None
    try:
        lat1 = float(record_a.get("lat") or record_a.get("latitude"))
        lon1 = float(record_a.get("lon") or record_a.get("longitude"))
        lat2 = float(record_b.get("lat") or record_b.get("latitude"))
        lon2 = float(record_b.get("lon") or record_b.get("longitude"))
        km_diff = round(_haversine_km(lat1, lon1, lat2, lon2), 2)
    except (ValueError, TypeError):
        pass
    
    # Date difference calculation
    days_diff = None
    da = _parse_date(record_a.get("date") or record_a.get("date_time"))
    db = _parse_date(record_b.get("date") or record_b.get("date_time"))
    if da and db:
        days_diff = abs((da - db).days)
        
    reasons = []
    reasons.append(f"Narrative Cosine Similarity: {sim:.4f} ({model_name.rsplit('/', 1)[-1]})")
    if km_diff is not None:
        reasons.append(f"Spatial Separation: {km_diff} km")
    else:
        reasons.append("Spatial Separation: Unknown (missing coordinates)")
        
    if days_diff is not None:
        reasons.append(f"Temporal Difference: {days_diff} days")
    else:
        reasons.append("Temporal Difference: Unknown (missing/unparseable dates)")
        
    # Decision Gate Logic
    is_dup = False
    confidence = "LOW"
    
    if sim >= 0.88 and (km_diff is None or km_diff <= 15.0) and (days_diff is None or days_diff <= 1):
        is_dup = True
        confidence = "HIGH"
        reasons.append("Verdict: High-confidence duplicate (High semantic similarity + matching location/date).")
    elif sim >= 0.80 and (km_diff is not None and km_diff <= 50.0) and (days_diff is not None and days_diff <= 3):
        is_dup = True
        confidence = "MEDIUM"
        reasons.append("Verdict: Probable duplicate (Strong semantic similarity across nearby space/time).")
    elif sim >= 0.82 and (km_diff is None or days_diff is None):
        is_dup = True
        confidence = "MEDIUM"
        reasons.append("Verdict: Candidate duplicate (High semantic similarity, partial metadata match).")
    else:
        is_dup = False
        confidence = "HIGH" if sim < 0.70 else "MEDIUM"
        reasons.append("Verdict: Distinct reports (Insufficient semantic/spatial/temporal convergence).")
        
    return {
        "is_duplicate": is_dup,
        "confidence": confidence,
        "reasons": reasons,
        "metrics": {
            "similarity_score": sim,
            "haversine_km": km_diff,
            "date_diff_days": days_diff
        }
    }

def _build_row_text(rec: Dict[str, Any], cols: List[str]) -> str:
    """Concatenate the selected embedding columns into the one string that
    actually gets encoded — shared by the pairwise engine and Stage B's
    canonical-similarity scoring, so "similarity" always means the same
    thing regardless of which stage computed it."""
    parts = []
    for c in cols:
        v = str(rec.get(c, "")).strip()
        if v and v.lower() not in ("nan", "none", "null"):
            parts.append(v)
    if not parts:
        # Fallback to any narrative text
        for fallback in ("narrative", "text", "description", "witness.notes"):
            v = str(rec.get(fallback, "")).strip()
            if v and v.lower() not in ("nan", "none", "null"):
                parts.append(v)
                break
    return " - ".join(parts) if parts else "Unknown Report"


class _UnionFind:
    """Minimal union-find over string node ids (dedup record ids)."""

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _row_completeness(row: Dict[str, Any]) -> int:
    """Count of non-empty fields — the heuristic used to pick the canonical
    row per cluster (most fully populated record wins)."""
    count = 0
    for v in row.values():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "null"):
            continue
        count += 1
    return count


def build_enriched_records(
    records: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    id_field: str = "id",
    text_duplicate_clusters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Bake Stage B's cluster membership onto every original row, for export
    to (or merging back into) the Data Explorer dataset — not just the rows
    that showed up in a pair. Adds four columns from the strict (text+date+
    location) clustering:
      dedup_cluster_id     — the cluster's id, or None if never grouped
      dedup_cluster_size   — members in the cluster (1 for an unclustered row)
      dedup_is_canonical   — True for the one row kept per cluster (and,
                              trivially, for every unclustered row)
      dedup_is_duplicate   — True for every non-canonical member — the set
                              you'd filter out to de-duplicate the dataset

    A cluster with size > 1 is exactly the "multiple witnesses reported the
    same event" case the Data Explorer view wants to surface.

    When text_duplicate_clusters is given (the independent text-only
    exact-duplicate pass — sim >= 0.88, date/location gates ignored — from
    run_advanced_dedup), four more columns are added the same way, prefixed
    dedup_text_*, plus a combined convenience flag:
      dedup_is_duplicate_any — True if the row is a non-canonical member of
                                *either* clustering. This is the column to
                                filter out to de-duplicate under both
                                criteria at once — it can never drop every
                                member of a connected group, since each
                                clustering always keeps its own canonical.

    id_field mirrors the id fallback run_cross_db_pipeline uses internally
    (id -> case_id -> "ROW-{n}") so membership lookups line up even for
    rows that had no explicit id column.
    """
    cluster_by_member: Dict[str, Dict[str, Any]] = {}
    for c in clusters:
        for member_id in c.get("member_ids", []):
            cluster_by_member[member_id] = c

    text_cluster_by_member: Dict[str, Dict[str, Any]] = {}
    for c in (text_duplicate_clusters or []):
        for member_id in c.get("member_ids", []):
            text_cluster_by_member[member_id] = c

    enriched = []
    for i, rec in enumerate(records):
        rid = str(rec.get(id_field) or rec.get("case_id") or f"ROW-{i + 1}")
        c = cluster_by_member.get(rid)
        out = dict(rec)
        if c:
            out["dedup_cluster_id"] = c["cluster_id"]
            out["dedup_cluster_size"] = c["size"]
            out["dedup_is_canonical"] = (rid == c["canonical_id"])
            out["dedup_is_duplicate"] = (c["size"] > 1 and rid != c["canonical_id"])
        else:
            out["dedup_cluster_id"] = None
            out["dedup_cluster_size"] = 1
            out["dedup_is_canonical"] = True
            out["dedup_is_duplicate"] = False

        if text_duplicate_clusters is not None:
            tc = text_cluster_by_member.get(rid)
            if tc:
                out["dedup_text_cluster_id"] = tc["cluster_id"]
                out["dedup_text_cluster_size"] = tc["size"]
                out["dedup_text_is_canonical"] = (rid == tc["canonical_id"])
                out["dedup_text_is_duplicate"] = (tc["size"] > 1 and rid != tc["canonical_id"])
            else:
                out["dedup_text_cluster_id"] = None
                out["dedup_text_cluster_size"] = 1
                out["dedup_text_is_canonical"] = True
                out["dedup_text_is_duplicate"] = False
            out["dedup_is_duplicate_any"] = bool(out["dedup_is_duplicate"] or out["dedup_text_is_duplicate"])

        enriched.append(out)
    return enriched


def attach_resolved_coords(
    records: List[Dict[str, Any]],
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    location_col: Optional[str] = None,
    state_col: Optional[str] = None,
    city_col: Optional[str] = None,
    use_gazetteer: bool = False,
) -> List[Dict[str, Any]]:
    """Write each row's effective coordinates back onto it as three columns:
      _resolved_lat / _resolved_lon  — what the location gate actually used
      _resolved_coord_source        — 'coords' | 'gazetteer' | None

    The pipeline resolves these internally per row (numeric lat/lon columns
    first, then the opt-in gazetteer city/state centroid) but never persisted
    them — so a dataset whose only location data is a city name exported with
    empty lat/lon and could not be mapped at all, even though the location
    gate had real coordinates for it during the run. Same resolution
    function, same tier order, same use_gazetteer gating as the pipeline, so
    the exported coordinates are exactly what the km gate compared."""
    out = []
    for rec in records:
        r = dict(rec)
        resolved = _resolve_effective_coord(rec, lat_col, lon_col, location_col, state_col, use_gazetteer, city_col)
        if resolved:
            (lat, lon), source = resolved
            r["_resolved_lat"] = lat
            r["_resolved_lon"] = lon
            r["_resolved_coord_source"] = source
        else:
            r["_resolved_lat"] = None
            r["_resolved_lon"] = None
            r["_resolved_coord_source"] = None
        out.append(r)
    return out


def build_cluster_edges(
    enriched_records: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    id_field: str = "id",
    date_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Per-cluster chronological movement chain, for tracking one UAP event
    across its sightings on a map: within each cluster, members are ordered
    by date and each consecutive pair becomes one edge (leg 1: earliest ->
    next, leg 2: next -> next...), rather than a star of members around the
    canonical row — following the legs in order follows the phenomenon
    through time and space. Undated members sort to the end of their chain;
    legs whose endpoints lack resolved coordinates still appear (the
    chronology is still meaningful) but with empty lat/lon and no leg_km.

    Rows are sorted by each cluster's earliest date, then cluster, then leg
    order — so the file reads chronologically overall while every cluster's
    chain stays contiguous. Feed it to a Kepler Arc/Line layer via the
    from_/to_ lat/lon columns; color by leg_km or to_canonical_similarity.
    """
    rec_by_id: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(enriched_records):
        rid = str(rec.get(id_field) or rec.get("case_id") or f"ROW-{i + 1}")
        rec_by_id[rid] = rec

    def _member_date(rid: str) -> Optional[datetime]:
        rec = rec_by_id.get(rid) or {}
        return _parse_date(rec.get(date_field)) if date_field else None

    keyed_edges: List[tuple] = []
    for c in clusters:
        sim_by_id = {m.get("id"): m.get("canonical_similarity") for m in c.get("member_previews", [])}
        dated = [(_member_date(rid), rid) for rid in c.get("member_ids", [])]
        dated.sort(key=lambda t: (t[0] is None, t[0] or datetime.max, t[1]))
        first_date = next((d for d, _ in dated if d is not None), None)

        for leg, ((d_from, id_from), (d_to, id_to)) in enumerate(zip(dated, dated[1:]), start=1):
            rf = rec_by_id.get(id_from) or {}
            rt = rec_by_id.get(id_to) or {}
            lat_f, lon_f = rf.get("_resolved_lat"), rf.get("_resolved_lon")
            lat_t, lon_t = rt.get("_resolved_lat"), rt.get("_resolved_lon")
            leg_km = None
            if None not in (lat_f, lon_f, lat_t, lon_t):
                leg_km = round(_haversine_km(lat_f, lon_f, lat_t, lon_t), 2)
            keyed_edges.append((
                (first_date is None, first_date or datetime.max, c["cluster_id"], leg),
                {
                    "cluster_id": c["cluster_id"],
                    "cluster_size": c["size"],
                    "leg_index": leg,
                    "from_id": id_from,
                    "to_id": id_to,
                    "from_date": d_from.isoformat() if d_from else None,
                    "to_date": d_to.isoformat() if d_to else None,
                    "from_lat": lat_f, "from_lon": lon_f,
                    "to_lat": lat_t, "to_lon": lon_t,
                    "leg_km": leg_km,
                    "from_is_canonical": id_from == c.get("canonical_id"),
                    "to_is_canonical": id_to == c.get("canonical_id"),
                    "from_source_dataset": rf.get("_source_dataset"),
                    "to_source_dataset": rt.get("_source_dataset"),
                    "from_canonical_similarity": sim_by_id.get(id_from),
                    "to_canonical_similarity": sim_by_id.get(id_to),
                },
            ))

    keyed_edges.sort(key=lambda t: t[0])
    return [edge for _, edge in keyed_edges]


def build_canonical_distance_points(
    enriched_records: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    id_field: str = "id",
    date_field: Optional[str] = None,
    text_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """One point per non-canonical cluster member: its temporal and
    haversine distance from that cluster's canonical record — every member
    measured directly against canonical, not against its chronological
    neighbor (build_cluster_edges' chain only connects adjacent dates, so a
    5-member cluster has only 4 edges and canonical is directly linked to at
    most 2 of them; here every member gets exactly one canonical-relative
    point). Feeds the Stage B spatio-temporal spread chart — how far
    (in time and distance) does a cluster's supporting evidence sit from the
    row Stage B picked as its representative."""
    rec_by_id: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(enriched_records):
        rid = str(rec.get(id_field) or rec.get("case_id") or f"ROW-{i + 1}")
        rec_by_id[rid] = rec

    effective_cols = list(text_cols) if text_cols else DEFAULT_EMBED_COLS

    def _preview(rid: str) -> str:
        text = _build_row_text(rec_by_id.get(rid) or {}, effective_cols)
        return text[:220] + ("..." if len(text) > 220 else "")

    points = []
    for c in clusters:
        canonical_id = c.get("canonical_id")
        canon_rec = rec_by_id.get(canonical_id) or {}
        canon_date = _parse_date(canon_rec.get(date_field)) if date_field else None
        canon_lat, canon_lon = canon_rec.get("_resolved_lat"), canon_rec.get("_resolved_lon")
        sim_by_id = {m.get("id"): m.get("canonical_similarity") for m in c.get("member_previews", [])}

        for member_id in c.get("member_ids", []):
            if member_id == canonical_id:
                continue
            rec = rec_by_id.get(member_id) or {}
            member_date = _parse_date(rec.get(date_field)) if date_field else None
            temporal_distance_days = (
                abs((member_date - canon_date).days) if (member_date and canon_date) else None
            )
            m_lat, m_lon = rec.get("_resolved_lat"), rec.get("_resolved_lon")
            haversine_km = None
            if None not in (m_lat, m_lon, canon_lat, canon_lon):
                haversine_km = round(_haversine_km(m_lat, m_lon, canon_lat, canon_lon), 2)
            points.append({
                "cluster_id": c["cluster_id"],
                "cluster_size": c["size"],
                "member_id": member_id,
                "canonical_id": canonical_id,
                "temporal_distance_days": temporal_distance_days,
                "haversine_km": haversine_km,
                "canonical_similarity": sim_by_id.get(member_id),
                "member_text": _preview(member_id),
                "canonical_text": _preview(canonical_id),
            })
    return points


# Column order for the pairs (Stage A) export CSV — mirrors the Interactive
# Multi-Gate Audit Trail table in the UI (id_a/id_b/similarity/distance/date
# diff/witness text/gate flags), plus similarity_bin (Semantic Similarity
# Bins) and the full untruncated narrative text + resolved date/coords for
# each side, so a row can be matched back to the source dataset by id or by
# narrative without carrying every original column along.
PAIRS_EXPORT_COLUMNS = [
    "id_a", "id_b", "similarity", "similarity_bin",
    "haversine_km", "location_name_similarity", "location_source", "date_diff_days",
    "date_a", "date_b", "lat_a", "lon_a", "lat_b", "lon_b",
    "is_similar_text", "is_similar_date", "is_similar_location",
    "is_similar_text_date", "is_similar_both", "is_similar_all",
    "text_a", "text_b",
]


def build_pairs_export_rows(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flattens the Stage A pairwise audit trail (run_advanced_dedup's
    `pairs`, including their nested `flags`) into one row per pair for the
    pairs_audit_trail.csv export — the full candidate-pair list, not just
    the subset that made it into a cluster, so a pair like Case-1820 vs
    Case-1840 is present in the export even if it never reached Stage B."""
    rows = []
    for p in pairs:
        flags = p.get("flags") or {}
        rows.append({
            "id_a": p.get("id_a"),
            "id_b": p.get("id_b"),
            "similarity": p.get("similarity"),
            "similarity_bin": p.get("bin"),
            "haversine_km": p.get("haversine_km"),
            "location_name_similarity": p.get("location_name_similarity"),
            "location_source": p.get("location_source"),
            "date_diff_days": p.get("date_diff_days"),
            "date_a": p.get("date_a"),
            "date_b": p.get("date_b"),
            "lat_a": p.get("lat_a"),
            "lon_a": p.get("lon_a"),
            "lat_b": p.get("lat_b"),
            "lon_b": p.get("lon_b"),
            "is_similar_text": flags.get("is_similar_text"),
            "is_similar_date": flags.get("is_similar_date"),
            "is_similar_location": flags.get("is_similar_location"),
            "is_similar_text_date": flags.get("is_similar_text_date"),
            "is_similar_both": flags.get("is_similar_both"),
            "is_similar_all": flags.get("is_similar_all"),
            # Full narrative text when present (runs computed after this
            # field was added); falls back to the 180-char preview for
            # pairs/zips exported by an older run.
            "text_a": p.get("text_a", p.get("text_a_preview")),
            "text_b": p.get("text_b", p.get("text_b_preview")),
        })
    return rows


def _finalize_clusters(
    cluster_id_groups: List[List[str]],
    row_by_id: Dict[str, Dict[str, Any]],
    preview_cols: Optional[List[str]],
    effective_cols: List[str],
    model_name: str,
    id_prefix: str,
    date_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Turn raw id groups (already filtered to size >= 2) into the
    cluster/member_preview shape the API returns — canonical selection,
    each member's cosine similarity to that canonical row (batched into one
    encode call across every group), and the quick-glance preview values.
    Shared by Stage B's strict (date+location+text) clusters and the
    text-only exact-duplicate pass, so "canonical" and "proximity" mean the
    same thing in both."""
    canonical_by_idx: Dict[int, str] = {
        i: sorted(ids, key=lambda rid: (-_row_completeness(row_by_id[rid]), str(rid)))[0]
        for i, ids in enumerate(cluster_id_groups)
    }

    canonical_similarity: Dict[str, float] = {}
    if cluster_id_groups:
        all_member_ids = [rid for ids in cluster_id_groups for rid in ids]
        member_texts = [_build_row_text(row_by_id[rid], effective_cols) for rid in all_member_ids]
        member_vecs = _encode_texts(member_texts, model_name)
        vec_by_id = {rid: member_vecs[i] for i, rid in enumerate(all_member_ids)}
        for i, ids in enumerate(cluster_id_groups):
            canonical_id = canonical_by_idx[i]
            canonical_vec = vec_by_id[canonical_id]
            for rid in ids:
                canonical_similarity[rid] = 1.0 if rid == canonical_id else round(
                    _cosine_sim(vec_by_id[rid], canonical_vec), 4
                )

    def _member_preview(rid: str) -> Dict[str, Any]:
        row = row_by_id.get(rid) or {}
        if preview_cols:
            values = {c: row.get(c) for c in preview_cols}
        else:
            values = {k: v for k, v in list(row.items())[:8]}
        for qc in QUICK_GLANCE_COLS:
            if qc not in values and row.get(qc) not in (None, ""):
                values[qc] = row.get(qc)
        preview: Dict[str, Any] = {"id": rid, "values": values}
        if rid in canonical_similarity:
            preview["canonical_similarity"] = canonical_similarity[rid]
        # Resolved date, for the Cluster Preview's "sort chronologically"
        # button — independent of whether date_col happens to also be one
        # of the embedding preview_cols.
        resolved_date = _resolve_date(row, date_col)
        if resolved_date:
            preview["date"] = resolved_date.isoformat()
        return preview

    clusters = []
    for i, ids in enumerate(cluster_id_groups):
        sorted_ids = sorted(ids)
        clusters.append({
            "cluster_id": f"{id_prefix}-{len(clusters) + 1}",
            "size": len(ids),
            "member_ids": sorted_ids,
            "canonical_id": canonical_by_idx[i],
            "preview_cols": preview_cols or [],
            "member_previews": [_member_preview(rid) for rid in sorted_ids],
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def run_advanced_dedup(
    records: List[Dict[str, Any]],
    cols: Optional[List[str]] = None,
    threshold: float = 0.80,
    date_diff_days: int = 3,
    max_km: float = 50.0,
    top_k: int = 5,
    max_pairs: Optional[int] = None,
    model_name: str = HARRIER_MODEL,
    use_llm_judge: bool = False,
    date_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    location_col: Optional[str] = None,
    state_col: Optional[str] = None,
    city_col: Optional[str] = None,
    use_gazetteer: bool = False,
    use_cluster_blocking: bool = False,
    block_min_cluster_size: int = 15,
) -> Dict[str, Any]:
    """
    Multi-Gate Batch Clustering Pipeline — two chained but separately-reported
    stages, each independently auditable:

      Stage A (screening): reuses the exact same live pairwise engine as the
      Cross-DB pipeline (respects the selected embedding columns and the
      threshold/date/km gates) to produce the full candidate-pair audit
      trail — nothing here is precomputed or cached.

      Stage B (clustering): unions only the pairs that satisfy every gate
      at once (`is_similar_all` — text AND date AND location all agree, the
      highest-confidence signal) into connected components via union-find,
      then picks one canonical row per cluster (most complete row wins).
      Using the strictest gate here — rather than text similarity alone —
      keeps a single borderline match from chaining unrelated clusters
      together.

    `use_llm_judge` is accepted for UI/API compatibility but is currently a
    no-op (no LLM confirmation call is made in this pipeline).
    """
    if not records:
        return {"status": "error", "message": "No records provided for deduplication."}

    stage_a = run_cross_db_pipeline(
        records_a=records,
        records_b=None,
        cols_a=cols,
        threshold=threshold,
        max_days=date_diff_days,
        max_km=max_km,
        top_k=top_k,
        max_pairs=max_pairs,
        model_name=model_name,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        location_col=location_col,
        state_col=state_col,
        city_col=city_col,
        use_gazetteer=use_gazetteer,
        use_cluster_blocking=use_cluster_blocking,
        block_min_cluster_size=block_min_cluster_size,
    )
    if stage_a.get("status") != "success":
        return stage_a

    pairs = stage_a["pairs"]

    uf = _UnionFind()
    # Independent second pass: unions purely on the Easy-Path "exact_duplicate"
    # text bin (cosine sim >= 0.88), ignoring date/location entirely. Kept as
    # its own union-find rather than merged into `uf` above — mixing the two
    # edge sets into one graph would let a single near-identical-but-generic
    # pair transitively chain together clusters that Stage A's location/date
    # gates deliberately kept apart. Catches the case where a pair is a
    # near-certain duplicate by text alone but never reached Stage B because
    # location couldn't be resolved (or was, and is just wrong/too coarse).
    text_uf = _UnionFind()
    row_by_id: Dict[str, Dict[str, Any]] = {}
    for p in pairs:
        row_by_id.setdefault(p["id_a"], p.get("row_a") or {})
        row_by_id.setdefault(p["id_b"], p.get("row_b") or {})
        if p["flags"]["is_similar_all"]:
            uf.union(p["id_a"], p["id_b"])
        if p["bin"] == "exact_duplicate":
            text_uf.union(p["id_a"], p["id_b"])

    def _groups(union_find: _UnionFind) -> List[List[str]]:
        buckets: Dict[str, List[str]] = {}
        for node_id in row_by_id:
            buckets.setdefault(union_find.find(node_id), []).append(node_id)
        return [ids for ids in buckets.values() if len(ids) >= 2]

    # Preview columns: the same embedding columns the caller selected (`cols`),
    # so the cluster preview shows exactly what the similarity was computed
    # from for every member — not an arbitrary slice of the canonical row.
    preview_cols = list(cols) if cols else None
    effective_cols = list(cols) if cols else DEFAULT_EMBED_COLS

    clusters = _finalize_clusters(
        _groups(uf), row_by_id, preview_cols, effective_cols, model_name, id_prefix="CLUSTER", date_col=date_col,
    )
    # Text-exact duplicates: near-certain re-filed duplicates by wording
    # alone, independent of whether Stage B's stricter gate accepted them.
    # strong_similar (0.80-0.88) pairs are deliberately *not* auto-merged
    # here — that band needs a human or LLM to confirm, not a threshold.
    text_duplicate_clusters = _finalize_clusters(
        _groups(text_uf), row_by_id, preview_cols, effective_cols, model_name, id_prefix="TEXTDUPE", date_col=date_col,
    )

    total_clusters = len(clusters)
    rows_in_clusters = sum(c["size"] for c in clusters)
    redundant_rows_saved = rows_in_clusters - total_clusters

    text_dup_rows_in_clusters = sum(c["size"] for c in text_duplicate_clusters)
    text_dup_redundant_rows_saved = text_dup_rows_in_clusters - len(text_duplicate_clusters)

    # Union of both duplicate sets — the actual row count you'd drop by
    # keeping only the canonical from *both* passes, since a row can be a
    # duplicate under either criterion independently.
    duplicate_ids: set = set()
    for c in clusters:
        duplicate_ids.update(rid for rid in c["member_ids"] if rid != c["canonical_id"])
    for c in text_duplicate_clusters:
        duplicate_ids.update(rid for rid in c["member_ids"] if rid != c["canonical_id"])

    return {
        "status": "success",
        "parameters": {
            "threshold": threshold,
            "date_diff_days": date_diff_days,
            "max_km": max_km,
            "use_llm_judge": use_llm_judge,
        },
        "summary": {
            **stage_a["summary"],
            "total_clusters": total_clusters,
            "rows_in_clusters": rows_in_clusters,
            "redundant_rows_saved": redundant_rows_saved,
            "flagged_pairs_count": len(pairs),
            "text_duplicate_clusters_count": len(text_duplicate_clusters),
            "text_duplicate_rows_in_clusters": text_dup_rows_in_clusters,
            "text_duplicate_redundant_rows_saved": text_dup_redundant_rows_saved,
            "combined_redundant_rows_saved": len(duplicate_ids),
        },
        "pairs": pairs,        # Stage A — full multi-gate audit trail
        "clusters": clusters,  # Stage B — strict (text+date+location) canonical clustering
        # Text-only exact-duplicate pass (sim >= 0.88, gates ignored) — union
        # with `clusters`' non-canonical members to get the full prunable set.
        "text_duplicate_clusters": text_duplicate_clusters,
    }


def run_cross_db_pipeline(
    records_a: List[Dict[str, Any]],
    records_b: Optional[List[Dict[str, Any]]] = None,
    cols_a: Optional[List[str]] = None,
    cols_b: Optional[List[str]] = None,
    threshold: float = 0.80,
    max_days: int = 3,
    max_km: float = 50.0,
    top_k: int = 5,
    max_pairs: Optional[int] = None,
    model_name: str = HARRIER_MODEL,
    max_rows: Optional[int] = None,
    date_col: Optional[str] = None,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    location_col: Optional[str] = None,
    state_col: Optional[str] = None,
    city_col: Optional[str] = None,
    # Dataset B's own column names, for between_datasets comparisons where
    # the two databases don't share a schema. Each defaults to the Dataset A
    # value above when not given, so single-schema callers (including every
    # within_dataset call) need to specify column names only once.
    date_col_b: Optional[str] = None,
    lat_col_b: Optional[str] = None,
    lon_col_b: Optional[str] = None,
    location_col_b: Optional[str] = None,
    state_col_b: Optional[str] = None,
    city_col_b: Optional[str] = None,
    location_name_threshold: float = 0.75,
    use_gazetteer: bool = False,
    use_cluster_blocking: bool = False,
    block_min_cluster_size: int = 15,
) -> Dict[str, Any]:
    """
    Cross-DB Similarity & Deduplication Pipeline supporting dual-path workflows:
    - Easy Path: Semantic similarity binning ('exact_duplicate', 'strong_similar', 'moderate_similar', 'distinct')
    - Hard Path: Interactive multi-gate boolean flags ('is_similar_text', 'is_similar_date', 'is_similar_location', 'is_similar_both', 'is_similar_all')

    date_col / lat_col / lon_col / location_col let the caller point at
    whichever columns actually hold that data instead of relying on the
    guessed common names (date_time/date, latitude/lat, longitude/lon) —
    those guesses stay the default so nothing breaks for datasets that
    already match them. When numeric coordinates don't resolve for a row,
    the location gate falls back through two more tiers, in order:
      1. (opt-in, use_gazetteer=True) an offline US Census Gazetteer
         city/state -> centroid lookup. Prefers a dedicated city_col (+
         state_col) when given — a clean, already-separated city name is a
         far more reliable gazetteer key than one parsed out of a free-text
         location_col sentence. Falls back to location_col (paired with
         state_col, or a parsed "City, ST" string in location_col alone)
         when city_col isn't given or doesn't resolve. Gives a real
         haversine distance. Off by default — it's US-only and a place name
         can collide across states/countries, so it shouldn't silently
         activate for every location-name dataset.
      2. text similarity on location_col, if the gazetteer is off or
         doesn't resolve.

    use_cluster_blocking: the default pairwise search computes a dense
    len(eval_a) x len(eval_b) cosine similarity matrix, which stops being
    feasible somewhere in the tens-of-thousands of rows (100k x 100k floats
    is 40GB before the matmul even runs). When True, rows are first grouped
    into UMAP+HDBSCAN blocks (same technique as the Analysis tab's cluster
    pipeline, see _hdbscan_block_labels) and only same-block rows are
    compared, trading a small amount of recall — a true duplicate pair that
    lands in two different blocks near a cluster boundary would be missed —
    for comparisons that scale with block size instead of dataset size.
    Off by default so behavior doesn't silently change for existing callers.
    """
    if not records_a:
        return {"status": "error", "message": "No candidate records provided for Dataset A."}

    mode = "between_datasets" if (records_b is not None and len(records_b) > 0) else "within_dataset"
    if mode == "within_dataset":
        records_b = records_a
        cols_b = cols_a
        date_col_b = date_col
        lat_col_b = lat_col
        lon_col_b = lon_col
        location_col_b = location_col
        state_col_b = state_col
        city_col_b = city_col
    else:
        date_col_b = date_col_b if date_col_b is not None else date_col
        lat_col_b = lat_col_b if lat_col_b is not None else lat_col
        lon_col_b = lon_col_b if lon_col_b is not None else lon_col
        location_col_b = location_col_b if location_col_b is not None else location_col
        state_col_b = state_col_b if state_col_b is not None else state_col
        city_col_b = city_col_b if city_col_b is not None else city_col

    cols_a = cols_a or DEFAULT_EMBED_COLS
    cols_b = cols_b or cols_a

    # No implicit truncation — every row the caller provides is evaluated.
    # Pass max_rows explicitly if you want to cap it (e.g. for a UI preview).
    eval_a = records_a[:max_rows] if max_rows else records_a
    eval_b = (records_b[:max_rows] if max_rows else records_b)

    # Resolve each row's coordinates once (not once per pair) — with top_k
    # candidates per row, a row can appear in many pairs, and both the
    # gazetteer dict lookup and the "City, ST" text parse are wasted work
    # if repeated per pair rather than cached per row up front. Dataset B
    # uses its own column mapping (identical to A's in within_dataset mode).
    coords_a = [_resolve_effective_coord(r, lat_col, lon_col, location_col, state_col, use_gazetteer, city_col) for r in eval_a]
    coords_b = coords_a if mode == "within_dataset" else [
        _resolve_effective_coord(r, lat_col_b, lon_col_b, location_col_b, state_col_b, use_gazetteer, city_col_b) for r in eval_b
    ]

    texts_a = [_build_row_text(r, cols_a) for r in eval_a]
    texts_b = [_build_row_text(r, cols_b) for r in eval_b] if mode == "between_datasets" else texts_a

    # Encode in one batched call per side (was one-row-at-a-time — didn't
    # scale once the 500-row cap was lifted).
    vecs_a = _encode_texts(texts_a, model_name)
    vecs_b = _encode_texts(texts_b, model_name) if mode == "between_datasets" else vecs_a
    
    def _local_top_k(block_vecs_a: np.ndarray, block_vecs_b: np.ndarray,
                      idx_a: list, idx_b: list, same_rows: bool) -> list:
        """Top-k (global_i, global_j, sim) candidates for one A/B slice —
        the same normalize + matmul + argsort top_k used for the full
        matrix, just scoped to whatever rows are passed in."""
        na = np.linalg.norm(block_vecs_a, axis=1, keepdims=True); na[na == 0] = 1e-10
        nb = np.linalg.norm(block_vecs_b, axis=1, keepdims=True); nb[nb == 0] = 1e-10
        local_sim = np.dot(block_vecs_a / na, (block_vecs_b / nb).T)
        if same_rows:
            for li, gi in enumerate(idx_a):
                for lj, gj in enumerate(idx_b):
                    if gi == gj:
                        local_sim[li, lj] = -1.0
        out = []
        for li in range(local_sim.shape[0]):
            row_sims = local_sim[li]
            top_local = np.argsort(row_sims)[::-1][:top_k]
            for lj in top_local:
                out.append((idx_a[li], idx_b[int(lj)], float(row_sims[lj])))
        return out

    block_stats = None
    candidates: list = []
    if use_cluster_blocking:
        labels_a, labels_b = _compute_block_labels(vecs_a, vecs_b, mode, block_min_cluster_size)
        blocks: Dict[Any, tuple] = {}
        for idx, lbl in enumerate(labels_a):
            blocks.setdefault(lbl, ([], []))[0].append(idx)
        for idx, lbl in enumerate(labels_b):
            blocks.setdefault(lbl, ([], []))[1].append(idx)
        compared_cells = 0
        for lbl, (idx_a, idx_b) in blocks.items():
            if not idx_a or not idx_b:
                continue
            candidates.extend(_local_top_k(
                vecs_a[idx_a], vecs_b[idx_b], idx_a, idx_b,
                same_rows=(mode == "within_dataset"),
            ))
            compared_cells += len(idx_a) * len(idx_b)
        block_stats = {
            "block_count": len(blocks),
            "noise_block_size": len(blocks.get(-1, ([], []))[0]),
            "full_matrix_cells": len(eval_a) * len(eval_b),
            "compared_cells": compared_cells,
        }
    else:
        # Cosine similarity matrix via matrix multiplication
        norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        norms_a[norms_a == 0] = 1e-10
        norms_b[norms_b == 0] = 1e-10

        sim_matrix = np.dot(vecs_a / norms_a, (vecs_b / norms_b).T)
        if mode == "within_dataset":
            np.fill_diagonal(sim_matrix, -1.0)

        for i in range(len(eval_a)):
            row_sims = sim_matrix[i]
            top_indices = np.argsort(row_sims)[::-1][:top_k]
            for j in top_indices:
                candidates.append((i, int(j), float(row_sims[j])))

    pairs = []
    # Build each candidate pair's full multi-gate record
    for i, j, sim in candidates:
            if sim < 0.60:
                continue

            rec_a = eval_a[i]
            rec_b = eval_b[j]
            # Within a single dataset, row i can appear as either the "A" or "B"
            # side of different pairs — an A-/B- prefixed fallback would then
            # label the same physical row two different ways and fracture any
            # graph (e.g. union-find clustering) built over these ids.
            if mode == "within_dataset":
                id_a = str(rec_a.get("id") or rec_a.get("case_id") or f"ROW-{i+1}")
                id_b = str(rec_b.get("id") or rec_b.get("case_id") or f"ROW-{j+1}")
            else:
                id_a = str(rec_a.get("id") or rec_a.get("case_id") or f"A-{i+1}")
                id_b = str(rec_b.get("id") or rec_b.get("case_id") or f"B-{j+1}")
            
            # Distance calculation — precomputed per row above (coords_a/b):
            # real lat/lon columns, else an offline US gazetteer centroid
            # lookup. Only falls all the way to location-name text
            # similarity when neither row resolved to any coordinate.
            haversine_km = None
            location_name_sim = None
            location_source = None
            resolved_a = coords_a[i]
            resolved_b = coords_b[j]
            if resolved_a and resolved_b:
                (lat_a, lon_a), location_source = resolved_a
                (lat_b, lon_b), _ = resolved_b
                haversine_km = round(_haversine_km(lat_a, lon_a, lat_b, lon_b), 2)
            elif location_col or location_col_b:
                location_name_sim = _location_name_similarity(
                    rec_a.get(location_col) if location_col else None,
                    rec_b.get(location_col_b) if location_col_b else None,
                )
                location_source = "name_text" if location_name_sim is not None else None

            # Date difference calculation — explicit date_col (per side) when given.
            date_diff_days = None
            da = _resolve_date(rec_a, date_col)
            db = _resolve_date(rec_b, date_col_b)
            if da and db:
                date_diff_days = abs((da - db).days)

            # Easy Path: Semantic Binning
            bin_label = "distinct"
            if sim >= 0.88:
                bin_label = "exact_duplicate"
            elif sim >= 0.80:
                bin_label = "strong_similar"
            elif sim >= 0.70:
                bin_label = "moderate_similar"

            # Hard Path: Multi-Gate Boolean Flags
            is_similar_text = bool(sim >= threshold)
            is_similar_date = bool(date_diff_days is not None and date_diff_days <= max_days)
            is_similar_location = bool(
                (haversine_km is not None and haversine_km <= max_km)
                or (location_name_sim is not None and location_name_sim >= location_name_threshold)
            )
            is_similar_both = bool(is_similar_date and is_similar_location)
            is_similar_text_date = bool(is_similar_text and is_similar_date)
            is_similar_all = bool(is_similar_text and is_similar_both)

            pairs.append({
                "id_a": id_a,
                "id_b": id_b,
                "similarity": round(sim, 4),
                "bin": bin_label,
                "haversine_km": haversine_km,
                "location_name_similarity": round(location_name_sim, 3) if location_name_sim is not None else None,
                # How the location gate was evaluated for this pair:
                # 'coords' (real lat/lon), 'gazetteer' (offline US city/state
                # centroid lookup), 'name_text' (fuzzy string match), or None.
                "location_source": location_source,
                "date_diff_days": date_diff_days,
                # Full (untruncated) narrative text — the *_preview fields
                # above are clipped to 180 chars for the UI table, but a
                # pairs export needs the complete text so it can be matched
                # back against the source dataset by ID or by narrative.
                "text_a": texts_a[i],
                "text_b": texts_b[j],
                "text_a_preview": texts_a[i][:180] + ("..." if len(texts_a[i]) > 180 else ""),
                "text_b_preview": texts_b[j][:180] + ("..." if len(texts_b[j]) > 180 else ""),
                "date_a": da.isoformat() if da else None,
                "date_b": db.isoformat() if db else None,
                "lat_a": resolved_a[0][0] if resolved_a else None,
                "lon_a": resolved_a[0][1] if resolved_a else None,
                "lat_b": resolved_b[0][0] if resolved_b else None,
                "lon_b": resolved_b[0][1] if resolved_b else None,
                "row_a": rec_a,
                "row_b": rec_b,
                "flags": {
                    "is_similar_text": is_similar_text,
                    "is_similar_date": is_similar_date,
                    "is_similar_location": is_similar_location,
                    "is_similar_both": is_similar_both,
                    "is_similar_text_date": is_similar_text_date,
                    "is_similar_all": is_similar_all
                }
            })
            if max_pairs is not None and len(pairs) >= max_pairs:
                break

    # Sort pairs descending by similarity
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Calculate summary KPIs across Easy & Hard paths
    bin_counts = {"exact_duplicate": 0, "strong_similar": 0, "moderate_similar": 0, "distinct": 0}
    gate_counts = {"similar_text": 0, "similar_date": 0, "similar_location": 0, "similar_text_date": 0, "similar_both": 0, "similar_all": 0}

    for p in pairs:
        b = p["bin"]
        if b in bin_counts:
            bin_counts[b] += 1
        f = p["flags"]
        if f["is_similar_text"]: gate_counts["similar_text"] += 1
        if f["is_similar_date"]: gate_counts["similar_date"] += 1
        if f["is_similar_location"]: gate_counts["similar_location"] += 1
        if f["is_similar_text_date"]: gate_counts["similar_text_date"] += 1
        if f["is_similar_both"]: gate_counts["similar_both"] += 1
        if f["is_similar_all"]: gate_counts["similar_all"] += 1
        
    return {
        "status": "success",
        "mode": mode,
        "summary": {
            "total_pairs_evaluated": len(pairs),
            "bins": bin_counts,
            "gate_counts": gate_counts,
            "block_stats": block_stats,
        },
        "pairs": pairs
    }

