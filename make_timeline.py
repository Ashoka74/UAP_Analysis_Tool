"""Prototype: deterministic core of the CUFOS-style UFO timeline renderer.

Stage (a) of the timeline strategy — NO LLM. Renders structured rows of
PURSUE_1_2_3_normalized_full into chronological timeline entries:

    {Year}, {Month} {Day-expr} — [{Time}] {Narrative verbatim}. ({Citations})

Implements transformations A (field normalization), B (date-expression
synthesis), D (citation assembly), E (chronological ordering). The narrative is
passed through verbatim from `sightingDetails.DenseNarrativeSection`; restyling
(stage C) and the secondary-bibliography join are intentionally out of scope.
"""
import os
import re
from pathlib import Path
import pandas as pd

SRC = "SUBDATASETS_V2/PURSUE_1_2_3_normalized_full.csv"
# Authoritative LLM-confirmed dedup clusters (dup_cluster_id / dup_role /
# dup_canonical_row, from apply_dedup.py) live here. When present, the timeline
# roots each entry at its canonical row and cites every member's pages+sources.
DEDUP_SRC = "PURSUE_1_2_3_dedup_flagged.csv"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# ── A. field normalization helpers ──────────────────────────────────────────
def _int_or_none(v):
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def fmt_time(row):
    """`23:00` -> `11:00 p.m.`; qualitative / missing -> ''."""
    for col in ("date_time.local_time", "sightingDetails.timeOfDay"):
        v = row.get(col)
        if pd.isna(v):
            continue
        m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(v))
        if not m:
            continue
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            continue
        suffix = "a.m." if h < 12 else "p.m."
        h12 = h % 12 or 12
        return f"{h12}:{mi:02d} {suffix}"
    return ""


def fmt_pages(token):
    """`page_0151-page_0155` -> `pp. 151–155`; `page_0096` -> `p. 96`."""
    if pd.isna(token):
        return ""
    nums = [int(n) for n in re.findall(r"page_(\d+)", str(token))]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    return f"p. {lo}" if lo == hi else f"pp. {lo}–{hi}"


# ── B. date-expression synthesis ────────────────────────────────────────────
def _day_expr_from_description(desc, month_name):
    """Pull a richer day expression out of the free-text description when the
    numeric day is only a point estimate. Conservative — returns None if nothing
    clearly day-like is found."""
    if not isinstance(desc, str):
        return None
    # "mid-/early-/late-December" fuzzy month
    m = re.search(r"\b(mid|early|late)[\s-]+" + re.escape(month_name), desc, re.I)
    if m:
        return f"{m.group(1).lower()}-{month_name}"
    # day range: "between 15th and 20th", "15-20", "14 to 28" (1–31 only,
    # so 4-digit times like 2115-2130 never match)
    m = re.search(r"\b([12]?\d|3[01])(?:st|nd|rd|th)?\s*(?:and|to|through|[-–])\s*"
                  r"([12]?\d|3[01])(?:st|nd|rd|th)?\b", desc)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 31 and 1 <= b <= 31 and a != b:
            return f"{a}–{b}"
    # alternates: "17 or 19"
    m = re.search(r"\b([12]?\d|3[01])(?:st|nd|rd|th)?\s+or\s+([12]?\d|3[01])(?:st|nd|rd|th)?\b", desc)
    if m:
        return f"{m.group(1)} or {m.group(2)}"
    return None


def fmt_date_header(row):
    """Year + Month + day-expression, reconciling numeric date with uncertainty
    and the free-text description."""
    year = _int_or_none(row.get("date_time.year"))
    month = _int_or_none(row.get("date_time.month"))
    day = _int_or_none(row.get("date_time.day"))
    unc = row.get("date_time.dateUncertainty")
    desc = row.get("date_time.description")

    if year is None:
        return None  # undatable — excluded from the timeline
    # `unknown` rows carry synthetic point-fills (month=1, day=1 etc.); trusting
    # them invents bogus "January 1" entries, so drop to year-only unless the
    # free-text description supplies a real month/day.
    if unc == "unknown":
        return str(year)
    if month is None or not (1 <= month <= 12):
        return str(year)
    month_name = MONTHS[month - 1]

    # Enrich the day expression from the description whenever the date isn't exact.
    day_expr = None
    if unc != "exact":
        day_expr = _day_expr_from_description(desc, month_name)
    if day_expr is None and day is not None and 1 <= day <= 31:
        day_expr = str(day)

    seg = f"{month_name} {day_expr}" if day_expr else month_name
    return f"{year}, {seg}"


# ── D. citation assembly — clickable, document-tagged provenance ─────────────
# Each reference is a Markdown link to the document's war.gov PDF (PDF | Image
# Link), labelled with the document identity + page(s). The URL is wrapped in
# <…> so the handful of filenames containing spaces stay valid CommonMark.
_LINK_STRIP = str.maketrans({"[": "(", "]": ")"})


def _doc_url(row):
    u = row.get("PDF | Image Link")
    if pd.notna(u) and str(u).strip().lower().startswith("http"):
        return str(u).strip()
    return ""


def _doc_label(row):
    """Short human identity for a document: Title prefix → source.ref → PDF stem."""
    lab = ""
    for src in (row.get("Title"), row.get("source.ref")):
        if pd.notna(src) and str(src).strip():
            lab = str(src).split(",")[0].strip()
            if lab:
                break
    if not lab:
        u = _doc_url(row)
        lab = Path(u).stem if u else ""
    lab = (lab or "PURSUE release").translate(_LINK_STRIP)
    return lab[:50].rstrip()


def _doc_ref(row):
    """`[label, pp. N–M](<pdf url>)` — clickable; falls back to plain text when
    no PDF URL is present."""
    pages = fmt_pages(row.get("pages")) or fmt_pages(row.get("source.pages"))
    label = _doc_label(row)
    text = f"{label}, {pages}" if pages else label
    url = _doc_url(row)
    return f"[{text}](<{url}>)" if url else text


def fmt_citation(row):
    ref = _doc_ref(row)
    return f"({ref})" if ref else ""


def fmt_citation_cluster(rows):
    """One clickable reference per distinct document in the cluster (deduped by
    PDF URL), so every member's page + source is individually traceable —
    e.g. `([fbi-photo-b14, p. 1](…); [fbi-photo-b17, p. 1](…); …)`."""
    seen, parts = set(), []
    for r in rows:
        ref = _doc_ref(r)
        if not ref:
            continue
        key = _doc_url(r) or ref
        if key not in seen:
            seen.add(key)
            parts.append(ref)
    return f"({'; '.join(parts)})" if parts else ""


# ── Dedupe-merge: block hard on date+location, confirm within block ─────────
def _loc_key(row):
    """Normalized location for blocking. '' when no location is available
    (rows with no location never block together — they stay separate)."""
    for c in ("location.city", "location.state",
              "sightingDetails.location.name", "location.description"):
        v = row.get(c)
        if pd.notna(v) and str(v).strip():
            s = re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()
            if s:
                return s
    return ""


def _block_key(row):
    """Candidate-blocking key: only EXACT, fully-dated rows with a location are
    eligible to merge. Everything else returns None ⇒ never a merge candidate
    (uncertain dates are the highest FP risk, so we refuse to auto-merge them)."""
    if row.get("date_time.dateUncertainty") != "exact":
        return None
    y, m, d = (_int_or_none(row.get("date_time.year")),
               _int_or_none(row.get("date_time.month")),
               _int_or_none(row.get("date_time.day")))
    loc = _loc_key(row)
    if None in (y, m, d) or not loc:
        return None
    return (y, m, d, loc)


def _narrative_sim(a, b):
    """Lexical proxy for the in-block semantic confirmer (token Jaccard + char
    ratio). Swap in cosine over real embeddings without changing the pipeline —
    TN/FP is governed by blocking, this only sharpens TP vs FP inside a block."""
    from difflib import SequenceMatcher
    a, b = (a or "").lower(), (b or "").lower()
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.5 * jac + 0.5 * SequenceMatcher(None, a, b).ratio()


SIM_TAU = 0.55          # conservative; in prod calibrate against a labeled set / LLM judge


def _cluster_block(idxs, narratives):
    """Clique (all-pairs ≥ τ) linkage within one block, so a single stray edge
    can't chain distinct events together. Returns list of index-clusters."""
    n = len(idxs)
    clusters = []
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            if all(_narrative_sim(narratives[g], narratives[j]) >= SIM_TAU for g in group):
                group.append(j); used[j] = True
        used[i] = True
        clusters.append([idxs[g] for g in group])
    return clusters


# ── E. ordering + assembly ──────────────────────────────────────────────────
_NARR = "sightingDetails.DenseNarrativeSection"


def _render(header, rows):
    """One timeline entry from a cluster of 1+ rows (canonical = highest
    trustScore, tiebreak longest narrative)."""
    canon = max(rows, key=lambda r: (
        pd.to_numeric(r.get("sightingDetails.trustScore"), errors="coerce") or 0,
        len(str(r.get(_NARR, ""))),
    ))
    time = fmt_time(canon)
    narrative = str(canon.get(_NARR, "")).strip()
    cite = fmt_citation_cluster(rows) if len(rows) > 1 else fmt_citation(canon)
    lead = f"{time} " if time else ""
    tag = f" _(merged ×{len(rows)})_" if len(rows) > 1 else ""
    return f"**{header}**{tag} — {lead}{narrative} {cite}".rstrip()


def _render_entry(header, root, members):
    """Canonical-rooted entry: date/time/narrative come from the canonical
    ``root`` row, while citations are unioned across ALL ``members`` of its
    cluster (pages + sources). Singletons pass ``members == [root]``."""
    time = fmt_time(root)
    narrative = str(root.get(_NARR, "")).strip()
    cite = fmt_citation_cluster(members) if len(members) > 1 else fmt_citation(root)
    lead = f"{time} " if time else ""
    tag = f" _(cluster ×{len(members)})_" if len(members) > 1 else ""
    return f"**{header}**{tag} — {lead}{narrative} {cite}".rstrip()


# Source-tier groupings for the `tiers=` filter. First-hand (primary +
# retrospective personal account) and second-hand (secondary) witness testimony;
# excludes press, reference_book, and unknown-provenance tiers.
TESTIMONY_TIERS = {"primary_investigative", "secondary_investigative",
                   "retrospective_personal_account"}


def build(n=40, src=None, tiers=None):
    """Dispatch on data shape. When the source carries authoritative dedup
    clusters (``dup_cluster_id`` / ``dup_role`` from apply_dedup.py), render one
    entry per cluster rooted at its canonical row and cite pages+sources from
    EVERY member. Otherwise fall back to in-file lexical clustering.

    tiers: optional set of allowed ``source.tier`` values; rows outside it are
    dropped before clustering (e.g. TESTIMONY_TIERS excludes press/books)."""
    src = src or (DEDUP_SRC if os.path.exists(DEDUP_SRC) else SRC)
    df = pd.read_csv(src, low_memory=False)
    if tiers is not None and "source.tier" in df.columns:
        df = df[df["source.tier"].isin(tiers)].copy()
    df = df[df["date_time.year"].notna()].copy()
    df = df[df["date_time.year"] > 1000]          # drop year=0 parse-error rows
    df["_y"] = df["date_time.year"].astype(float)
    df["_m"] = df["date_time.month"].fillna(13).astype(float)   # undated month sorts last within year
    df["_d"] = df["date_time.day"].fillna(32).astype(float)
    df["_t"] = df["date_time.local_time"].fillna("99:99").astype(str)
    if "dup_cluster_id" in df.columns:
        return _build_clustered(df, n)
    return _build_legacy(df, n)


def _root_score(r):
    """Canonical-selection key (matches apply_dedup): highest trustScore,
    tiebreak longest narrative."""
    ts = pd.to_numeric(r.get("sightingDetails.trustScore"), errors="coerce")
    return (ts if pd.notna(ts) else 0, len(str(r.get(_NARR, ""))))


def _build_clustered(df, n):
    """One entry per cluster present in ``df``, rooted at its best SURVIVING
    member (canonical rule above — so an unfiltered build roots at the stored
    canonical, and a tier-filtered build re-roots when the canonical was dropped)
    and citing every surviving member. Singletons render alone."""
    clustered = df[df["dup_cluster_id"].notna()]
    singles = df[df["dup_cluster_id"].isna()]

    render = []   # (root_row, [member_rows])
    for _cid, g in clustered.groupby("dup_cluster_id"):
        members = [r for _, r in g.iterrows()]
        render.append((max(members, key=_root_score), members))
    for _, s in singles.iterrows():
        render.append((s, [s]))

    render.sort(key=lambda rm: (rm[0]["_y"], rm[0]["_m"], rm[0]["_d"], rm[0]["_t"]))
    if n is not None:
        render = render[:n]

    entries, n_clusters = [], 0
    for root, members in render:
        header = fmt_date_header(root)
        if header is None:
            continue
        if len(members) > 1:
            n_clusters += 1
        entries.append(_render_entry(header, root, members))
    return entries, len(render), n_clusters, []


def _build_legacy(df, n):
    """Fallback: in-file lexical block→clique clustering (no dup_cluster_id)."""
    df = df.sort_values(["_y", "_m", "_d", "_t"]).head(n).reset_index(drop=True)
    rows = [r for _, r in df.iterrows()]

    # 1) Block: only exact, fully-dated, located rows are merge candidates.
    blocks = {}
    singletons = []
    for i, r in enumerate(rows):
        bk = _block_key(r)
        (blocks.setdefault(bk, []) if bk is not None else singletons).append(i)

    # 2) Confirm within each block via clique linkage on narrative similarity.
    clusters = []          # each = list of row-indices
    for bk, idxs in blocks.items():
        if len(idxs) == 1:
            clusters.append(idxs)
        else:
            clusters.extend(_cluster_block(idxs, [str(rows[i].get(_NARR, "")) for i in idxs]))
    clusters.extend([i] if isinstance(i, int) else [i] for i in singletons)

    # 3) Review list: pairs we DID NOT merge but that look related — surfaced
    #    as recoverable FNs instead of being silently kept apart. Two kinds:
    #      • same-block but below the merge τ (e.g. the Camp Hood 0.52 case) —
    #        the borderline band an LLM judge / calibrated τ should resolve.
    #      • cross-block (different date/location) yet ≥ τ — possible dup whose
    #        date/location metadata disagrees.
    REVIEW_FLOOR = 0.45
    row_block = {i: _block_key(r) for i, r in enumerate(rows)}
    review = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if any(i in c and j in c for c in clusters):    # already merged
                continue
            s = _narrative_sim(str(rows[i].get(_NARR, "")), str(rows[j].get(_NARR, "")))
            same_block = row_block[i] is not None and row_block[i] == row_block[j]
            if same_block and s >= REVIEW_FLOOR:
                kind = "same-block <τ"
            elif not same_block and s >= SIM_TAU:
                kind = "cross-block ≥τ"
            else:
                continue
            review.append((s, kind, fmt_date_header(rows[i]), fmt_date_header(rows[j])))

    # 4) Render, ordered by the canonical row's date key.
    def _ckey(c):
        r = rows[c[0]]
        return (r["_y"], r["_m"], r["_d"], r["_t"])
    clusters.sort(key=_ckey)

    entries = []
    for c in clusters:
        crows = [rows[i] for i in c]
        header = fmt_date_header(crows[0]) or fmt_date_header(
            max(crows, key=lambda r: len(str(r.get(_NARR, "")))))
        if header is None:
            continue
        entries.append(_render(header, crows))

    n_merged = sum(1 for c in clusters if len(c) > 1)
    return entries, len(rows), n_merged, review


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "40"
    n = None if arg.lower() in ("full", "all") else int(arg)
    testimony = len(sys.argv) > 2 and sys.argv[2].lower() == "testimony"
    tiers = TESTIMONY_TIERS if testimony else None

    base = "timeline_full" if n is None else f"timeline_first{n}"
    out_path = f"{base}_testimony.md" if testimony else f"{base}.md"
    scope = ("full dataset" if n is None else f"first {n} by date") + (
        " · first/second-hand investigative testimony only (no press/books)" if testimony else "")

    entries, n_rows, n_merged, review = build(n, tiers=tiers)
    out = (
        "# UFO Timeline — PURSUE releases (canonical-rooted, cluster-aware)\n\n"
        f"*{scope}: {len(entries)} entries; {n_merged} are clusters rooted at the "
        "canonical row, citing pages + sources from every member (each a clickable "
        "war.gov PDF link). Narrative verbatim from `DenseNarrativeSection`.*\n\n"
        + "\n\n".join(entries) + "\n"
    )
    with open(out_path, "w") as f:
        f.write(out)
    print(f"Wrote {out_path} — {len(entries)} entries "
          f"({n_merged} clusters rooted at canonical).\n")
    for e in entries[:8]:
        print(e + "\n")
    if review:
        print("─" * 70)
        print(f"REVIEW LIST — {len(review)} related pairs NOT auto-merged "
              "(potential FNs for LLM judge / τ calibration):")
        for s, kind, ha, hb in sorted(review, reverse=True)[:10]:
            print(f"  sim={s:.2f}  [{kind}]  «{ha}»  vs  «{hb}»")
