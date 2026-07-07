"""Analysis service — the Categorical Association Explorer (Cramér's V) that runs
directly on raw dataset columns, ported from analyzing.py. Reuses
``uap_analyzer.cramers_v`` for the per-pair statistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical label so missingness isn't fragmented into nan / None / "" etc.
_MISSING_LABEL = "(missing)"
_NULL_STR_TOKENS = {"nan", "none", "null", "<na>", "nat", ""}
_CV_TOL = 1e-6  # values within this of 0 / 1 are "trivial"


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def band_columns(df: pd.DataFrame, high_threshold: int = 30) -> tuple[dict, dict]:
    """Bucket columns into categorical bands by cardinality (see analyzing.py)."""
    bands: dict[str, list[str]] = {
        "binary": [], "low": [], "medium": [], "high": [], "constant": [],
    }
    nunique_map: dict[str, int] = {}
    for c in df.columns:
        nu = _safe_nunique(df[c])
        nunique_map[c] = nu
        if nu <= 1:
            bands["constant"].append(c)
        elif nu == 2:
            bands["binary"].append(c)
        elif nu <= 9:
            bands["low"].append(c)
        elif nu < high_threshold:
            bands["medium"].append(c)
        else:
            bands["high"].append(c)
    return bands, nunique_map


def _eligible_categorical(bands: dict) -> list[str]:
    """Columns the explorer scores by default: binary + low + medium cardinality
    (high-cardinality / free-text and constant columns are unsuitable for Cramér's V)."""
    return bands["binary"] + bands["low"] + bands["medium"]


def group_by_parent(columns: list[str], sep: str = ".") -> list[dict]:
    """Group nested, ``sep``-separated column names by their top-level parent
    segment (the part before the first separator), preserving first-seen order.

    e.g. ['craft.shape', 'craft.color', 'state'] ->
        [{"parent": "craft", "columns": ["craft.shape", "craft.color"],
          "leaves": ["shape", "color"], "nested": True},
         {"parent": "state", "columns": ["state"], "leaves": ["state"],
          "nested": False}]

    Columns without the separator form their own single-member, non-nested group
    so the frontend can render them as standalone chips.
    """
    order: list[str] = []
    groups: dict[str, list[str]] = {}
    for c in columns:
        name = str(c)
        parent = name.split(sep, 1)[0] if sep in name else name
        if parent not in groups:
            groups[parent] = []
            order.append(parent)
        groups[parent].append(c)
    out = []
    for parent in order:
        members = groups[parent]
        nested = len(members) > 1 or (sep in str(members[0]))
        leaves = [str(m).split(sep, 1)[1] if sep in str(m) else str(m) for m in members]
        out.append({"parent": parent, "columns": members,
                    "leaves": leaves, "nested": nested})
    return out


# ── Known exact semantic duplicates (MasterSCU_v1) ──────────────────────────
# Field pairs that encode the SAME category twice at different paths — the
# "definitional overlap" that skews XGBoost gain (importance splits across the
# twins, and each twin trivially "predicts" the other at V≈1). Per the SCU
# priority, the engagement_type member is kept and the anomaly.* re-encoding is
# dropped from the DEFAULT selection (still selectable manually). Unit twins
# (metric/imperial re-encodings of one measurement) keep the metric member.
# Both sides are matched by path suffix so this works on MasterSCU-shaped
# (`engagement.engagement_type.*`) and flat SCU_v2-shaped columns alike.
SEMANTIC_DUPLICATE_GROUPS: list[dict] = [
    # definitional twins — keep engagement_type (SCU-filter priority)
    {"keep": ["engagement_type.radical_flight"], "drop": ["anomaly.flight"],
     "reason": "anomaly.flight re-encodes engagement_type.radical_flight"},
    {"keep": ["engagement_type.occupant_observed", "engagement_type.occupant_encounter"],
     "drop": ["anomaly.occupant"],
     "reason": "anomaly.occupant re-encodes the occupant engagement types"},
    {"keep": ["engagement_type.electronic_transmissions"], "drop": ["anomaly.signal"],
     "reason": "anomaly.signal re-encodes engagement_type.electronic_transmissions"},
    # keep assessment.contradictsUap — it is the SCU gate input
    {"keep": ["assessment.contradictsUap"], "drop": ["anomaly.validated"],
     "reason": "anomaly.validated is the inverse re-encoding of assessment.contradictsUap"},
    {"keep": ["witness.count"], "drop": ["witness.countFreeform"],
     "reason": "free-text re-encoding of witness.count"},
    # unit twins — deterministic conversions of one measurement (keep metric)
    {"keep": ["object.size_meters"], "drop": ["object.size_feet"], "reason": "unit twin (m/ft)"},
    {"keep": ["object.altitude_meters"], "drop": ["object.altitude_feet"], "reason": "unit twin (m/ft)"},
    {"keep": ["witness.distance_from_uap_meters"], "drop": ["witness.distance_from_uap_feet"],
     "reason": "unit twin (m/ft)"},
    {"keep": ["witness.distance_from_nhi_meters"], "drop": ["witness.distance_from_nhi_feet"],
     "reason": "unit twin (m/ft)"},
    {"keep": ["behavior.distance_covered_km"], "drop": ["behavior.distance_covered_mi"],
     "reason": "unit twin (km/mi)"},
    {"keep": ["location.associated_facility_distance_km"],
     "drop": ["location.associated_facility_distance_mi"], "reason": "unit twin (km/mi)"},
    {"keep": ["performance.speed_kmh"], "drop": ["performance.speed_mph"],
     "reason": "unit twin (km/h / mph)"},
    {"keep": ["entities.height_meters"], "drop": ["entities.Height"],
     "reason": "unit twin (metric / freeform height)"},
    {"keep": ["date_time.duration_min"], "drop": ["date_time.duration"],
     "reason": "freeform re-encoding of duration_min"},
]


def _suffix_find(columns: list[str], path: str) -> str | None:
    """First column equal to ``path`` or ending in ``.path`` (case-insensitive)."""
    pl = path.lower()
    for c in columns:
        cl = str(c).lower()
        if cl == pl or cl.endswith("." + pl):
            return c
    return None


def dedupe_semantic(columns: list[str]) -> tuple[list[str], list[dict]]:
    """Drop the shadowed member of each known duplicate group from ``columns``
    (only when a kept member is present too). Returns (kept_columns, removed)
    where removed = [{kept, dropped, reason}] for UI transparency."""
    out = list(columns)
    removed: list[dict] = []
    for g in SEMANTIC_DUPLICATE_GROUPS:
        kept = next((c for k in g["keep"] if (c := _suffix_find(out, k))), None)
        if kept is None:
            continue   # canonical member absent — nothing shadows the twin
        dropped = [c for d in g["drop"] if (c := _suffix_find(out, d))]
        if dropped:
            out = [c for c in out if c not in dropped]
            removed.append({"kept": kept, "dropped": dropped, "reason": g["reason"]})
    return out, removed


def column_groups(df: pd.DataFrame, *, high_threshold: int = 30) -> dict:
    """Eligible categorical columns for the explorer, grouped by dotted parent.

    Cheap (only cardinality counting) so the frontend can render the parent-group
    selector before computing the full Cramér's V matrix. The DEFAULT selection
    (``eligible``) excludes known exact semantic duplicates (unit twins and
    anomaly.* re-encodings of engagement types) so baseline XGBoost gain isn't
    diluted across definitional twins; the dropped columns stay in ``groups``
    and can be re-selected manually.
    """
    bands, nunique_map = band_columns(df, high_threshold=high_threshold)
    eligible_all = _eligible_categorical(bands)
    eligible, removed = dedupe_semantic(eligible_all)
    return {
        "eligible": eligible,
        "groups": group_by_parent(eligible_all),
        "bands": bands,
        "nunique": nunique_map,
        "semantic_duplicates_removed": removed,
    }


def _coalesce(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.mask(s.str.lower().isin(_NULL_STR_TOKENS), _MISSING_LABEL)


def compute_cramers_v_df(df: pd.DataFrame, cols: list[str],
                         drop_missing: bool = False) -> pd.DataFrame:
    from uap_analyzer import cramers_v

    cv = pd.DataFrame(index=cols, columns=cols, data=np.nan, dtype=float)
    cache = {c: _coalesce(df[c]) for c in cols}
    for i, c1 in enumerate(cols):
        cv.at[c1, c1] = 1.0
        for c2 in cols[i + 1:]:
            a, b = cache[c1], cache[c2]
            if drop_missing:
                keep = (a != _MISSING_LABEL) & (b != _MISSING_LABEL)
                a, b = a[keep], b[keep]
            v = 0.0 if len(a) == 0 else float(cramers_v(pd.crosstab(a, b)))
            cv.at[c1, c2] = v
            cv.at[c2, c1] = v
    return cv


# Cochran's rule of thumb: a chi-square test is unreliable when > 20% of cells
# have expected frequency < 5, or any cell has expected < 1.
_SPARSE_FRAC = 0.20


def _table_test(ct: pd.DataFrame) -> dict:
    """Association test + sparsity diagnostics for one contingency table.

    Returns ``{p, test, sparse, sparse_frac}``. Sparse 2×2 tables fall back to
    Fisher's exact test (valid at any cell count); larger sparse tables keep the
    chi-square p but are FLAGGED so the UI can warn that both V and p are
    unreliable there.
    """
    from scipy.stats import chi2_contingency, fisher_exact

    out = {"p": None, "test": "chi2", "sparse": False, "sparse_frac": 0.0}
    if ct.shape[0] < 2 or ct.shape[1] < 2 or ct.values.sum() == 0:
        return out
    try:
        chi2, p, dof, expected = chi2_contingency(ct)
        frac = float((expected < 5).mean())
        out["sparse_frac"] = round(frac, 3)
        out["sparse"] = bool(frac > _SPARSE_FRAC or (expected < 1).any())
        out["p"] = float(p)
    except ValueError:
        return out
    if out["sparse"] and ct.shape == (2, 2):
        try:
            _odds, p_f = fisher_exact(ct.values)
            out["p"] = float(p_f)
            out["test"] = "fisher"
        except ValueError:
            pass
    return out


def _matrix_with_stats(df: pd.DataFrame, cols: list[str],
                       drop_missing: bool = False) -> tuple[pd.DataFrame, dict]:
    """Cramér's V matrix + per-pair test stats in ONE pass over the crosstabs.

    Returns ``(cv_df, stats)`` where stats maps ``(c1, c2)`` (upper triangle) to
    ``{p, q, test, sparse, sparse_frac}``. p-values are Benjamini–Hochberg
    FDR-adjusted (``q``) across ALL k(k-1)/2 pair tests actually computed — the
    multiple-comparisons correction a screening matrix needs (claude/gemini
    review items 4.5 / 2.2).
    """
    from uap_analyzer import cramers_v

    cv = pd.DataFrame(index=cols, columns=cols, data=np.nan, dtype=float)
    cache = {c: _coalesce(df[c]) for c in cols}
    stats: dict[tuple, dict] = {}
    for i, c1 in enumerate(cols):
        cv.at[c1, c1] = 1.0
        for c2 in cols[i + 1:]:
            a, b = cache[c1], cache[c2]
            if drop_missing:
                keep = (a != _MISSING_LABEL) & (b != _MISSING_LABEL)
                a, b = a[keep], b[keep]
            if len(a) == 0:
                v, entry = 0.0, {"p": None, "test": "chi2", "sparse": False, "sparse_frac": 0.0}
            else:
                ct = pd.crosstab(a, b)
                v = float(cramers_v(ct))
                entry = _table_test(ct)
            cv.at[c1, c2] = v
            cv.at[c2, c1] = v
            stats[(c1, c2)] = entry

    # BH-FDR across every test with a p-value.
    keys = [k for k, s in stats.items() if s["p"] is not None]
    if keys:
        from statsmodels.stats.multitest import multipletests
        _rej, q, _a1, _a2 = multipletests([stats[k]["p"] for k in keys], method="fdr_bh")
        for k, qv in zip(keys, q):
            stats[k]["q"] = float(qv)
    return cv, stats


def cramers_v_ci(a: pd.Series, b: pd.Series, *, n_boot: int = 500, seed: int = 42,
                 alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap confidence interval for Cramér's V between two aligned
    categorical series. Returns ``[lo, hi]`` at the (1-alpha) level, or None when
    there are too few rows to resample meaningfully. A CI is what reviewers expect
    instead of a bare point estimate — it exposes the instability of V on sparse
    contingency cells (a wide interval = don't trust the point value)."""
    from uap_analyzer import cramers_v

    a_arr = a.to_numpy()
    b_arr = b.to_numpy()
    n = len(a_arr)
    if n < 20:
        return None
    rng = np.random.default_rng(seed)
    vs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)        # resample rows with replacement
        vs[i] = float(cramers_v(pd.crosstab(a_arr[idx], b_arr[idx])))
    lo, hi = np.quantile(vs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [round(float(lo), 3), round(float(hi), 3)]


def _is_trivial_v(v: float, tol: float = _CV_TOL) -> bool:
    return (v <= tol) or (v >= 1.0 - tol)


def pairs_table(cv_df: pd.DataFrame, exclude_trivial: bool = True) -> tuple[list[dict], int]:
    rows, n_excluded = [], 0
    cols = list(cv_df.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v = cv_df.at[c1, c2]
            if pd.isna(v):
                continue
            v = float(v)
            if exclude_trivial and _is_trivial_v(v):
                n_excluded += 1
                continue
            rows.append({"a": c1, "b": c2, "v": round(v, 3)})
    rows.sort(key=lambda r: r["v"], reverse=True)
    return rows, n_excluded


def high_correlation_columns(cv_df: pd.DataFrame, strong_threshold: float = 0.30,
                             exclude_trivial: bool = True) -> list[str]:
    if cv_df is None or getattr(cv_df, "empty", True):
        return []
    out = []
    for col in cv_df.columns:
        others = cv_df[col].drop(labels=[col], errors="ignore")
        for v in others:
            if pd.isna(v):
                continue
            v = float(v)
            if exclude_trivial and _is_trivial_v(v):
                continue
            if v >= strong_threshold:
                out.append(col)
                break
    return out


def cramers_v_report(df: pd.DataFrame, columns: list[str] | None = None, *,
                     drop_missing: bool = False, exclude_trivial: bool = True,
                     strong_threshold: float = 0.30, high_threshold: int = 30,
                     ci_top_n: int = 0) -> dict:
    """Full explorer payload: column bands, the Cramér's V matrix, the ranked
    pair table, and the high-correlation column shortlist."""
    bands, nunique_map = band_columns(df, high_threshold=high_threshold)

    if columns:
        cols = [c for c in columns if c in df.columns]
    else:
        # Default selection mirrors the explorer: binary + low + medium cardinality.
        cols = _eligible_categorical(bands)

    if len(cols) < 2:
        return {
            "labels": [], "matrix": [], "pairs": [], "n_excluded": 0,
            "high_correlation_columns": [],
            "bands": bands, "nunique": nunique_map, "selected_columns": cols,
            "groups": group_by_parent(cols),
        }

    cv, pair_stats = _matrix_with_stats(df, cols, drop_missing=drop_missing)
    pairs, n_excluded = pairs_table(cv, exclude_trivial=exclude_trivial)
    high = high_correlation_columns(cv, strong_threshold, exclude_trivial)

    # Attach the per-pair test stats (raw p, BH-FDR q across ALL computed pairs,
    # test used, sparsity flag) so the UI can show significance honestly.
    for p in pairs:
        s = pair_stats.get((p["a"], p["b"])) or pair_stats.get((p["b"], p["a"]))
        if s:
            if s.get("p") is not None:
                p["p"] = round(s["p"], 4)
            if s.get("q") is not None:
                p["q"] = round(s["q"], 4)
            p["test"] = s["test"]
            if s.get("sparse"):
                p["sparse"] = True

    # Optional bootstrap 95% CIs for the strongest pairs (bounded so the matrix
    # stays cheap — the full matrix would be n_boot × k² crosstabs).
    if ci_top_n and pairs:
        cache = {c: _coalesce(df[c]) for c in cols}
        for p in pairs[:ci_top_n]:
            a, b = cache[p["a"]], cache[p["b"]]
            if drop_missing:
                keep = (a != _MISSING_LABEL) & (b != _MISSING_LABEL)
                a, b = a[keep], b[keep]
            ci = cramers_v_ci(a, b)
            if ci:
                p["ci"] = ci

    matrix = [[None if pd.isna(v) else round(float(v), 3) for v in cv.loc[r]] for r in cols]
    n_tests = sum(1 for s in pair_stats.values() if s.get("p") is not None)
    n_sparse = sum(1 for s in pair_stats.values() if s.get("sparse"))
    return {
        "labels": cols,
        "matrix": matrix,
        "pairs": pairs,
        "n_tests": n_tests,               # pair tests entering the BH-FDR correction
        "n_sparse": n_sparse,             # pairs with Cochran-sparse tables (V/p unreliable)
        "fdr_method": "benjamini-hochberg",
        "n_excluded": n_excluded,
        "high_correlation_columns": high,
        "bands": bands,
        "nunique": nunique_map,
        "selected_columns": cols,
        "groups": group_by_parent(cols),
    }


def contingency(df: pd.DataFrame, c1: str, c2: str, drop_missing: bool = False,
                top_n: int = 15) -> dict:
    """Crosstab + Cramér's V for a single pair, for the heatmap drill-down."""
    from uap_analyzer import cramers_v

    if c1 not in df.columns or c2 not in df.columns:
        raise ValueError("Both columns must exist in the dataset.")
    a, b = _coalesce(df[c1]), _coalesce(df[c2])
    if drop_missing:
        keep = (a != _MISSING_LABEL) & (b != _MISSING_LABEL)
        a, b = a[keep], b[keep]
    if len(a) == 0:
        return {"row_labels": [], "col_labels": [], "matrix": [], "v": 0.0, "n": 0}

    ct = pd.crosstab(a, b)
    v = float(cramers_v(ct))
    ci = cramers_v_ci(a, b)   # 95% bootstrap CI on the full pair (pre-trim)
    test = _table_test(ct)    # p, chi2/fisher, Cochran sparsity flag (pre-trim)
    # Trim to the top_n most frequent categories on each axis for display.
    row_order = ct.sum(axis=1).sort_values(ascending=False).index[:top_n]
    col_order = ct.sum(axis=0).sort_values(ascending=False).index[:top_n]
    ct = ct.loc[row_order, col_order]
    return {
        "row_labels": [str(x) for x in ct.index.tolist()],
        "col_labels": [str(x) for x in ct.columns.tolist()],
        "matrix": ct.values.astype(int).tolist(),
        "v": round(v, 3),
        "ci": ci,                 # [lo, hi] 95% bootstrap CI, or None
        "p": round(test["p"], 4) if test["p"] is not None else None,
        "test": test["test"],     # "chi2" | "fisher"
        "sparse": test["sparse"], # Cochran rule: >20% expected<5 or any <1
        "sparse_frac": test["sparse_frac"],
        "n": int(len(a)),
    }


# Min rows for a stratum to count toward the conditional test (tiny strata give
# unstable per-stratum statistics and inflate the pooled dof).
_COND_MIN_STRATUM = 10
_COND_MAX_LEVELS = 20


def conditional_association(df: pd.DataFrame, c1: str, c2: str, condition_on: str, *,
                           drop_missing: bool = False, top_n: int = 15) -> dict:
    """Test whether the c1–c2 association survives conditioning on a third field Z.

    This is the formal answer to the "is this pair redundant / confounded by a
    third variable?" caveat that pairwise Cramér's V can't address. Reports the
    marginal V, the within-stratum V for each level of Z, and a pooled conditional
    -independence test: the Cochran–Mantel–Haenszel test when both variables are
    binary (2×k×K), else a stratified-χ² test (the sum of the per-stratum χ²,
    which is itself χ² under conditional independence). A verdict compares the
    marginal association to the n-weighted mean within-stratum association.
    """
    from scipy.stats import chi2, chi2_contingency
    from uap_analyzer import cramers_v

    for col in (c1, c2, condition_on):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if len({c1, c2, condition_on}) < 3:
        raise ValueError("Pick three distinct columns (pair + a different conditioner).")

    a = _coalesce(df[c1]); b = _coalesce(df[c2]); z = _coalesce(df[condition_on])
    frame = pd.DataFrame({"a": a, "b": b, "z": z})
    if drop_missing:
        frame = frame[(frame["a"] != _MISSING_LABEL) & (frame["b"] != _MISSING_LABEL)
                      & (frame["z"] != _MISSING_LABEL)]
    if len(frame) == 0:
        raise ValueError("No rows left after coalescing/at least one column is empty.")

    n_levels = frame["z"].nunique()
    if n_levels > _COND_MAX_LEVELS:
        raise ValueError(
            f"Conditioner '{condition_on}' has {n_levels} levels (> {_COND_MAX_LEVELS}); "
            "pick a lower-cardinality field to condition on."
        )

    marginal_v = float(cramers_v(pd.crosstab(frame["a"], frame["b"])))
    a_bin = frame["a"].nunique() == 2
    b_bin = frame["b"].nunique() == 2

    strata, used_n, used_v_weighted = [], 0, 0.0
    pooled_chi2, pooled_dof, n_dropped = 0.0, 0, 0
    tables_2x2 = []
    for level, g in frame.groupby("z", observed=True):
        n_g = len(g)
        ct = pd.crosstab(g["a"], g["b"])
        if n_g < _COND_MIN_STRATUM or ct.shape[0] < 2 or ct.shape[1] < 2:
            n_dropped += 1
            continue
        v_g = float(cramers_v(ct))
        strata.append({"level": str(level), "n": int(n_g), "v": round(v_g, 3)})
        used_n += n_g
        used_v_weighted += v_g * n_g
        try:
            chi2_g, _p, dof_g, _exp = chi2_contingency(ct)
            pooled_chi2 += float(chi2_g)
            pooled_dof += int(dof_g)
        except ValueError:
            pass
        if a_bin and b_bin and ct.shape == (2, 2):
            tables_2x2.append(ct.to_numpy())

    strata.sort(key=lambda s: s["n"], reverse=True)
    mean_cond_v = round(used_v_weighted / used_n, 3) if used_n else None

    # Pooled conditional-independence test.
    test: dict = {"method": None, "statistic": None, "dof": None, "p_value": None}
    if a_bin and b_bin and len(tables_2x2) >= 1:
        try:
            from statsmodels.stats.contingency_tables import StratifiedTable
            st = StratifiedTable([t.astype(float) for t in tables_2x2])
            res = st.test_null_odds()
            test = {
                "method": "Cochran–Mantel–Haenszel",
                "statistic": round(float(res.statistic), 3),
                "dof": 1,
                "p_value": float(res.pvalue),
                "pooled_odds_ratio": round(float(st.oddsratio_pooled), 3),
            }
        except Exception:
            test["method"] = None
    if test["method"] is None and pooled_dof > 0:
        test = {
            "method": "Stratified χ²",
            "statistic": round(pooled_chi2, 3),
            "dof": int(pooled_dof),
            "p_value": float(chi2.sf(pooled_chi2, pooled_dof)),
        }

    # Verdict: marginal vs n-weighted within-stratum association.
    verdict = "inconclusive"
    if mean_cond_v is not None and test["p_value"] is not None:
        drop = marginal_v - mean_cond_v
        sig = test["p_value"] < 0.05
        if not sig and mean_cond_v < 0.5 * max(marginal_v, 1e-9):
            verdict = "explained_by_z"       # association collapses within strata
        elif sig and drop < 0.05:
            verdict = "persists"             # holds up conditioning on Z
        elif sig:
            verdict = "attenuated"           # survives but weaker
        else:
            verdict = "weak_or_absent"

    return {
        "c1": c1, "c2": c2, "condition_on": condition_on,
        "marginal_v": round(marginal_v, 3),
        "mean_conditional_v": mean_cond_v,
        "strata": strata[:top_n],
        "n_strata_used": len(strata),
        "n_strata_dropped": n_dropped,
        "n": int(len(frame)),
        "test": test,
        "verdict": verdict,
    }


# ── XGBoost feature importance on raw categorical columns ───────────────────
# Cap on a target column's class count — XGBoost multi:softmax with hundreds of
# classes is slow and the importances are meaningless. The explorer only feeds
# binary/low/medium-cardinality columns, so this is just a safety net.
_XGB_MAX_TARGET_CLASSES = 50

_GPU_CACHED: bool | None = None


def _gpu_available() -> bool:
    """Cached CUDA check via torch only — avoids importing the heavy
    ``uap_analyzer`` module (torch + transformers + sentence-transformers) just
    to read a flag, which would add tens of seconds to the first CV call."""
    global _GPU_CACHED
    if _GPU_CACHED is None:
        try:
            import torch
            _GPU_CACHED = bool(torch.cuda.is_available())
        except Exception:
            _GPU_CACHED = False
    return _GPU_CACHED


def _xgb_cv_accuracy(x: pd.DataFrame, y: pd.Series, n_classes: int, *, n_splits: int = 5) -> dict | None:
    """Stratified k-fold cross-validation accuracy via native ``xgboost.cv``.

    Returns ``{cv_mean, cv_std, cv_folds}`` or ``None`` when CV isn't feasible
    (the rarest class has < 2 samples). A single ``xgb.cv`` call runs all folds
    through the native path on the GPU when available, with early stopping — far
    cheaper than sklearn ``cross_val_score`` (which clones + refits a wrapper
    estimator per fold, on CPU). Comparing this CV mean to the single-split
    ``accuracy`` is the practical overfit check.
    """
    from collections import Counter
    import xgboost as xgb

    folds = min(n_splits, min(Counter(y).values()))
    if folds < 2:
        return None
    dtrain = xgb.DMatrix(x, label=y, enable_categorical=True)
    multi = n_classes > 2
    metric = "merror" if multi else "error"
    params = {
        "objective": "multi:softmax" if multi else "binary:logistic",
        "max_depth": 4, "eta": 0.1, "tree_method": "hist",
        "device": "cuda" if _gpu_available() else "cpu",
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2,
        "eval_metric": metric, "nthread": -1,
    }
    if multi:
        params["num_class"] = n_classes
    cvres = xgb.cv(
        params, dtrain, num_boost_round=400, nfold=folds,
        stratified=True, early_stopping_rounds=20, seed=42, as_pandas=True,
    )
    mean_col, std_col = f"test-{metric}-mean", f"test-{metric}-std"
    best = int(cvres[mean_col].idxmin())          # fewest errors = best round
    return {
        "cv_mean": round(1.0 - float(cvres[mean_col].iloc[best]), 3),
        "cv_std": round(float(cvres[std_col].iloc[best]), 3),
        "cv_folds": int(folds),
    }


def _xgb_quick_fit(x: pd.DataFrame, y, n_classes: int, *,
                   num_boost_round: int = 40, seed: int = 42, dmat=None) -> dict:
    """A fast importance-only fit (no eval/early-stopping) for the permutation and
    bootstrap resamples — fewer rounds keep the significance pass affordable.

    If ``dmat`` is passed (a label-carrying ``DMatrix`` built once from ``x``),
    reuse it and only re-stamp the label. The permutation null holds ``x`` fixed
    and just shuffles ``y``, so this skips re-parsing the DataFrame and
    re-quantizing the feature bins on every refit — XGBoost caches the quantized
    matrix on the ``DMatrix`` object, and the label is not part of that cache."""
    import xgboost as xgb

    multi = n_classes > 2
    params = {
        "objective": "multi:softmax" if multi else "binary:logistic",
        "max_depth": 6, "eta": 0.3, "tree_method": "hist",
        "device": "cuda" if _gpu_available() else "cpu", "nthread": -1, "seed": seed,
    }
    if multi:
        params["num_class"] = n_classes
    if dmat is None:
        dmat = xgb.DMatrix(x, label=y, enable_categorical=True)
    else:
        dmat.set_info(label=y)
    bst = xgb.train(params, dmat, num_boost_round=num_boost_round, verbose_eval=False)
    return {k: float(v) for k, v in bst.get_score(importance_type="gain").items()}


def _null_importance(x: pd.DataFrame, y: pd.Series, n_classes: int, real_imp: dict, *,
                     n_perm: int = 20, seed: int = 42) -> dict:
    """Permutation null for gain importance: shuffle the target, refit, and count
    how often each feature's *null* gain reaches its real gain. Empirical p-value
    ``(1 + #{null ≥ real}) / (n_perm + 1)`` — small p ⇒ the feature's importance is
    unlikely under no real relationship. Only features actually used (real gain > 0)
    get a p-value; the rest are non-significant by construction."""
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    feats = [f for f in x.columns if real_imp.get(f, 0.0) > 0]
    if not feats:
        return {}
    ge = {f: 0 for f in feats}
    y_arr = y.to_numpy()
    # x is identical across every permutation — build (and quantize) it once, then
    # only swap the shuffled label per refit.
    dmat = xgb.DMatrix(x, label=y_arr, enable_categorical=True)
    for p in range(n_perm):
        yp = rng.permutation(y_arr)
        imp = _xgb_quick_fit(x, yp, n_classes, seed=seed + p + 1, dmat=dmat)
        for f in feats:
            if imp.get(f, 0.0) >= real_imp[f]:
                ge[f] += 1
    return {f: round((1 + ge[f]) / (n_perm + 1), 3) for f in feats}


def _stability_selection(x: pd.DataFrame, y: pd.Series, n_classes: int, *,
                         n_boot: int = 20, seed: int = 42) -> dict:
    """Selection frequency under row bootstrapping: refit on B resamples and record
    how often each feature is used in any split. A feature that survives most
    resamples is a robust predictor; one that flickers in and out is fragile —
    far more honest than a single run's importance."""
    rng = np.random.default_rng(seed)
    feats = list(x.columns)
    sel = {f: 0 for f in feats}
    n = len(x)
    done = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x.iloc[idx], y.iloc[idx]
        if yb.nunique() < 2:
            continue
        imp = _xgb_quick_fit(xb, yb, n_classes, seed=seed + 1000 + b)
        for f in feats:
            if imp.get(f, 0.0) > 0:
                sel[f] += 1
        done += 1
    if done == 0:
        return {}
    return {f: round(sel[f] / done, 3) for f in feats}


def xgboost_importance(df: pd.DataFrame, columns: list[str], *,
                       test_size: float = 0.2, random_state: int = 42,
                       with_cv: bool = True, with_significance: bool = False,
                       n_perm: int = 20, n_boot: int = 20) -> dict:
    """Per-column XGBoost feature importance computed *directly* on the selected
    raw categorical columns — predict each column from the others and report the
    gain-based importance of every other column, plus the test accuracy.

    This mirrors ``analyzing.py``'s ``analyze_and_predict`` loop but runs on the
    raw values (the same set used by the Cramér's V explorer) instead of cluster
    labels, so feature importance is available without the embedding/cluster
    pipeline. Returns ``{results: {col: {feature_importance, accuracy}}, ...}``.
    """
    from sklearn.model_selection import train_test_split
    from uap_analyzer import train_xgboost

    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return {
            "results": {}, "columns": cols, "skipped": {},
            "message": "Select at least two categorical columns for feature importance.",
        }

    # Coalesce missingness the same way Cramér's V does, then category-encode.
    new_data = pd.DataFrame({c: _coalesce(df[c]) for c in cols}).astype("category")
    data_nums = new_data.apply(lambda s: s.cat.codes)

    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for col in cols:
        n_classes = len(new_data[col].cat.categories)
        if n_classes < 2:
            skipped[col] = "constant column (one class)"
            continue
        if n_classes > _XGB_MAX_TARGET_CLASSES:
            skipped[col] = f"too many classes ({n_classes}) to predict"
            continue
        try:
            x = data_nums.drop(columns=[col])
            y = data_nums[col]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state,
            )
            bst, accuracy, _ = train_xgboost(x_train, y_train, x_test, y_test, n_classes)
            # Gain-based importance; only features used in a split appear.
            imp = {k: float(v) for k, v in bst.get_score(importance_type="gain").items()}
            imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
            entry = {"feature_importance": imp, "accuracy": round(float(accuracy), 3)}
            if with_cv:
                try:                       # CV is best-effort — never drop the column over it
                    cv = _xgb_cv_accuracy(x, y, n_classes)
                except Exception:
                    cv = None
                if cv:
                    entry.update(cv)
            if with_significance:          # opt-in, slower: permutation null + bootstrap stability
                try:
                    entry["null_p"] = _null_importance(x, y, n_classes, imp,
                                                       n_perm=n_perm, seed=random_state)
                except Exception:
                    pass
                try:
                    entry["selection_freq"] = _stability_selection(x, y, n_classes,
                                                                   n_boot=n_boot, seed=random_state)
                except Exception:
                    pass
            results[col] = entry
        except Exception as e:  # noqa: BLE001 — one bad target shouldn't sink the rest
            skipped[col] = str(e)

    return {"results": results, "columns": cols, "skipped": skipped}


# ── 2nd pass: PCA latent-index XGBoost ──────────────────────────────────────
# Optional second pass that attacks the multicollinearity / "definitional
# overlap" the first pass surfaces (high Cramér's V between fields makes XGBoost
# split their gain and adds correlated noise dimensions). Cramér's V identifies
# redundancy clusters; each is collapsed into ONE PCA latent index ("Radar
# index"); XGBoost is re-fit on [latent indices + the non-redundant passthrough
# columns]. So the 2nd pass is *informed by* both first-pass artifacts: Cramér's
# V (which columns are redundant) and gain importance (optional dead-feature
# pruning + the de-dilution comparison shown in the UI).


def _redundancy_clusters(cv_df: pd.DataFrame, cols: list[str],
                         strong_threshold: float) -> list[list[str]]:
    """Connected components of the graph with an edge between two columns when
    their Cramér's V ≥ ``strong_threshold``. Returns only components with ≥ 2
    members (singletons are non-redundant "free" columns), each ordered as in
    ``cols`` so latent-index membership reads in a stable order."""
    parent = {c: c for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v = cv_df.at[c1, c2]
            if pd.notna(v) and float(v) >= strong_threshold:
                ra, rb = find(c1), find(c2)
                if ra != rb:
                    parent[ra] = rb

    comps: dict[str, list[str]] = {}
    for c in cols:
        comps.setdefault(find(c), []).append(c)
    order = {c: i for i, c in enumerate(cols)}
    return [sorted(m, key=order.get) for m in comps.values() if len(m) >= 2]


def _cluster_root(cv_df: pd.DataFrame, members: list[str]) -> str:
    """The most central member — highest summed Cramér's V to its cluster-mates.
    This is the "root predictor" (e.g. ``coded_radar`` for the radar cluster); the
    latent index is named after it."""
    best, best_s = members[0], -1.0
    for m in members:
        s = sum(
            float(cv_df.at[m, o])
            for o in members if o != m and pd.notna(cv_df.at[m, o])
        )
        if s > best_s:
            best_s, best = s, m
    return best


def _fit_latent_index(cat_block: pd.DataFrame) -> tuple[pd.Series, float, list[dict]]:
    """One-hot → standardise → PCA(1) over a cluster's columns. Returns the single
    component (aligned to the input index), its explained-variance ratio, and the
    per-source-column loading (share of component variance), sorted descending."""
    from sklearn.decomposition import PCA

    parts, owners = [], []
    for m in cat_block.columns:
        dummies = pd.get_dummies(cat_block[m].astype(str))
        if dummies.shape[1] == 0:
            continue
        parts.append(dummies.to_numpy(dtype=float))
        owners.extend([m] * dummies.shape[1])
    if not parts:
        raise ValueError("cluster produced no one-hot columns")

    x = np.hstack(parts)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    xs = (x - mu) / sd
    if np.allclose(xs, 0.0):
        raise ValueError("cluster columns are constant after coalescing")

    pca = PCA(n_components=1, random_state=42)
    comp = pca.fit_transform(xs)[:, 0]
    evr = float(pca.explained_variance_ratio_[0])

    loading_vec = pca.components_[0]
    owners_arr = np.asarray(owners)
    contrib = {m: float(np.sum(loading_vec[owners_arr == m] ** 2)) for m in cat_block.columns}
    total = sum(contrib.values()) or 1.0
    loadings = sorted(
        ({"feature": m, "weight": round(contrib[m] / total, 3)} for m in cat_block.columns),
        key=lambda d: d["weight"], reverse=True,
    )
    return pd.Series(comp, index=cat_block.index), evr, loadings


def _features_for_target(col: str, cols: list[str], clusters_meta: list[dict],
                         comp_cols: dict, member_to_index: dict, data_nums: pd.DataFrame,
                         dead: set, index) -> pd.DataFrame:
    """Build the design matrix for predicting ``col`` from the others: each
    redundancy cluster contributes its PCA latent index, except a cluster that
    contains the target (its cluster-mates fall back to raw columns so the index
    never leaks the target into its own prediction); non-redundant columns pass
    through as raw category codes."""
    feat = pd.DataFrame(index=index)
    for meta in clusters_meta:
        name = meta["index_name"]
        if col in meta["members"]:
            for m in meta["members"]:
                if m != col:
                    feat[m] = data_nums[m]
        else:
            feat[name] = comp_cols[name]
    for c in cols:
        if c != col and c not in member_to_index and c not in dead:
            feat[c] = data_nums[c]
    return feat


def xgboost_pca_importance(df: pd.DataFrame, columns: list[str], *,
                           strong_threshold: float = 0.30, with_cv: bool = True,
                           prune_uninformative: bool = False,
                           with_significance: bool = False, n_perm: int = 20, n_boot: int = 20,
                           test_size: float = 0.2, random_state: int = 42) -> dict:
    """Two-pass XGBoost feature importance.

    Pass 1 is the standard per-column gain importance (``xgboost_importance``).
    Pass 2 collapses each Cramér's V redundancy cluster into a single PCA latent
    index and re-fits XGBoost on [latent indices + non-redundant columns], so a
    cluster's de-diluted combined signal shows up as one feature. Per target the
    PCA components are excluded for any cluster the target belongs to (its
    cluster-mates fall back to raw columns) so a latent index never leaks the
    target into its own prediction.

    Returns ``{first_pass, second_pass, clusters, columns, skipped,
    strong_threshold, n_clusters, pruned, message}``.
    """
    from sklearn.model_selection import train_test_split
    from uap_analyzer import train_xgboost

    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return {
            "first_pass": {}, "second_pass": {}, "clusters": [], "columns": cols,
            "skipped": {}, "strong_threshold": strong_threshold, "n_clusters": 0,
            "pruned": [], "message": "Select at least two categorical columns.",
        }

    first = xgboost_importance(
        df, cols, test_size=test_size, random_state=random_state, with_cv=with_cv,
        with_significance=with_significance, n_perm=n_perm, n_boot=n_boot,
    )
    first_results = first["results"]
    skipped = dict(first["skipped"])

    cv_df = compute_cramers_v_df(df, cols, drop_missing=False)
    cluster_lists = _redundancy_clusters(cv_df, cols, strong_threshold)

    new_data = pd.DataFrame({c: _coalesce(df[c]) for c in cols}).astype("category")
    data_nums = new_data.apply(lambda s: s.cat.codes)

    # Fit each cluster's latent index once (independent of the target).
    clusters_meta: list[dict] = []
    comp_cols: dict[str, pd.Series] = {}
    member_to_index: dict[str, str] = {}
    for members in cluster_lists:
        root = _cluster_root(cv_df, members)
        # NB: XGBoost rejects '[', ']' and '<' in feature names (they break the
        # DMatrix), so the latent-index label uses a middot separator instead.
        index_name = f"PCA·{root}"
        try:
            comp, evr, loadings = _fit_latent_index(new_data[members])
        except Exception:  # noqa: BLE001 — a degenerate cluster just stays raw
            continue
        comp_cols[index_name] = comp
        member_to_index.update({m: index_name for m in members})
        # member importance summed from pass 1 (per target) is shown in the UI as
        # the de-dilution comparison; here we record the static cluster definition.
        clusters_meta.append({
            "index_name": index_name, "root": root, "members": members,
            "explained_variance": round(evr, 3), "loadings": loadings,
        })

    # Optional: drop free (non-clustered) columns that pass 1 never split on.
    dead: set[str] = set()
    if prune_uninformative:
        used = set()
        for r in first_results.values():
            used.update(r.get("feature_importance", {}).keys())
        dead = {c for c in cols if c not in member_to_index and c not in used}

    second: dict[str, dict] = {}
    for col in cols:
        if col not in first_results:
            continue  # constant / too-many-classes — already in skipped
        n_classes = len(new_data[col].cat.categories)
        try:
            feat = _features_for_target(
                col, cols, clusters_meta, comp_cols, member_to_index,
                data_nums, dead, new_data.index,
            )
            if feat.shape[1] < 1:
                skipped[col] = "no features left after PCA collapse"
                continue

            y = data_nums[col]
            x_train, x_test, y_train, y_test = train_test_split(
                feat, y, test_size=test_size, random_state=random_state,
            )
            bst, accuracy, _ = train_xgboost(x_train, y_train, x_test, y_test, n_classes)
            imp = {k: float(v) for k, v in bst.get_score(importance_type="gain").items()}
            imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
            entry = {
                "feature_importance": imp,
                "accuracy": round(float(accuracy), 3),
                "n_features": int(feat.shape[1]),
            }
            if with_cv:
                try:
                    cv = _xgb_cv_accuracy(feat, y, n_classes)
                except Exception:
                    cv = None
                if cv:
                    entry.update(cv)
            second[col] = entry
        except Exception as e:  # noqa: BLE001
            skipped[col] = str(e)

    if not clusters_meta:
        msg = (f"No redundancy clusters at Cramér's V ≥ {strong_threshold:.2f}. "
               "Lower the strong threshold to collapse correlated columns, or the "
               "selected columns are already non-redundant.")
    else:
        msg = None

    return {
        "first_pass": first_results,
        "second_pass": second,
        "clusters": clusters_meta,
        "columns": cols,
        "skipped": skipped,
        "strong_threshold": strong_threshold,
        "n_clusters": len(clusters_meta),
        "pruned": sorted(dead),
        "message": msg,
    }


# ── Model-based imputation of missing values ────────────────────────────────
# Reuses the per-target predict-from-the-others setup, but trains each model on
# the rows where the target is *observed* and predicts the rows where it is
# missing. The model's accuracy (CV when available) is returned per column so the
# UI can colour each predicted fill by how reliable its model is.
# Cap high enough to cover a full grid load (≤50k rows) so the Data Explorer
# overlay can fill every missing cell, not just a sample.
_IMPUTE_MAX_SAMPLE = 50000  # row-level predicted fills returned for display / CSV / overlay


def xgboost_impute(df: pd.DataFrame, columns: list[str], *,
                   with_cv: bool = True, use_pca: bool = False,
                   strong_threshold: float = 0.30, n_imputations: int = 1,
                   test_size: float = 0.2, random_state: int = 42,
                   max_sample: int = _IMPUTE_MAX_SAMPLE) -> dict:
    """Predict each selected column's missing values from the other columns.

    For every column with missing cells, an XGBoost model is trained on the rows
    where that column is observed and used to predict the missing rows. Returns
    per column: the model accuracy (+ CV) used to colour the fills, the predicted
    value distribution, and a row-level list of the predictions (with per-cell
    confidence) that the Data Explorer overlays onto the missing cells.

    ``n_imputations`` > 1 turns on **multiple imputation**: instead of taking the
    single most-likely class, it draws M values per cell from the model's predicted
    class probabilities, reports the modal value, and records a per-cell *agreement*
    (how often the modal value was drawn) — an honest cell-level uncertainty on top
    of the column-level model accuracy. Single imputation understates uncertainty;
    for inference, pool an estimand across the M completed datasets with Rubin's
    rules. Note also the **missingness mechanism**: this assumes the value is
    predictable from the observed fields (≈ MAR); if a field is blank *because of*
    its own value (MNAR — common in UAP report curation), the fills are biased.

    When ``use_pca`` is set the predictors are the same PCA latent indices the 2nd
    pass builds (redundancy clusters collapsed), so imputation benefits from the
    decorrelated feature space too.
    """
    from collections import Counter
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return {"results": {}, "columns": cols, "skipped": {}, "total_missing": 0,
                "message": "Select at least two categorical columns to impute."}

    new_data = pd.DataFrame({c: _coalesce(df[c]) for c in cols}).astype("category")
    data_nums = new_data.apply(lambda s: s.cat.codes)

    # Optional PCA predictor space (same construction as the 2nd pass).
    clusters_meta: list[dict] = []
    comp_cols: dict[str, pd.Series] = {}
    member_to_index: dict[str, str] = {}
    if use_pca:
        cv_df = compute_cramers_v_df(df, cols, drop_missing=False)
        for members in _redundancy_clusters(cv_df, cols, strong_threshold):
            root = _cluster_root(cv_df, members)
            index_name = f"PCA·{root}"
            try:
                comp, _evr, _load = _fit_latent_index(new_data[members])
            except Exception:  # noqa: BLE001
                continue
            comp_cols[index_name] = comp
            member_to_index.update({m: index_name for m in members})
            clusters_meta.append({"index_name": index_name, "members": members})

    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    total_missing = 0
    for col in cols:
        cats = list(new_data[col].cat.categories)
        if _MISSING_LABEL not in cats:
            continue  # nothing missing in this column
        miss_code = cats.index(_MISSING_LABEL)
        y_full = data_nums[col]
        miss_mask = (y_full == miss_code).to_numpy()
        n_missing = int(miss_mask.sum())
        if n_missing == 0:
            continue
        total_missing += n_missing
        try:
            feat = _features_for_target(
                col, cols, clusters_meta, comp_cols, member_to_index,
                data_nums, set(), new_data.index,
            )
            if feat.shape[1] < 1:
                skipped[col] = "no predictor columns available"
                continue

            obs_mask = ~miss_mask
            x_obs = feat[obs_mask]
            y_obs_raw = y_full[obs_mask]
            # Re-map observed labels to contiguous 0..k-1 (the missing class is
            # excluded from the label space — we never predict "(missing)").
            uniq = sorted(pd.unique(y_obs_raw))
            remap = {c: i for i, c in enumerate(uniq)}
            inv = {i: c for c, i in remap.items()}
            k = len(uniq)
            x_miss = feat[miss_mask]

            if k < 2:
                # Only one observed value — impute it directly (no model needed).
                label = str(cats[uniq[0]])
                preds = [label] * n_missing
                confs: list[float | None] = [None] * n_missing
                accuracy = None
                entry_cv: dict = {}
            else:
                import xgboost as xgb
                y_obs = y_obs_raw.map(remap)
                x_train, x_test, y_train, y_test = train_test_split(
                    x_obs, y_obs, test_size=test_size, random_state=random_state,
                )
                # Train for class probabilities (softprob) so we can both take the
                # argmax (single imputation) and sample (multiple imputation).
                params = {
                    "objective": "multi:softprob", "num_class": k,
                    "max_depth": 6, "eta": 0.3, "tree_method": "hist",
                    "device": "cuda" if _gpu_available() else "cpu", "nthread": -1,
                }
                bst = xgb.train(
                    params, xgb.DMatrix(x_train, label=y_train, enable_categorical=True),
                    num_boost_round=100,
                    evals=[(xgb.DMatrix(x_test, label=y_test, enable_categorical=True), "eval")],
                    early_stopping_rounds=10, verbose_eval=False,
                )
                accuracy = round(float(accuracy_score(
                    y_test, bst.predict(xgb.DMatrix(x_test, enable_categorical=True)).argmax(axis=1))), 3)
                proba = np.atleast_2d(bst.predict(xgb.DMatrix(x_miss, enable_categorical=True)))
                if n_imputations and n_imputations > 1:
                    # Multiple imputation: draw M class samples per cell; modal value
                    # + agreement fraction = a per-cell uncertainty.
                    rng = np.random.default_rng(random_state)
                    preds, confs = [], []
                    for prow in proba:
                        p = prow / prow.sum()
                        draws = rng.choice(k, size=n_imputations, p=p)
                        vals, counts = np.unique(draws, return_counts=True)
                        modal = int(vals[counts.argmax()])
                        preds.append(str(cats[inv[modal]]))
                        confs.append(round(float(counts.max() / n_imputations), 3))
                else:
                    arg = proba.argmax(axis=1)
                    preds = [str(cats[inv[int(p)]]) for p in arg]
                    confs = [None] * len(arg)
                entry_cv = {}
                if with_cv:
                    try:
                        cvres = _xgb_cv_accuracy(x_obs, y_obs, k)
                    except Exception:
                        cvres = None
                    if cvres:
                        entry_cv = cvres

            dist = Counter(preds)
            row_ids = [str(r) for r in df.index[miss_mask]]
            sample = [{"row": rid, "value": val, "conf": conf}
                      for rid, val, conf in zip(row_ids[:max_sample], preds[:max_sample], confs[:max_sample])]
            valid_confs = [c for c in confs if c is not None]
            entry = {
                "accuracy": accuracy,
                "n_missing": n_missing,
                "n_features": int(feat.shape[1]),
                "n_imputations": int(n_imputations),
                "mean_conf": round(float(np.mean(valid_confs)), 3) if valid_confs else None,
                "predictions": [{"value": v, "count": int(c)}
                                for v, c in dist.most_common()],
                "sample": sample,
                "sample_truncated": n_missing > max_sample,
            }
            entry.update(entry_cv)
            results[col] = entry
        except Exception as e:  # noqa: BLE001 — one bad target shouldn't sink the rest
            skipped[col] = str(e)

    if total_missing == 0:
        msg = "No missing values in the selected columns — nothing to impute."
    elif not results:
        msg = "No column could be imputed (need ≥ 2 observed classes and predictors)."
    else:
        msg = None

    return {
        "results": results,
        "columns": cols,
        "skipped": skipped,
        "total_missing": total_missing,
        "used_pca": bool(use_pca and clusters_meta),
        "n_imputations": int(n_imputations),
        "message": msg,
    }
