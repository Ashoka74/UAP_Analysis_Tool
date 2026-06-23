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


def column_groups(df: pd.DataFrame, *, high_threshold: int = 30) -> dict:
    """Eligible categorical columns for the explorer, grouped by dotted parent.

    Cheap (only cardinality counting) so the frontend can render the parent-group
    selector before computing the full Cramér's V matrix.
    """
    bands, nunique_map = band_columns(df, high_threshold=high_threshold)
    eligible = _eligible_categorical(bands)
    return {
        "eligible": eligible,
        "groups": group_by_parent(eligible),
        "bands": bands,
        "nunique": nunique_map,
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
                     strong_threshold: float = 0.30, high_threshold: int = 30) -> dict:
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

    cv = compute_cramers_v_df(df, cols, drop_missing=drop_missing)
    pairs, n_excluded = pairs_table(cv, exclude_trivial=exclude_trivial)
    high = high_correlation_columns(cv, strong_threshold, exclude_trivial)

    matrix = [[None if pd.isna(v) else round(float(v), 3) for v in cv.loc[r]] for r in cols]
    return {
        "labels": cols,
        "matrix": matrix,
        "pairs": pairs,
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
    # Trim to the top_n most frequent categories on each axis for display.
    row_order = ct.sum(axis=1).sort_values(ascending=False).index[:top_n]
    col_order = ct.sum(axis=0).sort_values(ascending=False).index[:top_n]
    ct = ct.loc[row_order, col_order]
    return {
        "row_labels": [str(x) for x in ct.index.tolist()],
        "col_labels": [str(x) for x in ct.columns.tolist()],
        "matrix": ct.values.astype(int).tolist(),
        "v": round(v, 3),
        "n": int(len(a)),
    }


# ── XGBoost feature importance on raw categorical columns ───────────────────
# Cap on a target column's class count — XGBoost multi:softmax with hundreds of
# classes is slow and the importances are meaningless. The explorer only feeds
# binary/low/medium-cardinality columns, so this is just a safety net.
_XGB_MAX_TARGET_CLASSES = 50


def xgboost_importance(df: pd.DataFrame, columns: list[str], *,
                       test_size: float = 0.2, random_state: int = 42) -> dict:
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
            results[col] = {"feature_importance": imp, "accuracy": round(float(accuracy), 3)}
        except Exception as e:  # noqa: BLE001 — one bad target shouldn't sink the rest
            skipped[col] = str(e)

    return {"results": results, "columns": cols, "skipped": skipped}
