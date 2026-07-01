"""SCU normalization service — wraps ``scu_normalizer`` to canonicalise parsed
UAP responses (country/US-state codes, witness roles, craft shape/size bands)
and derive the SCU five-criterion eligibility gate, plus an eligibility filter
with the same presets as the Streamlit page.
"""
from __future__ import annotations

from typing import Any


# Preset → ordered list of criterion keys, mirroring render_scu_filter() in
# parsing.py. Built lazily so importing this module doesn't import the
# normalizer (and its data tables) until a request actually needs it.
def _presets(criteria_keys: list[str]) -> dict[str, list[str] | None]:
    post_1975_keys = ["post_1975_window"] + [k for k in criteria_keys if k != "in_scu_window"]
    return {
        "Full five-criterion gate (1945-1975, recommended)": criteria_keys,
        "Post-1975 five-criterion gate (1975 onwards)": post_1975_keys,
        "Phase-3 analog — no window / anomaly / credibility gates": [
            "has_core_fields", "has_investigation_channel",
            "has_engagement_signal", "day_night_resolved", "military_public_known",
        ],
        "SCU window only (1945-1975)": ["in_scu_window"],
        "Post-1975 window only (1975 onwards)": ["post_1975_window"],
        "Data quality only — Criterion 2": ["has_core_fields"],
    }


def criteria_info() -> dict:
    """Criterion keys/labels and named presets for the filter UI."""
    import scu_normalizer

    criteria = [
        {"key": k, "column": c, "label": lbl} for k, c, lbl in scu_normalizer.SCU_CRITERIA
    ]
    extra = [
        {"key": k, "column": c, "label": lbl} for k, c, lbl in scu_normalizer.SCU_EXTRA_CRITERIA
    ]
    keys = [k for k, _c, _l in scu_normalizer.SCU_CRITERIA]
    return {"criteria": criteria, "extra_criteria": extra, "presets": _presets(keys)}


def _raw_df_from_parsed(parsed_responses: dict):
    import pandas as pd

    return pd.json_normalize(list(parsed_responses.values()))


def normalize_df(raw_df) -> dict:
    """Run scu_normalizer.normalize() on an already-structured DataFrame.

    Used by both the parsed-responses path and the uploaded-dataset path (a
    previously-parsed CSV/XLSX or a parsed_responses JSON), so SCU normalization
    doesn't require re-running the LLM extractor. Returns the normalized
    DataFrame plus the audit dict, its markdown render, and headline metrics.
    """
    import scu_normalizer

    if raw_df is None or raw_df.empty:
        raise ValueError("Empty dataset — nothing to normalize.")

    norm_df, audit = scu_normalizer.normalize(raw_df)
    audit_md = scu_normalizer.audit_to_markdown(audit)

    metrics = {
        "rows": int(audit.get("output_rows", len(norm_df))),
        "scu_eligible": int(audit.get("scu_eligible_count", 0)),
        "in_scu_window": int(audit.get("in_scu_window_count", 0)),
        "has_credible_witness": int(audit.get("has_credible_witness_count", 0)),
    }
    return {"df": norm_df, "audit": audit, "audit_markdown": audit_md, "metrics": metrics}


def normalize(parsed_responses: dict) -> dict:
    """Run scu_normalizer.normalize() on parsed responses (parsing-flow path)."""
    if not parsed_responses:
        raise ValueError("No parsed responses to normalize.")
    return normalize_df(_raw_df_from_parsed(parsed_responses))


def filter_eligibility(norm_df, criterion_keys: list[str]) -> dict:
    """AND the selected SCU criterion columns and return the surviving rows plus
    a cumulative funnel (one stage per criterion)."""
    import scu_normalizer

    stages, values, mask = scu_normalizer.incremental_funnel(norm_df, criterion_keys)
    filtered = norm_df[mask.values] if len(mask) == len(norm_df) else norm_df[mask]
    funnel = [{"stage": s, "count": v} for s, v in zip(stages, values)]
    return {"df": filtered, "funnel": funnel, "n_passed": int(mask.sum())}
