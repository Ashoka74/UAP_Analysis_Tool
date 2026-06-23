"""Parsing service — LLM feature extraction of raw UAP report text into
structured JSON, mirroring the core of the Streamlit ``parsing.py`` page.

Scope (core tier): schema registry + deep-merge + custom fields, cost
estimation, and the *client-parallel* run mode (OpenAI / DeepSeek). The
OpenAI server-batch path, SCU xlsx export and embeddings→HDF5 are intentionally
left out of this first pass.

Heavy/optional imports (``uap_analyzer``, ``config``) are pulled in lazily so
importing this module never drags in torch or validates secrets.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable


# ── Schema registry ────────────────────────────────────────────────────────
# Rebuilt directly from config.py (no Streamlit dependency) so it stays in
# sync with parsing.py's SCHEMA_FORMATS / SCHEMA_FORMAT_GROUPS mapping.
_SCHEMA_SPECS: list[tuple[str, str]] = [
    ("SCU_v1", "FORMAT_SCU_V1"),
    ("Default UAP Format", "FORMAT_LONG"),
    ("SCU Spreadsheet", "FORMAT_LONG_XLSX"),
    ("SCU_v2", "FORMAT_SCU_V2"),
    ("SCU_v3", "FORMAT_SCU_V3"),
    ("UFOSETI (RU)", "FORMAT_UFOSETI_RU"),
    ("NUFORC", "FORMAT_NUFORC"),
    ("Blue Book (USAF)", "FORMAT_BLUE_BOOK"),
    ("UK National Archives", "FORMAT_UK_NATIONAL_ARCHIVES"),
    ("COBEPS — PAN Notifications (BE)", "FORMAT_COBEPS_NOTIFICATIONS_PAN"),
    ("COBEPS — COB 2021 (BE)", "FORMAT_COBEPS_COB_2021"),
    ("GEP (DE)", "FORMAT_GEP"),
    ("UPDB / NICAP", "FORMAT_UPDB_NICAP"),
    ("OVNIBASE (FR)", "FORMAT_OVNIBASE"),
    ("UFOSETI (raw)", "FORMAT_UFOSETI"),
    ("UAP Sightings GitHub", "FORMAT_UAP_SIGHTINGS_GITHUB"),
    ("Weinstein Pilot Catalog", "FORMAT_WEINSTEIN_PILOT_CATALOG"),
    ("Petrowitsch LATAM", "FORMAT_PETROWITSCH_LATAM"),
    ("UFOCAT 2023", "FORMAT_UFOCAT"),
    ("CISU / CISUCAT (IT)", "FORMAT_CISU_CISUCAT"),
    ("CUFOC 1977 Italy-France", "FORMAT_CUFOC_1977_ITALY_FRANCE"),
    ("UAPCHECK Registry", "FORMAT_UAPCHECK"),
]

SCHEMA_FORMAT_GROUPS: dict[str, list[str]] = {
    "Canonical & SCU": ["SCU_v1", "Default UAP Format", "SCU Spreadsheet", "SCU_v2", "SCU_v3"],
    "Government & official archives": ["Blue Book (USAF)", "UK National Archives"],
    "European national databases": [
        "COBEPS — PAN Notifications (BE)", "COBEPS — COB 2021 (BE)",
        "GEP (DE)", "OVNIBASE (FR)", "CISU / CISUCAT (IT)", "CUFOC 1977 Italy-France",
    ],
    "Research catalogs": [
        "UPDB / NICAP", "UFOCAT 2023", "Weinstein Pilot Catalog", "Petrowitsch LATAM",
    ],
    "Public & crowd-sourced": ["NUFORC", "UAP Sightings GitHub", "UAPCHECK Registry"],
    "UFOSETI / SETI": ["UFOSETI (RU)", "UFOSETI (raw)"],
}


def _schema_registry() -> dict[str, Any]:
    """Map each human label to its schema (dict or JSON string), loaded from config."""
    import config

    registry: dict[str, Any] = {}
    for label, attr in _SCHEMA_SPECS:
        if hasattr(config, attr):
            registry[label] = getattr(config, attr)
    return registry


# ── Schema merge helpers (ported from parsing.py) ──────────────────────────
def _fmt_as_dict(v: Any) -> dict:
    """Coerce a schema entry (dict or JSON string) to a dict; {} on failure."""
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except Exception:
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _flatten_dotted(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to {dotted_key: leaf_value}."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            flat.update(_flatten_dotted(v, key))
        else:
            flat[key] = v
    return flat


def list_schemas() -> dict:
    """Public: schema labels + logical groupings for the picker."""
    registry = _schema_registry()
    labels = list(registry.keys())
    groups = {g: [c for c in cols if c in registry] for g, cols in SCHEMA_FORMAT_GROUPS.items()}
    grouped = {c for cols in groups.values() for c in cols}
    ungrouped = [c for c in labels if c not in grouped]
    if ungrouped:
        groups["Other"] = ungrouped
    return {"labels": labels, "groups": groups}


def merge_schema(labels: list[str], custom_fields: dict | None = None) -> dict:
    """Deep-merge the selected schema labels (plus optional dotted custom fields)
    into a single JSON template. Returns the merged dict, its pretty JSON string,
    the flattened leaf-field list, and the extraction key for FORMAT_LONG."""
    registry = _schema_registry()
    merged: dict = {}
    for label in labels:
        if label not in registry:
            raise ValueError(f"Unknown schema: {label!r}")
        merged = _deep_merge(merged, _fmt_as_dict(registry[label]))

    if custom_fields:
        merged = _deep_merge(merged, custom_fields)

    top_keys = set(merged.keys())
    # FORMAT_LONG is the only schema whose sole top-level key is sightingDetails.
    extract_key = "sightingDetails" if top_keys == {"sightingDetails"} else None

    flat = _flatten_dotted(merged)
    fields = [
        {"path": k, "description": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}
        for k, v in flat.items()
    ]
    return {
        "schema": merged,
        "schema_json": json.dumps(merged, indent=2),
        "fields": fields,
        "extract_key": extract_key,
    }


# ── Schema ↔ dataset coverage diff ─────────────────────────────────────────
def _norm_token(s: Any) -> str:
    """Normalize a name for fuzzy matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _leaf(path: str) -> str:
    """Last dotted segment of a (possibly nested) field/column name."""
    return str(path).split(".")[-1]


def schema_coverage(merged: dict, dataset_columns: list[str]) -> dict:
    """Compare the merged schema's leaf fields against ``dataset_columns``.

    A schema field counts as *present* when some dataset column shares its
    normalized leaf name (last dotted segment, case/punctuation-insensitive) —
    e.g. schema ``sightingDetails.objectDescription.shape`` matches a dataset
    column ``shape`` or ``object.shape``. Returns per-field present flags, the
    dataset columns that matched, and the dataset-only (unmatched) columns.
    """
    flat = _flatten_dotted(merged)
    col_by_token: dict[str, str] = {}
    for c in dataset_columns:
        col_by_token.setdefault(_norm_token(_leaf(c)), c)

    coverage = []
    matched_cols: set[str] = set()
    for path in flat:
        match = col_by_token.get(_norm_token(_leaf(path)))
        if match is not None:
            matched_cols.add(match)
        coverage.append({
            "path": path,
            "leaf": _leaf(path),
            "present": match is not None,
            "matched_column": match,
        })

    schema_tokens = {_norm_token(_leaf(p)) for p in flat}
    db_only = [c for c in dataset_columns if _norm_token(_leaf(c)) not in schema_tokens]
    n_present = sum(1 for c in coverage if c["present"])
    return {
        "coverage": coverage,
        "summary": {
            "present": n_present,
            "missing": len(coverage) - n_present,
            "total": len(coverage),
            "db_only": len(db_only),
        },
        "matched_columns": sorted(matched_cols),
        "db_only_columns": db_only,
    }


def prune_schema_to_paths(merged: dict, paths: list[str]) -> dict:
    """Rebuild a nested schema dict holding only the given dotted leaf paths,
    preserving each leaf's original description/value and the original nesting
    (so a FORMAT_LONG ``sightingDetails`` wrapper is kept)."""
    flat = _flatten_dotted(merged)
    keep = set(paths)
    out: dict = {}
    for path, val in flat.items():
        if path not in keep:
            continue
        parts = path.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def schema_coverage_report(labels: list[str], dataset_columns: list[str],
                           custom_fields: dict | None = None) -> dict:
    """Coverage diff plus ready-to-use extraction-schema variants for the three
    modes the parsing UI offers:

    - ``all``      → the full merged schema (extract every field).
    - ``missing``  → schema pruned to fields absent from the dataset (🔴 only).
    - ``database`` → empty schema; keep the dataset columns as-is (no extraction).
    """
    merged = merge_schema(labels, custom_fields)["schema"]
    cov = schema_coverage(merged, dataset_columns)

    missing_paths = [c["path"] for c in cov["coverage"] if not c["present"]]
    all_paths = [c["path"] for c in cov["coverage"]]
    missing_schema = prune_schema_to_paths(merged, missing_paths)

    variants = {
        "all": {"schema_json": json.dumps(merged, indent=2), "n_fields": len(all_paths)},
        "missing": {"schema_json": json.dumps(missing_schema, indent=2),
                    "n_fields": len(missing_paths)},
        "database": {"schema_json": "{}", "n_fields": 0},
    }
    return {**cov, "variants": variants}


# ── Cost estimation ────────────────────────────────────────────────────────
def estimate(descriptions: list[str], schema_json: str, model: str,
             use_cache: bool = True, use_batch: bool = False) -> dict:
    from uap_analyzer import estimate_cost
    return estimate_cost(descriptions, schema_json, model=model,
                         use_cache=use_cache, use_batch=use_batch)


def available_models() -> dict:
    from uap_analyzer import OPENAI_MODELS, DEEPSEEK_MODELS
    return {"openai": list(OPENAI_MODELS), "deepseek": list(DEEPSEEK_MODELS)}


# ── Parsed-response → flat DataFrame (ported from convert_cached_data_to_df) ─
def parsed_responses_to_df(parsed_responses: dict):
    import pandas as pd

    if not parsed_responses:
        return pd.DataFrame()
    parsed_df_raw = pd.DataFrame(parsed_responses).T
    if set(parsed_df_raw.columns) == {"sightingDetails"}:
        df = pd.json_normalize(parsed_df_raw["sightingDetails"].tolist())
        df.index = parsed_df_raw.index
    else:
        df = pd.json_normalize(list(parsed_responses.values()))
        df.index = parsed_df_raw.index
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df


# ── Run the client-parallel parse ──────────────────────────────────────────
def run_parse(descriptions: list[str], schema_json: str, *, provider: str,
              model: str, api_key: str, max_workers: int = 10,
              progress_callback: Callable[[int, int, int], None] | None = None) -> dict:
    """Parse a list of raw report texts into structured JSON via OpenAI/DeepSeek.

    Returns a dict with ``parsed_responses`` (keyed by description), the flat
    ``df`` (pandas DataFrame), and any ``errors``.
    """
    from uap_analyzer import UAPParser

    texts = [str(d) for d in descriptions if d is not None and str(d).strip()]
    if not texts:
        raise ValueError("No non-empty descriptions to parse.")

    parser = UAPParser(
        api_key=api_key,
        model=model,
        provider=("deepseek" if provider == "deepseek" else "openai"),
        use_batch=False,
        col=texts,
    )
    parser.process_descriptions(
        texts, schema_json, max_workers=max_workers, progress_callback=progress_callback,
    )
    parsed = parser.parse_responses()
    df = parsed_responses_to_df(parsed)
    return {
        "parsed_responses": parsed,
        "df": df,
        "errors": list(parser.last_errors),
        "n_ok": len(parsed),
        "n_total": len(texts),
    }
