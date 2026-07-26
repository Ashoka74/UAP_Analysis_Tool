"""
scu_normalizer.py

Normalizes a parsed UAP DataFrame into a clean, analysis-ready form aligned
with the SCU UAP Activity Pattern / Methodology studies. This is the importable
module behind the optional "Apply SCU normalization" step in parsing.py.

What it does (and only this):
  1. Drops columns that are entirely empty.
  2. Normalizes country, state, and location_type to canonical forms.
  3. Splits the witness role field (comma-string OR JSON array) into a clean
     primary role + multi-flag boolean columns.
  4. Normalizes military.military_public; resolves `Mixed` with a documented
     fallback (facility set => Military; else Public).
  5. Parses numberOfWitnesses free-text into a numeric witness_count_num field.
  6. Normalizes craft.size to canonical bands; canonicalizes craft.primary_shape.
  7. Coerces Y/N/U/P/S flag columns to consistent uppercase single chars.
  8. Validates trustScore is on a 0-100 scale and adds a trust_band column.
  9. Builds the SCU FIVE-CRITERION eligibility gate as derived columns:
       - in_scu_window                  (1945-1975 study window)
       - has_core_fields                (Criterion 2 — date/time/location/desc)
       - has_investigation_channel      (Criterion 4 — independent investigator)
       - has_credible_witness           (Criterion 5 — accepted witness class)
       - has_anomalous_characterization (Criterion 3 — anomalous struct/flight/occ)
       - has_engagement_signal          (Phase-3 — >=1 of nine activities)
       - day_night_resolved             (date_time.day_night resolved to D or N)
       - military_public_known          (military_public_resolved is populated)
       - post_1975_window               (year >= 1975 — companion-study window)
       - reports_within_1_month         (report filed within 1 month of sighting)
       - reports_within_1_year          (report filed within 1 year — SCU Criterion 1)
       - timeliness_status              (Criterion 1 — tri-state, cannot enforce)
       - scu_eligible                   (conjunction of all of the above)
       - scu_phase{1,2,3,4}_candidate   (phase approximations)

It is deliberately conservative: it does NOT impute, de-duplicate, or rewrite
free-text narrative columns.

`normalize(df)` returns `(normalized_df, audit_dict)`.
`audit_to_markdown(audit)` renders the audit dict as a Markdown string.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "parsed_reports_with_agency.xlsx"
DEFAULT_OUTPUT = "parsed_reports_normalized.xlsx"
DEFAULT_AUDIT = "normalization_audit.md"

# Columns that are dropped ONLY when they are confirmed entirely empty.
EMPTY_COLUMNS_TO_DROP = [
    "sightingDetails.evidence.url",
    "source.duplicate",
    "sightingDetails.weatherConditions",
    "sightingDetails.uapCharacteristics",
    "sightingDetails.observerDetails",
    "sightingDetails.additionalInformation",
    "sightingDetails.evidence",
]

# The nine SCU activity categories as they appear in this dataset.
ENGAGEMENT_TYPE_COLUMNS = [
    "engagement_type.interactive_flight",
    "engagement_type.radical_flight",
    "engagement_type.loitering",
    "engagement_type.electronic_transmissions",
    "engagement_type.interference_weapons",
    "engagement_type.military_intrusions",
    "engagement_type.occupant_encounter",
    "engagement_type.occupant_observed",
    "engagement_type.close_approach",
]

# Engagement flag columns (Y/N/U).
ENGAGEMENT_FLAG_COLUMNS = [
    "engagement_flags.aircraft_engagement",
    "engagement_flags.aircraft_encounters",
    "engagement_flags.active_radar_jamming",
    "engagement_flags.over_military_installation",
    "engagement_flags.during_missile_test",
    "engagement_flags.radar_tracking",
    "engagement_flags.radio_interference",
    "engagement_flags.radar_jamming",
    "engagement_flags.directed_radar",
    "engagement_flags.coded_radar",
    "engagement_flags.multiple_interactive_flight",
]

# Effects flag columns (Y/N/U).
EFFECTS_FLAG_COLUMNS = [
    "effects.atomic_related",
    "effects.communication",
    "effects.physical_effects",
]

# Performance / 5-observables flag columns (Y/N/U).
PERFORMANCE_FLAG_COLUMNS = [
    "performance.hypersonic",
    "performance.instantaneous_acceleration",
    "performance.low_observability",
    "performance.trans_medium_travel",
    "performance.positive_lift",
]

# ── Input-column contract + auto-mapping ────────────────────────────────────
# The raw input columns this normalizer reads. Any parsed schema whose fields
# sit at different dotted paths (e.g. MasterSCU_v1 nests `object.primary_shape`,
# `engagement.engagement_type.*`, `behavior.performance.*` where this code reads
# `craft.primary_shape`, `engagement_type.*`, `performance.*`) is auto-mapped
# onto these names by ``resolve_column_mapping`` before normalization, so the
# five-criterion gate isn't silently starved of NaN. A manual override map wins
# over the automatic match.
EXPECTED_INPUT_COLUMNS = [
    "location.country", "location.state", "location.type",
    "date_time.year", "date_time.month", "date_time.day", "date_time.day_night",
    "craft.primary_shape", "craft.size",
    "witness.roles", "witness.type", "witness.count",
    "assessment.contradictsUap", "sightingDetails.trustScore",
    "investigation.source", "investigation.timeliness",
    "military.facility_name", "military.facility_type", "military.military_public",
    "sightingDetails.uapCharacteristics.presenceHumanoids",
    *EFFECTS_FLAG_COLUMNS, *ENGAGEMENT_TYPE_COLUMNS, *ENGAGEMENT_FLAG_COLUMNS,
    *PERFORMANCE_FLAG_COLUMNS,
]


# Known-schema aliases, consulted after an exact hit but before the suffix /
# leaf / fuzzy heuristics. These pin the gate inputs whose heuristic match is
# ambiguous or impossible against MasterSCU_v1's layout — e.g. `craft.size`'s
# leaf ("size") collides with `entities.tools.size`, and `investigation.source`
# has no path/leaf relative in the master schema (its equivalent is the
# top-level `source.name`). First candidate present in the data wins.
CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "craft.primary_shape": ("object.primary_shape",),
    "craft.size": ("object.size",),
    "investigation.source": ("source.name",),
    "sightingDetails.trustScore": ("assessment.trustScore",),
    "sightingDetails.uapCharacteristics.presenceHumanoids": ("entities.presenceHumanoids",),
    # For datasets without a witness.type column (e.g. Mini-SCU parses) the
    # leaf heuristic mis-matches it to location.type; pin it to witness.roles,
    # the same Military/Public/... vocabulary in array form. Exact matches
    # still win, and the gate prefers witness.roles anyway when both exist.
    "witness.type": ("witness.roles",),
}


def _match_one(canonical: str, actual_cols: list[str], fuzzy: bool, threshold: float):
    """Best actual column for one expected canonical name. Cascade:
    exact → suffix (canonical is a tail of a deeper path) → unique leaf name →
    fuzzy (difflib ratio on full path or leaf). Returns (actual, method, score)
    or (None, "unmatched", 0.0)."""
    import difflib

    lower = {a: a.lower() for a in actual_cols}
    cl = canonical.lower()
    leaf = canonical.split(".")[-1].lower()

    if canonical in actual_cols:
        return canonical, "exact", 1.0

    # known-schema alias (e.g. MasterSCU_v1 re-parented / renamed fields)
    for alias in CANONICAL_ALIASES.get(canonical, ()):
        hit = next((a for a in actual_cols if lower[a] == alias.lower()
                    or lower[a].endswith("." + alias.lower())), None)
        if hit is not None:
            return hit, "alias", 0.98

    # suffix: an actual path that ends in ".<canonical>" (extra parent prefix)
    suf = [a for a in actual_cols if lower[a] == cl or lower[a].endswith("." + cl)]
    if len(suf) == 1:
        return suf[0], "suffix", 0.99
    # unique leaf-name match
    leafm = [a for a in actual_cols if a.split(".")[-1].lower() == leaf]
    if len(leafm) == 1:
        return leafm[0], "leaf", 0.9
    if not fuzzy:
        if suf:
            return suf[0], "suffix?", 0.7
        return (leafm[0], "leaf?", 0.7) if leafm else (None, "unmatched", 0.0)
    # fuzzy — restrict to leaf candidates when several, else all columns
    cands = leafm or suf or actual_cols
    best, best_r = None, 0.0
    for a in cands:
        r = max(
            difflib.SequenceMatcher(None, cl, lower[a]).ratio(),
            difflib.SequenceMatcher(None, leaf, a.split(".")[-1].lower()).ratio(),
        )
        if r > best_r:
            best_r, best = r, a
    if best is not None and best_r >= threshold:
        note = "fuzzy" + ("/ambiguous" if len(cands) > 1 and cands is leafm else "")
        return best, note, round(best_r, 2)
    return None, "unmatched", 0.0


def resolve_column_mapping(actual_cols, manual: dict | None = None, *,
                           fuzzy: bool = True, threshold: float = 0.80,
                           expected: list | None = None) -> dict:
    """Map each expected canonical input column to an actual dataframe column.

    ``manual`` (``{canonical: actual}``) overrides the automatic match. Returns
    ``{"mapping": {canonical: actual}, "methods": {canonical: how}, "scores":
    {...}, "unmatched": [...]}`` — the mapping is what ``apply_column_mapping``
    aliases, and the rest is audit surfaced to the UI for manual correction."""
    actual = list(actual_cols)
    manual = manual or {}
    expected = expected if expected is not None else EXPECTED_INPUT_COLUMNS
    mapping, methods, scores, unmatched = {}, {}, {}, []
    for canon in expected:
        if canon in manual and manual[canon] in actual:
            mapping[canon], methods[canon], scores[canon] = manual[canon], "manual", 1.0
            continue
        a, how, sc = _match_one(canon, actual, fuzzy, threshold)
        if a is None:
            unmatched.append(canon)
        else:
            mapping[canon], methods[canon], scores[canon] = a, how, sc
    return {"mapping": mapping, "methods": methods, "scores": scores, "unmatched": unmatched}


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Alias matched actual columns to the canonical names ``normalize`` reads.
    Non-destructive: copies ``df[actual]`` to a new ``canonical`` column (only
    when the canonical isn't already present and actual != canonical)."""
    out = df.copy()
    for canon, actual in mapping.items():
        if actual == canon or canon in out.columns or actual not in out.columns:
            continue
        out[canon] = out[actual]
    return out


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# ISO-2 country codes used as canonical form. Anything unmapped passes through
# unchanged and gets logged in the audit.
COUNTRY_MAP: dict[str, str] = {
    "US": "US", "USA": "US", "United States": "US",
    "US (Guam is unincorporated territory)": "US",
    "Canada": "CA", "CA": "CA",
    "France": "FR", "FR": "FR",
    "Germany": "DE", "DE": "DE",
    "UK": "GB", "GB": "GB", "United Kingdom": "GB",
    "Italy": "IT", "IT": "IT",
    "Spain": "ES",
    "Portugal": "PT", "PT": "PT",
    "Netherlands": "NL", "NL": "NL",
    "Sweden": "SE", "SE": "SE",
    "Norway": "NO", "NO": "NO",
    "Finland": "FI",
    "Denmark": "DK", "DK": "DK",
    "Austria": "AT", "AT": "AT",
    "Greece": "GR", "GR": "GR",
    "Russia": "RU", "RU": "RU", "USSR": "SU", "Soviet Union": "SU", "SU": "SU",
    "Kazakhstan": "KZ", "KZ": "KZ",
    "Turkmenistan": "TM",
    "Azerbaijan": "AZ",
    "Japan": "JP", "JP": "JP",
    "South Korea": "KR", "KR": "KR",
    "Philippines": "PH", "PH": "PH",
    "Iran": "IR", "IR": "IR",
    "Iraq": "IQ", "IQ": "IQ",
    "Syria": "SY", "SY": "SY",
    "Turkey": "TR",
    "United Arab Emirates": "AE", "AE": "AE",
    "Brunei": "BN", "BN": "BN",
    "Australia": "AU", "AU": "AU",
    "Papua New Guinea": "PG", "PG": "PG",
    "Marshall Islands": "MH",
    "Mexico": "MX", "MX": "MX",
    "Cuba": "CU",
    "Paraguay": "PY", "PY": "PY",
    "Argentina": "AR",
    "Colombia": "CO",
    "Uruguay": "UY", "UY": "UY",
    "Chile": "CL", "CL": "CL",
    "Puerto Rico": "PR", "PR": "PR",
    "Panama": "PA", "PA": "PA",
    "Guam": "GU", "GU": "GU",
    "South Africa": "ZA",
    "Madagascar": "MG",
    "Algeria": "DZ",
    "Zimbabwe": "ZW", "ZW": "ZW",
    "Georgia": "GE", "GE": "GE",
    "Antarctica": "AQ", "AQ": "AQ",
    "International Waters": "INTL_WATERS", "INTL_WATERS": "INTL_WATERS",
    "Arabian Gulf": "INTL_WATERS",
    "Scandinavia": "SCAND", "SCAND": "SCAND",
    "NO, SE": "SCAND",
    "Africa": "AFRICA",
    "Multiple": "MULTIPLE", "MULTIPLE": "MULTIPLE",
    "Moon": "MOON", "MOON": "MOON",
    "Unknown": "UNKNOWN", "UNKNOWN": "UNKNOWN",
    "Unknown (likely Middle East per CENTCOM)": "UNKNOWN",
    "Unknown (Persian Gulf region)": "UNKNOWN",
}

US_STATE_NAME_TO_CODE: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
    "Puerto Rico": "PR", "Guam": "GU",
}

VALID_US_STATE_CODES = set(US_STATE_NAME_TO_CODE.values())

LOCATION_TYPE_MAP: dict[str, str] = {
    "population centre": "Population centre",
    "population center": "Population centre",
    "military base": "Military base",
    "atomic site": "Atomic site",
    "waste facility": "Waste facility",
    "water body": "Water body",
    "airport": "Airport",
    "desert": "Desert",
    "forest": "Forest",
    "ocean": "Ocean",
    "river": "River",
    "space flight": "Space flight",
    "missile testing": "Missile testing",
    "rural": "Rural",
    "mixed": "Mixed",
    "other": "Other",
    "unknown": "Unknown",
}

CRAFT_SHAPE_MAP: dict[str, str] = {
    "disc": "Disc", "disk": "Disc",
    "sphere": "Sphere", "spherical": "Sphere", "ball": "Sphere",
    "round": "Sphere", "circle": "Sphere", "circular": "Sphere",
    "orb": "Orb",
    "light": "Light",
    "fireball": "Fireball",
    "cigar": "Cigar",
    "cylinder": "Cylinder",
    "egg": "Egg",
    "triangle": "Triangle",
    "chevron": "Chevron",
    "cone": "Cone",
    "teardrop": "Teardrop",
    "oval": "Oval",
    "rectangle": "Rectangle",
    "diamond": "Diamond",
    "boomerang": "Boomerang",
    "saturn": "Saturn",
    "cross": "Cross",
    "tic-tac": "Tic-Tac",
    "cube": "Cube",
    "pyramid": "Pyramid",
    "dome": "Dome",
    "crescent": "Crescent",
    "other": "Other",
    "unknown": "Unknown",
}

SIZE_BAND_MAP: dict[str, str] = {
    "tiny": "Tiny",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
    "very large": "Very large",
    "massive": "Massive",
    "unknown": "Unknown",
}

SIZE_RANGE_FOR_BAND: dict[str, str] = {
    "Tiny": "<0.5 m",
    "Small": "0.5-3 m",
    "Medium": "3-10 m",
    "Large": "10-50 m",
    "Very large": "50-200 m",
    "Massive": ">200 m",
    "Unknown": "",
}

WITNESS_ROLE_TOKENS: dict[str, str] = {
    "civilian": "Civilian",
    "public": "Public",
    "military": "Military",
    "pilot": "Pilot",
    "police": "Police",
    "scientist": "Scientist",
    "intelligence": "Intelligence",
    "astronaut": "Astronaut",
    "politician": "Politician",
    "security": "Security",
    "army": "Military",
    "navy": "Military",
    "air force": "Military",
    "unknown": "Unknown",
    "unspecified": "Unknown",
}

# Accepted SCU credible witness classes (Criterion 5).
CREDIBLE_WITNESS_CLASSES = {
    "Military", "Pilot", "Police", "Public", "Civilian",
    "Scientist", "Intelligence", "Astronaut", "Politician", "Security",
}

WITNESS_COUNT_KEYWORD_MAP: dict[str, int] = {
    "multiple": 3,
    "many": 5,
    "several": 3,
    "scores": 40,
    "hundreds": 100,
    "thousands": 1000,
    "tens of thousands": 10000,
}

# Tokens that mean "this cell is empty" once stringified.
_EMPTY_TOKENS = {"", "nan", "none", "null", "<na>", "na"}

# Ordered SCU five-criterion eligibility gate: (key, boolean column, label).
# This is the canonical cascade order used by incremental_funnel(), the audit
# report, and the SCU filter UI in parsing.py.
SCU_CRITERIA = [
    ("in_scu_window",                  "in_scu_window",                  "In SCU window (1945-1975)"),
    ("has_core_fields",                "has_core_fields",                "Core fields present — Criterion 2"),
    ("has_investigation_channel",      "has_investigation_channel",      "Independent investigation — Criterion 4"),
    ("has_credible_witness",           "has_credible_witness",           "Credible witness class — Criterion 5"),
    ("has_anomalous_characterization", "has_anomalous_characterization", "Anomalous characterization — Criterion 3"),
    ("has_engagement_signal",          "has_engagement_signal",          "Engagement signal, 1 of 9 — Phase-3"),
    ("day_night_resolved",             "day_night_resolved",             "Day/night resolved (D or N)"),
    ("military_public_known",          "military_public_known",          "Military/public resolved"),
]

# Additional criteria available to presets and custom filters but NOT part of
# the canonical 1945-1975 five-criterion gate (SCU_CRITERIA).
SCU_EXTRA_CRITERIA = [
    ("post_1975_window",       "post_1975_window",       "Post-1975 window (1975 onwards)"),
    ("reports_within_1_month", "reports_within_1_month",  "Report filed within 1 month of sighting"),
    ("reports_within_1_year",  "reports_within_1_year",   "Report filed within 1 year of sighting"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_or_none(value: object) -> str | None:
    """Return a stripped string, or None for NaN / empty / 'nan'-string cells."""
    if isinstance(value, (list, dict)):
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    s = str(value).strip()
    if s.lower() in _EMPTY_TOKENS:
        return None
    return s if s else None


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if present, else an all-NaN Series aligned to df."""
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index)


def _is_empty_series(series: pd.Series) -> bool:
    """True if every value in the series is empty (NaN, '', 'nan', [], {})."""
    def _empty(v: object) -> bool:
        if isinstance(v, (list, dict)):
            return len(v) == 0
        try:
            if pd.isna(v):
                return True
        except (ValueError, TypeError):
            return False
        return str(v).strip().lower() in _EMPTY_TOKENS
    return bool(series.map(_empty).all())


def normalize_country(raw: object) -> tuple[str | None, bool]:
    """Map a country string to an ISO-2 code. Returns (code, was_changed)."""
    s = _strip_or_none(raw)
    if s is None:
        return None, False
    mapped = COUNTRY_MAP.get(s)
    if mapped is None:
        for k, v in COUNTRY_MAP.items():
            if k.lower() == s.lower():
                return v, (v != s)
        return s, False  # unmapped — pass through, logged
    return mapped, (mapped != s)


def normalize_us_state(raw: object) -> tuple[str | None, bool]:
    """Map a US state to its 2-letter code. Non-US / multi-state kept as-is."""
    s = _strip_or_none(raw)
    if s is None:
        return None, False
    if s.upper() in VALID_US_STATE_CODES and len(s) == 2:
        return (s.upper(), True) if s != s.upper() else (s, False)
    if s in US_STATE_NAME_TO_CODE:
        return US_STATE_NAME_TO_CODE[s], True
    title = s.title()
    if title in US_STATE_NAME_TO_CODE:
        return US_STATE_NAME_TO_CODE[title], True
    return s, False


def normalize_location_type(raw: object) -> tuple[str | None, bool]:
    """Canonicalize location.type to one of the documented buckets."""
    s = _strip_or_none(raw)
    if s is None:
        return None, False
    key = s.lower()
    if key in LOCATION_TYPE_MAP:
        canon = LOCATION_TYPE_MAP[key]
        return canon, (canon != s)
    for sep in (",", "/"):
        if sep in s:
            first = s.split(sep)[0].strip().lower()
            if first in LOCATION_TYPE_MAP:
                return LOCATION_TYPE_MAP[first], True
    return s, False


def normalize_craft_shape(raw: object) -> tuple[str | None, bool]:
    """Canonicalize craft.primary_shape."""
    s = _strip_or_none(raw)
    if s is None:
        return None, False
    key = s.lower()
    if key in CRAFT_SHAPE_MAP:
        canon = CRAFT_SHAPE_MAP[key]
        return canon, (canon != s)
    return s, False


def normalize_craft_size(raw: object) -> tuple[str | None, str, bool]:
    """Map a craft.size string to a canonical band + documented metre range.

    Example: "Small (0.5-3 m)" -> ("Small", "0.5-3 m", True)
             "Medium"          -> ("Medium", "3-10 m", True)
             "Unknown"         -> ("Unknown", "", False)
    """
    s = _strip_or_none(raw)
    if s is None:
        return None, "", False
    bare = re.sub(r"\s*\(.*?\)\s*", "", s).strip()
    key = bare.lower()
    band: str | None = None
    if key in SIZE_BAND_MAP:
        band = SIZE_BAND_MAP[key]
    else:
        for token in SIZE_BAND_MAP:
            if key.startswith(token):
                band = SIZE_BAND_MAP[token]
                break
    if band is None:
        return s, "", False
    range_str = SIZE_RANGE_FOR_BAND.get(band, "")
    changed = bool((band != s) or (range_str and range_str not in s))
    return band, range_str, changed


def _roles_from_value(value: object) -> tuple[list[str], bool]:
    """Return canonical witness roles from either a JSON array or a combo string.

    Returns (roles, had_multi).
    """
    # JSON array (FORMAT_SCU_V2 witness.roles)
    if isinstance(value, (list, tuple, np.ndarray)):
        seen: list[str] = []
        for tok in value:
            token = re.sub(r"\s*\(.*?\)\s*", "", str(tok)).strip().lower()
            token = re.sub(r"\?+$", "", token).strip()
            canon = WITNESS_ROLE_TOKENS.get(token)
            if canon and canon not in seen:
                seen.append(canon)
        return seen, len(seen) > 1
    # Comma / semicolon / "and"-separated string (legacy witness.type)
    s = _strip_or_none(value)
    if s is None:
        return [], False
    parts = re.split(r"[,;/]| and ", s)
    seen = []
    for part in parts:
        token = re.sub(r"\s*\(.*?\)\s*", "", part).strip().lower()
        token = re.sub(r"\?+$", "", token).strip()
        if not token:
            continue
        canon = WITNESS_ROLE_TOKENS.get(token)
        if canon and canon not in seen:
            seen.append(canon)
    return seen, len(seen) > 1


def parse_witness_count(raw: object) -> tuple[int | None, bool]:
    """Convert numberOfWitnesses free text to a lower-bound integer.
    Returns (count, was_parsed_from_text)."""
    if isinstance(raw, (list, dict)):
        return None, False
    try:
        if pd.isna(raw):
            return None, False
    except (ValueError, TypeError):
        pass
    s = str(raw).strip()
    if not s or s.lower() in _EMPTY_TOKENS:
        return None, False
    if re.fullmatch(r"\d+", s):
        return int(s), False
    if re.fullmatch(r"\d+\.0", s):  # numeric column stringified as "3.0"
        return int(float(s)), False
    m = re.fullmatch(r"(\d+)\+", s)
    if m:
        return int(m.group(1)), True
    m = re.search(
        r"\b(?:at least|over|approximately|approx\.?|nearly|about|around)\s+(\d+)\b",
        s, flags=re.I,
    )
    if m:
        return int(m.group(1)), True
    s_low = s.lower()
    if "several hundred" in s_low:
        return 300, True
    if "tens of thousands" in s_low:
        return 10000, True
    for kw, val in WITNESS_COUNT_KEYWORD_MAP.items():
        if kw in s_low:
            return val, True
    m = re.match(r"^\s*(\d+)\b", s)
    if m:
        return int(m.group(1)), True
    return None, True  # parsed but couldn't extract a number


def normalize_flag(raw: object) -> tuple[str | None, bool]:
    """Normalize Y/N/P/S/U single-character flag values to uppercase."""
    s = _strip_or_none(raw)
    if s is None:
        return None, False
    upper = s.upper()
    if upper in {"YES", "TRUE", "1"}:
        return ("Y", upper != "Y")
    if upper in {"NO", "FALSE", "0"}:
        return ("N", upper != "N")
    if len(upper) == 1 and upper in {"Y", "N", "P", "S", "U", "D"}:
        return (upper, upper != s)
    return s, False


def trust_band(score: float | None) -> str | None:
    """Bin a 0-100 trustScore into one of five bands."""
    try:
        if score is None or pd.isna(score):
            return None
    except (ValueError, TypeError):
        return None
    s = float(score)
    if s < 20:
        return "very_low"
    if s < 40:
        return "low"
    if s < 60:
        return "mid"
    if s < 80:
        return "high"
    return "very_high"


def resolve_military_public(
    raw_military_public: object,
    facility_name: object,
    facility_type: object,
) -> tuple[str | None, str]:
    """Apply the SCU binary rule. `Mixed` is resolved as:
       - facility_name OR facility_type set => Military
       - else Public
    Returns (resolved, rule_used)."""
    s = _strip_or_none(raw_military_public)
    if s is None:
        return None, "passthrough_null"
    if s == "Military":
        return "Military", "kept"
    if s == "Public":
        return "Public", "kept"
    if s == "Mixed":
        if _strip_or_none(facility_name) or _strip_or_none(facility_type):
            return "Military", "mixed_resolved_facility_set"
        return "Public", "mixed_resolved_default_public"
    return s, "passthrough_unknown"


def _is_truthy_flag(series: pd.Series) -> pd.Series:
    """Boolean mask: True where the value reads as an affirmative flag."""
    return series.astype(str).str.strip().str.lower().isin(
        {"y", "yes", "true", "1", "p", "s"}
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame, *, column_map: dict | None = None,
              auto_map: bool = True, fuzzy: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run all normalizations and the SCU five-criterion gate.

    Before normalizing, input columns are auto-mapped onto the canonical names
    this code reads (``auto_map``, exact→suffix→leaf→fuzzy) so schemas that nest
    fields differently (e.g. MasterSCU_v1's ``object.*`` / ``engagement.*`` /
    ``behavior.performance.*``) still resolve. ``column_map`` (``{canonical:
    actual}``) is a manual override that wins over the automatic match. The
    resolved mapping + how each column matched is returned in the audit.

    Returns (normalized_df, audit_dict).
    """
    audit: dict[str, object] = {
        "input_rows": len(df),
        "input_cols": len(df.columns),
        "dropped_columns": [],
        "unmapped_countries": {},
        "unmapped_states_us": {},
        "unmapped_location_types": {},
        "unmapped_craft_shapes": {},
        "unmapped_craft_sizes": {},
        "witness_unparseable": {},
        "mixed_resolved_to_military": 0,
        "mixed_resolved_to_public": 0,
    }

    out = df.copy()

    # 0. Auto-map the parsed schema's columns onto the canonical names the gate
    #    reads (+ any manual override), so re-parented fields aren't lost as NaN.
    if auto_map or column_map:
        res = resolve_column_mapping(list(out.columns), manual=column_map, fuzzy=fuzzy)
        out = apply_column_mapping(out, res["mapping"])
        audit["column_mapping"] = res["mapping"]
        audit["column_mapping_methods"] = res["methods"]
        audit["column_mapping_scores"] = res["scores"]
        audit["column_mapping_unmatched"] = res["unmatched"]
        # Surfaced so a UI can render the editable mapping (canonical ← actual).
        audit["input_columns"] = list(df.columns)
        audit["expected_input_columns"] = list(EXPECTED_INPUT_COLUMNS)

    # 1. Drop columns that are entirely empty.
    present_to_drop = [
        c for c in EMPTY_COLUMNS_TO_DROP
        if c in out.columns and _is_empty_series(out[c])
    ]
    out = out.drop(columns=present_to_drop)
    audit["dropped_columns"] = present_to_drop

    # 2. Country.
    if "location.country" in out.columns:
        new_vals, changed_mask, unmapped = [], [], []
        country_keys_lower = {k.lower() for k in COUNTRY_MAP}
        for v in out["location.country"]:
            code, changed = normalize_country(v)
            new_vals.append(code)
            changed_mask.append(changed)
            raw = _strip_or_none(v)
            if raw is not None and raw.lower() not in country_keys_lower:
                unmapped.append(raw)
        out["location_country_iso"] = new_vals
        audit["country_changes"] = int(np.sum(changed_mask))
        audit["unmapped_countries"] = (
            pd.Series(unmapped).value_counts().to_dict() if unmapped else {}
        )

    # 3. US state.
    if "location.state" in out.columns:
        new_vals, changed_mask, unmapped_us = [], [], []
        is_us = _col(out, "location_country_iso") == "US"
        for v, us in zip(out["location.state"], is_us):
            code, changed = normalize_us_state(v)
            new_vals.append(code)
            changed_mask.append(changed)
            if (us and code is not None and code not in VALID_US_STATE_CODES
                    and len(str(code)) > 2):
                unmapped_us.append(str(code))
        out["location_state_norm"] = new_vals
        audit["state_changes"] = int(np.sum(changed_mask))
        audit["unmapped_states_us"] = (
            pd.Series(unmapped_us).value_counts().to_dict() if unmapped_us else {}
        )

    # 4. Location type.
    if "location.type" in out.columns:
        new_vals, changed_mask, unmapped = [], [], []
        for v in out["location.type"]:
            canon, changed = normalize_location_type(v)
            new_vals.append(canon)
            changed_mask.append(changed)
            raw = _strip_or_none(v)
            if raw is not None:
                key = raw.lower()
                first_token = re.split(r"[,/]", raw)[0].strip().lower()
                if key not in LOCATION_TYPE_MAP and first_token not in LOCATION_TYPE_MAP:
                    unmapped.append(raw)
        out["location_type_norm"] = new_vals
        audit["location_type_changes"] = int(np.sum(changed_mask))
        audit["unmapped_location_types"] = (
            pd.Series(unmapped).value_counts().to_dict() if unmapped else {}
        )

    # 5. Craft primary shape.
    if "craft.primary_shape" in out.columns:
        new_vals, changed_mask, unmapped = [], [], []
        for v in out["craft.primary_shape"]:
            canon, changed = normalize_craft_shape(v)
            new_vals.append(canon)
            changed_mask.append(changed)
            raw = _strip_or_none(v)
            if raw is not None and raw.lower() not in CRAFT_SHAPE_MAP:
                unmapped.append(raw)
        out["craft_primary_shape_norm"] = new_vals
        audit["craft_shape_changes"] = int(np.sum(changed_mask))
        audit["unmapped_craft_shapes"] = (
            pd.Series(unmapped).value_counts().to_dict() if unmapped else {}
        )

    # 6. Craft size.
    if "craft.size" in out.columns:
        bands, ranges, changed_mask, unmapped = [], [], [], []
        for v in out["craft.size"]:
            band, rng, changed = normalize_craft_size(v)
            bands.append(band)
            ranges.append(rng)
            changed_mask.append(changed)
            raw = _strip_or_none(v)
            if raw is not None and band is None:
                unmapped.append(raw)
        out["craft_size_band"] = bands
        out["craft_size_range"] = ranges
        audit["craft_size_changes"] = int(np.sum(changed_mask))
        audit["unmapped_craft_sizes"] = (
            pd.Series(unmapped).value_counts().to_dict() if unmapped else {}
        )

    # 7. Witness roles — accepts either witness.roles (array) or witness.type.
    witness_col = (
        "witness.roles" if "witness.roles" in out.columns
        else "witness.type" if "witness.type" in out.columns
        else None
    )
    if witness_col is not None:
        primary_roles, multi_flag = [], []
        is_military, is_pilot, is_police = [], [], []
        is_civilian_public, is_scientist = [], []
        for v in out[witness_col]:
            roles, multi = _roles_from_value(v)
            primary_roles.append(roles[0] if roles else None)
            multi_flag.append(multi)
            is_military.append("Military" in roles)
            is_pilot.append("Pilot" in roles)
            is_police.append("Police" in roles)
            is_civilian_public.append(("Civilian" in roles) or ("Public" in roles))
            is_scientist.append("Scientist" in roles)
        out["witness_primary_role"] = primary_roles
        out["witness_multi_role"] = multi_flag
        out["witness_is_military"] = is_military
        out["witness_is_pilot"] = is_pilot
        out["witness_is_police"] = is_police
        out["witness_is_civilian_public"] = is_civilian_public
        out["witness_is_scientist"] = is_scientist

    # Guarantee the witness flags exist (so the credibility gate never KeyErrors).
    for _wc in ["witness_is_military", "witness_is_pilot", "witness_is_police",
                "witness_is_civilian_public", "witness_is_scientist"]:
        if _wc not in out.columns:
            out[_wc] = False

    # 8. Witness count parse.
    wc_col = "sightingDetails.observerDetails.numberOfWitnesses"
    if wc_col in out.columns:
        counts, was_text, unparseable = [], [], []
        for v in out[wc_col]:
            n, parsed = parse_witness_count(v)
            counts.append(n)
            was_text.append(parsed and n is not None)
            if parsed and n is None:
                unparseable.append(str(v))
        out["witness_count_num"] = counts
        out["witness_count_from_text"] = was_text
        audit["witness_unparseable"] = (
            pd.Series(unparseable).value_counts().to_dict() if unparseable else {}
        )

    # 9. Military/public binary with Mixed resolution.
    if "military.military_public" in out.columns:
        resolved, rule_used = [], []
        fnames = _col(out, "military.facility_name")
        ftypes = _col(out, "military.facility_type")
        for mp, fname, ftype in zip(out["military.military_public"], fnames, ftypes):
            v, rule = resolve_military_public(mp, fname, ftype)
            resolved.append(v)
            rule_used.append(rule)
        out["military_public_resolved"] = resolved
        out["military_public_rule"] = rule_used
        audit["mixed_resolved_to_military"] = int(
            sum(r == "mixed_resolved_facility_set" for r in rule_used)
        )
        audit["mixed_resolved_to_public"] = int(
            sum(r == "mixed_resolved_default_public" for r in rule_used)
        )

    # 10. Flag column uppercase normalization.
    flag_cols = (
        ENGAGEMENT_TYPE_COLUMNS
        + ENGAGEMENT_FLAG_COLUMNS
        + EFFECTS_FLAG_COLUMNS
        + PERFORMANCE_FLAG_COLUMNS
        + ["engagement_type.no_engagement", "date_time.day_night",
           "investigation.reports_within_1_month_of_sighting",
           "investigation.reports_within_1_year_of_sighting"]
    )
    for c in flag_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: normalize_flag(v)[0])

    # 11. Trust score validation + banding.
    if "sightingDetails.trustScore" in out.columns:
        ts = pd.to_numeric(out["sightingDetails.trustScore"], errors="coerce")
        out["sightingDetails.trustScore"] = ts
        audit["trust_out_of_range_count"] = int(
            ts[(ts < 0) | (ts > 100)].notna().sum()
        )
        out["trust_band"] = ts.map(trust_band)
    else:
        ts = pd.Series([np.nan] * len(out), index=out.index)

    # 12. SCU FIVE-CRITERION eligibility gate (derived columns).

    # Study windows — the 1945-1975 SCU master window and the post-1975
    # companion-study window (year >= 1975, inclusive — the "1975 onwards" set;
    # overlaps the master window on 1975 by design).
    year = pd.to_numeric(_col(out, "date_time.year"), errors="coerce")
    out["in_scu_window"] = ((year >= 1945) & (year <= 1975)).fillna(False)
    out["post_1975_window"] = (year >= 1975).fillna(False)

    # Criterion 2 — core fields (date + country).
    out["has_core_fields"] = (
        _col(out, "date_time.year").notna()
        & _col(out, "date_time.month").notna()
        & _col(out, "date_time.day").notna()
        & _col(out, "location_country_iso").notna()
    )

    # Criterion 4 — independent investigation channel.
    out["has_investigation_channel"] = (
        _col(out, "investigation.source").map(lambda v: _strip_or_none(v) is not None)
    )

    # Phase-3 signal — >= 1 of the nine engagement activities (P or S).
    eng_present = [c for c in ENGAGEMENT_TYPE_COLUMNS if c in out.columns]
    if eng_present:
        out["has_engagement_signal"] = (
            out[eng_present].apply(lambda s: s.isin(["P", "S"])).any(axis=1)
        )
    else:
        out["has_engagement_signal"] = pd.Series([False] * len(out), index=out.index)

    # Criterion 5 — accepted credible witness class.
    out["has_credible_witness"] = (
        out["witness_is_military"].astype(bool)
        | out["witness_is_pilot"].astype(bool)
        | out["witness_is_police"].astype(bool)
        | out["witness_is_civilian_public"].astype(bool)
        | out["witness_is_scientist"].astype(bool)
    )

    # Criterion 3 — anomalous characterization (structure / flight / occupant).
    shape_norm = _col(out, "craft_primary_shape_norm").astype(str)
    has_anom_shape = (
        shape_norm.str.strip().str.lower().ne("unknown")
        & ~shape_norm.str.strip().str.lower().isin(_EMPTY_TOKENS)
    )
    perf_present = [c for c in PERFORMANCE_FLAG_COLUMNS if c in out.columns]
    if perf_present:
        has_anom_flight = (
            out[perf_present].apply(lambda s: _is_truthy_flag(s)).any(axis=1)
        )
    else:
        has_anom_flight = pd.Series([False] * len(out), index=out.index)
    has_anom_occupant = (
        _col(out, "engagement_type.occupant_observed").isin(["P", "S"])
        | _col(out, "engagement_type.occupant_encounter").isin(["P", "S"])
        | _col(out, "sightingDetails.uapCharacteristics.presenceHumanoids")
            .astype(str).str.strip().str.upper().str.startswith("Y")
    )
    out["has_anomalous_characterization"] = (
        has_anom_shape | has_anom_flight | has_anom_occupant
    )

    # Source itself denies a UAP (FORMAT_SCU_V2 assessment.contradictsUap).
    out["contradicts_uap"] = (
        _col(out, "assessment.contradictsUap")
        .astype(str).str.strip().str.lower().isin({"true", "1", "y", "yes"})
    )

    # Criterion 1 — timeliness. Cannot be enforced from columns alone; expose
    # a tri-state flag rather than forcing a (potentially false) boolean.
    out["timeliness_status"] = out["has_investigation_channel"].map(
        {True: "presumed_timely_via_source", False: "unknown_no_investigation"}
    ).fillna("unknown_no_investigation")

    out["day_night_resolved"] = _col(out, "date_time.day_night").isin(["D", "N"])
    out["military_public_known"] = _col(out, "military_public_resolved").map(
        lambda v: _strip_or_none(v) is not None
    )
    out["reports_within_1_month"] = _is_truthy_flag(
        _col(out, "investigation.reports_within_1_month_of_sighting")
    )
    # Within one year is implied by within one month, so OR them together.
    out["reports_within_1_year"] = (
        _is_truthy_flag(_col(out, "investigation.reports_within_1_year_of_sighting"))
        | out["reports_within_1_month"]
    )

    # Corrected scu_eligible — the full SCU five-criterion eligibility gate.
    out["scu_eligible"] = (
        out["in_scu_window"]                     # 1945 <= year <= 1975
        & out["has_core_fields"]                 # Criterion 2 — date/time/location/description
        & out["has_investigation_channel"]       # Criterion 4 — independent investigator
        & out["has_credible_witness"]            # Criterion 5 — accepted witness class
        & out["has_anomalous_characterization"]  # Criterion 3 — anomalous structure/flight/occupants
        & out["has_engagement_signal"]           # Phase-3 — at least 1 of 9 activities
        & out["day_night_resolved"]              # day/night classifiable (D or N)
        & out["military_public_known"]           # military/public resolved
    )

    # Phase 1-4 candidate approximations (proxies — see corrections doc §4.6).
    over_mil = _is_truthy_flag(_col(out, "engagement_flags.over_military_installation"))
    atomic = _is_truthy_flag(_col(out, "effects.atomic_related"))
    ac_eng = _is_truthy_flag(_col(out, "engagement_flags.aircraft_engagement"))
    ac_enc = _is_truthy_flag(_col(out, "engagement_flags.aircraft_encounters"))
    out["scu_phase1_candidate"] = out["scu_eligible"] & over_mil
    out["scu_phase2_candidate"] = out["scu_eligible"] & (atomic | over_mil)
    out["scu_phase3_candidate"] = out["scu_eligible"]
    out["scu_phase4_candidate"] = out["scu_eligible"] & (
        ac_eng | ac_enc | out["has_engagement_signal"]
    )

    # Cumulative cascade funnel for the audit (full eight-criterion gate).
    _stages, _values, _ = incremental_funnel(out, [k for k, _, _ in SCU_CRITERIA])
    audit["cascade_funnel"] = dict(zip(_stages, _values))

    audit["in_scu_window_count"] = int(out["in_scu_window"].sum())
    audit["post_1975_window_count"] = int(out["post_1975_window"].sum())
    audit["outside_window_count"] = int((~out["in_scu_window"]).sum())
    audit["has_core_fields_count"] = int(out["has_core_fields"].sum())
    audit["has_investigation_channel_count"] = int(
        out["has_investigation_channel"].sum()
    )
    audit["has_credible_witness_count"] = int(out["has_credible_witness"].sum())
    audit["has_anomalous_characterization_count"] = int(
        out["has_anomalous_characterization"].sum()
    )
    audit["has_engagement_signal_count"] = int(out["has_engagement_signal"].sum())
    audit["reports_within_1_month_count"] = int(out["reports_within_1_month"].sum())
    audit["reports_within_1_year_count"] = int(out["reports_within_1_year"].sum())
    audit["contradicts_uap_count"] = int(out["contradicts_uap"].sum())
    audit["scu_eligible_count"] = int(out["scu_eligible"].sum())
    audit["scu_eligible_with_trust_ge_60"] = int(
        (out["scu_eligible"] & (ts >= 60)).sum()
    )

    # Manual-review hook — eligible rows whose narrative looks suspicious.
    text_cols = [
        c for c in [
            "sightingDetails.DenseNarrativeSection", "case_text.text",
            "assessment.notes", "anomaly.summary",
        ] if c in out.columns
    ]
    if text_cols:
        pat = re.compile(
            r"no actual uap|sarcastic|hoax|misidentif|not a uap|not a ufo",
            re.IGNORECASE,
        )
        joined = out[text_cols].astype(str).agg(" ".join, axis=1)
        suspicious = out["scu_eligible"] & joined.str.contains(pat)
        audit["suspicious_eligible_count"] = int(suspicious.sum())
        audit["suspicious_eligible_rows"] = [
            str(i) for i in out.index[suspicious].tolist()[:20]
        ]

    audit["output_rows"] = len(out)
    audit["output_cols"] = len(out.columns)
    return out, audit


# ---------------------------------------------------------------------------
# Incremental-filter funnel
# ---------------------------------------------------------------------------

def incremental_funnel(
    df: pd.DataFrame, criterion_keys: list[str]
) -> tuple[list[str], list[int], pd.Series]:
    """Cumulative row count as each SCU criterion is ANDed in, in order.

    Returns (stage_labels, stage_values, final_mask):
      - stage_labels — ["All parsed rows", <label per criterion>...]
      - stage_values — row count surviving up to and including each stage
      - final_mask   — boolean Series for rows passing every selected criterion

    Unknown keys, or keys whose column is absent, are recorded as a stage with
    no additional drop (the mask is left unchanged for that step).
    """
    _registry = SCU_CRITERIA + SCU_EXTRA_CRITERIA
    col_by_key = {k: c for k, c, _ in _registry}
    label_by_key = {k: lbl for k, _, lbl in _registry}
    stages = ["All parsed rows"]
    values = [len(df)]
    mask = pd.Series([True] * len(df), index=df.index)
    for key in criterion_keys:
        col = col_by_key.get(key)
        if col is not None and col in df.columns:
            mask = mask & df[col].fillna(False).astype(bool)
        stages.append(label_by_key.get(key, key))
        values.append(int(mask.sum()))
    return stages, values, mask


# ---------------------------------------------------------------------------
# Audit-report renderer
# ---------------------------------------------------------------------------

def audit_to_markdown(audit: dict) -> str:
    """Render the audit dict produced by normalize() as a Markdown string.

    Each section is followed by a *proto-code* block — compact pseudocode that
    mirrors the corresponding logic in `normalize()` — so a reader can see how
    every number above it was derived without opening the source.
    """
    lines: list[str] = ["# Normalization audit\n"]

    def _proto(*code_lines: str) -> None:
        """Append a fenced proto-code (pseudocode) explanation block."""
        lines.append("> _Proto-code — how the figures above are computed:_")
        lines.append("```text")
        lines.extend(code_lines)
        lines.append("```")
        lines.append("")

    lines.append(
        "_This report is produced by `scu_normalizer.normalize()`. The proto-code "
        "blocks below each section are pseudocode mirrors of the real logic, not "
        "runnable code._\n"
    )

    lines.append(f"- Input rows: **{audit.get('input_rows', 0)}**")
    lines.append(f"- Input columns: **{audit.get('input_cols', 0)}**")
    lines.append(f"- Output rows: **{audit.get('output_rows', 0)}**")
    lines.append(f"- Output columns: **{audit.get('output_cols', 0)}**\n")
    _proto(
        "input_rows  = len(df)                  # rows are never dropped,",
        "output_rows = len(normalized_df)       #   so output_rows == input_rows",
        "input_cols  = len(df.columns)",
        "output_cols = input_cols",
        "            - len(dropped_columns)     # entirely-empty cols removed",
        "            + len(derived_columns)     # *_norm / *_iso / SCU gate flags added",
    )

    lines.append("## Dropped (entirely empty) columns")
    dropped = audit.get("dropped_columns", [])
    if dropped:
        lines.extend(f"- `{c}`" for c in dropped)
    else:
        lines.append("_None._")
    lines.append("")
    _proto(
        "for col in EMPTY_COLUMNS_TO_DROP:      # curated drop-list",
        "    if col in df and every value in df[col] is NaN/'' :",
        "        drop col   and  record it here",
    )

    def _section(title: str, key: str) -> None:
        lines.append(f"## {title}")
        data = audit.get(key, {})
        if not data:
            lines.append("_No changes recorded._\n")
            return
        if isinstance(data, dict):
            for k, v in sorted(data.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{k}` × {v}")
        else:
            lines.append(f"- {data}")
        lines.append("")

    lines.append(
        f"## Country normalization\n- Values changed: "
        f"**{audit.get('country_changes', 0)}**\n"
    )
    _section("Unmapped country values (passed through unchanged)", "unmapped_countries")
    _proto(
        "for v in df['location.country']:",
        "    iso, changed = normalize_country(v)   # COUNTRY_MAP[v.lower()] -> ISO-2",
        "    if changed: country_changes += 1",
        "    if v.lower() not in COUNTRY_MAP: record v as unmapped (verbatim)",
        "-> writes column  location_country_iso",
    )

    lines.append(
        f"## US state normalization\n- Values changed: "
        f"**{audit.get('state_changes', 0)}**\n"
    )
    _section("Unmapped US-state values", "unmapped_states_us")
    _proto(
        "is_us = (location_country_iso == 'US')",
        "for v, us in zip(df['location.state'], is_us):",
        "    code, changed = normalize_us_state(v)  # name/abbrev -> 2-letter code",
        "    if changed: state_changes += 1",
        "    if us and code not in VALID_US_STATE_CODES and len(code) > 2:",
        "        record code as unmapped",
        "-> writes column  location_state_norm",
    )

    lines.append(
        f"## Location type normalization\n- Values changed: "
        f"**{audit.get('location_type_changes', 0)}**\n"
    )
    _section("Unmapped location types", "unmapped_location_types")
    _proto(
        "for v in df['location.type']:",
        "    canon, changed = normalize_location_type(v)",
        "    key         = v.lower()",
        "    first_token = v.split on ',' or '/' [0].lower()",
        "    if key not in LOCATION_TYPE_MAP and first_token not in LOCATION_TYPE_MAP:",
        "        record v as unmapped",
        "-> writes column  location_type_norm",
    )

    lines.append(
        f"## Craft shape normalization\n- Values changed: "
        f"**{audit.get('craft_shape_changes', 0)}**\n"
    )
    _section("Unmapped craft shapes", "unmapped_craft_shapes")
    _proto(
        "for v in df['craft.primary_shape']:",
        "    canon, changed = normalize_craft_shape(v)  # CRAFT_SHAPE_MAP[v.lower()]",
        "    if v.lower() not in CRAFT_SHAPE_MAP: record v as unmapped",
        "-> writes column  craft_primary_shape_norm",
    )

    lines.append(
        f"## Craft size normalization\n- Values changed: "
        f"**{audit.get('craft_size_changes', 0)}**\n"
    )
    _section("Unmapped craft sizes", "unmapped_craft_sizes")
    _proto(
        "for v in df['craft.size']:",
        "    band, range_, changed = normalize_craft_size(v)  # text -> size band",
        "    if band is None: record v as unmapped",
        "-> writes columns  craft_size_band, craft_size_range",
    )

    _section("Witness-count strings that could not be reduced to a number",
             "witness_unparseable")
    _proto(
        "for v in df['sightingDetails.observerDetails.numberOfWitnesses']:",
        "    n, parsed = parse_witness_count(v)   # pull an integer out of free text",
        "    if parsed and n is None: record v as unparseable",
        "-> writes columns  witness_count_num, witness_count_from_text",
    )

    lines.append("## Military / public Mixed resolution")
    lines.append(
        f"- `Mixed` → **Military** (facility set): "
        f"{audit.get('mixed_resolved_to_military', 0)}"
    )
    lines.append(
        f"- `Mixed` → **Public** (no facility):    "
        f"{audit.get('mixed_resolved_to_public', 0)}\n"
    )
    _proto(
        "for mp, fname, ftype in military columns:",
        "    value, rule = resolve_military_public(mp, fname, ftype)",
        "    if mp == 'Mixed':",
        "        if fname or ftype is set -> 'Military'  (mixed_resolved_facility_set)",
        "        else                     -> 'Public'    (mixed_resolved_default_public)",
        "-> writes columns  military_public_resolved, military_public_rule",
    )

    lines.append("## Trust score")
    lines.append(
        f"- Out-of-range scores (< 0 or > 100): "
        f"{audit.get('trust_out_of_range_count', 0)}\n"
    )
    _proto(
        "ts = to_numeric(df['sightingDetails.trustScore'], errors='coerce')",
        "trust_out_of_range_count = count( ts.notna() and (ts < 0 or ts > 100) )",
        "trust_band               = trust_band(ts)   # e.g. low / medium / high",
    )

    lines.append("## SCU five-criterion cascade (cumulative)")
    funnel = audit.get("cascade_funnel", {})
    if funnel:
        for k, v in funnel.items():
            lines.append(f"- {k}: **{v}**")
    else:
        lines.append("_Not computed._")
    lines.append("")
    _proto(
        "mask = [True] * len(df)                 # start with every row",
        "for key in SCU_CRITERIA (in cascade order):",
        "    mask = mask AND df[criterion_column(key)]   # absent col -> no drop",
        "    record surviving_count = mask.sum()  # each value above is cumulative",
        "# values only ever shrink: each stage ANDs one more criterion in.",
    )

    lines.append("## SCU gate counts (after normalization)")
    lines.append(f"- `in_scu_window` (1945-1975): **{audit.get('in_scu_window_count', 0)}**")
    lines.append(f"- `post_1975_window` (1975 onwards): **{audit.get('post_1975_window_count', 0)}**")
    lines.append(f"- rows outside the SCU window: **{audit.get('outside_window_count', 0)}**")
    lines.append(f"- `has_core_fields`: **{audit.get('has_core_fields_count', 0)}**")
    lines.append(
        f"- `has_investigation_channel`: "
        f"**{audit.get('has_investigation_channel_count', 0)}**"
    )
    lines.append(
        f"- `has_credible_witness`: **{audit.get('has_credible_witness_count', 0)}**"
    )
    lines.append(
        f"- `has_anomalous_characterization`: "
        f"**{audit.get('has_anomalous_characterization_count', 0)}**"
    )
    lines.append(
        f"- `has_engagement_signal`: **{audit.get('has_engagement_signal_count', 0)}**"
    )
    lines.append(
        f"- `reports_within_1_month`: "
        f"**{audit.get('reports_within_1_month_count', 0)}**"
    )
    lines.append(
        f"- `reports_within_1_year`: "
        f"**{audit.get('reports_within_1_year_count', 0)}**"
    )
    lines.append(f"- `contradicts_uap`: **{audit.get('contradicts_uap_count', 0)}**")
    lines.append(f"- **`scu_eligible = True`: {audit.get('scu_eligible_count', 0)}**")
    lines.append(
        f"- `scu_eligible` AND `trustScore >= 60`: "
        f"**{audit.get('scu_eligible_with_trust_ge_60', 0)}**\n"
    )
    _proto(
        "year = to_numeric(date_time.year)",
        "in_scu_window     = 1945 <= year <= 1975              # SCU master window",
        "post_1975_window  = year >= 1975                      # companion study",
        "outside_window    = NOT in_scu_window",
        "",
        "# --- the eight gate flags (each is a boolean column) ---",
        "has_core_fields            = year & month & day & country_iso all present  # Crit 2",
        "has_investigation_channel  = investigation.source is non-empty             # Crit 4",
        "has_credible_witness       = witness is any of                             # Crit 5",
        "                             {Military, Pilot, Police, Civilian/Public, Scientist}",
        "has_anomalous_characterization =                                           # Crit 3",
        "       anomalous_shape  OR  performance_flag(P/S)  OR  occupant_observed",
        "has_engagement_signal      = any of 9 engagement_type cols in {P, S}       # Phase-3",
        "day_night_resolved         = date_time.day_night in {D, N}",
        "military_public_known      = military_public_resolved is non-empty",
        "reports_within_1_month     = truthy(investigation.reports_within_1_month_of_sighting)",
        "reports_within_1_year      = truthy(...within_1_year...) OR reports_within_1_month",
        "contradicts_uap            = assessment.contradictsUap in {true,1,y,yes}",
        "",
        "# --- Criterion 1 (timeliness) cannot be proven from columns alone, so it",
        "#     is exposed as a tri-state flag and NOT ANDed into scu_eligible. ---",
        "",
        "scu_eligible = in_scu_window AND has_core_fields",
        "             AND has_investigation_channel AND has_credible_witness",
        "             AND has_anomalous_characterization AND has_engagement_signal",
        "             AND day_night_resolved AND military_public_known",
        "",
        "scu_eligible_with_trust_ge_60 = scu_eligible AND (trustScore >= 60)",
        "# each *_count above = that column's .sum() (True == 1).",
    )

    if "suspicious_eligible_count" in audit:
        lines.append("## Manual-review hook")
        lines.append(
            f"- Eligible rows with suspicious narrative "
            f"(`no actual uap`, `sarcastic`, `hoax`, `misidentif`): "
            f"**{audit.get('suspicious_eligible_count', 0)}**"
        )
        rows = audit.get("suspicious_eligible_rows", [])
        if rows:
            lines.append(f"- Row indices (first 20): `{', '.join(rows)}`")
        lines.append("")
        _proto(
            "text = concat(narrative / notes / summary columns, per row)",
            "pattern = /no actual uap|sarcastic|hoax|misidentif|not a uap|not a ufo/i",
            "suspicious = scu_eligible AND text matches pattern",
            "# flags rows that PASS the gate but read like a debunk -> eyeball them.",
        )

    return "\n".join(lines)


def write_audit_markdown(audit: dict, path: str | Path) -> None:
    """Write the audit Markdown report to disk."""
    Path(path).write_text(audit_to_markdown(audit), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point (standalone use)
# ---------------------------------------------------------------------------

def main(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    audit_path: str = DEFAULT_AUDIT,
) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_excel(input_path)
    print(f"  rows={len(df)}  cols={len(df.columns)}")

    print("Normalizing ...")
    out, audit = normalize(df)

    print(f"Writing normalized file -> {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(output_path, index=False)

    print(f"Writing audit report     -> {audit_path}")
    write_audit_markdown(audit, audit_path)

    print("Done.")
    print(f"  SCU-eligible rows:                 {audit['scu_eligible_count']}")
    print(f"  SCU-eligible AND trustScore >= 60: {audit['scu_eligible_with_trust_ge_60']}")


if __name__ == "__main__":
    main()
