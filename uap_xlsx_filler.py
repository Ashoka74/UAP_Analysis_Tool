"""
uap_xlsx_filler.py
==================
Converts UAPParser.parse_responses() output into rows that can be written
back into "UAP Activities Data 1975 Onwards.xlsx".

Typical usage
-------------
    from uap_analyzer import UAPParser
    from uap_xlsx_filler import build_dataframe, append_to_xlsx
    from config import FORMAT_LONG

    parser = UAPParser(api_key=API_KEY)
    parser.process_descriptions(raw_texts, FORMAT_LONG)
    parsed = parser.parse_responses()

    df = build_dataframe(parser, parsed)
    append_to_xlsx(df, "UAP Activities Data 1975 Onwards.xlsx")
"""

import re
import pandas as pd
import numpy as np
from openpyxl import load_workbook

# ---------------------------------------------------------------------------
# Spreadsheet constants
# ---------------------------------------------------------------------------

XLSX_PATH    = "UAP Activities Data 1975 Onwards.xlsx"
SHEET_NAME   = "Data"
HEADER_ROW   = 9    # 1-indexed (openpyxl); this is row index 8 in pandas (0-indexed)
EXAMPLE_ROW  = 10   # 1-indexed; the "Example" row — skip when reading data
SKIPROWS_PD  = 8    # pd.read_excel skiprows to land on the header

# ---------------------------------------------------------------------------
# Mapping: (group_key, json_field) -> exact spreadsheet column name
#
# These must match the strings in HEADER_ROW exactly, including Unicode
# characters and trailing whitespace present in the original file.
# ---------------------------------------------------------------------------

FIELD_MAP: dict[tuple[str, str], str] = {
    # source
    ("source", "name"):                          "Source",
    ("source", "ref"):                           "Source Ref",
    ("source", "duplicate"):                     "Duplicate",
    # date_time
    ("date_time", "year"):                       "Year",
    ("date_time", "month"):                      "Month",
    ("date_time", "day"):                        "Day",
    ("date_time", "day_night"):                  "Day Night",
    ("date_time", "local_time"):                 "Local Time (start) 24h",
    ("date_time", "local_time_code"):            "Local Time Code",
    ("date_time", "duration_min"):               "Duration Min",
    ("date_time", "gmt_start"):                  "GMT (start)",
    ("date_time", "description"):                "Date, Time General Description",
    # location
    ("location", "country"):                     "Country",
    ("location", "state"):                       "State or Area",
    ("location", "city"):                        "City or Closest City",
    ("location", "latitude"):                    "Latitude",
    ("location", "longitude"):                   "Longitude",
    ("location", "description"):                 "Location Description",
    ("location", "type"):                        "Location Type",
    # witness
    ("witness", "count"):                        "Number Of Witnes",
    ("witness", "type"):                         "Type",
    ("witness", "description"):                  "Witness Description",
    # investigation
    ("investigation", "source"):                 "Investigated Source",
    ("investigation", "description"):            "Investigated Description",
    # craft
    ("craft", "primary_shape"):                  "Primary Shape",
    ("craft", "secondary_shape"):                "Secondary Shape",
    ("craft", "colour"):                         "Colour",
    ("craft", "size"):                           "Size",
    ("craft", "description"):                    "Shape Description",
    # performance / 5 observables
    ("performance", "speed_mph"):                "Speed miles per hour",
    ("performance", "acceleration_g"):           "Acceleration g",
    ("performance", "hypersonic"):               "Hypersonic Velocities Without Signatures",
    ("performance", "instantaneous_acceleration"): "Instantaneous Acceleration",
    ("performance", "low_observability"):        "Low Observability (Cloaking / Stealth)",
    ("performance", "trans_medium_travel"):      "Trans‑Medium Travel",   # ‑ = ‑
    ("performance", "positive_lift"):            "Positive Lift Without Aerodynamic Surfaces",
    # military
    ("military", "reported_by"):                 "Reported By",
    ("military", "military_public"):             "Military / Public",
    ("military", "facility_name"):               "Primary Facility Name",
    ("military", "facility_type"):               "Facility TYPE",
    ("military", "comments"):                    "Comments From Source",
    # effects
    ("effects", "atomic_related"):               "Atomic Related",
    ("effects", "communication"):                "Communication",
    ("effects", "physical_effects"):             "Physical Effects",
    ("effects", "text"):                         "Com / Effects Text",
    # engagement_flags — Y/N columns (AU-BE, cols 46-56)
    ("engagement_flags", "aircraft_engagement"):        "Aircraft Engagement",
    ("engagement_flags", "aircraft_encounters"):        "Aircraft Encounters",
    ("engagement_flags", "active_radar_jamming"):       "Active Radar jamming",
    ("engagement_flags", "over_military_installation"): "Over Military Installation",
    ("engagement_flags", "during_missile_test"):        "During missile, rocket and high-altitude balloon tests",
    ("engagement_flags", "radar_tracking"):             "Radar Tracking",
    ("engagement_flags", "radio_interference"):         "Radio interference in the form of noise on audio receivers",
    ("engagement_flags", "radar_jamming"):              "Radar Interference / jamming of receivers-displays",
    ("engagement_flags", "directed_radar"):             "Directed radar frequence transmissions – mimicking the frequencies used",
    ("engagement_flags", "coded_radar"):                "Coded radar frequence transmissions:\xa0 \xa0Identification Friend or Foe",
    ("engagement_flags", "multiple_interactive_flight"): "Multiple interactive flight  \xa0",
    # engagement_type — P/S/blank columns (BF-BO, cols 57-66)
    # Constraint: exactly one field = P; others S or blank
    ("engagement_type", "interactive_flight"):          "Interactive Flight",
    ("engagement_type", "radical_flight"):              "Radical Flight",
    ("engagement_type", "loitering"):                   "Loitering",
    ("engagement_type", "electronic_transmissions"):    "Electronic transmissions ",
    ("engagement_type", "interference_weapons"):        "Interference weapons systems ",
    ("engagement_type", "military_intrusions"):         "Intrusions at military installations ",
    ("engagement_type", "occupant_encounter"):          "Occupant encounter",
    ("engagement_type", "occupant_observed"):           "Occupant observed",
    ("engagement_type", "close_approach"):              "Close Approach",
    ("engagement_type", "no_engagement"):               "No Engagement Type",
    # case text
    ("case_text", "text"):                       "Case Text Dump",
}

# ---------------------------------------------------------------------------
# Dtype enforcement
# ---------------------------------------------------------------------------

_INT_COLS = {"Year", "Month", "Day", "Duration Min", "Number Of Witnes"}
_FLOAT_COLS = {"Latitude", "Longitude", "Speed miles per hour", "Acceleration g"}

# Columns that must contain only Y / N / ""
_YN_COLS = {
    "Atomic Related", "Communication", "Physical Effects",
    "Aircraft Engagement", "Aircraft Encounters", "Active Radar jamming",
    "Over Military Installation",
    "During missile, rocket and high-altitude balloon tests",
    "Radar Tracking",
    "Radio interference in the form of noise on audio receivers",
    "Radar Interference / jamming of receivers-displays",
    "Directed radar frequence transmissions – mimicking the frequencies used",
    "Coded radar frequence transmissions:\xa0 \xa0Identification Friend or Foe",
    "Multiple interactive flight  \xa0",
    # 5 observables
    "Hypersonic Velocities Without Signatures",
    "Instantaneous Acceleration",
    "Low Observability (Cloaking / Stealth)",
    "Trans‑Medium Travel",
    "Positive Lift Without Aerodynamic Surfaces",
}

# Columns that must contain only P / S / ""
_PSB_COLS = {
    "Interactive Flight", "Radical Flight", "Loitering",
    "Electronic transmissions ", "Interference weapons systems ",
    "Intrusions at military installations ",
    "Occupant encounter", "Occupant observed", "Close Approach",
    "No Engagement Type",
}

_TRUTHY = re.compile(r"^(y|yes|true|1)$", re.I)
_FALSY  = re.compile(r"^(n|no|false|0)$", re.I)

_DAY_RE   = re.compile(r"^(d|day|daytime|dawn|dusk|morning|afternoon|evening)$", re.I)
_NIGHT_RE = re.compile(r"^(n|night|nighttime|after\s*dark)$", re.I)
_UNK_RE   = re.compile(r"^(u|unk|unknown|\?)$", re.I)

_PRIMARY_RE   = re.compile(r"^(p|primary)$", re.I)
_SECONDARY_RE = re.compile(r"^(s|secondary)$", re.I)


def _norm_yn(val) -> str:
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    if _TRUTHY.match(s):
        return "Y"
    if _FALSY.match(s):
        return "N"
    return s if s in ("Y", "N") else ""


def _norm_day_night(val) -> str:
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    if s in ("D", "N", "U"):
        return s
    if _DAY_RE.match(s):
        return "D"
    if _NIGHT_RE.match(s):
        return "N"
    if _UNK_RE.match(s):
        return "U"
    return ""


def _norm_psb(val) -> str:
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    if _PRIMARY_RE.match(s):
        return "P"
    if _SECONDARY_RE.match(s):
        return "S"
    return s if s in ("P", "S") else ""


def _norm_facility(val, valid: list[str]) -> str:
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    if s in valid:
        return s
    sl = s.lower()
    for v in valid:
        if v.lower() == sl:
            return v
    return s  # keep original if no match


def _norm_time(val) -> str:
    """Normalise time values to HH:MM strings."""
    if pd.isna(val) or val == "":
        return ""
    s = str(val).strip()
    m = re.match(r"^(\d{1,2})[:\.](\d{2})$", s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return s


def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes and enum values on all known columns in-place (returns copy)."""
    from config import FACILITY_TYPES

    df = df.copy()

    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "Day Night" in df.columns:
        df["Day Night"] = df["Day Night"].apply(_norm_day_night).replace("", np.nan)

    for col in _YN_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_norm_yn).replace("", np.nan)

    for col in _PSB_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_norm_psb).replace("", np.nan)

    if "Facility TYPE" in df.columns:
        df["Facility TYPE"] = df["Facility TYPE"].apply(
            lambda v: _norm_facility(v, FACILITY_TYPES)
        ).replace("", np.nan)

    for col in ("Local Time (start) 24h", "GMT (start)"):
        if col in df.columns:
            df[col] = df[col].apply(_norm_time).replace("", np.nan)

    return df


# Groups to process in order (must match FORMAT_LONG top-level keys)
_GROUPS = [
    "source", "date_time", "location", "witness", "investigation",
    "craft", "performance", "military", "effects",
    "engagement_flags", "engagement_type", "case_text",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_existing(xlsx_path: str = XLSX_PATH) -> pd.DataFrame:
    """
    Load the existing data rows from the spreadsheet.

    Skips the 8 metadata/header rows and the example row (row 10).
    Returns an empty DataFrame (just the columns) if no real data exists yet.
    """
    # skip rows 0-7 (metadata) and row 9 (example row, 0-indexed)
    skip = list(range(SKIPROWS_PD)) + [SKIPROWS_PD + 1]
    df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME, skiprows=skip, header=0)
    df = df.dropna(how="all")
    return df


def build_dataframe(parser, parsed_responses: dict) -> pd.DataFrame:
    """
    Convert the output of UAPParser.parse_responses() into a DataFrame whose
    columns match the spreadsheet exactly.

    Parameters
    ----------
    parser :
        A UAPParser instance (needs the responses_to_df method).
    parsed_responses :
        Dict returned by parser.parse_responses():
        { original_text: { group_key: { field: value, ... }, ... }, ... }

    Returns
    -------
    pd.DataFrame  with column names matching the spreadsheet header row.
    """
    parts: list[pd.DataFrame] = []

    for group in _GROUPS:
        try:
            group_df = parser.responses_to_df(group, parsed_responses)
        except (KeyError, ValueError):
            continue

        # Rename json field names → exact spreadsheet column names
        rename = {}
        for json_field in group_df.columns:
            xlsx_col = FIELD_MAP.get((group, json_field))
            if xlsx_col:
                rename[json_field] = xlsx_col
        group_df = group_df.rename(columns=rename)

        # Drop any columns we don't have a mapping for
        known = set(FIELD_MAP.values())
        group_df = group_df[[c for c in group_df.columns if c in known]]

        parts.append(group_df)

    if not parts:
        raise ValueError("No groups could be extracted from parsed_responses.")

    full_df = pd.concat(parts, axis=1)

    # Resolve duplicate column names (can happen if the LLM returns extra keys)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]

    # Replace empty strings and "None" strings with NaN for clean writing
    full_df = full_df.replace({"": np.nan, "None": np.nan, None: np.nan}).infer_objects(copy=False)

    # Enforce dtypes and enum values
    full_df = _coerce_df(full_df)

    return full_df


def build_dataframe_from_csv(csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a flat CSV DataFrame (columns in ``group.field`` format as
    produced by ``responses_to_df`` with no col argument) into a DataFrame
    with exact spreadsheet column names, ready for ``append_to_xlsx``.

    Only works for CSVs exported in the SCU multi-section format where
    column names follow the ``group.field`` convention (e.g. ``source.name``,
    ``date_time.year``).  Default UAP format CSVs (single-section) are not
    supported here — use ``build_dataframe`` with the JSON instead.
    """
    # Reverse map: "group.field" -> xlsx column name
    reverse_map = {
        f"{group}.{field}": xlsx_col
        for (group, field), xlsx_col in FIELD_MAP.items()
    }
    renamed = csv_df.rename(columns=reverse_map)
    known_cols = set(FIELD_MAP.values())
    renamed = renamed[[c for c in renamed.columns if c in known_cols]]

    if renamed.empty or renamed.shape[1] == 0:
        raise ValueError(
            "No recognisable columns found. Make sure the CSV was exported in "
            "SCU format (columns like 'source.name', 'date_time.year', …). "
            "For other formats, upload the parsed_responses.json instead."
        )

    renamed = renamed.replace({"": np.nan, "None": np.nan, None: np.nan}).infer_objects(copy=False)
    return _coerce_df(renamed)


def append_to_xlsx(
    new_df: pd.DataFrame,
    xlsx_path: str = XLSX_PATH,
    output_path: str | None = None,
    start_internal_number: int | None = None,
) -> str:
    """
    Append new rows to the spreadsheet, preserving all existing formatting,
    dropdowns, and merged cells.

    Writes into the 'Data' sheet starting at the first empty row after the
    existing data (always at least row 11 = first row after the example).

    Parameters
    ----------
    new_df :
        DataFrame returned by build_dataframe().
    xlsx_path :
        Path to the source spreadsheet.
    output_path :
        Where to save the result. Defaults to xlsx_path (in-place).
    start_internal_number :
        If given, auto-number the Internal Number column starting here.
        If None, leaves Internal Number blank.

    Returns
    -------
    str  path of the written file.
    """
    output_path = output_path or xlsx_path

    wb = load_workbook(xlsx_path)
    ws = wb[SHEET_NAME]

    # Build col_name → column_index mapping from the header row
    col_index: dict[str, int] = {}
    for cell in ws[HEADER_ROW]:
        if cell.value is not None:
            col_index[cell.value] = cell.column

    # First row we can write to: max(EXAMPLE_ROW + 1, ws.max_row + 1)
    first_data_row = max(EXAMPLE_ROW + 1, ws.max_row + 1)

    for offset, (_, row_data) in enumerate(new_df.iterrows()):
        row_num = first_data_row + offset

        if start_internal_number is not None:
            internal_col = col_index.get("Internal Number")
            if internal_col:
                ws.cell(row=row_num, column=internal_col).value = (
                    start_internal_number + offset
                )

        for col_name, value in row_data.items():
            if pd.isna(value):
                continue
            xlsx_col = col_index.get(col_name)
            if xlsx_col is None:
                continue
            # Convert pandas nullable types to plain Python for openpyxl
            if hasattr(value, "item"):
                value = value.item()
            ws.cell(row=row_num, column=xlsx_col).value = value

    wb.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Reading helper (for verification / downstream analysis)
# ---------------------------------------------------------------------------

def read_data(xlsx_path: str = XLSX_PATH) -> pd.DataFrame:
    """
    Read all real data rows (skip metadata + example), return as DataFrame.
    Equivalent to load_existing() but forces dtype cleanup.
    """
    df = load_existing(xlsx_path)
    # Coerce numeric columns
    for col in ["Year", "Month", "Day", "Duration Min",
                "Latitude", "Longitude", "Speed miles per hour", "Acceleration g"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
