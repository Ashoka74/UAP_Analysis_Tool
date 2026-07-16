"""Offline US city/state -> lat/lon centroid lookup.

Built from the US Census Bureau's Gazetteer Places file (2024,
geo_data/us_places_gazetteer.csv) — free, public domain, no API key, no
rate limit, no network call at lookup time. ~32k incorporated places +
census-designated places, loaded once into an in-memory dict; lookups are
then a single dict hit, so this scales to hundreds of thousands of rows
with no meaningful latency (contrast with live geocoding APIs like
Nominatim, which are rate-limited to ~1 req/sec and unusable at this scale).

Trade-off: this gives a place *centroid* (e.g. "downtown Phoenix"), not the
sighting's actual coordinates — coarser than real geocoding, but a real
haversine distance in km, which is strictly more informative than a bare
text-similarity score when no lat/lon exists at all.
"""
import os
import re
from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd

_GAZETTEER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "geo_data", "us_places_gazetteer.csv")

_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "puerto rico": "PR",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
_VALID_ABBRS = set(_STATE_NAME_TO_ABBR.values())

_PLACE_SUFFIX_RE = re.compile(r"\s+(CDP|city|town|village|borough|township)$", re.IGNORECASE)


def normalize_state(s: Optional[str]) -> Optional[str]:
    """'Arizona' / 'AZ' / ' az ' -> 'AZ'. None if not a recognizable US state."""
    s = str(s or "").strip()
    if not s:
        return None
    if len(s) == 2 and s.isalpha() and s.upper() in _VALID_ABBRS:
        return s.upper()
    return _STATE_NAME_TO_ABBR.get(s.lower())


def normalize_place(name: Optional[str]) -> str:
    """Lowercase + strip the common incorporation-type suffix Census place
    names carry ('Phoenix city' -> 'phoenix'), so lookups match whether or
    not the caller's data includes that suffix."""
    name = str(name or "").strip()
    name = _PLACE_SUFFIX_RE.sub("", name)
    return name.lower()


def parse_city_state(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """Best-effort split of a combined 'City, ST' / 'City, State' /
    'City, State, USA' free-text location string into (city, state).
    Returns None if there's no comma-separated state component."""
    text = str(text or "").strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


@lru_cache(maxsize=1)
def _load_index() -> dict:
    df = pd.read_csv(_GAZETTEER_PATH)
    index: dict = {}
    for state, place, lat, lon in zip(df["state"], df["place"], df["lat"], df["lon"]):
        key = (str(state).upper(), str(place).lower())
        index.setdefault(key, (float(lat), float(lon)))
    return index


def geocode_us_place(place: Optional[str], state: Optional[str]) -> Optional[Tuple[float, float]]:
    """Approximate (lat, lon) centroid for a US city/place name within a
    state, from the offline Gazetteer index. None if unresolvable (unknown
    place, missing/unrecognized state, blank input)."""
    norm_state = normalize_state(state)
    if not norm_state:
        return None
    norm_place = normalize_place(place)
    if not norm_place:
        return None
    return _load_index().get((norm_state, norm_place))
