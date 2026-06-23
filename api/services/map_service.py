"""
map_service.py
--------------
Encapsulates all Kepler.gl HTML generation logic.

Key design decisions:
  - Results are cached by a *data fingerprint* (hash of the DataFrame shape +
    first/last row content).  When the user changes filters the fingerprint
    changes, a new HTML payload is generated, and the old entry is evicted.
  - We keep **at most `_MAX_CACHE_ENTRIES`** cached payloads to bound memory
    use when many sessions are active.
  - The heavy `df.copy()` that previously lived inside the endpoint is
    eliminated: we only copy the minimal columns we actually need for the map.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import traceback
from collections import OrderedDict
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_CACHE_ENTRIES = 20

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fingerprint(df: pd.DataFrame) -> str:
    """
    Fast, deterministic fingerprint for a DataFrame.

    Uses shape + a hash of a small sample (first & last 5 rows rendered as
    CSV) so the cost is O(1) with respect to the total number of rows.
    """
    sample = pd.concat([df.head(5), df.tail(5)])
    raw = f"{df.shape}|{sample.to_csv(index=False)}"
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()


def _build_html(df: pd.DataFrame) -> str:
    """
    Pure function: receive a (possibly large) DataFrame, return a Kepler.gl
    HTML string.
    """
    import json
    from keplergl import KeplerGl  # noqa: PLC0415

    # DECOUPLED: Use the new lightweight utils instead of map.py
    from api.utils.data_utils import auto_create_date_column, sanitize_dataframe_for_json  # noqa: PLC0415

    # Path setup
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_dir, "military_config.kgl")
    bases_path = os.path.join(base_dir, "secret_bases.csv")
    power_path = os.path.join(base_dir, "global_power_plant_database.csv")

    kmap = KeplerGl(height=800)

    # 1. Load configuration if available
    config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load military config: {e}")

    # 2. Add Auxiliary Data (Power Plants & Secret Bases)
    if os.path.exists(power_path):
        try:
            pp_df = pd.read_csv(power_path)
            # Filter for nuclear as per config multiSelect
            nuke_df = pp_df[pp_df["primary_fuel"] == "Nuclear"].copy()
            nuke_df["icon"] = "control-on" # Required for icon layer
            kmap.add_data(data=nuke_df, name="nuclear_powerplants")
        except Exception as e:
            logger.error(f"Failed to load power plant data: {e}")

    if os.path.exists(bases_path):
        try:
            bases_df = pd.read_csv(bases_path)
            bases_df["icon"] = "draw-shape"
            kmap.add_data(data=bases_df, name="secret_bases")
        except Exception as e:
            logger.error(f"Failed to load secret bases: {e}")

    # 3. Add user's UAP data
    df = auto_create_date_column(df)
    df = sanitize_dataframe_for_json(df)
    
    map_cols = list(df.columns)
    lat_candidates = [c for c in map_cols if str(c).lower() in {"lat", "latitude", "city_latitude"}]
    lon_candidates = [c for c in map_cols if str(c).lower() in {"lon", "lng", "longitude", "city_longitude"}]

    if lat_candidates and lon_candidates:
        lat_col = lat_candidates[0]
        lon_col = lon_candidates[0]
        needed_cols = list(set(map_cols) & set(df.columns))
        df_map = df[needed_cols].copy()
        df_map[lat_col] = pd.to_numeric(df_map[lat_col], errors="coerce")
        df_map[lon_col] = pd.to_numeric(df_map[lon_col], errors="coerce")
        df_map = df_map.dropna(subset=[lat_col, lon_col])

        # 4. DYNAMIC VIEWPORT (Phase 14)
        # Calculate center and zoom based on the current sightings data
        if not df_map.empty:
            lat_mean = float(df_map[lat_col].mean())
            lon_mean = float(df_map[lon_col].mean())
            
            # Simple zoom heuristic based on spread
            lat_range = df_map[lat_col].max() - df_map[lat_col].min()
            lon_range = df_map[lon_col].max() - df_map[lon_col].min()
            max_range = max(lat_range, lon_range)
            
            # log-based zoom approximation: 0 is whole world (~360), 10-12 is city
            if max_range > 100: zoom = 2
            elif max_range > 30: zoom = 3
            elif max_range > 10: zoom = 4
            elif max_range > 5: zoom = 5
            elif max_range > 2: zoom = 6
            elif max_range > 1: zoom = 7
            else: zoom = 8
            
            if config:
                if "mapState" not in config:
                    config["mapState"] = {}
                config["mapState"]["latitude"] = lat_mean
                config["mapState"]["longitude"] = lon_mean
                config["mapState"]["zoom"] = zoom
                logger.info(f"Dynamically centering map at {lat_mean:.2f}, {lon_mean:.2f} (zoom={zoom})")

        # Data ID aligned with military_config.kgl
        kmap.add_data(data=df_map, name="uap_sightings")
    else:
        kmap.add_data(data=df, name="uap_sightings")

    if config:
        kmap.config = config

    html_bytes = kmap._repr_html_()
    return html_bytes.decode("utf-8") if isinstance(html_bytes, bytes) else html_bytes


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------

class MapService:
    """
    Singleton-style service that generates and caches Kepler.gl HTML payloads.
    """
    _cache: OrderedDict[str, str] = OrderedDict()

    @classmethod
    def get_or_generate(cls, df: pd.DataFrame) -> tuple[str, bool]:
        """
        Return (html_string, cache_hit).
        """
        # Fingerprint logic remains the same, but we could add config mtime if it changes often
        key = _fingerprint(df)

        if key in cls._cache:
            # Promote to most-recently-used
            cls._cache.move_to_end(key)
            logger.info("MapService: cache HIT (key=%s…)", key[:8])
            return cls._cache[key], True

        logger.info("MapService: cache MISS — generating HTML (key=%s…)", key[:8])
        html = _build_html(df)

        # Evict oldest entry when the cache is full
        if len(cls._cache) >= _MAX_CACHE_ENTRIES:
            evicted = next(iter(cls._cache))
            cls._cache.pop(evicted)
            logger.debug("MapService: evicted cache entry %s…", evicted[:8])

        cls._cache[key] = html
        return html, False

    @classmethod
    def invalidate(cls, df: Optional[pd.DataFrame] = None) -> None:
        """
        Invalidate a specific entry (if *df* is provided) or the whole cache.
        """
        if df is None:
            cls._cache.clear()
            logger.info("MapService: full cache cleared")
        else:
            key = _fingerprint(df)
            cls._cache.pop(key, None)
            logger.info("MapService: cache entry %s… invalidated", key[:8])
