"""SCU normalizer: Cramér's V guards, input-column auto-mapping, the gate.

Shipped failure: MasterSCU_v1 re-parents fields (object.*, engagement.
engagement_type.*, behavior.performance.*), and the exact-match-only reader
silently fed NaN to 3 of the 5 SCU criteria.
"""
import json

import pandas as pd
import pytest

import scu_normalizer as S


MASTER_RECORD = {
    "date_time": {"year": 1960, "month": 5, "day": 3, "day_night": "N"},
    "location": {"country": "USA", "state": "Texas", "type": "military base"},
    "object": {"primary_shape": "disc", "size": "car"},
    "behavior": {"performance": {"hypersonic": "P"}},
    "engagement": {"engagement_type": {"occupant_observed": "P"},
                   "engagement_flags": {"radar_tracking": "P"}},
    "effects": {"atomic_related": "P"},
    "assessment": {"contradictsUap": "", "trustScore": 80},
    "witness": {"roles": ["pilot"], "count": 2, "type": "military"},
    "source": {"name": "Blue Book"},
    "investigation": {"timeliness": "prompt"},
}


def test_cramers_v_degenerate_table_returns_zero():
    from uap_analyzer import cramers_v
    # single row/column -> denominator would be <= 0 (division-by-zero class)
    ct = pd.crosstab(pd.Series(["a", "a", "a"]), pd.Series(["x", "y", "x"]))
    assert cramers_v(ct) == 0.0


def test_mapping_resolves_master_paths(master_schema):
    def leaves(o, pre=""):
        out = []
        if isinstance(o, dict):
            for k, v in o.items():
                out += leaves(v, f"{pre}.{k}" if pre else k)
        elif isinstance(o, list):
            out += leaves(o[0], pre) if (o and isinstance(o[0], dict)) else [pre]
        else:
            out.append(pre)
        return out

    res = S.resolve_column_mapping(leaves(master_schema))
    m, how = res["mapping"], res["methods"]
    assert m["craft.primary_shape"] == "object.primary_shape"
    assert m["craft.size"] == "object.size" and how["craft.size"] == "alias"
    assert m["investigation.source"] == "source.name" and how["investigation.source"] == "alias"
    assert m["engagement_type.occupant_observed"] == "engagement.engagement_type.occupant_observed"
    assert m["performance.hypersonic"] == "behavior.performance.hypersonic"
    # every gate input resolves deterministically against the full master schema
    bad = [c for c in S.EXPECTED_INPUT_COLUMNS
           if how.get(c, "unmatched") not in ("exact", "suffix", "alias", "leaf")]
    assert bad == []


def test_exact_column_beats_alias():
    df = pd.DataFrame({"investigation.source": ["MUFON"], "source.name": ["press"]})
    res = S.resolve_column_mapping(list(df.columns))
    assert res["mapping"]["investigation.source"] == "investigation.source"
    assert res["methods"]["investigation.source"] == "exact"


def test_gate_fires_on_master_shaped_record():
    out, audit = S.normalize(pd.json_normalize([MASTER_RECORD]))
    row = out.iloc[0]
    for crit in ("in_scu_window", "has_core_fields", "has_investigation_channel",
                 "has_credible_witness", "has_anomalous_characterization",
                 "has_engagement_signal"):
        assert bool(row[crit]), f"{crit} should fire on a fully-populated record"
    assert row["craft_primary_shape_norm"] == "Disc"
    assert audit["column_mapping_methods"]["craft.primary_shape"] == "alias"


def test_manual_column_map_override():
    df = pd.json_normalize([MASTER_RECORD])
    out, audit = S.normalize(df, column_map={"craft.size": "object.primary_shape"})
    assert audit["column_mapping"]["craft.size"] == "object.primary_shape"
    assert audit["column_mapping_methods"]["craft.size"] == "manual"


# ── Mini-SCU "tunnel" schema — the cheap first-pass funnel ──────────────────

def _leaves(d, pfx=""):
    out = set()
    for k, v in d.items():
        p = f"{pfx}{k}"
        if isinstance(v, dict):
            out |= _leaves(v, p + ".")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out |= _leaves(v[0], p + ".")
        else:
            out.add(p)
    return out


def test_mini_scu_is_strict_subset_of_scu_v3():
    """Mini-SCU is derived from SCU_v3, so every leaf (and its verbatim
    definition) must exist in SCU_v3 — no invented fields, and a real shrink."""
    from config import FORMAT_MINI_SCU, FORMAT_SCU_V3
    mini, v3 = _leaves(FORMAT_MINI_SCU), _leaves(FORMAT_SCU_V3)
    assert mini < v3, f"MINI leaves outside SCU_v3: {sorted(mini - v3)}"
    assert len(mini) < len(v3) / 3            # a genuine funnel, not a rename
    # definitions inherited byte-for-byte (extraction quality must not drift)
    assert (FORMAT_MINI_SCU["engagement_type"]["occupant_observed"]
            == FORMAT_SCU_V3["engagement_type"]["occupant_observed"])


def test_mini_scu_covers_every_gate_input():
    """The whole point: a record carrying ONLY Mini-SCU fields, at SCU_v3's
    native flat paths, must clear the five-criterion gate with no auto-mapping
    (Mini uses native paths, so nothing needs renaming)."""
    rec = {
        "date_time": {"year": 1965, "month": 7, "day": 3, "day_night": "N"},
        "location": {"country": "US"},
        "investigation": {"source": "NICAP",
                          "reports_within_1_month_of_sighting": "Y",
                          "reports_within_1_year_of_sighting": "Y"},
        "witness": {"roles": ["Pilot", "Military"]},
        "craft": {"primary_shape": "Disc"},
        "performance": {"hypersonic": "Y", "instantaneous_acceleration": "U",
                        "low_observability": "U", "trans_medium_travel": "U",
                        "positive_lift": "Y"},
        "engagement_type": {"interactive_flight": "P", "radical_flight": "S",
                            "loitering": "", "electronic_transmissions": "",
                            "interference_weapons": "", "military_intrusions": "",
                            "occupant_encounter": "", "occupant_observed": "",
                            "close_approach": "S", "no_engagement": ""},
        "military": {"military_public": "Military"},
        "assessment": {"contradictsUap": False, "notes": ""},
    }
    out, _ = S.normalize(pd.json_normalize([rec]))
    assert bool(out["scu_eligible"].iloc[0]) is True
    for c in ("in_scu_window", "has_core_fields", "has_investigation_channel",
              "has_credible_witness", "has_anomalous_characterization",
              "has_engagement_signal", "day_night_resolved", "military_public_known"):
        assert bool(out[c].iloc[0]), f"{c} not satisfied by Mini-SCU fields"

    # negative control: strip the anomaly + engagement signal -> gate must reject
    rec2 = json.loads(json.dumps(rec))
    rec2["craft"]["primary_shape"] = "Unknown"
    rec2["performance"] = {k: "U" for k in rec["performance"]}
    rec2["engagement_type"] = {k: "" for k in rec["engagement_type"]}
    out2, _ = S.normalize(pd.json_normalize([rec2]))
    assert bool(out2["scu_eligible"].iloc[0]) is False
