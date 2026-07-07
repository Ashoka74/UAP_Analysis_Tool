"""UAPParser: fail-fast on dead accounts, schema completion/pruning, batch splitting.

Each test reproduces a shipped failure:
- an unfunded account (insufficient_quota == HTTP 429) retried 10x per row
  across 10 workers — hundreds of doomed API calls;
- gpt-4o-mini dropped schema keys entirely, silently deleting columns;
- one record nested eleven top-level blocks inside `military` (146 phantom cols);
- a 9.5k-report batch blew the 200 MiB input-file cap.
"""
import json
import threading

import pytest

from conftest import make_parser, mock_batch_client


class _Resp:
    class _Choice:
        class _Msg:
            content = '{"a": "x"}'
        message = _Msg()
    choices = [_Choice()]


def _client_with(create_fn):
    return type("C", (), {"chat": type("Ch", (), {
        "completions": type("Co", (), {"create": staticmethod(create_fn)})()
    })()})()


# ── Fail-fast / circuit breaker ─────────────────────────────────────────────

def test_insufficient_quota_aborts_run_without_storm(no_sleep):
    calls = {"n": 0, "lock": threading.Lock()}

    def quota_create(**kw):
        with calls["lock"]:
            calls["n"] += 1
        raise Exception(
            "Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
            "'type': 'insufficient_quota', 'code': 'insufficient_quota'}}"
        )

    p = make_parser()
    p.client = _client_with(quota_create)
    p.process_descriptions([f"r{i}" for i in range(50)], '{"a": ""}', max_workers=5)

    # Before the circuit breaker: up to 50 rows x 10 retries = 500 calls.
    assert calls["n"] <= 15, f"rate-limit storm not contained ({calls['n']} calls)"
    assert p.responses == {}
    assert p.last_errors and p.last_errors[0].startswith("FATAL:")


def test_transient_rate_limit_still_retries(no_sleep):
    state = {"n": 0}

    def flaky_create(**kw):
        state["n"] += 1
        if state["n"] <= 2:
            raise Exception("Error code: 429 - Rate limit reached, code: rate_limit_exceeded")
        return _Resp()

    p = make_parser()
    p.client = _client_with(flaky_create)
    p.process_descriptions(["only-one"], '{"a": ""}', max_workers=1)
    assert state["n"] == 3           # two 429s, then success
    assert len(p.responses) == 1


# ── Schema completion + strict prune ────────────────────────────────────────

def test_schema_complete_pads_dropped_keys(master_schema_text):
    p = make_parser()
    p.responses = {"r": json.dumps(
        {"engagement": {"engagement_type": {"occupant_observed": "P"}}}
    )}
    rec = p.parse_responses(format_long=master_schema_text)["r"]
    et = rec["engagement"]["engagement_type"]
    assert "electronic_transmissions" in et and et["electronic_transmissions"] == ""
    assert "interference_weapons" in et
    assert "contact" in rec           # whole dropped block padded back, blank


def test_schema_prune_rescues_misplaced_blocks(master_schema):
    from uap_analyzer import schema_prune

    bad = {
        "military": {"facility_name": "PMRF",
                     "investigation": {"timeliness": "prompt"},
                     "narrative": {"rawText": "the real narrative"}},
        "classification": {"HYNEK": "DD", "assessment": {"trustScore": 77},
                           "contradictsUap": "Y", "FLYBY": "FB1"},
        "assessment": {"explanationCategory": "drone"},
    }
    rec, info = schema_prune(bad, master_schema)
    assert rec["investigation"]["timeliness"] == "prompt"          # suffix rescue
    assert rec["narrative"]["rawText"] == "the real narrative"
    assert rec["assessment"]["trustScore"] == 77                   # unique-leaf rescue
    assert rec["assessment"]["contradictsUap"] == "Y"
    assert rec["assessment"]["explanationCategory"] == "drone"     # never overwritten
    assert rec["military"]["facility_name"] == "PMRF"              # legit field kept
    assert "FLYBY" not in rec.get("classification", {})            # invention dropped
    assert "classification.FLYBY" in info["dropped"]


def test_parse_responses_yields_no_offschema_columns(master_schema_text, master_schema):
    import pandas as pd
    from uap_analyzer import _template_leaf_paths

    p = make_parser()
    p.responses = {
        "r1": json.dumps({"military": {"investigation": {"timeliness": "prompt"}}}),
        "r2": json.dumps({"location": {"country": "FR"}}),
    }
    parsed = p.parse_responses(format_long=master_schema_text)
    df = pd.json_normalize(list(parsed.values()))
    leaves = set(_template_leaf_paths(master_schema))
    assert [c for c in df.columns if c not in leaves] == []


# ── Batch API: byte-aware splitting + .jsonl output import ─────────────────

def test_batch_split_respects_byte_cap_and_global_ids():
    files, batches = [], []
    p = make_parser()
    p.client = mock_batch_client(files, batches)
    p._BATCH_MAX_BYTES = 20_000   # ~1.3 KB/line x 30 -> must split into >= 2 files

    descs = [f"report number {i} " + "x" * 200 for i in range(30)]
    ids = p._submit_batch_openai(descs, json.dumps({"a": ""}))

    assert len(ids) >= 2                                    # forced split
    assert all(len(data) <= 20_000 for _n, data in files)   # cap respected
    lines = [ln for _n, data in files for ln in data.decode().splitlines()]
    assert len(lines) == 30                                 # nothing lost
    cids = [json.loads(ln)["custom_id"] for ln in lines]
    assert cids == [str(i) for i in range(30)]              # GLOBAL, ordered


def test_batch_jsonl_roundtrip_rekeys_and_prunes(master_schema_text):
    from uap_analyzer import batch_jsonl_to_parsed

    def line(cid, content):
        return json.dumps({"custom_id": str(cid), "response": {"body": {"choices": [
            {"message": {"content": json.dumps(content)}}]}}, "error": None})

    jsonl = "\n".join([
        line(0, {"location": {"country": "USA"}}),
        line(1, {"military": {"investigation": {"timeliness": "prompt"}}}),
        json.dumps({"custom_id": "2", "response": None,
                    "error": {"message": "rate limited"}}),
    ])
    parsed, errs = batch_jsonl_to_parsed(
        jsonl, format_long=master_schema_text, descriptions=["A", "B", "C"])
    assert set(parsed) == {"A", "B"}                        # re-keyed to inputs
    assert parsed["B"]["investigation"]["timeliness"] == "prompt"   # pruned+rescued
    assert len(errs) == 1 and "rate limited" in errs[0]


def test_extract_json_no_quadratic_blowup_on_degenerate_output():
    """A truncated/degenerate 600 KB model response (huge whitespace runs, broken
    JSON) froze batch imports for minutes via catastrophic regex backtracking in
    the fence-stripping fallback. It must now fail fast."""
    import time
    from uap_analyzer import _extract_json

    junk = '{"a": "b",' + ('   \n' * 150_000) + '"c": '   # broken, whitespace-heavy
    t0 = time.time()
    assert _extract_json(junk) is None
    assert time.time() - t0 < 2.0, "fallback chain is superlinear again"


def test_deepseek_insufficient_balance_is_fatal(no_sleep):
    """DeepSeek reports a drained account as HTTP 402 'Insufficient Balance' —
    not a 429 — so it must trip the same circuit breaker as insufficient_quota,
    while concurrency-limit 429s stay transient (they clear as requests drain)."""
    calls = {"n": 0, "lock": threading.Lock()}

    def balance_create(**kw):
        with calls["lock"]:
            calls["n"] += 1
        raise Exception(
            "Error code: 402 - {'error': {'message': 'Insufficient Balance', "
            "'type': 'unknown_error', 'code': 'invalid_request_error'}}"
        )

    p = make_parser()
    p.client = _client_with(balance_create)
    p.process_descriptions([f"r{i}" for i in range(50)], '{"a": ""}', max_workers=5)
    assert calls["n"] <= 15, f"drained DeepSeek account stormed ({calls['n']} calls)"
    assert p.last_errors and p.last_errors[0].startswith("FATAL:")
