"""End-to-end API flows over the FastAPI TestClient (per-test session isolation)."""
import io
import json

import numpy as np
import pandas as pd
import pytest


def _upload_csv(client, df, endpoint="/api/data/upload"):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return client.post(endpoint, files={"file": ("d.csv", buf, "text/csv")})


@pytest.fixture(scope="module")
def demo_df():
    rng = np.random.RandomState(3)
    n = 250
    z = rng.randint(0, 2, n)

    def noisy(x, p):
        flip = rng.rand(n) < p
        return np.where(flip, 1 - x, x)

    return pd.DataFrame({
        "A": noisy(z, 0.12), "B": noisy(z, 0.12), "Z": z,
        "shape": rng.randint(0, 4, n),
    }).astype(str)


def test_dashboard_summary_works_with_no_data(api_client):
    r = api_client.get("/api/dashboard/summary")
    assert r.status_code == 200 and r.json()["loaded"] is False


def test_parse_use_loaded_requires_dataset(api_client):
    assert api_client.post("/api/parse/use-loaded").status_code == 400


def test_cramers_contingency_conditional_flow(api_client, demo_df):
    assert _upload_csv(api_client, demo_df).status_code == 200

    r = api_client.post("/api/analysis/cramers-v",
                        json={"columns": list(demo_df.columns), "ci_top_n": 2})
    rep = r.json()
    assert r.status_code == 200 and rep["fdr_method"] == "benjamini-hochberg"
    assert rep["pairs"][0].get("q") is not None
    assert rep["pairs"][0].get("ci") is not None          # ci_top_n applied

    r = api_client.post("/api/analysis/contingency", json={"col1": "A", "col2": "B"})
    assert r.status_code == 200 and r.json()["p"] is not None

    r = api_client.post("/api/analysis/conditional",
                        json={"col1": "A", "col2": "B", "condition_on": "Z"})
    assert r.status_code == 200 and r.json()["verdict"] == "explained_by_z"


def test_batch_import_display_cap_vs_full_export(api_client, master_schema_text):
    """The JSON payload is preview-capped (500 rows); /api/parse/export is not."""
    n = 2100
    lines = [json.dumps({"custom_id": str(i), "response": {"body": {"choices": [
        {"message": {"content": json.dumps({"location": {"country": "US"},
                                            "record": {"id": str(i)}})}}]}}})
        for i in range(n)]
    r = api_client.post(
        "/api/parse/upload-batch",
        files={"file": ("out.jsonl", io.BytesIO("\n".join(lines).encode()), "application/jsonl")},
    )
    j = r.json()
    assert r.status_code == 200 and j["n_ok"] == n
    assert j["data"]["returned_rows"] == 500               # preview cap (30 MB payloads froze the browser)
    assert j["data"]["total_rows"] == n                    # honest total

    r = api_client.get("/api/parse/export")
    assert r.status_code == 200
    assert r.text.count("\n") >= n                         # header + ALL rows


def test_scu_normalize_upload_and_remap(api_client):
    rec = {"date_time": {"year": 1960, "month": 5, "day": 3},
           "location": {"country": "USA"},
           "object": {"primary_shape": "disc", "size": "car"},
           "engagement": {"engagement_type": {"occupant_observed": "P"}},
           "source": {"name": "Blue Book"},
           "witness": {"roles": ["pilot"], "type": "military"}}
    r = api_client.post("/api/scu/normalize-upload", files={
        "file": ("p.json", io.BytesIO(json.dumps([rec]).encode()), "application/json")})
    j = r.json()
    assert r.status_code == 200
    assert j["mapping"]["methods"]["investigation.source"] == "alias"

    r = api_client.post("/api/scu/remap",
                        json={"column_map": {"craft.size": "object.primary_shape"}})
    assert r.status_code == 200
    assert r.json()["mapping"]["methods"]["craft.size"] == "manual"

    # full exports: normalized always; filtered only after a filter ran
    r = api_client.get("/api/scu/export?scope=filtered")
    assert r.status_code == 400                            # nothing filtered yet
    r = api_client.get("/api/scu/export")
    assert r.status_code == 200 and r.text.count("\n") >= 1
    r = api_client.post("/api/scu/filter", json={"criterion_keys": ["has_core_fields"]})
    assert r.status_code == 200
    r = api_client.get("/api/scu/export?scope=filtered")
    assert r.status_code == 200
