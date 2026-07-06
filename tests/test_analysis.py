"""analysis_service statistics: CI, FDR, sparsity, conditional independence, dedup."""
import numpy as np
import pandas as pd
import pytest

from api.services import analysis_service as A


@pytest.fixture(scope="module")
def confounded_df():
    """A,B both track Z (marginally associated, conditionally independent)."""
    rng = np.random.RandomState(0)
    n = 300
    z = rng.randint(0, 2, n)

    def noisy(x, p):
        flip = rng.rand(n) < p
        return np.where(flip, 1 - x, x)

    return pd.DataFrame({
        "A": noisy(z, 0.1), "B": noisy(z, 0.1), "Z": z,
        "noise": rng.randint(0, 3, n),
        "rare": np.where(rng.rand(n) < 0.02, 1, 0),
    }).astype(str)


def test_bootstrap_ci_brackets_point_estimate(confounded_df):
    a = A._coalesce(confounded_df["A"])
    b = A._coalesce(confounded_df["B"])
    ci = A.cramers_v_ci(a, b, n_boot=200)
    from uap_analyzer import cramers_v
    v = float(cramers_v(pd.crosstab(a, b)))
    assert ci is not None and ci[0] <= v <= ci[1]
    assert ci[1] - ci[0] < 0.5          # not absurdly wide on n=300


def test_fdr_and_sparse_flags(confounded_df):
    rep = A.cramers_v_report(confounded_df, list(confounded_df.columns))
    assert rep["fdr_method"] == "benjamini-hochberg"
    assert rep["n_tests"] > 0
    by_pair = {(p["a"], p["b"]): p for p in rep["pairs"]}
    ab = by_pair.get(("A", "B")) or by_pair.get(("B", "A"))
    assert ab["q"] is not None and ab["q"] < 0.05            # real signal survives FDR
    assert all(p.get("q", 0) >= p.get("p", 0) - 1e-12
               for p in rep["pairs"] if p.get("q") is not None)   # BH never lowers p
    # 'rare' has ~6 positive cases -> Cochran-sparse tables must be flagged
    sparse_pairs = [p for p in rep["pairs"] if p.get("sparse")]
    assert any("rare" in (p["a"], p["b"]) for p in sparse_pairs)


def test_sparse_2x2_uses_fisher(confounded_df):
    ct = A.contingency(confounded_df, "A", "rare")
    assert ct["sparse"] is True
    assert ct["test"] == "fisher"
    assert ct["p"] is not None


def test_conditional_association_detects_confounder(confounded_df):
    res = A.conditional_association(confounded_df, "A", "B", "Z")
    assert res["verdict"] == "explained_by_z"
    assert res["test"]["method"] == "Cochran–Mantel–Haenszel"
    assert res["test"]["p_value"] > 0.05          # no association within strata
    assert res["marginal_v"] > 0.3                # strong marginal association


def test_dedupe_semantic_prioritizes_engagement_type():
    cols = [
        "engagement.engagement_type.radical_flight", "anomaly.flight",
        "engagement.engagement_type.occupant_observed", "anomaly.occupant",
        "assessment.contradictsUap", "anomaly.validated",
        "witness.count", "witness.countFreeform",
        "location.country",
    ]
    kept, removed = A.dedupe_semantic(cols)
    assert "anomaly.flight" not in kept
    assert "anomaly.occupant" not in kept
    assert "anomaly.validated" not in kept
    assert "witness.countFreeform" not in kept
    assert "engagement.engagement_type.radical_flight" in kept
    assert "assessment.contradictsUap" in kept
    assert len(removed) == 4


def test_dedupe_keeps_twin_when_canonical_absent():
    # anomaly.flight alone (no radical_flight) must NOT be dropped
    kept, removed = A.dedupe_semantic(["anomaly.flight", "location.country"])
    assert "anomaly.flight" in kept and removed == []
