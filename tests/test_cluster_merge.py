"""Cluster-term merging: the three cluster_cosine bugs.

Shipped failure: a column where HDBSCAN labeled every point noise crashed the
whole Streamlit analysis run ("Expected 2D array, got 1D array: array=[]").
Two silent bugs rode along: noise rows (-1) wrapped to the LAST cluster's name,
and the merge map was keyed by row position but looked up by cluster id.
"""
import uap_analyzer as U


def _analyzer():
    return U.UAPAnalyzer.__new__(U.UAPAnalyzer)   # skip heavy __init__


def test_all_noise_column_does_not_crash(fake_embedder):
    a = _analyzer()
    out = a.merge_similar_clusters(cluster_terms=[], cluster_labels=[-1] * 5)
    assert out == ["Noise"] * 5


def test_noise_rows_are_not_mislabeled(fake_embedder):
    a = _analyzer()
    out = a.merge_similar_clusters(
        cluster_terms=["disc", "triangle"],
        cluster_labels=[0, 1, -1, 0],
        similarity_threshold=0.9999,
    )
    # Pre-fix, the -1 row wrapped to cluster_terms[-1] == "triangle".
    assert out == ["disc", "triangle", "Noise", "disc"]


def test_similar_clusters_actually_merge(fake_embedder):
    a = _analyzer()
    out = a.merge_similar_clusters(
        cluster_terms=["disc", "disk", "triangle"],
        cluster_labels=[0, 1, 2, -1, 1],
        similarity_threshold=0.95,
    )
    assert out == ["disc", "disc", "triangle", "Noise", "disc"]


def test_single_cluster_short_circuits(fake_embedder):
    a = _analyzer()
    out = a.merge_similar_clusters(cluster_terms=["disc"], cluster_labels=[0, -1, 0])
    assert out == ["disc", "Noise", "disc"]
