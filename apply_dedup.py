"""Apply the tuned flag-similar-events pipeline to the full PURSUE Excel.

Tier-1/2 (programmatic): block on (year, month); keep pairs passing the loop's
best gates (same-day date overlap + field-agreement ≥0.5) AND embedding cosine
≥ APPLY_EMBED_MIN — a high-confidence candidate set.
Tier-3 (LLM): Gemini confirms each candidate is the SAME real-world event
(drops same-day flap pairs the programmatic gates can't separate).
Confirmed pairs → union-find clusters → canonical row per cluster (highest
trustScore). Writes a flagged workbook; nothing is deleted (FP ≫ FN).
"""
import os
import re
import sys
import itertools
from collections import defaultdict

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # openpyxl-forbidden control chars


def _sanitize(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: _ILLEGAL.sub("", v) if isinstance(v, str) else v)
    return df

import numpy as np
import pandas as pd

import make_timeline as mt
import geo_dedup as gd
import eval_pairs as ep

SRC = mt.SRC
OUT_XLSX = "PURSUE_1_2_3_dedup_flagged.xlsx"
OUT_PAIRS = "autoresearch/loop-260618-2053/confirmed_dup_pairs.csv"
APPLY_EMBED_MIN = 0.80


def candidate_pairs(df, emb):
    dd = df[df["date_time.year"].notna() & (df["date_time.year"] > 1000)]
    by_ym = defaultdict(list)
    for i, r in dd.iterrows():
        y = mt._int_or_none(r["date_time.year"])
        m = mt._int_or_none(r.get("date_time.month"))
        by_ym[(y, m)].append(i)
    cands = []
    for v in by_ym.values():
        for a, b in itertools.combinations(v, 2):
            es = float(emb[a] @ emb[b])
            if es < APPLY_EMBED_MIN:
                continue
            ra, rb = df.loc[a], df.loc[b]
            s = mt._narrative_sim(str(ra.get(mt._NARR, "")), str(rb.get(mt._NARR, "")))
            if gd.predict_same_event(ra, rb, s, embed_sim=es):
                cands.append((a, b, round(es, 4)))
    return cands


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def main():
    df = pd.read_csv(SRC)

    # Tier-3 results are cached — reuse them (the LLM calls are the expensive part).
    if os.path.exists(OUT_PAIRS) and os.path.getsize(OUT_PAIRS) > 50:
        cdf = pd.read_csv(OUT_PAIRS)
        confirmed = [(int(r.i_id), int(r.j_id), float(r.embed_sim), str(r.llm_reason))
                     for r in cdf.itertuples()]
        print(f"loaded {len(confirmed)} LLM-confirmed pairs from cache (skipping LLM)")
    else:
        emb = np.load("narr_emb.npy")
        cands = candidate_pairs(df, emb)
        print(f"high-confidence candidate pairs (embed≥{APPLY_EMBED_MIN}): {len(cands)}")
        labels = ep.judge(df, cands)               # Tier-3: LLM confirmation
        sim_by_pair = {k: cands[k][2] for k in range(len(cands))}
        confirmed = [(cands[k][0], cands[k][1], sim_by_pair[k], r)
                     for k, (same, r) in labels.items() if same]
        print(f"LLM-confirmed same-event pairs: {len(confirmed)} / {len(cands)}")
        pd.DataFrame([{"i_id": a, "j_id": b, "embed_sim": s, "llm_reason": r}
                      for a, b, s, r in confirmed]).to_csv(OUT_PAIRS, index=False)

    # cluster confirmed pairs
    uf = UF()
    sim_edge = {}
    for a, b, s, _r in confirmed:
        uf.union(a, b)
        sim_edge[a] = max(sim_edge.get(a, 0), s)
        sim_edge[b] = max(sim_edge.get(b, 0), s)
    groups = defaultdict(list)
    for a, b, _s, _r in confirmed:
        groups[uf.find(a)].append(a); groups[uf.find(b)].append(b)
    clusters = {root: sorted(set(v)) for root, v in groups.items()}

    # annotate dataframe (nothing deleted)
    df["dup_cluster_id"] = ""
    df["dup_cluster_size"] = 0
    df["dup_role"] = ""
    df["dup_canonical_row"] = ""
    df["dup_match_embed_sim"] = np.nan
    df["dup_llm_reason"] = ""
    reason_by_node = {}
    for a, b, _s, r in confirmed:
        reason_by_node.setdefault(a, r); reason_by_node.setdefault(b, r)

    cluster_rows = []
    for cid, (root, members) in enumerate(sorted(clusters.items()), start=1):
        def _score(i):
            ts = pd.to_numeric(df.loc[i].get("sightingDetails.trustScore"), errors="coerce")
            return (ts if pd.notna(ts) else 0, len(str(df.loc[i].get(mt._NARR, ""))))
        canon = max(members, key=_score)
        for i in members:
            df.at[i, "dup_cluster_id"] = cid
            df.at[i, "dup_cluster_size"] = len(members)
            df.at[i, "dup_role"] = "canonical" if i == canon else "duplicate"
            df.at[i, "dup_canonical_row"] = canon
            df.at[i, "dup_match_embed_sim"] = round(sim_edge.get(i, np.nan), 4)
            df.at[i, "dup_llm_reason"] = reason_by_node.get(i, "")
        yrs = sorted({mt._int_or_none(df.loc[i].get("date_time.year")) for i in members})
        cluster_rows.append({
            "cluster_id": cid, "size": len(members),
            "year(s)": ",".join(str(y) for y in yrs if y),
            "canonical_row": canon,
            "canonical_title": str(df.loc[canon].get("Title", ""))[:60],
            "member_rows": ",".join(map(str, members)),
        })

    df["is_duplicate"] = df["dup_role"] == "duplicate"
    n_clusters = len(clusters)
    n_dups = int((df["dup_role"] == "duplicate").sum())
    print(f"clusters={n_clusters} | rows in clusters={int((df['dup_cluster_id']!='').sum())} | "
          f"redundant (non-canonical) rows={n_dups}")

    df = _sanitize(df)
    clusters_df = _sanitize(pd.DataFrame(cluster_rows))
    df.to_csv(OUT_XLSX.replace(".xlsx", ".csv"), index=False)   # robust fallback
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="flagged", index=False)
        clusters_df.to_excel(xw, sheet_name="clusters", index=False)
    print(f"wrote {OUT_XLSX} (sheets: flagged {df.shape}, clusters [{n_clusters}])")
    return n_clusters, n_dups


if __name__ == "__main__":
    main()
