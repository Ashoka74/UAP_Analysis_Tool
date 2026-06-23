"""Build the LLM gold set ONCE for the flag-similar-events autoresearch loop.

Blocks the full PURSUE set on (year, month), samples candidate pairs across
narrative-similarity bands so the gold set spans the decision boundary, then
asks Gemini "same real-world event?" for each. Caches to gold_pairs.csv keyed by
stable CSV row index. NOT edited by the loop — the loop's Verify reads the cache.

    uv run python eval_pairs.py --dry     # show the candidate pool, no LLM
    uv run python eval_pairs.py           # judge + write gold_pairs.csv
"""
import sys
import json
import time
import tomllib
import itertools

import numpy as np
import pandas as pd

import make_timeline as mt

SRC = mt.SRC
NARR = mt._NARR
GOLD = "gold_pairs.csv"
MODEL = "models/gemini-3.1-pro-preview"
SIM_EDGES = [0.0, 0.2, 0.4, 0.55, 0.70, 0.85, 1.01]
PER_BAND = 28           # ~165 pairs total across 6 bands
GROUP_PAIR_CAP = 250    # cap pairs per (year,month) before scoring
SEED = 7


def load():
    df = pd.read_csv(SRC)
    df = df[df["date_time.year"].notna() & (df["date_time.year"] > 1000)]
    return df                      # keep original CSV index as the stable id


def _ctx(row):
    y = mt._int_or_none(row.get("date_time.year"))
    m = mt._int_or_none(row.get("date_time.month"))
    d = mt._int_or_none(row.get("date_time.day"))
    loc = row.get("sightingDetails.location.name") or row.get("location.name") or "?"
    tod = row.get("sightingDetails.timeOfDay") or row.get("date_time.local_time") or ""
    nar = str(row.get(NARR, "")).strip()
    return f"Date: {y}-{m}-{d} {tod}; Location: {loc}\n{nar}"


def candidates(df):
    rng = np.random.default_rng(SEED)
    by_ym = {}
    for i, r in df.iterrows():
        y = mt._int_or_none(r["date_time.year"])
        m = mt._int_or_none(r.get("date_time.month"))
        if y is None:
            continue
        by_ym.setdefault((y, m), []).append(i)
    pairs = []
    for ids in by_ym.values():
        gp = list(itertools.combinations(ids, 2))
        if len(gp) > GROUP_PAIR_CAP:
            sel = rng.choice(len(gp), GROUP_PAIR_CAP, replace=False)
            gp = [gp[k] for k in sel]
        pairs.extend(gp)
    return pairs


def sample_bands(df, pairs):
    rng = np.random.default_rng(SEED + 1)
    need = set(p for pr in pairs for p in pr)
    narr = {i: str(df.loc[i, NARR]) for i in need}
    scored = [(a, b, mt._narrative_sim(narr[a], narr[b])) for a, b in pairs]
    out = []
    for lo, hi in zip(SIM_EDGES[:-1], SIM_EDGES[1:]):
        band = [t for t in scored if lo <= t[2] < hi]
        if len(band) > PER_BAND:
            sel = rng.choice(len(band), PER_BAND, replace=False)
            band = [band[k] for k in sel]
        out.extend(band)
        print(f"  band [{lo:.2f},{hi:.2f}): {len(band)} sampled "
              f"(of {sum(1 for t in scored if lo <= t[2] < hi)})")
    return out


def judge(df, sampled):
    with open(".streamlit/secrets.toml", "rb") as f:
        key = tomllib.load(f)["GEMINI_KEY"]
    import google.generativeai as genai
    genai.configure(api_key=key)
    model = genai.GenerativeModel(MODEL, generation_config={
        "response_mime_type": "application/json", "temperature": 0.0,
        "max_output_tokens": 65536})
    labels = {}
    CHUNK = 50
    for c0 in range(0, len(sampled), CHUNK):
        chunk = sampled[c0:c0 + CHUNK]
        blocks = [f"=== PAIR {k} ===\nRecord A:\n{_ctx(df.loc[a])}\n\nRecord B:\n{_ctx(df.loc[b])}"
                  for k, (a, b, _s) in enumerate(chunk, start=c0)]
        prompt = (
            "You are a UFO-report deduplication judge. For each numbered pair, decide "
            "whether records A and B describe the SAME real-world sighting/event — the "
            "same object seen by the same witness(es) at the same place and time — "
            "allowing for differences in wording, date/location precision, missing "
            "fields, or one source quoting another. Distinct sightings on the same day "
            "in the same region (a 'flap') are NOT the same event.\n\n"
            f"Return ONLY a JSON array of exactly {len(chunk)} objects:\n"
            '{"id": <pair id>, "same_event": <true|false>, "reason": "<short>"}\n\n'
            + "\n\n".join(blocks))
        resp = model.generate_content(prompt)
        try:
            arr = json.loads(resp.text)
        except json.JSONDecodeError:
            import re
            arr = json.loads(re.search(r"\[.*\]", resp.text, re.DOTALL).group(0))
        for v in arr:
            labels[int(v["id"])] = (bool(v.get("same_event", False)), str(v.get("reason", "")))
        print(f"  judged {min(c0+CHUNK, len(sampled))}/{len(sampled)}")
    return labels


if __name__ == "__main__":
    df = load()
    pairs = candidates(df)
    print(f"datable rows: {len(df)} | candidate pairs (blocked, capped): {len(pairs):,}")
    sampled = sample_bands(df, pairs)
    print(f"sampled for judging: {len(sampled)} pairs")
    if "--dry" in sys.argv:
        sys.exit(0)
    labels = judge(df, sampled)
    rows = []
    for k, (a, b, s) in enumerate(sampled):
        if k not in labels:
            continue
        same, reason = labels[k]
        rows.append({"i_id": a, "j_id": b, "narrative_sim": round(s, 4),
                     "llm_same_event": int(same), "llm_reason": reason})
    out = pd.DataFrame(rows)
    out.to_csv(GOLD, index=False)
    print(f"\nWrote {GOLD}: {len(out)} labeled pairs | "
          f"positives(same_event)={int(out['llm_same_event'].sum())} "
          f"negatives={int((1-out['llm_same_event']).sum())}")
