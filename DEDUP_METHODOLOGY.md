# Sighting Deduplication Methodology

This document describes how UAP-Data-Analysis-Tool deduplicates sighting
records. Unlike a multi-source merge (different databases reporting
independently on the same event), this repo's dedupe problem is
**within-source**: the same real-world sighting can appear more than once in
`SUBDATASETS_V2/PURSUE_1_2_3_normalized_full.csv` because the underlying
document release (FBI/CIA/military file dumps assembled into the PURSUE
corpus) was re-scanned, re-OCR'd, or re-filed across multiple releases.

The corpus in scope: **1,791 rows**, spanning 1833–2025.

## Two implementations, two purposes

The repo carries a lightweight fallback and a tuned, LLM-verified production
pipeline. `make_timeline.py` picks between them automatically:

| | Legacy fallback (`_build_legacy`) | Production pipeline (`apply_dedup.py`) |
|---|---|---|
| Used when | `PURSUE_1_2_3_dedup_flagged.csv` doesn't exist yet | Pre-computed dedup columns are present |
| Blocking | exact `(year, month, day)` + normalized location string | `(year, month)` only — coarser, cast a wide net |
| Confirmer | lexical `_narrative_sim` ≥ `SIM_TAU` (0.55), clique linkage | embedding cosine + tuned structured gates, then an LLM |
| Cost | free, instant, no network calls | embeddings + ~300 Gemini calls |
| Output | in-memory clusters for one `build()` call | persisted `dup_cluster_id`/`dup_role`/canonical columns |

The legacy path exists so the timeline renderer always produces *something*
reasonable without requiring embeddings or an API key; the production path is
what actually ships in `PURSUE_1_2_3_dedup_flagged.xlsx`.

## Production pipeline: three tiers

### Tier 1/2 (programmatic) — `apply_dedup.candidate_pairs` + `geo_dedup.predict_same_event`

1. **Block** on `(year, month)` — every same-year-month pair is a candidate.
2. **Embedding pre-filter**: keep only pairs with narrative-embedding cosine
   ≥ `APPLY_EMBED_MIN = 0.80` (MiniLM narrative embeddings, `narr_emb.npy`).
3. **`predict_same_event`** — a conjunctive rule over four independent
   signals, each individually tunable (`PREDICT_PARAMS`):

   | Gate | Current value | What it checks |
   |---|---|---|
   | Text-similarity floor (`tau`) | **0.10** | lexical `_narrative_sim` (Jaccard + `SequenceMatcher`), or embedding cosine if `use_embed` |
   | Date-overlap gate | **on**, `t_max_days = 0` | plausible date intervals (uncertainty-scaled, see below) must overlap |
   | Location gate | **off** | normalized location / shared state match — tested, hurt precision (see calibration) |
   | Field-agreement gate | **on**, `field_min = 0.5` | ≥50% of `{country, state, craft shape, craft size, witness role, year}` that are populated on both rows must agree |

Only pairs passing all active gates become Tier-3 candidates.

### Geo/date signal — `geo_dedup.py`

Adds the location axis the exact-string block key was missing:
- **Geocoding**: forward-geocodes `location.name`/city+state via Nominatim,
  cached to `geocode_cache.csv`. Native lat/lon wins when present. A geocode
  is only "trusted" for tight blocking when it resolves to town-or-finer
  precision (`place_rank ≥ 12`, `importance ≥ 0.15`) — coarse state/country
  centroids never collapse two rows onto one point.
- **`date_gap_days`**: gap between two rows' plausible date *intervals*, not
  point dates. Each row's interval half-width scales with its stated
  uncertainty (`exact`→0 days, `approximate_month`→15d, `approximate_year`→182d,
  `unknown`→3650d), so a vague date can still overlap a precise one.
- **`proximity_score = exp(-km/30) * exp(-gap_days/3)`** — a smooth
  blocking/ranking signal, **never the merge verdict itself**. Narrative
  similarity stays a separate, independent column so the two never conflate.

### Tier 3 (LLM confirmation) — `eval_pairs.py` / `apply_dedup.py`

High-confidence candidates go to Gemini (`models/gemini-3.1-pro-preview`) with
both records' date/location/narrative, and the model answers "same
real-world sighting?" — explicitly instructed that same-day sightings in the
same region ("a flap") are *not* the same event. Confirmed pairs are cached
to `autoresearch/loop-260618-2053/confirmed_dup_pairs.csv` so re-runs never
re-pay for LLM calls on unchanged data.

### Clustering & canonical selection

Confirmed pairs are merged via **union-find** (unlike a purely pairwise
advisory table, this repo *does* take transitive closure — if A↔B and B↔C
are both confirmed, A/B/C become one cluster). Canonical row per cluster =
highest `sightingDetails.trustScore`, tiebreak longest narrative. **Nothing
is deleted** — every row keeps its original data; the workbook adds
`dup_cluster_id`, `dup_cluster_size`, `dup_role` (`canonical`/`duplicate`),
`dup_canonical_row`, `dup_match_embed_sim`, `dup_llm_reason`, and
`is_duplicate`. `make_timeline.py` then renders one timeline entry per
cluster, rooted at the canonical row, with citations unioned across every
member's pages + sources.

## Calibration: the autoresearch tuning loop

`PREDICT_PARAMS` wasn't hand-picked — it's the output of a 9-iteration
autoresearch loop (`autoresearch/loop-260618-2053/`) that hill-climbed
against a small Gemini-labeled gold set (96 pairs: 45 positive / 51 negative,
stratified across narrative-similarity bands via `eval_pairs.py`), scored by
F1 of `predict_same_event` vs. the LLM labels (`geo_dedup.py verify`).

| Iter | Change | F1 | Kept? |
|---|---|---|---|
| 0 | baseline: τ=0.55, sim-only, no gates | 0.316 | — |
| 1 | lower τ 0.55→0.30 | 0.674 | ✅ |
| 2 | + date gate (t_max=2d) | 0.681 | ✅ |
| 3 | + location gate | 0.667 | ❌ discarded |
| 4 | + field-agreement gate ≥0.5 | 0.682 | ✅ |
| 5 | lower τ 0.30→0.10 | 0.804 | ✅ |
| 6 | tighten field_min 0.5→0.7 | 0.759 | ❌ discarded |
| 7 | tighten date window t_max 2→0 | **0.812** | ✅ (final) |
| 8 | field_min 0.5→0.6 | 0.784 | ❌ discarded |
| 9 | high-sim bypass (accept if sim≥0.6 regardless of gates) | 0.804 | ❌ discarded |
| 10 | swap confirmer to MiniLM embedding cosine (τ=0.78) | 0.764 | ❌ discarded |
| 11 | embedding confirmer at low floor (τ=0.45) | 0.796 | ❌ discarded |

**F1 0.316 → 0.812 (+157%)** over the run. Findings that shaped the final
rule:
- **Lowering** the text-similarity floor helped more than raising it — true
  duplicates are often *lexically dissimilar* (different scanning artifacts,
  paraphrased OCR); the structured gates (date overlap, field agreement) are
  what actually carries the signal once τ stops filtering them out.
- **Strict location matching backfired** — location-string drift between
  releases loses true-duplicate recall without removing false positives, so
  the location gate stays off.
- **An embedding-cosine confirmer did not beat the lexical+structured rule**
  (0.764 and 0.796 vs. 0.812) — the residual false positives are same-day
  "flap" pairs that remain *semantically* similar (embedding cosine up to
  0.95) even though they're different events. No similarity metric alone
  separates them; that's explicitly deferred to the Tier-3 LLM judge.
- The gold set is small (96 pairs); the loop's own recommendation is to grow
  labels over the unresolved/borderline similarity band before tightening
  further, rather than keep hill-climbing on 96 pairs.

## Results (current `PURSUE_1_2_3_dedup_flagged.xlsx`)

| Metric | Value |
|---|---|
| Total rows | 1,791 |
| Rows in a duplicate cluster (any role) | 634 (35.4%) |
| Rows flagged `duplicate` (non-canonical) | 357 (19.9%) |
| Clusters | 277 |
| Canonical rows | 277 |
| Embedding-cosine range within confirmed matches | 0.800 – 0.986 (mean 0.894) |

Cluster size distribution:

| Size | Count |
|---|---|
| 2 | 222 |
| 3 | 37 |
| 4 | 13 |
| 5 | 3 |
| 6 | 2 |

## Validation / testing

- `geo_dedup.silver_label` — a heuristic auto-labeler used only to sanity-check
  band ratios and threshold sweeps before the LLM gold set existed: same
  exact-block + matching craft shape ⇒ positive; date gap > 30 days or
  geo distance > 250 km ⇒ negative; everything else is left `None` (the
  borderline band that requires the LLM/gold set).
- `geo_dedup.band_ratio` / `geo_dedup.sweep` — diagnostic tools that print
  duplicate rate by geo/date band and a precision/recall grid over
  `(proximity_score, narrative_sim)` thresholds, so a threshold change can be
  sanity-checked against real duplicate-rate curves before it's adopted.
- `eval_pairs.py` — builds and caches the LLM gold set (`gold_pairs.csv`),
  never edited by the tuning loop; the loop's Verify step (`geo_dedup.py
  verify`) always re-scores against this frozen file.

## A related but distinct subsystem

`analyzing.py`'s `dedupe_semantic` (exercised by
`tests/test_analysis.py::test_dedupe_semantic_prioritizes_engagement_type`)
is a **feature-column** deduplicator, not a sighting-record one: it drops
redundant/derived structured columns (e.g. `anomaly.flight` when
`engagement.engagement_type.radical_flight` already encodes the same signal)
before XGBoost training, so correlated twin features don't dilute feature
importance. It shares no code with the sighting-level pipeline above — see
`paper/methodology.tex` §"Redundancy control" for its own methodology.

## File inventory

| File | Role |
|---|---|
| `make_timeline.py` | Renders the canonical-rooted timeline; owns the legacy fallback dedupe (`_block_key`, `_narrative_sim`, `_cluster_block`) and the clustered-source renderer (`_build_clustered`) |
| `geo_dedup.py` | Geocoding, haversine/date-gap proximity scoring, the tunable `predict_same_event` rule, and the calibration harness (`silver_label`, `band_ratio`, `sweep`, `verify_f1`) |
| `apply_dedup.py` | Runs the full pipeline end-to-end: candidate generation → gates → Gemini Tier-3 confirmation → union-find clustering → writes `PURSUE_1_2_3_dedup_flagged.xlsx`/`.csv` |
| `eval_pairs.py` | Builds/caches the stratified LLM gold set (`gold_pairs.csv`) used to score every tuning iteration |
| `autoresearch/loop-260618-2053/` | The tuning-loop run: `results.tsv` (iteration log), `findings.md`/`evals-summary.md` (summary), `confirmed_dup_pairs.csv` (Tier-3 LLM cache), `handoff.json` |
| `PURSUE_1_2_3_dedup_flagged.xlsx` | Output workbook: `flagged` sheet (every row + dedup columns) and `clusters` sheet (one row per cluster) |
| `geocode_cache.csv` | Sidecar cache of Nominatim geocode results, keyed by query params |
