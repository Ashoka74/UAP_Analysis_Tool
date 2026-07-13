# PURSUE Corpus Report — Deduplication, EDA & Manifest Coverage

**Corpus:** `SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv` (1,873 rows × 363 columns)
**Cross-referenced against:** the official war.gov UAP release manifest (334 rows,
releases 1–4, published 5/8/26 – 7/10/26)
**Figures:** `figures/01`–`12` (PDF + PNG, 300 DPI, colorblind-safe)

---

## 1. Executive Summary

- The working corpus merges **PURSUE_1_2_3** (1,791 rows) with the newer **PURSUE_4**
  release (82 rows) — see [`figures/01_source_composition`](figures/01_source_composition.png).
- Deduplication (embedding cosine blocking + Gemini Tier-3 confirmation) found
  **251 clusters** covering 578 rows, of which **327 are redundant** (non-canonical)
  duplicates — a ~17.5% reduction in the merged corpus.
- **4 of those 251 clusters span both releases** — i.e. 6 PURSUE_4 documents
  re-describe an event already present in PURSUE_1_2_3. This is the exact scenario
  cross-release dedup was built to catch (see §3).
- Cross-referencing against the official 334-row war.gov manifest shows the corpus
  covers **165 of 216 linked documents (76.4%)**. The **51 uncovered documents are
  entirely non-text media** — 27 images and 24 PDFs — none image-derived, since the
  extraction pipeline has no image-content path (see §4).

---

## 2. Corpus Overview

| | PURSUE_1_2_3 | PURSUE_4 | Combined |
|---|---:|---:|---:|
| Rows | 1,791 | 82 | 1,873 |
| Share | 95.6% | 4.4% | 100% |

- **Timeline** ([`02_year_distribution`](figures/02_year_distribution.png),
  [`03_year_eras`](figures/03_year_eras.png)): sightings are heavily concentrated
  in **1945–1975** (1,461 rows, 78% of dated rows), dominated by a sharp 1947 spike
  — consistent with the "flying disc" reporting wave of that summer.
- **Geography** ([`04_top_countries`](figures/04_top_countries.png)): 1,383 US
  reports; "UNKNOWN" (148) and "INTL_WATERS" (51) are the next largest buckets.
- **Craft shape** ([`05_craft_shapes`](figures/05_craft_shapes.png)): "Unknown"
  (464), Sphere (438), and Disc (313) are the three largest categories.
- **Trust scoring** ([`06_trust_score_distribution`](figures/06_trust_score_distribution.png),
  [`07_trust_bands`](figures/07_trust_bands.png)): mean trust score 60.0 (σ=18.2);
  778 reports rated "high" trust, 202 "very_high".
- **Day/night** ([`08_day_night`](figures/08_day_night.png)): 710 night, 617 day,
  487 unknown/unspecified.
- **Source agency** ([`09_agencies`](figures/09_agencies.png)): DOD (77) and FBI (2)
  are the only two agency values populated in this column — most rows carry agency
  attribution elsewhere (`source.agency`) or not at all.
- **Engagement / performance anomalies** ([`10_engagement_performance_rates`](figures/10_engagement_performance_rates.png)):
  "low observability" is flagged in 41.3% of reports; "over military installation"
  in 20.6%; "aircraft encounters" in 16.3%.
- **Investigation status** ([`11_timeliness_status`](figures/11_timeliness_status.png)):
  1,637 reports "presumed timely via source" vs. 154 "unknown, no investigation."
- **Missingness:** average column missingness is 59.4%; 175 of 363 columns are
  >90% empty — expected for a 30-year span of heterogeneously-structured source
  documents merged into one schema.

---

## 3. Deduplication Results

Methodology (see `merge_and_dedup_pursue4.py`): block candidate pairs by
(year, month) → filter by narrative-embedding cosine ≥ 0.80 → gate on date overlap
+ field agreement (relaxed to cosine + date/location only for any pair touching a
PURSUE_4 row, which lacks the enrichment columns PURSUE_1_2_3 has) → Gemini
Tier-3 "same real-world event?" confirmation → union-find clustering → canonical
row = highest trust score / longest narrative per cluster.

| Metric | Value |
|---|---:|
| Candidate pairs (post-blocking, cosine ≥ 0.80) | 503 |
| LLM-confirmed same-event pairs | 384 |
| Clusters formed | 251 |
| Rows in a cluster | 578 |
| Redundant (non-canonical) rows | 327 |
| Largest cluster | 6 rows |
| Clusters of size ≥ 3 | 53 |
| **Cross-release clusters (PURSUE_1_2_3 ↔ PURSUE_4)** | **4** |
| PURSUE_4 rows involved in any cluster | 6 / 82 (7.3%) |

Nothing is deleted — every row keeps `dup_cluster_id` / `dup_role` /
`dup_canonical_row` / `dup_llm_reason` so a downstream consumer can choose to
collapse to canonical rows or keep everything with duplicates flagged. Full
detail: `PURSUE_1_2_3_4_dedup_flagged.xlsx` (sheets `flagged`, `clusters`).

---

## 4. Document Coverage vs. the Official Manifest

The war.gov manifest lists 334 released assets (158 in release 1 batch dated
5/8/26, 64 dated 5/22/26, 72 dated 6/12/26, 40 dated 7/10/26 = release 4). Of
these, **216 have a direct downloadable link** (183 PDF + 27 IMG + 12 VID with a
link; the remaining 118 VID/AUD entries are referenced only by DVIDS ID, with no
direct file link to cross-check against).

Joining the corpus to the manifest on that shared link column
([`12_document_coverage`](figures/12_document_coverage.png)):

| Type | Released (linked) | Represented in corpus | Coverage |
|---|---:|---:|---:|
| PDF | 189 | 165 | 87.3% |
| IMG | 27 | 0 | 0% |
| **Total** | **216** | **165** | **76.4%** |

**Every one of the 27 released images has zero corresponding extracted report.**
This is not a bug in this corpus — it's a structural gap: the extraction pipeline
(`extract_reports.py` / `pdf_to_reports.py`) parses narrative text out of PDFs; it
has no path for turning a standalone photo (FBI photo series, NASA STS-80 mission
imagery, Apollo/Gemini frames) into a structured sighting report. This is the same
gap flagged earlier for multimodal embeddings — `pipeline/embed_media.py` handles
these via image embeddings into pgvector, but that's a separate representation
from a `DenseNarrativeSection`-style report row, so it doesn't close this
coverage number.

The 24 uncovered **PDF** documents are a mix of:
- **Release 4 documents not yet OCR'd/extracted** at merge time (14 of the 24 —
  e.g. `DOW-UAP-D089`–`D097`, `CIA-UAP-D020/D021`, `DOE-UAP-D004/D005`) — release 4
  landed 7/10/26, the same day this corpus was assembled, so some may simply be
  mid-pipeline.
- **Older gaps from releases 1–3** (10 of the 24) — e.g. FBI serial/section files,
  NASA Gemini debriefings, one malformed filename (`59_64634_711.5612[7-2852.pdf`
  — the `[` is almost certainly a transcription artifact in the manifest itself).

Full list with title/agency/release date: `pipeline_data/missing_documents.csv`
(51 rows — gitignored, regenerate via
`python dedupe_all_reports.py --manifest <manifest.csv>`).

**Extraction density** — reports-per-document, PDF-linked, is heavily right-skewed:
median 1, mean 10.9, max 237 (a single Sandia general-correspondence bundle and an
FBI section file each yielded 237 individual sighting reports). 43 of 165 covered
documents (26%) contain more than one extracted report.

---

## 5. Data Quality Notes

- The manifest's `Type` column contains both `"PDF"` and `"PDF "` (trailing space)
  as distinct string values for 6 rows — a source-side artifact, normalized in
  `dedupe_all_reports.py` before aggregation but left as-is in the raw manifest.
- One manifest filename contains a literal `[` character
  (`59_64634_711.5612[7-2852.pdf`) — worth confirming against the live PDF link
  before treating as a genuine missing-coverage case.
- `date_time.year` contains at least one `0` sentinel value in the merged corpus
  (excluded from the year-distribution figures via a `> 1400` floor).

---

## Reproducing this report

```bash
uv run python merge_and_dedup_pursue4.py        # builds the merged + deduped corpus (cached)
uv run python dedupe_all_reports.py \
    --manifest pipeline_data/uap_manifest_export.csv   # figures + coverage cross-ref
```

Manifest source: `https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-data.csv`
(auto-fetched and cached by `pipeline/check_new_release.py`; a browser export of
the same endpoint was used for this cross-reference and confirmed byte-identical
on the 17 substantive columns).
