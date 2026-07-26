# Exploratory Data Analysis Report: PURSUE_1_2_3_4_merged.csv

**Generated:** 2026-07-13

---

## Executive Summary

This report provides a comprehensive exploratory data analysis of
`SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv` — the merged UAP sighting-report
corpus produced by `merge_and_dedup_pursue4.py` (PURSUE_1_2_3 + PURSUE_4). It
covers file/format identification, structure and data-type analysis, missing-data
and duplicate-detection, outlier screening, correlation analysis, and
recommendations for downstream use. This is the general/statistical companion to
[`PURSUE_corpus_report.md`](PURSUE_corpus_report.md), which covers the domain
findings (deduplication results, manifest coverage); this report focuses on data
quality and structure of the file itself.

---

## Basic Information

- **Filename:** `PURSUE_1_2_3_4_merged.csv`
- **Full Path:** `SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv`
- **File Size:** 17.8 MB (17,768,608 bytes)
- **Last Modified:** 2026-07-11T16:47:54
- **Extension:** `.csv`
- **Format Category:** General Scientific Data — tabular (Comma-Separated Values)

---

## File Type Details

### Format Description
Plain-text comma-separated tabular data — a UTF-8 CSV with a header row, no BOM.

### Typical Data Content
Experimental/observational records; here, structured UAP sighting reports with
nested-field-style flattened column names (`sightingDetails.*`, `date_time.*`,
`location.*`, `craft.*`, `engagement_flags.*`, `performance.*`).

### Common Use Cases
Cross-tool data exchange, spreadsheet/BI import, direct pandas ingestion for
statistical and ML pipelines.

### Python Libraries for Reading
- `pandas`: `pd.read_csv('PURSUE_1_2_3_4_merged.csv', low_memory=False)`
  — `low_memory=False` recommended given 363 mixed-type columns
- `polars`: faster alternative for repeated large reads
- `csv` (stdlib): only for streaming/row-level validation, not analysis

---

## Data Structure Analysis

### Overview
1,873 rows × 363 columns. Two source releases concatenated with a provenance
tag (`_source_dataset`): 1,791 rows from PURSUE_1_2_3, 82 from PURSUE_4.

### Dimensions
| | Value |
|---|---:|
| Rows | 1,873 |
| Columns | 363 |
| Cells | 679,899 |

### Data Types
| dtype | Column count |
|---|---:|
| object (text/mixed) | 309 |
| float64 | 53 |
| int64 | 1 |

54 columns (15%) are numeric; the remaining 85% are text/categorical, consistent
with a document-extraction corpus where most fields are free text, enum-like
categories, or sparsely-populated flags.

---

## Quality Assessment

### Completeness
- **Missing values:** average column missingness is 59.4%; 175 of 363 columns
  (48%) are more than 90% empty. This is expected — the schema is a superset
  across two extraction eras and multiple document types, so no single row
  populates every field. Core fields are well covered:
  `sightingDetails.DenseNarrativeSection` and `sightingDetails.trustScore` are
  100% populated; `date_time.year` is 89% populated.
- **Data coverage:** see [`PURSUE_corpus_report.md`](PURSUE_corpus_report.md) §4
  for coverage against the official 334-document release manifest (76.4% of
  linked documents have at least one extracted report).

### Validity
- **Range check (`date_time.year`):** 1 row has a `0` sentinel value (should be
  null, not zero) and 2 rows have implausible future years (**2030, 2040**) —
  likely transcription/OCR errors on a two-digit year. All other years fall in a
  plausible 1833–2026 range.
- **Format compliance:** encoding is valid UTF-8 with no BOM; no malformed rows
  encountered on load.
- **Consistency:** three witness-count columns
  (`sightingDetails.observerDetails.numberOfWitnesses`, `witness.count`,
  `witness_count_num`) are **perfectly correlated (r=1.00)** — they are the same
  value carried under three names from different pipeline stages, not
  independent measurements. Downstream analysis should pick one and drop the
  other two to avoid accidental double-weighting.

### Integrity
- **Duplicate rows (exact, all 363 columns):** **0** — no literal copy-paste
  duplicates. (Near-duplicate *reports describing the same event* are a separate,
  semantic concept, handled by the embedding+LLM dedup pipeline — see
  `PURSUE_corpus_report.md` §3 — and are not detectable by exact-row comparison.)
- **Duplicate column names:** 0.
- **Legacy artifact columns:** `Unnamed: 0` and `Unnamed: 0.1` are stray
  pandas row-index columns baked in from an earlier save/reload cycle upstream
  of this merge. They are **not** the same index (608 of 1,791 overlapping rows
  have different values between the two), carry no analytical meaning, and
  `Unnamed: 0.1` is null for all 82 PURSUE_4 rows (that source file never had
  it). **Recommend dropping both** before analysis or publication.
- **File corruption check:** file parses cleanly under `pd.read_csv`; no
  encoding errors, no ragged rows detected.

---

## Statistical Summary

### Numerical Variables
Of 54 numeric columns, 9 have ≥50% coverage and are meaningfully analyzable:

| Column | n (coverage) | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| `sightingDetails.trustScore` | 1,873 (100%) | 60.0 | 18.2 | 5 | 93 |
| `date_time.year` | 1,661 (89%) | 1954.1 | 52.0 | 0 | 2040 |
| `date_time.month` | 1,637 (87%) | 6.6 | 3.2 | 1 | 12 |
| `date_time.day` | 1,596 (85%) | 14.7 | 9.2 | 1 | 31 |
| `source.chunkCount` | 1,473 (79%) | 1.2 | 0.9 | 1 | 11 |
| `witness_count_num` | 1,338 (71%) | 3.2 | 14.3 | 1 | 200 |
| `witness.count` | 1,402 (75%) | 3.1 | 13.9 | 1 | 200 |
| `sightingDetails.observerDetails.numberOfWitnesses` | 1,385 (74%) | 3.2 | 14.1 | 1 | 200 |
| `Unnamed: 0.1` (legacy index — see Integrity) | 1,791 (96%) | 444.8 | 350.2 | 0 | 1,184 |

### Categorical Variables
| Column | Top values |
|---|---|
| `_source_dataset` | PURSUE_1_2_3 1,791 · PURSUE_4 82 |
| `trust_band` | high 778 · mid 604 · very_high 202 · low 130 · very_low 77 · (missing 82) |
| `date_time.day_night` | N 710 · D 617 · U 487 · (missing 59) |
| `location.country` | US 1,383 · UNKNOWN 148 · INTL_WATERS 51 (+12 more, see figures/04) |
| `craft.primary_shape` | Unknown 464 · Sphere 438 · Disc 313 (+9 more, see figures/05) |

### Distributions
Witness count is extremely right-skewed: median is 1 witness, but 14 reports
claim >20 witnesses and 6 reports claim exactly 200 — almost certainly a
data-entry ceiling/rounding value rather than 6 independently-verified
200-witness events, and worth spot-checking before using this field in any
per-report statistical model. See `figures/06_trust_score_distribution` and
`figures/02_year_distribution` for the two best-populated numeric distributions.

---

## Data Characteristics

### Temporal Properties
- **Time range:** 1833–2026 (excluding the 3 invalid sentinel/future values
  noted above); the effective analytical range is 1945–2026.
- **Sampling rate:** irregular — event-driven historical reports, not a regular
  time series. 78% of dated rows (1,461/1,873) fall in 1945–1975.
- **Missing time points:** not applicable (event log, not a regular series);
  212 rows (11%) have no `date_time.year` at all.

### Spatial Properties
- Location is carried primarily as free-text (`location.country`,
  `location.name`) rather than as consistent lat/lon — a `location.latitude`/
  `location.longitude` pair exists but was not among the ≥50%-coverage numeric
  columns, so precise geospatial analysis (e.g. the haversine proximity checks
  in `geo_dedup.py`) is only available for a minority subset of rows.

### Experimental Metadata
- **Instrument/method:** not applicable — this is a document-extraction corpus,
  not sensor data. The closest analog is `source.chunkCount` (how many OCR
  chunks a source PDF was split into) and `_source_dataset` (which release
  batch a row came from).

---

## Key Findings

1. **Data volume:** 1,873 rows × 363 columns, merged from two release batches
   at a 95.6% / 4.4% split (PURSUE_1_2_3 / PURSUE_4).
2. **Data quality:** core fields (narrative, trust score) are fully populated;
   the long tail of 175 columns >90% empty is structural (schema superset
   across heterogeneous source documents), not a defect. Two stray legacy
   index columns and 3 fully-redundant witness-count columns should be dropped
   before modeling.
3. **Notable patterns:** an extreme concentration of reports in 1947
   (a documented reporting "flap"), and 3 redundant witness-count columns that
   are perfectly correlated (r=1.00), confirming they encode the same value.
4. **Potential issues:** 3 invalid `date_time.year` values (one `0` sentinel,
   two implausible future years 2030/2040); 6 reports with a suspicious
   round-number 200-witness count; 0 exact duplicate rows (the semantic-level
   duplicates — 327 of them — are handled separately and already flagged via
   `dup_cluster_id`/`dup_role`, not visible to a naive row-duplicate check).

---

## Visualizations

Generated by `dedupe_all_reports.py` (Okabe-Ito colorblind-safe palette, 300 DPI
PDF+PNG):

### Distribution Plots
- `figures/02_year_distribution` — full sighting-year histogram
- `figures/06_trust_score_distribution` — trust score histogram + mean line
- `figures/03_year_eras` — sightings binned by era

### Correlation Analysis
Correlation matrix computed over the 9 well-populated numeric columns (see
Statistical Summary); the only structurally meaningful correlation is the
r=1.00 identity among the three witness-count columns. No figure was generated
for this matrix since, once those 3 redundant columns are collapsed to one,
only 6 numeric columns with sparse pairwise overlap remain — not dense enough
for a meaningful heatmap.

### Categorical Breakdowns
- `figures/04_top_countries`, `figures/05_craft_shapes`, `figures/07_trust_bands`,
  `figures/08_day_night`, `figures/09_agencies`,
  `figures/10_engagement_performance_rates`, `figures/11_timeliness_status`

---

## Recommendations for Further Analysis

### Immediate Actions
1. Drop `Unnamed: 0` and `Unnamed: 0.1` (legacy, non-identical index artifacts).
2. Pick one of the three witness-count columns (`witness_count_num` is the
   cleanest name) and drop the other two.
3. Null out the invalid `date_time.year` values (`0`, `2030`, `2040`) rather
   than treating them as real dates in any temporal analysis.

### Preprocessing Steps
- Cap or winsorize `witness_count_num` before using it in any model — the
  200-witness rows will dominate a mean/variance calculation for what is
  otherwise a median-1 field.
- Consider dropping columns >95% missing for any model-facing (not archival)
  export — a large majority of the 363 columns are effectively empty for a
  given row.
- Apply the existing `dup_cluster_id`/`dup_role` columns to collapse to
  canonical-only rows if the downstream use case needs one row per real-world
  event rather than one row per source mention (327 rows would drop).

### Analytical Approaches
- Given the extreme class imbalance and skew (1947 spike, witness-count
  outliers, 89%-vs-100% coverage gaps across fields), prefer robust/rank-based
  statistics (median, IQR, Spearman) over mean/Pearson for any cross-field
  analysis.
- Geospatial analysis should first assess actual `location.latitude`/
  `location.longitude` coverage (below the 50% threshold used in this report)
  before committing to a proximity-based method.

### Tools and Methods
- **Recommended software:** pandas/polars for tabular work already in use;
  no format conversion needed.
- **Statistical methods:** IQR-based outlier flagging (as used here) is
  adequate for a first pass, but is noisy on the strongly-peaked `date_time.year`
  distribution (297/1,661 rows flagged as "outliers" purely because the bulk of
  the data sits in a narrow 1947-centered band) — a domain-aware era-binning
  approach (as already used in `dedupe_all_reports.py`'s era figure) is more
  informative than a blanket IQR fence for this field.
- **Visualization tools:** matplotlib (already used throughout this session,
  see `figures/`), Okabe-Ito palette for colorblind accessibility.

---

## Data Processing Workflow

```
PURSUE_1_2_3_normalized_full.csv (1,791 rows)  ─┐
                                                  ├──► merge_and_dedup_pursue4.py
PURSUE_4_normalized.csv (82 rows)              ─┘         │
                                                            ▼
                                          PURSUE_1_2_3_4_merged.csv (this file)
                                                            │
                                              ┌─────────────┼─────────────┐
                                              ▼             ▼             ▼
                                    dedupe_all_reports.py  this EDA   PURSUE_corpus_report.md
                                    (figures/*.pdf/png)     report    (domain findings)
```
