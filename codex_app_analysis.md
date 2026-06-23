# UAP Analysis App: Improvement and Debugging Notes

## Scope Reviewed
- Backend API: `backend/main.py`
- Frontend app/store/pages: `frontend/src/**`
- Project positioning/features: `README.md`

## High-Priority Debugging and Reliability Issues

### 1) Global in-memory backend state is shared across all users/sessions
- File: `backend/main.py`
- Current behavior: `state` is a single process-level dictionary storing dataset, filtered data, analysis results, and query context.
- Risk:
  - Cross-user data leakage.
  - Race conditions (one user can overwrite another user’s data mid-session).
  - Non-deterministic behavior in multi-worker deployment.
- Improvement:
  - Introduce per-session/project IDs and isolate state in Redis or a DB cache keyed by session.
  - Add dataset ownership + TTL cleanup.

### 2) Dashboard “Analysis Runs” counter is effectively broken
- File: `backend/main.py`
- Current behavior: `/api/dashboard/summary` reports `analyzed_columns` from `state["col_names"]`, but `col_names` is never populated in `run_analysis`.
- Impact: dashboard can show incorrect/always-zero run stats.
- Improvement:
  - Set `state["col_names"] = req.columns` (validated columns only) during analysis.
  - Consider storing analysis timestamp and run count.

### 3) Numeric filtering ignores partial ranges
- Files: `backend/main.py`, `frontend/src/components/data/FilterPanel.tsx`
- Current behavior: numeric filter only applies when both `min_val` and `max_val` are provided.
- Impact: user-provided minimum-only or maximum-only constraints do nothing.
- Improvement:
  - Support all combinations: min-only, max-only, and bounded range.

### 4) Filtering error handling is silent on frontend
- File: `frontend/src/components/data/DataExplorer.tsx`
- Current behavior: `handleFilter` catches errors and suppresses them.
- Impact: user cannot tell whether filtering failed or returned no matches.
- Improvement:
  - Surface backend error in UI (same pattern used by upload/load errors).

### 5) Backend query endpoint has expensive prompt construction and token-risk
- File: `backend/main.py`
- Current behavior: concatenates up to 500 rows of full text into one prompt string.
- Risks:
  - Large latency/cost spikes.
  - Prompt truncation or model failure on long text columns.
- Improvement:
  - Add token-aware chunking/sampling and optional map-reduce summarization.
  - Enforce max characters/tokens per request with clear user feedback.

### 6) CORS config is overly permissive and potentially invalid for credentials
- File: `backend/main.py`
- Current behavior: `allow_origins=["*"]` with `allow_credentials=True`.
- Risk: browser credential behavior can be inconsistent; security posture is weak for production.
- Improvement:
  - Use explicit origin allowlist per environment.
  - Keep credentials disabled unless required.

## Data-Analysis Quality Gaps (Core Product)

### 7) Analysis pipeline is currently mock/simulated, not aligned with README claims
- File: `backend/main.py`
- Current behavior: analysis uses value counts + random 2D points + correlation proxy for “XGBoost-like” output.
- Impact: users may interpret synthetic outputs as real model outputs.
- Improvement:
  - Label this mode explicitly as `demo/mock` in API/UI.
  - Add a production pipeline path using real embeddings + UMAP/HDBSCAN + trained model artifacts.

### 8) Cluster assignment is too coarse for high-cardinality text fields
- File: `backend/main.py`
- Current behavior: top 32 frequent values become “clusters”; all others mapped to `Other`.
- Impact: weak signal extraction for long-tail UAP narratives.
- Improvement:
  - Use embedding-based similarity clustering with min-cluster-size tuning.
  - Keep top terms as labels only, not as cluster definitions.

### 9) “XGBoost” results are correlation-based placeholders
- File: `backend/main.py`
- Current behavior: feature importance derived from absolute correlation among category codes, with random “accuracy”.
- Impact: misleading ML interpretation.
- Improvement:
  - Either rename section (`Association Importance`) or run real train/validation with metrics and confidence intervals.

### 10) Cramer’s V stability safeguards are minimal
- File: `backend/main.py`
- Current behavior: exceptions are swallowed to `0.0` values.
- Impact: matrix can hide data-quality problems.
- Improvement:
  - Return diagnostics (insufficient contingency size, sparse table warning, low sample counts).

## UX and Feature Improvements for Analysis Workflows

### 11) Add reproducibility controls
- Current gap: random projections are generated without surfaced seed controls; pipeline details are hidden.
- Improvement:
  - UI inputs for random seed and analysis config profile.
  - Persist configuration alongside results.

### 12) Add time/location-first analysis modules
- Context: UAP datasets are usually spatiotemporal.
- Improvement ideas:
  - Temporal anomaly detection (daily/weekly trend breaks).
  - Geo heatmaps + hotspot evolution over time.
  - Co-occurrence matrices for shape/light/motion features.

### 13) Add model/result provenance panel
- Improvement:
  - Track dataset hash, row count after filters, analysis timestamp, selected columns, pipeline version.
  - Show this metadata in Analysis and export payloads.

### 14) Improve filter capabilities for real EDA
- Current gap: categorical filter relies on top-values list and cannot easily search rare categories.
- Improvement:
  - Add searchable categorical picker and “include nulls/exclude nulls”.
  - Add reusable saved filter presets.

### 15) Add export/reporting features
- Improvement:
  - Export filtered dataset, correlation matrix, and feature-importance JSON/CSV.
  - One-click markdown/PDF report with charts and configuration metadata.

## Engineering Quality and Maintainability

### 16) Add automated tests for core API behaviors
- Suggested minimal suite:
  - `/api/data/load`, `/api/data/filter`, `/api/analyze/run`, `/api/dashboard/summary`, `/api/query/gemini` failure paths.
  - Numeric filter edge cases (min-only/max-only).
  - State isolation once sessionized.

### 17) Add request/analysis observability
- Improvement:
  - Structured logging + request IDs.
  - Timing metrics per stage (load/filter/analyze/query).
  - Distinguish user errors (4xx) from pipeline errors (5xx).

### 18) Clarify mode separation: demo vs production
- Improvement:
  - Feature flags/environment variable to select mock vs full analysis backend.
  - UI badges and warning copy to prevent scientific misinterpretation.

## Suggested Implementation Order
1. Fix state isolation and dashboard run-count correctness.
2. Fix filtering behavior + frontend error surfacing.
3. Mark current analysis mode as mock and rename misleading outputs.
4. Add reproducibility/provenance metadata and exports.
5. Introduce real embedding + clustering + model pipeline behind a feature flag.
