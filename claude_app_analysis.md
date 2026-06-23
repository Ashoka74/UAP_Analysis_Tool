# UAP Analysis Tool — Improvement & Debugging Report

## 1. Critical Bugs

### 1.1 Variable Name Typo in Clustering (`uap_analyzer.py:316`)
`self.clusters_labels` is assigned instead of `self.cluster_labels`, creating a new attribute and silently breaking downstream cluster references.

### 1.2 Bare `except:` Clauses (`uap_analyzer.py:987-990, 792`)
Multiple bare `except:` blocks swallow all exceptions including `KeyboardInterrupt` and `SystemExit`. These should catch specific exceptions (`json.JSONDecodeError`, `ValueError`, etc.).

### 1.3 Race Condition in Concurrent Parsing (`uap_analyzer.py:966-978`)
`ThreadPoolExecutor(max_workers=32)` writes to a shared `self.responses` dict without any locking mechanism. This can corrupt data under load.

### 1.4 Division by Zero in Cramér's V (`uap_analyzer.py:758-761`)
```python
return np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))
```
If either dimension is 1, the denominator is 0. No guard exists.

---

## 2. Error Handling Gaps

| Area | Issue |
|------|-------|
| **GPU fallback** | No `try/except` around CUDA operations; crashes if GPU unavailable |
| **Empty DataFrames** | No validation before passing to UMAP, HDBSCAN, or XGBoost |
| **Single-valued columns** | Clustering and correlation break on columns with only one unique value |
| **API timeouts** | No connection/read timeouts on Gemini, OpenAI, or INTERMAGNET calls |
| **HDF5 loading** | Backend assumes the HDF5 file exists at a fixed path with no fallback |
| **File uploads** | No size limits enforced; no validation of CSV/Excel structure |
| **Regex injection** | User text input passed directly to `str.contains()` without `re.escape()` |

---

## 3. Architecture & Code Quality

### 3.1 God Object Anti-Pattern
`UAPAnalyzer` handles embedding, dimensionality reduction, clustering, TF-IDF naming, XGBoost classification, and Cramér's V correlation all in one class. Consider splitting into:
- `EmbeddingService`
- `ClusteringPipeline`
- `StatisticalAnalyzer`
- `FeatureImportanceAnalyzer`

### 3.2 Duplicate Methods
`merge_similar_clusters()` and `merge_similar_clusters2()` (lines 263-356) are near-identical. Consolidate into a single parameterized method.

### 3.3 In-Memory State Management (backend/main.py)
The backend stores all session state in a plain Python dict. This means:
- No persistence across server restarts
- No multi-user isolation
- Memory grows unbounded with concurrent sessions

### 3.4 Hardcoded Data Truncation (`streamlit_uap_clean.py:239`)
`.head(10000)` silently drops rows beyond 10k. Users are not warned about data loss.

### 3.5 Wildcard CORS (`backend/main.py:30`)
`allow_origins=["*"]` allows any domain to call the API. Should be restricted to the frontend origin.

---

## 4. Data Analysis Feature Improvements

### 4.1 Temporal Analysis
- **Time-series decomposition**: Detect seasonal and trend components in sighting frequency over time (e.g., monthly/yearly cycles).
- **Change-point detection**: Identify statistically significant shifts in sighting patterns using algorithms like PELT or Bayesian Online Change Point Detection.
- **Temporal clustering**: Group sightings by time windows and compare feature distributions across eras.

### 4.2 Enhanced Geospatial Analysis
- **Spatial autocorrelation** (Moran's I): Quantify whether sightings cluster geographically beyond random chance.
- **Kernel density estimation**: Generate continuous heatmaps instead of discrete point maps.
- **Proximity analysis**: Correlate sighting density with distance to military bases, nuclear plants, airports, and flight corridors.
- **Voronoi tessellation**: Partition geography into regions of influence per cluster.

### 4.3 Advanced Clustering
- **Silhouette score / Davies-Bouldin index**: Automatically evaluate cluster quality and suggest optimal `min_cluster_size`.
- **Hierarchical HDBSCAN tree**: Expose the cluster hierarchy for interactive drill-down.
- **Ensemble clustering**: Combine HDBSCAN + KMeans + spectral clustering via consensus for more robust assignments.
- **Outlier analysis**: Surface and profile noise points (HDBSCAN label `-1`) instead of discarding them.

### 4.4 Natural Language & Text Mining
- **Topic modeling** (BERTopic / LDA): Extract latent themes from witness descriptions beyond TF-IDF keywords.
- **Sentiment analysis**: Score witness reports for emotional intensity, fear, certainty, etc.
- **Named entity extraction**: Pull out specific aircraft types, locations, agencies, and dates from free text.
- **Cross-report similarity network**: Build a graph of similar reports and detect communities.

### 4.5 Statistical Rigor
- **Multiple hypothesis correction**: Apply Bonferroni or FDR correction to chi-squared tests across many column pairs.
- **Effect size reporting**: Report Cohen's w alongside p-values for contingency tests.
- **Confidence intervals**: Add bootstrap CIs to feature importance scores and cluster statistics.
- **Bayesian alternatives**: Offer Bayesian correlation and classification as alternatives to frequentist methods.

### 4.6 Interactive Exploration
- **Linked brushing**: Selecting points on a scatter plot should filter the data table, map, and histograms simultaneously.
- **Drill-down from clusters**: Click a cluster to view its members, top features, and representative reports.
- **Comparison mode**: Side-by-side analysis of two clusters or two time periods.
- **Custom derived columns**: Let users create calculated fields (e.g., `duration_minutes / distance_km`).

### 4.7 Export & Reporting
- **PDF/HTML report generation**: One-click export of the full analysis pipeline with charts and summary text.
- **Reproducibility logs**: Record all parameter choices (UMAP neighbors, cluster size, etc.) so analyses can be replicated.
- **Data export**: Export filtered/clustered data as CSV with cluster labels and embeddings.

---

## 5. Performance Improvements

| Bottleneck | Current | Suggested |
|------------|---------|-----------|
| Embedding computation | CPU fallback is slow for >5k rows | Batch with `encode(batch_size=256)`, cache embeddings to disk |
| UMAP on large datasets | O(n log n), no progress feedback | Use `umap.parametric_umap` or pre-reduce with PCA to 50 dims first |
| XGBoost training | Single-threaded default | Set `nthread=-1`, use `early_stopping_rounds` |
| TF-IDF vectorization | Rebuilds on every run | Cache vectorizer and matrix in session state |
| HDF5 loading | Loads full 1.8GB file into memory | Use `pd.read_hdf()` with `where` clause for lazy loading |
| Frontend re-renders | Full data sent on every filter | Implement server-side pagination and send only visible rows |

---

## 6. Testing (Currently Zero Coverage)

### Priority Test Targets
1. **Clustering pipeline**: empty input, single row, all-null columns, single-valued features
2. **API endpoints**: request validation, error responses, concurrent requests
3. **Statistical functions**: known-answer tests for Cramér's V, chi-squared, feature importance
4. **Data loading**: corrupted files, missing columns, encoding issues, oversized uploads
5. **Frontend components**: render tests, API error states, filter interactions

### Suggested Setup
- `pytest` + `pytest-cov` for backend
- `vitest` + `@testing-library/react` for frontend
- CI pipeline via GitHub Actions

---

## 7. Security

- **API keys**: Entered via text input and stored in session state in plaintext. Use environment variables or a secrets manager.
- **CORS**: Wildcard `*` origin should be replaced with explicit frontend URL.
- **Input sanitization**: User-provided regex and column names should be escaped before use in queries.
- **Rate limiting**: No rate limiting on API endpoints; vulnerable to abuse.
- **Dependency pinning**: All requirements use `>=` with no upper bounds, risking breaking changes on install.

---

## 8. Dependency Cleanup

| Package | Status |
|---------|--------|
| `st-paywall>=0.1.8` | Unused — remove |
| `cohere>=5.5.8` | Imported but never called — remove or integrate |
| `protobuf>=4.25.3` | Transitive dependency conflict risk — pin version |
| `sentence_transformers` | Two different models loaded (`all-mpnet-base-v2` and `e5-large-v2`) — standardize on one |

---

## 9. Logging & Observability

- Replace all `print()` statements with Python `logging` module.
- Add structured logging with context (user session, operation, duration).
- Instrument key operations (embedding time, clustering time, API latency) with timing metrics.
- Add health check endpoint that validates dependencies (GPU, model files, HDF5 availability).

---

## Summary: Top 10 Action Items

| # | Priority | Action |
|---|----------|--------|
| 1 | Critical | Fix `clusters_labels` typo → `cluster_labels` |
| 2 | Critical | Replace bare `except:` with specific exception types |
| 3 | Critical | Add thread-safe locking for concurrent parsing |
| 4 | High | Add division-by-zero guard in Cramér's V |
| 5 | High | Restrict CORS to frontend origin |
| 6 | High | Add unit tests for core pipeline |
| 7 | Medium | Split `UAPAnalyzer` into focused services |
| 8 | Medium | Implement temporal and geospatial analysis features |
| 9 | Medium | Add proper logging and performance instrumentation |
| 10 | Low | Clean up unused dependencies |
