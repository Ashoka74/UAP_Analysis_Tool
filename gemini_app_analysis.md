# UAP Analysis App: Gemini's Code Review and Improvement Plan

Below is an analysis identifying potential improvements, debugging areas, and feature additions for the current UAP Analysis Application, with a strong focus on data-analysis pipelines and UI responsiveness.

---

## 1. Machine Learning & Modeling Improvements

### 1.1 Inconsistent Embedding Models
**Issue**: In `uap_analyzer.py`, `SentenceTransformer` is initialized twice with two different models: `all-mpnet-base-v2` at the global level and `embaas/sentence-transformers-e5-large-v2` inside the `UAPAnalyzer` `__init__` method. 
**Improvement**: Standardize on a single embedding model to prevent memory bloat on the GPU and ensure consistent vector space representations. Expose the model choice as a configurable parameter.

### 1.2 Hardcoded Dimensionality Reduction and Clustering Params
**Issue**: In `streamlit_uap_clean.py`, the UMAP parameters (`n_neighbors=15`, `min_dist=0.1`) and HDBSCAN parameters (`min_cluster_size=15`) are entirely hardcoded.
**Improvement**: For a true Data-Analysis EDA tool, these hyperparameters should be exposed to the user via Streamlit sliders. Different distributions of UAP textual data require tuning these parameters to find meaningful clusters versus noise.

### 1.3 XGBoost Overfitting & Explainability
**Issue**: The XenBoost classification in `train_xgboost` is strictly run with standard parameters (`max_depth=6`, `eta=0.3`, `num_round=100`) without any hyperparameter tuning or cross-validation. Furthermore, feature importance is just calculated globally via `plot_importance`.
**Improvement**: 
- **SHAP values**: Add SHAP (SHapley Additive exPlanations) for local and global model explainability. It provides clear insights into *why* a UAP sighting was categorized into a specific cluster based on other columns.
- **Cross-Validation**: Implement K-Fold cross-validation to guarantee the accuracy score isn't a fluke based on the random `train_test_split`.

### 1.4 Better Topic Modeling
**Issue**: Cluster naming is done using a simple TF-IDF approach taking the most frequent terms. This often results in generic or somewhat disconnected hyphenated labels.
**Improvement**: Integrate **BERTopic**, which handles embedding, dimensionality reduction, clustering, and topic representation (using c-TF-IDF or LLMs) under the hood much more cohesively.

---

## 2. Statistical Robustness

### 2.1 Cramer's V Division by Zero
**Bug**: In `cramers_v(confusion_matrix)`, the formula returns `np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))`. If `k_corr` or `r_corr` equals 1, this causes a `ZeroDivisionError` or numpy runtime warning for division by zero.
**Fix**: Add a check: if `min((k_corr-1), (r_corr-1)) == 0`, simply return `0.0`.

### 2.2 Chi-Square Test Validity
**Issue**: Using `chi2_contingency` on tables with very small frequencies (such as sparse clusters of UAP sightings) violates the test's assumption that the expected frequency in each cell should be >= 5.
**Fix**: Provide visual warnings if the contingency table is too sparse. Consider implementing Fisher's Exact Test as a fallback for small sample sizes, or combine minor clusters before testing.

---

## 3. App Architecture & Performance

### 3.1 Hardcoded Data Limits
**Issue**: `parsed_files_distance_embeds.h5` is aggressively truncated: `.head(10000)` without informing the user.
**Improvement**: Allow users to toggle the sample size, or implement lazy loading. If computational limits are an issue, implement chunked processing or Dask DataFrames.

### 3.2 Lack of Caching for Heavy ML Tasks
**Issue**: Operations like UMAP and embedding creation are computationally expensive and will lock up the Streamlit UI on every state change. 
**Fix**: Wrap the heavy ML processing functions (like `analyze_and_predict` and `preprocess_data`) in `@st.cache_data` or `@st.cache_resource`. This ensures that if the user makes a minor UI change, the embeddings and XGBoost models don't retrain completely from scratch.

### 3.3 State Management Bottlenecks
**Issue**: `streamlit_uap_clean.py` stores large DataFrames and multiple analyzer objects directly into `st.session_state`. This consumes massive amounts of RAM and slows down page re-rendering.
**Fix**: Clear large intermediate objects from the state when moving to the next stage, keeping only the reduced data or metrics needed for plotting.

---

## 4. Advanced Data-Analysis Features to Add

### 4.1 Spatiotemporal Analysis
UAP data is inherently tied to *where* and *when*. 
- **Time-Series Forecasting**: Add visualizations to plot sighting frequency over time. Run anomaly detection to highlight "waves" or spikes in reports.
- **Geospatial Mapping**: If coordinates/locations exist in the data, use `folium` or `plotly.express.scatter_map` to visualize UAP hotspots and automatically cluster them geographically.

### 4.2 LLM Token Limit Crash (Gemini Integration)
**Bug**: `gemini_query` directly joins all passed data into one big prompt: `context = '\n'.join(filtered)`. If the user has thousands of rows selected, this will easily exceed the token limit of Gemini and crash the app with an API error.
**Fix**: Implement token counting and truncation, or construct a Retrieval-Augmented Generation (RAG) approach that only sends the top-k most relevant rows to Gemini for summarization.

### 4.3 Advanced Filtering 
**Issue**: The current filter function `filter_dataframe` is powerful but basic. 
**Improvement**: Add the capability to filter by specific ML clusters created dynamically. Allow users to filter the dataset strictly to the instances the XGBoost model confidently predicted or misclassified (predictive error analysis).
