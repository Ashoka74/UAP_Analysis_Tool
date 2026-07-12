export interface ColumnInfo {
  name: string;
  dtype: string;
  unique: number;
  non_null: number;
}

export interface ColumnStat {
  name: string;
  dtype: string;
  non_null: number;
  unique: number;
  min?: number;
  max?: number;
  mean?: number;
  top_values?: { value: string; count: number }[];
}

export interface DataResponse {
  columns: string[];
  rows: Record<string, unknown>[];
  // Stable per-row id aligned with `rows` (DataFrame index label), used to overlay
  // model-imputed fills onto the right cells in the Data Explorer.
  row_ids?: string[];
  total_rows: number;
  returned_rows: number;
}

export interface LoadDataResponse {
  status: string;
  data: DataResponse;
  column_stats: ColumnStat[];
}

export interface ClusterTrace {
  name: string;
  x: number[];
  y: number[];
  text: string[];
  count: number;
}

export interface ClusterViz {
  traces: ClusterTrace[];
  title: string;
}

export interface DistributionItem {
  label: string;
  count: number;
}

export interface ColumnAnalysis {
  cluster_count: number;
  distribution: DistributionItem[];
  total_points: number;
}

export interface MultimodalResult {
  source_type: string;
  parent_id: string;
  source_id: string | null;
  similarity: number | null;
  start_seconds: number | null;
  end_seconds: number | null;
  page: number | null;
  embedded_text: string | null;
  source_url: string;
  media_embed_url: string | null;
  release: string | null;
  release_date: string | null;
  chunk_matches: number;
}

export interface MultimodalSearchResponse {
  status: string;
  n_results: number;
  results: MultimodalResult[];
}

export interface XGBoostResult {
  feature_importance: Record<string, number>;
  accuracy: number;
  // Stratified k-fold cross-validation accuracy (optional — present on the
  // direct /api/analysis/xgboost path, absent on the cluster pipeline).
  cv_mean?: number;
  cv_std?: number;
  cv_folds?: number;
  // Number of features the model was fit on (present on the PCA 2nd pass, where
  // redundancy clusters are collapsed so the count drops vs the raw 1st pass).
  n_features?: number;
  // Opt-in significance: permutation-null empirical p-value and bootstrap
  // selection frequency per feature (only present when requested).
  null_p?: Record<string, number>;
  selection_freq?: Record<string, number>;
}

export interface CramersVData {
  labels: string[];
  matrix: number[][];
}

export interface AnalysisResponse {
  status: string;
  analysis_mode?: string;
  mock_mode?: boolean;
  warnings?: string[];
  results: Record<string, ColumnAnalysis>;
  cluster_viz: Record<string, ClusterViz>;
  cramers_v: CramersVData | null;
  xgboost: Record<string, XGBoostResult>;
  processed_data: DataResponse;
}

export interface DashboardSummary {
  loaded: boolean;
  total_rows: number;
  total_columns: number;
  columns?: string[];
  dtypes?: Record<string, string>;
  analyzed: boolean;
  analyzed_columns: number;
  analysis_runs?: number;
  last_analysis_at?: string | null;
  analysis_mode?: string;
  null_counts?: Record<string, number>;
  memory_mb?: number;
}

export type PageId =
  | 'dashboard'
  | 'data'
  | 'parsing'
  | 'analysis'
  | 'query'
  | 'rag'
  | 'scu'
  | 'map'
  | 'magnetic'
  | 'clusters'
  | 'dedup';

// ── Parsing ────────────────────────────────────────────────────────────────
export interface SchemaListResponse {
  labels: string[];
  groups: Record<string, string[]>;
  models: { openai: string[]; deepseek: string[] };
}

export interface SchemaField {
  path: string;
  description: string;
}

export interface SchemaMergeResponse {
  schema: Record<string, unknown>;
  schema_json: string;
  fields: SchemaField[];
  extract_key: string | null;
}

export interface ParseUploadResponse {
  status: string;
  filename: string;
  data: DataResponse;
  columns: string[];
  total_rows: number;
}

export type CoverageMode = 'all' | 'missing' | 'database';

export interface SchemaCoverageField {
  path: string;
  leaf: string;
  present: boolean;
  matched_column: string | null;
}

export interface SchemaCoverageResponse {
  coverage: SchemaCoverageField[];
  summary: { present: number; missing: number; total: number; db_only: number };
  matched_columns: string[];
  db_only_columns: string[];
  variants: Record<CoverageMode, { schema_json: string; n_fields: number }>;
}

export interface CostEstimate {
  total_usd: number;
  cost_input_usd: number;
  cost_cached_usd: number;
  cost_output_usd: number;
  total_input_tokens: number;
  output_tokens: number;
  prefix_tokens: number;
  avg_desc_tokens: number;
  batch_discount_pct: number;
  model: string;
  provider: string;
  n_queries: number;
}

export interface ParseRunResponse {
  status: string;
  n_ok: number;
  n_total: number;
  n_failed: number;
  errors: string[];
  // Set when the run aborted on an unfunded/invalid account (no quota, bad key,
  // billing) instead of failing row-by-row.
  fatal?: boolean;
  fatal_message?: string | null;
  data: DataResponse;
}

// ── SCU ────────────────────────────────────────────────────────────────────
export interface ScuCriterion {
  key: string;
  column: string;
  label: string;
}

export interface ScuCriteriaResponse {
  criteria: ScuCriterion[];
  extra_criteria: ScuCriterion[];
  presets: Record<string, string[]>;
}

// Input-column mapping resolved by the normalizer (exact/suffix/leaf/fuzzy/manual)
// so schemas that nest fields differently (e.g. MasterSCU_v1) still feed the gate.
export interface ScuColumnMapping {
  resolved: Record<string, string>;   // canonical -> actual dataframe column
  methods: Record<string, string>;    // canonical -> how it matched
  scores: Record<string, number>;     // canonical -> match confidence 0..1
  unmatched: string[];                // canonical inputs with no source column
  expected: string[];                 // all canonical inputs the gate reads
  actual_columns: string[];           // columns present in the raw input
}

export interface ScuNormalizeResponse {
  status: string;
  metrics: {
    rows: number;
    scu_eligible: number;
    in_scu_window: number;
    has_credible_witness: number;
  };
  audit_markdown: string;
  mapping?: ScuColumnMapping;
  data: DataResponse;
}

export interface ScuFilterResponse {
  status: string;
  funnel: { stage: string; count: number }[];
  n_passed: number;
  data: DataResponse;
}

// ── RAG ────────────────────────────────────────────────────────────────────
export interface RagSearchResponse {
  status: string;
  n_results: number;
  searched_columns: string[];
  data: DataResponse;
}

// ── Cramér's V explorer ────────────────────────────────────────────────────
export interface ColumnGroup {
  parent: string;
  columns: string[];
  leaves: string[];
  nested: boolean;
}

export interface ColumnGroupsResponse {
  eligible: string[];
  groups: ColumnGroup[];
  bands: Record<string, string[]>;
  nunique: Record<string, number>;
  // Known exact semantic duplicates (unit twins / anomaly.* re-encodings of
  // engagement types) excluded from the default selection to avoid diluting
  // XGBoost gain — still present in `groups` for manual re-selection.
  semantic_duplicates_removed?: { kept: string; dropped: string[]; reason: string }[];
}

export interface CramersVResponse {
  labels: string[];
  matrix: (number | null)[][];
  pairs: {
    a: string; b: string; v: number; ci?: [number, number];
    p?: number;            // raw association-test p-value
    q?: number;            // Benjamini–Hochberg FDR-adjusted (across all pair tests)
    test?: 'chi2' | 'fisher';
    sparse?: boolean;      // Cochran rule violated — V and p unreliable
  }[];
  n_tests?: number;        // pair tests entering the FDR correction
  n_sparse?: number;
  fdr_method?: string;
  n_excluded: number;
  high_correlation_columns: string[];
  bands: Record<string, string[]>;
  nunique: Record<string, number>;
  selected_columns: string[];
  groups: ColumnGroup[];
}

export interface ContingencyResponse {
  row_labels: string[];
  col_labels: string[];
  matrix: number[][];
  v: number;
  ci?: [number, number] | null;   // 95% bootstrap CI on Cramér's V
  p?: number | null;              // association-test p-value (chi2 / fisher)
  test?: 'chi2' | 'fisher';
  sparse?: boolean;               // Cochran rule violated (expected<5 in >20% of cells)
  sparse_frac?: number;
  n: number;
}

// Conditional-association drill-down: does A–B survive conditioning on Z?
export interface ConditionalResponse {
  c1: string;
  c2: string;
  condition_on: string;
  marginal_v: number;
  mean_conditional_v: number | null;
  strata: { level: string; n: number; v: number }[];
  n_strata_used: number;
  n_strata_dropped: number;
  n: number;
  test: {
    method: string | null;
    statistic: number | null;
    dof: number | null;
    p_value: number | null;
    pooled_odds_ratio?: number;
  };
  verdict: 'persists' | 'attenuated' | 'explained_by_z' | 'weak_or_absent' | 'inconclusive';
}

export interface XgboostImportanceResponse {
  results: Record<string, XGBoostResult>;
  columns: string[];
  skipped: Record<string, string>;
  message?: string;
}

// 2nd pass: each Cramér's V redundancy cluster collapsed into one PCA latent index.
export interface LatentCluster {
  index_name: string;            // e.g. "PCA[coded_radar]" — the key used in importances
  root: string;                  // most central member (the "root predictor")
  members: string[];
  explained_variance: number;    // variance share captured by the single component
  loadings: { feature: string; weight: number }[]; // per-member contribution share
}

export interface XgboostPcaResponse {
  first_pass: Record<string, XGBoostResult>;
  second_pass: Record<string, XGBoostResult>;
  clusters: LatentCluster[];
  columns: string[];
  skipped: Record<string, string>;
  strong_threshold: number;
  n_clusters: number;
  pruned: string[];
  message?: string | null;
}

// Model-based imputation: predict each column's missing cells from the others.
export interface ImputationResult {
  accuracy: number | null;       // model accuracy used to colour the fills (null = single observed class)
  cv_mean?: number;
  cv_std?: number;
  cv_folds?: number;
  n_missing: number;
  n_features: number;
  n_imputations: number;         // M draws (>1 = multiple imputation)
  mean_conf: number | null;      // mean per-cell agreement across the M draws
  predictions: { value: string; count: number }[]; // predicted-fill distribution
  // Row-level predictions; conf = per-cell agreement across draws (null = single imputation).
  sample: { row: string; value: string; conf: number | null }[];
  sample_truncated: boolean;
}

export interface XgboostImputeResponse {
  results: Record<string, ImputationResult>;
  columns: string[];
  skipped: Record<string, string>;
  total_missing: number;
  used_pca: boolean;
  n_imputations: number;
  message?: string | null;
}

// ── Deduplication Studio ───────────────────────────────────────────────────
export interface SimpleSimilarityResponse {
  similarity_score: number;
  is_similar: boolean;
  model?: string;
  reason?: string;
}

export interface SimpleDuplicateResponse {
  is_duplicate: boolean;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  reasons: string[];
  metrics: {
    similarity_score: number;
    haversine_km: number | null;
    date_diff_days: number | null;
  };
}

export interface FlaggedPair {
  id_a: string;
  id_b: string;
  similarity: number;
  llm_same_event: boolean;
  llm_reason: string;
}

export interface AdvancedDedupResponse {
  status: string;
  parameters: {
    threshold: number;
    date_diff_days: number;
    max_km: number;
    use_llm_judge: boolean;
  };
  summary: {
    total_clusters: number;
    rows_in_clusters: number;
    redundant_rows_saved: number;
    flagged_pairs_count: number;
  };
  flagged_pairs: FlaggedPair[];
}

export interface CrossDbPair {
  id_a: string;
  id_b: string;
  similarity: number;
  bin: 'exact_duplicate' | 'strong_similar' | 'moderate_similar' | 'distinct';
  haversine_km: number | null;
  date_diff_days: number | null;
  text_a_preview: string;
  text_b_preview: string;
  row_a?: Record<string, any>;
  row_b?: Record<string, any>;
  flags: {
    is_similar_text: boolean;
    is_similar_date: boolean;
    is_similar_location: boolean;
    is_similar_both: boolean;
    is_similar_all: boolean;
  };
}

export interface CrossDbPipelineResponse {
  status: string;
  mode: string;
  summary: {
    total_pairs_evaluated: number;
    bins: {
      exact_duplicate: number;
      strong_similar: number;
      moderate_similar: number;
      distinct: number;
    };
    gate_counts: {
      similar_text: number;
      similar_date: number;
      similar_location: number;
      similar_both: number;
      similar_all: number;
    };
  };
  pairs: CrossDbPair[];
}

