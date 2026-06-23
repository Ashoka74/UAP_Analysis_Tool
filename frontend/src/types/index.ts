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

export interface XGBoostResult {
  feature_importance: Record<string, number>;
  accuracy: number;
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
  | 'clusters';

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

export interface ScuNormalizeResponse {
  status: string;
  metrics: {
    rows: number;
    scu_eligible: number;
    in_scu_window: number;
    has_credible_witness: number;
  };
  audit_markdown: string;
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
}

export interface CramersVResponse {
  labels: string[];
  matrix: (number | null)[][];
  pairs: { a: string; b: string; v: number }[];
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
  n: number;
}

export interface XgboostImportanceResponse {
  results: Record<string, XGBoostResult>;
  columns: string[];
  skipped: Record<string, string>;
  message?: string;
}
