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
  null_counts?: Record<string, number>;
  memory_mb?: number;
}

export type PageId = 'dashboard' | 'data' | 'analysis' | 'query';
