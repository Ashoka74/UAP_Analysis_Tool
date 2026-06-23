import type {
  LoadDataResponse,
  AnalysisResponse,
  DashboardSummary,
  SchemaListResponse,
  SchemaMergeResponse,
  SchemaCoverageResponse,
  ParseUploadResponse,
  CostEstimate,
  ParseRunResponse,
  ScuCriteriaResponse,
  ScuNormalizeResponse,
  ScuFilterResponse,
  RagSearchResponse,
  CramersVResponse,
  ContingencyResponse,
  ColumnGroupsResponse,
  XgboostImportanceResponse,
} from '../types';

// API origin is configurable for split deployments (e.g. frontend on Vercel,
// backend on Hugging Face Spaces / Render). Set VITE_API_BASE to the backend
// origin at build time, e.g. "https://user-uap.hf.space". When unset it falls
// back to a same-origin "/api", which the Vite dev proxy (vite.config.ts) and
// a Vercel `/api` rewrite both handle transparently.
const API_ORIGIN = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '');
const BASE = `${API_ORIGIN}/api`;

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  loadData(type = 'west', rows = 15000): Promise<LoadDataResponse> {
    return request(`/data/load?type=${type}&rows=${rows}`);
  },

  uploadFile(file: File): Promise<LoadDataResponse> {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE}/data/upload`, { method: 'POST', body: form }).then(
      async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || `Upload failed: ${res.status}`);
        }
        return res.json();
      }
    );
  },

  filterData(
    filters: {
      column: string;
      type: string;
      values?: string[];
      min_val?: number;
      max_val?: number;
      pattern?: string;
    }[]
  ): Promise<LoadDataResponse> {
    return request('/data/filter', {
      method: 'POST',
      body: JSON.stringify(filters),
    });
  },

  getColumns(): Promise<{ columns: { name: string; dtype: string; unique: number; non_null: number }[] }> {
    return request('/data/columns');
  },

  getColumnValues(
    column: string,
    search = '',
    limit = 50
  ): Promise<{
    column: string;
    values: { value: string; count: number }[];
    total_matches: number;
  }> {
    return request(
      `/data/column-values?column=${encodeURIComponent(column)}&search=${encodeURIComponent(search)}&limit=${limit}`
    );
  },

  runAnalysis(
    columns: string[],
    opts: {
      enable_tfidf?: boolean;
      min_cluster_size?: number;
      n_neighbors?: number;
      min_dist?: number;
      top_n?: number;
    } = {}
  ): Promise<AnalysisResponse> {
    return request('/analyze/run', {
      method: 'POST',
      body: JSON.stringify({ columns, ...opts }),
    });
  },

  // ── Parsing ───────────────────────────────────────────────────────────
  getSchemas(): Promise<SchemaListResponse> {
    return request('/parse/schemas');
  },

  mergeSchema(
    labels: string[],
    customFields?: Record<string, unknown>
  ): Promise<SchemaMergeResponse> {
    return request('/parse/schema-merge', {
      method: 'POST',
      body: JSON.stringify({ labels, custom_fields: customFields ?? null }),
    });
  },

  schemaCoverage(
    labels: string[],
    columns: string[],
    customFields?: Record<string, unknown>
  ): Promise<SchemaCoverageResponse> {
    return request('/parse/schema-coverage', {
      method: 'POST',
      body: JSON.stringify({ labels, columns, custom_fields: customFields ?? null }),
    });
  },

  uploadParseFile(file: File): Promise<ParseUploadResponse> {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE}/parse/upload`, { method: 'POST', body: form }).then(
      async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || `Upload failed: ${res.status}`);
        }
        return res.json();
      }
    );
  },

  estimateParse(
    columns: string[],
    formatJson: string,
    model: string,
    useBatch = false
  ): Promise<CostEstimate> {
    return request('/parse/estimate', {
      method: 'POST',
      body: JSON.stringify({ columns, format_json: formatJson, model, use_batch: useBatch }),
    });
  },

  runParse(payload: {
    columns: string[];
    format_json: string;
    provider: string;
    model: string;
    api_key: string;
    max_workers?: number;
    keep_columns?: string[];
  }): Promise<ParseRunResponse> {
    return request('/parse/run', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  // ── SCU normalization ─────────────────────────────────────────────────
  getScuCriteria(): Promise<ScuCriteriaResponse> {
    return request('/scu/criteria');
  },

  scuNormalize(): Promise<ScuNormalizeResponse> {
    return request('/scu/normalize', { method: 'POST', body: '{}' });
  },

  scuFilter(criterionKeys: string[]): Promise<ScuFilterResponse> {
    return request('/scu/filter', {
      method: 'POST',
      body: JSON.stringify({ criterion_keys: criterionKeys }),
    });
  },

  // ── RAG search (Cohere) ───────────────────────────────────────────────
  ragSearch(
    columns: string[],
    question: string,
    cohereKey: string,
    topN = 50
  ): Promise<RagSearchResponse> {
    return request('/rag/search', {
      method: 'POST',
      body: JSON.stringify({ columns, question, cohere_key: cohereKey, top_n: topN }),
    });
  },

  // ── Cramér's V explorer ───────────────────────────────────────────────
  cramersV(payload: {
    columns?: string[] | null;
    drop_missing?: boolean;
    exclude_trivial?: boolean;
    strong_threshold?: number;
    high_threshold?: number;
    source?: string;
  }): Promise<CramersVResponse> {
    return request('/analysis/cramers-v', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  contingency(payload: {
    col1: string;
    col2: string;
    drop_missing?: boolean;
    source?: string;
  }): Promise<ContingencyResponse> {
    return request('/analysis/contingency', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  columnGroups(payload: {
    source?: string;
    high_threshold?: number;
  }): Promise<ColumnGroupsResponse> {
    return request('/analysis/column-groups', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  xgboostImportance(columns: string[], source = 'dataset'): Promise<XgboostImportanceResponse> {
    return request('/analysis/xgboost', {
      method: 'POST',
      body: JSON.stringify({ columns, source }),
    });
  },

  queryGemini(question: string, columns: string[], geminiKey: string): Promise<{ status: string; response: string; context_rows_used?: number; columns_used?: string[] }> {
    return request('/query/gemini', {
      method: 'POST',
      body: JSON.stringify({ question, columns, gemini_key: geminiKey }),
    });
  },

  getDashboardSummary(): Promise<DashboardSummary> {
    return request('/dashboard/summary');
  },

  healthCheck(): Promise<{ status: string; version: string }> {
    return request('/health');
  },
};
