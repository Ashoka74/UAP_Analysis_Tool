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
  MultimodalSearchResponse,
  CramersVResponse,
  ContingencyResponse,
  ConditionalResponse,
  ColumnGroupsResponse,
  XgboostImportanceResponse,
  XgboostPcaResponse,
  XgboostImputeResponse,
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
      start?: string;
      end?: string;
      drop_null?: boolean;
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
      enable_llm?: boolean;
      llm_sample_size?: number;
      llm_max_words?: number;
      llm_provider?: string;
      llm_model?: string;
      llm_api_key?: string;
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

  // Normalize an uploaded, already-structured dataset (parsed CSV/XLSX or a
  // parsed_responses JSON) — no prior LLM parse required.
  scuNormalizeUpload(file: File): Promise<ScuNormalizeResponse> {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE}/scu/normalize-upload`, { method: 'POST', body: form }).then(
      async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || `Upload failed: ${res.status}`);
        }
        return res.json();
      }
    );
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

  // ── Multimodal RAG (Neon + pgvector) ──────────────────────────────────
  multimodalReleases(databaseUrl: string): Promise<{ releases: string[] }> {
    return request('/rag/multimodal/releases', {
      method: 'POST',
      body: JSON.stringify({ database_url: databaseUrl }),
    });
  },

  multimodalSearch(p: {
    databaseUrl: string; geminiKey: string; query: string;
    sourceType?: string; release?: string; limit?: number; threshold?: number; groupByParent?: boolean;
  }): Promise<MultimodalSearchResponse> {
    return request('/rag/multimodal/search', {
      method: 'POST',
      body: JSON.stringify({
        database_url: p.databaseUrl, gemini_key: p.geminiKey, query: p.query,
        source_type: p.sourceType, release: p.release, limit: p.limit, threshold: p.threshold,
        group_by_parent: p.groupByParent,
      }),
    });
  },

  multimodalSearchImage(p: {
    file: File; databaseUrl: string; geminiKey: string;
    sourceType?: string; release?: string; limit?: number; threshold?: number; groupByParent?: boolean;
  }): Promise<MultimodalSearchResponse> {
    const form = new FormData();
    form.append('file', p.file);
    form.append('database_url', p.databaseUrl);
    form.append('gemini_key', p.geminiKey);
    if (p.sourceType) form.append('source_type', p.sourceType);
    if (p.release) form.append('release', p.release);
    form.append('limit', String(p.limit ?? 12));
    form.append('threshold', String(p.threshold ?? 0.2));
    form.append('group_by_parent', String(p.groupByParent ?? true));
    return fetch(`${BASE}/rag/multimodal/search-image`, { method: 'POST', body: form }).then(
      async (res) => {
        if (!res.ok) {
          const b = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(b.detail || `Image search failed: ${res.status}`);
        }
        return res.json();
      }
    );
  },

  // ── Cramér's V explorer ───────────────────────────────────────────────
  cramersV(payload: {
    columns?: string[] | null;
    drop_missing?: boolean;
    exclude_trivial?: boolean;
    strong_threshold?: number;
    high_threshold?: number;
    source?: string;
    ci_top_n?: number;            // bootstrap 95% CI for the top-N strongest pairs
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

  // Does the col1–col2 association survive conditioning on a third field?
  conditional(payload: {
    col1: string;
    col2: string;
    condition_on: string;
    drop_missing?: boolean;
    source?: string;
  }): Promise<ConditionalResponse> {
    return request('/analysis/conditional', {
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

  xgboostImportance(
    columns: string[],
    source = 'dataset',
    withCv = true,
    withSignificance = false,
  ): Promise<XgboostImportanceResponse> {
    return request('/analysis/xgboost', {
      method: 'POST',
      body: JSON.stringify({ columns, source, with_cv: withCv, with_significance: withSignificance }),
    });
  },

  // 2nd pass — collapse Cramér's V redundancy clusters into PCA latent indices,
  // then re-fit XGBoost. Returns both passes + the latent-index definitions.
  xgboostPca(
    columns: string[],
    source = 'dataset',
    withCv = true,
    strongThreshold = 0.3,
    pruneUninformative = false,
    withSignificance = false,
  ): Promise<XgboostPcaResponse> {
    return request('/analysis/xgboost-pca', {
      method: 'POST',
      body: JSON.stringify({
        columns, source, with_cv: withCv,
        strong_threshold: strongThreshold, prune_uninformative: pruneUninformative,
        with_significance: withSignificance,
      }),
    });
  },

  // Predict (impute) each selected column's missing values from the others; the
  // per-column model accuracy colours the fills in the UI. usePca swaps in the
  // 2nd-pass PCA latent-index predictor space.
  xgboostImpute(
    columns: string[],
    source = 'dataset',
    withCv = true,
    usePca = false,
    strongThreshold = 0.3,
    nImputations = 1,
  ): Promise<XgboostImputeResponse> {
    return request('/analysis/xgboost-impute', {
      method: 'POST',
      body: JSON.stringify({
        columns, source, with_cv: withCv, use_pca: usePca,
        strong_threshold: strongThreshold, n_imputations: nImputations,
      }),
    });
  },

  queryGemini(question: string, columns: string[], geminiKey: string): Promise<{ status: string; response: string; context_rows_used?: number; columns_used?: string[] }> {
    return request('/query/gemini', {
      method: 'POST',
      body: JSON.stringify({ question, columns, gemini_key: geminiKey }),
    });
  },

  // AI interpretation of a rendered results table (Cramér's V / XGBoost).
  interpretAnalysis(
    kind: 'cramers' | 'xgboost',
    context: string,
    geminiKey: string,
    question?: string,
  ): Promise<{ status: string; response: string }> {
    return request('/analysis/interpret', {
      method: 'POST',
      body: JSON.stringify({ kind, context, gemini_key: geminiKey, question }),
    });
  },

  getDashboardSummary(): Promise<DashboardSummary> {
    return request('/dashboard/summary');
  },

  healthCheck(): Promise<{ status: string; version: string }> {
    return request('/health');
  },
};
