import type {
  LoadDataResponse,
  AnalysisResponse,
  DashboardSummary,
} from '../types';

const BASE = '/api';

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
  loadData(rows = 10000): Promise<LoadDataResponse> {
    return request(`/data/load?rows=${rows}`);
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

  runAnalysis(columns: string[]): Promise<AnalysisResponse> {
    return request('/analyze/run', {
      method: 'POST',
      body: JSON.stringify({ columns }),
    });
  },

  queryGemini(question: string, column: string, geminiKey: string): Promise<{ status: string; response: string }> {
    return request('/query/gemini', {
      method: 'POST',
      body: JSON.stringify({ question, column, gemini_key: geminiKey }),
    });
  },

  getDashboardSummary(): Promise<DashboardSummary> {
    return request('/dashboard/summary');
  },

  healthCheck(): Promise<{ status: string; version: string }> {
    return request('/health');
  },
};
