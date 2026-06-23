import { useState } from 'react';
import { Search, AlertTriangle, Key, Sparkles } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { DataTable } from '../data/DataTable';
import type { RagSearchResponse } from '../../types';

export function RagSearchPage() {
  const { data, dataLoaded, cohereKey, setCohereKey, setPage } = useStore();
  const [selectedCols, setSelectedCols] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const [topN, setTopN] = useState(50);
  const [result, setResult] = useState<RagSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const columns = data?.columns ?? [];

  const toggle = (c: string) =>
    setSelectedCols((p) => (p.includes(c) ? p.filter((x) => x !== c) : [...p, c]));

  const runSearch = async () => {
    if (!selectedCols.length) {
      setError('Select at least one column to search.');
      return;
    }
    if (!question.trim()) {
      setError('Enter a question.');
      return;
    }
    if (!cohereKey) {
      setError('Enter your Cohere API key.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.ragSearch(selectedCols, question, cohereKey, topN);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  if (!dataLoaded) {
    return (
      <Panel title="No Data Loaded">
        <p className="text-sm text-text-muted">
          Load a dataset first from the{' '}
          <button onClick={() => setPage('data')} className="text-accent hover:underline">
            Data Explorer
          </button>
          . RAG search reranks the loaded (and filtered) dataset.
        </p>
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      <Panel title="Dataset RAG Search" subtitle="Semantic rerank over the loaded dataset via Cohere">
        <div className="space-y-4">
          {/* Config row */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs text-text-muted">Cohere API Key</label>
              <div className="relative">
                <Key className="absolute left-2.5 top-2 h-3.5 w-3.5 text-text-muted" />
                <input
                  type="password"
                  value={cohereKey}
                  onChange={(e) => setCohereKey(e.target.value)}
                  placeholder="Enter API key..."
                  className="w-full rounded border border-border bg-deep py-1.5 pl-8 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-muted">Top results: {topN}</label>
              <input
                type="range"
                min={10}
                max={100}
                step={5}
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                className="mt-2 w-full accent-accent"
              />
            </div>
          </div>

          {/* Columns */}
          <div>
            <label className="mb-1.5 block text-xs text-text-muted">Columns to search</label>
            <div className="flex flex-wrap gap-2">
              {columns.map((c) => (
                <button
                  key={c}
                  onClick={() => toggle(c)}
                  className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                    selectedCols.includes(c)
                      ? 'border-accent bg-accent-dim/30 text-accent-bright'
                      : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          {/* Question */}
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Sparkles className="absolute left-3 top-2.5 h-4 w-4 text-text-muted" />
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && runSearch()}
                placeholder="Ask a question to rank relevant reports..."
                className="w-full rounded-md border border-border bg-deep py-2 pl-9 pr-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
              />
            </div>
            <button
              onClick={runSearch}
              disabled={loading}
              className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
            >
              <Search className="h-4 w-4" />
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
        </div>
      </Panel>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Reranking with Cohere..." />}

      {result && (
        <Panel
          title="Reranked Results"
          subtitle={`${result.n_results} results · searched ${result.searched_columns.join(', ')}`}
          noPad
        >
          <DataTable data={result.data} maxHeight="calc(100vh - 380px)" />
        </Panel>
      )}
    </div>
  );
}
