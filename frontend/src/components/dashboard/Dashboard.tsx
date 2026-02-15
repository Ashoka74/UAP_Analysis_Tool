import { useEffect, useState, useCallback } from 'react';
import {
  Database,
  Columns3,
  BrainCircuit,
  HardDrive,
  Upload,
  Play,
  AlertTriangle,
} from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { StatCard } from './StatCard';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';

export function Dashboard() {
  const { summary, setSummary, setPage, dataLoaded, setData } = useStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    try {
      const s = await api.getDashboardSummary();
      setSummary(s);
    } catch {
      // Summary endpoint might not have data yet
    }
  }, [setSummary]);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  const handleLoadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.loadData(10000);
      setData(res.data, res.column_stats);
      await fetchSummary();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Records"
          value={summary?.total_rows?.toLocaleString() ?? '---'}
          icon={<Database className="h-5 w-5" />}
          color="text-accent"
        />
        <StatCard
          label="Columns"
          value={summary?.total_columns ?? '---'}
          icon={<Columns3 className="h-5 w-5" />}
          color="text-purple"
        />
        <StatCard
          label="Analysis Runs"
          value={summary?.analyzed_columns ?? 0}
          icon={<BrainCircuit className="h-5 w-5" />}
          color="text-cyan"
        />
        <StatCard
          label="Memory Usage"
          value={summary?.memory_mb ? `${summary.memory_mb} MB` : '---'}
          icon={<HardDrive className="h-5 w-5" />}
          color="text-orange"
        />
      </div>

      {/* Quick actions & status */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Quick Actions */}
        <Panel title="Quick Actions" className="lg:col-span-1">
          <div className="space-y-3">
            <button
              onClick={handleLoadData}
              disabled={loading}
              className="flex w-full items-center gap-3 rounded-md border border-border bg-raised px-4 py-3 text-sm text-text-primary transition-colors hover:border-accent hover:bg-elevated disabled:opacity-50"
            >
              <Upload className="h-4 w-4 text-accent" />
              {loading ? 'Loading Dataset...' : 'Load Parsed Dataset (HDF5)'}
            </button>
            <button
              onClick={() => setPage('data')}
              className="flex w-full items-center gap-3 rounded-md border border-border bg-raised px-4 py-3 text-sm text-text-primary transition-colors hover:border-accent hover:bg-elevated"
            >
              <Database className="h-4 w-4 text-purple" />
              Open Data Explorer
            </button>
            <button
              onClick={() => setPage('analysis')}
              disabled={!dataLoaded}
              className="flex w-full items-center gap-3 rounded-md border border-border bg-raised px-4 py-3 text-sm text-text-primary transition-colors hover:border-accent hover:bg-elevated disabled:opacity-50"
            >
              <Play className="h-4 w-4 text-cyan" />
              Run Analysis Pipeline
            </button>
          </div>
          {loading && <LoadingSpinner text="Loading dataset..." />}
          {error && (
            <div className="mt-3 flex items-center gap-2 rounded-md bg-danger/10 px-3 py-2 text-sm text-danger">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          )}
        </Panel>

        {/* Data schema overview */}
        <Panel title="Data Schema" subtitle={summary?.loaded ? `${summary.total_columns} columns detected` : 'Load data to view schema'} className="lg:col-span-2">
          {summary?.columns && summary.columns.length > 0 ? (
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase text-text-muted">
                    <th className="pb-2 pr-4">Column</th>
                    <th className="pb-2 pr-4">Type</th>
                    <th className="pb-2">Nulls</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.columns.map((col) => (
                    <tr key={col} className="border-b border-border/50">
                      <td className="py-1.5 pr-4 font-mono text-xs text-text-primary">{col}</td>
                      <td className="py-1.5 pr-4">
                        <span className="rounded bg-elevated px-1.5 py-0.5 text-xs text-text-secondary">
                          {summary.dtypes?.[col] ?? '?'}
                        </span>
                      </td>
                      <td className="py-1.5 text-xs text-text-muted">
                        {summary.null_counts?.[col] ?? 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-text-muted">No data loaded. Use &quot;Load Parsed Dataset&quot; to begin.</p>
          )}
        </Panel>
      </div>

      {/* System status */}
      <Panel title="System Pipeline">
        <div className="flex items-center gap-2">
          {['Data Ingestion', 'Embedding', 'Dimensionality Reduction', 'Clustering', 'TF-IDF Naming', 'Merge Clusters', 'XGBoost', 'Visualization'].map(
            (step, i) => (
              <div key={step} className="flex items-center gap-2">
                <div
                  className={`rounded-md border px-3 py-1.5 text-xs font-medium ${
                    i === 0 && dataLoaded
                      ? 'border-success/30 bg-success/10 text-success'
                      : 'border-border bg-raised text-text-muted'
                  }`}
                >
                  {step}
                </div>
                {i < 7 && (
                  <div className="h-px w-4 bg-border" />
                )}
              </div>
            )
          )}
        </div>
      </Panel>
    </div>
  );
}
