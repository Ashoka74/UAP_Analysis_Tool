import { useState, useCallback } from 'react';
import { Upload, Database, FileSpreadsheet, AlertTriangle } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { DataTable } from './DataTable';
import { FilterPanel } from './FilterPanel';
import type { ColumnStat, DataResponse } from '../../types';

export function DataExplorer() {
  const { data, columnStats, setData, dataLoaded } = useStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeData, setActiveData] = useState<DataResponse | null>(null);
  const [activeStats, setActiveStats] = useState<ColumnStat[]>([]);

  const displayData = activeData ?? data;
  const displayStats = activeStats.length > 0 ? activeStats : columnStats;

  const handleLoadHDF5 = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.loadData(10000);
      setData(res.data, res.column_stats);
      setActiveData(null);
      setActiveStats([]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.uploadFile(file);
      setData(res.data, res.column_stats);
      setActiveData(null);
      setActiveStats([]);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFilter = useCallback(
    async (
      filters: {
        column: string;
        type: string;
        values?: string[];
        min_val?: number;
        max_val?: number;
        pattern?: string;
      }[]
    ) => {
      if (filters.length === 0) {
        setActiveData(null);
        setActiveStats([]);
        return;
      }
      setLoading(true);
      try {
        const res = await api.filterData(filters);
        setActiveData(res.data);
        setActiveStats(res.column_stats);
      } catch {
        // Fall back to unfiltered
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return (
    <div className="space-y-4">
      {/* Data source controls */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={handleLoadHDF5}
          disabled={loading}
          className="flex items-center gap-2 rounded-md border border-border bg-surface px-4 py-2 text-sm text-text-primary transition-colors hover:border-accent hover:bg-elevated disabled:opacity-50"
        >
          <Database className="h-4 w-4 text-accent" />
          Load Parsed Dataset
        </button>

        <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border bg-surface px-4 py-2 text-sm text-text-primary transition-colors hover:border-purple hover:bg-elevated">
          <Upload className="h-4 w-4 text-purple" />
          Upload CSV / Excel
          <input type="file" accept=".csv,.xlsx,.xls" onChange={handleUpload} className="hidden" />
        </label>

        {dataLoaded && displayData && (
          <div className="ml-auto flex items-center gap-3 text-xs text-text-muted">
            <FileSpreadsheet className="h-4 w-4" />
            <span>
              {displayData.returned_rows.toLocaleString()} / {displayData.total_rows.toLocaleString()} rows
            </span>
            <span>{displayData.columns.length} columns</span>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Processing data..." />}

      {/* Filters */}
      {dataLoaded && displayStats.length > 0 && (
        <FilterPanel columns={displayStats} onApply={handleFilter} />
      )}

      {/* Column Stats */}
      {dataLoaded && displayStats.length > 0 && (
        <Panel title="Column Statistics" subtitle={`${displayStats.length} columns`}>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
            {displayStats.map((col) => (
              <div
                key={col.name}
                className="rounded border border-border/50 bg-raised p-2.5"
              >
                <p className="truncate text-xs font-medium text-text-primary" title={col.name}>
                  {col.name}
                </p>
                <p className="mt-0.5 text-[10px] text-text-muted">
                  {col.dtype} | {col.unique} unique
                </p>
                {col.top_values && col.top_values.length > 0 && (
                  <div className="mt-1.5 space-y-0.5">
                    {col.top_values.slice(0, 3).map((tv) => (
                      <div key={tv.value} className="flex items-center gap-1">
                        <div
                          className="h-1 rounded-full bg-accent/60"
                          style={{
                            width: `${Math.min(100, (tv.count / (col.non_null || 1)) * 100)}%`,
                          }}
                        />
                        <span className="whitespace-nowrap text-[9px] text-text-muted">
                          {tv.value.slice(0, 15)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* Data Table */}
      {dataLoaded && displayData && (
        <Panel title="Data View" noPad>
          <DataTable data={displayData} maxHeight="calc(100vh - 420px)" />
        </Panel>
      )}
    </div>
  );
}
