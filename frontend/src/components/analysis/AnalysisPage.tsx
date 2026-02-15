import { useState } from 'react';
import { Play, AlertTriangle, CheckCircle2, Layers, BarChart3, Grid3x3 } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ClusterVisualization } from './ClusterVisualization';
import { CorrelationHeatmap } from './CorrelationHeatmap';
import { XGBoostResults } from './XGBoostResults';
import { DistributionChart } from './DistributionChart';
import type { AnalysisResponse } from '../../types';

type TabId = 'clusters' | 'correlation' | 'xgboost' | 'distribution';

export function AnalysisPage() {
  const { data, dataLoaded, analysisResults, setAnalysisResults, analysisRunning, setAnalysisRunning, setPage } = useStore();
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('clusters');

  const columns = data?.columns ?? [];
  const results: AnalysisResponse | null = analysisResults;

  const toggleColumn = (col: string) => {
    setSelected((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const runAnalysis = async () => {
    if (selected.length === 0) return;
    setError(null);
    setAnalysisRunning(true);
    try {
      const res = await api.runAnalysis(selected);
      setAnalysisResults(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
      setAnalysisRunning(false);
    }
  };

  if (!dataLoaded) {
    return (
      <Panel title="No Data Loaded">
        <p className="text-sm text-text-muted">
          Load a dataset first from the{' '}
          <button onClick={() => setPage('dashboard')} className="text-accent hover:underline">
            Dashboard
          </button>{' '}
          or{' '}
          <button onClick={() => setPage('data')} className="text-accent hover:underline">
            Data Explorer
          </button>.
        </p>
      </Panel>
    );
  }

  const tabs: { id: TabId; label: string; icon: typeof Layers }[] = [
    { id: 'clusters', label: 'Clusters', icon: Layers },
    { id: 'correlation', label: 'Correlation', icon: Grid3x3 },
    { id: 'xgboost', label: 'Feature Importance', icon: BarChart3 },
    { id: 'distribution', label: 'Distribution', icon: BarChart3 },
  ];

  return (
    <div className="space-y-4">
      {/* Column selector */}
      <Panel
        title="Select Columns for Analysis"
        subtitle="Choose columns to run through the clustering and ML pipeline"
        actions={
          <button
            onClick={runAnalysis}
            disabled={selected.length === 0 || analysisRunning}
            className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            {analysisRunning ? 'Running...' : 'Run Analysis'}
          </button>
        }
      >
        <div className="flex flex-wrap gap-2">
          {columns.map((col) => (
            <button
              key={col}
              onClick={() => toggleColumn(col)}
              className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                selected.includes(col)
                  ? 'border-accent bg-accent-dim/30 text-accent-bright'
                  : 'border-border bg-raised text-text-secondary hover:border-border-bright'
              }`}
            >
              {col}
            </button>
          ))}
        </div>
        {selected.length > 0 && (
          <p className="mt-2 text-xs text-text-muted">
            {selected.length} column{selected.length > 1 ? 's' : ''} selected
          </p>
        )}
      </Panel>

      {analysisRunning && <LoadingSpinner text="Running analysis pipeline..." />}

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <>
          {/* Success message */}
          <div className="flex items-center gap-2 rounded-md border border-success/30 bg-success/10 px-4 py-2.5 text-sm text-success">
            <CheckCircle2 className="h-4 w-4" />
            Analysis complete for {Object.keys(results.results).length} column(s)
          </div>

          {/* Result tabs */}
          <div className="flex gap-1 border-b border-border">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-1.5 border-b-2 px-4 py-2.5 text-xs font-medium transition-colors ${
                  activeTab === id
                    ? 'border-accent text-accent'
                    : 'border-transparent text-text-muted hover:text-text-secondary'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>

          {/* Cluster visualizations */}
          {activeTab === 'clusters' && (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              {Object.entries(results.cluster_viz).map(([col, viz]) => (
                <Panel key={col} title={viz.title}>
                  <ClusterVisualization viz={viz} />
                </Panel>
              ))}
            </div>
          )}

          {/* Correlation heatmap */}
          {activeTab === 'correlation' && results.cramers_v && (
            <Panel title="Cramer's V Correlation Matrix">
              <CorrelationHeatmap data={results.cramers_v} height={500} />
            </Panel>
          )}
          {activeTab === 'correlation' && !results.cramers_v && (
            <Panel title="Correlation">
              <p className="text-sm text-text-muted">
                Select at least 2 columns to compute correlation analysis.
              </p>
            </Panel>
          )}

          {/* XGBoost */}
          {activeTab === 'xgboost' && results.xgboost && Object.keys(results.xgboost).length > 0 && (
            <XGBoostResults results={results.xgboost} />
          )}
          {activeTab === 'xgboost' && (!results.xgboost || Object.keys(results.xgboost).length === 0) && (
            <Panel title="Feature Importance">
              <p className="text-sm text-text-muted">
                Select at least 2 columns to run feature importance analysis.
              </p>
            </Panel>
          )}

          {/* Distribution */}
          {activeTab === 'distribution' && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
              {Object.entries(results.results).map(([col, analysis]) => (
                <Panel key={col} title={`${col} (${analysis.cluster_count} clusters)`}>
                  <DistributionChart
                    data={analysis.distribution}
                    title={col}
                    height={280}
                  />
                </Panel>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
