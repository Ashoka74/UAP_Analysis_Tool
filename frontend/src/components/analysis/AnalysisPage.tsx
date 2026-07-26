import { useState } from 'react';
import { Play, AlertTriangle, CheckCircle2, Layers, BarChart3, Grid3x3, Settings2, Network, Sparkles } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ClusterVisualization } from './ClusterVisualization';
import { CorrelationHeatmap } from './CorrelationHeatmap';
import { XGBoostResults } from './XGBoostResults';
import { DistributionChart } from './DistributionChart';
import { CramersVExplorer } from './CramersVExplorer';
import { EvidenceGuide } from './EvidenceGuide';
import type { AnalysisResponse, XGBoostResult, XgboostPcaResponse, XgboostImputeResponse } from '../../types';

type TabId = 'clusters' | 'correlation' | 'xgboost' | 'distribution' | 'association';

export function AnalysisPage() {
  const {
    data, dataLoaded, analysisResults, setAnalysisResults, analysisRunning, setAnalysisRunning, setPage,
    setCramersSelected, setCramersAutoRun,
    analysisActiveTab, setAnalysisActiveTab,
    analysisXgbHandoff: assocXgboost, analysisXgbHandoffPca: assocPca,
    analysisXgbHandoffImpute: assocImpute, setAnalysisXgbHandoff,
    openaiKey, setOpenaiKey, geminiKey, setGeminiKey, deepseekKey, setDeepseekKey,
  } = useStore();
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Store-backed (not useState) — AnalysisPage unmounts on page navigation
  // (see App.tsx), so local state here doesn't survive a trip to another tab
  // and back. Same reasoning for the Cramér's V hand-off results above:
  // assocXgboost/assocPca/assocImpute now come from the store (previously
  // local useState here, which silently dropped the XGBoost run handed off
  // from the Cramér's V explorer the moment you left the page).
  const activeTab = analysisActiveTab as TabId;
  const setActiveTab = (tab: TabId) => setAnalysisActiveTab(tab);

  const handleAssocXgboost = (
    r: Record<string, XGBoostResult>,
    extras?: { pca?: XgboostPcaResponse; impute?: XgboostImputeResponse },
  ) => {
    setAnalysisXgbHandoff(r, extras);
    setActiveTab('xgboost');
  };

  // Reverse hand-off: push the analyzed columns from the Feature Importance tab
  // into the Cramér's V explorer and auto-compute associations there.
  const handleXgbToAssoc = (cols: string[]) => {
    setCramersSelected(cols);
    setCramersAutoRun(true);
    setActiveTab('association');
  };

  // Cluster pipeline tuning (mirrors analyzing.py controls)
  const [showParams, setShowParams] = useState(false);
  const [enableTfidf, setEnableTfidf] = useState(false);
  const [enableLlm, setEnableLlm] = useState(false);
  const [llmModel, setLlmModel] = useState('gpt-4o-mini');
  const [minClusterSize, setMinClusterSize] = useState(15);
  const [nNeighbors, setNNeighbors] = useState(15);
  const [minDist, setMinDist] = useState(0.1);

  // AI cluster naming uses an LLM; pick the provider/key from the model id and
  // reuse the matching key already held in the store (set on Parsing / AI Query).
  const llmProvider = llmModel.includes('gemini') ? 'google' : llmModel.startsWith('deepseek') ? 'deepseek' : 'openai';
  const llmKey = llmProvider === 'google' ? geminiKey : llmProvider === 'deepseek' ? deepseekKey : openaiKey;
  const setLlmKey = llmProvider === 'google' ? setGeminiKey : llmProvider === 'deepseek' ? setDeepseekKey : setOpenaiKey;

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
      const res = await api.runAnalysis(selected, {
        enable_tfidf: enableTfidf,
        min_cluster_size: minClusterSize,
        n_neighbors: nNeighbors,
        min_dist: minDist,
        ...(enableLlm
          ? { enable_llm: true, llm_provider: llmProvider, llm_model: llmModel, llm_api_key: llmKey }
          : {}),
      });
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

  const tabs: { id: TabId; label: string; icon: typeof Layers; needsResults: boolean }[] = [
    { id: 'clusters', label: 'Clusters', icon: Layers, needsResults: true },
    { id: 'correlation', label: 'Correlation', icon: Grid3x3, needsResults: true },
    { id: 'xgboost', label: 'Feature Importance', icon: BarChart3, needsResults: true },
    { id: 'distribution', label: 'Distribution', icon: BarChart3, needsResults: true },
    { id: 'association', label: "Cramér's V Explorer", icon: Network, needsResults: false },
  ];

  return (
    <div className="space-y-4">
      {/* Column selector */}
      <Panel
        title="Select Columns for Analysis"
        subtitle="Choose columns to run through the clustering and ML pipeline"
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowParams((s) => !s)}
              className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
                showParams ? 'border-accent text-accent' : 'border-border text-text-secondary hover:text-text-primary'
              }`}
            >
              <Settings2 className="h-3.5 w-3.5" /> Params
            </button>
            <button
              onClick={runAnalysis}
              disabled={selected.length === 0 || analysisRunning}
              className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5" />
              {analysisRunning ? 'Running...' : 'Run Analysis'}
            </button>
          </div>
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

        {showParams && (
          <div className="mt-3 space-y-3 rounded-md border border-border/50 bg-raised/50 p-3">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <label className="flex items-center gap-2 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={enableTfidf}
                onChange={(e) => setEnableTfidf(e.target.checked)}
                className="accent-accent"
              />
              TF-IDF cluster naming + merging
            </label>
            <label className="flex items-center gap-2 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={enableLlm}
                onChange={(e) => setEnableLlm(e.target.checked)}
                className="accent-accent"
              />
              <span className="inline-flex items-center gap-1">
                <Sparkles className="h-3 w-3 text-purple" /> AI cluster naming (LLM)
              </span>
            </label>
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">
                Min cluster size: {minClusterSize}
              </label>
              <input
                type="range" min={2} max={50} value={minClusterSize}
                onChange={(e) => setMinClusterSize(Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">
                UMAP neighbors: {nNeighbors}
              </label>
              <input
                type="range" min={2} max={100} value={nNeighbors}
                onChange={(e) => setNNeighbors(Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">
                UMAP min dist: {minDist.toFixed(2)}
              </label>
              <input
                type="range" min={0} max={1} step={0.05} value={minDist}
                onChange={(e) => setMinDist(Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>
          </div>

          {enableLlm && (
            <div className="grid grid-cols-1 gap-3 border-t border-border/40 pt-3 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-[11px] text-text-muted">LLM model (names each cluster)</label>
                <select
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                  className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                >
                  {['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4o', 'models/gemini-3.1-pro-preview'].map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-[11px] text-text-muted">
                  {llmProvider === 'google' ? 'Gemini' : llmProvider === 'deepseek' ? 'DeepSeek' : 'OpenAI'} API key
                </label>
                <input
                  type="password"
                  value={llmKey}
                  onChange={(e) => setLlmKey(e.target.value)}
                  placeholder="Reused from Parsing / AI Query if already set…"
                  className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                />
              </div>
            </div>
          )}
          </div>
        )}
      </Panel>

      {analysisRunning && <LoadingSpinner text="Running analysis pipeline..." />}

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {/* Success message */}
      {results && (
        <>
          <div className="flex items-center gap-2 rounded-md border border-success/30 bg-success/10 px-4 py-2.5 text-sm text-success">
            <CheckCircle2 className="h-4 w-4" />
            Analysis complete for {Object.keys(results.results).length} column(s)
          </div>
          {results.mock_mode && (
            <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/10 px-4 py-2.5 text-sm text-warning">
              <AlertTriangle className="h-4 w-4" />
              Mock analysis mode is enabled. Outputs are for demo/debugging, not scientific conclusions.
            </div>
          )}
        </>
      )}

      {/* Tab bar — always available; the Cramér's V explorer needs no analysis run */}
      <div className="flex flex-wrap gap-1 border-b border-border">
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

      {/* Association explorer — independent of the cluster pipeline. Its XGBoost
          run is handed to the Feature Importance tab via handleAssocXgboost. */}
      {activeTab === 'association' && (
        <div className="space-y-4">
          <EvidenceGuide />
          <CramersVExplorer source="dataset" onXgboost={handleAssocXgboost} />
        </div>
      )}

      {/* Feature importance — independent of the cluster pipeline: it shows the
          explorer-driven results when present, otherwise the pipeline's. */}
      {activeTab === 'xgboost' && (() => {
        const usingAssoc = !!assocXgboost && Object.keys(assocXgboost).length > 0;
        const xgb = usingAssoc ? assocXgboost : results?.xgboost ?? null;
        if (xgb && Object.keys(xgb).length > 0) {
          return (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                {usingAssoc ? (
                  <div className="flex items-center gap-2 rounded-md border border-purple/30 bg-purple/10 px-4 py-2.5 text-xs text-text-secondary">
                    <Network className="h-4 w-4 text-purple" />
                    Computed directly from your Cramér's V column selection — each column predicted
                    from the others{assocPca ? ', with an optional PCA 2nd pass collapsing redundancy clusters' : ''}.
                  </div>
                ) : (
                  <span className="text-xs text-text-muted">
                    Feature importance for {Object.keys(xgb).length} column(s).
                  </span>
                )}
                <button
                  onClick={() => handleXgbToAssoc(Object.keys(xgb))}
                  title="Send these columns to the Cramér's V Explorer and compute associations"
                  className="flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
                >
                  <Network className="h-3.5 w-3.5" /> Cramér's V →
                </button>
              </div>
              <EvidenceGuide />
              <XGBoostResults
                results={xgb}
                pca={usingAssoc ? assocPca ?? undefined : undefined}
                impute={usingAssoc ? assocImpute ?? undefined : undefined}
              />
            </div>
          );
        }
        return (
          <Panel title="Feature Importance">
            <p className="text-sm text-text-muted">
              No feature-importance results yet. Run the cluster pipeline above, or open the{' '}
              <button onClick={() => setActiveTab('association')} className="text-accent hover:underline">
                Cramér's V Explorer
              </button>
              , select columns, and click <span className="text-accent">Feature Importance →</span>.
            </p>
          </Panel>
        );
      })()}

      {/* Result-dependent tabs */}
      {activeTab !== 'association' && activeTab !== 'xgboost' && !results && (
        <Panel title="Run the analysis pipeline">
          <p className="text-sm text-text-muted">
            Select columns above and click <span className="text-accent">Run Analysis</span> to
            populate clustering, correlation, feature-importance and distribution views. The{' '}
            <button onClick={() => setActiveTab('association')} className="text-accent hover:underline">
              Cramér's V Explorer
            </button>{' '}
            works directly on raw columns without running the pipeline.
          </p>
        </Panel>
      )}

      {results && (
        <>
          {/* Cluster visualizations */}
          {activeTab === 'clusters' && (
            <div className="flex flex-col gap-4">
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
