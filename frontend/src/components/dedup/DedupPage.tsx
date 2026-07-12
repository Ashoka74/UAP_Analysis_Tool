import { useState } from 'react';
import {
  GitCompare,
  CheckCircle2,
  AlertTriangle,
  Play,
  Sliders,
  Sparkles,
  Layers,
  Database,
  ShieldAlert,
} from 'lucide-react';
import type {
  SimpleSimilarityResponse,
  SimpleDuplicateResponse,
  AdvancedDedupResponse,
  CrossDbPipelineResponse,
} from '../../types';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';

export function DedupPage() {
  const { data, dataLoaded } = useStore();
  const [dataSourceMode, setDataSourceMode] = useState<'real_dataset' | 'sample_mock'>('sample_mock');
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced' | 'cross_db'>('simple');



  // Simple Tab States
  const [textA, setTextA] = useState(
    'A large triangular craft with 3 green lights hovered silently over Phoenix for 10 minutes.'
  );
  const [textB, setTextB] = useState(
    'Witness observed a silent triangular UFO with green lights near Phoenix on the same evening.'
  );
  const [latA, setLatA] = useState('33.4484');
  const [lonA, setLonA] = useState('-112.0740');
  const [dateA, setDateA] = useState('1997-03-13');

  const [latB, setLatB] = useState('33.4500');
  const [lonB, setLonB] = useState('-112.0710');
  const [dateB, setDateB] = useState('1997-03-13');

  const [simResult, setSimResult] = useState<SimpleSimilarityResponse | null>(null);
  const [dupResult, setDupResult] = useState<SimpleDuplicateResponse | null>(null);
  const [simpleLoading, setSimpleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Advanced Tab States
  const [threshold, setThreshold] = useState(0.80);
  const [dateDiffDays, setDateDiffDays] = useState(3);
  const [maxKm, setMaxKm] = useState(50.0);
  const [useLlmJudge, setUseLlmJudge] = useState(false);
  const [advResult, setAdvResult] = useState<AdvancedDedupResponse | null>(null);
  const [advLoading, setAdvLoading] = useState(false);

  // Cross-DB Pipeline States
  const [crossPathMode, setCrossPathMode] = useState<'easy' | 'hard'>('easy');
  const [crossCmpMode, setCrossCmpMode] = useState<'within' | 'between'>('within');
  const [crossBinFilter, setCrossBinFilter] = useState<string>('All');
  const [crossGateFilter, setCrossGateFilter] = useState<string>('show_all');
  const [crossResult, setCrossResult] = useState<CrossDbPipelineResponse | null>(null);
  const [crossLoading, setCrossLoading] = useState(false);
  const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0);
  const [splitViewMode, setSplitViewMode] = useState<'single_pair' | 'batch_table'>('single_pair');

  const handleRunCrossDb = async () => {
    setCrossLoading(true);
    setError(null);
    try {
      let recordsA: any[] = [];
      let recordsB: any[] | null = null;
      if (dataSourceMode === 'real_dataset' && dataLoaded && data?.rows && data.rows.length > 0) {
        const mappedRows = data.rows.slice(0, 200).map((row: any, idx: number) => {
          const id = row.id || row.locus_tag || row.case_id || `Case-${idx + 1}`;
          const notes = row.witness_notes || row.narrative || row.description || row.summary || row.text || 'UAP sighting report';
          const dt = row.date_time || row.date || row.datetime || '2000-01-01';
          const lat = row.latitude || row.lat || 0;
          const lon = row.longitude || row.lon || row.lng || 0;
          return { ...row, id: String(id), witness_notes: String(notes), date_time: String(dt), latitude: Number(lat) || 0, longitude: Number(lon) || 0 };
        });

        if (crossCmpMode === 'between') {
          const half = Math.floor(mappedRows.length / 2);
          recordsA = mappedRows.slice(0, half);
          recordsB = mappedRows.slice(half);
        } else {
          recordsA = mappedRows;
          recordsB = null;
        }
      } else {
        const sampleRecords = [
          { id: 'NUFORC-114209', witness_notes: textA, date_time: dateA, latitude: latA, longitude: lonA, shape: 'Triangle', city: 'Phoenix', state: 'AZ', duration: '10 mins', database: 'NUFORC' },
          { id: 'MUFON-88912', witness_notes: textB, date_time: dateB, latitude: latB, longitude: lonB, shape: 'Triangle', city: 'Phoenix', state: 'AZ', duration: '8 mins', database: 'MUFON' },
          { id: 'BLUEBOOK-1092', witness_notes: 'Triangular craft observed near Phoenix airport with silent motion.', date_time: '1997-03-13', latitude: '33.4400', longitude: '-112.0700', shape: 'Triangle', city: 'Phoenix', state: 'AZ', duration: '12 mins', database: 'Project Blue Book' },
          { id: 'NUFORC-67210', witness_notes: 'Green fireball streaked across night sky over California coast.', date_time: '2025-08-14', latitude: '36.7783', longitude: '-119.4179', shape: 'Fireball', city: 'Fresno', state: 'CA', duration: '15 secs', database: 'NUFORC' },
          { id: 'MUFON-55319', witness_notes: 'Bright green fireball seen exploding high over California ocean.', date_time: '2025-08-14', latitude: '36.7800', longitude: '-119.4100', shape: 'Fireball', city: 'Fresno', state: 'CA', duration: '12 secs', database: 'MUFON' }
        ];
        recordsA = sampleRecords;
        recordsB = crossCmpMode === 'between' ? sampleRecords.slice(1) : null;
      }

      const dataRes = await api.runCrossDbPipeline({
        records_a: recordsA,
        records_b: recordsB,
        cols_a: ['witness_notes'],
        cols_b: ['witness_notes'],
        threshold: threshold,
        max_days: dateDiffDays,
        max_km: maxKm
      });
      setCrossResult(dataRes as CrossDbPipelineResponse);
      setSelectedPairIndex(0);
    } catch (err: any) {
      setError(err.message || 'Cross-DB pipeline evaluation failed. Check backend connection and dataset schema.');
    } finally {
      setCrossLoading(false);
    }
  };

  // API Call: Check Similarity
  const handleCheckSimilarity = async () => {
    setSimpleLoading(true);
    setError(null);
    setSimResult(null);
    setDupResult(null);
    try {
      const dataRes = await api.checkSimpleSimilarity({ text_a: textA, text_b: textB });
      setSimResult(dataRes as SimpleSimilarityResponse);
    } catch (err: any) {
      setError(err.message || 'Similarity check failed.');
    } finally {
      setSimpleLoading(false);
    }
  };

  // API Call: Check Duplicate
  const handleCheckDuplicate = async () => {
    setSimpleLoading(true);
    setError(null);
    setSimResult(null);
    setDupResult(null);
    try {
      const dataRes = await api.checkSimpleDuplicate({
        record_a: { narrative: textA, lat: latA, lon: lonA, date: dateA },
        record_b: { narrative: textB, lat: latB, lon: lonB, date: dateB },
      });
      setDupResult(dataRes as SimpleDuplicateResponse);
    } catch (err: any) {
      setError(err.message || 'Duplicate check failed.');
    } finally {
      setSimpleLoading(false);
    }
  };

  // API Call: Run Advanced Dedup
  const handleRunAdvanced = async () => {
    setAdvLoading(true);
    setError(null);
    try {
      const dataRes = await api.runAdvancedDedup({
        threshold,
        date_diff_days: dateDiffDays,
        max_km: maxKm,
        use_llm_judge: useLlmJudge,
      });
      setAdvResult(dataRes as AdvancedDedupResponse);
    } catch (err: any) {
      setError(err.message || 'Advanced deduplication failed.');
    } finally {
      setAdvLoading(false);
    }
  };


  return (
    <div className="flex flex-col gap-6 p-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="flex flex-col gap-2 rounded-xl bg-gradient-to-r from-accent/20 via-abyss to-accent-dim/10 border border-accent/30 p-6 backdrop-blur-md shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-accent/20 border border-accent/40 text-accent-bright shadow-inner">
              <GitCompare className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-text-primary">
                Harrier Deduplication Studio
              </h1>
              <p className="text-xs text-text-secondary">
                Cross-database entity resolution, semantic pairwise gating, and transitive graph clustering
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-elevated/80 border border-border text-xs text-accent-bright font-mono">
            <Sparkles className="h-3.5 w-3.5" />
            harrier-oss-v1-270m
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-2 mt-4 border-b border-border/60 pb-1">
          <button
            onClick={() => setActiveTab('simple')}
            className={`flex items-center gap-2 px-4 py-2 rounded-t-lg font-medium text-sm transition-all duration-200 ${
              activeTab === 'simple'
                ? 'bg-accent/20 text-accent-bright border-b-2 border-accent-bright shadow-sm'
                : 'text-text-secondary hover:text-text-primary hover:bg-elevated/40'
            }`}
          >
            <GitCompare className="h-4 w-4" />
            Simple Pair Checker (is_similar / is_duplicate)
          </button>
          <button
            onClick={() => setActiveTab('advanced')}
            className={`flex items-center gap-2 px-4 py-2 rounded-t-lg font-medium text-sm transition-all duration-200 ${
              activeTab === 'advanced'
                ? 'bg-accent/20 text-accent-bright border-b-2 border-accent-bright shadow-sm'
                : 'text-text-secondary hover:text-text-primary hover:bg-elevated/40'
            }`}
          >
            <Layers className="h-4 w-4" />
            Advanced Dedupe Studio (Batch Clustering)
          </button>
          <button
            onClick={() => setActiveTab('cross_db')}
            className={`flex items-center gap-2 px-4 py-2 rounded-t-lg font-medium text-sm transition-all duration-200 ${
              activeTab === 'cross_db'
                ? 'bg-accent/20 text-accent-bright border-b-2 border-accent-bright shadow-sm'
                : 'text-text-secondary hover:text-text-primary hover:bg-elevated/40'
            }`}
          >
            <Database className="h-4 w-4" />
            Cross-DB Pipeline (Easy vs Hard Path)
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-300">
          <AlertTriangle className="h-5 w-5 shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* SIMPLE TAB */}
      {activeTab === 'simple' && (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Record A Input Card */}
            <div className="flex flex-col gap-4 rounded-xl border border-border bg-abyss/80 p-5 shadow-md">
              <div className="flex items-center justify-between border-b border-border/50 pb-3">
                <span className="text-sm font-semibold text-accent-bright flex items-center gap-2">
                  <Database className="h-4 w-4" /> Record A (Candidate Sighting)
                </span>
                <span className="text-xs text-text-muted font-mono">Input A</span>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-text-secondary">Narrative Description</label>
                <textarea
                  rows={4}
                  value={textA}
                  onChange={(e) => setTextA(e.target.value)}
                  className="w-full rounded-md border border-border bg-elevated/60 p-3 text-sm text-text-primary focus:border-accent focus:outline-none transition-colors"
                  placeholder="Enter sighting narrative A..."
                />
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-xs font-medium text-text-secondary">Date</label>
                  <input
                    type="text"
                    value={dateA}
                    onChange={(e) => setDateA(e.target.value)}
                    className="w-full rounded-md border border-border bg-elevated/60 p-2 text-xs text-text-primary font-mono mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-text-secondary">Latitude</label>
                  <input
                    type="text"
                    value={latA}
                    onChange={(e) => setLatA(e.target.value)}
                    className="w-full rounded-md border border-border bg-elevated/60 p-2 text-xs text-text-primary font-mono mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-text-secondary">Longitude</label>
                  <input
                    type="text"
                    value={lonA}
                    onChange={(e) => setLonA(e.target.value)}
                    className="w-full rounded-md border border-border bg-elevated/60 p-2 text-xs text-text-primary font-mono mt-1"
                  />
                </div>
              </div>
            </div>

            {/* Record B Input Card */}
            <div className="flex flex-col gap-4 rounded-xl border border-border bg-abyss/80 p-5 shadow-md">
              <div className="flex items-center justify-between border-b border-border/50 pb-3">
                <span className="text-sm font-semibold text-accent-bright flex items-center gap-2">
                  <Database className="h-4 w-4" /> Record B (Target Sighting)
                </span>
                <span className="text-xs text-text-muted font-mono">Input B</span>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-text-secondary">Narrative Description</label>
                <textarea
                  rows={4}
                  value={textB}
                  onChange={(e) => setTextB(e.target.value)}
                  className="w-full rounded-md border border-border bg-elevated/60 p-3 text-sm text-text-primary focus:border-accent focus:outline-none transition-colors"
                  placeholder="Enter sighting narrative B..."
                />
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-xs font-medium text-text-secondary">Date</label>
                  <input
                    type="text"
                    value={dateB}
                    onChange={(e) => setDateB(e.target.value)}
                    className="w-full rounded-md border border-border bg-elevated/60 p-2 text-xs text-text-primary font-mono mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-text-secondary">Latitude</label>
                  <input
                    type="text"
                    value={latB}
                    onChange={(e) => setLatB(e.target.value)}
                    className="w-full rounded-md border border-border bg-elevated/60 p-2 text-xs text-text-primary font-mono mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-text-secondary">Longitude</label>
                  <input
                    type="text"
                    value={lonB}
                    onChange={(e) => setLonB(e.target.value)}
                    className="w-full rounded-md border border-border bg-elevated/60 p-2 text-xs text-text-primary font-mono mt-1"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-center gap-4 py-2">
            <button
              onClick={handleCheckSimilarity}
              disabled={simpleLoading}
              className="flex items-center gap-2 rounded-lg bg-elevated px-6 py-3 font-semibold text-sm text-text-primary border border-border hover:bg-accent/20 hover:border-accent-bright transition-all duration-200 shadow-md disabled:opacity-50"
            >
              <Sparkles className="h-4 w-4 text-accent-bright" />
              Check Similarity (is_similar)
            </button>
            <button
              onClick={handleCheckDuplicate}
              disabled={simpleLoading}
              className="flex items-center gap-2 rounded-lg bg-accent px-6 py-3 font-semibold text-sm text-white hover:bg-accent-bright transition-all duration-200 shadow-lg shadow-accent/20 disabled:opacity-50"
            >
              <GitCompare className="h-4 w-4" />
              Check Duplicate (is_duplicate)
            </button>
          </div>

          {/* Similarity Result Panel */}
          {simResult && (
            <div className="flex flex-col gap-3 rounded-xl border border-accent/40 bg-gradient-to-br from-abyss via-elevated/40 to-abyss p-6 shadow-xl animate-fade-in">
              <div className="flex items-center justify-between border-b border-border/60 pb-3">
                <span className="text-sm font-bold tracking-wide text-text-primary flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-accent-bright" /> Semantic Similarity Verdict
                </span>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-bold ${
                    simResult.is_similar
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  }`}
                >
                  {simResult.is_similar ? 'HIGH SIMILARITY (is_similar = true)' : 'BELOW THRESHOLD (is_similar = false)'}
                </span>
              </div>
              <div className="flex items-center gap-6 mt-2">
                <div className="flex flex-col items-center justify-center p-4 rounded-xl bg-abyss border border-border w-44">
                  <span className="text-3xl font-extrabold text-accent-bright font-mono">
                    {(simResult.similarity_score * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-text-muted mt-1">Cosine Score</span>
                </div>
                <div className="flex-1 space-y-2">
                  <div className="w-full bg-abyss rounded-full h-3 border border-border overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        simResult.is_similar ? 'bg-gradient-to-r from-emerald-500 to-accent-bright' : 'bg-amber-500'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, simResult.similarity_score * 100))}%` }}
                    />
                  </div>
                  <p className="text-xs text-text-secondary">{simResult.reason}</p>
                </div>
              </div>
            </div>
          )}

          {/* Duplicate Screening Result Panel */}
          {dupResult && (
            <div className="flex flex-col gap-4 rounded-xl border border-accent/40 bg-gradient-to-br from-abyss via-elevated/50 to-abyss p-6 shadow-xl animate-fade-in">
              <div className="flex items-center justify-between border-b border-border/60 pb-3">
                <span className="text-sm font-bold tracking-wide text-text-primary flex items-center gap-2">
                  <GitCompare className="h-4 w-4 text-accent-bright" /> Multi-Gate Deduplication Verdict
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold ${
                      dupResult.confidence === 'HIGH'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    }`}
                  >
                    Confidence: {dupResult.confidence}
                  </span>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold ${
                      dupResult.is_duplicate
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                    }`}
                  >
                    {dupResult.is_duplicate ? 'TRUE DUPLICATE (is_duplicate = true)' : 'DISTINCT EVENT (is_duplicate = false)'}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex flex-col p-3 rounded-lg bg-abyss border border-border">
                  <span className="text-xs text-text-muted">Semantic Similarity</span>
                  <span className="text-lg font-bold text-accent-bright font-mono mt-1">
                    {(dupResult.metrics.similarity_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex flex-col p-3 rounded-lg bg-abyss border border-border">
                  <span className="text-xs text-text-muted">Spatial Separation</span>
                  <span className="text-lg font-bold text-text-primary font-mono mt-1">
                    {dupResult.metrics.haversine_km !== null
                      ? `${dupResult.metrics.haversine_km} km`
                      : 'Unknown'}
                  </span>
                </div>
                <div className="flex flex-col p-3 rounded-lg bg-abyss border border-border">
                  <span className="text-xs text-text-muted">Temporal Separation</span>
                  <span className="text-lg font-bold text-text-primary font-mono mt-1">
                    {dupResult.metrics.date_diff_days !== null
                      ? `${dupResult.metrics.date_diff_days} days`
                      : 'Unknown'}
                  </span>
                </div>
              </div>

              <div className="flex flex-col gap-1.5 mt-2 bg-abyss/60 p-4 rounded-lg border border-border/60">
                <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
                  Decision Audit Trail
                </span>
                {dupResult.reasons.map((r, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-text-primary">
                    <CheckCircle2 className="h-3.5 w-3.5 text-accent-bright shrink-0" />
                    <span>{r}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ADVANCED TAB */}
      {activeTab === 'advanced' && (
        <div className="flex flex-col gap-6">
          {/* Controls Panel */}
          <div className="flex flex-col gap-5 rounded-xl border border-border bg-abyss/90 p-6 shadow-md">
            <div className="flex items-center justify-between border-b border-border/60 pb-3">
              <span className="text-sm font-semibold text-accent-bright flex items-center gap-2">
                <Sliders className="h-4 w-4" /> Multi-Gate Screening Parameters
              </span>
              <span className="text-xs text-text-muted">Configure spatial/temporal gating and thresholding</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
              <div>
                <label className="text-xs font-medium text-text-secondary flex justify-between">
                  <span>Cosine Similarity Threshold</span>
                  <span className="font-mono text-accent-bright">{threshold}</span>
                </label>
                <input
                  type="range"
                  min="0.60"
                  max="0.95"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full mt-2 accent-accent"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-text-secondary flex justify-between">
                  <span>Max Temporal Diff (Days)</span>
                  <span className="font-mono text-accent-bright">{dateDiffDays} d</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="30"
                  step="1"
                  value={dateDiffDays}
                  onChange={(e) => setDateDiffDays(parseInt(e.target.value))}
                  className="w-full mt-2 accent-accent"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-text-secondary flex justify-between">
                  <span>Max Spatial Radius (Haversine)</span>
                  <span className="font-mono text-accent-bright">{maxKm} km</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="200"
                  step="5"
                  value={maxKm}
                  onChange={(e) => setMaxKm(parseFloat(e.target.value))}
                  className="w-full mt-2 accent-accent"
                />
              </div>

              <div className="flex flex-col justify-center">
                <label className="flex items-center gap-3 cursor-pointer p-2 rounded-lg border border-border bg-elevated/40 hover:bg-elevated">
                  <input
                    type="checkbox"
                    checked={useLlmJudge}
                    onChange={(e) => setUseLlmJudge(e.target.checked)}
                    className="h-4 w-4 accent-accent rounded"
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-semibold text-text-primary">Enable LLM Tier-3 Judge</span>
                    <span className="text-[10px] text-text-muted">Gemini 3.5 confirmation on borderline pairs</span>
                  </div>
                </label>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleRunAdvanced}
                disabled={advLoading}
                className="flex items-center gap-2 rounded-lg bg-accent px-6 py-2.5 font-semibold text-sm text-white hover:bg-accent-bright transition-all duration-200 shadow-lg shadow-accent/20 disabled:opacity-50"
              >
                <Play className="h-4 w-4 fill-current" />
                Run Advanced Deduplication & Clustering
              </button>
            </div>
          </div>

          {/* Results Display */}
          {advResult && (
            <div className="flex flex-col gap-6 animate-fade-in">
              {/* KPI Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="flex flex-col p-5 rounded-xl border border-accent/40 bg-gradient-to-br from-abyss to-elevated/60 shadow-lg">
                  <span className="text-xs text-text-secondary font-medium">Total Clusters Formed</span>
                  <span className="text-3xl font-extrabold text-accent-bright font-mono mt-1">
                    {advResult.summary.total_clusters}
                  </span>
                  <span className="text-[11px] text-text-muted mt-1">Union-Find connected components</span>
                </div>

                <div className="flex flex-col p-5 rounded-xl border border-border bg-abyss/80 shadow-md">
                  <span className="text-xs text-text-secondary font-medium">Rows in Clusters</span>
                  <span className="text-3xl font-extrabold text-text-primary font-mono mt-1">
                    {advResult.summary.rows_in_clusters}
                  </span>
                  <span className="text-[11px] text-text-muted mt-1">Total candidate reports linked</span>
                </div>

                <div className="flex flex-col p-5 rounded-xl border border-emerald-500/40 bg-gradient-to-br from-abyss to-emerald-950/20 shadow-lg">
                  <span className="text-xs text-emerald-400 font-medium">Redundant Rows Saved</span>
                  <span className="text-3xl font-extrabold text-emerald-300 font-mono mt-1">
                    {advResult.summary.redundant_rows_saved}
                  </span>
                  <span className="text-[11px] text-emerald-400/80 mt-1">Non-canonical duplicates pruned</span>
                </div>

                <div className="flex flex-col p-5 rounded-xl border border-border bg-abyss/80 shadow-md">
                  <span className="text-xs text-text-secondary font-medium">Flagged Pairs Displayed</span>
                  <span className="text-3xl font-extrabold text-text-primary font-mono mt-1">
                    {advResult.summary.flagged_pairs_count}
                  </span>
                  <span className="text-[11px] text-text-muted mt-1">Top high-confidence matches</span>
                </div>
              </div>

              {/* Flagged Table */}
              <div className="flex flex-col rounded-xl border border-border bg-abyss/90 shadow-lg overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-border bg-elevated/40">
                  <span className="text-sm font-semibold text-text-primary flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-accent-bright" /> High-Confidence Flagged Sighting Pairs
                  </span>
                  <span className="text-xs text-text-muted font-mono">
                    Showing {advResult.flagged_pairs.length} screened pairs
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-border text-text-secondary bg-abyss/60 font-mono">
                        <th className="p-3">Record A ID</th>
                        <th className="p-3">Record B ID</th>
                        <th className="p-3">Similarity</th>
                        <th className="p-3">LLM Verdict</th>
                        <th className="p-3">Justification & Context</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/40 text-text-primary">
                      {advResult.flagged_pairs.map((pair, idx) => (
                        <tr key={idx} className="hover:bg-elevated/30 transition-colors">
                          <td className="p-3 font-mono text-accent-bright font-semibold">{pair.id_a}</td>
                          <td className="p-3 font-mono text-text-secondary">{pair.id_b}</td>
                          <td className="p-3 font-mono">
                            <span className="px-2 py-0.5 rounded bg-accent/20 border border-accent/40 text-accent-bright">
                              {(pair.similarity * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="p-3">
                            <span
                              className={`px-2 py-0.5 rounded font-bold ${
                                pair.llm_same_event
                                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                  : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                              }`}
                            >
                              {pair.llm_same_event ? 'SAME EVENT' : 'DISTINCT'}
                            </span>
                          </td>
                          <td className="p-3 text-text-secondary">{pair.llm_reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* CROSS-DB TAB */}
      {activeTab === 'cross_db' && (
        <div className="flex flex-col gap-6 animate-fade-in">
          <div className="flex flex-col gap-4 rounded-xl border border-border bg-abyss/80 p-6 shadow-md">
            <h3 className="text-lg font-bold text-accent-bright flex items-center gap-2">
              <Database className="h-5 w-5" /> Cross-DB Similarity & Deduplication Pipeline
            </h3>
            <p className="text-xs text-text-secondary">
              Compare candidate records either within a single dataset or across two distinct databases using Harrier cosine similarities. Choose your screening workflow:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div
                onClick={() => setCrossPathMode('easy')}
                className={`cursor-pointer p-4 rounded-xl border transition-all ${
                  crossPathMode === 'easy'
                    ? 'border-accent-bright bg-accent/20 shadow-md'
                    : 'border-border/60 bg-elevated/40 hover:border-border'
                }`}
              >
                <div className="flex items-center gap-2 font-bold text-sm text-text-primary">
                  <Sparkles className="h-4 w-4 text-amber-400" /> a. Easy Path: Semantic Similarity Bins
                </div>
                <p className="text-xs text-text-muted mt-1">
                  Automatically classify candidate pairs into Exact Duplicates (≥0.88), Strong Similar (0.80-0.88), Moderate Similar (0.70-0.80), and Distinct (&lt;0.70).
                </p>
              </div>

              <div
                onClick={() => setCrossPathMode('hard')}
                className={`cursor-pointer p-4 rounded-xl border transition-all ${
                  crossPathMode === 'hard'
                    ? 'border-accent-bright bg-accent/20 shadow-md'
                    : 'border-border/60 bg-elevated/40 hover:border-border'
                }`}
              >
                <div className="flex items-center gap-2 font-bold text-sm text-text-primary">
                  <Sliders className="h-4 w-4 text-cyan-400" /> b. Hard Path: Interactive Multi-Gate Overview
                </div>
                <p className="text-xs text-text-muted mt-1">
                  Manual batch oversight with dynamic toggle buttons for Similar Text, Similar Date, Similar Location, Similar Both, or Similar All (Full Convergence).
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-border/50">
              <span className="text-xs font-semibold text-text-secondary">Comparison Scope:</span>
              <button
                onClick={() => setCrossCmpMode('within')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                  crossCmpMode === 'within' ? 'bg-accent/30 border-accent-bright text-white' : 'bg-elevated/40 border-border text-text-muted'
                }`}
              >
                Within Dataset (DB1 vs DB1)
              </button>
              <button
                onClick={() => setCrossCmpMode('between')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                  crossCmpMode === 'between' ? 'bg-accent/30 border-accent-bright text-white' : 'bg-elevated/40 border-border text-text-muted'
                }`}
              >
                Between Datasets (DB1 vs DB2)
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-border/50">
              <span className="text-xs font-semibold text-text-secondary">Data Source:</span>
              <button
                onClick={() => setDataSourceMode('real_dataset')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-1.5 ${
                  dataSourceMode === 'real_dataset'
                    ? 'bg-emerald-500/30 border-emerald-400 text-white font-bold'
                    : 'bg-elevated/40 border-border text-text-muted hover:border-emerald-500/40'
                }`}
              >
                <Database className="h-3.5 w-3.5 text-emerald-400" />
                <span>Real Loaded Dataset ({dataLoaded && data?.rows ? `${data.rows.length.toLocaleString()} rows` : '0 loaded'})</span>
              </button>
              <button
                onClick={() => setDataSourceMode('sample_mock')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-1.5 ${
                  dataSourceMode === 'sample_mock'
                    ? 'bg-amber-500/30 border-amber-400 text-white font-bold'
                    : 'bg-elevated/40 border-border text-text-muted hover:border-amber-500/40'
                }`}
              >
                <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                <span>Interactive Sample Mockup (5 test cases)</span>
              </button>
            </div>


            <div className="flex justify-start pt-2">
              <button
                onClick={handleRunCrossDb}
                disabled={crossLoading}
                className="flex items-center gap-2 rounded-lg bg-accent px-6 py-3 font-semibold text-sm text-white hover:bg-accent-bright transition-all shadow-lg disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {crossLoading ? 'Running Harrier Cross-DB Pipeline...' : 'Execute Cross-DB Similarity Pipeline'}
              </button>
            </div>
          </div>

          {crossResult && (() => {
            const filteredPairs = crossResult.pairs.filter((p) => {
              if (crossPathMode === 'easy' && crossBinFilter !== 'All') {
                return p.bin === crossBinFilter;
              }
              if (crossPathMode === 'hard' && crossGateFilter !== 'show_all') {
                if (crossGateFilter === 'similar_text') return p.flags.is_similar_text;
                if (crossGateFilter === 'similar_date') return p.flags.is_similar_date;
                if (crossGateFilter === 'similar_location') return p.flags.is_similar_location;
                if (crossGateFilter === 'similar_both') return p.flags.is_similar_both;
                if (crossGateFilter === 'similar_all') return p.flags.is_similar_all;
              }
              return true;
            });
            const activePair = filteredPairs[selectedPairIndex] || filteredPairs[0] || crossResult.pairs[0];

            return (
              <div className="flex flex-col gap-6 animate-fade-in">
                {crossPathMode === 'easy' ? (
                  <div className="flex flex-col gap-6">
                    {/* Easy Path Bin Summary Cards */}
                    <div className="flex flex-col gap-4 rounded-xl border border-border bg-abyss/80 p-6 shadow-md">
                      <h4 className="text-sm font-bold text-accent-bright flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-amber-400" /> Semantic Similarity Bins Overview
                      </h4>
                      <p className="text-xs text-text-muted">Candidate pairs classified by Harrier semantic cosine thresholds:</p>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div
                          onClick={() => setCrossBinFilter(crossBinFilter === 'exact_duplicate' ? 'All' : 'exact_duplicate')}
                          className={`cursor-pointer p-4 rounded-xl border transition-all flex flex-col ${
                            crossBinFilter === 'exact_duplicate'
                              ? 'bg-rose-500/20 border-rose-400 shadow-md scale-[1.02]'
                              : 'bg-elevated/30 border-border/60 hover:border-rose-500/40'
                          }`}
                        >
                          <span className="text-xs font-semibold text-rose-300">Exact Duplicate (≥0.88)</span>
                          <span className="text-2xl font-bold font-mono text-white mt-1">{crossResult.summary.bins.exact_duplicate}</span>
                        </div>
                        <div
                          onClick={() => setCrossBinFilter(crossBinFilter === 'strong_similar' ? 'All' : 'strong_similar')}
                          className={`cursor-pointer p-4 rounded-xl border transition-all flex flex-col ${
                            crossBinFilter === 'strong_similar'
                              ? 'bg-amber-500/20 border-amber-400 shadow-md scale-[1.02]'
                              : 'bg-elevated/30 border-border/60 hover:border-amber-500/40'
                          }`}
                        >
                          <span className="text-xs font-semibold text-amber-300">Strong Similar (0.80-0.88)</span>
                          <span className="text-2xl font-bold font-mono text-white mt-1">{crossResult.summary.bins.strong_similar}</span>
                        </div>
                        <div
                          onClick={() => setCrossBinFilter(crossBinFilter === 'moderate_similar' ? 'All' : 'moderate_similar')}
                          className={`cursor-pointer p-4 rounded-xl border transition-all flex flex-col ${
                            crossBinFilter === 'moderate_similar'
                              ? 'bg-cyan-500/20 border-cyan-400 shadow-md scale-[1.02]'
                              : 'bg-elevated/30 border-border/60 hover:border-cyan-500/40'
                          }`}
                        >
                          <span className="text-xs font-semibold text-cyan-300">Moderate Similar (0.70-0.80)</span>
                          <span className="text-2xl font-bold font-mono text-white mt-1">{crossResult.summary.bins.moderate_similar}</span>
                        </div>
                        <div
                          onClick={() => setCrossBinFilter(crossBinFilter === 'distinct' ? 'All' : 'distinct')}
                          className={`cursor-pointer p-4 rounded-xl border transition-all flex flex-col ${
                            crossBinFilter === 'distinct'
                              ? 'bg-emerald-500/20 border-emerald-400 shadow-md scale-[1.02]'
                              : 'bg-elevated/30 border-border/60 hover:border-emerald-500/40'
                          }`}
                        >
                          <span className="text-xs font-semibold text-emerald-300">Distinct (&lt;0.70)</span>
                          <span className="text-2xl font-bold font-mono text-white mt-1">{crossResult.summary.bins.distinct}</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 pt-2">
                        <span className="text-xs font-medium text-text-secondary">Filter Table by Bin:</span>
                        {['All', 'exact_duplicate', 'strong_similar', 'moderate_similar', 'distinct'].map((b) => (
                          <button
                            key={b}
                            onClick={() => setCrossBinFilter(b)}
                            className={`px-3 py-1 rounded text-xs font-mono border transition-colors ${
                              crossBinFilter === b ? 'bg-accent text-white border-accent-bright' : 'bg-elevated/40 text-text-secondary border-border'
                            }`}
                          >
                            {b}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Simple Table for Easy Path */}
                    <div className="flex flex-col rounded-xl border border-border bg-abyss/90 shadow-lg overflow-hidden">
                      <div className="flex items-center justify-between p-4 border-b border-border bg-elevated/40">
                        <span className="text-sm font-semibold text-text-primary flex items-center gap-2">
                          <Database className="h-4 w-4 text-accent-bright" /> Candidate Pair Evaluation Matrix
                        </span>
                        <span className="text-xs text-text-muted font-mono">
                          Showing {filteredPairs.length} pairs
                        </span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b border-border text-text-secondary bg-abyss/60 font-mono">
                              <th className="p-3">Record A ID</th>
                              <th className="p-3">Record B ID</th>
                              <th className="p-3">Score</th>
                              <th className="p-3">Bin Category</th>
                              <th className="p-3">Distance</th>
                              <th className="p-3">Date Gap</th>
                              <th className="p-3">Multi-Gate Status</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border/40 text-text-primary">
                            {filteredPairs.map((pair, idx) => (
                              <tr key={idx} className="hover:bg-elevated/30 transition-colors">
                                <td className="p-3 font-mono text-accent-bright font-semibold">{pair.id_a}</td>
                                <td className="p-3 font-mono text-text-secondary">{pair.id_b}</td>
                                <td className="p-3 font-mono font-bold">{(pair.similarity * 100).toFixed(1)}%</td>
                                <td className="p-3 font-mono">
                                  <span className="px-2 py-0.5 rounded bg-elevated border border-border text-xs">
                                    {pair.bin}
                                  </span>
                                </td>
                                <td className="p-3 font-mono">{pair.haversine_km !== null ? `${pair.haversine_km} km` : '—'}</td>
                                <td className="p-3 font-mono">{pair.date_diff_days !== null ? `${pair.date_diff_days} days` : '—'}</td>
                                <td className="p-3">
                                  <div className="flex gap-1">
                                    <span title="Similar Text" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_text ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'opacity-25'}`}>TXT</span>
                                    <span title="Similar Date" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_date ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'opacity-25'}`}>DAT</span>
                                    <span title="Similar Location" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_location ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' : 'opacity-25'}`}>LOC</span>
                                    <span title="Full Duplicate" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_all ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'opacity-25'}`}>ALL</span>
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Hard Path: Split View Data-Explorer Aligned Database Rows */
                  <div className="flex flex-col gap-6">
                    {/* Interactive Multi-Gate Filter Buttons */}
                    <div className="flex flex-col gap-4 rounded-xl border border-border bg-abyss/80 p-6 shadow-md">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <h4 className="text-sm font-bold text-text-primary flex items-center gap-2">
                            <Sliders className="h-4 w-4 text-cyan-400" /> Interactive Multi-Gate Split View & Filter Overview
                          </h4>
                          <p className="text-xs text-text-muted mt-0.5">Select a multi-gate filter on the left to slice candidate pairs, and inspect side-by-side aligned database rows on the right:</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-text-secondary font-medium">Right Panel View:</span>
                          <button
                            onClick={() => setSplitViewMode('single_pair')}
                            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                              splitViewMode === 'single_pair' ? 'bg-accent text-white border-accent-bright shadow' : 'bg-elevated/40 text-text-muted border-border hover:text-white'
                            }`}
                          >
                            Side-by-Side Aligned Inspector
                          </button>
                          <button
                            onClick={() => setSplitViewMode('batch_table')}
                            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                              splitViewMode === 'batch_table' ? 'bg-accent text-white border-accent-bright shadow' : 'bg-elevated/40 text-text-muted border-border hover:text-white'
                            }`}
                          >
                            Batch Data-Explorer Table ({filteredPairs.length})
                          </button>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-2 pt-2 border-t border-border/40">
                        <button
                          onClick={() => { setCrossGateFilter('similar_text'); setSelectedPairIndex(0); }}
                          className={`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-2 ${
                            crossGateFilter === 'similar_text' ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-md' : 'bg-elevated/40 border-border text-text-secondary hover:border-cyan-500/40'
                          }`}
                        >
                          📝 Similar Text Only ({crossResult.summary.gate_counts.similar_text})
                        </button>
                        <button
                          onClick={() => { setCrossGateFilter('similar_date'); setSelectedPairIndex(0); }}
                          className={`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-2 ${
                            crossGateFilter === 'similar_date' ? 'bg-amber-500/20 border-amber-400 text-amber-200 shadow-md' : 'bg-elevated/40 border-border text-text-secondary hover:border-amber-500/40'
                          }`}
                        >
                          📅 Similar Date Only ({crossResult.summary.gate_counts.similar_date})
                        </button>
                        <button
                          onClick={() => { setCrossGateFilter('similar_location'); setSelectedPairIndex(0); }}
                          className={`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-2 ${
                            crossGateFilter === 'similar_location' ? 'bg-purple-500/20 border-purple-400 text-purple-200 shadow-md' : 'bg-elevated/40 border-border text-text-secondary hover:border-purple-500/40'
                          }`}
                        >
                          📍 Similar Location Only ({crossResult.summary.gate_counts.similar_location})
                        </button>
                        <button
                          onClick={() => { setCrossGateFilter('similar_both'); setSelectedPairIndex(0); }}
                          className={`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-2 ${
                            crossGateFilter === 'similar_both' ? 'bg-pink-500/20 border-pink-400 text-pink-200 shadow-md' : 'bg-elevated/40 border-border text-text-secondary hover:border-pink-500/40'
                          }`}
                        >
                          ⚡ Similar Both (Spatial+Temporal) ({crossResult.summary.gate_counts.similar_both})
                        </button>
                        <button
                          onClick={() => { setCrossGateFilter('similar_all'); setSelectedPairIndex(0); }}
                          className={`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-2 ${
                            crossGateFilter === 'similar_all' ? 'bg-emerald-500/20 border-emerald-400 text-emerald-200 shadow-md' : 'bg-elevated/40 border-border text-text-secondary hover:border-emerald-500/40'
                          }`}
                        >
                          🎯 Similar All / Duplicates ({crossResult.summary.gate_counts.similar_all})
                        </button>
                        <button
                          onClick={() => { setCrossGateFilter('show_all'); setSelectedPairIndex(0); }}
                          className={`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-2 ${
                            crossGateFilter === 'show_all' ? 'bg-accent/30 border-accent-bright text-white shadow-md' : 'bg-elevated/40 border-border text-text-secondary hover:border-accent/50'
                          }`}
                        >
                          🌐 Show All Evaluated ({crossResult.pairs.length})
                        </button>
                      </div>
                    </div>

                    {/* Split View Container */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                      {/* Left side: Simplified Output List */}
                      <div className="lg:col-span-4 flex flex-col rounded-xl border border-border bg-abyss/90 shadow-lg overflow-hidden">
                        <div className="flex items-center justify-between p-3.5 border-b border-border bg-elevated/40">
                          <span className="text-xs font-semibold text-text-primary flex items-center gap-1.5">
                            <Layers className="h-3.5 w-3.5 text-accent-bright" /> Simplified Candidate Pairs
                          </span>
                          <span className="px-2 py-0.5 rounded-full bg-accent/20 border border-accent/40 text-[11px] font-mono text-accent-bright">
                            {filteredPairs.length} matches
                          </span>
                        </div>

                        <div className="max-h-[580px] overflow-y-auto divide-y divide-border/40">
                          {filteredPairs.length === 0 ? (
                            <div className="p-6 text-center text-xs text-text-muted">
                              No candidate pairs match the active multi-gate filter.
                            </div>
                          ) : (
                            filteredPairs.map((pair, idx) => {
                              const isSelected = activePair && activePair.id_a === pair.id_a && activePair.id_b === pair.id_b;
                              return (
                                <div
                                  key={idx}
                                  onClick={() => setSelectedPairIndex(idx)}
                                  className={`p-3.5 cursor-pointer transition-all flex flex-col gap-2 ${
                                    isSelected
                                      ? 'bg-accent/25 border-l-4 border-accent-bright'
                                      : 'hover:bg-elevated/40 border-l-4 border-transparent'
                                  }`}
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="font-mono text-xs font-bold text-text-primary flex items-center gap-1.5">
                                      <span className="text-accent-bright">{pair.id_a}</span>
                                      <span className="text-text-muted">↔</span>
                                      <span className="text-text-secondary">{pair.id_b}</span>
                                    </span>
                                    <span className="px-1.5 py-0.5 rounded bg-accent/30 text-accent-bright font-mono text-[11px] font-bold">
                                      {(pair.similarity * 100).toFixed(1)}%
                                    </span>
                                  </div>

                                  <div className="flex items-center justify-between text-[11px] text-text-muted">
                                    <div className="flex gap-1">
                                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_text ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'opacity-25'}`}>TXT</span>
                                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_date ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'opacity-25'}`}>DAT</span>
                                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_location ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' : 'opacity-25'}`}>LOC</span>
                                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_all ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'opacity-25'}`}>ALL</span>
                                    </div>
                                    <span className="font-mono text-[10px]">
                                      {pair.haversine_km !== null ? `${pair.haversine_km}km` : ''} 
                                      {pair.date_diff_days !== null ? ` | ${pair.date_diff_days}d` : ''}
                                    </span>
                                  </div>
                                </div>
                              );
                            })
                          )}
                        </div>
                      </div>

                      {/* Right side: Aligned Database Rows Data-Explorer */}
                      <div className="lg:col-span-8 flex flex-col rounded-xl border border-border bg-abyss/90 shadow-lg overflow-hidden">
                        {splitViewMode === 'single_pair' ? (
                          activePair ? (
                            <div className="flex flex-col">
                              {/* Inspector Header */}
                              <div className="flex flex-wrap items-center justify-between p-4 border-b border-border bg-elevated/40 gap-3">
                                <div className="flex flex-col">
                                  <span className="text-sm font-bold text-text-primary flex items-center gap-2">
                                    <Database className="h-4 w-4 text-emerald-400" /> Aligned Database Rows — Pair #{activePair.id_a} vs #{activePair.id_b}
                                  </span>
                                  <span className="text-xs text-text-muted mt-0.5">
                                    Comparing Record A and Record B side-by-side aligned directly from database columns
                                  </span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  <span className="px-2.5 py-1 rounded bg-cyan-500/20 border border-cyan-500/40 text-cyan-200 text-xs font-mono font-bold">
                                    Sim: {(activePair.similarity * 100).toFixed(1)}%
                                  </span>
                                  {activePair.haversine_km !== null && (
                                    <span className="px-2.5 py-1 rounded bg-purple-500/20 border border-purple-500/40 text-purple-200 text-xs font-mono font-bold">
                                      Dist: {activePair.haversine_km} km
                                    </span>
                                  )}
                                  {activePair.date_diff_days !== null && (
                                    <span className="px-2.5 py-1 rounded bg-amber-500/20 border border-amber-500/40 text-amber-200 text-xs font-mono font-bold">
                                      Time Gap: {activePair.date_diff_days} days
                                    </span>
                                  )}
                                </div>
                              </div>

                              {/* Aligned Side-by-Side Comparison Table */}
                              <div className="overflow-x-auto max-h-[580px] overflow-y-auto">
                                <table className="w-full text-left border-collapse text-xs">
                                  <thead>
                                    <tr className="border-b border-border text-text-secondary bg-abyss/60 font-mono sticky top-0 backdrop-blur-md">
                                      <th className="p-3.5 w-1/4">Database Attribute</th>
                                      <th className="p-3.5 w-3/8 border-l border-border/50 text-accent-bright">
                                        Database Row A (#{activePair.id_a})
                                      </th>
                                      <th className="p-3.5 w-3/8 border-l border-border/50 text-text-primary">
                                        Database Row B (#{activePair.id_b})
                                      </th>
                                      <th className="p-3.5 w-1/6 border-l border-border/50">Multi-Gate Alignment</th>
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-border/40 text-text-primary">
                                    {/* Record ID Row */}
                                    <tr className="hover:bg-elevated/30 transition-colors">
                                      <td className="p-3.5 font-semibold font-mono text-text-secondary bg-elevated/10">Primary Record ID</td>
                                      <td className="p-3.5 font-mono text-accent-bright border-l border-border/50 font-bold">{activePair.id_a}</td>
                                      <td className="p-3.5 font-mono text-text-primary border-l border-border/50 font-bold">{activePair.id_b}</td>
                                      <td className="p-3.5 border-l border-border/50 text-text-muted font-mono">Candidate Pair</td>
                                    </tr>

                                    {/* Witness Notes / Narrative Row */}
                                    <tr className="hover:bg-elevated/30 transition-colors bg-accent/5">
                                      <td className="p-3.5 font-semibold text-text-primary align-top bg-elevated/10">
                                        Witness Notes / Narrative Text
                                      </td>
                                      <td className="p-3.5 text-text-primary border-l border-border/50 leading-relaxed align-top">
                                        {activePair.row_a?.witness_notes || activePair.row_a?.narrative || activePair.text_a_preview || '—'}
                                      </td>
                                      <td className="p-3.5 text-text-primary border-l border-border/50 leading-relaxed align-top">
                                        {activePair.row_b?.witness_notes || activePair.row_b?.narrative || activePair.text_b_preview || '—'}
                                      </td>
                                      <td className="p-3.5 border-l border-border/50 align-top">
                                        <span className={`px-2 py-1 rounded text-xs font-mono font-bold block text-center ${
                                          activePair.flags.is_similar_text ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'bg-elevated text-text-muted border border-border'
                                        }`}>
                                          {activePair.flags.is_similar_text ? '✅ GATE PASSED' : '❌ BELOW THRESH'}
                                          <span className="block text-[10px] font-normal mt-0.5">{(activePair.similarity * 100).toFixed(1)}% Cosine</span>
                                        </span>
                                      </td>
                                    </tr>

                                    {/* Date & Time Row */}
                                    <tr className="hover:bg-elevated/30 transition-colors">
                                      <td className="p-3.5 font-semibold text-text-secondary align-top bg-elevated/10">Timestamp / Date</td>
                                      <td className="p-3.5 font-mono text-text-primary border-l border-border/50 align-top">
                                        {String(activePair.row_a?.date_time || activePair.row_a?.date || '—')}
                                      </td>
                                      <td className="p-3.5 font-mono text-text-primary border-l border-border/50 align-top">
                                        {String(activePair.row_b?.date_time || activePair.row_b?.date || '—')}
                                      </td>
                                      <td className="p-3.5 border-l border-border/50 align-top">
                                        {activePair.date_diff_days !== null ? (
                                          <span className={`px-2 py-1 rounded text-xs font-mono font-bold block text-center ${
                                            activePair.flags.is_similar_date ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-elevated text-text-muted border border-border'
                                          }`}>
                                            {activePair.flags.is_similar_date ? '✅ GATE PASSED' : '❌ OUT OF RANGE'}
                                            <span className="block text-[10px] font-normal mt-0.5">Δ {activePair.date_diff_days} days</span>
                                          </span>
                                        ) : (
                                          <span className="text-text-muted font-mono text-center block">No Date Data</span>
                                        )}
                                      </td>
                                    </tr>

                                    {/* Coordinates Row */}
                                    <tr className="hover:bg-elevated/30 transition-colors">
                                      <td className="p-3.5 font-semibold text-text-secondary align-top bg-elevated/10">Spatial Coordinates</td>
                                      <td className="p-3.5 font-mono text-text-primary border-l border-border/50 align-top">
                                        {activePair.row_a?.latitude !== undefined ? `${Number(activePair.row_a.latitude).toFixed(4)}, ${Number(activePair.row_a.longitude).toFixed(4)}` : '—'}
                                      </td>
                                      <td className="p-3.5 font-mono text-text-primary border-l border-border/50 align-top">
                                        {activePair.row_b?.latitude !== undefined ? `${Number(activePair.row_b.latitude).toFixed(4)}, ${Number(activePair.row_b.longitude).toFixed(4)}` : '—'}
                                      </td>
                                      <td className="p-3.5 border-l border-border/50 align-top">
                                        {activePair.haversine_km !== null ? (
                                          <span className={`px-2 py-1 rounded text-xs font-mono font-bold block text-center ${
                                            activePair.flags.is_similar_location ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' : 'bg-elevated text-text-muted border border-border'
                                          }`}>
                                            {activePair.flags.is_similar_location ? '✅ GATE PASSED' : '❌ OUT OF RANGE'}
                                            <span className="block text-[10px] font-normal mt-0.5">Δ {activePair.haversine_km} km</span>
                                          </span>
                                        ) : (
                                          <span className="text-text-muted font-mono text-center block">No Lat/Lon</span>
                                        )}
                                      </td>
                                    </tr>

                                    {/* Additional Aligned Database Columns */}
                                    {(() => {
                                      const rowAKeys = activePair.row_a ? Object.keys(activePair.row_a) : [];
                                      const rowBKeys = activePair.row_b ? Object.keys(activePair.row_b) : [];
                                      const ignoreKeys = ['id', 'locus_tag', 'case_id', 'witness_notes', 'narrative', 'description', 'summary', 'text', 'date_time', 'date', 'datetime', 'latitude', 'lat', 'longitude', 'lon', 'lng'];
                                      const allOtherKeys = Array.from(new Set([...rowAKeys, ...rowBKeys])).filter(k => !ignoreKeys.includes(k.toLowerCase()));

                                      return allOtherKeys.map((key) => {
                                        const valA = activePair.row_a?.[key] !== undefined ? String(activePair.row_a[key]) : '—';
                                        const valB = activePair.row_b?.[key] !== undefined ? String(activePair.row_b[key]) : '—';
                                        const isExact = valA !== '—' && valB !== '—' && valA.toLowerCase() === valB.toLowerCase();

                                        return (
                                          <tr key={key} className="hover:bg-elevated/30 transition-colors">
                                            <td className="p-3.5 font-semibold text-text-secondary align-top bg-elevated/10 capitalize">
                                              {key.replace(/_/g, ' ')}
                                            </td>
                                            <td className="p-3.5 text-text-primary border-l border-border/50 align-top font-mono">
                                              {valA}
                                            </td>
                                            <td className="p-3.5 text-text-primary border-l border-border/50 align-top font-mono">
                                              {valB}
                                            </td>
                                            <td className="p-3.5 border-l border-border/50 align-top">
                                              {isExact ? (
                                                <span className="px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-[11px] font-mono font-bold block text-center">
                                                  ✅ Exact Match
                                                </span>
                                              ) : (
                                                <span className="text-text-muted text-[11px] font-mono text-center block">
                                                  Differs
                                                </span>
                                              )}
                                            </td>
                                          </tr>
                                        );
                                      });
                                    })()}

                                    {/* Final Multi-Gate Verdict Row */}
                                    <tr className="bg-abyss/80 border-t-2 border-border font-bold">
                                      <td className="p-4 text-accent-bright bg-elevated/20">Full Multi-Gate Verdict</td>
                                      <td colSpan={2} className="p-4 border-l border-border/50 text-white">
                                        {activePair.flags.is_similar_all ? (
                                          <span className="flex items-center gap-2 text-emerald-300">
                                            <CheckCircle2 className="h-4 w-4" /> Candidate Duplicate — Triggered Text + Spatial + Temporal Convergence
                                          </span>
                                        ) : activePair.flags.is_similar_both ? (
                                          <span className="flex items-center gap-2 text-pink-300">
                                            <Sliders className="h-4 w-4" /> Strong Spatial + Temporal Correlation (Distinct Narrative)
                                          </span>
                                        ) : (
                                          <span className="text-text-secondary">
                                            Partial Match — Classified as {activePair.bin.replace(/_/g, ' ')}
                                          </span>
                                        )}
                                      </td>
                                      <td className="p-4 border-l border-border/50 text-center">
                                        <span className={`px-3 py-1 rounded-full text-xs font-bold font-mono ${
                                          activePair.flags.is_similar_all ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400' : 'bg-elevated text-text-secondary border border-border'
                                        }`}>
                                          {activePair.flags.is_similar_all ? 'DUPLICATE' : 'DISTINCT'}
                                        </span>
                                      </td>
                                    </tr>
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          ) : (
                            <div className="p-12 text-center text-sm text-text-muted">
                              Select a candidate pair from the left panel to inspect aligned database rows side-by-side.
                            </div>
                          )
                        ) : (
                          /* Batch Data-Explorer Table View across all filtered pairs */
                          <div className="flex flex-col">
                            <div className="flex items-center justify-between p-4 border-b border-border bg-elevated/40">
                              <span className="text-sm font-bold text-text-primary flex items-center gap-2">
                                <Database className="h-4 w-4 text-accent-bright" /> Batch Data-Explorer Aligned Table
                              </span>
                              <span className="text-xs text-text-muted font-mono">
                                Showing all {filteredPairs.length} filtered candidate pairs
                              </span>
                            </div>

                            <div className="overflow-x-auto max-h-[580px] overflow-y-auto">
                              <table className="w-full text-left border-collapse text-xs">
                                <thead>
                                  <tr className="border-b border-border text-text-secondary bg-abyss/60 font-mono sticky top-0 backdrop-blur-md">
                                    <th className="p-3">Pair ID (A ↔ B)</th>
                                    <th className="p-3">Harrier Sim</th>
                                    <th className="p-3">Aligned Dates (Row A vs Row B)</th>
                                    <th className="p-3">Aligned Coordinates (Row A vs Row B)</th>
                                    <th className="p-3">Side-by-Side Witness Notes Preview</th>
                                    <th className="p-3">Multi-Gate Status</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-border/40 text-text-primary">
                                  {filteredPairs.map((pair, idx) => (
                                    <tr
                                      key={idx}
                                      onClick={() => { setSelectedPairIndex(idx); setSplitViewMode('single_pair'); }}
                                      className="hover:bg-elevated/30 transition-colors cursor-pointer"
                                    >
                                      <td className="p-3 font-mono font-bold text-accent-bright whitespace-nowrap">
                                        {pair.id_a} <span className="text-text-muted">↔</span> {pair.id_b}
                                      </td>
                                      <td className="p-3 font-mono font-bold">
                                        {(pair.similarity * 100).toFixed(1)}%
                                      </td>
                                      <td className="p-3 font-mono text-[11px] whitespace-nowrap">
                                        <div><span className="text-emerald-400">A:</span> {String(pair.row_a?.date_time || pair.row_a?.date || '—')}</div>
                                        <div><span className="text-cyan-400">B:</span> {String(pair.row_b?.date_time || pair.row_b?.date || '—')}</div>
                                        <div className="text-text-muted text-[10px] mt-0.5">Gap: {pair.date_diff_days ?? '—'}d</div>
                                      </td>
                                      <td className="p-3 font-mono text-[11px] whitespace-nowrap">
                                        <div><span className="text-emerald-400">A:</span> {pair.row_a?.latitude !== undefined ? `${Number(pair.row_a.latitude).toFixed(2)}, ${Number(pair.row_a.longitude).toFixed(2)}` : '—'}</div>
                                        <div><span className="text-cyan-400">B:</span> {pair.row_b?.latitude !== undefined ? `${Number(pair.row_b.latitude).toFixed(2)}, ${Number(pair.row_b.longitude).toFixed(2)}` : '—'}</div>
                                        <div className="text-text-muted text-[10px] mt-0.5">Dist: {pair.haversine_km ?? '—'}km</div>
                                      </td>
                                      <td className="p-3 text-[11px] max-w-md">
                                        <div className="line-clamp-1 border-b border-border/30 pb-1 mb-1"><span className="font-bold text-emerald-400">A:</span> {pair.text_a_preview}</div>
                                        <div className="line-clamp-1"><span className="font-bold text-cyan-400">B:</span> {pair.text_b_preview}</div>
                                      </td>
                                      <td className="p-3">
                                        <div className="flex gap-1">
                                          <span title="Similar Text" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_text ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'opacity-25'}`}>TXT</span>
                                          <span title="Similar Date" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_date ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'opacity-25'}`}>DAT</span>
                                          <span title="Similar Location" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_location ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' : 'opacity-25'}`}>LOC</span>
                                          <span title="Full Duplicate" className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${pair.flags.is_similar_all ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'opacity-25'}`}>ALL</span>
                                        </div>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}

