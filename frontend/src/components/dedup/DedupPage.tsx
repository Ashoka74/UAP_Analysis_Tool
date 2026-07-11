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
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';
import type {
  SimpleSimilarityResponse,
  SimpleDuplicateResponse,
  AdvancedDedupResponse,
} from '../../types';

export function DedupPage() {
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced'>('simple');

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

  // API Call: Check Similarity
  const handleCheckSimilarity = async () => {
    setSimpleLoading(true);
    setError(null);
    setSimResult(null);
    setDupResult(null);
    try {
      const res = await fetch('/api/dedup/simple/similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_a: textA, text_b: textB }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: SimpleSimilarityResponse = await res.json();
      setSimResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
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
      const res = await fetch('/api/dedup/simple/duplicate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          record_a: { narrative: textA, lat: latA, lon: lonA, date: dateA },
          record_b: { narrative: textB, lat: latB, lon: lonB, date: dateB },
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: SimpleDuplicateResponse = await res.json();
      setDupResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSimpleLoading(false);
    }
  };

  // API Call: Run Advanced Dedup
  const handleRunAdvanced = async () => {
    setAdvLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/dedup/advanced/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threshold,
          date_diff_days: dateDiffDays,
          max_km: maxKm,
          use_llm_judge: useLlmJudge,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: AdvancedDedupResponse = await res.json();
      setAdvResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
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
    </div>
  );
}
