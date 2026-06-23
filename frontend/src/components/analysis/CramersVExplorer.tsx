import { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Play, AlertTriangle, Grid3x3, Layers, BarChart3 } from 'lucide-react';
import { api } from '../../api/client';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { XGBoostResults } from './XGBoostResults';
import type { CramersVResponse, ContingencyResponse, ColumnGroup, XGBoostResult } from '../../types';

interface Props {
  source: 'dataset' | 'parsed';
  // When provided, XGBoost feature importance computed from the selected columns
  // is handed off to the parent (e.g. the Feature Importance tab) instead of
  // rendering inline.
  onXgboost?: (results: Record<string, XGBoostResult>) => void;
}

export function CramersVExplorer({ source, onXgboost }: Props) {
  const [report, setReport] = useState<CramersVResponse | null>(null);
  const [contingency, setContingency] = useState<ContingencyResponse | null>(null);
  const [pair, setPair] = useState<{ a: string; b: string } | null>(null);

  const [dropMissing, setDropMissing] = useState(false);
  const [excludeTrivial, setExcludeTrivial] = useState(true);
  const [strong, setStrong] = useState(0.3);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // XGBoost feature importance run directly on the selected columns.
  const [xgbLoading, setXgbLoading] = useState(false);
  const [localXgb, setLocalXgb] = useState<Record<string, XGBoostResult> | null>(null);

  // Eligible categorical columns, grouped by their dotted parent (e.g. craft.*).
  const [groups, setGroups] = useState<ColumnGroup[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [groupsError, setGroupsError] = useState<string | null>(null);

  // Load the parent groups up front — cheap (cardinality only, no matrix), so the
  // selector is usable before the first Compute. Defaults to all eligible columns,
  // which matches the explorer's prior "compute everything" behavior.
  useEffect(() => {
    let cancelled = false;
    setGroupsError(null);
    api
      .columnGroups({ source })
      .then((res) => {
        if (cancelled) return;
        setGroups(res.groups);
        setSelected(new Set(res.eligible));
      })
      .catch((e) => {
        if (!cancelled) setGroupsError(e instanceof Error ? e.message : 'Could not load columns');
      });
    return () => {
      cancelled = true;
    };
  }, [source]);

  // Flattened in grouped order so the matrix keeps related columns adjacent.
  const orderedEligible = useMemo(() => groups.flatMap((g) => g.columns), [groups]);
  const nestedGroups = groups.filter((g) => g.nested);
  const standaloneCols = groups.filter((g) => !g.nested).flatMap((g) => g.columns);

  const toggleCol = (c: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });

  const toggleGroup = (g: ColumnGroup) =>
    setSelected((prev) => {
      const next = new Set(prev);
      const allOn = g.columns.every((c) => next.has(c));
      for (const c of g.columns) {
        if (allOn) next.delete(c);
        else next.add(c);
      }
      return next;
    });

  const selectAll = () => setSelected(new Set(orderedEligible));
  const clearAll = () => setSelected(new Set());

  const run = async () => {
    if (orderedEligible.length && selected.size < 2) {
      setError('Select at least two columns (or whole parent groups) to compute associations.');
      return;
    }
    setLoading(true);
    setError(null);
    setContingency(null);
    setPair(null);
    try {
      const cols = orderedEligible.filter((c) => selected.has(c));
      const res = await api.cramersV({
        source,
        columns: cols.length ? cols : undefined,
        drop_missing: dropMissing,
        exclude_trivial: excludeTrivial,
        strong_threshold: strong,
      });
      setReport(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Cramér’s V failed');
    } finally {
      setLoading(false);
    }
  };

  const runXgboost = async () => {
    const cols = orderedEligible.filter((c) => selected.has(c));
    if (cols.length < 2) {
      setError('Select at least two columns to run feature importance.');
      return;
    }
    setXgbLoading(true);
    setError(null);
    try {
      const res = await api.xgboostImportance(cols, source);
      if (!Object.keys(res.results).length) {
        setError(res.message || 'No feature-importance results (need ≥2 non-constant columns).');
        return;
      }
      if (onXgboost) onXgboost(res.results);
      else setLocalXgb(res.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Feature importance failed');
    } finally {
      setXgbLoading(false);
    }
  };

  const loadContingency = async (a: string, b: string) => {
    setPair({ a, b });
    try {
      const res = await api.contingency({ col1: a, col2: b, drop_missing: dropMissing, source });
      setContingency(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Contingency failed');
    }
  };

  // Lower-triangle masked matrix for the heatmap
  const masked = report
    ? report.matrix.map((row, i) => row.map((val, j) => (j > i ? null : val)))
    : [];

  return (
    <div className="space-y-4">
      <Panel
        title="Categorical Association Explorer (Cramér's V)"
        subtitle="Pairwise association across the selected categorical columns"
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={runXgboost}
              disabled={xgbLoading || selected.size < 2}
              title="Train XGBoost on the selected columns and send the result to Feature Importance"
              className="flex items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
            >
              <BarChart3 className="h-3.5 w-3.5" />
              {xgbLoading ? 'Training…' : 'Feature Importance →'}
            </button>
            <button
              onClick={run}
              disabled={loading}
              className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5" />
              {loading ? 'Computing…' : 'Compute'}
            </button>
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={dropMissing}
              onChange={(e) => setDropMissing(e.target.checked)}
              className="accent-accent"
            />
            Drop missing (complete-case pairs)
          </label>
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={excludeTrivial}
              onChange={(e) => setExcludeTrivial(e.target.checked)}
              className="accent-accent"
            />
            Exclude trivial (V≈0 / V≈1)
          </label>
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            Strong ≥ {strong.toFixed(2)}
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={strong}
              onChange={(e) => setStrong(Number(e.target.value))}
              className="w-28 accent-accent"
            />
          </label>
        </div>
      </Panel>

      {/* Column / parent-group selector */}
      <Panel
        title="Columns"
        subtitle="Add whole parent groups (nested dot-separated names) or individual columns"
        actions={
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-text-muted">
              {selected.size}/{orderedEligible.length} selected
            </span>
            <button
              onClick={selectAll}
              className="rounded border border-border px-2 py-0.5 text-text-secondary transition-colors hover:border-accent hover:text-accent"
            >
              All
            </button>
            <button
              onClick={clearAll}
              className="rounded border border-border px-2 py-0.5 text-text-secondary transition-colors hover:border-accent hover:text-accent"
            >
              Clear
            </button>
          </div>
        }
      >
        {groupsError && <p className="text-xs text-danger">{groupsError}</p>}
        {!groupsError && orderedEligible.length === 0 && (
          <p className="text-xs text-text-muted">
            No categorical-eligible columns found for this source (binary/low/medium cardinality).
          </p>
        )}

        <div className="space-y-3">
          {nestedGroups.map((g) => {
            const sel = g.columns.filter((c) => selected.has(c)).length;
            const all = sel === g.columns.length;
            return (
              <div key={g.parent} className="rounded-md border border-border/50 bg-raised/40 p-2">
                <button
                  onClick={() => toggleGroup(g)}
                  title={all ? 'Remove whole group' : 'Add whole group'}
                  className={`mb-1.5 flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs font-semibold transition-colors ${
                    all ? 'text-accent-bright' : sel ? 'text-accent' : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  <Layers className="h-3.5 w-3.5" />
                  {g.parent}
                  <span className="font-normal text-text-muted">
                    ({sel}/{g.columns.length})
                  </span>
                </button>
                <div className="flex flex-wrap gap-1.5 pl-1">
                  {g.columns.map((c, i) => (
                    <button
                      key={c}
                      onClick={() => toggleCol(c)}
                      title={c}
                      className={`rounded-md border px-2 py-1 text-[11px] transition-colors ${
                        selected.has(c)
                          ? 'border-accent bg-accent-dim/30 text-accent-bright'
                          : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      {g.leaves[i]}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}

          {standaloneCols.length > 0 && (
            <div>
              {nestedGroups.length > 0 && (
                <p className="mb-1.5 text-[11px] font-medium text-text-muted">Ungrouped columns</p>
              )}
              <div className="flex flex-wrap gap-1.5">
                {standaloneCols.map((c) => (
                  <button
                    key={c}
                    onClick={() => toggleCol(c)}
                    className={`rounded-md border px-2 py-1 text-[11px] transition-colors ${
                      selected.has(c)
                        ? 'border-accent bg-accent-dim/30 text-accent-bright'
                        : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </Panel>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Computing Cramér's V matrix..." />}
      {xgbLoading && <LoadingSpinner text="Training XGBoost on the selected columns..." />}

      {localXgb && (
        <Panel
          title="Feature Importance (XGBoost)"
          subtitle="Each selected column predicted from the others — gain-based importance"
        >
          <XGBoostResults results={localXgb} />
        </Panel>
      )}

      {report && report.labels.length < 2 && (
        <Panel title="Not enough categorical columns">
          <p className="text-sm text-text-muted">
            Fewer than two suitable categorical columns were selected (binary/low/medium cardinality).
            High-cardinality, free-text and constant columns are excluded automatically.
          </p>
        </Panel>
      )}

      {report && report.labels.length >= 2 && (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
          {/* Heatmap */}
          <div className="xl:col-span-2">
            <Panel title="Association Matrix" subtitle="Click a cell to drill into the contingency table" noPad>
              <div className="p-2">
                <Plot
                  data={[
                    {
                      z: masked,
                      x: report.labels,
                      y: report.labels,
                      type: 'heatmap',
                      colorscale: [
                        [0, '#0d1117'],
                        [0.25, '#1f3a5f'],
                        [0.5, '#3d6098'],
                        [0.75, '#d29922'],
                        [1, '#f85149'],
                      ],
                      zmin: 0,
                      zmax: 1,
                      hoverongaps: false,
                      colorbar: {
                        title: { text: "Cramér's V", font: { color: '#8b949e', size: 10 } },
                        tickfont: { color: '#8b949e', size: 9 },
                      },
                    },
                  ]}
                  layout={{
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#8b949e', size: 9 },
                    margin: { l: 130, r: 30, t: 20, b: 130 },
                    xaxis: { tickangle: -45, automargin: true },
                    yaxis: { automargin: true },
                    height: Math.max(360, report.labels.length * 26),
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                  onClick={(e: Readonly<{ points?: Array<{ x?: unknown; y?: unknown }> }>) => {
                    const pt = e.points?.[0];
                    if (pt && pt.x != null && pt.y != null) {
                      loadContingency(String(pt.y), String(pt.x));
                    }
                  }}
                />
              </div>
            </Panel>

            {/* Contingency drilldown */}
            {pair && contingency && (
              <Panel
                title={`Contingency: ${pair.a} × ${pair.b}`}
                subtitle={`Cramér's V = ${contingency.v} · N = ${contingency.n.toLocaleString()}`}
                className="mt-4"
                noPad
              >
                <div className="overflow-auto p-2" style={{ maxHeight: 360 }}>
                  <table className="border-collapse text-[11px]">
                    <thead>
                      <tr>
                        <th className="sticky left-0 bg-deep px-2 py-1 text-left text-text-muted">
                          {pair.a} \ {pair.b}
                        </th>
                        {contingency.col_labels.map((c) => (
                          <th key={c} className="px-2 py-1 text-text-muted" title={c}>
                            <div className="max-w-24 truncate">{c}</div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {contingency.row_labels.map((r, i) => (
                        <tr key={r} className="border-t border-border/30">
                          <td className="sticky left-0 bg-surface px-2 py-1 font-medium text-text-secondary" title={r}>
                            <div className="max-w-32 truncate">{r}</div>
                          </td>
                          {contingency.matrix[i].map((v, j) => (
                            <td key={j} className="px-2 py-1 text-center text-text-secondary">
                              {v || ''}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            )}
          </div>

          {/* Pairs + high-corr columns */}
          <div className="space-y-4">
            <Panel
              title="Strongest Pairs"
              subtitle={report.n_excluded ? `${report.n_excluded} trivial pairs hidden` : undefined}
            >
              <div className="max-h-96 space-y-1 overflow-y-auto">
                {report.pairs.slice(0, 40).map((p, i) => (
                  <button
                    key={i}
                    onClick={() => loadContingency(p.a, p.b)}
                    className="flex w-full items-center justify-between gap-2 rounded border border-border/40 bg-raised px-2.5 py-1.5 text-left text-[11px] hover:border-accent"
                  >
                    <span className="truncate text-text-secondary">
                      {p.a} <span className="text-text-muted">×</span> {p.b}
                    </span>
                    <span
                      className={`shrink-0 font-mono font-semibold ${
                        p.v >= strong ? 'text-accent-bright' : 'text-text-muted'
                      }`}
                    >
                      {p.v.toFixed(3)}
                    </span>
                  </button>
                ))}
                {report.pairs.length === 0 && (
                  <p className="text-xs text-text-muted">No non-trivial pairs found.</p>
                )}
              </div>
            </Panel>

            {report.high_correlation_columns.length > 0 && (
              <Panel title={`High-Correlation Columns (≥ ${strong.toFixed(2)})`}>
                <div className="flex flex-wrap gap-1.5">
                  {report.high_correlation_columns.map((c) => (
                    <span
                      key={c}
                      className="rounded-md border border-purple/40 bg-purple/10 px-2 py-1 text-[11px] text-text-primary"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}

      {!report && !loading && (
        <Panel title="Cramér's V">
          <p className="flex items-center gap-2 text-sm text-text-muted">
            <Grid3x3 className="h-4 w-4" />
            Pick parent groups / columns above, then click{' '}
            <span className="text-accent">Compute</span> to score categorical associations
            across the {source === 'parsed' ? 'parsed' : 'loaded'} dataset.
          </p>
        </Panel>
      )}
    </div>
  );
}
