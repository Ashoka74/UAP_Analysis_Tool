import { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Play, AlertTriangle, Grid3x3, Layers, BarChart3, Boxes, Wand2 } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { XGBoostResults } from './XGBoostResults';
import { AiInterpret } from './AiInterpret';
import type { ColumnGroup, XGBoostResult, XgboostPcaResponse, XgboostImputeResponse, ConditionalResponse } from '../../types';

interface Props {
  source: 'dataset' | 'parsed';
  // When provided, XGBoost feature importance computed from the selected columns
  // is handed off to the parent (e.g. the Feature Importance tab) instead of
  // rendering inline. `extras` carries the optional 2nd-pass PCA payload and the
  // missing-value imputation payload when those toggles are on.
  onXgboost?: (
    results: Record<string, XGBoostResult>,
    extras?: { pca?: XgboostPcaResponse; impute?: XgboostImputeResponse },
  ) => void;
}

const VERDICT_META: Record<string, { label: string; cls: string }> = {
  persists: { label: 'Persists within strata', cls: 'bg-success/15 text-success border-success/40' },
  attenuated: { label: 'Survives but weaker', cls: 'bg-warning/15 text-warning border-warning/40' },
  explained_by_z: { label: 'Explained by Z (confounded)', cls: 'bg-danger/15 text-danger border-danger/40' },
  weak_or_absent: { label: 'Weak / absent', cls: 'bg-raised text-text-muted border-border' },
  inconclusive: { label: 'Inconclusive', cls: 'bg-raised text-text-muted border-border' },
};
const fmtP = (p: number | null) => (p == null ? '—' : p < 1e-3 ? '<0.001' : p.toFixed(3));

export function CramersVExplorer({ source, onXgboost }: Props) {
  // Persistent explorer state lives in the global store so it survives tab/page
  // switches (this component unmounts whenever another Analysis tab is active).
  const {
    cramersReport: report, setCramersReport,
    cramersParams, setCramersParams,
    cramersContingency, setCramersContingency,
    cramersSelected, setCramersSelected,
    cramersLocalXgb: localXgb, setCramersLocalXgb,
    cramersLocalPca: localPca, setCramersLocalPca,
    cramersLocalImpute: localImpute, setCramersLocalImpute,
    setImputation,
    cramersAutoRun, setCramersAutoRun,
  } = useStore();

  const { dropMissing, excludeTrivial, strong } = cramersParams;
  const { pair, data: contingency } = cramersContingency;

  // Transient UI state — fine to reset on remount.
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [xgbLoading, setXgbLoading] = useState(false);
  const [withCv, setWithCv] = useState(true);   // cross-validate (slower)
  const [withPca, setWithPca] = useState(false); // 2nd pass: collapse redundancy → PCA indices
  const [prunePca, setPrunePca] = useState(false); // drop columns pass 1 never split on
  const [withSig, setWithSig] = useState(false); // permutation null + stability (slower)
  const [withImpute, setWithImpute] = useState(false); // predict each column's missing values
  const [nImput, setNImput] = useState(20);      // multiple-imputation draws (1 = single)
  const [withCi, setWithCi] = useState(false);   // bootstrap CIs on the strongest pairs

  // Eligible categorical columns, grouped by their dotted parent (e.g. craft.*).
  const [groups, setGroups] = useState<ColumnGroup[]>([]);
  const [groupsError, setGroupsError] = useState<string | null>(null);
  // Exact semantic duplicates deselected by default (unit twins, anomaly.*
  // re-encodings of engagement types) — shown as a note for transparency.
  const [dedup, setDedup] = useState<{ kept: string; dropped: string[]; reason: string }[]>([]);

  // Selection is a Set in the UI but persisted as an array in the store.
  const selected = useMemo(() => new Set(cramersSelected ?? []), [cramersSelected]);

  // Load the parent groups up front — cheap (cardinality only, no matrix), so the
  // selector is usable before the first Compute. Seeds the selection to all
  // eligible columns only on the FIRST load; a cached or handed-in selection is
  // preserved across remounts.
  useEffect(() => {
    let cancelled = false;
    setGroupsError(null);
    api
      .columnGroups({ source })
      .then((res) => {
        if (cancelled) return;
        setGroups(res.groups);
        setDedup(res.semantic_duplicates_removed ?? []);
        if (useStore.getState().cramersSelected === null) {
          setCramersSelected(res.eligible);
        }
      })
      .catch((e) => {
        if (!cancelled) setGroupsError(e instanceof Error ? e.message : 'Could not load columns');
      });
    return () => {
      cancelled = true;
    };
  }, [source, setCramersSelected]);

  // Flattened in grouped order so the matrix keeps related columns adjacent.
  const orderedEligible = useMemo(() => groups.flatMap((g) => g.columns), [groups]);
  const nestedGroups = groups.filter((g) => g.nested);
  const standaloneCols = groups.filter((g) => !g.nested).flatMap((g) => g.columns);

  const toggleCol = (c: string) => {
    const next = new Set(selected);
    if (next.has(c)) next.delete(c);
    else next.add(c);
    setCramersSelected([...next]);
  };

  const toggleGroup = (g: ColumnGroup) => {
    const next = new Set(selected);
    const allOn = g.columns.every((c) => next.has(c));
    for (const c of g.columns) {
      if (allOn) next.delete(c);
      else next.add(c);
    }
    setCramersSelected([...next]);
  };

  const selectAll = () => setCramersSelected([...orderedEligible]);
  const clearAll = () => setCramersSelected([]);

  const run = async () => {
    if (orderedEligible.length && selected.size < 2) {
      setError('Select at least two columns (or whole parent groups) to compute associations.');
      return;
    }
    setLoading(true);
    setError(null);
    setCramersContingency(null, null);
    try {
      const cols = orderedEligible.filter((c) => selected.has(c));
      const res = await api.cramersV({
        source,
        columns: cols.length ? cols : undefined,
        drop_missing: dropMissing,
        exclude_trivial: excludeTrivial,
        strong_threshold: strong,
        ci_top_n: withCi ? 40 : 0,
      });
      setCramersReport(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Cramér’s V failed');
    } finally {
      setLoading(false);
    }
  };

  // Compute once automatically when a selection was handed in from another tab
  // (e.g. the Feature Importance tab's "Cramér's V →" action).
  useEffect(() => {
    if (cramersAutoRun && groups.length > 0) {
      setCramersAutoRun(false);
      run();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cramersAutoRun, groups]);

  const runXgboost = async () => {
    const cols = orderedEligible.filter((c) => selected.has(c));
    if (cols.length < 2) {
      setError('Select at least two columns to run feature importance.');
      return;
    }
    setXgbLoading(true);
    setError(null);
    try {
      // Headline feature importance — the PCA path returns the 1st pass as the
      // headline plus the 2nd-pass payload; otherwise the plain importance.
      let headline: Record<string, XGBoostResult> | null = null;
      let pca: XgboostPcaResponse | undefined;
      if (withPca) {
        const res = await api.xgboostPca(cols, source, withCv, strong, prunePca, withSig);
        if (Object.keys(res.first_pass).length) { headline = res.first_pass; pca = res; }
      } else {
        const res = await api.xgboostImportance(cols, source, withCv, withSig);
        if (Object.keys(res.results).length) headline = res.results;
      }
      if (!headline) {
        setError('No feature-importance results (need ≥2 non-constant columns).');
        return;
      }

      // Optional: predict each column's missing values from the others.
      let impute: XgboostImputeResponse | undefined;
      if (withImpute) {
        impute = await api.xgboostImpute(cols, source, withCv, withPca, strong, nImput);
        setImputation(impute);   // share with the Data Explorer grid overlay
      }

      if (onXgboost) onXgboost(headline, { pca, impute });
      else {
        setCramersLocalXgb(headline);
        setCramersLocalPca(pca ?? null);
        setCramersLocalImpute(impute ?? null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Feature importance failed');
    } finally {
      setXgbLoading(false);
    }
  };

  const loadContingency = async (a: string, b: string) => {
    setCramersContingency({ a, b }, null);
    setCond(null);          // reset conditional drill-down for the new pair
    setCondZ('');
    try {
      const res = await api.contingency({ col1: a, col2: b, drop_missing: dropMissing, source });
      setCramersContingency({ a, b }, res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Contingency failed');
    }
  };

  // Conditional-association drill-down: does the current pair survive conditioning
  // on a third field Z? (the formal confounding / redundancy check)
  const [condZ, setCondZ] = useState('');
  const [cond, setCond] = useState<ConditionalResponse | null>(null);
  const [condLoading, setCondLoading] = useState(false);
  const runConditional = async () => {
    if (!pair || !condZ) return;
    setCondLoading(true);
    setError(null);
    try {
      const res = await api.conditional({
        col1: pair.a, col2: pair.b, condition_on: condZ, drop_missing: dropMissing, source,
      });
      setCond(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Conditional test failed');
    } finally {
      setCondLoading(false);
    }
  };

  // Lower-triangle masked matrix for the heatmap
  const masked = report
    ? report.matrix.map((row, i) => row.map((val, j) => (j > i ? null : val)))
    : [];

  // Compact association summary handed to the AI interpreter.
  const buildContext = () => {
    if (!report) return '';
    const lines: string[] = [
      `Cramér's V categorical-association analysis over ${report.labels.length} columns (source: ${source}).`,
      `Columns: ${report.labels.join(', ')}.`,
      `Strong-association threshold: V ≥ ${strong.toFixed(2)}.`,
    ];
    if (report.high_correlation_columns.length) {
      lines.push(`High-correlation columns (≥ ${strong.toFixed(2)}): ${report.high_correlation_columns.join(', ')}.`);
    }
    lines.push('Strongest pairs (a × b: V):');
    report.pairs.slice(0, 25).forEach((p) => lines.push(`- ${p.a} × ${p.b}: ${p.v.toFixed(3)}`));
    return lines.join('\n');
  };

  return (
    <div className="space-y-4">
      <Panel
        title="Categorical Association Explorer (Cramér's V)"
        subtitle="Pairwise association (bias-corrected Cramér's V, Bergsma 2013) across the selected categorical columns"
        actions={
          <div className="flex items-center gap-2">
            <label
              className="flex items-center gap-1 text-[11px] text-text-muted"
              title="Cross-validate each target (slower) — adds CV accuracy ± std and the overfit check. Uncheck for a fast holdout-only run."
            >
              <input
                type="checkbox"
                checked={withCv}
                onChange={(e) => setWithCv(e.target.checked)}
                className="accent-accent"
              />
              CV
            </label>
            <label
              className={`flex items-center gap-1 text-[11px] ${withPca ? 'text-purple' : 'text-text-muted'}`}
              title="2nd pass: collapse each Cramér's V redundancy cluster (V ≥ strong threshold) into one PCA latent index, then re-fit XGBoost. Surfaces the de-diluted combined signal of correlated columns."
            >
              <input
                type="checkbox"
                checked={withPca}
                onChange={(e) => setWithPca(e.target.checked)}
                className="accent-purple"
              />
              <Boxes className="h-3 w-3" /> PCA 2nd pass
            </label>
            <label
              className={`flex items-center gap-1 text-[11px] ${withImpute ? 'text-accent' : 'text-text-muted'}`}
              title="Predict each selected column's MISSING values from the others using the trained models. Each predicted fill is coloured by its model's accuracy (confidence)."
            >
              <input
                type="checkbox"
                checked={withImpute}
                onChange={(e) => setWithImpute(e.target.checked)}
                className="accent-accent"
              />
              <Wand2 className="h-3 w-3" /> Predict missing
            </label>
            <label
              className={`flex items-center gap-1 text-[11px] ${withSig ? 'text-warning' : 'text-text-muted'}`}
              title="Significance for feature importance (SLOWER): a permutation null (shuffle the target, refit) gives an empirical p-value per feature, and bootstrap resampling gives a selection frequency. Reports how often a feature survives, not a single run."
            >
              <input
                type="checkbox"
                checked={withSig}
                onChange={(e) => setWithSig(e.target.checked)}
                className="accent-warning"
              />
              Significance
            </label>
            <button
              onClick={runXgboost}
              disabled={xgbLoading || selected.size < 2}
              title={
                withPca
                  ? "Train XGBoost twice — raw columns, then with redundancy clusters collapsed into PCA latent indices — and send both to Feature Importance"
                  : 'Train XGBoost on the selected columns and send the result to Feature Importance'
              }
              className="flex items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
            >
              <BarChart3 className="h-3.5 w-3.5" />
              {xgbLoading ? 'Training…' : withPca ? 'Feature Importance + PCA →' : 'Feature Importance →'}
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
              onChange={(e) => setCramersParams({ dropMissing: e.target.checked })}
              className="accent-accent"
            />
            Drop missing (complete-case pairs)
          </label>
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={excludeTrivial}
              onChange={(e) => setCramersParams({ excludeTrivial: e.target.checked })}
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
              onChange={(e) => setCramersParams({ strong: Number(e.target.value) })}
              className="w-28 accent-accent"
            />
          </label>
          <label
            className="flex items-center gap-2 text-xs text-text-secondary"
            title="Add a 95% bootstrap confidence interval to the strongest pairs and to the contingency drill-down. A wide interval means the point V is unstable (e.g. sparse cells) — don't over-read it."
          >
            <input
              type="checkbox"
              checked={withCi}
              onChange={(e) => setWithCi(e.target.checked)}
              className="accent-accent"
            />
            Bootstrap 95% CI (top pairs)
          </label>
          {withImpute && (
            <label
              className="flex items-center gap-2 text-xs text-accent"
              title="Multiple imputation: draw this many values per missing cell from the model's predicted class probabilities, then report the modal value + a per-cell agreement. 1 = single (best-guess) imputation. >1 exposes per-cell uncertainty."
            >
              Imputations (M)
              <input
                type="number"
                min={1}
                max={200}
                value={nImput}
                onChange={(e) => setNImput(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
                className="w-16 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary focus:border-accent focus:outline-none"
              />
            </label>
          )}
          {withPca && (
            <label
              className="flex items-center gap-2 text-xs text-purple"
              title="Before the 2nd pass, drop non-redundant columns that the 1st pass never split on (zero gain everywhere). Redundancy-cluster members are always kept — they live on inside their PCA index."
            >
              <input
                type="checkbox"
                checked={prunePca}
                onChange={(e) => setPrunePca(e.target.checked)}
                className="accent-purple"
              />
              Drop 1st-pass-unused columns
            </label>
          )}
        </div>
        {withPca && (
          <p className="mt-2 flex items-center gap-1.5 text-[11px] text-text-muted">
            <Boxes className="h-3 w-3 text-purple" />
            2nd pass collapses redundancy clusters at the <span className="text-purple">strong ≥ {strong.toFixed(2)}</span>{' '}
            threshold into PCA latent indices — tune the slider to control which columns are treated as redundant.
          </p>
        )}
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

        {dedup.length > 0 && (
          <p
            className="mb-2 text-[11px] text-text-muted"
            title={dedup
              .map((d) => `${d.dropped.join(', ')} — ${d.reason} (kept ${d.kept})`)
              .join('\n')}
          >
            ⚖️ {dedup.reduce((n, d) => n + d.dropped.length, 0)} exact semantic duplicate(s)
            deselected by default ({dedup.map((d) => d.dropped.map((c) => c.split('.').pop()).join(', ')).join(', ')})
            — twins of kept fields; re-select manually if needed.
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
          <XGBoostResults results={localXgb} pca={localPca ?? undefined} impute={localImpute ?? undefined} />
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
        <AiInterpret kind="cramers" buildContext={buildContext} label="Interpret associations with AI" />
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
                subtitle={
                  `Cramér's V = ${contingency.v}` +
                  (contingency.ci ? ` (95% CI ${contingency.ci[0]}–${contingency.ci[1]})` : '') +
                  ` · N = ${contingency.n.toLocaleString()}`
                }
                className="mt-4"
                noPad
              >
                <div className="overflow-auto p-2" style={{ maxHeight: 320 }}>
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

                {/* Conditional-association drill-down (CMH / stratified χ²) */}
                <div className="border-t border-border/60 p-3">
                  <div className="flex flex-wrap items-center gap-2 text-[11px]">
                    <span className="font-semibold text-text-primary">Condition on a third field:</span>
                    <select
                      value={condZ}
                      onChange={(e) => setCondZ(e.target.value)}
                      className="rounded border border-border bg-deep px-2 py-1 text-[11px] text-text-primary focus:border-accent focus:outline-none"
                    >
                      <option value="">— pick Z —</option>
                      {orderedEligible
                        .filter((c) => c !== pair.a && c !== pair.b)
                        .map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                    </select>
                    <button
                      onClick={runConditional}
                      disabled={!condZ || condLoading}
                      title="Is the association confounded by / explained by Z? Per-stratum Cramér's V + a pooled CMH / stratified-χ² test."
                      className="rounded-md border border-border bg-raised px-2.5 py-1 text-[11px] font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                    >
                      {condLoading ? 'Testing…' : 'Test'}
                    </button>
                  </div>
                  {cond && (
                    <div className="mt-2 space-y-2">
                      <div className="flex flex-wrap items-center gap-2 text-[11px]">
                        <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${(VERDICT_META[cond.verdict] ?? VERDICT_META.inconclusive).cls}`}>
                          {(VERDICT_META[cond.verdict] ?? VERDICT_META.inconclusive).label}
                        </span>
                        <span className="text-text-secondary">
                          Marginal V <b className="text-text-primary">{cond.marginal_v.toFixed(3)}</b>
                          {' → '}mean within-stratum V{' '}
                          <b className="text-text-primary">{cond.mean_conditional_v != null ? cond.mean_conditional_v.toFixed(3) : '—'}</b>
                        </span>
                        {cond.test.method && (
                          <span className="text-text-muted">
                            {cond.test.method}: p = {fmtP(cond.test.p_value)}
                            {cond.test.pooled_odds_ratio != null ? `, OR ${cond.test.pooled_odds_ratio}` : ''}
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-text-muted">
                        {cond.n_strata_used} strata used
                        {cond.n_strata_dropped ? `, ${cond.n_strata_dropped} dropped (too small)` : ''} · conditioning on{' '}
                        <span className="text-text-secondary">{cond.condition_on}</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {cond.strata.map((s) => (
                          <span
                            key={s.level}
                            className="rounded border border-border/50 bg-raised px-1.5 py-0.5 text-[10px] text-text-secondary"
                            title={`${s.n.toLocaleString()} rows`}
                          >
                            <span className="max-w-24 truncate">{s.level}</span>: V={s.v.toFixed(2)}{' '}
                            <span className="text-text-muted">(n={s.n})</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
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
                    <span className="flex shrink-0 flex-col items-end leading-tight">
                      <span
                        className={`font-mono font-semibold ${
                          p.v >= strong ? 'text-accent-bright' : 'text-text-muted'
                        }`}
                      >
                        {p.v.toFixed(3)}
                      </span>
                      {p.ci && (
                        <span className="font-mono text-[9px] text-text-muted" title="95% bootstrap CI">
                          [{p.ci[0].toFixed(2)}, {p.ci[1].toFixed(2)}]
                        </span>
                      )}
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
