import { useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import Plot from 'react-plotly.js';
import { Download, BarChart3, Table2, Trophy, AlertTriangle, Star, Activity, Layers, Boxes, Wand2 } from 'lucide-react';
import { AiInterpret } from './AiInterpret';
import type { XGBoostResult, XgboostPcaResponse, XgboostImputeResponse } from '../../types';

interface Props {
  results: Record<string, XGBoostResult>;
  // When present, enables the optional 2nd pass: each Cramér's V redundancy
  // cluster is collapsed into one PCA latent index and XGBoost is re-fit. The
  // `results` prop carries the 1st pass (== pca.first_pass) and this carries the
  // 2nd pass + latent-index definitions.
  pca?: XgboostPcaResponse;
  // When present, the trained models predicted each column's missing values;
  // the predicted fills are shown coloured by each model's accuracy.
  impute?: XgboostImputeResponse;
}

// Accuracy → colour ramp (red → amber at 0.6 → green at ~0.85). Used to tint the
// imputed-value chips so the colour itself encodes the model's reliability.
function accuracyRgb(a: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, a));
  const lerp = (x: number, y: number, p: number) => Math.round(x + (y - x) * p);
  if (t < 0.6) {
    const p = t / 0.6;
    return [lerp(248, 210, p), lerp(81, 153, p), lerp(73, 34, p)];
  }
  const p = Math.min(1, (t - 0.6) / 0.25);
  return [lerp(210, 63, p), lerp(153, 185, p), lerp(34, 80, p)];
}

function accuracyChipStyle(a: number | null): CSSProperties {
  const [r, g, b] = a == null ? [139, 148, 158] : accuracyRgb(a);
  return {
    backgroundColor: `rgba(${r}, ${g}, ${b}, 0.16)`,
    color: `rgb(${r}, ${g}, ${b})`,
    borderColor: `rgba(${r}, ${g}, ${b}, 0.4)`,
  };
}

function toCsv(rows: (string | number)[][]): string {
  return rows
    .map((r) =>
      r
        .map((c) => {
          const s = String(c ?? '');
          return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
        })
        .join(','),
    )
    .join('\n');
}

function downloadCsv(filename: string, csv: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const signedPct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
// Prefer the CV mean as the headline score when present, else the holdout.
const scoreOf = (r: XGBoostResult) => r.cv_mean ?? r.accuracy;

function accuracyClass(a: number) {
  return a >= 0.8 ? 'bg-success/20 text-success' : a >= 0.6 ? 'bg-warning/20 text-warning' : 'bg-danger/20 text-danger';
}

export function XGBoostResults({ results, pca, impute }: Props) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const hasPca = !!pca && Array.isArray(pca.clusters);
  // Default to showing the 2nd pass when it exists — that's the point of running it.
  const [pass, setPass] = useState<'first' | 'second'>(hasPca ? 'second' : 'first');

  const showSecond = hasPca && pass === 'second';
  const displayResults = showSecond ? pca!.second_pass : results;
  const otherResults = hasPca ? (showSecond ? pca!.first_pass : pca!.second_pass) : null;

  const columns = Object.keys(displayResults);
  const hasCv = columns.some((c) => displayResults[c].cv_mean != null);
  // Permutation null p-value + bootstrap selection frequency (opt-in significance).
  const hasSig = columns.some((c) => displayResults[c].null_p || displayResults[c].selection_freq);

  // Tidy (target, feature, importance) rows — the saveable table / CSV.
  const tableRows = useMemo(() => {
    const rows: {
      target: string; accuracy: number; cvMean: number | null; cvStd: number | null;
      cvFolds: number | null; feature: string; importance: number;
      nullP: number | null; selFreq: number | null;
    }[] = [];
    for (const col of Object.keys(displayResults)) {
      const r = displayResults[col];
      const base = { target: col, accuracy: r.accuracy, cvMean: r.cv_mean ?? null, cvStd: r.cv_std ?? null, cvFolds: r.cv_folds ?? null };
      const feats = Object.entries(r.feature_importance);
      const lookup = (f: string) => ({
        nullP: r.null_p?.[f] ?? null,
        selFreq: r.selection_freq?.[f] ?? null,
      });
      if (feats.length === 0) rows.push({ ...base, feature: '(no splits)', importance: 0, nullP: null, selFreq: null });
      for (const [feature, importance] of feats) rows.push({ ...base, feature, importance, ...lookup(feature) });
    }
    return rows;
  }, [displayResults]);

  const exportCsv = () => {
    const header = ['target', 'accuracy', 'cv_mean', 'cv_std', 'cv_folds', 'feature', 'importance',
      ...(hasSig ? ['perm_p', 'selection_freq'] : [])];
    const body = tableRows.map((r) => [
      r.target, r.accuracy, r.cvMean ?? '', r.cvStd ?? '', r.cvFolds ?? '', r.feature, r.importance.toFixed(6),
      ...(hasSig ? [r.nullP ?? '', r.selFreq ?? ''] : []),
    ]);
    downloadCsv(`xgboost_feature_importance_${showSecond ? 'pca' : 'raw'}.csv`, toCsv([header, ...body]));
  };

  // Compact results summary handed to the AI interpreter. On the 2nd pass it also
  // explains each latent index so the model reads "PCA[...]" features correctly.
  const buildContext = () => {
    const lines: string[] = [];
    if (showSecond && pca) {
      lines.push(
        'XGBoost 2nd pass: each Cramér’s V redundancy cluster (V ≥ ' +
          `${pca.strong_threshold.toFixed(2)}) was collapsed into ONE PCA latent index (one-hot → PCA(1), ` +
          '≈ Multiple Correspondence Analysis), then XGBoost was re-fit on [latent indices + non-redundant ' +
          'columns]. A "PCA·root" feature is that decorrelated latent index (named after its most central ' +
          'member); its importance is the cluster’s combined, de-diluted signal.',
      );
      if (pca.clusters.length) {
        lines.push('Latent indices:');
        for (const c of pca.clusters) {
          const top = c.loadings.slice(0, 6).map((l) => `${l.feature} ${l.weight.toFixed(2)}`).join(', ');
          lines.push(`- ${c.index_name} (var ${pct(c.explained_variance)}): ${top}`);
        }
      }
      lines.push('');
    }
    lines.push(`Feature importance (${showSecond ? '2nd pass / PCA latent indices' : '1st pass / raw columns'}):`);
    for (const col of Object.keys(displayResults)) {
      const r = displayResults[col];
      const cv = r.cv_mean != null ? `, ${r.cv_folds}-fold CV ${pct(r.cv_mean)} ± ${pct(r.cv_std ?? 0)}` : '';
      const delta = otherResults?.[col]
        ? ` (Δ vs 1st ${signedPct(scoreOf(r) - scoreOf(otherResults[col]))})`
        : '';
      lines.push(`Target "${col}": holdout accuracy ${pct(r.accuracy)}${cv}${showSecond ? delta : ''}.`);
      const tops = Object.entries(r.feature_importance).slice(0, 8);
      lines.push(
        tops.length
          ? `  Top features by gain: ${tops.map(([f, v]) => `${f}=${v.toFixed(3)}`).join(', ')}`
          : '  No features were used in any split.',
      );
    }
    return lines.join('\n');
  };

  // Side highlights — same spirit as the Cramér's V explorer's side column.
  const highlights = useMemo(() => {
    const cols = Object.keys(displayResults);
    const perTarget = cols.map((col) => {
      const r = displayResults[col];
      const score = scoreOf(r); // prefer CV when present
      const gap = r.cv_mean != null ? r.accuracy - r.cv_mean : null;
      const top = Object.entries(r.feature_importance)[0];
      return { col, accuracy: r.accuracy, cvMean: r.cv_mean ?? null, score, gap, topFeature: top?.[0] ?? null };
    });
    const bestPredicted = [...perTarget].sort((a, b) => b.score - a.score).slice(0, 5);
    const hardest = [...perTarget].sort((a, b) => a.score - b.score).slice(0, 5);
    // Overall feature influence: normalise each target's gains to sum 1, then
    // accumulate across targets so different-scale gains are comparable.
    const influence: Record<string, number> = {};
    for (const col of cols) {
      const imps = Object.entries(displayResults[col].feature_importance);
      const total = imps.reduce((s, [, v]) => s + v, 0) || 1;
      for (const [f, v] of imps) influence[f] = (influence[f] ?? 0) + v / total;
    }
    const topFeatures = Object.entries(influence).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const overfit = perTarget
      .filter((p) => p.gap != null && p.gap > 0.1)
      .sort((a, b) => (b.gap ?? 0) - (a.gap ?? 0))
      .slice(0, 5);
    return { bestPredicted, hardest, topFeatures, overfit };
  }, [displayResults]);

  // Aggregate first-vs-second comparison for the headline strip.
  const comparison = useMemo(() => {
    if (!hasPca || !pca) return null;
    const cols = Object.keys(pca.second_pass).filter((c) => pca.first_pass[c]);
    if (!cols.length) return null;
    const mean = (rs: Record<string, XGBoostResult>) =>
      cols.reduce((s, c) => s + scoreOf(rs[c]), 0) / cols.length;
    const firstMean = mean(pca.first_pass);
    const secondMean = mean(pca.second_pass);
    const dims1 = cols.length; // 1st pass uses (n-1) raw columns per target — report feature counts instead
    const avgDims2 =
      cols.reduce((s, c) => s + (pca.second_pass[c].n_features ?? 0), 0) / (cols.length || 1);
    return { firstMean, secondMean, delta: secondMean - firstMean, nClusters: pca.n_clusters, avgDims2, dims1 };
  }, [hasPca, pca]);

  // Map a latent-index feature name back to its cluster for tooltips / labels.
  const clusterByIndex = useMemo(() => {
    const m: Record<string, XgboostPcaResponse['clusters'][number]> = {};
    if (pca) for (const c of pca.clusters) m[c.index_name] = c;
    return m;
  }, [pca]);

  // Imputation rows — one per column with predicted missing values, best model
  // (by score) first so the most reliable fills sit at the top.
  const imputeRows = useMemo(() => {
    if (!impute) return [];
    return Object.entries(impute.results)
      .map(([col, r]) => ({ col, ...r, score: r.cv_mean ?? r.accuracy ?? -1 }))
      .sort((a, b) => b.score - a.score);
  }, [impute]);

  const exportImputations = () => {
    if (!impute) return;
    const header = ['target', 'model_accuracy', 'cv_mean', 'row', 'predicted_value'];
    const body: (string | number)[][] = [];
    for (const [col, r] of Object.entries(impute.results)) {
      for (const s of r.sample) body.push([col, r.accuracy ?? '', r.cv_mean ?? '', s.row, s.value]);
    }
    downloadCsv('imputed_missing_values.csv', toCsv([header, ...body]));
  };

  return (
    <div className="space-y-3">
      {/* Action bar: pass toggle (when PCA present) + chart/table toggle + CSV */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          {hasPca && (
            <div className="inline-flex rounded-md border border-border bg-raised p-0.5 text-xs">
              <button
                onClick={() => setPass('first')}
                title="Raw columns — the standard single-pass importance"
                className={`flex items-center gap-1.5 rounded px-2.5 py-1 transition-colors ${
                  pass === 'first' ? 'bg-accent-dim text-white' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                1st pass
              </button>
              <button
                onClick={() => setPass('second')}
                title="PCA latent indices — redundancy clusters collapsed, then XGBoost re-fit"
                className={`flex items-center gap-1.5 rounded px-2.5 py-1 transition-colors ${
                  pass === 'second' ? 'bg-purple/80 text-white' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                <Boxes className="h-3.5 w-3.5" /> 2nd pass · PCA
              </button>
            </div>
          )}
          <div className="inline-flex rounded-md border border-border bg-raised p-0.5 text-xs">
            <button
              onClick={() => setView('chart')}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 transition-colors ${
                view === 'chart' ? 'bg-accent-dim text-white' : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" /> Charts
            </button>
            <button
              onClick={() => setView('table')}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 transition-colors ${
                view === 'table' ? 'bg-accent-dim text-white' : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              <Table2 className="h-3.5 w-3.5" /> Table
            </button>
          </div>
        </div>
        <button
          onClick={exportCsv}
          title="Save accuracy, cross-validation and per-feature importance as a CSV table"
          className="flex items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
        >
          <Download className="h-3.5 w-3.5" /> Download CSV
        </button>
      </div>

      {/* PCA: comparison strip + "no clusters" message */}
      {hasPca && pca!.message && (
        <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/5 px-4 py-2.5 text-xs text-warning">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {pca!.message}
        </div>
      )}
      {comparison && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 rounded-md border border-purple/30 bg-purple/5 px-4 py-2.5 text-[11px] text-text-secondary">
          <span className="flex items-center gap-1.5 font-semibold text-text-primary">
            <Boxes className="h-3.5 w-3.5 text-purple" /> 2nd pass · PCA
          </span>
          <span>
            {comparison.nClusters} redundancy cluster{comparison.nClusters === 1 ? '' : 's'} collapsed
            {' '}(V ≥ {pca!.strong_threshold.toFixed(2)})
          </span>
          <span>
            Mean {hasCv ? 'CV' : 'holdout'}: <span className="font-mono text-text-primary">{pct(comparison.firstMean)}</span>
            {' → '}
            <span className="font-mono text-text-primary">{pct(comparison.secondMean)}</span>
            {' '}
            <span className={`font-mono font-semibold ${comparison.delta >= 0 ? 'text-success' : 'text-warning'}`}>
              ({signedPct(comparison.delta)})
            </span>
          </span>
          <span className="text-text-muted">avg ~{Math.round(comparison.avgDims2)} features / target</span>
          {pca!.pruned.length > 0 && (
            <span className="text-text-muted">{pca!.pruned.length} uninformative column(s) dropped</span>
          )}
        </div>
      )}

      <AiInterpret kind="xgboost" buildContext={buildContext} label="Interpret results with AI" />

      {/* Model-based imputation — predicted missing values, coloured by accuracy */}
      {impute && (
        <div className="rounded-lg border border-accent/30 bg-surface">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
            <div>
              <h4 className="flex items-center gap-1.5 text-sm font-semibold text-text-primary">
                <Wand2 className="h-3.5 w-3.5 text-accent" /> Predicted missing values
              </h4>
              <p className="text-xs text-text-muted">
                Each column’s missing cells predicted from the others{impute.used_pca ? ' (via PCA latent indices)' : ''} ·{' '}
                fill colour encodes the model’s {hasCv ? 'CV ' : ''}accuracy
                {impute.n_imputations > 1 ? ` · multiple imputation, M=${impute.n_imputations}` : ''}
              </p>
            </div>
            {imputeRows.length > 0 && (
              <button
                onClick={exportImputations}
                title="Download every predicted fill (target, row, value) as a CSV"
                className="flex items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
              >
                <Download className="h-3.5 w-3.5" /> Imputations CSV
              </button>
            )}
          </div>

          {impute.message ? (
            <p className="px-4 py-3 text-sm text-text-muted">{impute.message}</p>
          ) : (
            <>
              <div className="overflow-auto" style={{ maxHeight: 420 }}>
                <table className="w-full border-collapse text-[11px]">
                  <thead className="sticky top-0 bg-deep">
                    <tr className="text-left text-text-muted">
                      <th className="px-3 py-2">Target</th>
                      <th className="px-3 py-2">Model accuracy</th>
                      <th className="px-3 py-2 text-right">Missing</th>
                      {impute.n_imputations > 1 && (
                        <th className="px-3 py-2 text-right" title="Mean per-cell agreement across the M draws. Low agreement = the cell is genuinely uncertain even when the column model is accurate.">Agreement</th>
                      )}
                      <th className="px-3 py-2">Predicted fills (value × count)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {imputeRows.map((r) => (
                      <tr key={r.col} className="border-t border-border/60 align-top">
                        <td className="px-3 py-2 font-medium text-text-secondary">{r.col}</td>
                        <td className="px-3 py-2">
                          <span
                            className="inline-block rounded-full border px-2 py-0.5 font-mono text-[10px] font-semibold"
                            style={accuracyChipStyle(r.accuracy)}
                            title={r.cv_mean != null ? `${r.cv_folds}-fold CV ${pct(r.cv_mean)} ± ${pct(r.cv_std ?? 0)}` : 'holdout accuracy'}
                          >
                            {r.accuracy == null ? 'n/a' : pct(r.accuracy)}
                          </span>
                          {r.cv_mean != null && (
                            <span className="ml-1.5 text-[10px] text-text-muted">CV {pct(r.cv_mean)}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-text-secondary">
                          {r.n_missing.toLocaleString()}
                        </td>
                        {impute.n_imputations > 1 && (
                          <td className="px-3 py-2 text-right font-mono text-text-muted">
                            {r.mean_conf != null ? pct(r.mean_conf) : '—'}
                          </td>
                        )}
                        <td className="px-3 py-2">
                          <div className="flex flex-wrap gap-1">
                            {r.predictions.slice(0, 12).map((p) => (
                              <span
                                key={p.value}
                                className="inline-flex items-center gap-1 rounded border px-1.5 py-0.5"
                                style={accuracyChipStyle(r.accuracy)}
                                title={`${p.count} cell(s) predicted "${p.value}" — model accuracy ${r.accuracy == null ? 'n/a' : pct(r.accuracy)}`}
                              >
                                <span className="max-w-32 truncate">{p.value || '∅'}</span>
                                <span className="font-mono opacity-70">×{p.count}</span>
                              </span>
                            ))}
                            {r.predictions.length > 12 && (
                              <span className="text-text-muted">+{r.predictions.length - 12} more</span>
                            )}
                          </div>
                          {r.sample_truncated && (
                            <p className="mt-1 text-[10px] text-text-muted">
                              Row-level CSV capped at {r.sample.length.toLocaleString()} of {r.n_missing.toLocaleString()} cells; the distribution above is complete.
                            </p>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap items-center gap-3 border-t border-border px-4 py-2 text-[10px] text-text-muted">
                <span>Colour = model accuracy:</span>
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(248,81,73)' }} /> low
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(210,153,34)' }} /> medium
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: 'rgb(63,185,80)' }} /> high
                </span>
                <span className="text-text-muted">— low-accuracy fills are unreliable; treat them as guesses, not facts.</span>
                {impute.n_imputations > 1 && (
                  <span className="w-full text-text-muted">
                    Multiple imputation (M={impute.n_imputations}): chips show the modal draw; <b>Agreement</b> is per-cell certainty.
                    For statistical inference, pool an estimand across the M completed datasets with Rubin’s rules.
                  </span>
                )}
                <span className="w-full text-text-muted">
                  Assumes values are missing-at-random given the observed fields (MAR). If a field is blank <i>because of</i> its
                  own value (MNAR — common in curated UAP reports), these fills are biased.
                </span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Latent-index definitions (2nd pass only) */}
      {showSecond && pca!.clusters.length > 0 && (
        <div className="rounded-lg border border-purple/30 bg-surface p-3">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-text-primary">
            <Layers className="h-3.5 w-3.5 text-purple" /> Latent indices
            <span
              className="font-normal text-text-muted"
              title="One-hot indicators → standardize → PCA(1) per cluster ≈ Multiple Correspondence Analysis (MCA), the canonical dimensionality reduction for categorical data."
            >
              redundant clusters collapsed via PCA on one-hot indicators (≈ MCA) — weights are each column’s share of the component
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {pca!.clusters.map((c) => (
              <div key={c.index_name} className="rounded-md border border-border/60 bg-raised/40 p-2.5">
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="font-mono text-[11px] font-semibold text-purple" title={`Root predictor: ${c.root}`}>
                    {c.index_name}
                  </span>
                  <span className="shrink-0 rounded-full bg-raised px-2 py-0.5 text-[10px] text-text-muted" title="Variance of the cluster captured by this single component">
                    {pct(c.explained_variance)} var
                  </span>
                </div>
                <div className="space-y-1">
                  {c.loadings.map((l) => (
                    <div key={l.feature} className="flex items-center gap-2 text-[10px]">
                      <span className="w-36 shrink-0 truncate text-text-secondary" title={l.feature}>{l.feature}</span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-deep">
                        <div
                          className="h-full rounded-full bg-purple/70"
                          style={{ width: `${Math.min(100, l.weight * 100)}%` }}
                        />
                      </div>
                      <span className="w-9 shrink-0 text-right font-mono text-text-muted">{l.weight.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {/* Main column */}
        <div className="space-y-4 xl:col-span-2">
          {view === 'chart'
            ? columns.map((col) => {
                const r = displayResults[col];
                const features = Object.keys(r.feature_importance);
                const importances = Object.values(r.feature_importance);
                const overfit = r.cv_mean != null && r.accuracy - r.cv_mean > 0.1;
                const other = otherResults?.[col];
                const delta = other ? scoreOf(r) - scoreOf(other) : null;
                return (
                  <div key={col} className="rounded-lg border border-border bg-surface">
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
                      <div>
                        <h4 className="text-sm font-semibold text-text-primary">{col}</h4>
                        <p className="text-xs text-text-muted">
                          Predicted from the {showSecond ? 'latent indices + other columns' : 'other columns'} · gain importance
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        {delta != null && (
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                              delta >= 0 ? 'bg-success/15 text-success' : 'bg-warning/15 text-warning'
                            }`}
                            title={`${hasCv ? 'CV' : 'holdout'} change vs 1st pass`}
                          >
                            {signedPct(delta)} vs 1st
                          </span>
                        )}
                        {overfit && (
                          <AlertTriangle
                            className="h-3.5 w-3.5 text-warning"
                            aria-label="Holdout accuracy is well above the CV mean — likely optimistic / overfit"
                          />
                        )}
                        {r.cv_mean != null && (
                          <span
                            className="rounded-full bg-raised px-2.5 py-0.5 text-xs text-text-secondary"
                            title={`${r.cv_folds}-fold stratified cross-validation`}
                          >
                            CV {pct(r.cv_mean)} ± {pct(r.cv_std ?? 0)}
                          </span>
                        )}
                        <span className="text-xs text-text-muted">Holdout:</span>
                        <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${accuracyClass(r.accuracy)}`}>
                          {pct(r.accuracy)}
                        </span>
                      </div>
                    </div>
                    <div className="p-4">
                      <Plot
                        data={[
                          {
                            y: features,
                            x: importances,
                            type: 'bar',
                            orientation: 'h',
                            marker: {
                              color: importances.map((v, i) => {
                                const t = v / Math.max(...importances, 0.001);
                                // Tint latent-index bars purple to set them apart from raw columns.
                                return features[i] in clusterByIndex
                                  ? `rgba(188, 140, 255, ${0.35 + t * 0.65})`
                                  : `rgba(88, 166, 255, ${0.3 + t * 0.7})`;
                              }),
                            },
                            hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
                          },
                        ]}
                        layout={{
                          paper_bgcolor: 'transparent',
                          plot_bgcolor: 'transparent',
                          font: { color: '#8b949e', size: 10 },
                          margin: { l: 140, r: 20, t: 10, b: 30 },
                          xaxis: {
                            title: { text: 'Importance (Gain)', font: { size: 10 } },
                            gridcolor: '#21283b',
                            zerolinecolor: '#30363d',
                          },
                          yaxis: { autorange: 'reversed' },
                          height: 200,
                        }}
                        config={{ responsive: true, displayModeBar: false }}
                        style={{ width: '100%' }}
                      />
                    </div>
                  </div>
                );
              })
            : (
              <div className="overflow-auto rounded-lg border border-border bg-surface" style={{ maxHeight: 560 }}>
                <table className="w-full border-collapse text-[11px]">
                  <thead className="sticky top-0 bg-deep">
                    <tr className="text-left text-text-muted">
                      <th className="px-3 py-2">Target</th>
                      <th className="px-3 py-2">Holdout</th>
                      {hasCv && <th className="px-3 py-2">CV (mean ± std)</th>}
                      <th className="px-3 py-2">Feature</th>
                      <th className="px-3 py-2 text-right">Importance</th>
                      {hasSig && (
                        <>
                          <th className="px-3 py-2 text-right" title="Permutation-null empirical p-value: how often a shuffled-target refit matched this gain. Lower = more significant.">perm p</th>
                          <th className="px-3 py-2 text-right" title="Bootstrap selection frequency: fraction of resamples where the feature was used in any split. Higher = more stable.">stability</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((r, i) => {
                      const firstOfGroup = i === 0 || tableRows[i - 1].target !== r.target;
                      const sig = r.nullP != null && r.nullP < 0.05;
                      return (
                        <tr key={`${r.target}-${r.feature}-${i}`} className={firstOfGroup ? 'border-t border-border/60' : ''}>
                          <td className="px-3 py-1 font-medium text-text-secondary">{firstOfGroup ? r.target : ''}</td>
                          <td className="px-3 py-1">{firstOfGroup ? pct(r.accuracy) : ''}</td>
                          {hasCv && (
                            <td className="px-3 py-1 text-text-muted">
                              {firstOfGroup && r.cvMean != null ? `${pct(r.cvMean)} ± ${pct(r.cvStd ?? 0)}` : ''}
                            </td>
                          )}
                          <td className={`px-3 py-1 ${r.feature in clusterByIndex ? 'font-mono text-purple' : 'text-text-secondary'}`}>
                            {r.feature}
                          </td>
                          <td className="px-3 py-1 text-right font-mono text-text-secondary">{r.importance.toFixed(3)}</td>
                          {hasSig && (
                            <>
                              <td className={`px-3 py-1 text-right font-mono ${sig ? 'text-success' : r.nullP != null ? 'text-text-muted' : 'text-text-muted/40'}`}>
                                {r.nullP != null ? (r.nullP < 0.001 ? '<0.001' : r.nullP.toFixed(3)) : '—'}
                                {sig ? ' *' : ''}
                              </td>
                              <td className="px-3 py-1 text-right font-mono text-text-muted">
                                {r.selFreq != null ? `${Math.round(r.selFreq * 100)}%` : '—'}
                              </td>
                            </>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
        </div>

        {/* Side highlights — like the Cramér's V explorer */}
        <div className="space-y-4">
          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-text-primary">
              <Trophy className="h-3.5 w-3.5 text-success" /> Best predicted
              <span className="font-normal text-text-muted">{hasCv ? '(by CV)' : '(by holdout)'}</span>
            </div>
            <div className="space-y-1">
              {highlights.bestPredicted.map((p) => (
                <div key={p.col} className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="truncate text-text-secondary" title={p.col}>{p.col}</span>
                  <span className="shrink-0 font-mono font-semibold text-success">{pct(p.score)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-text-primary">
              <Activity className="h-3.5 w-3.5 text-danger" /> Hardest to predict
            </div>
            <div className="space-y-1">
              {highlights.hardest.map((p) => (
                <div key={p.col} className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="truncate text-text-secondary" title={p.col}>{p.col}</span>
                  <span className="shrink-0 font-mono font-semibold text-text-muted">{pct(p.score)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-text-primary">
              <Star className="h-3.5 w-3.5 text-accent" /> Most influential features
              <span className="font-normal text-text-muted">overall</span>
            </div>
            <div className="space-y-1">
              {highlights.topFeatures.map(([f, v]) => (
                <div key={f} className="flex items-center justify-between gap-2 text-[11px]">
                  <span
                    className={`truncate ${f in clusterByIndex ? 'font-mono text-purple' : 'text-text-secondary'}`}
                    title={f in clusterByIndex ? `Latent index of: ${clusterByIndex[f].members.join(', ')}` : f}
                  >
                    {f}
                  </span>
                  <span className="shrink-0 font-mono text-accent-bright">{v.toFixed(2)}</span>
                </div>
              ))}
              {highlights.topFeatures.length === 0 && (
                <p className="text-[11px] text-text-muted">No features used in any split.</p>
              )}
            </div>
          </div>

          {highlights.overfit.length > 0 && (
            <div className="rounded-lg border border-warning/30 bg-warning/5 p-3">
              <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-warning">
                <AlertTriangle className="h-3.5 w-3.5" /> Overfit watch
              </div>
              <p className="mb-2 text-[11px] text-text-muted">Holdout much higher than CV — treat accuracy as optimistic.</p>
              <div className="space-y-1">
                {highlights.overfit.map((p) => (
                  <div key={p.col} className="flex items-center justify-between gap-2 text-[11px]">
                    <span className="truncate text-text-secondary" title={p.col}>{p.col}</span>
                    <span className="shrink-0 font-mono text-warning">+{pct(p.gap ?? 0)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
