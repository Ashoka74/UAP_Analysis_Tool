import { useEffect, useMemo, useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  Play,
  Filter,
  FileText,
  Upload,
  GitCompare,
  RefreshCw,
} from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { DataTable } from '../data/DataTable';
import { Markdown } from '../common/Markdown';
import type {
  ScuCriteriaResponse,
  ScuNormalizeResponse,
  ScuFilterResponse,
  ScuColumnMapping,
} from '../../types';

// Badge styling per match method — exact matches are quiet, heuristic ones
// (suffix/leaf/fuzzy) stand out for review, manual overrides are accented.
const METHOD_STYLE: Record<string, string> = {
  exact: 'bg-raised text-text-muted border-border',
  manual: 'bg-accent-dim/30 text-accent-bright border-accent',
  suffix: 'bg-success/10 text-success border-success/40',
  leaf: 'bg-warning/10 text-warning border-warning/40',
  fuzzy: 'bg-purple/10 text-purple border-purple/40',
  unmatched: 'bg-danger/10 text-danger border-danger/40',
};
const methodStyle = (m: string) =>
  METHOD_STYLE[m.split('/')[0].replace('?', '')] ?? METHOD_STYLE.fuzzy;

export function ScuPage() {
  const { parsedReady, setPage } = useStore();
  const [criteria, setCriteria] = useState<ScuCriteriaResponse | null>(null);
  const [norm, setNorm] = useState<ScuNormalizeResponse | null>(null);
  const [filtered, setFiltered] = useState<ScuFilterResponse | null>(null);

  const [preset, setPreset] = useState<string>('');
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);

  const [loading, setLoading] = useState(false);
  const [filterLoading, setFilterLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Manual column-map overrides ({canonical: actual}) applied on top of the
  // automatic mapping via /api/scu/remap.
  const [remapLoading, setRemapLoading] = useState(false);
  const applyRemap = async (columnMap: Record<string, string>) => {
    setRemapLoading(true);
    setError(null);
    setFiltered(null);
    try {
      const res = await api.scuRemap(columnMap);
      setNorm(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Re-normalization failed');
    } finally {
      setRemapLoading(false);
    }
  };

  useEffect(() => {
    api.getScuCriteria().then(setCriteria).catch(() => {});
  }, []);

  const normalize = async () => {
    setLoading(true);
    setError(null);
    setFiltered(null);
    try {
      const res = await api.scuNormalize();
      setNorm(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Normalization failed');
    } finally {
      setLoading(false);
    }
  };

  // Normalize an uploaded, already-structured dataset (no prior LLM parse).
  const normalizeUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setFiltered(null);
    try {
      const res = await api.scuNormalizeUpload(file);
      setNorm(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Normalization failed');
    } finally {
      setLoading(false);
    }
  };

  const applyPreset = (name: string) => {
    setPreset(name);
    if (criteria && name && criteria.presets[name]) {
      setSelectedKeys(criteria.presets[name]);
    }
  };

  const runFilter = async () => {
    if (!selectedKeys.length) return;
    setFilterLoading(true);
    setError(null);
    try {
      const res = await api.scuFilter(selectedKeys);
      setFiltered(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Filter failed');
    } finally {
      setFilterLoading(false);
    }
  };

  const allCriteria = criteria ? [...criteria.criteria, ...criteria.extra_criteria] : [];
  const maxFunnel = filtered ? Math.max(...filtered.funnel.map((f) => f.count), 1) : 1;

  return (
    <div className="space-y-4">
      <Panel
        title="SCU Normalization"
        subtitle="Canonicalise parsed data and derive the SCU five-criterion eligibility gate"
        actions={
          <div className="flex items-center gap-2">
            <label
              title="Normalize an already-structured dataset (parsed CSV / XLSX / parsed_responses JSON) without re-parsing"
              className="flex cursor-pointer items-center gap-2 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
            >
              <Upload className="h-3.5 w-3.5" />
              Upload dataset
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.json"
                className="hidden"
                disabled={loading}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) normalizeUpload(f);
                  e.target.value = '';
                }}
              />
            </label>
            <button
              onClick={normalize}
              disabled={loading || !parsedReady}
              title={parsedReady ? undefined : 'No parsed data in session — parse a dataset or upload one'}
              className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5" />
              {loading ? 'Normalizing…' : 'Run on parsed data'}
            </button>
          </div>
        }
      >
        {!parsedReady && !norm && (
          <p className="text-sm text-text-muted">
            Run SCU normalization on the parsed responses from the{' '}
            <button onClick={() => setPage('parsing')} className="text-accent hover:underline">
              Parsing
            </button>{' '}
            step — or <span className="text-accent">Upload dataset</span> to normalize an
            already-structured file (a previously-parsed CSV/XLSX or a <code>parsed_responses</code> JSON)
            without re-running the extractor.
          </p>
        )}
        {parsedReady && !norm && (
          <p className="text-sm text-text-muted">
            Parsed data is ready. Click <span className="text-accent">Run on parsed data</span> (or{' '}
            <span className="text-accent">Upload dataset</span>) to canonicalise country/state codes,
            witness roles and craft shape/size bands, and compute the SCU eligibility criteria.
          </p>
        )}
      </Panel>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Normalizing dataset..." />}

      {norm && (
        <>
          {/* Metrics */}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard label="Rows" value={norm.metrics.rows} icon={FileText} />
            <MetricCard label="SCU-eligible" value={norm.metrics.scu_eligible} icon={ShieldCheck} accent />
            <MetricCard label="In 1945–1975 window" value={norm.metrics.in_scu_window} icon={FileText} />
            <MetricCard label="Credible witness" value={norm.metrics.has_credible_witness} icon={FileText} />
          </div>

          {/* Input-column mapping — review the auto-match, override manually */}
          {norm.mapping && (
            <ColumnMappingPanel
              mapping={norm.mapping}
              busy={remapLoading}
              onApply={applyRemap}
            />
          )}

          {/* Eligibility filter */}
          <Panel title="SCU Eligibility Filter" actions={<Filter className="h-4 w-4 text-text-muted" />}>
            {criteria && (
              <div className="space-y-3">
                <div>
                  <label className="mb-1 block text-xs text-text-muted">Preset</label>
                  <select
                    value={preset}
                    onChange={(e) => applyPreset(e.target.value)}
                    className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                  >
                    <option value="">Custom selection</option>
                    {Object.keys(criteria.presets).map((p) => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
                <div className="flex flex-wrap gap-2">
                  {allCriteria.map((c) => (
                    <button
                      key={c.key}
                      onClick={() => {
                        setPreset('');
                        setSelectedKeys((p) =>
                          p.includes(c.key) ? p.filter((k) => k !== c.key) : [...p, c.key]
                        );
                      }}
                      title={c.label}
                      className={`rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
                        selectedKeys.includes(c.key)
                          ? 'border-accent bg-accent-dim/30 text-accent-bright'
                          : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      {c.label}
                    </button>
                  ))}
                </div>
                <button
                  onClick={runFilter}
                  disabled={!selectedKeys.length || filterLoading}
                  className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
                >
                  <Filter className="h-3.5 w-3.5" />
                  {filterLoading ? 'Filtering…' : 'Apply Filter'}
                </button>
              </div>
            )}

            {/* Funnel */}
            {filtered && (
              <div className="mt-4 space-y-1.5">
                <p className="text-xs font-medium text-text-secondary">
                  Eligibility funnel — {filtered.n_passed.toLocaleString()} rows pass all criteria
                </p>
                {filtered.funnel.map((f, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="w-56 shrink-0 truncate text-[11px] text-text-muted" title={f.stage}>
                      {f.stage}
                    </span>
                    <div className="h-4 flex-1 overflow-hidden rounded bg-deep">
                      <div
                        className="h-full rounded bg-accent/60"
                        style={{ width: `${(f.count / maxFunnel) * 100}%` }}
                      />
                    </div>
                    <span className="w-16 shrink-0 text-right text-[11px] font-mono text-text-secondary">
                      {f.count.toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Panel>

          {/* Audit report */}
          {norm.audit_markdown && (
            <Panel title="Normalization Audit">
              <div className="max-h-96 overflow-auto rounded bg-deep/40 p-3">
                <Markdown>{norm.audit_markdown}</Markdown>
              </div>
            </Panel>
          )}

          {/* Data table — filtered result if present, else normalized. The JSON
              payload is a 500-row preview; the button downloads the full set. */}
          {(() => {
            const active = filtered ? filtered.data : norm.data;
            const scope: 'filtered' | 'normalized' = filtered ? 'filtered' : 'normalized';
            return (
              <Panel
                title={filtered ? 'Filtered (SCU-eligible) Rows' : 'Normalized Data'}
                subtitle={
                  `${active.returned_rows.toLocaleString()} of ${active.total_rows.toLocaleString()} rows shown` +
                  (active.total_rows > active.returned_rows ? ' · table preview is capped, download for all rows' : '')
                }
                actions={
                  <button
                    onClick={() =>
                      api.scuExportCsv(scope).catch((e) =>
                        setError(e instanceof Error ? e.message : 'Export failed'))
                    }
                    title={`Download the full ${scope} dataset as CSV (the in-table export only has the displayed rows)`}
                    className="flex items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
                  >
                    <FileText className="h-3.5 w-3.5" />
                    Download full CSV ({active.total_rows.toLocaleString()} rows)
                  </button>
                }
                noPad
              >
                <DataTable data={active} maxHeight="calc(100vh - 420px)" />
              </Panel>
            );
          })()}
        </>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  icon: Icon,
  accent,
}: {
  label: string;
  value: number;
  icon: typeof ShieldCheck;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center gap-2">
        <Icon className={`h-4 w-4 ${accent ? 'text-accent' : 'text-text-muted'}`} />
        <p className="text-[10px] uppercase tracking-wider text-text-muted">{label}</p>
      </div>
      <p className={`mt-1 text-2xl font-bold ${accent ? 'text-accent-bright' : 'text-text-primary'}`}>
        {value.toLocaleString()}
      </p>
    </div>
  );
}

// Review / override how raw input columns were mapped onto the canonical fields
// the SCU gate reads. Heuristic matches (suffix/leaf/fuzzy) and unmatched fields
// are shown by default; exact matches are behind a toggle. Overrides re-run the
// normalizer via /api/scu/remap.
function ColumnMappingPanel({
  mapping,
  busy,
  onApply,
}: {
  mapping: ScuColumnMapping;
  busy: boolean;
  onApply: (columnMap: Record<string, string>) => void;
}) {
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [showExact, setShowExact] = useState(false);

  // Reset pending overrides whenever a new mapping arrives (post-remap).
  useEffect(() => setOverrides({}), [mapping]);

  const rows = useMemo(() => {
    const method = (c: string) =>
      mapping.unmatched.includes(c) ? 'unmatched' : mapping.methods[c] ?? 'unmatched';
    return mapping.expected
      .map((c) => ({
        canonical: c,
        actual: mapping.resolved[c] ?? '',
        method: method(c),
        score: mapping.scores[c],
      }))
      .filter((r) => showExact || r.method !== 'exact')
      .sort((a, b) => {
        // unmatched first, then heuristic, then manual/exact
        const rank = (m: string) => (m === 'unmatched' ? 0 : m === 'exact' ? 2 : 1);
        return rank(a.method) - rank(b.method) || a.canonical.localeCompare(b.canonical);
      });
  }, [mapping, showExact]);

  const nHeuristic = mapping.expected.filter((c) => {
    const m = mapping.methods[c];
    return m && m !== 'exact' && m !== 'manual';
  }).length;
  const dirty = Object.keys(overrides).length > 0;

  return (
    <Panel
      title="Input Column Mapping"
      subtitle="How your dataset's columns were matched to the fields the SCU gate reads — override any wrong match"
      actions={
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-[11px] text-text-muted">
            <input
              type="checkbox"
              checked={showExact}
              onChange={(e) => setShowExact(e.target.checked)}
              className="accent-accent"
            />
            Show exact matches
          </label>
          <button
            onClick={() => onApply(overrides)}
            disabled={busy || !dirty}
            title={dirty ? 'Re-run normalization with your manual mapping' : 'Change a mapping below first'}
            className="flex items-center gap-1.5 rounded-md bg-accent-dim px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${busy ? 'animate-spin' : ''}`} />
            {busy ? 'Re-normalizing…' : `Apply mapping${dirty ? ` (${Object.keys(overrides).length})` : ''}`}
          </button>
        </div>
      }
    >
      <p className="mb-2 flex items-center gap-1.5 text-[11px] text-text-muted">
        <GitCompare className="h-3.5 w-3.5" />
        {mapping.unmatched.length} unmatched · {nHeuristic} heuristic match(es) (suffix / leaf-name /
        fuzzy) — review these; exact matches are hidden by default.
      </p>
      <div className="max-h-80 overflow-auto rounded border border-border/40">
        <table className="w-full border-collapse text-[11px]">
          <thead className="sticky top-0 bg-deep">
            <tr className="text-left text-text-muted">
              <th className="px-3 py-2">Expected field (gate input)</th>
              <th className="px-3 py-2">Mapped from</th>
              <th className="px-3 py-2">Match</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const pending = overrides[r.canonical];
              const shown = pending ?? r.actual;
              return (
                <tr key={r.canonical} className="border-t border-border/30">
                  <td className="px-3 py-1.5 font-mono text-text-secondary">{r.canonical}</td>
                  <td className="px-3 py-1.5">
                    <select
                      value={shown}
                      onChange={(e) => {
                        const v = e.target.value;
                        setOverrides((p) => {
                          const next = { ...p };
                          if (!v || v === r.actual) delete next[r.canonical];
                          else next[r.canonical] = v;
                          return next;
                        });
                      }}
                      className={`w-full max-w-xs rounded border bg-deep px-2 py-1 text-[11px] focus:border-accent focus:outline-none ${
                        pending ? 'border-accent text-accent-bright' : 'border-border text-text-primary'
                      }`}
                    >
                      <option value="">— not mapped —</option>
                      {mapping.actual_columns.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </td>
                  <td className="px-3 py-1.5">
                    <span
                      className={`inline-block rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                        pending ? methodStyle('manual') : methodStyle(r.method)
                      }`}
                      title={r.score != null ? `confidence ${r.score}` : undefined}
                    >
                      {pending ? 'manual (pending)' : r.method}
                    </span>
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td colSpan={3} className="px-3 py-3 text-center text-text-muted">
                  Every gate input matched its column exactly — nothing to review.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
