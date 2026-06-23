import { useEffect, useMemo, useState } from 'react';
import {
  Upload,
  FileSearch,
  AlertTriangle,
  CheckCircle2,
  Play,
  DollarSign,
  Key,
  Layers,
  ShieldCheck,
  GitCompare,
  Database,
  Plus,
} from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { DataTable } from '../data/DataTable';
import type {
  SchemaListResponse,
  CostEstimate,
  ParseRunResponse,
  DataResponse,
  SchemaCoverageResponse,
  CoverageMode,
} from '../../types';

const COVERAGE_MODES: { id: CoverageMode; label: string; hint: string }[] = [
  { id: 'missing', label: 'Add missing only', hint: 'Extract only schema fields the dataset is missing (🔴)' },
  { id: 'all', label: 'All fields', hint: 'Extract every schema field, present or not' },
  { id: 'database', label: 'Database only', hint: 'No extraction — keep the uploaded columns as-is' },
];

export function ParsingPage() {
  const {
    openaiKey,
    setOpenaiKey,
    deepseekKey,
    setDeepseekKey,
    setParsedReady,
    setPage,
  } = useStore();

  const [schemas, setSchemas] = useState<SchemaListResponse | null>(null);
  const [selectedSchemas, setSelectedSchemas] = useState<string[]>([]);
  const [mergedFormat, setMergedFormat] = useState('');
  const [fieldCount, setFieldCount] = useState(0);

  // Schema ↔ dataset coverage diff + extraction mode (missing / all / database)
  const [coverage, setCoverage] = useState<SchemaCoverageResponse | null>(null);
  const [coverageMode, setCoverageMode] = useState<CoverageMode>('all');

  const [source, setSource] = useState<DataResponse | null>(null);
  const [sourceColumns, setSourceColumns] = useState<string[]>([]);
  const [textColumns, setTextColumns] = useState<string[]>([]);
  const [keepColumns, setKeepColumns] = useState<string[]>([]);

  const [provider, setProvider] = useState<'openai' | 'deepseek'>('openai');
  const [model, setModel] = useState('gpt-4o-mini');
  const [maxWorkers, setMaxWorkers] = useState(10);

  const [estimate, setEstimate] = useState<CostEstimate | null>(null);
  const [result, setResult] = useState<ParseRunResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load schema registry + model lists on mount
  useEffect(() => {
    api
      .getSchemas()
      .then((s) => {
        setSchemas(s);
        if (s.labels.length > 0) setSelectedSchemas([s.labels[0]]);
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load schemas'));
  }, []);

  // Keep the model valid for the selected provider
  useEffect(() => {
    if (!schemas) return;
    const list = provider === 'openai' ? schemas.models.openai : schemas.models.deepseek;
    if (list.length && !list.includes(model)) setModel(list[0]);
  }, [provider, schemas]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-merge schemas / recompute the coverage diff whenever the schema
  // selection or the uploaded dataset columns change. With a dataset present we
  // fetch the diff (which also carries the per-mode extraction schemas); the
  // effective mergedFormat is then derived from the active mode below.
  useEffect(() => {
    if (selectedSchemas.length === 0) {
      setMergedFormat('');
      setFieldCount(0);
      setCoverage(null);
      return;
    }
    if (sourceColumns.length) {
      api
        .schemaCoverage(selectedSchemas, sourceColumns)
        .then(setCoverage)
        .catch((e) => setError(e instanceof Error ? e.message : 'Coverage diff failed'));
    } else {
      setCoverage(null);
      api
        .mergeSchema(selectedSchemas)
        .then((m) => {
          setMergedFormat(m.schema_json);
          setFieldCount(m.fields.length);
        })
        .catch((e) => setError(e instanceof Error ? e.message : 'Schema merge failed'));
    }
  }, [selectedSchemas, sourceColumns]);

  // Derive the effective extraction schema from the chosen coverage mode.
  useEffect(() => {
    if (!coverage) return;
    const variant = coverage.variants[coverageMode];
    setMergedFormat(variant.schema_json);
    setFieldCount(variant.n_fields);
  }, [coverage, coverageMode]);

  const apiKey = provider === 'openai' ? openaiKey : deepseekKey;
  const setApiKey = provider === 'openai' ? setOpenaiKey : setDeepseekKey;
  const modelList = schemas
    ? provider === 'openai'
      ? schemas.models.openai
      : schemas.models.deepseek
    : [];

  // "Database only" mode yields an empty schema ({}), so there is nothing to
  // estimate or extract — the uploaded columns are kept as-is.
  const extractionReady = !!mergedFormat && mergedFormat.trim() !== '{}';

  const addToKeep = (cols: string[]) =>
    setKeepColumns((prev) => Array.from(new Set([...prev, ...cols])));

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setEstimate(null);
    try {
      const res = await api.uploadParseFile(file);
      setSource(res.data);
      setSourceColumns(res.columns);
      setTextColumns(res.columns.length ? [res.columns[0]] : []);
      setKeepColumns([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const toggle = (list: string[], v: string) =>
    list.includes(v) ? list.filter((x) => x !== v) : [...list, v];

  const runEstimate = async () => {
    if (!textColumns.length || !extractionReady) return;
    setError(null);
    try {
      const est = await api.estimateParse(textColumns, mergedFormat, model);
      setEstimate(est);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Estimate failed');
    }
  };

  const runParse = async () => {
    if (!textColumns.length || !extractionReady) {
      setError('Select at least one text column and a schema with fields to extract.');
      return;
    }
    if (!apiKey) {
      setError(`Enter your ${provider === 'openai' ? 'OpenAI' : 'DeepSeek'} API key.`);
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const res = await api.runParse({
        columns: textColumns,
        format_json: mergedFormat,
        provider,
        model,
        api_key: apiKey,
        max_workers: maxWorkers,
        keep_columns: keepColumns,
      });
      setResult(res);
      setParsedReady(res.n_ok > 0);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Parsing failed');
    } finally {
      setRunning(false);
    }
  };

  const groups = useMemo(() => (schemas ? Object.entries(schemas.groups) : []), [schemas]);

  return (
    <div className="space-y-4">
      {/* Upload */}
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border bg-surface px-4 py-2 text-sm text-text-primary transition-colors hover:border-purple hover:bg-elevated">
          <Upload className="h-4 w-4 text-purple" />
          Upload Raw Reports (CSV / XLSX / JSON)
          <input type="file" accept=".csv,.xlsx,.xls,.json" onChange={handleUpload} className="hidden" />
        </label>
        {source && (
          <div className="ml-auto flex items-center gap-2 text-xs text-text-muted">
            <FileSearch className="h-4 w-4" />
            <span>{source.total_rows.toLocaleString()} rows · {sourceColumns.length} columns</span>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Loading raw dataset..." />}

      {!source && !loading && (
        <Panel title="LLM Feature Extraction">
          <p className="text-sm text-text-muted">
            Upload a dataset of raw UAP report text. Select the column(s) holding the narrative,
            choose one or more output schemas, then run GPT/DeepSeek extraction into structured JSON.
          </p>
        </Panel>
      )}

      {source && coverage && (
        <Panel
          title="Schema ↔ Dataset Coverage"
          subtitle="How many merged-schema fields the uploaded dataset already provides"
          actions={
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1 text-success">
                <span className="h-2 w-2 rounded-full bg-success" /> {coverage.summary.present} present
              </span>
              <span className="flex items-center gap-1 text-danger">
                <span className="h-2 w-2 rounded-full bg-danger" /> {coverage.summary.missing} missing
              </span>
              <span className="flex items-center gap-1 text-text-muted">
                <Database className="h-3 w-3" /> {coverage.summary.db_only} DB-only
              </span>
            </div>
          }
        >
          {/* Extraction-mode selector */}
          <div className="mb-2 flex flex-wrap gap-1.5">
            {COVERAGE_MODES.map((m) => {
              const active = coverageMode === m.id;
              const n = coverage.variants[m.id].n_fields;
              const Icon = m.id === 'missing' ? Plus : m.id === 'all' ? Layers : Database;
              return (
                <button
                  key={m.id}
                  onClick={() => setCoverageMode(m.id)}
                  title={m.hint}
                  className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                    active
                      ? 'border-accent bg-accent-dim/30 text-accent-bright'
                      : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {m.label}
                  {m.id !== 'database' && <span className="text-text-muted">({n})</span>}
                </button>
              );
            })}
          </div>
          <p className="mb-3 flex items-center gap-1.5 text-[11px] text-text-muted">
            <GitCompare className="h-3.5 w-3.5" />
            {COVERAGE_MODES.find((m) => m.id === coverageMode)?.hint}
          </p>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {/* Schema fields (🟢 present / 🔴 missing) */}
            <div>
              <p className="mb-1.5 text-[11px] font-semibold text-text-secondary">Schema fields</p>
              <div className="max-h-56 overflow-y-auto rounded border border-border/40 bg-deep/40 p-2">
                <div className="flex flex-wrap gap-1.5">
                  {coverage.coverage.map((f) => {
                    // Which fields get extracted in the active mode.
                    const extracted =
                      coverageMode === 'all' ? true : coverageMode === 'missing' ? !f.present : false;
                    return (
                      <span
                        key={f.path}
                        title={`${f.path}${f.matched_column ? `  ←  ${f.matched_column}` : ''}`}
                        className={`flex items-center gap-1 rounded border px-2 py-0.5 text-[11px] ${
                          f.present
                            ? 'border-success/40 bg-success/10 text-success'
                            : 'border-danger/40 bg-danger/10 text-danger'
                        } ${extracted ? '' : 'opacity-40'}`}
                      >
                        {f.present ? '🟢' : '🔴'} {f.leaf}
                      </span>
                    );
                  })}
                </div>
              </div>
              <p className="mt-1 text-[10px] text-text-muted">
                Dimmed chips aren’t extracted in this mode · matched by field (leaf) name.
              </p>
            </div>

            {/* Database-only columns (in dataset, not in schema) */}
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <p className="text-[11px] font-semibold text-text-secondary">
                  Database-only columns ({coverage.db_only_columns.length})
                </p>
                {coverage.db_only_columns.length > 0 && (
                  <button
                    onClick={() => addToKeep(coverage.db_only_columns)}
                    className="flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:border-accent hover:text-accent"
                    title="Append these to the carry-through columns"
                  >
                    <Plus className="h-3 w-3" /> Carry through
                  </button>
                )}
              </div>
              <div className="max-h-56 overflow-y-auto rounded border border-border/40 bg-deep/40 p-2">
                {coverage.db_only_columns.length ? (
                  <div className="flex flex-wrap gap-1.5">
                    {coverage.db_only_columns.map((c) => (
                      <span
                        key={c}
                        className="rounded border border-border bg-raised px-2 py-0.5 text-[11px] text-text-secondary"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-[11px] text-text-muted">Every dataset column maps to a schema field.</p>
                )}
              </div>
              {coverage.matched_columns.length > 0 && (
                <button
                  onClick={() => addToKeep(coverage.matched_columns)}
                  className="mt-1.5 flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:border-accent hover:text-accent"
                  title="Append the dataset columns that already cover schema fields to carry-through"
                >
                  <Plus className="h-3 w-3" /> Carry through {coverage.matched_columns.length} matched column(s)
                </button>
              )}
            </div>
          </div>
        </Panel>
      )}

      {source && (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
          {/* Left: column + schema config */}
          <div className="space-y-4 xl:col-span-1">
            <Panel title="1 · Text Columns" subtitle="Concatenated with ' - ' per row">
              <div className="flex flex-wrap gap-2">
                {sourceColumns.map((c) => (
                  <button
                    key={c}
                    onClick={() => setTextColumns((p) => toggle(p, c))}
                    className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${
                      textColumns.includes(c)
                        ? 'border-accent bg-accent-dim/30 text-accent-bright'
                        : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </Panel>

            <Panel title="2 · Output Schema(s)" subtitle={`${fieldCount} leaf fields merged`}>
              <div className="max-h-72 space-y-3 overflow-y-auto pr-1">
                {groups.map(([group, labels]) => (
                  <div key={group}>
                    <p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-text-muted">
                      {group}
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {labels.map((l) => (
                        <button
                          key={l}
                          onClick={() => setSelectedSchemas((p) => toggle(p, l))}
                          className={`rounded-md border px-2 py-1 text-[11px] transition-colors ${
                            selectedSchemas.includes(l)
                              ? 'border-purple bg-purple/20 text-text-primary'
                              : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                          }`}
                        >
                          {l}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="Carry-through columns" subtitle="Appended to parsed output (optional)">
              <div className="flex flex-wrap gap-2">
                {sourceColumns.map((c) => (
                  <button
                    key={c}
                    onClick={() => setKeepColumns((p) => toggle(p, c))}
                    className={`rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
                      keepColumns.includes(c)
                        ? 'border-accent bg-accent-dim/30 text-accent-bright'
                        : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </Panel>
          </div>

          {/* Middle: provider/model/key + schema preview */}
          <div className="space-y-4 xl:col-span-1">
            <Panel title="3 · Provider & Model">
              <div className="space-y-3">
                <div className="flex gap-2">
                  {(['openai', 'deepseek'] as const).map((p) => (
                    <button
                      key={p}
                      onClick={() => setProvider(p)}
                      className={`flex-1 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                        provider === p
                          ? 'border-accent bg-accent-dim/30 text-accent-bright'
                          : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      {p === 'openai' ? 'OpenAI' : 'DeepSeek'}
                    </button>
                  ))}
                </div>
                <div>
                  <label className="mb-1 block text-xs text-text-muted">Model</label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                  >
                    {modelList.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-text-muted">
                    {provider === 'openai' ? 'OpenAI' : 'DeepSeek'} API Key
                  </label>
                  <div className="relative">
                    <Key className="absolute left-2.5 top-2 h-3.5 w-3.5 text-text-muted" />
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="Enter API key..."
                      className="w-full rounded border border-border bg-deep py-1.5 pl-8 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                    />
                  </div>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-text-muted">
                    Max parallel workers: {maxWorkers}
                  </label>
                  <input
                    type="range"
                    min={1}
                    max={64}
                    value={maxWorkers}
                    onChange={(e) => setMaxWorkers(Number(e.target.value))}
                    className="w-full accent-accent"
                  />
                </div>
              </div>
            </Panel>

            <Panel
              title="Schema Preview"
              actions={
                <button
                  onClick={runEstimate}
                  disabled={!textColumns.length || !extractionReady}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-raised px-3 py-1 text-xs text-text-secondary hover:border-accent hover:text-accent disabled:opacity-50"
                >
                  <DollarSign className="h-3.5 w-3.5" /> Estimate Cost
                </button>
              }
            >
              <textarea
                value={mergedFormat}
                readOnly
                className="h-44 w-full resize-none rounded border border-border bg-deep p-2 font-mono text-[10px] text-text-secondary focus:outline-none"
              />
            </Panel>
          </div>

          {/* Right: estimate + run */}
          <div className="space-y-4 xl:col-span-1">
            {estimate && (
              <Panel title="Estimated Cost">
                <div className="grid grid-cols-2 gap-3">
                  <Metric label="Total" value={`$${estimate.total_usd.toFixed(4)}`} highlight />
                  <Metric label="Queries" value={estimate.n_queries.toLocaleString()} />
                  <Metric label="Input tokens" value={estimate.total_input_tokens.toLocaleString()} />
                  <Metric label="Output tokens" value={estimate.output_tokens.toLocaleString()} />
                </div>
                <p className="mt-2 text-[10px] text-text-muted">
                  {estimate.model} · {estimate.provider} · avg {estimate.avg_desc_tokens} tok/query
                </p>
              </Panel>
            )}

            <Panel title="4 · Run Extraction">
              <button
                onClick={runParse}
                disabled={running || !textColumns.length || !extractionReady || !apiKey}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-accent-dim px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {running ? 'Parsing…' : 'Parse Dataset'}
              </button>
              {!extractionReady && coverageMode === 'database' ? (
                <p className="mt-2 text-[11px] text-warning">
                  Database-only mode: no extraction needed — your uploaded columns are kept as-is.
                  Switch to “Add missing only” or “All fields” to run the LLM.
                </p>
              ) : (
                <p className="mt-2 text-[11px] text-text-muted">
                  {textColumns.length} text column(s), {fieldCount} field(s) to extract ({coverageMode}).
                </p>
              )}
            </Panel>

            {result && (
              <Panel title="Result">
                <div className="flex items-center gap-2 text-sm text-success">
                  <CheckCircle2 className="h-4 w-4" />
                  {result.n_ok} / {result.n_total} parsed
                </div>
                {result.n_failed > 0 && (
                  <p className="mt-1 text-xs text-warning">{result.n_failed} failed</p>
                )}
                <button
                  onClick={() => setPage('scu')}
                  className="mt-3 flex w-full items-center justify-center gap-2 rounded-md border border-purple/40 bg-purple/10 px-3 py-2 text-xs font-medium text-text-primary hover:bg-purple/20"
                >
                  <ShieldCheck className="h-3.5 w-3.5" /> Run SCU Normalization →
                </button>
              </Panel>
            )}
          </div>
        </div>
      )}

      {running && <LoadingSpinner text="Sending reports to the LLM… this may take a while." />}

      {/* Parsed output table */}
      {result && result.data.columns.length > 0 && (
        <Panel
          title="Parsed Output"
          subtitle={`${result.data.returned_rows} rows · ${result.data.columns.length} fields`}
          actions={<Layers className="h-4 w-4 text-text-muted" />}
          noPad
        >
          <DataTable data={result.data} maxHeight="calc(100vh - 360px)" />
        </Panel>
      )}
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded border border-border/50 bg-raised p-2.5">
      <p className="text-[10px] uppercase tracking-wider text-text-muted">{label}</p>
      <p className={`mt-0.5 text-sm font-semibold ${highlight ? 'text-accent-bright' : 'text-text-primary'}`}>
        {value}
      </p>
    </div>
  );
}
