import { useEffect, useState } from 'react';
import { Search, AlertTriangle, Key, Database, Type, Image as ImageIcon, FileText, Film, AudioLines } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { LoadingSpinner } from '../common/LoadingSpinner';
import type { MultimodalResult } from '../../types';

const SOURCE_TYPES = [
  { id: '', label: 'All media' },
  { id: 'video_chunk', label: 'Video' },
  { id: 'audio_clip', label: 'Audio' },
  { id: 'pdf_page', label: 'PDF pages' },
];

export function MultimodalSearch() {
  const { dbUrl, setDbUrl, geminiKey, setGeminiKey } = useStore();
  const [mode, setMode] = useState<'text' | 'image'>('text');
  const [query, setQuery] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState('');
  const [release, setRelease] = useState('');
  const [releases, setReleases] = useState<string[]>([]);
  const [limit, setLimit] = useState(12);
  const [threshold, setThreshold] = useState(0.2);
  const [groupByParent, setGroupByParent] = useState(true);

  const [results, setResults] = useState<MultimodalResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Populate the release filter once a DB URL is present.
  useEffect(() => {
    if (!dbUrl) return;
    let cancelled = false;
    api.multimodalReleases(dbUrl).then((r) => !cancelled && setReleases(r.releases)).catch(() => {});
    return () => { cancelled = true; };
  }, [dbUrl]);

  const run = async () => {
    setError(null);
    if (!dbUrl || !geminiKey) {
      setError('Enter the Neon database URL and a Gemini API key.');
      return;
    }
    if (mode === 'text' && !query.trim()) {
      setError('Enter a text query.');
      return;
    }
    if (mode === 'image' && !imageFile) {
      setError('Choose an image to search with.');
      return;
    }
    setLoading(true);
    setResults(null);
    try {
      const common = { databaseUrl: dbUrl, geminiKey, sourceType: sourceType || undefined, release: release || undefined, limit, threshold, groupByParent };
      const res =
        mode === 'text'
          ? await api.multimodalSearch({ ...common, query })
          : await api.multimodalSearchImage({ ...common, file: imageFile! });
      setResults(res.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <Panel
        title="Multi-Modal Semantic Search"
        subtitle="Gemini 768-d text/image embeddings over the Neon + pgvector media archive (video · audio · PDF pages)"
      >
        <div className="space-y-4">
          {/* Connection */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs text-text-muted">Neon database URL</label>
              <div className="relative">
                <Database className="absolute left-2.5 top-2 h-3.5 w-3.5 text-text-muted" />
                <input
                  type="password"
                  value={dbUrl}
                  onChange={(e) => setDbUrl(e.target.value)}
                  placeholder="postgresql://…  (direct endpoint, sslmode=require)"
                  className="w-full rounded border border-border bg-deep py-1.5 pl-8 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-muted">Gemini API key</label>
              <div className="relative">
                <Key className="absolute left-2.5 top-2 h-3.5 w-3.5 text-text-muted" />
                <input
                  type="password"
                  value={geminiKey}
                  onChange={(e) => setGeminiKey(e.target.value)}
                  placeholder="Must match the gemini-embedding-2 model family"
                  className="w-full rounded border border-border bg-deep py-1.5 pl-8 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Query mode */}
          <div className="inline-flex rounded-md border border-border bg-raised p-0.5 text-xs">
            <button
              onClick={() => setMode('text')}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 transition-colors ${mode === 'text' ? 'bg-accent-dim text-white' : 'text-text-secondary hover:text-text-primary'}`}
            >
              <Type className="h-3.5 w-3.5" /> Text
            </button>
            <button
              onClick={() => setMode('image')}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 transition-colors ${mode === 'image' ? 'bg-accent-dim text-white' : 'text-text-secondary hover:text-text-primary'}`}
            >
              <ImageIcon className="h-3.5 w-3.5" /> Image
            </button>
          </div>

          {mode === 'text' ? (
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && run()}
              placeholder="Describe what to find — e.g. 'spherical object pulsing over water'"
              className="w-full rounded-md border border-border bg-deep px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
            />
          ) : (
            <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border bg-raised px-3 py-2 text-xs text-text-secondary transition-colors hover:border-accent">
              <ImageIcon className="h-4 w-4" />
              {imageFile ? imageFile.name : 'Choose an image (jpg / png / webp)…'}
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
              />
            </label>
          )}

          {/* Filters */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">Media type</label>
              <select
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value)}
                className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                {SOURCE_TYPES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">Release</label>
              <select
                value={release}
                onChange={(e) => setRelease(e.target.value)}
                className="w-full rounded border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="">All releases</option>
                {releases.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">Results: {limit}</label>
              <input type="range" min={5} max={50} value={limit} onChange={(e) => setLimit(Number(e.target.value))} className="mt-2 w-full accent-accent" />
            </div>
            <div>
              <label className="mb-1 block text-[11px] text-text-muted">Min similarity: {threshold.toFixed(2)}</label>
              <input type="range" min={0} max={0.9} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="mt-2 w-full accent-accent" />
            </div>
          </div>

          <label
            className="flex w-fit items-center gap-1.5 text-[11px] text-text-muted"
            title="Video/audio chunks of one asset share a near-identical, title-based embedding — collapse them so each source appears once instead of flooding the results with repeated titles. PDF pages stay distinct."
          >
            <input
              type="checkbox"
              checked={groupByParent}
              onChange={(e) => setGroupByParent(e.target.checked)}
              className="accent-accent"
            />
            Group video/audio chunks by source
          </label>

          <button
            onClick={run}
            disabled={loading}
            className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
          >
            <Search className="h-4 w-4" />
            {loading ? 'Searching…' : 'Search archive'}
          </button>
        </div>
      </Panel>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Embedding the query and searching pgvector…" />}

      {results && (
        <div className="space-y-3">
          <p className="text-xs text-text-muted">{results.length} result(s)</p>
          {results.map((r, i) => <ResultCard key={i} r={r} />)}
          {results.length === 0 && (
            <Panel title="No matches">
              <p className="text-sm text-text-muted">Nothing above the similarity threshold — lower it or broaden the media filter.</p>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}

function ResultCard({ r }: { r: MultimodalResult }) {
  const Icon = r.source_type === 'video_chunk' ? Film : r.source_type === 'audio_clip' ? AudioLines : FileText;
  return (
    <div className="rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <span className="flex items-center gap-2 text-xs font-medium text-text-primary">
          <Icon className="h-3.5 w-3.5 text-accent" />
          <code className="text-text-secondary">{r.parent_id}</code>
          <span className="text-text-muted">· {r.source_type}</span>
          {r.page != null && <span className="text-text-muted">· page {r.page}</span>}
          {r.start_seconds != null && (
            <span className="text-text-muted">· {r.start_seconds.toFixed(0)}s→{(r.end_seconds ?? 0).toFixed(0)}s</span>
          )}
          {r.chunk_matches > 1 && (
            <span className="text-text-muted" title="Number of near-duplicate chunks of this asset that matched">
              · {r.chunk_matches} segments
            </span>
          )}
        </span>
        <span className="shrink-0 rounded-full bg-raised px-2 py-0.5 font-mono text-[11px] text-accent-bright">
          {r.similarity != null ? r.similarity.toFixed(3) : '—'}
        </span>
      </div>
      <div className="space-y-2 p-4">
        {r.embedded_text && (
          <p className="text-xs leading-relaxed text-text-secondary">
            {r.embedded_text.length > 600 ? r.embedded_text.slice(0, 600) + '…' : r.embedded_text}
          </p>
        )}
        {r.media_embed_url && (
          <iframe
            src={r.media_embed_url}
            title={r.parent_id}
            className="w-full rounded border border-border"
            height={r.source_type === 'audio_clip' ? 170 : 360}
            allowFullScreen
          />
        )}
        {r.source_type === 'video_chunk' && r.start_seconds != null && (
          <p className="text-[11px] text-text-muted">
            DVIDS embeds ignore timestamps — this chunk is {r.start_seconds.toFixed(1)}s→{(r.end_seconds ?? 0).toFixed(1)}s; seek manually.
          </p>
        )}
        {r.source_url && (
          <a href={r.source_url} target="_blank" rel="noreferrer" className="inline-block text-xs text-accent hover:underline">
            {r.source_type === 'pdf_page' ? `Open full PDF (war.gov)${r.page != null ? ` · page ${r.page}` : ''}` : 'Open source page'} ↗
          </a>
        )}
        <div className="text-[10px] text-text-muted">{r.release} · {r.release_date}</div>
      </div>
    </div>
  );
}
