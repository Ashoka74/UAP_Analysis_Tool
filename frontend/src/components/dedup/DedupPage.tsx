import { useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Play,
  Sparkles,
  Layers,
  Database,
  FileText,
  Calendar,
  MapPin,
  Puzzle,
  Zap,
  Target,
  Globe,
  Download,
  Upload,
} from 'lucide-react';
import type {
  AdvancedDedupResponse,
  CrossDbPair,
  DedupCluster,
  CanonicalDistancePoint,
} from '../../types';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';
import { ClusterSpatioTemporalChart } from './ClusterSpatioTemporalChart';

function DedupeProgressBar({ loading, title }: { loading: boolean; title?: string }) {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');

  useEffect(() => {
    if (!loading) {
      setProgress(100);
      return;
    }
    setProgress(8);
    setStage('Stage 1/4: Extracting & normalizing report metadata & composite text features...');

    const t1 = setTimeout(() => {
      setProgress(32);
      setStage('Stage 2/4: Computing dense vector embeddings using Harrier (microsoft/harrier-oss-v1-270m)...');
    }, 700);

    const t2 = setTimeout(() => {
      setProgress(64);
      setStage('Stage 3/4: Evaluating pairwise cosine similarity & screening multi-gate spatial/temporal criteria...');
    }, 1800);

    const t3 = setTimeout(() => {
      setProgress(86);
      setStage('Stage 4/4: Constructing transitive duplicate clustering graph & allocating similarity bins...');
    }, 3500);

    const t4 = setTimeout(() => {
      setProgress(95);
      setStage('Finalizing batch matrix evaluation across candidate pairs...');
    }, 5500);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
    };
  }, [loading]);

  if (!loading) return null;

  return (
    <div className="mt-3 flex flex-col gap-2 rounded-md border border-border bg-raised p-3">
      <div className="flex items-center justify-between text-xs font-medium">
        <span className="flex items-center gap-2 text-accent">
          <Sparkles className="h-3.5 w-3.5 animate-spin" /> {title || 'Running Harrier Deduplication Pipeline...'}
        </span>
        <span className="font-mono text-text-secondary">{progress}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-deep">
        <div
          className="h-full rounded-full bg-accent transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className="truncate font-mono text-[11px] text-text-muted">{stage}</span>
    </div>
  );
}

// Full side-by-side comparison for one candidate pair — shared by the
// Cross-DB tab's Preview and the Batch Cluster tab's Stage A audit trail
// preview, since both operate on the same CrossDbPair shape.
function AlignedInspector({ pair }: { pair: CrossDbPair }) {
  return (
    <div className="flex flex-col">
      {/* Inspector Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex flex-col">
          <span className="flex items-center gap-2 text-sm font-medium text-text-primary">
            <Database className="h-4 w-4 text-text-muted" /> Pair #{pair.id_a} vs #{pair.id_b}
          </span>
          <span className="mt-0.5 text-xs text-text-muted">
            Record A and Record B, aligned side-by-side by database column
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="rounded border border-border bg-deep px-2.5 py-1 font-mono text-xs text-text-secondary">
            Sim: {(pair.similarity * 100).toFixed(1)}%
          </span>
          {pair.haversine_km !== null && (
            <span className="rounded border border-border bg-deep px-2.5 py-1 font-mono text-xs text-text-secondary">
              Dist: {pair.haversine_km} km
            </span>
          )}
          {pair.date_diff_days !== null && (
            <span className="rounded border border-border bg-deep px-2.5 py-1 font-mono text-xs text-text-secondary">
              Time Gap: {pair.date_diff_days} days
            </span>
          )}
        </div>
      </div>

      {/* Aligned Side-by-Side Comparison Table */}
      <div className="max-h-[580px] overflow-x-auto overflow-y-auto">
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="sticky top-0 border-b border-border bg-deep font-mono text-text-secondary">
              <th className="w-1/4 p-3.5">Database Attribute</th>
              <th className="w-3/8 border-l border-border/50 p-3.5 text-accent">
                Row A (#{pair.id_a})
              </th>
              <th className="w-3/8 border-l border-border/50 p-3.5 text-text-primary">
                Row B (#{pair.id_b})
              </th>
              <th className="w-1/6 border-l border-border/50 p-3.5">Gate Alignment</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40 text-text-primary">
            {/* Record ID Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 font-mono font-medium text-text-secondary">Primary Record ID</td>
              <td className="border-l border-border/50 p-3.5 font-mono font-medium text-accent">{pair.id_a}</td>
              <td className="border-l border-border/50 p-3.5 font-mono font-medium text-text-primary">{pair.id_b}</td>
              <td className="border-l border-border/50 p-3.5 font-mono text-text-muted">Candidate Pair</td>
            </tr>

            {/* Witness Notes / Narrative Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 align-top font-medium text-text-primary">
                Witness Notes / Narrative Text
              </td>
              <td className="border-l border-border/50 p-3.5 align-top leading-relaxed text-text-primary">
                {pair.row_a?.witness_notes || pair.row_a?.narrative || pair.text_a_preview || '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top leading-relaxed text-text-primary">
                {pair.row_b?.witness_notes || pair.row_b?.narrative || pair.text_b_preview || '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top">
                <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                  pair.flags.is_similar_text ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                }`}>
                  <span className="flex items-center gap-1">
                    {pair.flags.is_similar_text ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                    {pair.flags.is_similar_text ? 'GATE PASSED' : 'BELOW THRESHOLD'}
                  </span>
                  <span className="text-[10px] font-normal">{(pair.similarity * 100).toFixed(1)}% Cosine</span>
                </span>
              </td>
            </tr>

            {/* Date & Time Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 align-top font-medium text-text-secondary">Timestamp / Date</td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {String(pair.row_a?.date_time || pair.row_a?.date || '—')}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {String(pair.row_b?.date_time || pair.row_b?.date || '—')}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top">
                {pair.date_diff_days !== null ? (
                  <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                    pair.flags.is_similar_date ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                  }`}>
                    <span className="flex items-center gap-1">
                      {pair.flags.is_similar_date ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {pair.flags.is_similar_date ? 'GATE PASSED' : 'OUT OF RANGE'}
                    </span>
                    <span className="text-[10px] font-normal">Δ {pair.date_diff_days} days</span>
                  </span>
                ) : (
                  <span className="block text-center font-mono text-text-muted">No Date Data</span>
                )}
              </td>
            </tr>

            {/* Coordinates Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 align-top font-medium text-text-secondary">Spatial Coordinates</td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {pair.row_a?.latitude !== undefined ? `${Number(pair.row_a.latitude).toFixed(4)}, ${Number(pair.row_a.longitude).toFixed(4)}` : '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {pair.row_b?.latitude !== undefined ? `${Number(pair.row_b.latitude).toFixed(4)}, ${Number(pair.row_b.longitude).toFixed(4)}` : '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top">
                {pair.haversine_km !== null ? (
                  <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                    pair.flags.is_similar_location ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                  }`}>
                    <span className="flex items-center gap-1">
                      {pair.flags.is_similar_location ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {pair.flags.is_similar_location ? 'GATE PASSED' : 'OUT OF RANGE'}
                    </span>
                    <span className="text-[10px] font-normal">Δ {pair.haversine_km} km</span>
                  </span>
                ) : pair.location_name_similarity !== null ? (
                  <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                    pair.flags.is_similar_location ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                  }`}>
                    <span className="flex items-center gap-1">
                      {pair.flags.is_similar_location ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {pair.flags.is_similar_location ? 'GATE PASSED' : 'BELOW THRESHOLD'}
                    </span>
                    <span className="text-[10px] font-normal">Name sim: {(pair.location_name_similarity * 100).toFixed(0)}%</span>
                  </span>
                ) : (
                  <span className="block text-center font-mono text-text-muted">No Lat/Lon</span>
                )}
              </td>
            </tr>

            {/* Additional Aligned Database Columns */}
            {(() => {
              const rowAKeys = pair.row_a ? Object.keys(pair.row_a) : [];
              const rowBKeys = pair.row_b ? Object.keys(pair.row_b) : [];
              const ignoreKeys = ['id', 'locus_tag', 'case_id', 'witness_notes', 'narrative', 'description', 'summary', 'text', 'date_time', 'date', 'datetime', 'latitude', 'lat', 'longitude', 'lon', 'lng'];
              const allOtherKeys = Array.from(new Set([...rowAKeys, ...rowBKeys])).filter(k => !ignoreKeys.includes(k.toLowerCase()));

              return allOtherKeys.map((key) => {
                const valA = pair.row_a?.[key] !== undefined ? String(pair.row_a[key]) : '—';
                const valB = pair.row_b?.[key] !== undefined ? String(pair.row_b[key]) : '—';
                const isExact = valA !== '—' && valB !== '—' && valA.toLowerCase() === valB.toLowerCase();

                return (
                  <tr key={key} className="hover:bg-elevated/30 transition-colors">
                    <td className="bg-deep/60 p-3.5 align-top font-medium capitalize text-text-secondary">
                      {key.replace(/_/g, ' ')}
                    </td>
                    <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                      {valA}
                    </td>
                    <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                      {valB}
                    </td>
                    <td className="border-l border-border/50 p-3.5 align-top">
                      {isExact ? (
                        <span className="flex items-center justify-center gap-1 rounded border border-success/30 bg-success/10 px-2 py-0.5 text-center font-mono text-[11px] font-medium text-success">
                          <CheckCircle2 className="h-3 w-3" /> Exact Match
                        </span>
                      ) : (
                        <span className="block text-center font-mono text-[11px] text-text-muted">
                          Differs
                        </span>
                      )}
                    </td>
                  </tr>
                );
              });
            })()}

            {/* Final Multi-Gate Verdict Row */}
            <tr className="border-t-2 border-border bg-deep font-medium">
              <td className="bg-elevated/40 p-4 text-text-primary">Full Multi-Gate Verdict</td>
              <td colSpan={2} className="border-l border-border/50 p-4 text-text-primary">
                {pair.flags.is_similar_all ? (
                  <span className="flex items-center gap-2 text-success">
                    <CheckCircle2 className="h-4 w-4" /> Candidate Duplicate — text + spatial + temporal convergence
                  </span>
                ) : pair.flags.is_similar_both ? (
                  <span className="flex items-center gap-2 text-warning">
                    <Zap className="h-4 w-4" /> Strong spatial + temporal correlation (distinct narrative)
                  </span>
                ) : (
                  <span className="text-text-secondary">
                    Partial match — classified as {pair.bin.replace(/_/g, ' ')}
                  </span>
                )}
              </td>
              <td className="border-l border-border/50 p-4 text-center">
                <span className={`rounded-full border px-3 py-1 font-mono text-xs font-medium ${
                  pair.flags.is_similar_all ? 'border-success/40 bg-success/20 text-success' : 'border-border bg-elevated text-text-secondary'
                }`}>
                  {pair.flags.is_similar_all ? 'DUPLICATE' : 'DISTINCT'}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Stage A audit-trail sort priority: Full Convergence > Date+Location >
// Text+Location > Text+Date > Text-only > everything else. Checked in this
// order rather than independently, since the flags overlap — is_similar_all
// already implies is_similar_both, so it has to be checked first to keep
// the buckets mutually exclusive.
function gateSortRank(p: CrossDbPair): number {
  if (p.flags.is_similar_all) return 0;
  if (p.flags.is_similar_both) return 1;
  if (p.flags.is_similar_text && p.flags.is_similar_location) return 2;
  if (p.flags.is_similar_text_date) return 3;
  if (p.flags.is_similar_text) return 4;
  return 5;
}

// Cluster Preview gradient: blue (just barely over the clustering threshold
// — likely a distinct witness's independent account of the same event) ->
// green -> red (near-verbatim text — likely the same document re-filed).
// `floor` anchors the cool end of the scale to whatever threshold this run
// actually used, so the gradient always spans the full range of similarity
// scores that could show up in a cluster, not just a fixed slice of it.
function canonicalSimilarityHue(sim: number, floor: number): number {
  const t = Math.max(0, Math.min(1, (sim - floor) / Math.max(1 - floor, 0.01)));
  return t < 0.5 ? 220 - 160 * (t / 0.5) : 140 - 280 * (t - 0.5);
}

function canonicalSimilarityColors(sim: number, floor: number): { text: string; bg: string; border: string } {
  const hue = canonicalSimilarityHue(sim, floor);
  return {
    text: `hsl(${hue}, 75%, 60%)`,
    bg: `hsla(${hue}, 75%, 55%, 0.16)`,
    border: `hsla(${hue}, 75%, 55%, 0.45)`,
  };
}

// Free-text craft color descriptions ("aluminum or eggshell white; shiny
// chromium-like") often name several colors in one cell, so each matched
// word/phrase is individually highlighted in place rather than tinting the
// whole cell one color — a cluster's color fields then read as a strip of
// inline swatches you can eyeball for agreement across members, without
// losing which word said what.
const COLOR_WORD_PATTERNS: [RegExp, string][] = [
  [/\b(?:chrome|chromium|aluminum|aluminium|silver(?:ish)?|metallic|gray|grey)\b/gi, '#9ca3af'],
  [/\b(?:white(?:ish)?|eggshell|pearl)\b/gi, '#e5e7eb'],
  [/\b(?:black(?:ish)?|dark)\b/gi, '#71717a'],
  [/\b(?:gold(?:en)?|brass|amber)\b/gi, '#eab308'],
  [/\b(?:yellow(?:ish)?)\b/gi, '#facc15'],
  [/\b(?:orange(?:ish)?)\b/gi, '#f97316'],
  [/\b(?:red(?:dish)?|crimson|scarlet)\b/gi, '#ef4444'],
  [/\b(?:pink(?:ish)?|magenta)\b/gi, '#ec4899'],
  [/\b(?:purple|violet|indigo|lavender)\b/gi, '#a855f7'],
  [/\b(?:blue(?:ish)?|azure|cyan)\b/gi, '#3b82f6'],
  [/\b(?:green(?:ish)?|emerald)\b/gi, '#22c55e'],
  [/\b(?:brown(?:ish)?|tan|bronze|copper|rust)\b/gi, '#a16207'],
];

// Finds every color-word match across all patterns and returns the text
// with each match wrapped in its own colored highlight, leaving everything
// else untouched. Overlapping matches keep whichever started first (ties
// broken by the longer match), so a phrase like "eggshell white" doesn't
// get double-highlighted for both "eggshell" and "white".
function highlightColorWords(text: string): ReactNode {
  const matches: { start: number; end: number; color: string }[] = [];
  for (const [re, color] of COLOR_WORD_PATTERNS) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) {
      matches.push({ start: m.index, end: m.index + m[0].length, color });
    }
  }
  if (matches.length === 0) return text;
  matches.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));

  const kept: typeof matches = [];
  let cursor = 0;
  for (const m of matches) {
    if (m.start < cursor) continue;
    kept.push(m);
    cursor = m.end;
  }

  const nodes: ReactNode[] = [];
  let last = 0;
  kept.forEach((m, i) => {
    if (m.start > last) nodes.push(text.slice(last, m.start));
    nodes.push(
      <span
        key={i}
        style={{ color: m.color, backgroundColor: `${m.color}2a`, borderRadius: '3px', padding: '0 2px' }}
      >
        {text.slice(m.start, m.end)}
      </span>,
    );
    last = m.end;
  });
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

const COLOR_FIELD_RE = /colou?r/i;

// Shared table + expandable member-preview for a set of clusters — used for
// both Stage B's strict (text+date+location) clusters and the independent
// text-only exact-duplicate pass, so the gradient/color-swatch preview
// behaves identically in both.
function ClusterTable({
  clusters,
  selectedId,
  onSelectId,
  simFloor,
  emptyMessage,
}: {
  clusters: DedupCluster[];
  selectedId: string | null;
  onSelectId: (id: string | null) => void;
  simFloor: number;
  emptyMessage: string;
}) {
  const selected = selectedId ? clusters.find((cl) => cl.cluster_id === selectedId) : null;
  // A pooled multi-database run can form hundreds of clusters — paginated
  // the same way as the Stage A pairs table, for the same reason (an
  // unvirtualized table with enough rows freezes the tab).
  const CLUSTER_PAGE_SIZE = 100;
  const [page, setPage] = useState(0);
  // Member column order in the expanded preview below — canonical proximity
  // (default backend order, roughly id-sorted), similarity to canonical
  // descending, or chronological by date.
  const [memberSort, setMemberSort] = useState<'default' | 'similarity' | 'date'>('default');
  const totalPages = Math.max(1, Math.ceil(clusters.length / CLUSTER_PAGE_SIZE));
  const clampedPage = Math.min(page, totalPages - 1);
  const pagedClusters = clusters.slice(clampedPage * CLUSTER_PAGE_SIZE, (clampedPage + 1) * CLUSTER_PAGE_SIZE);
  return (
    <>
      {clusters.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b border-border bg-deep font-mono text-text-secondary">
                <th className="p-3">Cluster ID</th>
                <th className="p-3">Size</th>
                <th className="p-3">Canonical Row ID</th>
                <th className="p-3">Member IDs</th>
                <th className="p-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40 text-text-primary">
              {pagedClusters.map((c) => (
                <tr key={c.cluster_id} className="hover:bg-elevated/30 transition-colors">
                  <td className="p-3 font-mono font-medium text-accent">{c.cluster_id}</td>
                  <td className="p-3 font-mono">{c.size}</td>
                  <td className="p-3 font-mono text-success">{c.canonical_id}</td>
                  <td className="p-3 font-mono text-[11px] text-text-secondary">{c.member_ids.join(', ')}</td>
                  <td className="p-3">
                    <button
                      onClick={() => onSelectId(selectedId === c.cluster_id ? null : c.cluster_id)}
                      className="rounded border border-border px-2 py-1 text-[11px] text-text-muted transition-colors hover:border-accent hover:text-accent"
                    >
                      {selectedId === c.cluster_id ? 'Hide' : 'Preview'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {clusters.length > CLUSTER_PAGE_SIZE && (
            <div className="flex items-center justify-between border-t border-border px-3 py-2 text-xs text-text-muted">
              <span>
                Showing {clampedPage * CLUSTER_PAGE_SIZE + 1}-{Math.min((clampedPage + 1) * CLUSTER_PAGE_SIZE, clusters.length)} of {clusters.length}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={clampedPage === 0}
                  className="rounded border border-border px-2 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
                >
                  Prev
                </button>
                <span className="font-mono">{clampedPage + 1} / {totalPages}</span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={clampedPage >= totalPages - 1}
                  className="rounded border border-border px-2 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <p className="p-4 text-xs text-text-muted">{emptyMessage}</p>
      )}

      {selected && (() => {
        const c = selected;
        // Union preview_cols with whatever keys actually showed up per
        // member — picks up quick-glance fields (craft color/shape) the
        // backend opportunistically attaches even when they weren't part
        // of the selected embedding columns, and falls back entirely to
        // observed keys when the run had no explicit column selection
        // (preview_cols empty).
        const cols = Array.from(new Set([
          ...c.preview_cols,
          ...c.member_previews.flatMap((m) => Object.keys(m.values)),
        ]));
        const hasDates = c.member_previews.some((m) => m.date);
        const sortedMembers = [...c.member_previews].sort((a, b) => {
          if (memberSort === 'similarity') {
            return (b.canonical_similarity ?? -1) - (a.canonical_similarity ?? -1);
          }
          if (memberSort === 'date') {
            if (!a.date && !b.date) return 0;
            if (!a.date) return 1;
            if (!b.date) return -1;
            return a.date.localeCompare(b.date);
          }
          return 0;
        });
        return (
          <div className="border-t border-border bg-deep p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-medium text-text-secondary">
                {c.cluster_id} — {c.member_previews.length} members, selected columns
              </span>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1 rounded-md border border-border p-0.5 text-[10px]">
                  {([
                    ['default', 'Default'],
                    ['similarity', 'Similarity'],
                    ['date', 'Chronological'],
                  ] as const).map(([key, label]) => (
                    <button
                      key={key}
                      onClick={() => setMemberSort(key)}
                      disabled={key === 'date' && !hasDates}
                      title={key === 'date' && !hasDates ? 'No resolved dates on this cluster’s members' : undefined}
                      className={`rounded px-2 py-1 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-30 ${
                        memberSort === key ? 'bg-accent-dim/30 text-accent-bright' : 'text-text-muted hover:text-text-secondary'
                      }`}
                    >
                      Sort: {label}
                    </button>
                  ))}
                </div>
                <span className="flex items-center gap-1.5 text-[10px] text-text-muted">
                  Proximity to canonical:
                  <span style={{ color: canonicalSimilarityColors(simFloor, simFloor).text }}>low</span>
                  →
                  <span style={{ color: canonicalSimilarityColors(1, simFloor).text }}>near-identical</span>
                </span>
              </div>
            </div>
            <div className="mt-2 overflow-x-auto rounded-md border border-border">
              <table className="w-full border-collapse text-left text-[11px]">
                <thead>
                  <tr className="border-b border-border bg-abyss font-mono text-text-secondary">
                    <th className="p-2">Column</th>
                    {sortedMembers.map((m) => {
                      const sim = m.canonical_similarity;
                      const simColors = sim != null ? canonicalSimilarityColors(sim, simFloor) : null;
                      return (
                        <th
                          key={m.id}
                          className={`border-l border-border/50 p-2 ${m.id === c.canonical_id ? 'text-accent' : 'text-text-primary'}`}
                          style={simColors ? { boxShadow: `inset 0 -3px 0 0 ${simColors.border}` } : undefined}
                        >
                          <div className="flex flex-col items-start gap-0.5">
                            <span>{m.id}{m.id === c.canonical_id ? ' (canonical)' : ''}</span>
                            {sim != null && !Number.isNaN(sim) && (
                              <span
                                className="rounded-full px-1.5 py-0.5 font-mono text-[10px] font-normal"
                                style={{ color: simColors!.text, backgroundColor: simColors!.bg }}
                                title="Cosine similarity to the cluster's canonical row"
                              >
                                {Math.round(sim * 100)}%
                              </span>
                            )}
                            {m.date && <span className="font-normal text-text-muted">{m.date}</span>}
                          </div>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40 text-text-primary">
                  {cols.map((col) => {
                    const isColorField = COLOR_FIELD_RE.test(col);
                    return (
                      <tr key={col} className="hover:bg-elevated/30 transition-colors">
                        <td className="bg-abyss/60 p-2 font-mono font-medium text-text-secondary">{col}</td>
                        {sortedMembers.map((m) => {
                          const raw = m.values[col];
                          const empty = raw == null || raw === '';
                          return (
                            <td key={m.id} className="border-l border-border/50 p-2 align-top">
                              {empty ? (
                                '—'
                              ) : isColorField && typeof raw === 'string' ? (
                                highlightColorWords(raw)
                              ) : (
                                String(raw)
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}
    </>
  );
}

// Column-name guesses shared by every dataset's field-mapping UI (Dataset A,
// inline in the component below; Dataset B, via useFieldMapping) — pulled to
// module scope so both can reference the same guess list without duplicating
// it. Auto-detected defaults keep a dataset that already matches one of
// these working with zero clicks; anything else falls back to manual mapping.
const NARRATIVE_COL_HINTS = ['witness_notes', 'narrative', 'description', 'case_text.text', 'summary', 'text'];
const DATE_COL_HINTS = ['date_time', 'date', 'sighting_date', 'sightingDetails.date'];
const LAT_COL_HINTS = ['latitude', 'lat', 'sightingDetails.location.latitude'];
const LON_COL_HINTS = ['longitude', 'lon', 'lng', 'sightingDetails.location.longitude'];
const LOCATION_NAME_HINTS = ['location.name', 'sightingDetails.location.name', 'location_name', 'city', 'city_name', 'location.city'];
const STATE_COL_HINTS = ['location.state', 'location_state_norm', 'state', 'sightingDetails.location.state'];
const CITY_COL_HINTS = ['location.city', 'city', 'city_name', 'sightingDetails.location.city'];

// Mirrors the backend's QUICK_GLANCE_COLS (api/services/dedup_service.py) —
// the only original columns worth carrying through a pooled run's payload
// beyond the _dedup_* fields, since they feed the Cluster Preview's
// color-word highlighting. Everything else in a source row is dropped by
// normalizeForPool; see the comment there for why.
const POOL_QUICK_GLANCE_COLS = [
  'craft.colour', 'sightingDetails.uapCharacteristics.color', 'sightingDetails.craft.colour',
  'craft.primary_shape', 'sightingDetails.uapCharacteristics.shape', 'sightingDetails.craft.primary_shape',
];

// One dataset's embedding-column + date/location field mapping. Dataset A
// keeps its own individual useState hooks below (unchanged); the Database
// Pool (Cross-DB tab) needs a *variable-length* list of these (2-6 total
// datasets, added/removed at runtime), which rules out one useState per
// field per dataset — React hooks can't be called a variable number of
// times. So pool datasets carry their mapping as plain object state instead,
// guessed once synchronously on upload (see guessFieldMapping) rather than
// via a per-dataset effect.
interface FieldMapping {
  embedCols: string[];
  embedColFilter: string;
  dateCol: string;
  locationMode: 'coords' | 'name';
  latCol: string;
  lonCol: string;
  locationCol: string;
  stateCol: string;
  cityCol: string;
}

interface PoolDataset extends FieldMapping {
  id: string;
  name: string;
  rows: any[];
  columns: string[];
}

// Max total datasets in a Database Pool comparison, including Dataset A.
const MAX_POOL_DATASETS = 6;

function guessFieldMapping(columns: string[]): FieldMapping {
  const guessedEmbed = NARRATIVE_COL_HINTS.filter((c) => columns.includes(c));
  const dateCol = DATE_COL_HINTS.find((c) => columns.includes(c)) ?? '';
  const latCol = LAT_COL_HINTS.find((c) => columns.includes(c)) ?? '';
  const lonCol = LON_COL_HINTS.find((c) => columns.includes(c)) ?? '';
  const locationCol = LOCATION_NAME_HINTS.find((c) => columns.includes(c)) ?? '';
  const stateCol = STATE_COL_HINTS.find((c) => columns.includes(c)) ?? '';
  const cityCol = CITY_COL_HINTS.find((c) => columns.includes(c)) ?? '';
  return {
    embedCols: guessedEmbed.length ? guessedEmbed : (columns[0] ? [columns[0]] : []),
    embedColFilter: '',
    dateCol,
    locationMode: latCol && lonCol ? 'coords' : locationCol ? 'name' : 'coords',
    latCol,
    lonCol,
    locationCol,
    stateCol,
    cityCol,
  };
}

export function DedupPage() {
  const { data, dataLoaded, setData, setPage } = useStore();
  const [activeTab, setActiveTab] = useState<'advanced' | 'cross_db'>('advanced');

  // Embedding column selection — shared by the Advanced (Batch Cluster) and
  // Cross-DB tabs, mirroring the Streamlit page-level column picker.
  const [embedCols, setEmbedCols] = useState<string[]>([]);
  const [embedColFilter, setEmbedColFilter] = useState('');

  useEffect(() => {
    if (dataLoaded && data?.columns?.length) {
      const guessed = NARRATIVE_COL_HINTS.filter((c) => data.columns.includes(c));
      setEmbedCols(guessed.length ? guessed : [data.columns[0]]);
    }
  }, [dataLoaded, data]);

  const toggleEmbedCol = (col: string) => {
    setEmbedCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  };

  const activeEmbedCols = embedCols.length > 0 ? embedCols : ['witness_notes'];

  // Field mapping (date / location) — same reasoning as the embedding column
  // picker above: the pipelines used to look for fixed column names
  // (date_time/date, latitude/lat, longitude/lon) and silently never fire
  // the date/location gates if a dataset used different names. Auto-detected
  // defaults below keep existing datasets working with zero clicks; leaving
  // these unset (a dataset that matches no hint) falls back to the
  // backend's own default-name guessing. (Hint lists are module-level —
  // see guessFieldMapping above, which the Database Pool uses directly.)
  const [dateCol, setDateCol] = useState<string>('');
  const [locationMode, setLocationMode] = useState<'coords' | 'name'>('coords');
  const [latCol, setLatCol] = useState<string>('');
  const [lonCol, setLonCol] = useState<string>('');
  const [locationCol, setLocationCol] = useState<string>('');
  const [stateCol, setStateCol] = useState<string>('');
  const [cityCol, setCityCol] = useState<string>('');

  useEffect(() => {
    if (dataLoaded && data?.columns?.length) {
      const guessedDate = DATE_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedLat = LAT_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedLon = LON_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedLoc = LOCATION_NAME_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedState = STATE_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedCity = CITY_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      setDateCol(guessedDate);
      setLatCol(guessedLat);
      setLonCol(guessedLon);
      setLocationCol(guessedLoc);
      setStateCol(guessedState);
      setCityCol(guessedCity);
      setLocationMode(guessedLat && guessedLon ? 'coords' : guessedLoc ? 'name' : 'coords');
    }
  }, [dataLoaded, data]);

  const activeDateCol = dateCol ? dateCol : undefined;
  const activeLatCol = locationMode === 'coords' && latCol ? latCol : undefined;
  const activeLonCol = locationMode === 'coords' && lonCol ? lonCol : undefined;
  const activeLocationCol = locationMode === 'name' && locationCol ? locationCol : undefined;
  const activeStateCol = locationMode === 'name' && stateCol ? stateCol : undefined;
  const activeCityCol = locationMode === 'name' && cityCol ? cityCol : undefined;



  const [error, setError] = useState<string | null>(null);

  // Advanced Tab States
  const [threshold, setThreshold] = useState(0.80);
  const [dateDiffDays, setDateDiffDays] = useState(3);
  const [maxKm, setMaxKm] = useState(50.0);
  // Shared by both Batch Cluster and Cross-DB: cluster-block the pairwise
  // search (UMAP+HDBSCAN over row embeddings) instead of a dense N x N
  // similarity matrix. Off by default — only worth it once a dataset is
  // large enough that the full matrix stops being cheap (tens of thousands
  // of rows); see the block_stats note next to the toggle.
  const [useClusterBlocking, setUseClusterBlocking] = useState(false);
  const [blockMinClusterSize, setBlockMinClusterSize] = useState(15);
  // Opt-in like cluster-blocking: off by default because the gazetteer is
  // US-only and a bare place name can collide across states/countries.
  // When off, the location gate falls straight to text-similarity on the
  // Location Name Column instead of resolving a real distance.
  const [useGazetteer, setUseGazetteer] = useState(false);
  const [advResult, setAdvResult] = useState<AdvancedDedupResponse | null>(null);
  // The exact record set the most recent run actually clustered — a plain
  // single-dataset run's buildDedupRecords() output, or a pool run's tagged/
  // normalized poolRecords. Export/Apply must reuse this rather than
  // recomputing buildDedupRecords() fresh: after a pool run, the clusters'
  // member_ids are "SourceName::id"-prefixed pool records, not Dataset A's
  // plain ids, so re-deriving from Dataset A alone would both drop the other
  // 3 databases' rows *and* fail to match any Dataset A row to its cluster.
  const [lastRunRecords, setLastRunRecords] = useState<any[] | null>(null);
  const [advLoading, setAdvLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [applyLoading, setApplyLoading] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [canonicalDistancePoints, setCanonicalDistancePoints] = useState<CanonicalDistancePoint[] | null>(null);
  const [canonicalDistancesLoading, setCanonicalDistancesLoading] = useState(false);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [selectedTextClusterId, setSelectedTextClusterId] = useState<string | null>(null);
  const [selectedAuditPairIndex, setSelectedAuditPairIndex] = useState<number>(0);
  const [auditViewMode, setAuditViewMode] = useState<'table' | 'preview'>('table');
  // A pool run over tens of thousands of rows can evaluate tens of thousands
  // of candidate pairs — rendering all of them as <tr> elements in one
  // unvirtualized table is what froze the tab, so the table is paginated
  // client-side. Filter/bin counts still come from `summary` (computed over
  // the full result), so they stay accurate regardless of which page is shown.
  const [auditPage, setAuditPage] = useState(0);
  const AUDIT_PAGE_SIZE = 100;
  // Same bin/gate filter pattern as the Cross-DB tab, applied to Stage A's
  // own pairs array.
  const [auditBinFilter, setAuditBinFilter] = useState<string>('All');
  const [auditGateFilter, setAuditGateFilter] = useState<string>('show_all');

  // Database Pool (Cross-DB tab) — Dataset A (the main loaded dataset above)
  // plus up to 5 more independently-uploaded datasets, each with its own
  // field mapping. "Execute Pipeline" normalizes every dataset's rows into
  // a common shape, tags each with its source name, pools them all into one
  // record list, and runs it through the exact same engine as the Batch
  // Cluster tab (run_advanced_dedup) — a cluster can end up with members
  // from 2, 3, or more different source databases. This reuses the
  // union-find/candidate-generation/Stage A+B UI wholesale instead of a
  // separate N-way comparison engine; results land in `advResult` and the
  // view switches to the Batch Cluster tab to show them.
  const [poolDatasets, setPoolDatasets] = useState<PoolDataset[]>([]);
  const [expandedPoolId, setExpandedPoolId] = useState<string | null>(null);
  const [poolUploadLoading, setPoolUploadLoading] = useState(false);
  const [poolUploadError, setPoolUploadError] = useState<string | null>(null);
  const [poolSourceSummary, setPoolSourceSummary] = useState<{ name: string; rows: number }[] | null>(null);
  const [crossLoading, setCrossLoading] = useState(false);

  // Loaded via the stateless parse endpoint (not /api/data/upload) so it
  // never clobbers the Data Explorer's active dataset (Dataset A).
  const handleAddPoolDataset = async (file: File) => {
    if (1 + poolDatasets.length >= MAX_POOL_DATASETS) return;
    setPoolUploadLoading(true);
    setPoolUploadError(null);
    try {
      const res = await api.parseDatasetFile(file);
      const mapping = guessFieldMapping(res.data.columns);
      const id = `${file.name}-${Date.now()}`;
      setPoolDatasets((prev) => [...prev, { id, name: file.name, rows: res.data.rows, columns: res.data.columns, ...mapping }]);
      setExpandedPoolId(id);
    } catch (err: any) {
      setPoolUploadError(err.message || 'Failed to load dataset.');
    } finally {
      setPoolUploadLoading(false);
    }
  };

  const updatePoolDataset = (id: string, patch: Partial<FieldMapping>) => {
    setPoolDatasets((prev) => prev.map((d) => (d.id === id ? { ...d, ...patch } : d)));
  };

  const togglePoolEmbedCol = (id: string, col: string) => {
    setPoolDatasets((prev) => prev.map((d) => (
      d.id === id ? { ...d, embedCols: d.embedCols.includes(col) ? d.embedCols.filter((c) => c !== col) : [...d.embedCols, col] } : d
    )));
  };

  const removePoolDataset = (id: string) => {
    setPoolDatasets((prev) => prev.filter((d) => d.id !== id));
  };

  // Missing date/lat/lon must stay missing, not fall back to a fake value —
  // '2000-01-01' or (0,0) would make every undated/unlocated row look
  // identical to every other one, silently passing the date/location
  // similarity gates for pairs with no real evidence. Shared by Dataset A
  // (the main loaded dataset) and every Database Pool dataset so they all
  // get identical id/date/lat/lon normalization.
  const normalizeRecords = (rows: any[]): any[] => {
    return rows.map((row: any, idx: number) => {
      const id = row.id || row.locus_tag || row.case_id || `Case-${idx + 1}`;
      const dt = row.date_time || row.date || row.datetime;
      const rawLat = row.latitude ?? row.lat;
      const rawLon = row.longitude ?? row.lon ?? row.lng;
      const lat = rawLat != null && rawLat !== '' ? Number(rawLat) : null;
      const lon = rawLon != null && rawLon !== '' ? Number(rawLon) : null;
      return {
        ...row,
        id: String(id),
        date_time: dt ? String(dt) : '',
        latitude: lat != null && !Number.isNaN(lat) ? lat : null,
        longitude: lon != null && !Number.isNaN(lon) ? lon : null,
      };
    });
  };

  // Builds the active-dataset record list used by the Advanced (Batch
  // Cluster) tab — the loaded dataset, mapped so id/date/lat/lon are
  // normalized (narrative text comes from whichever columns are selected
  // via activeEmbedCols, so we don't synthesize a text field here). Every
  // loaded row is sent — no implicit truncation (matches run_advanced_dedup's
  // "no cap unless asked" default) — so for large datasets, turn on
  // Cluster-Blocked Comparison above instead of silently comparing fewer
  // rows than loaded.
  const buildDedupRecords = (): any[] => {
    if (!dataLoaded || !data?.rows) return [];
    return normalizeRecords(data.rows);
  };

  // Normalizes one Database Pool dataset's rows into a common shape the
  // backend already knows how to consume regardless of the original column
  // names: one concatenated narrative field, one date field, and either
  // coordinates or a location name — exactly the tiers _resolve_effective_
  // coord already falls through per row, so rows from a lat/lon dataset and
  // rows from a location-name dataset can be pooled and compared together
  // with zero backend changes. IDs are prefixed with the source name so two
  // datasets that happen to reuse the same row id (e.g. both "Case-1") don't
  // collide in the union-find.
  const normalizeForPool = (rows: any[], mapping: FieldMapping, sourceName: string): any[] => {
    const cols = mapping.embedCols.length ? mapping.embedCols : Object.keys(rows[0] ?? {}).slice(0, 1);
    return rows.map((row: any, idx: number) => {
      const text = cols
        .map((c) => row[c])
        .filter((v) => v != null && String(v).trim() !== '')
        .join(' - ');
      const dateVal = mapping.dateCol ? row[mapping.dateCol] : undefined;
      const latVal = mapping.locationMode === 'coords' && mapping.latCol ? row[mapping.latCol] : undefined;
      const lonVal = mapping.locationMode === 'coords' && mapping.lonCol ? row[mapping.lonCol] : undefined;
      const locVal = mapping.locationMode === 'name' && mapping.locationCol ? row[mapping.locationCol] : undefined;
      const cityVal = mapping.locationMode === 'name' && mapping.cityCol ? row[mapping.cityCol] : undefined;
      const stateVal = mapping.locationMode === 'name' && mapping.stateCol ? row[mapping.stateCol] : undefined;
      const lat = latVal != null && latVal !== '' ? Number(latVal) : null;
      const lon = lonVal != null && lonVal !== '' ? Number(lonVal) : null;
      const rowId = row.id || row.locus_tag || row.case_id || `ROW-${idx + 1}`;
      // Deliberately NOT spreading ...row here — a source database can carry
      // hundreds of columns that are irrelevant to clustering, and spreading
      // all of them onto every pooled row (across every dataset, all sent in
      // one request) is what froze the tab at 21k rows x 4 x 300 cols. Only
      // what the backend actually uses (the _dedup_* fields) plus a couple
      // of preview-only extras ride along.
      const out: Record<string, unknown> = {
        id: `${sourceName}::${String(rowId)}`,
        _source_dataset: sourceName,
        _dedup_text: text || 'Unknown Report',
        _dedup_date: dateVal != null ? String(dateVal) : '',
        _dedup_lat: lat != null && !Number.isNaN(lat) ? lat : null,
        _dedup_lon: lon != null && !Number.isNaN(lon) ? lon : null,
        _dedup_location_name: locVal != null ? String(locVal) : '',
        _dedup_city: cityVal != null ? String(cityVal) : '',
        _dedup_state: stateVal != null ? String(stateVal) : '',
      };
      for (const qc of POOL_QUICK_GLANCE_COLS) {
        if (row[qc] != null && row[qc] !== '') out[qc] = row[qc];
      }
      return out;
    });
  };

  // Past roughly 5-10k pooled rows, the dense similarity matrix the pipeline
  // builds without Cluster-Blocked Comparison gets expensive fast (it's
  // O(rows^2) — 21k rows is ~440M matrix cells) — pooling multiple databases
  // is exactly the scenario that crosses this threshold, so it's worth
  // flagging even though the toggle itself lives in the shared panel above.
  const totalPoolRows = (data?.rows?.length ?? 0) + poolDatasets.reduce((n, d) => n + d.rows.length, 0);
  const POOL_BLOCKING_RECOMMENDED_AT = 5000;

  const handleRunPooledDedup = async () => {
    if (!dataLoaded || !data?.rows) {
      setError('Load Dataset A from the Dashboard or Data Explorer first.');
      return;
    }
    if (poolDatasets.length === 0) {
      setError('Add at least one more database below before running the pool comparison.');
      return;
    }
    setCrossLoading(true);
    setError(null);
    setCanonicalDistancePoints(null);
    try {
      const datasetAMapping: FieldMapping = {
        embedCols: activeEmbedCols, embedColFilter: '', dateCol, locationMode, latCol, lonCol, locationCol, stateCol, cityCol,
      };
      const sources = [{ name: 'Dataset A', rows: data.rows, mapping: datasetAMapping }, ...poolDatasets.map((d) => ({ name: d.name, rows: d.rows, mapping: d }))];
      const poolRecords = sources.flatMap((s) => normalizeForPool(s.rows, s.mapping, s.name));
      setLastRunRecords(poolRecords);

      const dataRes = await api.runAdvancedDedup({
        records: poolRecords,
        cols: ['_dedup_text'],
        threshold,
        date_diff_days: dateDiffDays,
        max_km: maxKm,
        use_llm_judge: false,
        date_col: '_dedup_date',
        lat_col: '_dedup_lat',
        lon_col: '_dedup_lon',
        location_col: '_dedup_location_name',
        state_col: '_dedup_state',
        city_col: '_dedup_city',
        use_gazetteer: useGazetteer,
        use_cluster_blocking: useClusterBlocking,
        block_min_cluster_size: blockMinClusterSize,
      });
      setAdvResult(dataRes as AdvancedDedupResponse);
      setPoolSourceSummary(sources.map((s) => ({ name: s.name, rows: s.rows.length })));
      setActiveTab('advanced');
    } catch (err: any) {
      setError(err.message || 'Database Pool comparison failed.');
    } finally {
      setCrossLoading(false);
    }
  };

  // API Call: Run Advanced Dedup (live multi-gate screening + union-find clustering)
  const handleRunAdvanced = async () => {
    setAdvLoading(true);
    setError(null);
    setPoolSourceSummary(null);
    setCanonicalDistancePoints(null);
    try {
      const records = buildDedupRecords();
      setLastRunRecords(records);
      const dataRes = await api.runAdvancedDedup({
        records,
        cols: activeEmbedCols,
        threshold,
        date_diff_days: dateDiffDays,
        max_km: maxKm,
        use_llm_judge: false,
        date_col: activeDateCol,
        lat_col: activeLatCol,
        lon_col: activeLonCol,
        location_col: activeLocationCol,
        state_col: activeStateCol,
        city_col: activeCityCol,
        use_gazetteer: useGazetteer,
        use_cluster_blocking: useClusterBlocking,
        block_min_cluster_size: blockMinClusterSize,
      });
      setAdvResult(dataRes as AdvancedDedupResponse);
    } catch (err: any) {
      setError(err.message || 'Advanced deduplication failed.');
    } finally {
      setAdvLoading(false);
    }
  };

  // Downloads deduped_dataset.csv (dedup_cluster_id/size/is_canonical/
  // is_duplicate baked onto every row from lastRunRecords — the exact set
  // Stage A/B actually ran on, whether that's Dataset A alone or a pool of
  // several databases) + dedup_run_metadata.json, zipped — the metadata
  // records every parameter this run used, for reproducibility.
  // The column mapping the most recent run actually resolved dates/locations
  // with — pool runs always use the normalized _dedup_* fields, plain runs
  // use whatever Dataset A's Field Mapping panel selected. Export/Apply send
  // this so the backend can (a) write _resolved_lat/_resolved_lon (incl.
  // gazetteer centroids) onto every row and (b) build cluster_edges.csv's
  // chronological chains from the same date column the gates compared.
  const lastRunMapping = () => (poolSourceSummary ? {
    date_col: '_dedup_date',
    lat_col: '_dedup_lat',
    lon_col: '_dedup_lon',
    location_col: '_dedup_location_name',
    state_col: '_dedup_state',
    city_col: '_dedup_city',
    use_gazetteer: useGazetteer,
  } : {
    date_col: activeDateCol,
    lat_col: activeLatCol,
    lon_col: activeLonCol,
    location_col: activeLocationCol,
    state_col: activeStateCol,
    city_col: activeCityCol,
    use_gazetteer: useGazetteer,
  });

  const handleExportDedup = async () => {
    if (!advResult?.clusters || !lastRunRecords) return;
    setExportLoading(true);
    setError(null);
    try {
      const mapping = lastRunMapping();
      await api.exportDedupResults({
        records: lastRunRecords,
        clusters: advResult.clusters as unknown as Record<string, unknown>[],
        id_field: 'id',
        text_duplicate_clusters: advResult.text_duplicate_clusters as unknown as Record<string, unknown>[] | undefined,
        // Every Stage A pair the run evaluated, not just the filtered/paged
        // subset currently shown — so a pair like Case-1820 vs Case-1840 is
        // in the export even if it's hidden behind an active filter, and so
        // a later importDedupZip() can replicate the full run.
        pairs: advResult.pairs as unknown as Record<string, unknown>[] | undefined,
        summary: advResult.summary as unknown as Record<string, unknown> | undefined,
        ...mapping,
        parameters: {
          threshold,
          date_diff_days: dateDiffDays,
          max_km: maxKm,
          ...(poolSourceSummary
            ? { pooled_datasets: poolSourceSummary, embed_cols: ['_dedup_text'] }
            : { embed_cols: activeEmbedCols }),
          date_col: mapping.date_col ?? null,
          lat_col: mapping.lat_col ?? null,
          lon_col: mapping.lon_col ?? null,
          location_col: mapping.location_col ?? null,
          state_col: mapping.state_col ?? null,
          city_col: mapping.city_col ?? null,
          use_gazetteer: useGazetteer,
          use_cluster_blocking: useClusterBlocking,
          block_min_cluster_size: blockMinClusterSize,
          model_name: 'microsoft/harrier-oss-v1-270m',
          top_k: 5,
        },
      });
    } catch (err: any) {
      setError(err.message || 'Dedup export failed.');
    } finally {
      setExportLoading(false);
    }
  };

  // Same enrichment as the export, but loads the result straight into the
  // Data Explorer store instead of downloading a file — no download/
  // re-upload round trip needed to filter/sort on the new dedup_* columns.
  // For a pool run, this replaces Dataset A in the Data Explorer with the
  // full pooled set (all source databases, tagged with _source_dataset) —
  // that's the whole point of "Apply" after a pool comparison.
  const handleApplyDedup = async () => {
    if (!advResult?.clusters || !lastRunRecords) return;
    setApplyLoading(true);
    setError(null);
    try {
      // Apply doesn't build edges, so it takes the coord mapping without date_col.
      const { lat_col, lon_col, location_col, state_col, city_col, use_gazetteer } = lastRunMapping();
      const applyMapping = { lat_col, lon_col, location_col, state_col, city_col, use_gazetteer };
      const res = await api.applyDedupToDataset({
        records: lastRunRecords,
        clusters: advResult.clusters as unknown as Record<string, unknown>[],
        id_field: 'id',
        text_duplicate_clusters: advResult.text_duplicate_clusters as unknown as Record<string, unknown>[] | undefined,
        ...applyMapping,
      });
      setData(res.data, res.column_stats);
    } catch (err: any) {
      setError(err.message || 'Applying dedup results to the dataset failed.');
    } finally {
      setApplyLoading(false);
    }
  };

  // Restores a previously-exported dedup_export.zip — for replication (share
  // a run, or come back to it later) with no re-fetch of the source dataset
  // and no re-running the embedding pipeline. deduped_dataset.csv becomes
  // the active Data Explorer dataset (same as Apply); when the zip also has
  // dedup_result.json, Stage A/B fully repopulate from it too. Works with no
  // dataset loaded yet — the zip is self-contained.
  const handleImportZip = async (file: File) => {
    setImportLoading(true);
    setError(null);
    try {
      const res = await api.importDedupZip(file);
      setData(res.data, res.column_stats);
      setCanonicalDistancePoints(null);
      if (res.dedup_result) {
        const dr = res.dedup_result;
        setAdvResult({
          status: 'success',
          pairs: dr.pairs as unknown as CrossDbPair[],
          clusters: dr.clusters as unknown as DedupCluster[],
          text_duplicate_clusters: dr.text_duplicate_clusters as unknown as DedupCluster[],
          summary: dr.summary as unknown as AdvancedDedupResponse['summary'],
          parameters: dr.parameters as unknown as AdvancedDedupResponse['parameters'],
        });
        // The imported rows already carry the dedup_* columns the original
        // run baked on — fine to reuse as the record set for another
        // Apply/Export pass; a later Run Pipeline overwrites them anyway.
        setLastRunRecords(res.data.rows);
        const params: Record<string, unknown> = dr.parameters || {};
        if (typeof params.threshold === 'number') setThreshold(params.threshold);
        if (typeof params.date_diff_days === 'number') setDateDiffDays(params.date_diff_days);
        if (typeof params.max_km === 'number') setMaxKm(params.max_km);
        if (typeof params.use_gazetteer === 'boolean') setUseGazetteer(params.use_gazetteer);
        if (typeof params.use_cluster_blocking === 'boolean') setUseClusterBlocking(params.use_cluster_blocking);
        if (typeof params.block_min_cluster_size === 'number') setBlockMinClusterSize(params.block_min_cluster_size);
        if (Array.isArray(params.pooled_datasets)) {
          setPoolSourceSummary(params.pooled_datasets as { name: string; rows: number }[]);
        }
        setActiveTab('advanced');
      } else {
        setError(
          'Dataset restored, but this zip has no dedup_result.json (exported before Stage A pairs were included) — ' +
          'Stage A/B tables can\'t be replicated from it. Re-run the pipeline to regenerate them.',
        );
      }
    } catch (err: any) {
      setError(err.message || 'Importing the dedup zip failed.');
    } finally {
      setImportLoading(false);
    }
  };

  // Every non-canonical cluster member's temporal + haversine distance from
  // its cluster's canonical record, for the Stage B spatio-temporal spread
  // chart — computed on demand (not auto-fetched) since it's a second
  // gazetteer-aware coordinate resolution pass over every clustered row.
  const handleLoadCanonicalDistances = async () => {
    if (!advResult?.clusters || !lastRunRecords) return;
    setCanonicalDistancesLoading(true);
    setError(null);
    try {
      const mapping = lastRunMapping();
      const res = await api.getCanonicalDistances({
        records: lastRunRecords,
        clusters: advResult.clusters as unknown as Record<string, unknown>[],
        id_field: 'id',
        cols: poolSourceSummary ? ['_dedup_text'] : activeEmbedCols,
        ...mapping,
      });
      setCanonicalDistancePoints(res.points);
    } catch (err: any) {
      setError(err.message || 'Computing cluster spatio-temporal distances failed.');
    } finally {
      setCanonicalDistancesLoading(false);
    }
  };


  const tabs: { id: 'advanced' | 'cross_db'; label: string; icon: typeof Layers }[] = [
    { id: 'advanced', label: 'Batch Cluster Pipeline', icon: Layers },
    { id: 'cross_db', label: 'Cross-DB Pipeline', icon: Database },
  ];

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
        <div className="mt-3 flex items-center gap-2 border-t border-border/50 pt-3">
          <span className="text-sm text-text-muted">Or replicate a previous run —</span>
          <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border bg-raised px-3 py-1.5 text-xs text-text-secondary transition-colors hover:border-accent hover:text-accent">
            <Upload className="h-3.5 w-3.5" />
            {importLoading ? 'Loading…' : 'Load Previous Run (.zip)'}
            <input
              type="file"
              accept=".zip"
              className="hidden"
              disabled={importLoading}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleImportZip(file);
                e.target.value = '';
              }}
            />
          </label>
        </div>
        {error && (
          <p className="mt-2 flex items-center gap-2 text-xs text-danger">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> {error}
          </p>
        )}
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      {/* Tab bar — page identity ("Dedupe Studio") already shown in the TopBar */}
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

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* SHARED: Data Source + Embedding Column Selection — feeds both the
          Advanced (Batch Cluster) and Cross-DB pipelines */}
      {(activeTab === 'advanced' || activeTab === 'cross_db') && (
        <Panel title="Data Source & Embedding Columns" subtitle="Feeds the Batch Cluster and Cross-DB pipelines below">
          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <Database className="h-3.5 w-3.5 text-text-muted" />
            Active dataset: {data?.rows ? `${data.rows.length.toLocaleString()} rows` : '0 rows'}
          </div>

          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
                <Layers className="h-3.5 w-3.5" /> Embedding Columns (Semantic Representation)
              </span>
              <span className="font-mono text-[10px] text-text-muted">
                {activeEmbedCols.length} selected
              </span>
            </div>
            {data?.columns?.length ? (
              <>
                <input
                  type="text"
                  value={embedColFilter}
                  onChange={(e) => setEmbedColFilter(e.target.value)}
                  placeholder="Filter columns..."
                  className="w-full rounded-md border border-border bg-deep px-3 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                />
                <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto p-1">
                  {data.columns
                    .filter((c) => c.toLowerCase().includes(embedColFilter.toLowerCase()))
                    .map((c) => (
                      <button
                        key={c}
                        onClick={() => toggleEmbedCol(c)}
                        className={`rounded border px-2 py-1 font-mono text-[11px] transition-colors ${
                          embedCols.includes(c)
                            ? 'border-accent bg-accent-dim/30 text-accent-bright'
                            : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        {c}
                      </button>
                    ))}
                </div>
                <p className="text-[10px] text-text-muted">
                  Selected columns are concatenated per row to build the narrative text used for semantic similarity.
                </p>
              </>
            ) : (
              <p className="text-[11px] text-text-muted">
                Load a dataset to pick which columns feed the narrative embedding.
              </p>
            )}
          </div>

          {/* Field Mapping — the date/location gates used to look for fixed
              column names (date_time/date, latitude/lat, longitude/lon) and
              silently never fire if a dataset used different ones. These
              selectors point at the real columns; auto-detected defaults
              above keep already-matching datasets working with no clicks. */}
          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
            <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
              <Calendar className="h-3.5 w-3.5" /> Field Mapping (Date & Location)
            </span>
            {data?.columns?.length ? (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label className="text-[10px] text-text-muted">Date Column</label>
                  <select
                    value={dateCol}
                    onChange={(e) => setDateCol(e.target.value)}
                    className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                  >
                    <option value="">(none — no date gating)</option>
                    {data.columns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-[10px] text-text-muted">Location Field Type</label>
                  <div className="mt-1 flex gap-1.5">
                    <button
                      onClick={() => setLocationMode('coords')}
                      className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                        locationMode === 'coords' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      Lat/Lon
                    </button>
                    <button
                      onClick={() => setLocationMode('name')}
                      className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                        locationMode === 'name' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      Location Name
                    </button>
                  </div>
                </div>

                {locationMode === 'coords' ? (
                  <>
                    <div>
                      <label className="text-[10px] text-text-muted">Latitude Column</label>
                      <select
                        value={latCol}
                        onChange={(e) => setLatCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] text-text-muted">Longitude Column</label>
                      <select
                        value={lonCol}
                        onChange={(e) => setLonCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <label className="text-[10px] text-text-muted">
                        Location Name Column — free-text description, matched via fuzzy text, or
                        parsed for the gazetteer lookup below when no City Column is set
                      </label>
                      <select
                        value={locationCol}
                        onChange={(e) => setLocationCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] text-text-muted">
                        City Column (optional, preferred) — a clean, already-separated city name
                        is a far more reliable gazetteer key than one parsed out of a free-text
                        Location Name sentence; falls back to that column when unset
                      </label>
                      <select
                        value={cityCol}
                        onChange={(e) => setCityCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] text-text-muted">
                        State Column (optional) — pairs with City Column (preferred) or Location
                        Name Column for gazetteer matching; if unset, a combined "City, ST" in
                        Location Name is parsed instead
                      </label>
                      <select
                        value={stateCol}
                        onChange={(e) => setStateCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div className="sm:col-span-2 flex items-center justify-between rounded-md border border-border/50 bg-deep/50 px-3 py-2">
                      <span className="text-[10px] text-text-muted">
                        US Census Gazetteer lookup (city/state → real lat/lon centroid) — US-only,
                        and a bare place name can collide across states/countries, so it's opt-in
                        rather than the default; off falls back to fuzzy text match only.
                      </span>
                      <button
                        onClick={() => setUseGazetteer(!useGazetteer)}
                        className={`ml-3 shrink-0 rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
                          useGazetteer
                            ? 'border-accent bg-accent-dim/30 text-accent-bright'
                            : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        {useGazetteer ? 'ON' : 'OFF'}
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <p className="text-[11px] text-text-muted">
                Load a dataset to map its date/location columns.
              </p>
            )}
          </div>

          {/* Cluster-Blocked Comparison — the pairwise search below normally
              builds a dense N x N similarity matrix, which stops being
              feasible somewhere in the tens of thousands of rows. Turning
              this on pre-groups rows with UMAP+HDBSCAN (the same technique
              the Analysis tab's cluster pipeline uses) and only compares
              rows within the same group, trading a small recall risk near
              cluster boundaries for comparisons that scale with group size
              instead of dataset size. */}
          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
                <Layers className="h-3.5 w-3.5" /> Cluster-Blocked Comparison
              </span>
              <button
                onClick={() => setUseClusterBlocking(!useClusterBlocking)}
                className={`rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
                  useClusterBlocking
                    ? 'border-accent bg-accent-dim/30 text-accent-bright'
                    : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                }`}
              >
                {useClusterBlocking ? 'ON' : 'OFF'}
              </button>
            </div>
            <p className="text-[10px] text-text-muted">
              Pre-groups rows by embedding similarity (UMAP+HDBSCAN) and only compares rows within the
              same group, instead of every row against every other row. Recommended for datasets past
              roughly 20-30k rows, where the full comparison matrix gets expensive; leave off for smaller
              datasets since a true duplicate pair split across two groups near a boundary would be missed.
            </p>
            {useClusterBlocking && (
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-text-muted">Min group size</label>
                <input
                  type="number"
                  min={2}
                  max={500}
                  value={blockMinClusterSize}
                  onChange={(e) => setBlockMinClusterSize(Math.max(2, parseInt(e.target.value, 10) || 15))}
                  className="w-20 rounded-md border border-border bg-deep px-2 py-1 text-xs text-text-primary focus:border-accent focus:outline-none"
                />
              </div>
            )}
          </div>
        </Panel>
      )}

      {/* ADVANCED TAB */}
      {activeTab === 'advanced' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Batch Cluster Pipeline"
            subtitle="Stage A screens every candidate pair live; Stage B unions the full-convergence pairs into canonical clusters"
            actions={
              <div className="flex items-center gap-2">
                <label
                  title="Restore Stage A/B from a previously exported dedup_export.zip — no recomputation"
                  className="flex cursor-pointer items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent"
                >
                  <Upload className="h-3.5 w-3.5" />
                  {importLoading ? 'Loading…' : 'Load Previous Run (.zip)'}
                  <input
                    type="file"
                    accept=".zip"
                    className="hidden"
                    disabled={importLoading}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleImportZip(file);
                      e.target.value = '';
                    }}
                  />
                </label>
                <button
                  onClick={handleRunAdvanced}
                  disabled={advLoading}
                  className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
                >
                  <Play className="h-3.5 w-3.5" />
                  {advLoading ? 'Running…' : 'Run Pipeline'}
                </button>
              </div>
            }
          >
            <div className="grid grid-cols-1 items-center gap-6 sm:grid-cols-3">
              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Cosine Similarity Threshold</span>
                  <span className="font-mono text-accent">{threshold}</span>
                </label>
                <input
                  type="range"
                  min="0.50"
                  max="1.00"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Temporal Diff (Days)</span>
                  <span className="font-mono text-accent">{dateDiffDays} d</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="1"
                  value={dateDiffDays}
                  onChange={(e) => setDateDiffDays(parseInt(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Spatial Radius (Haversine)</span>
                  <span className="font-mono text-accent">{maxKm} km</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="200"
                  step="5"
                  value={maxKm}
                  onChange={(e) => setMaxKm(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>
            </div>

            <DedupeProgressBar loading={advLoading} title="Batch Clustering & Graph Screening" />
          </Panel>

          {/* Results Display */}
          {advResult && advResult.status !== 'success' && (
            <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{advResult.message || 'Advanced deduplication failed.'}</span>
            </div>
          )}
          {advResult && advResult.status === 'success' && (() => {
            const summary = advResult.summary;
            // Full Convergence first, then Date+Location, Text+Location,
            // Text+Date, Text-only, then everything else — similarity
            // descending within each bucket.
            const pairs = [...(advResult.pairs ?? [])].sort((a, b) => {
              const rankDiff = gateSortRank(a) - gateSortRank(b);
              return rankDiff !== 0 ? rankDiff : b.similarity - a.similarity;
            });
            const clusters = advResult.clusters ?? [];
            if (!summary) return null;
            const filteredPairs = pairs.filter((p) => {
              if (auditBinFilter !== 'All' && p.bin !== auditBinFilter) return false;
              if (auditGateFilter !== 'show_all') {
                if (auditGateFilter === 'similar_text' && !p.flags.is_similar_text) return false;
                if (auditGateFilter === 'similar_date' && !p.flags.is_similar_date) return false;
                if (auditGateFilter === 'similar_location' && !p.flags.is_similar_location) return false;
                if (auditGateFilter === 'similar_text_date' && !p.flags.is_similar_text_date) return false;
                if (auditGateFilter === 'similar_both' && !p.flags.is_similar_both) return false;
                if (auditGateFilter === 'similar_all' && !p.flags.is_similar_all) return false;
              }
              return true;
            });
            // Gate button counts scoped to the active Semantic Similarity Bin —
            // summary.gate_counts is a fixed global total, so with a bin
            // selected its numbers stop matching what pressing a gate button
            // would actually show (e.g. "Similar Location (10)" including
            // pairs outside the selected bin). Recomputed from `pairs` on
            // every render so the two filter panels always agree with each
            // other and with the table below.
            const binScopedPairs = auditBinFilter === 'All' ? pairs : pairs.filter((p) => p.bin === auditBinFilter);
            const liveGateCounts = {
              similar_text: binScopedPairs.filter((p) => p.flags.is_similar_text).length,
              similar_date: binScopedPairs.filter((p) => p.flags.is_similar_date).length,
              similar_location: binScopedPairs.filter((p) => p.flags.is_similar_location).length,
              similar_text_date: binScopedPairs.filter((p) => p.flags.is_similar_text_date).length,
              similar_both: binScopedPairs.filter((p) => p.flags.is_similar_both).length,
              similar_all: binScopedPairs.filter((p) => p.flags.is_similar_all).length,
            };
            const totalAuditPages = Math.max(1, Math.ceil(filteredPairs.length / AUDIT_PAGE_SIZE));
            const clampedAuditPage = Math.min(auditPage, totalAuditPages - 1);
            const pagedPairs = filteredPairs.slice(clampedAuditPage * AUDIT_PAGE_SIZE, (clampedAuditPage + 1) * AUDIT_PAGE_SIZE);
            return (
              <div className="flex flex-col gap-4">
                {poolSourceSummary && (
                  <p className="flex flex-wrap items-center gap-1.5 text-[11px] text-text-muted">
                    <Database className="h-3 w-3" /> Pooled from {poolSourceSummary.length} databases:{' '}
                    {poolSourceSummary.map((s, i) => (
                      <span key={s.name}>
                        {i > 0 && ', '}
                        <span className="text-text-secondary">{s.name}</span> ({s.rows.toLocaleString()})
                      </span>
                    ))}
                  </p>
                )}
                {/* KPI Cards */}
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                  <div className="flex flex-col rounded-md border border-border bg-raised p-3">
                    <span className="text-xs text-text-muted">Total Clusters Formed</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-accent">{summary.total_clusters}</span>
                    <span className="mt-1 text-[11px] text-text-muted">Union-find connected components</span>
                  </div>
                  <div className="flex flex-col rounded-md border border-border bg-raised p-3">
                    <span className="text-xs text-text-muted">Rows in Clusters</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{summary.rows_in_clusters}</span>
                    <span className="mt-1 text-[11px] text-text-muted">Total candidate reports linked</span>
                  </div>
                  <div className="flex flex-col rounded-md border border-success/30 bg-success/10 p-3">
                    <span className="text-xs text-success">Redundant Rows Saved</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-success">{summary.redundant_rows_saved}</span>
                    <span className="mt-1 text-[11px] text-success/80">Non-canonical duplicates pruned</span>
                  </div>
                  <div className="flex flex-col rounded-md border border-warning/30 bg-warning/10 p-3">
                    <span className="text-xs text-warning">Text-Exact Duplicates</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-warning">{summary.text_duplicate_redundant_rows_saved}</span>
                    <span className="mt-1 text-[11px] text-warning/80">
                      Sim ≥0.88, gates ignored — {summary.text_duplicate_clusters_count} groups
                    </span>
                  </div>
                  <div className="flex flex-col rounded-md border border-accent/30 bg-accent-dim/10 p-3">
                    <span className="text-xs text-accent">Combined Prunable</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-accent">{summary.combined_redundant_rows_saved}</span>
                    <span className="mt-1 text-[11px] text-accent/80">Drop these to dedupe by both criteria</span>
                  </div>
                </div>

                {summary.block_stats && (
                  <p className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <Layers className="h-3 w-3" /> Cluster-blocked: {summary.block_stats.block_count} groups cut the comparison
                    matrix from {summary.block_stats.full_matrix_cells.toLocaleString()} to{' '}
                    {summary.block_stats.compared_cells.toLocaleString()} cells
                    {summary.block_stats.noise_block_size > 0 && ` (${summary.block_stats.noise_block_size} rows ungrouped)`}.
                  </p>
                )}

                {/* Semantic Similarity Bins — cards double as filter toggles */}
                <Panel title="Semantic Similarity Bins" subtitle="Click a bin to filter Stage A below; click again to clear">
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    {([
                      ['exact_duplicate', 'Exact Duplicate (≥0.88)'],
                      ['strong_similar', 'Strong Similar (0.80-0.88)'],
                      ['moderate_similar', 'Moderate Similar (0.70-0.80)'],
                      ['distinct', 'Distinct (<0.70)'],
                    ] as const).map(([key, label]) => (
                      <div
                        key={key}
                        onClick={() => { setAuditBinFilter(auditBinFilter === key ? 'All' : key); setSelectedAuditPairIndex(0); }}
                        className={`flex cursor-pointer flex-col rounded-md border p-3 transition-colors ${
                          auditBinFilter === key
                            ? 'border-accent bg-accent-dim/30'
                            : 'border-border bg-raised hover:border-border-bright'
                        }`}
                      >
                        <span className="text-xs text-text-secondary">{label}</span>
                        <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{summary.bins[key]}</span>
                      </div>
                    ))}
                  </div>
                </Panel>

                {/* Interactive Multi-Gate Filter */}
                <Panel
                  title="Interactive Multi-Gate Filter"
                  subtitle={
                    auditBinFilter === 'All'
                      ? 'Select a filter to slice the Stage A pairs below'
                      : `Select a filter to slice the Stage A pairs below — counts scoped to the "${auditBinFilter.replace(/_/g, ' ')}" bin selected above`
                  }
                >
                  <div className="flex flex-wrap gap-2">
                    {([
                      ['similar_text', FileText, `Similar Text (${liveGateCounts.similar_text})`],
                      ['similar_date', Calendar, `Similar Date (${liveGateCounts.similar_date})`],
                      ['similar_location', MapPin, `Similar Location (${liveGateCounts.similar_location})`],
                      ['similar_text_date', Puzzle, `Text + Date (${liveGateCounts.similar_text_date})`],
                      ['similar_both', Zap, `Both (Spatial+Temporal) (${liveGateCounts.similar_both})`],
                      ['similar_all', Target, `Full Convergence (${liveGateCounts.similar_all})`],
                      ['show_all', Globe, `Show All (${binScopedPairs.length})`],
                    ] as const).map(([key, Icon, label]) => (
                      <button
                        key={key}
                        onClick={() => { setAuditGateFilter(key); setSelectedAuditPairIndex(0); }}
                        className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
                          auditGateFilter === key ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        <Icon className="h-3.5 w-3.5" /> {label}
                      </button>
                    ))}
                  </div>
                </Panel>

                {/* Stage A: Pairwise Audit Trail */}
                <Panel
                  title="Stage A — Multi-Gate Pairwise Audit Trail"
                  subtitle="Every candidate pair the screening engine evaluated, with each gate's individual verdict"
                  actions={
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-text-muted">{filteredPairs.length} / {pairs.length} pairs</span>
                      <div className="flex overflow-hidden rounded border border-border">
                        <button
                          onClick={() => setAuditViewMode('table')}
                          className={`px-2.5 py-1 text-[11px] font-medium transition-colors ${
                            auditViewMode === 'table' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                          }`}
                        >
                          Table
                        </button>
                        <button
                          onClick={() => setAuditViewMode('preview')}
                          className={`border-l border-border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                            auditViewMode === 'preview' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                          }`}
                        >
                          Preview
                        </button>
                      </div>
                    </div>
                  }
                  noPad
                >
                  {auditViewMode === 'table' ? (
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-xs">
                        <thead>
                          <tr className="border-b border-border bg-deep font-mono text-text-secondary">
                            <th className="p-3">ID A</th>
                            <th className="p-3">ID B</th>
                            <th className="p-3">Similarity</th>
                            <th className="p-3">Distance</th>
                            <th className="p-3">Date Diff</th>
                            <th className="p-3">Witness Notes Preview</th>
                            <th className="p-3">Text</th>
                            <th className="p-3">Date</th>
                            <th className="p-3">Location</th>
                            <th className="p-3">Full Convergence</th>
                            <th className="p-3"></th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40 text-text-primary">
                          {pagedPairs.length === 0 ? (
                            <tr>
                              <td colSpan={11} className="p-6 text-center text-xs text-text-muted">
                                No candidate pairs match the active filters.
                              </td>
                            </tr>
                          ) : pagedPairs.map((pair, localIdx) => {
                            const idx = clampedAuditPage * AUDIT_PAGE_SIZE + localIdx;
                            return (
                            <tr
                              key={idx}
                              onClick={() => { setSelectedAuditPairIndex(idx); setAuditViewMode('preview'); }}
                              className="cursor-pointer hover:bg-elevated/30 transition-colors"
                            >
                              <td className="p-3 font-mono font-medium text-accent">{pair.id_a}</td>
                              <td className="p-3 font-mono text-text-secondary">{pair.id_b}</td>
                              <td className="p-3 font-mono">
                                <span className="rounded border border-accent/40 bg-accent/10 px-2 py-0.5 text-accent">
                                  {(pair.similarity * 100).toFixed(1)}%
                                </span>
                              </td>
                              <td className="p-3 font-mono text-text-secondary">{pair.haversine_km != null ? `${pair.haversine_km} km` : '—'}</td>
                              <td className="p-3 font-mono text-text-secondary">{pair.date_diff_days != null ? `${pair.date_diff_days} d` : '—'}</td>
                              <td className="max-w-md p-3 text-[11px]">
                                <div className="mb-1 line-clamp-1 border-b border-border/30 pb-1"><span className="font-medium text-text-secondary">A:</span> {pair.text_a_preview}</div>
                                <div className="line-clamp-1"><span className="font-medium text-text-secondary">B:</span> {pair.text_b_preview}</div>
                              </td>
                              {(['is_similar_text', 'is_similar_date', 'is_similar_location', 'is_similar_all'] as const).map((flagKey) => (
                                <td key={flagKey} className="p-3">
                                  <span className={pair.flags[flagKey] ? 'font-medium text-success' : 'text-text-muted'}>
                                    {pair.flags[flagKey] ? '✓' : '—'}
                                  </span>
                                </td>
                              ))}
                              <td className="p-3">
                                <button
                                  onClick={(e) => { e.stopPropagation(); setSelectedAuditPairIndex(idx); setAuditViewMode('preview'); }}
                                  className="flex items-center gap-1 rounded border border-border px-2 py-1 text-[11px] text-text-muted transition-colors hover:border-accent hover:text-accent"
                                >
                                  <Layers className="h-3 w-3" /> Preview
                                </button>
                              </td>
                            </tr>
                            );
                          })}
                        </tbody>
                      </table>
                      {filteredPairs.length > AUDIT_PAGE_SIZE && (
                        <div className="flex items-center justify-between border-t border-border px-3 py-2 text-xs text-text-muted">
                          <span>
                            Showing {clampedAuditPage * AUDIT_PAGE_SIZE + 1}-{Math.min((clampedAuditPage + 1) * AUDIT_PAGE_SIZE, filteredPairs.length)} of {filteredPairs.length}
                          </span>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setAuditPage((p) => Math.max(0, p - 1))}
                              disabled={clampedAuditPage === 0}
                              className="rounded border border-border px-2 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
                            >
                              Prev
                            </button>
                            <span className="font-mono">{clampedAuditPage + 1} / {totalAuditPages}</span>
                            <button
                              onClick={() => setAuditPage((p) => Math.min(totalAuditPages - 1, p + 1))}
                              disabled={clampedAuditPage >= totalAuditPages - 1}
                              className="rounded border border-border px-2 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
                            >
                              Next
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : filteredPairs[selectedAuditPairIndex] || filteredPairs[0] ? (
                    <AlignedInspector pair={filteredPairs[selectedAuditPairIndex] || filteredPairs[0]} />
                  ) : (
                    <div className="p-12 text-center text-sm text-text-muted">
                      No candidate pairs to preview — adjust the filters above.
                    </div>
                  )}
                </Panel>

                {/* Stage B: Canonical Clusters */}
                <Panel
                  title="Stage B — Canonical Clusters"
                  subtitle="Connected components over the full-convergence pairs above, one canonical (most-complete) row picked per cluster"
                  actions={
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs text-text-muted">{clusters.length} clusters</span>
                      <button
                        onClick={handleApplyDedup}
                        disabled={applyLoading}
                        title="Loads this dataset (with dedup_cluster_id/size/is_canonical/is_duplicate columns baked in) as the active Data Explorer dataset — no download/re-upload needed"
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                      >
                        <Upload className="h-3 w-3" /> {applyLoading ? 'Applying...' : 'Apply to Dataset'}
                      </button>
                      <button
                        onClick={handleExportDedup}
                        disabled={exportLoading}
                        title="Downloads the full dataset with dedup_cluster_id/size/is_canonical/is_duplicate columns baked in, plus a metadata JSON recording this run's parameters — zipped together"
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                      >
                        <Download className="h-3 w-3" /> {exportLoading ? 'Exporting...' : 'Export Enriched Dataset (.zip)'}
                      </button>
                    </div>
                  }
                  noPad
                >
                  <ClusterTable
                    clusters={clusters}
                    selectedId={selectedClusterId}
                    onSelectId={setSelectedClusterId}
                    simFloor={advResult?.parameters?.threshold ?? 0.6}
                    emptyMessage="No clusters formed — no pairs reached full convergence (text + date + location) at the current settings."
                  />
                </Panel>

                {/* Spatio-Temporal Cluster Spread — computed on demand since
                    it's a second gazetteer-aware coordinate resolution pass
                    over every clustered row */}
                {clusters.length > 0 && (
                  <Panel
                    title="Spatio-Temporal Cluster Spread"
                    subtitle="Every cluster member's temporal + spatial distance from its cluster's canonical record"
                    actions={
                      <button
                        onClick={handleLoadCanonicalDistances}
                        disabled={canonicalDistancesLoading}
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                      >
                        <Play className="h-3 w-3" />
                        {canonicalDistancesLoading ? 'Computing…' : canonicalDistancePoints ? 'Refresh Chart' : 'Load Chart'}
                      </button>
                    }
                    noPad
                  >
                    {canonicalDistancePoints ? (
                      <ClusterSpatioTemporalChart points={canonicalDistancePoints} onSelectCluster={setSelectedClusterId} />
                    ) : (
                      <p className="p-4 text-xs text-text-muted">
                        Click "Load Chart" to plot every cluster member's temporal/spatial distance from its
                        canonical record — resolves coordinates (incl. gazetteer, if enabled) for all
                        {' '}{clusters.reduce((n, c) => n + c.size, 0).toLocaleString()} clustered rows.
                      </p>
                    )}
                  </Panel>
                )}

                {/* Text-Only Exact Duplicates — independent of Stage B's gate */}
                <Panel
                  title="Text-Exact Duplicates (date/location ignored)"
                  subtitle="Pairs at sim ≥0.88 on the selected embedding columns alone — near-certain re-filed duplicates that Stage B's stricter gate may have rejected (often because location couldn't be resolved). strong_similar (0.80-0.88) pairs are intentionally left out of this pass — use the Semantic Similarity Bins filter above to review those by hand."
                  actions={
                    <span className="font-mono text-xs text-text-muted">
                      {(advResult.text_duplicate_clusters ?? []).length} groups
                    </span>
                  }
                  noPad
                >
                  <ClusterTable
                    clusters={advResult.text_duplicate_clusters ?? []}
                    selectedId={selectedTextClusterId}
                    onSelectId={setSelectedTextClusterId}
                    simFloor={0.88}
                    emptyMessage="No text-exact duplicates found at the current threshold."
                  />
                </Panel>
              </div>
            );
          })()}
        </div>
      )}

      {/* CROSS-DB TAB */}
      {activeTab === 'cross_db' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Cross-DB Pipeline — Database Pool"
            subtitle="Pool Dataset A (loaded above) with up to 5 more independently-mapped databases and run one clustering pass across all of them"
            actions={
              <button
                onClick={handleRunPooledDedup}
                disabled={crossLoading || poolDatasets.length === 0}
                title={poolDatasets.length === 0 ? 'Add at least one more database below first' : undefined}
                className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                {crossLoading ? 'Running…' : 'Execute Pipeline'}
              </button>
            }
          >
            <div className="flex flex-wrap items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5 text-text-secondary">
                <Database className="h-3.5 w-3.5 text-text-muted" />
                {1 + poolDatasets.length} / {MAX_POOL_DATASETS} databases pooled — {totalPoolRows.toLocaleString()} total rows
              </span>
            </div>
            {totalPoolRows > POOL_BLOCKING_RECOMMENDED_AT && !useClusterBlocking && (
              <p className="mt-2 flex items-center gap-1.5 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-[11px] text-warning">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                {totalPoolRows.toLocaleString()} pooled rows without Cluster-Blocked Comparison builds a dense
                ~{Math.round((totalPoolRows * totalPoolRows) / 1_000_000).toLocaleString()}M-cell similarity matrix —
                turn on Cluster-Blocked Comparison in the panel above before running this.
              </p>
            )}

            <div className="mt-3 grid grid-cols-1 items-center gap-6 border-t border-border/50 pt-3 sm:grid-cols-3">
              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Cosine Similarity Threshold</span>
                  <span className="font-mono text-accent">{threshold}</span>
                </label>
                <input
                  type="range"
                  min="0.50"
                  max="1.00"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Temporal Diff (Days)</span>
                  <span className="font-mono text-accent">{dateDiffDays} d</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="1"
                  value={dateDiffDays}
                  onChange={(e) => setDateDiffDays(parseInt(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Spatial Radius (Haversine)</span>
                  <span className="font-mono text-accent">{maxKm} km</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="200"
                  step="5"
                  value={maxKm}
                  onChange={(e) => setMaxKm(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>
            </div>

            <DedupeProgressBar loading={crossLoading} title="Cross-Database Similarity Pipeline" />
          </Panel>

          <Panel
            title={`Database Pool (${1 + poolDatasets.length} / ${MAX_POOL_DATASETS})`}
            subtitle="Dataset A above is always included; add up to 5 more independently-mapped databases to pool with it"
            actions={
              1 + poolDatasets.length < MAX_POOL_DATASETS ? (
                <label className="flex cursor-pointer items-center gap-2 rounded-md border border-border bg-raised px-3 py-1.5 text-xs text-text-secondary transition-colors hover:border-accent hover:text-accent">
                  <Upload className="h-3.5 w-3.5" />
                  {poolUploadLoading ? 'Loading…' : 'Add Database'}
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls,.json"
                    className="hidden"
                    disabled={poolUploadLoading}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleAddPoolDataset(file);
                      e.target.value = '';
                    }}
                  />
                </label>
              ) : (
                <span className="text-[11px] text-text-muted">Max {MAX_POOL_DATASETS} databases</span>
              )
            }
          >
            {poolUploadError && <p className="mb-2 text-xs text-danger">{poolUploadError}</p>}

            <div className="flex items-center justify-between rounded-md border border-border/50 bg-deep/50 px-3 py-2">
              <span className="flex items-center gap-1.5 text-xs text-text-secondary">
                <Database className="h-3.5 w-3.5 text-text-muted" />
                <span className="font-medium text-text-primary">Dataset A</span>
                <span className="text-text-muted">(active dataset) — {data?.rows ? `${data.rows.length.toLocaleString()} rows` : '0 rows'}</span>
              </span>
              <span className="text-[10px] text-text-muted">Mapped via the Data Source & Embedding Columns panel above</span>
            </div>

            {poolDatasets.length === 0 ? (
              <p className="mt-3 text-xs text-text-muted">
                Add at least one more database above to run a pool comparison.
              </p>
            ) : (
              <div className="mt-3 flex flex-col gap-2">
                {poolDatasets.map((d) => {
                  const isExpanded = expandedPoolId === d.id;
                  return (
                    <div key={d.id} className="rounded-md border border-border">
                      <div className="flex items-center justify-between px-3 py-2">
                        <button
                          onClick={() => setExpandedPoolId(isExpanded ? null : d.id)}
                          className="flex flex-1 items-center gap-2 text-left text-xs text-text-secondary"
                        >
                          <Database className="h-3.5 w-3.5 text-text-muted" />
                          <span className="font-medium text-text-primary">{d.name}</span>
                          <span className="text-text-muted">— {d.rows.length.toLocaleString()} rows, {d.columns.length} columns</span>
                        </button>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setExpandedPoolId(isExpanded ? null : d.id)}
                            className="rounded border border-border px-2 py-1 text-[11px] text-text-muted transition-colors hover:border-accent hover:text-accent"
                          >
                            {isExpanded ? 'Collapse' : 'Map Columns'}
                          </button>
                          <button
                            onClick={() => removePoolDataset(d.id)}
                            className="rounded border border-danger/30 px-2 py-1 text-[11px] text-danger transition-colors hover:border-danger hover:bg-danger/10"
                          >
                            Remove
                          </button>
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="border-t border-border/50 p-3">
                          <div className="flex flex-col gap-2">
                            <div className="flex items-center justify-between">
                              <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
                                <Layers className="h-3.5 w-3.5" /> Embedding Columns
                              </span>
                              <span className="font-mono text-[10px] text-text-muted">{d.embedCols.length} selected</span>
                            </div>
                            <input
                              type="text"
                              value={d.embedColFilter}
                              onChange={(e) => updatePoolDataset(d.id, { embedColFilter: e.target.value })}
                              placeholder="Filter columns..."
                              className="w-full rounded-md border border-border bg-deep px-3 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                            />
                            <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto p-1">
                              {d.columns
                                .filter((c) => c.toLowerCase().includes(d.embedColFilter.toLowerCase()))
                                .map((c) => (
                                  <button
                                    key={c}
                                    onClick={() => togglePoolEmbedCol(d.id, c)}
                                    className={`rounded border px-2 py-1 font-mono text-[11px] transition-colors ${
                                      d.embedCols.includes(c)
                                        ? 'border-accent bg-accent-dim/30 text-accent-bright'
                                        : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                                    }`}
                                  >
                                    {c}
                                  </button>
                                ))}
                            </div>
                          </div>

                          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
                            <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
                              <Calendar className="h-3.5 w-3.5" /> Field Mapping
                            </span>
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                              <div>
                                <label className="text-[10px] text-text-muted">Date Column</label>
                                <select
                                  value={d.dateCol}
                                  onChange={(e) => updatePoolDataset(d.id, { dateCol: e.target.value })}
                                  className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                                >
                                  <option value="">(none — no date gating)</option>
                                  {d.columns.map((c) => (
                                    <option key={c} value={c}>{c}</option>
                                  ))}
                                </select>
                              </div>

                              <div>
                                <label className="text-[10px] text-text-muted">Location Field Type</label>
                                <div className="mt-1 flex gap-1.5">
                                  <button
                                    onClick={() => updatePoolDataset(d.id, { locationMode: 'coords' })}
                                    className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                                      d.locationMode === 'coords' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                                    }`}
                                  >
                                    Lat/Lon
                                  </button>
                                  <button
                                    onClick={() => updatePoolDataset(d.id, { locationMode: 'name' })}
                                    className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                                      d.locationMode === 'name' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                                    }`}
                                  >
                                    Location Name
                                  </button>
                                </div>
                              </div>

                              {d.locationMode === 'coords' ? (
                                <>
                                  <div>
                                    <label className="text-[10px] text-text-muted">Latitude Column</label>
                                    <select
                                      value={d.latCol}
                                      onChange={(e) => updatePoolDataset(d.id, { latCol: e.target.value })}
                                      className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                                    >
                                      <option value="">(none)</option>
                                      {d.columns.map((c) => (
                                        <option key={c} value={c}>{c}</option>
                                      ))}
                                    </select>
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-text-muted">Longitude Column</label>
                                    <select
                                      value={d.lonCol}
                                      onChange={(e) => updatePoolDataset(d.id, { lonCol: e.target.value })}
                                      className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                                    >
                                      <option value="">(none)</option>
                                      {d.columns.map((c) => (
                                        <option key={c} value={c}>{c}</option>
                                      ))}
                                    </select>
                                  </div>
                                </>
                              ) : (
                                <>
                                  <div>
                                    <label className="text-[10px] text-text-muted">Location Name Column</label>
                                    <select
                                      value={d.locationCol}
                                      onChange={(e) => updatePoolDataset(d.id, { locationCol: e.target.value })}
                                      className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                                    >
                                      <option value="">(none)</option>
                                      {d.columns.map((c) => (
                                        <option key={c} value={c}>{c}</option>
                                      ))}
                                    </select>
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-text-muted">
                                      City Column (optional, preferred) — falls back to Location Name when unset
                                    </label>
                                    <select
                                      value={d.cityCol}
                                      onChange={(e) => updatePoolDataset(d.id, { cityCol: e.target.value })}
                                      className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                                    >
                                      <option value="">(none)</option>
                                      {d.columns.map((c) => (
                                        <option key={c} value={c}>{c}</option>
                                      ))}
                                    </select>
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-text-muted">State Column (optional)</label>
                                    <select
                                      value={d.stateCol}
                                      onChange={(e) => updatePoolDataset(d.id, { stateCol: e.target.value })}
                                      className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                                    >
                                      <option value="">(none)</option>
                                      {d.columns.map((c) => (
                                        <option key={c} value={c}>{c}</option>
                                      ))}
                                    </select>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
                <p className="text-[10px] text-text-muted">
                  The US Census Gazetteer toggle in the Dataset A panel above applies to every database in the
                  pool — it's one shared setting, not per-dataset.
                </p>
              </div>
            )}
          </Panel>

        </div>
      )}
    </div>
  );
}

