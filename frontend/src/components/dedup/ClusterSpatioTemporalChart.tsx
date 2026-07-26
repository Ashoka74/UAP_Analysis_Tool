import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import type { PlotMouseEvent } from 'plotly.js';
import { Eye, EyeOff } from 'lucide-react';
import type { CanonicalDistancePoint } from '../../types';

const COLORS = [
  '#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#39d2c0',
  '#f85149', '#d29922', '#79c0ff', '#56d364', '#ffa657',
  '#d2a8ff', '#a5d6ff', '#7ee787', '#ffd8b5', '#e2c5ff',
  '#76e3ea', '#ff7b72', '#e3b341', '#87ceeb', '#ff69b4',
];

// Legend gets unusable past a few dozen entries — one row per cluster —
// so it's hidden by default for larger runs and left available as a toggle
// rather than always rendered.
const LEGEND_AUTO_SHOW_THRESHOLD = 25;

function wrapHoverText(text: string, width = 60, maxLines = 10): string {
  const s = (text ?? '').toString();
  if (!s || ['nan', 'none', 'null', ''].includes(s.trim().toLowerCase())) return '(no text)';
  const escaped = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const words = escaped.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > width && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  const truncated = lines.length > maxLines ? [...lines.slice(0, maxLines), '…'] : lines;
  return truncated.join('<br>');
}

interface Props {
  points: CanonicalDistancePoint[];
  height?: number;
  onSelectCluster?: (clusterId: string) => void;
}

export function ClusterSpatioTemporalChart({ points, height = 520, onSelectCluster }: Props) {
  const [showLegend, setShowLegend] = useState<boolean | null>(null);

  const { traces, clusterCount, maxGroupCount, xRange, yRange } = useMemo(() => {
    const byCluster = new Map<string, CanonicalDistancePoint[]>();
    for (const p of points) {
      if (p.temporal_distance_days == null || p.haversine_km == null) continue;
      const list = byCluster.get(p.cluster_id) ?? [];
      list.push(p);
      byCluster.set(p.cluster_id, list);
    }
    const clusterIds = Array.from(byCluster.keys());

    // Per-cluster spread = std dev of haversine_km around that cluster's own
    // mean (a single scalar "how consistent is this cluster's spatial
    // distance from canonical," independent of x) — cheap, and doesn't
    // require a valid line fit for 1-point clusters (spread undefined ->
    // default opacity, same as the reference script).
    const spreadByCluster = new Map<string, number>();
    for (const [cid, pts] of byCluster) {
      if (pts.length < 2) continue;
      const ys = pts.map((p) => p.haversine_km as number);
      const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
      const variance = ys.reduce((a, b) => a + (b - mean) ** 2, 0) / ys.length;
      spreadByCluster.set(cid, Math.sqrt(variance));
    }
    const maxSpread = Math.max(0, ...Array.from(spreadByCluster.values()));

    function opacityFor(cid: string): number {
      const spread = spreadByCluster.get(cid);
      if (spread == null || Number.isNaN(spread)) return 0.85;
      if (maxSpread === 0) return 1.0;
      const norm = spread / maxSpread;
      return Math.max(0.15, Math.min(1.0, 1.0 - 0.8 * norm));
    }

    // Near-duplicate members very often land on the exact same (temporal,
    // haversine) distance from canonical — same date, same coordinates —
    // so plotting one marker per member silently stacks N markers on top
    // of each other with zero visual sign anything's underneath. Grouped
    // by exact (x, y) instead: one marker per distinct location, sized by
    // how many members share it, hover listing all of them.
    const MAX_HOVER_MEMBERS = 6;
    const built: any[] = [];
    let maxGroupCount = 1;
    clusterIds.forEach((cid) => {
      const pts = byCluster.get(cid) ?? [];
      const groups = new Map<string, CanonicalDistancePoint[]>();
      for (const p of pts) {
        const key = `${p.temporal_distance_days}|${p.haversine_km}`;
        const g = groups.get(key) ?? [];
        g.push(p);
        groups.set(key, g);
      }
      for (const g of groups.values()) maxGroupCount = Math.max(maxGroupCount, g.length);
    });

    clusterIds.forEach((cid, i) => {
      const pts = byCluster.get(cid) ?? [];
      const groups = new Map<string, CanonicalDistancePoint[]>();
      for (const p of pts) {
        const key = `${p.temporal_distance_days}|${p.haversine_km}`;
        const g = groups.get(key) ?? [];
        g.push(p);
        groups.set(key, g);
      }
      const grouped = Array.from(groups.values()).sort(
        (a, b) => (a[0].temporal_distance_days as number) - (b[0].temporal_distance_days as number),
      );

      const color = COLORS[i % COLORS.length];
      const op = opacityFor(cid);
      const x = grouped.map((g) => g[0].temporal_distance_days);
      const y = grouped.map((g) => g[0].haversine_km);
      // sqrt scaling so a 10x pile-up doesn't dwarf everything else, capped
      // so one massive overlap can't blow out the whole plot's scale.
      const sizes = grouped.map((g) => Math.min(28, 7 + 5 * Math.sqrt(g.length - 1)));
      const text = grouped.map((g) => {
        const p0 = g[0];
        const header =
          `<b>${p0.cluster_id}</b> (size ${p0.cluster_size})<br>` +
          `temporal: ${p0.temporal_distance_days}d | haversine: ${p0.haversine_km}km` +
          (g.length > 1 ? `<br><b>${g.length} members overlap at this exact distance</b>` : '');
        const shown = g.slice(0, MAX_HOVER_MEMBERS);
        const memberLines = shown
          .map((p) => `<br><br><b>Member (${p.member_id}):</b><br>${wrapHoverText(p.member_text)}`)
          .join('');
        const more = g.length > MAX_HOVER_MEMBERS ? `<br><br>…and ${g.length - MAX_HOVER_MEMBERS} more` : '';
        return (
          header + memberLines + more +
          `<br><br><b>Canonical (${p0.canonical_id}):</b><br>${wrapHoverText(p0.canonical_text)}`
        );
      });

      // Direct connections between a cluster's own distinct points only —
      // never across clusters — so the line network stays local and
      // readable instead of a tangle spanning the whole chart, and opacity
      // fades out loosely-scattered clusters so tight ones (likely true
      // duplicates) are what visually pop.
      if (grouped.length >= 2) {
        built.push({
          x, y, mode: 'lines', type: 'scattergl', name: cid, legendgroup: cid,
          line: { color, width: 1.5 },
          opacity: op, showlegend: false, hoverinfo: 'skip',
        });
      }
      built.push({
        x, y, text, mode: 'markers', type: 'scattergl', name: `${cid} (${pts.length})`, legendgroup: cid,
        marker: {
          color, opacity: op, size: sizes,
          line: { width: grouped.some((g) => g.length > 1) ? 1 : 0, color: 'rgba(230,237,243,0.5)' },
        },
        showlegend: true, hoverinfo: 'text', customdata: grouped.map(() => cid),
      });
    });

    // Fixed axis range computed once from the *full* dataset, not left to
    // Plotly's autorange. Autorange recomputes from whatever traces are
    // currently visible, so isolating a single cluster via the legend
    // (double-click) rescales the axes to fit just that cluster — and if
    // its points all collapse to one exact coordinate (e.g. every member
    // 0 days / 0 km from canonical), the range collapses to zero width on
    // both axes, which breaks Plotly's hover hit-testing entirely for
    // whatever's left visible. A stable range up front sidesteps that, and
    // as a bonus the chart no longer jumps around every time a cluster is
    // toggled in the legend.
    const allX = points.map((p) => p.temporal_distance_days).filter((v): v is number => v != null);
    const allY = points.map((p) => p.haversine_km).filter((v): v is number => v != null);
    const maxX = allX.length ? Math.max(...allX) : 1;
    const maxY = allY.length ? Math.max(...allY) : 1;
    const xRange: [number, number] = [-Math.max(maxX * 0.08, 0.5), maxX + Math.max(maxX * 0.08, 0.5)];
    const yRange: [number, number] = [-Math.max(maxY * 0.08, 0.5), maxY + Math.max(maxY * 0.08, 0.5)];

    return { traces: built, clusterCount: clusterIds.length, maxGroupCount, xRange, yRange };
  }, [points]);

  const legendVisible = showLegend ?? clusterCount <= LEGEND_AUTO_SHOW_THRESHOLD;

  const handleClick = (event: Readonly<PlotMouseEvent>) => {
    const pt = event.points[0] as any;
    const cid = pt?.customdata as string | undefined;
    if (cid && onSelectCluster) onSelectCluster(cid);
  };

  if (traces.length === 0) {
    return (
      <p className="p-4 text-xs text-text-muted">
        No cluster members have both a resolved date and resolved coordinates to plot — check the Field
        Mapping panel above.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2 p-4">
      <div className="flex items-center justify-between">
        <p className="max-w-2xl text-[11px] text-text-muted">
          Each point is one non-canonical cluster member, plotted by its temporal and spatial distance from
          that cluster's canonical record — canonical itself sits conceptually at the origin. Points in the
          same cluster share a color and are connected directly; opacity is scaled by that cluster's spatial
          spread, so tight, consistent clusters (likely true duplicates) stay opaque while loosely scattered
          ones fade. Members that land on the exact same distance (e.g. same date + location) are merged
          into one larger marker sized by overlap count{maxGroupCount > 1 ? ` (up to ${maxGroupCount} here)` : ''}
          {' '}rather than silently stacking invisibly — hover to see everyone at that point. Click a point to
          jump to that cluster above.
        </p>
        <button
          onClick={() => setShowLegend(!legendVisible)}
          className="flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent"
        >
          {legendVisible ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
          {legendVisible ? 'Hide' : 'Show'} Legend ({clusterCount})
        </button>
      </div>

      <Plot
        data={traces}
        onClick={handleClick}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: '#111820',
          font: { color: '#8b949e', size: 10 },
          margin: { l: 55, r: legendVisible ? 160 : 20, t: 20, b: 45 },
          xaxis: {
            title: { text: 'Temporal distance from canonical (days)' },
            gridcolor: '#21283b',
            zerolinecolor: '#30363d',
            range: xRange,
          },
          yaxis: {
            title: { text: 'Haversine distance from canonical (km)' },
            gridcolor: '#21283b',
            zerolinecolor: '#30363d',
            range: yRange,
          },
          legend: {
            font: { size: 8, color: '#8b949e' },
            bgcolor: 'transparent',
            x: 1.02,
            y: 1,
          },
          showlegend: legendVisible,
          height,
          hovermode: 'closest',
        }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
