import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import type { PlotMouseEvent } from 'plotly.js';
import { Download, MousePointerClick } from 'lucide-react';
import type { ClusterViz } from '../../types';
import { useStore } from '../../store/useStore';

const COLORS = [
  '#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#39d2c0',
  '#f85149', '#d29922', '#79c0ff', '#56d364', '#ffa657',
  '#d2a8ff', '#a5d6ff', '#7ee787', '#ffd8b5', '#e2c5ff',
  '#76e3ea', '#ff7b72', '#e3b341', '#87ceeb', '#ff69b4',
];

// Word-wraps + HTML-escapes raw narrative text into <br>-joined lines, so
// Plotly hover tooltips show a readable paragraph instead of one unbroken
// line (and stray '<'/'>' in source text can't be read as markup).
function wrapHoverText(text: string, width = 60, maxLines = 14): string {
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

function downloadCsv(rows: Record<string, unknown>[], columns: string[], filename: string) {
  const headers = columns.join(',');
  const body = rows.map((row) =>
    columns.map((col) => {
      const val = row[col] == null ? '' : String(row[col]);
      return `"${val.replace(/"/g, '""')}"`;
    }).join(',')
  );
  const csvContent = [headers, ...body].join('\n');
  // Blob URL, not a data: URI — data: URIs are capped at a few MB by browsers.
  const blob = new Blob(['﻿' + csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

interface Props {
  viz: ClusterViz;
  height?: number;
}

export function ClusterVisualization({ viz, height = 450 }: Props) {
  const { data } = useStore();
  const [selectedTraceIdx, setSelectedTraceIdx] = useState<number | null>(null);

  const traces = useMemo(
    () =>
      viz.traces.map((t, i) => ({
        x: t.x,
        y: t.y,
        text: t.text.map((txt) => wrapHoverText(txt)),
        customdata: t.row_ids,
        name: `${t.name} (${t.count})`,
        type: 'scatter' as const,
        mode: 'markers' as const,
        marker: {
          color: COLORS[i % COLORS.length],
          size: 5,
          opacity: 0.8,
        },
        hoverinfo: 'text' as const,
      })),
    [viz]
  );

  const selectedTrace = selectedTraceIdx != null ? viz.traces[selectedTraceIdx] : null;

  // Cross-reference the selected cluster's row_ids against the loaded
  // dataset's row_ids (same alignment convention as DataTable's imputation
  // overlay) to pull back the full raw rows, not just this column's text.
  const selectedRows = useMemo(() => {
    if (!selectedTrace || !data?.rows || !data.row_ids) return null;
    const idxByRowId = new Map(data.row_ids.map((id, i) => [id, i]));
    const rows: Record<string, unknown>[] = [];
    for (const rowId of selectedTrace.row_ids) {
      const i = idxByRowId.get(rowId);
      if (i != null) rows.push(data.rows[i]);
    }
    return rows;
  }, [selectedTrace, data]);

  const handleClick = (event: Readonly<PlotMouseEvent>) => {
    const pt = event.points[0];
    if (pt) setSelectedTraceIdx(pt.curveNumber);
  };

  const handleDownloadCsv = () => {
    if (!selectedRows || !data?.columns) return;
    downloadCsv(selectedRows, data.columns, `${viz.title.replace(/\s+/g, '_')}_cluster_${selectedTraceIdx}.csv`);
  };

  return (
    <div className="flex flex-col gap-3 lg:flex-row">
      <div className="min-w-0 flex-1">
        <Plot
          data={traces}
          onClick={handleClick}
          layout={{
            title: { text: viz.title, font: { color: '#e6edf3', size: 14 } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: '#111820',
            font: { color: '#8b949e', size: 10 },
            margin: { l: 40, r: 20, t: 40, b: 40 },
            xaxis: {
              gridcolor: '#21283b',
              zerolinecolor: '#30363d',
              showticklabels: false,
            },
            yaxis: {
              gridcolor: '#21283b',
              zerolinecolor: '#30363d',
              showticklabels: false,
            },
            legend: {
              font: { size: 9, color: '#8b949e' },
              bgcolor: 'transparent',
              x: 1.02,
              y: 1,
            },
            height,
            showlegend: true,
          }}
          config={{ responsive: true, displayModeBar: true, displaylogo: false }}
          style={{ width: '100%' }}
        />
        <p className="mt-1.5 flex items-center gap-1 text-[10px] text-text-muted">
          <Download className="h-3 w-3" /> Use the camera icon in the chart's toolbar to export it as a PNG.
        </p>
      </div>

      <div className="flex flex-col gap-2 rounded-lg border border-border bg-abyss/60 p-3 lg:w-80 lg:shrink-0">
        {!selectedTrace || !selectedRows ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 py-8 text-center text-[11px] text-text-muted">
            <MousePointerClick className="h-4 w-4" />
            Click a point to inspect that cluster's raw rows here.
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between border-b border-border/50 pb-2">
              <div className="flex flex-col">
                <span className="text-xs font-semibold text-text-primary">{selectedTrace.name}</span>
                <span className="text-[10px] text-text-muted">{selectedRows.length} rows</span>
              </div>
              <button
                onClick={handleDownloadCsv}
                className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[10px] text-text-muted transition-colors hover:border-accent/50 hover:text-accent-bright"
              >
                <Download className="h-3 w-3" /> CSV
              </button>
            </div>
            <div className="max-h-96 overflow-auto">
              <table className="w-full border-collapse text-left text-[10px]">
                <thead>
                  <tr className="border-b border-border/50 text-text-secondary">
                    {(data?.columns ?? []).map((col) => (
                      <th key={col} className="whitespace-nowrap px-2 py-1 font-mono">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30 text-text-primary">
                  {selectedRows.map((row, i) => (
                    <tr key={i} className="hover:bg-elevated/30">
                      {(data?.columns ?? []).map((col) => (
                        <td key={col} className="whitespace-nowrap px-2 py-1">
                          {row[col] == null || row[col] === '' ? '—' : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
