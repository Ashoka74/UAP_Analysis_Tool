import Plot from 'react-plotly.js';
import type { ClusterViz } from '../../types';

const COLORS = [
  '#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#39d2c0',
  '#f85149', '#d29922', '#79c0ff', '#56d364', '#ffa657',
  '#d2a8ff', '#a5d6ff', '#7ee787', '#ffd8b5', '#e2c5ff',
  '#76e3ea', '#ff7b72', '#e3b341', '#87ceeb', '#ff69b4',
];

interface Props {
  viz: ClusterViz;
  height?: number;
}

export function ClusterVisualization({ viz, height = 450 }: Props) {
  const traces = viz.traces.map((t, i) => ({
    x: t.x,
    y: t.y,
    text: t.text,
    name: `${t.name} (${t.count})`,
    type: 'scatter' as const,
    mode: 'markers' as const,
    marker: {
      color: COLORS[i % COLORS.length],
      size: 5,
      opacity: 0.8,
    },
    hoverinfo: 'text' as const,
  }));

  return (
    <Plot
      data={traces}
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
  );
}
