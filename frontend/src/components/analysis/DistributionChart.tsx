import Plot from 'react-plotly.js';
import type { DistributionItem } from '../../types';

interface Props {
  data: DistributionItem[];
  title: string;
  height?: number;
}

const COLORS = [
  '#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#39d2c0',
  '#f85149', '#d29922', '#79c0ff', '#56d364', '#ffa657',
];

export function DistributionChart({ data, title, height = 300 }: Props) {
  return (
    <Plot
      data={[
        {
          labels: data.map((d) => d.label),
          values: data.map((d) => d.count),
          type: 'pie',
          hole: 0.5,
          marker: { colors: COLORS },
          textfont: { color: '#e6edf3', size: 9 },
          hovertemplate: '%{label}<br>Count: %{value}<br>%{percent}<extra></extra>',
        },
      ]}
      layout={{
        title: { text: title, font: { color: '#e6edf3', size: 13 } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#8b949e', size: 9 },
        margin: { l: 10, r: 10, t: 40, b: 10 },
        showlegend: true,
        legend: { font: { size: 8, color: '#8b949e' }, bgcolor: 'transparent' },
        height,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
}
