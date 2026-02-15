import Plot from 'react-plotly.js';
import type { CramersVData } from '../../types';

interface Props {
  data: CramersVData;
  height?: number;
}

export function CorrelationHeatmap({ data, height = 400 }: Props) {
  // Mask upper triangle
  const masked = data.matrix.map((row, i) =>
    row.map((val, j) => (j > i ? null : val))
  );

  // Annotation text
  const annotations = [];
  for (let i = 0; i < data.labels.length; i++) {
    for (let j = 0; j <= i; j++) {
      annotations.push({
        x: data.labels[j],
        y: data.labels[i],
        text: masked[i][j] != null ? masked[i][j]!.toFixed(2) : '',
        font: { color: '#e6edf3', size: 10 },
        showarrow: false,
      });
    }
  }

  return (
    <Plot
      data={[
        {
          z: masked,
          x: data.labels,
          y: data.labels,
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
            title: { text: "Cramer's V", font: { color: '#8b949e', size: 10 } },
            tickfont: { color: '#8b949e', size: 9 },
          },
        },
      ]}
      layout={{
        title: {
          text: "Cramer's V Correlation Matrix",
          font: { color: '#e6edf3', size: 14 },
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#8b949e', size: 10 },
        margin: { l: 100, r: 40, t: 40, b: 100 },
        xaxis: { tickangle: -45 },
        annotations,
        height,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
}
