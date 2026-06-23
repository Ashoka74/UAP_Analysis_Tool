import Plot from 'react-plotly.js';
import type { XGBoostResult } from '../../types';

interface Props {
  results: Record<string, XGBoostResult>;
}

export function XGBoostResults({ results }: Props) {
  const columns = Object.keys(results);

  return (
    <div className="space-y-4">
      {columns.map((col) => {
        const r = results[col];
        const features = Object.keys(r.feature_importance);
        const importances = Object.values(r.feature_importance);

        return (
          <div key={col} className="rounded-lg border border-border bg-surface">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <div>
                <h4 className="text-sm font-semibold text-text-primary">{col}</h4>
                <p className="text-xs text-text-muted">Feature importance analysis</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-text-muted">Accuracy:</span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                    r.accuracy >= 0.8
                      ? 'bg-success/20 text-success'
                      : r.accuracy >= 0.6
                        ? 'bg-warning/20 text-warning'
                        : 'bg-danger/20 text-danger'
                  }`}
                >
                  {(r.accuracy * 100).toFixed(1)}%
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
                      color: importances.map((v) => {
                        const t = v / Math.max(...importances, 0.001);
                        return `rgba(88, 166, 255, ${0.3 + t * 0.7})`;
                      }),
                    },
                    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
                  },
                ]}
                layout={{
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#8b949e', size: 10 },
                  margin: { l: 120, r: 20, t: 10, b: 30 },
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
      })}
    </div>
  );
}
