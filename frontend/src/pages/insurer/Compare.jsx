// Side-by-side insurer comparison.
//
// The whole design problem here is that comparison INVITES the meaningless
// cross-class query — a life insurer's book against a general insurer's. Rather
// than block it (universal balance-sheet lines are legitimately comparable), the
// UI shows only metrics every selected insurer reports, and says loudly when the
// selection spans classes.
import { useMemo } from 'react';
import { Line } from 'react-chartjs-2';

const SERIES = ['#1e3a8a', '#b45309', '#047857', '#7c3aed'];

function fmt(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (unit === 'inr') {
    const cr = value / 1e7;
    if (Math.abs(cr) >= 1e5) return `₹${(cr / 1e5).toFixed(2)} L Cr`;
    return `₹${cr.toLocaleString('en-IN', { maximumFractionDigits: 0 })} Cr`;
  }
  if (unit === 'percent') return `${value.toFixed(2)}%`;
  if (unit === 'ratio') return value.toFixed(2);
  if (unit === 'rate') return value.toLocaleString('en-IN', { maximumFractionDigits: 1 });
  return Math.round(value).toLocaleString('en-IN');
}

function TrendRow({ metric, insurers }) {
  const data = useMemo(() => {
    const labels = [...new Set(
      Object.values(metric.trends || {}).flat().map((p) => p.fy),
    )].sort();
    return {
      labels,
      datasets: insurers.map((ins, i) => {
        const byFy = Object.fromEntries((metric.trends?.[ins.id] || []).map((p) => [p.fy, p.v]));
        return {
          label: ins.name,
          data: labels.map((f) => (f in byFy
            ? (metric.unit === 'inr' ? byFy[f] / 1e7 : byFy[f]) : null)),
          borderColor: SERIES[i % SERIES.length],
          backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3,
          spanGaps: false,   // a gap in filings must LOOK like a gap, never interpolate
        };
      }),
    };
  }, [metric, insurers]);
  const opts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false } },
    scales: { x: { display: false }, y: { display: false } },
  }), []);
  if (!metric.trends || !Object.keys(metric.trends).length) return null;
  return <div className="cmp-trend"><Line data={data} options={opts} /></div>;
}

export default function Compare({ data, insurers }) {
  if (!data) return null;
  const ins = data.insurers || [];

  return (
    <div className="cmp">
      {data.mixed_class && (
        <div className="cmp-warn">
          <i className="fas fa-triangle-exclamation" />
          <div>
            <strong>Comparing across insurer classes ({data.classes.join(', ')}).</strong>{' '}
            Only lines that every selected insurer reports are shown — class-specific
            metrics (premium, solvency, claims) are hidden because they mean different
            things in each class. Market share is suppressed for the same reason.
          </div>
        </div>
      )}

      <div className="cmp-legend">
        {ins.map((x, i) => (
          <span key={x.id} className="cmp-chip">
            <i className="cmp-dot" style={{ background: SERIES[i % SERIES.length] }} />
            {x.name} <em>{x.class}</em>
          </span>
        ))}
      </div>

      {!data.mixed_class && data.market_share?._class && (
        <div className="cmp-share">
          <div className="cmp-share-title">
            Market share — {data.market_share._class} segment, {data.market_share._fy}
          </div>
          <div className="cmp-share-bars">
            {ins.map((x, i) => {
              const v = data.market_share[x.id];
              if (v === undefined) return null;
              const max = Math.max(...ins.map((y) => data.market_share[y.id] || 0));
              return (
                <div key={x.id} className="cmp-share-row">
                  <span className="cmp-share-name">{x.name}</span>
                  <span className="cmp-share-bar">
                    <span style={{ width: `${max ? (v / max) * 100 : 0}%`,
                      background: SERIES[i % SERIES.length] }} />
                  </span>
                  <span className="cmp-share-val">{v}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="cmp-tablewrap">
        <table className="cmp-table">
          <thead>
            <tr>
              <th>Metric</th>
              {ins.map((x, i) => (
                <th key={x.id}><i className="cmp-dot" style={{ background: SERIES[i % SERIES.length] }} />{x.name}</th>
              ))}
              <th className="cmp-trendhead">Trend</th>
            </tr>
          </thead>
          <tbody>
            {(data.metrics || []).map((m) => (
              <tr key={m.id} className={m.derived ? 'is-derived' : ''}>
                <th scope="row">
                  {m.label}
                  <span className="cmp-fy">{m.fy}</span>
                  {m.note && <span className="cmp-note">{m.note}</span>}
                </th>
                {ins.map((x) => {
                  const v = m.values?.[x.id];
                  const tone = m.best === x.id ? 'is-best' : m.worst === x.id ? 'is-worst' : '';
                  return <td key={x.id} className={tone}>{fmt(v, m.unit)}</td>;
                })}
                <td className="cmp-trendcell"><TrendRow metric={m} insurers={ins} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="cmp-foot">
        Green marks the better figure <em>for that metric's direction</em> — lowest is best for
        grievances and expenses, highest for premium and profit. Metrics are compared on the most
        recent year all selected insurers share, so a fresher filing never wins by default.
      </p>
    </div>
  );
}
