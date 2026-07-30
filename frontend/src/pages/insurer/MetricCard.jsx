// A KPI card that flips to its underlying data.
//
// Double-click (or the small table icon) turns the chart over to show every year
// as a table, selectable and copyable — because a supervisor who is about to cite
// a number wants the series, not a sparkline. Copy emits TSV so it pastes
// straight into Excel as columns.
import { useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';

export function fmtVal(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (unit === 'inr') {
    const cr = value / 1e7;
    if (Math.abs(cr) >= 1e5) return `₹${(cr / 1e5).toFixed(2)} L Cr`;
    if (Math.abs(cr) >= 1) return `₹${cr.toLocaleString('en-IN', { maximumFractionDigits: 0 })} Cr`;
    return `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  }
  if (unit === 'percent') return `${value.toFixed(2)}%`;
  if (unit === 'ratio') return value.toFixed(2);
  if (unit === 'rate') return value.toLocaleString('en-IN', { maximumFractionDigits: 1 });
  return Math.round(value).toLocaleString('en-IN');
}

// Raw numbers for the table — a supervisor copying data wants the figure, not the
// display abbreviation. ₹ is shown in crore because that is the unit of record.
function tableVal(v, unit) {
  if (v === null || v === undefined) return '';
  if (unit === 'inr') return (v / 1e7).toFixed(2);
  if (unit === 'percent' || unit === 'ratio') return v.toFixed(2);
  return String(Math.round(v));
}

export default function MetricCard({ k, children, onToast }) {
  const [flipped, setFlipped] = useState(false);
  const trend = k.trend || [];
  const unitHead = k.unit === 'inr' ? 'Value (₹ Cr)'
    : k.unit === 'percent' ? 'Value (%)' : k.unit === 'ratio' ? 'Ratio' : 'Value';

  const chart = useMemo(() => ({
    labels: trend.map((p) => p.fy),
    datasets: [{
      data: trend.map((p) => (k.unit === 'inr' ? p.v / 1e7 : p.v)),
      borderColor: 'rgba(30,58,138,0.9)', backgroundColor: 'rgba(30,58,138,0.10)',
      borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, spanGaps: false,
    }],
  }), [trend, k.unit]);

  const opts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false }, tooltip: { enabled: true,
      callbacks: { label: (c) => fmtVal(k.unit === 'inr' ? c.parsed.y * 1e7 : c.parsed.y, k.unit) } } },
    scales: { x: { display: false }, y: { display: false } },
  }), [k.unit]);

  const copy = async () => {
    const tsv = ['Year\t' + unitHead,
      ...trend.map((p) => `${p.fy}\t${tableVal(p.v, k.unit)}`)].join('\n');
    try {
      await navigator.clipboard.writeText(tsv);
      onToast && onToast(`${k.label}: ${trend.length} years copied`);
    } catch {
      onToast && onToast('Could not copy', true);
    }
  };

  return (
    <div className={`i360-card ${flipped ? 'is-flipped' : ''}`}
      onDoubleClick={() => setFlipped((f) => !f)}
      title={flipped ? 'Double-click to show the chart' : 'Double-click to show the data table'}>
      <div className="i360-head">
        <span className="i360-label">
          {k.label}
          {k.context && <span className="i360-ctx" title={k.context}>{k.context}</span>}
        </span>
        <span className="i360-fy">
          {k.fy}
          {k.stale && <span className="i360-stale" title={`no ${k.selected_fy || 'selected-year'} filing; showing ${k.fy}`}>STALE</span>}
          <button type="button" className="i360-flip" aria-label={flipped ? 'Show chart' : 'Show data table'}
            onClick={(e) => { e.stopPropagation(); setFlipped((f) => !f); }}>
            <i className={`fas ${flipped ? 'fa-chart-line' : 'fa-table'}`} />
          </button>
        </span>
      </div>

      {!flipped ? (
        <>
          <div className="i360-value">
            {fmtVal(k.value, k.unit)}
            {k.computed && <i className="fas fa-calculator i360-calc"
              title={`Computed from ${k.components?.length || 0} leaf items — not a stored figure`} />}
          </div>
          {children}
          {trend.length > 1
            ? <div className="i360-spark"><Line data={chart} options={opts} /></div>
            : <div className="i360-spark-empty">single year</div>}
          {k.computed && k.components?.length > 0 && (
            <details className="i360-prov">
              <summary>= sum of {k.components.length} items</summary>
              {k.what && <p className="i360-what">{k.what}</p>}
              <ul>{k.components.map((c) => <li key={c}>{c}</li>)}</ul>
            </details>
          )}
        </>
      ) : (
        <div className="i360-table-wrap">
          <table className="i360-table">
            <thead><tr><th>Year</th><th>{unitHead}</th></tr></thead>
            <tbody>
              {trend.slice().reverse().map((p) => (
                <tr key={p.fy} className={p.fy === k.fy ? 'on' : ''}>
                  <td>{p.fy}</td><td>{tableVal(p.v, k.unit)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <button type="button" className="i360-copy" onClick={(e) => { e.stopPropagation(); copy(); }}>
            <i className="far fa-copy" /> Copy {trend.length} years
          </button>
        </div>
      )}
    </div>
  );
}
