// Side-by-side insurer comparison.
//
// The whole design problem here is that comparison INVITES the meaningless
// cross-class query — a life insurer's book against a general insurer's. Rather
// than block it (universal balance-sheet lines are legitimately comparable), the
// UI says loudly when the selection spans classes.
//
// A row is shown when ANY selected insurer reports it, with a dash for those that
// do not and a coverage count when it is short of the full set. Requiring every
// insurer to report a row hid seven useful columns because of one missing eighth.
// The year is the part that stays strict: one year per row for everyone in it.
import { useMemo } from 'react';
import { Line } from 'react-chartjs-2';

// Eight categorical slots, in fixed order. Not eyeballed — the previous four were,
// and two of them failed the checks: #1e3a8a sat outside the lightness band and a
// hand-picked teal read as grey. Validated for the light surface: worst adjacent
// colour-blind separation ΔE 9.1 (target ≥8), worst normal-vision ΔE 19.6 (floor 15).
//
// Three slots fall below 3:1 contrast against white, which is allowed only where
// identity does not rest on colour alone. It does not here: every column carries the
// insurer's NAME, and each legend chip pairs its dot with the name. The colour is a
// cross-reference into the table, never the label itself.
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                '#e87ba4', '#008300', '#4a3aa7', '#e34948'];

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

function TrendRow({ metric, insurers, colorOf }) {
  const data = useMemo(() => {
    const labels = [...new Set(
      Object.values(metric.trends || {}).flat().map((p) => p.fy),
    )].sort();
    return {
      labels,
      datasets: insurers.map((ins) => {
        const byFy = Object.fromEntries((metric.trends?.[ins.id] || []).map((p) => [p.fy, p.v]));
        return {
          label: ins.name,
          data: labels.map((f) => (f in byFy
            ? (metric.unit === 'inr' ? byFy[f] / 1e7 : byFy[f]) : null)),
          borderColor: colorOf(ins.id),
          backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3,
          spanGaps: false,   // a gap in filings must LOOK like a gap, never interpolate
        };
      }),
    };
  }, [metric, insurers, colorOf]);
  const opts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false } },
    scales: { x: { display: false }, y: { display: false } },
  }), []);
  if (!metric.trends || !Object.keys(metric.trends).length) return null;
  return <div className="cmp-trend"><Line data={data} options={opts} /></div>;
}

export default function Compare({ data, insurers, slotOf, colourSlots = 8 }) {
  if (!data) return null;
  const ins = data.insurers || [];
  // Colour follows the INSURER, not its position in the list. Keyed on selection
  // order, removing the second of four repainted the two below it — the same
  // insurer changed colour without its data changing, which is exactly what a
  // legend must never do. slotOf holds a slot per insurer for as long as it is
  // selected; the index fallback keeps this component usable on its own.
  const colorOf = (id) => SERIES[((slotOf ? slotOf(id) : ins.findIndex((x) => x.id === id)) || 0) % SERIES.length];
  // Past the palette the sparkline would repeat hues, and a 12-line sparkline in a
  // 118px cell is unreadable regardless. Drop the column rather than draw a lie —
  // every number is still in the table.
  const showTrend = ins.length <= colourSlots;

  return (
    <div className="cmp">
      {data.mixed_class && (
        <div className="cmp-warn">
          <i className="fas fa-triangle-exclamation" />
          <div>
            <strong>Comparing across insurer classes ({data.classes.join(', ')}).</strong>{' '}
            Class-specific metrics mean different things in each class, so a row here is
            only as comparable as its label suggests, and class market share is suppressed.
            Check the coverage note on a row before reading across it. <strong>Where their books overlap on a line of business</strong>,
            that line is compared directly at the top — a standalone health insurer and a
            general insurer genuinely compete on health, and are ranked there against every
            insurer writing it.
          </div>
        </div>
      )}

      <div className="cmp-legend">
        {ins.map((x) => (
          <span key={x.id} className="cmp-chip">
            <i className="cmp-dot" style={{ background: colorOf(x.id) }} />
            {x.name} <em>{x.segment || x.class}</em>
            {x.status && x.status !== 'active' && (
              <span className={`cmp-status is-${x.status}`}>
                {x.status === 'not_writing' ? 'not writing' : 'closed'}
              </span>
            )}
          </span>
        ))}
      </div>

      {!data.mixed_class && data.market_share?._class && (
        <div className="cmp-share">
          <div className="cmp-share-title">
            Market share — {data.market_share._class} segment, {data.market_share._fy}
          </div>
          <div className="cmp-share-bars">
            {ins.map((x) => {
              const v = data.market_share[x.id];
              if (v === undefined) return null;
              const max = Math.max(...ins.map((y) => data.market_share[y.id] || 0));
              return (
                <div key={x.id} className="cmp-share-row">
                  <span className="cmp-share-name">{x.name}</span>
                  <span className="cmp-share-bar">
                    <span style={{ width: `${max ? (v / max) * 100 : 0}%`,
                      background: colorOf(x.id) }} />
                  </span>
                  <span className="cmp-share-val">{v}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Now rare — it takes a selection where no insurer reports any tracked KPI at
          all, rather than merely disagreeing on years. Still worth handling: an empty
          headed table reads as a broken page rather than as a finding. */}
      {(data.metrics || []).length === 0 ? (
        <div className="cmp-empty">
          <i className="fas fa-circle-info" />
          <div>
            <strong>None of these {ins.length} insurers reports a metric IRIS tracks for comparison.</strong>
            <p>
              This usually means the selection is all specialised or newly licensed
              insurers whose filings sit outside the compared set. Try one alongside an
              established insurer of the same class.
            </p>
          </div>
        </div>
      ) : (
      <div className="cmp-tablewrap">
        <table className="cmp-table">
          <thead>
            <tr>
              <th>Metric</th>
              {ins.map((x) => (
                <th key={x.id}><i className="cmp-dot" style={{ background: colorOf(x.id) }} />{x.name}</th>
              ))}
              {showTrend && <th className="cmp-trendhead">Trend</th>}
            </tr>
          </thead>
          <tbody>
            {(data.metrics || []).map((m) => (
              <tr key={m.id} className={m.derived ? 'is-derived' : ''}>
                <th scope="row">
                  {m.label}
                  <span className="cmp-fy">{m.fy}</span>
                  {/* Only when short of the full set. A dash then reads as "this
                      insurer did not report it", not as a broken cell — and it stops
                      the green "best" mark being read as best-of-all. */}
                  {m.coverage != null && m.selected != null && m.coverage < m.selected && (
                    <span className="cmp-cover" title={`${m.coverage} of ${m.selected} selected insurers reported this in ${m.fy}`}>
                      {m.coverage}/{m.selected} reported
                    </span>
                  )}
                  {m.note && <span className="cmp-note">{m.note}</span>}
                </th>
                {ins.map((x) => {
                  const v = m.values?.[x.id];
                  const tone = m.best === x.id ? 'is-best' : m.worst === x.id ? 'is-worst' : '';
                  return <td key={x.id} className={tone}>{fmt(v, m.unit)}</td>;
                })}
                {showTrend && <td className="cmp-trendcell"><TrendRow metric={m} insurers={ins} colorOf={colorOf} /></td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}

      <p className="cmp-foot">
        Green marks the better figure <em>for that metric's direction</em> — lowest is best for
        grievances and expenses, highest for premium and profit, and it ranks only the insurers
        that reported the row. Each row is a single year for everyone in it — the year the most
        selected insurers filed — so a fresher filing never wins by default; an insurer without
        that year shows <strong>—</strong> rather than a figure from a different one. Rows every
        insurer reports come first.
      </p>
    </div>
  );
}
