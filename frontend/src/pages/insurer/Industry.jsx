// Industry trends — the market-level view.
//
// The premium basis DIFFERS per class and the classes are never summed. There is
// no valid cross-class premium line: `gross_premium` looks universal but is a
// minor sub-line for life insurers, which would have put the whole life industry
// at ₹2,386 Cr. Each panel therefore labels the basis it used.
import { useMemo } from 'react';
import { Line, Bar } from 'react-chartjs-2';

const CLS_COLOR = { General: '#1e3a8a', Life: '#b45309', SAHI: '#047857',
  'Non-Life': '#7c3aed', Reinsurance: '#0e7490',
  'Health (line)': '#059669', 'Motor (line)': '#dc2626',
  'Fire (line)': '#ea580c', 'Marine (line)': '#0891b2' };
const CLS_LABEL = { General: 'General', Life: 'Life', SAHI: 'Standalone Health',
  'Non-Life': 'Non-Life (Gen + SAHI)', Reinsurance: 'Reinsurance (incl. FRB)',
  'Health (line)': 'Health', 'Motor (line)': 'Motor',
  'Fire (line)': 'Fire', 'Marine (line)': 'Marine' };

const lakhCr = (v) => `₹${(v / 1e12).toFixed(2)} L Cr`;

// HHI is the standard concentration measure (sum of squared percentage shares).
// The thresholds are the competition-authority convention, worth stating on screen
// so the number is readable by someone meeting it for the first time.
function hhiBand(h) {
  if (h >= 2500) return { label: 'highly concentrated', tone: 'bad' };
  if (h >= 1500) return { label: 'moderately concentrated', tone: 'warn' };
  return { label: 'competitive', tone: 'good' };
}

export default function Industry({ data }) {
  if (!data) return null;
  const classes = Object.keys(data.size || {});

  const sizeChart = useMemo(() => {
    const labels = [...new Set(Object.values(data.size || {}).flat().map((p) => p.fy))].sort();
    return {
      labels,
      datasets: classes.map((c) => {
        const by = Object.fromEntries((data.size[c] || []).map((p) => [p.fy, p.v]));
        return {
          label: CLS_LABEL[c] || c,
          data: labels.map((f) => (f in by ? by[f] / 1e12 : null)),
          borderColor: CLS_COLOR[c], backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, tension: 0.3, spanGaps: false,
        };
      }),
    };
  }, [data, classes]);

  const hhiChart = useMemo(() => {
    const labels = [...new Set(Object.values(data.concentration || {}).flat().map((p) => p.fy))].sort();
    return {
      labels,
      datasets: classes.map((c) => {
        const by = Object.fromEntries((data.concentration[c] || []).map((p) => [p.fy, p.hhi]));
        return {
          label: CLS_LABEL[c] || c,
          data: labels.map((f) => (f in by ? by[f] : null)),
          borderColor: CLS_COLOR[c], backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, tension: 0.3, spanGaps: false,
        };
      }),
    };
  }, [data, classes]);

  const conductChart = useMemo(() => ({
    labels: (data.conduct || []).map((p) => p.fy),
    datasets: [{
      label: 'Grievances per lakh policies',
      data: (data.conduct || []).map((p) => p.per_lakh),
      borderColor: '#b91c1c', backgroundColor: 'rgba(185,28,28,0.10)',
      borderWidth: 2.5, pointRadius: 0, tension: 0.3, fill: true,
    }],
  }), [data]);

  const channelChart = useMemo(() => ({
    labels: (data.channel_mix || []).map((c) => c.name),
    datasets: [{
      data: (data.channel_mix || []).map((c) => c.pct),
      backgroundColor: '#1e3a8a', borderRadius: 4, barThickness: 16,
    }],
  }), [data]);

  const lineOpts = (yLabel) => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: true, position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
    scales: {
      x: { grid: { display: false }, ticks: { font: { size: 10 } } },
      y: { title: { display: !!yLabel, text: yLabel, font: { size: 10 } }, ticks: { font: { size: 10 } } },
    },
  });

  const latestConduct = (data.conduct || []).slice(-1)[0];
  const prevConduct = (data.conduct || []).slice(-2)[0];

  return (
    <div className="ind">
      <div className="ind-kindhead">By licence class</div>
      <div className="ind-cards">
        {classes.filter((c) => (data.segment_kind || {})[c] !== 'line').map((c) => {
          const ser = data.size[c] || [];
          const last = ser[ser.length - 1];
          const cagr = data.growth_cagr?.[c];
          return (
            <div key={c} className="ind-card" style={{ borderLeftColor: CLS_COLOR[c] }}>
              <div className="ind-card-t">{CLS_LABEL[c] || c}</div>
              <div className="ind-card-v">{last ? lakhCr(last.v) : '—'}</div>
              <div className="ind-card-s">
                {last?.fy} · <strong>{cagr}%</strong> CAGR
                <span className="ind-basis">basis: {data.premium_basis?.[c]}</span>
              </div>
            </div>
          );
        })}
      </div>

      {classes.some((c) => (data.segment_kind || {})[c] === 'line') && (
        <>
          <div className="ind-kindhead">
            By line of business
            <span>cuts across classes — every general insurer writes health too</span>
          </div>
          <div className="ind-cards">
            {classes.filter((c) => (data.segment_kind || {})[c] === 'line').map((c) => {
              const ser = data.size[c] || [];
              const last = ser[ser.length - 1];
              return (
                <div key={c} className="ind-card is-line" style={{ borderLeftColor: CLS_COLOR[c] }}>
                  <div className="ind-card-t">{CLS_LABEL[c] || c}</div>
                  <div className="ind-card-v">{last ? lakhCr(last.v) : '—'}</div>
                  <div className="ind-card-s">
                    {last?.fy} · <strong>{data.growth_cagr?.[c]}%</strong> CAGR
                    <span className="ind-basis">basis: {data.premium_basis?.[c]}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      <div className="ind-note">
        Each class uses its <strong>own premium basis</strong> and the classes are deliberately
        not summed — no single premium line is comparable across Life, General and Standalone
        Health, so an "industry total" here would be arithmetic on incompatible definitions.
        {' '}Line segments are a different cut of the same market — the Health <em>line</em>
        (₹1.27 L Cr, competitive) is over three times the Standalone Health <em>class</em>
        (₹0.38 L Cr, highly concentrated), because every general insurer writes health.
        Never add a class segment to a line segment.
      </div>

      <div className="ind-panels">
        <section className="ind-panel">
          <h3>Market size by class</h3>
          <div className="ind-chart"><Line data={sizeChart} options={lineOpts('₹ lakh crore')} /></div>
        </section>

        <section className="ind-panel">
          <h3>
            Market concentration (HHI)
            <span className="ind-sub">sum of squared shares · &lt;1500 competitive · &gt;2500 highly concentrated</span>
          </h3>
          <div className="ind-chart"><Line data={hhiChart} options={lineOpts('HHI')} /></div>
          <div className="ind-hhi-now">
            {classes.map((c) => {
              const ser = data.concentration?.[c] || [];
              const last = ser[ser.length - 1];
              if (!last) return null;
              const b = hhiBand(last.hhi);
              return (
                <span key={c} className={`ind-hhi tone-${b.tone}`}>
                  <i style={{ background: CLS_COLOR[c] }} />
                  {CLS_LABEL[c] || c}: <strong>{last.hhi}</strong> {b.label}
                  <em>top-5 {last.top5}% of {last.n}</em>
                </span>
              );
            })}
          </div>
        </section>

        <section className="ind-panel">
          <h3>
            Industry conduct
            <span className="ind-sub">grievances per lakh policies — size-normalised</span>
          </h3>
          <div className="ind-chart"><Line data={conductChart} options={lineOpts('per lakh')} /></div>
          {latestConduct && prevConduct && (
            <div className={`ind-delta ${latestConduct.per_lakh > prevConduct.per_lakh ? 'worse' : 'better'}`}>
              {latestConduct.fy}: <strong>{latestConduct.per_lakh}</strong> per lakh policies
              {' '}({latestConduct.per_lakh > prevConduct.per_lakh ? '▲' : '▼'}{' '}
              {Math.abs(((latestConduct.per_lakh - prevConduct.per_lakh) / prevConduct.per_lakh) * 100).toFixed(1)}%
              {' '}vs {prevConduct.fy}) · {Math.round(latestConduct.reported).toLocaleString('en-IN')} grievances
            </div>
          )}
        </section>

        <section className="ind-panel">
          <h3>
            Distribution channel mix
            <span className="ind-sub">share of premium, {data.channel_fy}</span>
          </h3>
          <div className="ind-chart">
            <Bar data={channelChart} options={{
              indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: false,
              plugins: { legend: { display: false },
                tooltip: { callbacks: { label: (c) => `${c.parsed.x}%` } } },
              scales: { x: { ticks: { font: { size: 10 }, callback: (v) => `${v}%` } },
                y: { grid: { display: false }, ticks: { font: { size: 10 } } } },
            }} />
          </div>
        </section>
      </div>
    </div>
  );
}
