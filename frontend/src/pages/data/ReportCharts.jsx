import { useMemo, useState } from 'react';
import { Line, Bar, Pie } from 'react-chartjs-2';
import '../../lib/charts.js';
import { EmptyState } from '../../components/UI.jsx';

const PALETTE = [
  '#1a237e', '#2563eb', '#06b6d4', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ef4444', '#0ea5e9', '#14b8a6', '#a855f7', '#f97316', '#64748b',
];

const num = (v) => {
  if (v == null) return null;
  const n = parseFloat(String(v).replace(/,/g, ''));
  return Number.isFinite(n) ? n : null;
};

// Unit lives in the metric name's trailing parentheses, e.g. "... (₹Crore)".
const unitOf = (metric) => {
  const m = metric.match(/\(([^()]*)\)\s*$/);
  return m ? m[1] : '';
};

const fmt = (v) => (v == null ? '—' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 }));

// Build a chart spec for one metric column out of the pivoted rows.
function buildChart(metric, rows, baseCols, dimension) {
  const years = [...new Set(rows.map((r) => r.Financial_Year).filter(Boolean))].sort();
  const isTime = baseCols.includes('Financial_Year') && years.length >= 2;

  // X axis: years for a time series, else the categorical column that varies most.
  let xCol;
  if (isTime) {
    xCol = 'Financial_Year';
  } else {
    const cat = baseCols.filter((c) => c !== 'Financial_Year');
    xCol = cat
      .map((c) => [c, new Set(rows.map((r) => r[c])).size])
      .sort((a, b) => b[1] - a[1])[0]?.[0] || dimension;
  }

  const xVals = isTime ? years
    : [...new Set(rows.map((r) => r[xCol]).filter((v) => v != null))].sort();

  // Series = the other base columns that actually vary (e.g. one line per sector).
  const seriesCols = baseCols.filter(
    (c) => c !== xCol && new Set(rows.map((r) => r[c])).size > 1,
  );
  const seriesKey = (r) => seriesCols.map((c) => r[c]).join(' · ');

  const grouped = {};
  rows.forEach((r) => {
    const k = seriesCols.length ? seriesKey(r) : '';
    (grouped[k] ||= {})[r[xCol]] = num(r[metric]);
  });
  const seriesNames = Object.keys(grouped);
  if (!seriesNames.some((s) => Object.values(grouped[s]).some((v) => v != null))) return null;

  const series = seriesNames.map((name) => ({
    name: name || metric,
    data: xVals.map((x) => (grouped[name][x] ?? null)),
  }));

  return { metric, isTime, labels: xVals, series, unit: unitOf(metric), multi: series.length > 1 };
}

// Turn a spec into Chart.js datasets for the chosen chart type.
function toChartData(spec, type, showLabels) {
  const { series, labels, unit } = spec;
  if (type === 'pie') {
    // One slice per series (latest value) when multi-series, else per category.
    const pieLabels = spec.multi ? series.map((s) => s.name) : labels;
    const pieVals = spec.multi
      ? series.map((s) => [...s.data].reverse().find((v) => v != null) ?? 0)
      : series[0].data.map((v) => v ?? 0);
    return {
      labels: pieLabels,
      datasets: [{
        data: pieVals,
        backgroundColor: pieLabels.map((_, i) => PALETTE[i % PALETTE.length]),
        borderColor: '#fff', borderWidth: 2,
      }],
    };
  }
  const datasets = series.map((s, i) => {
    const color = PALETTE[i % PALETTE.length];
    if (type === 'line') {
      return {
        label: s.name, data: s.data, borderColor: color, backgroundColor: `${color}22`,
        borderWidth: 2, tension: 0.3, pointRadius: 3, pointHoverRadius: 5,
        pointBackgroundColor: '#fff', pointBorderColor: color, spanGaps: true, fill: false,
      };
    }
    // Bar: a single series gets a distinct colour per category for readability.
    const bg = spec.multi ? color : labels.map((_, j) => PALETTE[j % PALETTE.length]);
    return { label: s.name, data: s.data, backgroundColor: bg, borderRadius: 4, maxBarThickness: 46 };
  });
  return { labels, datasets };
}

function ChartCard({ spec, type, showLabels }) {
  const effType = type === 'auto' ? (spec.isTime ? 'line' : 'bar') : type;
  const isPie = effType === 'pie';
  const data = toChartData(spec, effType, showLabels);

  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: isPie ? undefined : { mode: 'index', intersect: false },
    plugins: {
      legend: {
        display: isPie || spec.multi,
        position: 'bottom',
        labels: {
          usePointStyle: true, pointStyle: 'circle',
          boxWidth: 7, boxHeight: 7, padding: 12, font: { size: 10.5 },
        },
      },
      datalabels: showLabels ? {
        display: (c) => (isPie ? true : c.dataset.data[c.dataIndex] != null),
        color: isPie ? '#fff' : '#334155',
        anchor: isPie ? 'center' : 'end',
        align: isPie ? 'center' : 'end',
        font: { size: 9.5, weight: '600' },
        formatter: (v) => fmt(v),
      } : { display: false },
      tooltip: {
        callbacks: {
          label: (c) => {
            const v = isPie ? c.parsed : c.parsed.y;
            return `${c.dataset.label || c.label}: ${fmt(v)}${spec.unit ? ` ${spec.unit}` : ''}`;
          },
        },
      },
    },
    scales: isPie ? undefined : {
      x: { grid: { display: false }, ticks: { font: { size: 11 } } },
      y: {
        beginAtZero: effType === 'bar',  // so small bars are visible against larger ones
        grid: { color: 'rgba(0,0,0,0.05)' },
        ticks: { font: { size: 11 }, callback: (v) => Number(v).toLocaleString('en-IN') },
        title: spec.unit ? { display: true, text: spec.unit, font: { size: 10 } } : undefined,
      },
    },
  };
  const Comp = isPie ? Pie : effType === 'bar' ? Bar : Line;
  const icon = isPie ? 'fa-chart-pie' : effType === 'bar' ? 'fa-chart-column' : 'fa-chart-line';
  return (
    <div className="chart-card anim-rise">
      <div className="chart-card-head">
        <i className={`fas ${icon}`} />
        <span>{spec.metric}</span>
      </div>
      <div className="chart-card-box">
        <Comp data={data} options={opts} />
      </div>
    </div>
  );
}

const TYPES = [
  ['auto', 'fa-wand-magic-sparkles', 'Auto'],
  ['line', 'fa-chart-line', 'Line'],
  ['bar', 'fa-chart-column', 'Bar'],
  ['pie', 'fa-chart-pie', 'Pie'],
];

export default function ReportCharts({ report, dimension }) {
  const [type, setType] = useState('auto');
  const [showLabels, setShowLabels] = useState(false);

  const specs = useMemo(() => {
    const cols = report?.columns || [];
    const rows = report?.rows || [];
    if (!rows.length) return [];
    const baseOrder = [dimension, 'Financial_Year', 'Line_of_Business', 'Class_of_Business', 'Quarter'];
    const baseCols = baseOrder.filter((c) => cols.includes(c));
    const metricCols = cols.filter((c) => !baseCols.includes(c) && c !== 'Source_File');
    return metricCols.map((m) => buildChart(m, rows, baseCols, dimension)).filter(Boolean);
  }, [report, dimension]);

  if (!specs.length) {
    return <EmptyState icon="fa-chart-area">No chartable metrics in this report.</EmptyState>;
  }
  return (
    <>
      <div className="chart-toolbar">
        <div className="chart-type-group">
          {TYPES.map(([t, icon, label]) => (
            <button
              key={t}
              className={`chart-type-btn ${type === t ? 'active' : ''}`}
              onClick={() => setType(t)}
              title={label}
            >
              <i className={`fas ${icon}`} /> {label}
            </button>
          ))}
        </div>
        <label className="chart-labels-toggle">
          <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
          Data labels
        </label>
      </div>
      <div className="charts-grid">
        {specs.map((s) => <ChartCard key={s.metric} spec={s} type={type} showLabels={showLabels} />)}
      </div>
    </>
  );
}
