import { useMemo } from 'react';
import { Line, Bar } from 'react-chartjs-2';
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

  const datasets = seriesNames.map((name, i) => {
    const color = PALETTE[i % PALETTE.length];
    const data = xVals.map((x) => (grouped[name][x] ?? null));
    return isTime
      ? {
        label: name || metric, data, borderColor: color, backgroundColor: `${color}22`,
        borderWidth: 2, tension: 0.3, pointRadius: 3, pointHoverRadius: 5,
        pointBackgroundColor: '#fff', pointBorderColor: color, spanGaps: true, fill: false,
      }
      : { label: name || metric, data, backgroundColor: color, borderRadius: 4, maxBarThickness: 46 };
  });

  return { metric, isTime, labels: xVals, datasets, unit: unitOf(metric), multi: datasets.length > 1 };
}

function ChartCard({ spec }) {
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: spec.multi, position: 'bottom', labels: { boxWidth: 12, font: { size: 11 }, usePointStyle: true } },
      datalabels: { display: false },
      tooltip: {
        callbacks: {
          label: (c) => `${c.dataset.label}: ${c.parsed.y?.toLocaleString('en-IN') ?? '—'}${spec.unit ? ` ${spec.unit}` : ''}`,
        },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { size: 11 } } },
      y: {
        beginAtZero: false,
        grid: { color: 'rgba(0,0,0,0.05)' },
        ticks: { font: { size: 11 }, callback: (v) => Number(v).toLocaleString('en-IN') },
        title: spec.unit ? { display: true, text: spec.unit, font: { size: 10 } } : undefined,
      },
    },
  };
  const Comp = spec.isTime ? Line : Bar;
  return (
    <div className="chart-card anim-rise">
      <div className="chart-card-head">
        <i className={`fas ${spec.isTime ? 'fa-chart-line' : 'fa-chart-column'}`} />
        <span>{spec.metric}</span>
      </div>
      <div className="chart-card-box">
        <Comp data={{ labels: spec.labels, datasets: spec.datasets }} options={opts} />
      </div>
    </div>
  );
}

export default function ReportCharts({ report, dimension }) {
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
  return <div className="charts-grid">{specs.map((s) => <ChartCard key={s.metric} spec={s} />)}</div>;
}
