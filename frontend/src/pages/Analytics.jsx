import { useEffect, useState } from 'react';
import { Doughnut, Line } from 'react-chartjs-2';
import '../lib/charts.js';
import PageHeader from '../components/PageHeader.jsx';
import { PageLoading, StatCard } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { api } from '../api.js';
import './analytics/analytics.css';

const DOUGHNUT_COLORS = ['#283593', '#2563eb', '#06b6d4', '#22c55e', '#f59e0b', '#8b5cf6'];

export default function Analytics() {
  const toast = useToast();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    api.get('/analytics').then(setData)
      .catch((e) => toast.error(e.message || 'Could not load analytics'))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  async function clearLogs() {
    if (!window.confirm('Clear all recorded crashes?')) return;
    try { await api.post('/clear-logs'); toast.success('Crashes cleared'); load(); }
    catch (e) { toast.error(e.message || 'Clear failed'); }
  }

  if (loading) return (<><PageHeader fullForm="System Health Monitoring" title="IRIS Analytics" scope="Live Traffic & Error Tracking" /><div className="page-body"><PageLoading /></div></>);

  const { stats, chart, monthly, yearly, logs, can_view_crash_details } = data;
  const hasTraffic = (chart.data || []).some((v) => v > 0);
  const hasMonthly = (monthly.data || []).some((v) => v > 0);

  return (
    <>
      <PageHeader fullForm="System Health Monitoring" title="IRIS Analytics" scope="Live Traffic & Error Tracking" />
      <div className="page-body">
        <div className="stat-grid" style={{ marginBottom: 22 }}>
          <StatCard label="Unique Users" value={stats.users} icon="fa-users" />
          <StatCard label="Total Requests" value={stats.total} icon="fa-arrow-trend-up" />
          <StatCard label="System Crashes" value={stats.errors} icon="fa-bug" tone={stats.errors > 0 ? 'bad' : 'good'} />
        </div>

        <div className="charts-row">
          <div className="card pad chart-wrapper" style={{ flex: 2 }}>
            <h4 className="chart-title"><i className="fas fa-chart-line" /> Monthly Traffic</h4>
            <div className="chart-box">
              {hasMonthly ? (
                <Line
                  data={{
                    labels: monthly.labels,
                    datasets: [{
                      label: 'Requests', data: monthly.data,
                      borderColor: '#1a237e', backgroundColor: 'rgba(37,99,235,0.12)',
                      borderWidth: 2, fill: true, tension: 0.3, pointRadius: 4,
                      pointBackgroundColor: '#fff', pointBorderColor: '#1a237e',
                    }],
                  }}
                  options={{
                    responsive: true, maintainAspectRatio: false,
                    layout: { padding: { top: 25, right: 10, left: 10 } },
                    plugins: {
                      legend: { display: false },
                      datalabels: { display: true, align: 'top', offset: 4, color: '#1a237e', font: { weight: 'bold' }, formatter: (v) => (v > 0 ? v : '') },
                    },
                    scales: { y: { beginAtZero: true, grace: '10%', grid: { color: '#eef2fb' } }, x: { grid: { display: false } } },
                  }}
                />
              ) : <div className="chart-empty">No traffic recorded yet.</div>}
            </div>
          </div>

          <div className="card pad chart-wrapper" style={{ flex: 1 }}>
            <h4 className="chart-title"><i className="fas fa-chart-pie" /> Module Usage</h4>
            <div className="chart-box">
              {hasTraffic ? (
                <Doughnut
                  data={{ labels: chart.labels, datasets: [{ data: chart.data, backgroundColor: DOUGHNUT_COLORS, borderWidth: 0 }] }}
                  options={{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                      legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } },
                      datalabels: {
                        display: true, color: '#fff', font: { weight: 'bold', size: 12 },
                        formatter: (value, ctx) => {
                          if (!value) return '';
                          const sum = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                          return Math.round((value * 100) / sum) + '%';
                        },
                      },
                    },
                  }}
                />
              ) : <div className="chart-empty">No module usage yet.</div>}
            </div>
          </div>
        </div>

        <div className="card pad" style={{ marginBottom: 22 }}>
          <h4 className="chart-title"><i className="fas fa-calendar-days" /> Yearly Overview</h4>
          <table className="data" style={{ marginTop: 8 }}>
            <thead><tr><th>Year</th><th>Total Interactions</th></tr></thead>
            <tbody>
              {yearly.length ? yearly.map((y) => (
                <tr key={y.year}><td>{y.year}</td><td style={{ fontWeight: 600 }}>{y.count}</td></tr>
              )) : <tr><td colSpan={2} style={{ textAlign: 'center', color: 'var(--faint)' }}>No historical data available.</td></tr>}
            </tbody>
          </table>
        </div>

        {stats.errors > 0 && can_view_crash_details ? (
          <div className="error-log">
            <div className="log-header">
              <span><i className="fas fa-bug" /> Recent Failures</span>
              <button className="btn btn-danger btn-sm" onClick={clearLogs}>Clear Crashes</button>
            </div>
            <div className="table-scroll">
              <table className="log-table">
                <thead><tr><th>Time</th><th>Endpoint</th><th>Error Message</th></tr></thead>
                <tbody>
                  {logs.map((l, i) => (
                    <tr key={i}><td>{l.timestamp}</td><td>{l.endpoint}</td><td className="err-msg">{l.error}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : stats.errors === 0 ? (
          <div className="no-failures"><i className="fas fa-circle-check" /><div>No system failures recorded.</div></div>
        ) : null}
      </div>
    </>
  );
}
