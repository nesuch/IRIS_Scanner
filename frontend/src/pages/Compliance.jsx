import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { EmptyState, PageLoading } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { api } from '../api.js';
import './compliance/compliance.css';

function parseAlert(msg) {
  const [title, rest = ''] = msg.split(/: (.+)/s);
  const [entity, detail = ''] = rest ? rest.split(/ - (.+)/s) : [''];
  return { title, entity, detail };
}

// Resolve the card's display status + styling for the active filter mode,
// mirroring the original filterMode() reclassification.
function cardView(co, mode) {
  const crit = co.has_critical, warn = co.has_warning;
  if (mode === 'critical') return { status: 'VIOLATION', cls: 'violation', showCrit: true, showWarn: false };
  if (mode === 'warning') return { status: 'WATCHLIST', cls: 'watchlist', showCrit: false, showWarn: true };
  if (crit && warn) return { status: 'VIOLATION & WATCH', cls: 'mixed', showCrit: true, showWarn: true };
  return { status: co.status, cls: co.status.toLowerCase(), showCrit: true, showWarn: true };
}

const num = (v) => (v === 'N/A' || v == null ? null : Number(v));
const inr = (v) => (v === 'N/A' || v == null ? 'N/A' : Number(v).toLocaleString('en-IN'));
const pct = (v) => (v === 'N/A' || v == null ? 'N/A' : `${v}%`);
const GROUPS = ['All', 'PSU', 'Private', 'Life', 'General', 'Health (SAHI)', 'Reinsurance'];

export default function Compliance() {
  const toast = useToast();
  const [data, setData] = useState(null);
  const [year, setYear] = useState('');
  const [mode, setMode] = useState('all');
  const [group, setGroup] = useState('All');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.get(`/compliance${year ? `?year=${encodeURIComponent(year)}` : ''}`)
      .then(setData)
      .catch((e) => toast.error(e.message || 'Could not load compliance data'))
      .finally(() => setLoading(false));
  }, [year]);

  const companies = (data?.companies || []).filter((co) => {
    if (mode === 'critical' && !co.has_critical) return false;
    if (mode === 'warning' && !co.has_warning) return false;
    if (group !== 'All' && !(co.tags || []).includes(group)) return false;
    return true;
  });

  return (
    <>
      <PageHeader fullForm="Regulatory Oversight" title="Compliance Status" scope="Live Monitoring: Solvency, EoM & Ratios" />
      <div className="page-body">
        <div className="controls-row">
          <div className="filter-group">
            <button className={`comp-filter ${mode === 'all' ? 'active' : ''}`} onClick={() => setMode('all')}>All View</button>
            <button className={`comp-filter danger ${mode === 'critical' ? 'active' : ''}`} onClick={() => setMode('critical')}>Violations Only</button>
            <button className={`comp-filter ${mode === 'warning' ? 'active' : ''}`} onClick={() => setMode('warning')}>Watchlist Only</button>
          </div>
          <select className="year-select" value={year} onChange={(e) => setYear(e.target.value)}>
            <option value="">Latest Available</option>
            {(data?.years || []).map((y) => <option key={y} value={y}>FY {y}</option>)}
          </select>
        </div>
        <div className="comp-groups">
          {GROUPS.map((g) => (
            <button key={g} className={`comp-chip ${group === g ? 'active' : ''}`} onClick={() => setGroup(g)}>{g}</button>
          ))}
        </div>

        {loading ? <PageLoading label="Loading compliance data…" />
          : companies.length === 0 ? <EmptyState icon="fa-shield-halved">No companies match this view.</EmptyState>
          : (
            <div className="compliance-grid stagger">
              {companies.map((co, i) => {
                const v = cardView(co, mode);
                const sol = num(co.metrics.solvency), exp = num(co.metrics.expenses),
                  comb = num(co.metrics.combined), clm = num(co.metrics.claims);
                const alerts = (co.alerts || []).filter((a) => (a.level === 'critical' ? v.showCrit : v.showWarn));
                return (
                  <div key={i} className={`company-card border-${v.cls}`}>
                    <div className="card-header">
                      <div className="card-title" title={co.name}>
                        {co.name}
                        {co.group && <span className="card-group">{co.group}</span>}
                      </div>
                      <span className={`status-badge status-${v.cls}`}>{v.status}</span>
                    </div>
                    <div className="metrics-grid">
                      <Metric val={co.metrics.solvency} label="Solvency" cls={sol != null && sol < 1.5 ? 'm-violation' : 'm-primary'} />
                      <Metric val={pct(co.metrics.expenses)} label="EoM" cls={exp != null && exp > 35 ? 'm-violation' : 'm-primary'} />
                      <Metric val={pct(co.metrics.combined)} label="Comb. Ratio" cls={comb != null && comb > 100 ? 'm-watchlist' : 'm-dark'} />
                      <Metric val={pct(co.metrics.claims)} label="Claims Ratio" cls={clm != null && clm > 90 ? 'm-watchlist' : 'm-dark'} />
                      <Metric val={inr(co.metrics.premium)} label="GDP (₹Cr)" cls="m-dark" />
                      <Metric val={inr(co.metrics.underwriting)} label="U/W P&L (₹Cr)" cls={num(co.metrics.underwriting) != null && num(co.metrics.underwriting) < 0 ? 'm-watchlist' : 'm-dark'} />
                    </div>
                    <div className="alert-area">
                      {alerts.length === 0 ? (
                        <div className="all-clear"><i className="fas fa-circle-check" />All key metrics within limits.</div>
                      ) : alerts.map((a, ai) => {
                        const p = parseAlert(a.msg);
                        return (
                          <div key={ai} className={`comp-alert ${a.level === 'critical' ? 'a-crit' : 'a-warn'}`}>
                            <i className={`fas ${a.level === 'critical' ? 'fa-circle-xmark' : 'fa-circle-exclamation'}`} />
                            <span><span className="a-title">{p.title}:</span> {p.entity && <span className="a-entity">{p.entity}</span>} {p.detail && <span className="a-detail">- {p.detail}</span>}</span>
                          </div>
                        );
                      })}
                    </div>
                    <div className="update-tag">Data: {co.last_updated}</div>
                  </div>
                );
              })}
            </div>
          )}
      </div>
    </>
  );
}

function Metric({ val, label, cls }) {
  return (
    <div className="metric-cell">
      <span className={`metric-val ${cls}`}>{val}</span>
      <span className="metric-label">{label}</span>
    </div>
  );
}
