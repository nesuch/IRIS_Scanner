// INSURER 360 — everything IRIS holds on one insurer, on one screen.
//
// Two supervisory rules the UI must never break, both enforced server-side but
// surfaced here so the reader can see them:
//   * peer context is ALWAYS within the insurer's own class (Life/General/SAHI/
//     Reinsurer/FRB). Only 23 of 957 metrics are reported by every class, so a
//     cross-class rank would be meaningless.
//   * a series the handbook stopped publishing is marked STALE rather than shown
//     as if it were current.
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import '../lib/charts.js';
import PageHeader from '../components/PageHeader.jsx';
import { PageLoading } from '../components/UI.jsx';
import { api } from '../api.js';
import Compare from './insurer/Compare.jsx';
import Industry from './insurer/Industry.jsx';
import Exceptions from './insurer/Exceptions.jsx';
import Picker from './insurer/Picker.jsx';
import MetricCard, { fmtVal } from './insurer/MetricCard.jsx';
import { useToast } from '../components/Toast.jsx';
import './insurer/insurer360.css';

const CLASS_ORDER = ['General', 'Life', 'SAHI', 'Reinsurer', 'FRB'];
const CLASS_LABEL = {
  General: 'General', Life: 'Life', SAHI: 'Standalone Health',
  Reinsurer: 'Reinsurer', FRB: 'Foreign Reinsurance Branch',
};

// ₹ values are stored in base units (paise-free rupees); Crore is how supervisors
// read them. Counts and ratios pass through with their own formatting.
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

// Percentile is only meaningful with a real cohort behind it, and its *colour*
// depends on the metric's direction — being top of the table for grievances is
// bad news, so we never colour by rank alone.
// YoY direction must respect the metric, not the arithmetic sign. Operating
// expenses falling 27% is GOOD news; colouring it red because the number went
// down would misread the page at a glance. Context-only metrics stay neutral.
function yoyTone(yoy, higherIsBetter) {
  if (higherIsBetter === null || higherIsBetter === undefined) return 'neutral';
  const improving = higherIsBetter ? yoy >= 0 : yoy < 0;
  return improving ? 'up' : 'down';
}

function pctTone(pct, higherIsBetter) {
  if (pct === null || pct === undefined || higherIsBetter === null) return '';
  const good = higherIsBetter ? pct >= 60 : pct <= 40;
  const bad = higherIsBetter ? pct <= 25 : pct >= 75;
  return good ? 'tone-good' : bad ? 'tone-bad' : '';
}

function Spark({ trend, unit }) {
  const data = useMemo(() => ({
    labels: trend.map((p) => p.fy),
    datasets: [{
      data: trend.map((p) => (unit === 'inr' ? p.v / 1e7 : p.v)),
      borderColor: 'rgba(30,58,138,0.9)',
      backgroundColor: 'rgba(30,58,138,0.10)',
      borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
    }],
  }), [trend, unit]);
  const opts = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
    scales: { x: { display: false }, y: { display: false } },
    animation: false,
  }), []);
  if (!trend || trend.length < 2) return <div className="i360-spark-empty">single year</div>;
  return <div className="i360-spark"><Line data={data} options={opts} /></div>;
}

export default function Insurer360() {
  const [classes, setClasses] = useState({});
  const [sel, setSel] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState('single');     // 'single' | 'compare'
  const [cmpIds, setCmpIds] = useState([]);       // up to 4
  const [cmp, setCmp] = useState(null);
  const [ind, setInd] = useState(null);
  const [exc, setExc] = useState(null);
  const [fy, setFy] = useState('');          // '' = latest
  const [mq, setMq] = useState('');          // all-metrics filter
  const toast = useToast();

  useEffect(() => {
    api.get('/insurer/list')
      .then((d) => {
        setClasses(d.classes || {});
        const first = (d.classes?.General || [])[0];
        if (first) setSel(first.id);
      })
      .catch((e) => setErr(e.message));
  }, []);

  const load = useCallback((id, year) => {
    if (!id) return;
    setLoading(true); setErr(null);
    api.get(`/insurer/${encodeURIComponent(id)}${year ? `?fy=${encodeURIComponent(year)}` : ''}`)
      .then((d) => setData(d))
      .catch((e) => { setErr(e.message); setData(null); })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { if (mode === 'single') load(sel, fy); }, [sel, fy, load, mode]);

  useEffect(() => {
    if (mode !== 'compare' || cmpIds.length < 2) { setCmp(null); return; }
    setLoading(true); setErr(null);
    const qs = cmpIds.map((i) => `id=${encodeURIComponent(i)}`).join('&');
    api.get(`/insurer/compare?${qs}`)
      .then((d) => setCmp(d))
      .catch((e) => { setErr(e.message); setCmp(null); })
      .finally(() => setLoading(false));
  }, [mode, cmpIds]);

  useEffect(() => {
    if (mode !== 'industry' || ind) return;
    setLoading(true); setErr(null);
    api.get('/insurer/industry')
      .then((d) => setInd(d))
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
    // Industry data is year-independent, so fetch once per mount.
  }, [mode]);

  useEffect(() => {
    if (mode !== 'alerts') return;
    setLoading(true); setErr(null);
    api.get(`/insurer/exceptions${fy ? `?fy=${encodeURIComponent(fy)}` : ''}`)
      .then((d) => setExc(d))
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
    // `exc` must NOT be a dependency: this effect SETS it, so including it makes
    // the effect retrigger itself forever and the list never settles.
  }, [mode, fy]);

  const nameOf = useCallback((id) => {
    for (const c of CLASS_ORDER) {
      const hit = (classes[c] || []).find((x) => x.id === id);
      if (hit) return hit.name;
    }
    return id;
  }, [classes]);

  return (
    <>
      <PageHeader fullForm="Supervisory Intelligence" title="Insurer 360"
        scope={mode === 'alerts' ? 'Exception worklist'
          : mode === 'industry' ? 'Market-level trends'
          : mode === 'compare'
          ? (cmpIds.length >= 2
              ? `Comparing ${cmpIds.length} insurers`
              : 'Comparison — pick at least two')
          : (data ? `${data.insurer.name} · ${CLASS_LABEL[data.insurer.class] || data.insurer.class}`
                  : 'Select an insurer')} />

      <div className="page-body i360">
        <div className="i360-picker">
          {mode === 'single' && <label>Insurer</label>}
          {mode === 'single' && (
            <Picker classes={classes} order={CLASS_ORDER} labels={CLASS_LABEL}
              value={sel} onChange={setSel} placeholder="Select an insurer…" />
          )}
          {(mode === 'single' || mode === 'alerts') && (data?.years?.length || fy) && (
            <span className="i360-year">
              <label htmlFor="i360-fy">Year</label>
              <select id="i360-fy" className="input" value={fy}
                onChange={(e) => { setFy(e.target.value); setExc(null); }}>
                <option value="">Latest</option>
                {(data?.years || []).slice().reverse().map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </span>
          )}
          {mode === 'single' && data && (
            <span className="i360-cohort">
              compared against <strong>{data.cohort_size}</strong> {CLASS_LABEL[data.insurer.class] || data.insurer.class} insurers
            </span>
          )}
          <span className="i360-modes">
            <button type="button" className={mode === 'single' ? 'on' : ''}
              onClick={() => setMode('single')}>Single</button>
            <button type="button" className={mode === 'compare' ? 'on' : ''}
              onClick={() => setMode('compare')}>Compare</button>
            <button type="button" className={mode === 'industry' ? 'on' : ''}
              onClick={() => setMode('industry')}>Industry</button>
            <button type="button" className={mode === 'alerts' ? 'on' : ''}
              onClick={() => setMode('alerts')}>Alerts</button>
          </span>
        </div>

        {mode === 'compare' && (
          <div className="cmp-picker">
            <select className="input" value="" aria-label="Add insurer to comparison"
              onChange={(e) => {
                const v = e.target.value;
                if (v && !cmpIds.includes(v) && cmpIds.length < 4) setCmpIds([...cmpIds, v]);
              }}>
              <option value="">Add insurer…{cmpIds.length >= 4 ? ' (max 4)' : ''}</option>
              {CLASS_ORDER.filter((c) => (classes[c] || []).length).map((c) => (
                <optgroup key={c} label={`${CLASS_LABEL[c] || c} (${classes[c].length})`}>
                  {classes[c].filter((i) => !cmpIds.includes(i.id))
                    .map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}
                </optgroup>
              ))}
            </select>
            {cmpIds.map((id) => (
              <span key={id} className="cmp-sel">
                {nameOf(id)}
                <button type="button" aria-label={`Remove ${nameOf(id)}`}
                  onClick={() => setCmpIds(cmpIds.filter((x) => x !== id))}>&times;</button>
              </span>
            ))}
            {cmpIds.length < 2 && <span className="i360-cohort">pick at least two</span>}
          </div>
        )}

        {err && <p className="iris-msg" style={{ color: 'var(--bad)' }}>{err}</p>}
        {loading && <PageLoading />}

        {!loading && mode === 'compare' && <Compare data={cmp} insurers={classes} />}
        {!loading && mode === 'industry' && <Industry data={ind} />}
        {!loading && mode === 'alerts' && (
          <Exceptions data={exc} onSelect={(id) => { setSel(id); setMode('single'); }} />
        )}

        {!loading && mode === 'single' && data && (
          <>
            {data.derived?.length > 0 && (
              <div className="i360-derived">
                {data.derived.map((d, i) => (
                  <div key={i} className={`i360-dcard ${d.stale ? 'is-stale' : ''}`}>
                    <div className="i360-dlabel">{d.label}</div>
                    <div className="i360-dvalue">
                      {d.value === null ? <span className="i360-na">n/a</span> : fmt(d.value, d.unit)}
                    </div>
                    <div className="i360-dnote">
                      <span className="i360-fy">{d.fy}</span>
                      {d.stale && <span className="i360-stale">STALE</span>}
                      {d.note && <span> {d.note}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="i360-grid">
              {data.kpis.map((k) => (
                <MetricCard key={k.id} k={{ ...k, selected_fy: data.selected_fy }}
                  onToast={(m, bad) => (bad ? toast.error(m) : toast.success(m))}>
                  <div className="i360-sub">
                    {k.yoy_pct !== null && (
                      <span className={`i360-yoy ${yoyTone(k.yoy_pct, k.higher_is_better)}`}
                        title={k.higher_is_better === null ? 'context metric — no better/worse direction' : undefined}>
                        <i className={`fas fa-arrow-${k.yoy_pct >= 0 ? 'up' : 'down'}`} /> {Math.abs(k.yoy_pct)}% YoY
                      </span>
                    )}
                    {k.percentile !== null && (
                      <span className={`i360-pct ${pctTone(k.percentile, k.higher_is_better)}`}
                        title={`Higher than ${k.percentile}% of the ${k.peer_n} peers in this class`}>
                        p{k.percentile} of {k.peer_n}
                      </span>
                    )}
                    {k.peer_median !== null && (
                      <span className="i360-med-inline">med {fmtVal(k.peer_median, k.unit)}</span>
                    )}
                  </div>
                </MetricCard>
              ))}
            </div>

            {(data.all_metrics || []).length > 0 && (
              <section className="i360-all">
                <div className="i360-all-head">
                  <h3>All reported metrics <span>{data.all_metrics.length}</span></h3>
                  <input className="input i360-mq" value={mq} onChange={(e) => setMq(e.target.value)}
                    placeholder="Filter metrics…" aria-label="Filter metrics" />
                </div>
                <div className="i360-all-grid">
                  {data.all_metrics
                    .filter((m) => !mq || m.label.toLowerCase().includes(mq.toLowerCase()))
                    .map((m) => (
                      <MetricCard key={m.id} k={{ ...m, selected_fy: data.selected_fy }}
                        onToast={(t, bad) => (bad ? toast.error(t) : toast.success(t))} />
                    ))}
                </div>
              </section>
            )}

            <p className="i360-foot">
              Peer percentiles are computed within <strong>{CLASS_LABEL[data.insurer.class] || data.insurer.class}</strong> only —
              metrics are not comparable across insurer classes. Figures are from the IRDAI Handbook
              ({data.years?.[0]}–{data.years?.[data.years.length - 1]}).
            </p>
          </>
        )}
      </div>
    </>
  );
}
