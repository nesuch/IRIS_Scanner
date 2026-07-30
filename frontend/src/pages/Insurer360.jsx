// INSURER 360 — everything IRIS holds on one insurer, on one screen.
//
// Two supervisory rules the UI must never break, both enforced server-side but
// surfaced here so the reader can see them:
//   * peer context is ALWAYS within the insurer's own class (Life/General/SAHI/
//     Reinsurer/FRB). Only 23 of 957 metrics are reported by every class, so a
//     cross-class rank would be meaningless.
//   * a series the handbook stopped publishing is marked STALE rather than shown
//     as if it were current.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

// No cap on how many insurers can be compared. Partial rows are shown with a
// coverage note and the metric column is pinned, so extra columns cost readability
// rather than correctness.
//
// COLOUR_SLOTS is a different limit: only eight validated categorical colours exist,
// so past eight the trend sparkline would repeat hues and a 12-line sparkline is
// unreadable anyway. Past that the trend column is dropped — the table still carries
// every number, and identity comes from the column header rather than a colour.
const COLOUR_SLOTS = 8;

// 'Specialised' must be listed here. The Picker builds its option list by walking
// this order, so a segment the API returns but this array omits vanishes from the UI —
// which is exactly what happened to AIC and ECGC when Specialised was split out.
const CLASS_ORDER = ['General', 'Life', 'SAHI', 'Specialised', 'Reinsurer', 'FRB'];
const CLASS_LABEL = {
  General: 'General', Life: 'Life', SAHI: 'Standalone Health',
  Specialised: 'Specialised', Reinsurer: 'Reinsurer', FRB: 'Foreign Reinsurance Branch',
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

// Bucket the metric list under its line of business, biggest group first so the main
// statements lead and one-off tables fall to the bottom. Filtering is applied before
// grouping, so a search never leaves an empty heading behind.
function metricGroups(all, q) {
  const s = (q || '').trim().toLowerCase();
  const hit = (all || []).filter((m) => !s
    || m.label.toLowerCase().includes(s)
    || (m.context || '').toLowerCase().includes(s));
  const by = new Map();
  hit.forEach((m) => {
    const g = (m.context || '').split('\u00b7')[0].trim() || 'Other';
    if (!by.has(g)) by.set(g, []);
    by.get(g).push(m);
  });
  return [...by.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
}

export default function Insurer360() {
  const [classes, setClasses] = useState({});
  const [sel, setSel] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState('single');     // 'single' | 'compare'
  const [cmpIds, setCmpIds] = useState([]);
  const [cmpFy, setCmpFy] = useState('');   // '' = let each row pick its best year
  const [cmpExtra, setCmpExtra] = useState([]);   // metric_ids added beyond the KPI set
  // A colour slot per insurer, held for as long as it stays selected. Without this
  // the colour came from position, so removing one repainted every insurer below it.
  // A ref, not state: it is written during the same event that calls setCmpIds, and
  // the render that follows reads the updated map.
  const slotRef = useRef(new Map());
  const slotOf = useCallback((id) => {
    const m = slotRef.current;
    if (!m.has(id)) m.set(id, [...Array(COLOUR_SLOTS).keys()]
      .find((n) => ![...m.values()].includes(n)) ?? 0);
    return m.get(id);
  }, []);
  const releaseSlot = (id) => { slotRef.current.delete(id); };
  // The combobox offers only what is not already selected, so the list shortens as
  // the comparison grows and you cannot add a duplicate.
  // Quick filters. Built from the registry's own public/private split rather than a
  // hardcoded list of names, so "PSU" cannot drift out of date or quietly include a
  // private insurer. Only ACTIVE insurers are added — a cohort you assemble in one
  // click should be the current market, not a graveyard; closed insurers stay
  // individually selectable.
  const quickSets = useMemo(() => {
    const all = Object.entries(classes).flatMap(([seg, list]) =>
      (list || []).map((i) => ({ ...i, seg })));
    const live = all.filter((i) => !i.status || i.status === 'active');
    const pick = (fn) => live.filter(fn).map((i) => i.id);
    const defs = [
      ['PSU — all', (i) => i.ownership === 'public'],
      ['PSU Life', (i) => i.ownership === 'public' && i.seg === 'Life'],
      ['PSU Non-Life', (i) => i.ownership === 'public' && i.seg === 'General'],
      ['PSU Specialised', (i) => i.ownership === 'public' && i.seg === 'Specialised'],
      ['Private Life', (i) => i.ownership === 'private' && i.seg === 'Life'],
      ['Private General', (i) => i.ownership === 'private' && i.seg === 'General'],
      ['Standalone Health', (i) => i.seg === 'SAHI'],
      ['Reinsurers & FRB', (i) => i.seg === 'Reinsurer' || i.seg === 'FRB'],
    ];
    return defs.map(([label, fn]) => ({ label, ids: pick(fn) })).filter((d) => d.ids.length > 1);
  }, [classes]);

  const applyQuick = (ids) => {
    slotRef.current = new Map();          // a fresh cohort gets fresh colours
    ids.forEach((id) => slotOf(id));
    setCmpIds(ids);
  };

  const cmpAvailable = useMemo(() => Object.fromEntries(
    Object.entries(classes).map(([c, list]) => [c, list.filter((i) => !cmpIds.includes(i.id))])
  ), [classes, cmpIds]);
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
    const qs = cmpIds.map((i) => `id=${encodeURIComponent(i)}`).join('&')
      + (cmpFy ? `&fy=${encodeURIComponent(cmpFy)}` : '')
      + cmpExtra.map((m) => `&m=${encodeURIComponent(m)}`).join('');
    api.get(`/insurer/compare?${qs}`)
      .then((d) => setCmp(d))
      .catch((e) => { setErr(e.message); setCmp(null); })
      .finally(() => setLoading(false));
  }, [mode, cmpIds, cmpFy, cmpExtra]);

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

        {mode === 'compare' && quickSets.length > 0 && (
          <div className="cmp-quick">
            <span className="cmp-quick-label">Quick sets</span>
            {quickSets.map((q) => (
              <button type="button" key={q.label} className="cmp-quick-chip"
                title={`Compare all ${q.ids.length} — replaces the current selection`}
                onClick={() => applyQuick(q.ids)}>
                {q.label}<span className="cmp-quick-n">{q.ids.length}</span>
              </button>
            ))}
          </div>
        )}

        {mode === 'compare' && (
          <div className="cmp-picker">
            {/* Same combobox as single mode. A native <select> of 87 near-identical
                company names is a wall to scroll; this filters as you type. Already
                excludes anything selected, so the list shrinks as you build up. */}
            <Picker classes={cmpAvailable} order={CLASS_ORDER} labels={CLASS_LABEL}
              value="" placeholder="Search insurers to add…"
              onChange={(v) => { if (v && !cmpIds.includes(v)) { slotOf(v); setCmpIds([...cmpIds, v]); } }} />
            {/* Year is the user's choice, not ours. Default keeps the per-row
                best-covered year; picking one compares everybody on that year and
                drops rows nobody filed then. */}
            {cmp?.years?.length > 0 && (
              <select className="input cmp-year" value={cmpFy} aria-label="Comparison year"
                onChange={(e) => setCmpFy(e.target.value)}>
                <option value="">Best year per metric</option>
                {cmp.years.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            )}
            {cmpIds.length > 0 && (
              <button type="button" className="cmp-clear" onClick={() => { slotRef.current = new Map(); setCmpIds([]); }}>
                Clear all
              </button>
            )}
            {cmpIds.map((id) => (
              <span key={id} className="cmp-sel">
                {nameOf(id)}
                <button type="button" aria-label={`Remove ${nameOf(id)}`}
                  onClick={() => { releaseSlot(id); setCmpIds(cmpIds.filter((x) => x !== id)); }}>&times;</button>
              </span>
            ))}
            {cmpIds.length < 2 && <span className="i360-cohort">pick at least two</span>}
          </div>
        )}

        {err && <p className="iris-msg" style={{ color: 'var(--bad)' }}>{err}</p>}
        {loading && <PageLoading />}

        {!loading && mode === 'compare' && <Compare data={cmp} insurers={classes} slotOf={slotOf} colourSlots={COLOUR_SLOTS}
            extra={cmpExtra} onExtraChange={setCmpExtra} />}
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

            {(data.lob_mix || []).length > 1 && (
              <section className="i360-lob">
                <h3>Book by line of business <span>{data.lob_mix[0].fy}</span></h3>
                <div className="i360-lob-bars">
                  {data.lob_mix.map((l) => (
                    <div key={l.lob} className="i360-lob-row">
                      <span className="i360-lob-name">{l.lob}</span>
                      <span className="i360-lob-bar">
                        <span style={{ width: `${l.pct}%` }} />
                      </span>
                      <span className="i360-lob-pct">{l.pct}%</span>
                      <span className="i360-lob-val">{fmt(l.value, 'inr')}</span>
                    </div>
                  ))}
                </div>
                <p className="i360-lob-note">
                  Gross direct premium split. Line-of-business market shares above are computed
                  across every insurer writing that line — a health book competes with general
                  insurers' health portfolios, not only with other standalone health insurers.
                </p>
              </section>
            )}

            {(data.computed || []).length > 0 && (
              <section className="i360-all i360-computed">
                <div className="i360-all-head">
                  <h3>Computed totals <span>{data.computed.length}</span></h3>
                  <p className="i360-computed-note">
                    Not stored — reconstructed from leaf items at query time. The database holds
                    only canonical facts, so every total here shows exactly what it was summed
                    from. A context missing any component is omitted rather than shown as a
                    partial sum.
                  </p>
                </div>
                <div className="i360-all-grid">
                  {data.computed.map((m) => (
                    <MetricCard key={m.id} k={{ ...m, selected_fy: data.selected_fy }}
                      onToast={(t, bad) => (bad ? toast.error(t) : toast.success(t))} />
                  ))}
                </div>
              </section>
            )}

            {(data.all_metrics || []).length > 0 && (
              <section className="i360-all">
                <div className="i360-all-head">
                  <h3>All reported metrics <span>{data.all_metrics.length}</span></h3>
                  <input className="input i360-mq" value={mq} onChange={(e) => setMq(e.target.value)}
                    placeholder="Filter metrics…" aria-label="Filter metrics" />
                </div>
                {/* Grouped under the statement or table each metric came from. Flat and
                    alphabetical, "1 Yr", "13th Month" and "31 To 90 Days" sit together
                    meaning nothing; under "Persistency" and "Death Claim Settlement
                    Duration" they read as what they are. The heading is the line of
                    business, which is the first half of the context already shown on
                    every card, so no new data is needed. */}
                {metricGroups(data.all_metrics, mq).map(([group, items]) => (
                  <div key={group} className="i360-group">
                    <h4 className="i360-group-head">
                      {group}<span>{items.length}</span>
                    </h4>
                    <div className="i360-all-grid">
                      {items.map((m) => (
                        <MetricCard key={m.id} k={{ ...m, selected_fy: data.selected_fy }}
                          onToast={(t, bad) => (bad ? toast.error(t) : toast.success(t))} />
                      ))}
                    </div>
                  </div>
                ))}
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
