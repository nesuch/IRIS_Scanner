import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Modal, EmptyState, TypingDots } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import FlagModal from '../components/FlagModal.jsx';
import ReportCharts from './data/ReportCharts.jsx';
import { api } from '../api.js';
import './data/data.css';

const ENTITY_GROUPS = [
  { name: 'SAHIs', keys: ['Star', 'Care', 'Aditya Birla', 'Niva', 'Manipal', 'Galaxy', 'Narayana'] },
  { name: 'PSUs', keys: ['New India', 'United India', 'Oriental', 'National'] },
  { name: 'Life', keys: ['LIC', 'HDFC Life', 'SBI Life', 'ICICI Prudential'] },
];

// Filter steps, in the cascade order the user picks them. Each step is scoped
// to the selections made in the steps before it (see combosMatching below).
const CATEGORIES = [
  ['entities', 'Entity'], ['lobs', 'Line of Biz'], ['classes', 'Class of Biz'],
  ['metrics', 'Metric'], ['years', 'Fin Year'], ['quarters', 'Quarter'],
];
const CASCADE_ORDER = ['entities', 'lobs', 'classes', 'metrics', 'years', 'quarters'];
const KEY_TO_COL = {
  entities: 'Entity', lobs: 'Line_of_Business', classes: 'Class_of_Business',
  metrics: 'Metric', years: 'Financial_Year', quarters: 'Quarter',
};
// What the "Entity" step is called per report view.
const entityNoun = (dim) => (dim === 'Insurer' ? 'Insurers' : dim === 'Industry' ? 'Sectors' : `${dim}s`);

const blankFilters = (dim = 'Insurer') => ({
  dimension: dim, entities: [], metrics: [], classes: [], years: [], quarters: [], lobs: [],
});

// Split an EWS message ("Title: Entity - detail") for structured display.
function parseAlert(msg) {
  const [title, rest = ''] = msg.split(/: (.+)/s);
  const [entity, detail = ''] = rest ? rest.split(/ - (.+)/s) : [''];
  return { title, entity, detail };
}

function FilterModal({ options, initial, onApply, onClose }) {
  const toast = useToast();
  const [draft, setDraft] = useState(initial);
  const [cat, setCat] = useState('entities');
  const [search, setSearch] = useState('');

  const entityList = options.entities?.[draft.dimension] || [];
  const dimOpts = options.by_dim?.[draft.dimension] || {};
  const cascade = dimOpts.cascade;   // combo tuple column order
  const combos = dimOpts.combos;     // distinct valid combinations

  const colIndex = (key) => (cascade ? cascade.indexOf(KEY_TO_COL[key]) : -1);

  // Combos consistent with every selection made *before* `uptoKey` in the cascade.
  const combosMatching = (d, uptoKey) => {
    if (!combos) return [];
    const upto = CASCADE_ORDER.indexOf(uptoKey);
    return combos.filter((row) => {
      for (let i = 0; i < upto; i += 1) {
        const k = CASCADE_ORDER[i];
        const sel = d[k];
        const ci = colIndex(k);
        if (sel?.length && ci >= 0 && !sel.includes(row[ci])) return false;
      }
      return true;
    });
  };

  // Valid options for one step, scoped to the earlier steps' selections.
  const optionsForCat = (d, key) => {
    if (!combos) return dimOpts[key] || options[key] || [];
    const ci = colIndex(key);
    if (ci < 0) return dimOpts[key] || [];
    return [...new Set(combosMatching(d, key).map((r) => r[ci]))].sort();
  };

  // After any change, drop downstream selections that are no longer valid.
  const prune = (d) => {
    if (!combos) return d;
    const nd = { ...d };
    CASCADE_ORDER.forEach((key) => {
      if (key === 'entities' || !nd[key]?.length) return;
      const valid = new Set(optionsForCat(nd, key));
      nd[key] = nd[key].filter((v) => valid.has(v));
    });
    return nd;
  };

  const toggle = (key, value) => setDraft((d) => {
    const set = new Set(d[key]);
    set.has(value) ? set.delete(value) : set.add(value);
    return prune({ ...d, [key]: [...set] });
  });
  const setEntities = (vals) => setDraft((d) => prune({ ...d, entities: vals }));
  // Select-all / clear for the active step (selects the full option list, not
  // just the search-filtered subset).
  const selectAllCat = (key) => setDraft((d) => {
    const all = combos ? optionsForCat(d, key) : (key === 'entities' ? entityList : (dimOpts[key] || options[key] || []));
    return prune({ ...d, [key]: [...all] });
  });
  const clearCat = (key) => setDraft((d) => prune({ ...d, [key]: [] }));
  // Switching dimension clears prior selections (they belong to the old dimension).
  const changeDim = (dim) => setDraft((d) => ({
    ...d, dimension: dim, entities: [], metrics: [], years: [], quarters: [], lobs: [], classes: [],
  }));

  // Gated stepper: a step stays locked until every prior step has a selection,
  // so the user must follow the cascade order (steps with no available options
  // for the current selections are auto-skipped rather than blocking forever).
  let firstOpen = -1;
  for (let i = 0; i < CASCADE_ORDER.length; i += 1) {
    const k = CASCADE_ORDER[i];
    if (draft[k]?.length) continue;
    if (combos && optionsForCat(draft, k).length === 0) continue; // empty step → skip
    firstOpen = i;
    break;
  }
  const isLocked = (i) => firstOpen !== -1 && i > firstOpen;

  // Never leave the active tab on a locked step (e.g. after clearing a step).
  useEffect(() => {
    const ci = CASCADE_ORDER.indexOf(cat);
    if (isLocked(ci)) { setCat(CASCADE_ORDER[firstOpen]); setSearch(''); }
  }, [firstOpen]);

  const count = draft.entities.length + draft.metrics.length;
  const baseList = combos
    ? optionsForCat(draft, cat)
    : (cat === 'entities' ? entityList : (dimOpts[cat] || options[cat] || []));
  const list = baseList.filter((o) => o.toLowerCase().includes(search.toLowerCase()));

  function apply() {
    if (!draft.entities.length || !draft.metrics.length) {
      toast.error('Please select at least one Entity and one Metric.');
      return;
    }
    onApply(draft);
  }

  return (
    <Modal title="Report Configuration" width="900px" onClose={onClose}
      footer={(
        <>
          <button className="btn btn-ghost btn-sm" onClick={() => setDraft(blankFilters(draft.dimension))}>
            <i className="fas fa-trash-can" /> Clear All
          </button>
          <button className="btn btn-primary btn-sm" onClick={apply}><i className="fas fa-check" /> Apply Filters</button>
        </>
      )}>
      <div className="dim-select">
        <label className="mono-label">Step 1 · Select Report View</label>
        <select className="select" value={draft.dimension} onChange={(e) => changeDim(e.target.value)}>
          {(options.dimensions?.length ? options.dimensions : ['Insurer']).map((d) => (
            <option key={d} value={d}>{d}-wise View</option>
          ))}
        </select>
      </div>

      <div className="filter-modal-body">
        <div className="filter-cats">
          {CATEGORIES.map(([key, label], i) => {
            const locked = isLocked(i);
            const done = draft[key]?.length > 0;
            return (
              <div key={key}
                className={`cat-item ${cat === key ? 'active' : ''} ${locked ? 'locked' : ''}`}
                onClick={() => { if (!locked) { setCat(key); setSearch(''); } }}>
                <span className="cat-label">
                  <span className="cat-step">{locked ? <i className="fas fa-lock" /> : done ? <i className="fas fa-check" /> : i + 2}</span>
                  {key === 'entities' ? entityNoun(draft.dimension) : label}
                </span>
                {done && <span className="cat-count">{draft[key].length}</span>}
              </div>
            );
          })}
        </div>

        <div className="filter-options-pane">
          <input className="input" placeholder="Search options…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ marginBottom: 12 }} />

          <div className="entity-chips">
            <span className="chip-btn" onClick={() => selectAllCat(cat)}><i className="fas fa-check-double" /> Select all</span>
            {cat === 'entities' && draft.dimension === 'Insurer' && ENTITY_GROUPS.map((g) => (
              <span key={g.name} className="chip-btn" onClick={() => setEntities(entityList.filter((e) => g.keys.some((k) => e.toLowerCase().includes(k.toLowerCase()))))}>{g.name}</span>
            ))}
            <span className="chip-btn danger" onClick={() => clearCat(cat)}>Clear</span>
          </div>

          {list.length === 0 ? (
            <div className="no-data-msg">{cat === 'entities'
              ? '⚠️ No entity data for this view. Try Admin → Sync Data.'
              : (search ? 'No options found.' : 'No options for the earlier selections — adjust a previous step.')}</div>
          ) : list.map((opt) => (
            <label key={opt} className="checkbox-item">
              <input type="checkbox" checked={draft[cat].includes(opt)} onChange={() => toggle(cat, opt)} />
              <span>{opt}</span>
            </label>
          ))}
        </div>
      </div>
      <div className="filter-count-note">{count} selection{count === 1 ? '' : 's'} · Entities &amp; Metrics required</div>
    </Modal>
  );
}

function ReportTable({ report, onExport, onFlag }) {
  const toast = useToast();
  const cols = report.columns || [];

  function copyData() {
    const keep = cols.filter((c) => c !== 'Source_File');
    const head = keep.map((c) => c.replace(/_/g, ' ')).join('\t');
    const body = report.rows.map((r) => keep.map((c) => r[c]).join('\t')).join('\n');
    navigator.clipboard.writeText(head + '\n' + body)
      .then(() => toast.success('Table copied! (Source filenames excluded)'))
      .catch(() => toast.error('Copy failed'));
  }

  return (
    <div className="data-table-wrapper anim-rise">
      <div className="data-table-header">
        <div className="dt-title"><strong>Financial Report</strong><span className="dt-rows">{report.rows.length} rows</span></div>
        <div className="dt-actions">
          <button className="btn btn-ghost btn-sm" onClick={copyData}><i className="fas fa-copy" /> Copy Data</button>
          <button className="btn btn-ghost btn-sm" onClick={onFlag}><i className="fas fa-flag" /> Flag</button>
          <button className="btn btn-sm dt-excel" onClick={onExport}><i className="fas fa-file-excel" /> Export Excel</button>
        </div>
      </div>
      <div className="data-table-scroll">
        <table className="dt-table">
          <thead>
            <tr>{cols.map((c) => <th key={c} className={c === 'Source_File' ? 'src-col' : ''}>{c.replace(/_/g, ' ')}</th>)}</tr>
          </thead>
          <tbody>
            {report.rows.map((row, ri) => (
              <tr key={ri}>
                {cols.map((c, ci) => (
                  <td key={c} className={`${c === 'Source_File' ? 'src-col' : ''} ${ci < 2 ? 'fw-bold' : ''}`} title={c === 'Source_File' ? row[c] : undefined}>{row[c]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function DataExplorer() {
  const toast = useToast();
  const [options, setOptions] = useState(null);
  const [filters, setFilters] = useState(blankFilters());
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modal, setModal] = useState(false);
  const [flagOpen, setFlagOpen] = useState(false);
  const [view, setView] = useState('chart');

  useEffect(() => {
    api.get('/data/options').then(setOptions).catch(() => toast.error('Could not load filter options'));
  }, []);

  async function applyFilters(draft) {
    setFilters(draft);
    setModal(false);
    setLoading(true);
    try {
      const data = await api.post('/data/filter', draft);
      setReport(data);
    } catch (err) {
      toast.error(err.message || 'Filter failed');
    } finally {
      setLoading(false);
    }
  }

  function exportExcel() {
    api.download('/data/download', filters, 'IRIS_Financial_Report.xlsx').catch((e) => toast.error(e.message));
  }

  const summary = useMemo(() => (report ? `Report: ${filters.dimension} View` : 'No filters applied'), [report, filters]);
  const filterCount = filters.entities.length + filters.metrics.length;

  return (
    <>
      <PageHeader fullForm="IRDAI's Regulatory Intelligence System" title="Data Explorer" scope="Scope: Financials & Market Data" />
      <div className="page-body">
        <div className="filter-actions-bar">
          <button className="btn btn-ghost filter-main-btn" onClick={() => setModal(true)} disabled={!options}>
            <i className="fas fa-sliders" /> Filters &amp; View
            {report && filterCount > 0 && <span className="filter-badge">{filterCount}</span>}
          </button>
          <span className="active-filters-text">{summary}</span>
          <button className="reset-link" onClick={() => { setReport(null); setFilters(blankFilters()); }}>Reset Dashboard</button>
        </div>

        {loading ? (
          <div className="dt-loading"><TypingDots /><div>Processing data…</div></div>
        ) : !report ? (
          <EmptyState icon="fa-chart-column">Select <strong>Filters</strong> to generate a report.</EmptyState>
        ) : (
          <>
            {report.risks?.length > 0 && (
              <div className="ews-box anim-rise">
                <div className="ews-head"><i className="fas fa-radiation" /> <strong>IRIS Early Warning System</strong></div>
                <div className="ews-list">
                  {report.risks.map((a, i) => {
                    const p = parseAlert(a.msg);
                    return (
                      <div key={i} className={`alert-base ${a.level === 'critical' ? 'alert-crit' : 'alert-warn'}`}>
                        <i className={`fas ${a.level === 'critical' ? 'fa-circle-xmark' : 'fa-chart-line'}`} />
                        <div className="alert-content">
                          <div className="alert-title">{p.title}</div>
                          {p.entity && <div className="alert-entity">{p.entity}</div>}
                          {p.detail && <div className="alert-detail">{p.detail}</div>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {report.missing?.length > 0 && (
              <div className="gaps-box anim-rise">
                <div className="gaps-head"><i className="fas fa-circle-info" /> <strong>Data Gaps Detected</strong></div>
                <ul>{report.missing.map((m, i) => <li key={i}>{m}</li>)}</ul>
              </div>
            )}

            {!report.rows?.length ? (
              <EmptyState icon="fa-folder-open">No matching records found.</EmptyState>
            ) : (
              <>
                <div className="view-toggle-bar">
                  <div className="view-toggle">
                    <button className={`vt-btn ${view === 'chart' ? 'active' : ''}`} onClick={() => setView('chart')}>
                      <i className="fas fa-chart-line" /> Dashboard
                    </button>
                    <button className={`vt-btn ${view === 'table' ? 'active' : ''}`} onClick={() => setView('table')}>
                      <i className="fas fa-table" /> Table
                    </button>
                  </div>
                  {view === 'chart' && (
                    <div className="dt-actions">
                      <button className="btn btn-ghost btn-sm" onClick={() => setFlagOpen(true)}><i className="fas fa-flag" /> Flag</button>
                      <button className="btn btn-sm dt-excel" onClick={exportExcel}><i className="fas fa-file-excel" /> Export Excel</button>
                    </div>
                  )}
                </div>
                {view === 'chart'
                  ? <ReportCharts report={report} dimension={filters.dimension} />
                  : <ReportTable report={report} onExport={exportExcel} onFlag={() => setFlagOpen(true)} />}
              </>
            )}
          </>
        )}
      </div>

      {modal && options && (
        <FilterModal options={options} initial={filters} onApply={applyFilters} onClose={() => setModal(false)} />
      )}

      {flagOpen && (
        <FlagModal kind="financial"
          target={`${filters.dimension} View · ${filters.entities.join(', ').slice(0, 120) || 'report'}`}
          detail={JSON.stringify({ filters, columns: report?.columns })}
          onClose={() => setFlagOpen(false)} />
      )}
    </>
  );
}
