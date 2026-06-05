import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Modal, EmptyState, TypingDots } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import FlagModal from '../components/FlagModal.jsx';
import { api } from '../api.js';
import './data/data.css';

const ENTITY_GROUPS = [
  { name: 'SAHIs', keys: ['Star', 'Care', 'Aditya Birla', 'Niva', 'Manipal', 'Galaxy', 'Narayana'] },
  { name: 'PSUs', keys: ['New India', 'United India', 'Oriental', 'National'] },
  { name: 'Life', keys: ['LIC', 'HDFC Life', 'SBI Life', 'ICICI Prudential'] },
];

const CATEGORIES = [
  ['entities', 'Entities'], ['metrics', 'Metrics'], ['classes', 'Class of Biz'],
  ['years', 'Fin Year'], ['quarters', 'Quarter'], ['lobs', 'Line of Biz'],
];

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
  const optionsFor = { entities: entityList, metrics: options.metrics, classes: options.classes,
    years: options.years, quarters: options.quarters, lobs: options.lobs };

  const toggle = (key, value) => setDraft((d) => {
    const set = new Set(d[key]);
    set.has(value) ? set.delete(value) : set.add(value);
    return { ...d, [key]: [...set] };
  });
  const setEntities = (vals) => setDraft((d) => ({ ...d, entities: vals }));
  const changeDim = (dim) => setDraft((d) => ({ ...d, dimension: dim, entities: [] }));

  const count = draft.entities.length + draft.metrics.length;
  const list = (optionsFor[cat] || []).filter((o) => o.toLowerCase().includes(search.toLowerCase()));

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
          {CATEGORIES.map(([key, label]) => (
            <div key={key} className={`cat-item ${cat === key ? 'active' : ''}`} onClick={() => { setCat(key); setSearch(''); }}>
              {key === 'entities' ? (draft.dimension === 'Insurer' ? 'Insurers' : draft.dimension + 's') : label}
              {draft[key]?.length > 0 && <span className="cat-count">{draft[key].length}</span>}
            </div>
          ))}
        </div>

        <div className="filter-options-pane">
          <input className="input" placeholder="Search options…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ marginBottom: 12 }} />

          {cat === 'entities' && (
            <div className="entity-chips">
              <span className="chip-btn" onClick={() => setEntities([...entityList])}>All {draft.dimension}s</span>
              {draft.dimension === 'Insurer' && ENTITY_GROUPS.map((g) => (
                <span key={g.name} className="chip-btn" onClick={() => setEntities(entityList.filter((e) => g.keys.some((k) => e.toLowerCase().includes(k.toLowerCase()))))}>{g.name}</span>
              ))}
              <span className="chip-btn danger" onClick={() => setEntities([])}>Clear</span>
            </div>
          )}

          {list.length === 0 ? (
            <div className="no-data-msg">{cat === 'entities' ? '⚠️ No entity data for this view. Try Admin → Sync Data.' : 'No options found.'}</div>
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
              <ReportTable report={report} onExport={exportExcel} onFlag={() => setFlagOpen(true)} />
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
