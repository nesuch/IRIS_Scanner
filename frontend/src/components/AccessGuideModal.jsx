import { useEffect, useMemo, useState } from 'react';
import { Modal, Spinner } from './UI.jsx';
import { useToast } from './Toast.jsx';
import { api } from '../api.js';

// Reference guide: which IRIS view + filter path reaches each Handbook table.
export default function AccessGuideModal({ onClose }) {
  const toast = useToast();
  const [rows, setRows] = useState(null);
  const [q, setQ] = useState('');
  const [view, setView] = useState('All');

  useEffect(() => {
    api.get('/guide')
      .then((d) => setRows(d.rows || []))
      .catch((e) => { toast.error(e.message || 'Could not load guide'); setRows([]); });
  }, []);

  const views = useMemo(
    () => ['All', ...[...new Set((rows || []).map((r) => r['IRIS View']).filter(Boolean))].sort()],
    [rows],
  );

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return (rows || []).filter((r) => {
      if (view !== 'All' && r['IRIS View'] !== view) return false;
      if (!needle) return true;
      return Object.values(r).some((v) => String(v).toLowerCase().includes(needle));
    });
  }, [rows, q, view]);

  return (
    <Modal title="Handbook → IRIS Access Guide" width="920px" onClose={onClose}
      footer={(
        <>
          <span className="guide-count">{filtered.length} of {rows?.length ?? 0} tables</span>
          <a className="btn btn-primary btn-sm" href="/api/guide/download">
            <i className="fas fa-file-excel" /> Download Excel
          </a>
        </>
      )}>
      <p className="guide-intro">Find where each Handbook table lives in IRIS and the exact filter path to reach it.</p>
      <div className="guide-controls">
        <input className="input" placeholder="Search table, title, or path…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
        <select className="select" value={view} onChange={(e) => setView(e.target.value)}>
          {views.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      </div>

      {rows === null ? (
        <div className="guide-loading"><Spinner size={16} /> Loading guide…</div>
      ) : (
        <div className="guide-table-wrap">
          <table className="guide-table">
            <thead>
              <tr><th>Table</th><th>Handbook Title</th><th>IRIS View</th><th>How to access</th></tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr><td colSpan={4} className="guide-empty">No tables match.</td></tr>
              ) : filtered.map((r, i) => (
                <tr key={i}>
                  <td className="guide-tno">{r.Part} · {r.Table}</td>
                  <td className="guide-title">{r['Handbook Table Title']}</td>
                  <td><span className="guide-viewchip">{r['IRIS View']}</span></td>
                  <td className="guide-path">{r['How to Access (filter path)']}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Modal>
  );
}
