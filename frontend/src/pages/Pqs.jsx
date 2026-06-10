import { useCallback, useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { EmptyState, PageLoading, Spinner, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';
import './pqs/pqs.css';

export default function Pqs() {
  const toast = useToast();
  const { user } = useAuth();
  const isAdmin = !!user?.is_admin;
  const [query, setQuery] = useState('');
  const [items, setItems] = useState(null);
  const [active, setActive] = useState(null);     // full reply being read (overlay)
  const [loading, setLoading] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);

  const fetchList = useCallback((q) => api.get(`/pq${q ? `?q=${encodeURIComponent(q)}` : ''}`)
    .then((d) => setItems(d.items || []))
    .catch((e) => { toast.error(e.message || 'Could not load PQs'); setItems([]); }), [toast]);

  useEffect(() => { fetchList(''); }, [fetchList]);
  useEffect(() => {  // debounced search
    const t = setTimeout(() => fetchList(query.trim()), 250);
    return () => clearTimeout(t);
  }, [query, fetchList]);

  function open(id) {
    setLoading(true);
    api.get(`/pq/${id}`).then(setActive)
      .catch((e) => toast.error(e.message || 'Could not load PQ'))
      .finally(() => setLoading(false));
  }

  async function del(id) {
    if (!window.confirm('Delete this Parliamentary Question? This cannot be undone.')) return;
    try {
      await api.post(`/pq/${id}/delete`);
      toast.success('Deleted');
      if (active?.id === id) setActive(null);
      fetchList(query.trim());
    } catch (e) { toast.error(e.message || 'Could not delete'); }
  }

  if (!items) return (<><PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions" /><div className="page-body"><PageLoading /></div></>);

  // ---- Reading view (overlay; list/search state preserved underneath) ----
  if (active) {
    return (
      <>
        <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions" />
        <div className="page-body">
          <div className="pq-read-bar">
            <button className="btn btn-ghost btn-sm" onClick={() => setActive(null)}><i className="fas fa-arrow-left" /> Back to results</button>
            <div className="pq-read-actions">
              {isAdmin && <RetagControl pq={active} onSaved={(tags) => { setActive({ ...active, tags }); fetchList(query.trim()); }} />}
              {isAdmin && <button className="btn btn-ghost btn-sm danger" onClick={() => del(active.id)}><i className="fas fa-trash" /> Delete</button>}
              {active.download_url && (
                <a className="btn btn-primary btn-sm" href={active.download_url} target="_blank" rel="noreferrer"><i className="fas fa-file-word" /> Download original (.docx)</a>
              )}
            </div>
          </div>
          <div className="pq-doc card">
            <div className="pq-doc-head">
              {active.house && <span className="pq-house">{active.house}</span>}
              {active.pq_no && <span className="pq-no">Q No. {active.pq_no}</span>}
              {active.date && <span className="pq-date">{active.date}</span>}
            </div>
            <h2 className="pq-doc-title">{active.title}</h2>
            {active.tags?.length > 0 && (
              <div className="pq-tags">{active.tags.map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
            )}
            <div className="pq-html" dangerouslySetInnerHTML={{ __html: active.html }} />
          </div>
        </div>
      </>
    );
  }

  // ---- Search view ----
  return (
    <>
      <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions">
        {isAdmin && <button className="btn btn-primary btn-sm" onClick={() => setUploadOpen(true)}><i className="fas fa-upload" /> Upload PQ</button>}
      </PageHeader>
      <div className="page-body">
        <div className="pq-searchbar">
          <i className="fas fa-magnifying-glass" />
          <input className="input" autoFocus value={query} onChange={(e) => setQuery(e.target.value)}
            placeholder="Search Parliamentary Questions — topic, subject, or tag (e.g. senior citizens, cashless)…" />
          {query && <button className="pq-clear" onClick={() => setQuery('')} aria-label="Clear">&times;</button>}
        </div>
        {loading && <div className="pq-loading"><Spinner size={16} /> Opening…</div>}
        {items.length === 0 ? (
          <EmptyState icon="fa-landmark">{query ? 'No Parliamentary Questions match your search.' : 'No Parliamentary Questions yet.'}</EmptyState>
        ) : (
          <>
            <div className="pq-result-count">{items.length} {items.length === 1 ? 'reply' : 'replies'}{query ? ' found' : ''}</div>
            <div className="pq-list">
              {items.map((p) => (
                <div key={p.id} className="pq-card" onClick={() => open(p.id)} role="button" tabIndex={0}
                  onKeyDown={(e) => { if (e.key === 'Enter') open(p.id); }}>
                  {isAdmin && <button className="pq-card-del" title="Delete" onClick={(e) => { e.stopPropagation(); del(p.id); }}><i className="fas fa-trash" /></button>}
                  <div className="pq-card-top">
                    {p.house && <span className="pq-house">{p.house}</span>}
                    {p.pq_no && <span className="pq-no">Q No. {p.pq_no}</span>}
                    {p.date && <span className="pq-date">{p.date}</span>}
                  </div>
                  <div className="pq-card-title">{p.subject || p.title}</div>
                  {p.snippet && <div className="pq-card-snip">{p.snippet}</div>}
                  {p.tags?.length > 0 && (
                    <div className="pq-tags">{p.tags.slice(0, 5).map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
                  )}
                  <span className="pq-card-open">Read full reply <i className="fas fa-arrow-right" /></span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
      {uploadOpen && <UploadModal onClose={() => setUploadOpen(false)} onDone={() => { setUploadOpen(false); fetchList(query.trim()); }} />}
    </>
  );
}

function RetagControl({ pq, onSaved }) {
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState((pq.tags || []).join(', '));
  const [busy, setBusy] = useState(false);

  async function save() {
    setBusy(true);
    try {
      const r = await api.post(`/pq/${pq.id}/retag`, { tags: val });
      toast.success('Tags updated');
      setEditing(false);
      onSaved(r.tags || []);
    } catch (e) { toast.error(e.message || 'Could not update'); }
    finally { setBusy(false); }
  }

  if (!editing) return <button className="btn btn-ghost btn-sm" onClick={() => { setVal((pq.tags || []).join(', ')); setEditing(true); }}><i className="fas fa-tags" /> Edit tags</button>;
  return (
    <span className="pq-retag">
      <input className="input" value={val} onChange={(e) => setVal(e.target.value)} placeholder="comma-separated tags" />
      <button className="btn btn-primary btn-sm" onClick={save} disabled={busy}>{busy ? <Spinner size={13} color="#fff" /> : 'Save'}</button>
      <button className="btn btn-ghost btn-sm" onClick={() => setEditing(false)} disabled={busy}>Cancel</button>
    </span>
  );
}

function UploadModal({ onClose, onDone }) {
  const toast = useToast();
  const [file, setFile] = useState(null);
  const [tags, setTags] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!file) { toast.error('Choose a .docx file'); return; }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('tags', tags);
      const r = await api.post('/pq/upload', fd);
      toast.success(`Added: ${r.title?.slice(0, 50) || 'PQ'}`);
      onDone();
    } catch (e) { toast.error(e.message || 'Upload failed'); }
    finally { setBusy(false); }
  }

  return (
    <Modal title="Upload Parliamentary Question" width="520px" onClose={onClose}
      footer={(
        <>
          <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={submit} disabled={busy || !file}>
            {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-upload" />} Upload &amp; publish
          </button>
        </>
      )}>
      <p className="guide-intro">Upload the approved reply as a Word file. IRIS renders it on screen (formatting + tables preserved), keeps the original for download, and makes it searchable here.</p>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Word document (.docx)</label>
        <input className="input" type="file" accept=".docx" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      </div>
      <div className="field">
        <label>Tags <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(comma-separated — drives search)</span></label>
        <input className="input" value={tags} onChange={(e) => setTags(e.target.value)}
          placeholder="e.g. senior citizens, claim repudiation, grievance" />
      </div>
    </Modal>
  );
}
