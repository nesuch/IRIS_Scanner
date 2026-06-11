import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';
import './search/search.css';   // reuse the universal-search shell (bottom bar, suggestions)
import './pqs/pqs.css';

export default function Pqs() {
  const toast = useToast();
  const { user } = useAuth();
  const isAdmin = !!user?.is_admin;
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);   // null = nothing searched yet (empty state)
  const [lastQuery, setLastQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [active, setActive] = useState(null);      // full reply being read (overlay)
  const [opening, setOpening] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [allTags, setAllTags] = useState([]);

  useEffect(() => { api.get('/pq/tags').then((d) => setAllTags(d.tags || [])).catch(() => setAllTags([])); }, []);
  useEffect(() => {
    document.documentElement.classList.add('app-fixed');
    return () => document.documentElement.classList.remove('app-fixed');
  }, []);

  async function runSearch(q) {
    if (!q.trim() || busy) return;
    setBusy(true); setSuggestions([]);
    try {
      const d = await api.get(`/pq?q=${encodeURIComponent(q.trim())}`);
      setResults(d.items || []);
      setLastQuery(q.trim());
    } catch (e) { toast.error(e.message || 'Search failed'); setResults([]); }
    finally { setBusy(false); }
  }

  function onSubmit(e) { e.preventDefault(); const q = query.trim(); if (!q) return; setQuery(''); runSearch(q); }
  function onKeyDown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(e); } }

  function onInput(e) {
    const val = e.target.value;
    setQuery(val);
    const t = val.trim().toLowerCase();
    if (t.length < 2) { setSuggestions([]); return; }
    // Suggest matching tags (the PQ vocabulary), not whole documents.
    setSuggestions(allTags.filter((tag) => tag.toLowerCase().includes(t)).slice(0, 8));
  }

  function pickTag(tag) { setQuery(''); setSuggestions([]); runSearch(tag); }

  function open(id) {
    setOpening(true);
    api.get(`/pq/${id}`).then(setActive)
      .catch((e) => toast.error(e.message || 'Could not load PQ'))
      .finally(() => setOpening(false));
  }

  async function del(id) {
    if (!window.confirm('Delete this Parliamentary Question? This cannot be undone.')) return;
    try {
      await api.post(`/pq/${id}/delete`);
      toast.success('Deleted');
      if (active?.id === id) setActive(null);
      if (lastQuery) runSearch(lastQuery); else setResults(null);
    } catch (e) { toast.error(e.message || 'Could not delete'); }
  }

  // ---- Reading view (overlay; search state preserved underneath) ----
  if (active) {
    return (
      <div className="search-shell">
        <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions" />
        <div className="pq-read-bar">
          <button className="btn btn-ghost btn-sm" onClick={() => setActive(null)}><i className="fas fa-arrow-left" /> Back to results</button>
          <div className="pq-read-actions">
            {isAdmin && <RetagControl pq={active} onSaved={(tags) => { setActive({ ...active, tags }); if (lastQuery) runSearch(lastQuery); }} />}
            {isAdmin && <button className="btn btn-ghost btn-sm danger" onClick={() => del(active.id)}><i className="fas fa-trash" /> Delete</button>}
            {active.download_url && (
              <a className="btn btn-primary btn-sm" href={active.download_url} target="_blank" rel="noreferrer"><i className="fas fa-file-word" /> Download original (.docx)</a>
            )}
          </div>
        </div>
        <div className="pq-read-scroll">
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
      </div>
    );
  }

  // ---- Search view (mirrors Universal Search: empty start, bottom bar, suggestions) ----
  return (
    <div className="search-shell">
      <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions">
        {isAdmin && <button className="btn btn-primary btn-sm" onClick={() => setUploadOpen(true)}><i className="fas fa-upload" /> Upload PQ</button>}
      </PageHeader>

      <div className="search-main">
        <div className={`chat-window ${results === null ? 'is-empty' : ''}`}>
          {results === null ? (
            <div className="chat-empty anim-fade">
              <i className="fas fa-landmark" />
              <p><strong>Search Parliamentary Questions.</strong><br />Try “senior citizens”, “cashless”, or “claim repudiation”.</p>
            </div>
          ) : results.length === 0 ? (
            <p className="iris-msg">No Parliamentary Questions match “{lastQuery}”.</p>
          ) : (
            <div className="pq-feed">
              <div className="pq-result-count">{results.length} {results.length === 1 ? 'reply' : 'replies'} for “{lastQuery}”</div>
              {opening && <div className="pq-loading"><Spinner size={16} /> Opening…</div>}
              <div className="pq-list">
                {results.map((p) => (
                  <div key={p.id} className="pq-card" onClick={() => open(p.id)} role="button" tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') open(p.id); }}>
                    {isAdmin && <button className="pq-card-del" title="Delete" onClick={(e) => { e.stopPropagation(); del(p.id); }}><i className="fas fa-trash" /></button>}
                    <div className="pq-card-top">
                      {p.house && <span className="pq-house">{p.house}</span>}
                      {p.pq_no && <span className="pq-no">Q No. {p.pq_no}</span>}
                      {p.date && <span className="pq-date">{p.date}</span>}
                    </div>
                    <div className="pq-card-title">{p.subject || p.title}</div>
                    {p.tags?.length > 0 && (
                      <div className="pq-tags">{p.tags.slice(0, 5).map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
                    )}
                    <span className="pq-card-open">Read full reply <i className="fas fa-arrow-right" /></span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="input-area">
        <form className="input-inner" onSubmit={onSubmit} autoComplete="off">
          <div className="search-wrapper">
            {suggestions.length > 0 && (
              <div className="suggestions-box">
                {suggestions.map((s, i) => (
                  <div className="suggestion-item" key={i} onMouseDown={(e) => { e.preventDefault(); pickTag(s); }}>
                    <span><i className="fas fa-tag" style={{ fontSize: 10, color: 'var(--muted)', marginRight: 7 }} />{s}</span>
                    <span className="badge badge-concept">Tag</span>
                  </div>
                ))}
              </div>
            )}
            <textarea className="search-input" placeholder="Search Parliamentary Questions — topic, subject, or tag…" value={query}
              onChange={onInput} onKeyDown={onKeyDown} rows={1} autoFocus />
            <button className="btn btn-primary search-submit" type="submit" disabled={busy} aria-label="Search">
              {busy ? <Spinner size={16} color="#fff" /> : <i className="fas fa-magnifying-glass" />} <span className="btn-label">Search</span>
            </button>
          </div>
        </form>
      </div>

      {uploadOpen && <UploadModal onClose={() => setUploadOpen(false)} onDone={() => { setUploadOpen(false); if (lastQuery) runSearch(lastQuery); }} />}
    </div>
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
      toast.success('Tags updated'); setEditing(false); onSaved(r.tags || []);
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
