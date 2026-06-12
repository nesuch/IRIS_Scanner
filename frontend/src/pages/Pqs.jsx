import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import FlagModal from '../components/FlagModal.jsx';
import { api } from '../api.js';
import './search/search.css';   // reuse the universal-search shell (bottom bar, suggestions)
import './pqs/pqs.css';

const NO_RESULT_REASONS = [
  'Missing PQ (should be here)',
  'Wrong / irrelevant results',
  'Search not working as expected',
  'Other',
];

// Controlled department vocabulary — mirrors PQ_DEPARTMENTS in api.py.
const DEPTS = [
  { code: 'HEALTH', label: 'Health' },
  { code: 'LIFE', label: 'Life' },
  { code: 'NONLIFE', label: 'Non-Life' },
];
const DEPT_LABEL = Object.fromEntries(DEPTS.map((d) => [d.code, d.label]));

// Deep-Scan suggestion chips — reads full reply bodies, not just headlines.
function DeepChips({ chips, onPick }) {
  if (!chips || chips.length === 0) return null;
  return (
    <div className="deep-chips">
      <div className="deep-chips-hint">Not finding it in titles &amp; tags? <strong>Deep Scan full reply text:</strong></div>
      <div className="deep-chips-row">
        {chips.map((c, i) => (
          <button key={i} className={`chip chip-${c.kind}`}
            onClick={() => onPick(c.text, c.mode, c.mode === 'all' ? `“${c.text}” (all words)` : `“${c.text}”`)}>
            {c.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// Reusable Health/Life/Non-Life multi-select for upload/edit/bulk.
function DeptPicker({ value, onChange }) {
  const set = new Set(value);
  const toggle = (code) => {
    const next = new Set(set);
    next.has(code) ? next.delete(code) : next.add(code);
    onChange(DEPTS.filter((d) => next.has(d.code)).map((d) => d.code));
  };
  return (
    <div className="pq-dept-pick">
      {DEPTS.map((d) => (
        <button type="button" key={d.code}
          className={`pq-dept-chip ${set.has(d.code) ? 'on' : ''}`}
          onClick={() => toggle(d.code)}>
          <i className={`fas ${set.has(d.code) ? 'fa-check' : 'fa-plus'}`} /> {d.label}
        </button>
      ))}
    </div>
  );
}

// One IRIS answer: the result set for a single query, with its own department
// refinement. Renders cards, the no-match deep-scan prompt, or a dept-empty note.
function PqResponse({ resp, dismissed, isAdmin, onOpen, onHide, onDelete, onDeep, onFlag }) {
  const [deptFilter, setDeptFilter] = useState(resp.initialDept || []);
  const results = resp.items || [];
  const toggleDept = (code) => setDeptFilter((cur) => cur.includes(code) ? cur.filter((c) => c !== code) : [...cur, code]);
  const visible = results.filter((p) => !dismissed.has(p.id)
    && (deptFilter.length === 0 || (p.departments || []).some((c) => deptFilter.includes(c))));

  // No matches at all — explain and offer Deep Scan (full reply bodies).
  if (results.length === 0) {
    return (
      <div className="pq-noresult">
        <p className="iris-msg">{resp.deep
          ? `I read every reply in full — none mention ${resp.label}.`
          : resp.browse ? 'There are no Parliamentary Questions in the database yet.'
          : `Nothing is tagged for “${resp.label}”.`}
          {onFlag && (
            <button className="flag-link" title="Report a missing PQ"
              onClick={() => onFlag({ _noresult: true, label: resp.label })}>
              <i className="fas fa-flag" /> Flag this
            </button>
          )}
        </p>
        {resp.qText && !resp.deep && (
          <button className="btn btn-primary btn-sm pq-deep-cta" onClick={() => onDeep(resp.qText, 'all')}>
            <i className="fas fa-binoculars" /> Deep scan every reply for “{resp.qText}”
          </button>
        )}
        <DeepChips chips={resp.chips} onPick={onDeep} />
      </div>
    );
  }

  return (
    <div className="pq-feed">
      <div className={`pq-foundvia ${resp.deep ? 'is-deep' : ''}`}>
        {resp.deep
          ? <><i className="fas fa-binoculars" /> Deep Scan · <strong>{resp.label}</strong></>
          : resp.browse
            ? <><i className="fas fa-layer-group" /> {resp.initialDept?.length ? <>Department · <strong>{resp.initialDept.map((c) => DEPT_LABEL[c]).join(' / ')}</strong></> : <>All Parliamentary replies</>}</>
            : <><i className="fas fa-tag" /> Found via <strong>Tags</strong> · {resp.label}</>}
      </div>
      {results.length > 0 && (
        <div className="pq-dept-bar">
          <span className="pq-dept-bar-label">Department:</span>
          {DEPTS.map((d) => {
            const n = results.filter((p) => (p.departments || []).includes(d.code)).length;
            return (
              <button key={d.code} className={`pq-dept-chip ${deptFilter.includes(d.code) ? 'on' : ''}`}
                onClick={() => toggleDept(d.code)}>
                {d.label}<span className="pq-dept-count">{n}</span>
              </button>
            );
          })}
          {deptFilter.length > 0 && <button className="pq-dept-clear" onClick={() => setDeptFilter([])}>Clear</button>}
        </div>
      )}
      <div className="pq-result-count">{visible.length} {visible.length === 1 ? 'reply' : 'replies'}{resp.deep ? ` mentioning ${resp.label} · deep scan` : resp.browse ? ' in the database' : ` tagged “${resp.label}”`}{deptFilter.length > 0 ? ` · ${deptFilter.map((c) => DEPT_LABEL[c]).join(' / ')}` : ''}</div>
      {visible.length === 0 ? (
        <p className="iris-msg">No {deptFilter.map((c) => DEPT_LABEL[c]).join(' / ')} PQs in this set.</p>
      ) : (
        <div className="pq-list">
          {visible.map((p) => (
            <div key={p.id} className="pq-card" onClick={() => onOpen(p.id)} role="button" tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter') onOpen(p.id); }}>
              <button className="pq-card-hide" title="Hide from results" onClick={(e) => { e.stopPropagation(); onHide(p.id); }}>&times;</button>
              {onFlag && <button className="pq-card-flag" title="Flag this reply" onClick={(e) => { e.stopPropagation(); onFlag(p); }}><i className="fas fa-flag" /></button>}
              {isAdmin && <button className="pq-card-del" title="Delete" onClick={(e) => { e.stopPropagation(); onDelete(p.id); }}><i className="fas fa-trash" /></button>}
              <div className="pq-card-top">
                {p.house && <span className="pq-house">{p.house}</span>}
                {(p.pq_no || p.date) && (
                  <span className="pq-meta-text">
                    {p.pq_no && <span className="pq-no">Q No. {p.pq_no}</span>}
                    {p.pq_no && p.date && <span className="pq-sep" />}
                    {p.date && <span className="pq-date"><i className="far fa-calendar" /> {p.date}</span>}
                  </span>
                )}
              </div>
              <div className="pq-card-title">{p.subject || p.title}</div>
              {((p.departments || []).length > 0 || p.tags?.length > 0) && (
                <div className="pq-card-meta-row">
                  {(p.departments || []).map((c) => <span key={c} className="pq-dept-badge">{DEPT_LABEL[c]}</span>)}
                  {(p.tags || []).slice(0, 5).map((t) => <span key={t} className="pq-tag">{t}</span>)}
                </div>
              )}
              <span className="pq-card-open">Read full reply <i className="fas fa-arrow-right" /></span>
            </div>
          ))}
        </div>
      )}
      {!resp.deep && <DeepChips chips={resp.chips} onPick={onDeep} />}
    </div>
  );
}

export default function Pqs() {
  const toast = useToast();
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin' || !!user?.is_admin;
  const isEditor = isAdmin || user?.role === 'editor';
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([]);   // [{ id, query (label), response, error }]
  const [dismissed, setDismissed] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [active, setActive] = useState(null);
  const [opening, setOpening] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [bulkOpen, setBulkOpen] = useState(false);
  const [allTags, setAllTags] = useState([]);
  const [flagTarget, setFlagTarget] = useState(null);   // PQ (or no-result) being flagged
  const chatRef = useRef(null);
  const lastUserRef = useRef(null);

  const loadTags = () => api.get('/pq/tags').then((d) => setAllTags(d.tags || [])).catch(() => setAllTags([]));
  useEffect(() => { loadTags(); }, []);
  useEffect(() => {
    document.documentElement.classList.add('app-fixed');
    return () => document.documentElement.classList.remove('app-fixed');
  }, []);

  function clearChat() { setHistory([]); setQuery(''); setSuggestions([]); setActive(null); setDismissed(new Set()); }

  // Re-clicking "Parliamentary Q&A" in the sidebar clears the conversation.
  useEffect(() => {
    const onReclick = () => clearChat();
    window.addEventListener('iris:reclick', onReclick);
    return () => window.removeEventListener('iris:reclick', onReclick);
  }, []);

  // Keep the latest question anchored near the top as the chat grows.
  useEffect(() => {
    if (lastUserRef.current && chatRef.current) {
      chatRef.current.scrollTop = Math.max(lastUserRef.current.offsetTop - 16, 0);
    }
  }, [history]);

  // Append a chat turn (You → IRIS) and resolve its response.
  async function ask(url, label, meta = {}) {
    if (busy) return;
    const id = Math.random().toString(36).slice(2);
    setHistory((h) => [...h, { id, query: label, response: null }]);
    setBusy(true);
    try {
      const d = await api.get(url);
      const response = { items: d.items || [], chips: d.chips || [], deep: !!d.deep, label, ...meta };
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (e) {
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, error: e.message || 'Search failed' } : x)));
      toast.error(e.message || 'Search failed');
    } finally { setBusy(false); }
  }

  // mode: 'q' free text | 'tag' exact tag | 'num' PQ number
  function runSearch(value, mode = 'q') {
    const v = value.trim(); if (!v) return;
    const key = mode === 'tag' ? 'tag' : mode === 'num' ? 'num' : 'q';
    const label = mode === 'num' ? `/${v}` : v;
    ask(`/pq?${key}=${encodeURIComponent(v)}`, label, { qText: mode === 'q' ? v : '' });
  }

  // Deep Scan — read every reply's full body. mode: all | phrase | word.
  function runDeepScan(text, mode = 'all', label = '') {
    const v = text.trim(); if (!v) return;
    const shown = label || `“${v}”`;
    ask(`/pq?deep=${encodeURIComponent(v)}&mode=${mode}`, `Deep Scan: ${shown}`, { deep: true, label: shown });
  }

  // Browse the whole library; `depts` pre-selects the department refinement.
  function browseAll(depts = []) {
    const label = depts.length ? `All ${depts.map((c) => DEPT_LABEL[c]).join(' / ')} PQs` : 'All PQs';
    ask('/pq', label, { browse: true, initialDept: depts });
  }

  function onSubmit(e) {
    e.preventDefault();
    const q = query.trim(); if (!q) return;
    setQuery(''); setSuggestions([]);
    if (q.startsWith('/')) { const d = q.replace(/\D/g, ''); if (d) runSearch(d, 'num'); }
    else runSearch(q, 'q');
  }
  function onKeyDown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(e); } }

  function onInput(e) {
    const val = e.target.value;
    setQuery(val);
    const t = val.trim();
    if (t.startsWith('/')) {                 // "/9000" -> suggest PQs by number
      const digits = t.replace(/\D/g, '');
      if (!digits) { setSuggestions([]); return; }
      api.get(`/pq?num=${digits}`)
        .then((d) => setSuggestions((d.items || []).slice(0, 30).map((p) => ({ ...p, _pq: true }))))
        .catch(() => setSuggestions([]));
      return;
    }
    const low = t.toLowerCase();
    if (low.length < 2) { setSuggestions([]); return; }
    setSuggestions(allTags.filter((tag) => tag.toLowerCase().includes(low)).slice(0, 30));
  }

  function pickTag(tag) { setQuery(''); setSuggestions([]); runSearch(tag, 'tag'); }

  function open(id) {
    setOpening(true);
    api.get(`/pq/${id}`).then(setActive)
      .catch((e) => toast.error(e.message || 'Could not load PQ'))
      .finally(() => setOpening(false));
  }

  function hide(id) { setDismissed((s) => new Set(s).add(id)); }

  async function del(id) {
    if (!window.confirm('Delete this Parliamentary Question? This cannot be undone.')) return;
    try {
      await api.post(`/pq/${id}/delete`);
      toast.success('Deleted');
      if (active?.id === id) setActive(null);
      hide(id); loadTags();
    } catch (e) { toast.error(e.message || 'Could not delete'); }
  }

  // ---- Reading view ----
  if (active) {
    return (
      <div className="search-shell">
        <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions" />
        <div className="pq-read-bar">
          <button className="btn btn-ghost btn-sm" onClick={() => setActive(null)}><i className="fas fa-arrow-left" /> Back to results</button>
          <div className="pq-read-actions">
            {isEditor && <EditMeta pq={active}
              onSaved={(title, tags, departments) => { setActive({ ...active, title, tags, departments }); loadTags(); }} />}
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
              {(active.pq_no || active.date) && (
                <span className="pq-meta-text">
                  {active.pq_no && <span className="pq-no">Q No. {active.pq_no}</span>}
                  {active.pq_no && active.date && <span className="pq-sep" />}
                  {active.date && <span className="pq-date"><i className="far fa-calendar" /> {active.date}</span>}
                </span>
              )}
            </div>
            <h2 className="pq-doc-title">{active.title}</h2>
            {active.departments?.length > 0 && (
              <div className="pq-doc-depts">{active.departments.map((c) => <span key={c} className="pq-dept-badge">{DEPT_LABEL[c]}</span>)}</div>
            )}
            {active.tags?.length > 0 && (
              <div className="pq-tags">{active.tags.map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
            )}
            <div className="pq-html" dangerouslySetInnerHTML={{ __html: active.html }} />
          </div>
        </div>
      </div>
    );
  }

  const empty = history.length === 0;

  // ---- Chat view ----
  return (
    <div className="search-shell">
      <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions">
        {!empty && <button className="btn btn-ghost btn-sm" onClick={clearChat}><i className="fas fa-arrow-rotate-left" /> Clear</button>}
        <button className="btn btn-ghost btn-sm" onClick={() => browseAll()}><i className="fas fa-list" /> All PQs</button>
        {isEditor && <button className="btn btn-ghost btn-sm" onClick={() => setBulkOpen(true)}><i className="fas fa-layer-group" /> Bulk upload</button>}
        {isEditor && <button className="btn btn-primary btn-sm" onClick={() => setUploadOpen(true)}><i className="fas fa-upload" /> Upload PQ</button>}
      </PageHeader>

      <div className="search-main">
        <div className={`chat-window ${empty ? 'is-empty' : ''}`} ref={chatRef}>
          {empty && (
            <div className="chat-empty anim-fade">
              <i className="fas fa-landmark" />
              <p><strong>Search Parliamentary Questions.</strong><br />Try a topic, a tag, or <b>/</b> + a PQ number (e.g. <b>/9000</b>).</p>
              <button className="btn btn-ghost btn-sm pq-empty-browse" onClick={() => browseAll()}><i className="fas fa-list" /> Browse all PQs in the database</button>
              <div className="pq-empty-depts">
                <span>or by department:</span>
                {DEPTS.map((d) => (
                  <button key={d.code} className="pq-dept-chip" onClick={() => browseAll([d.code])}>{d.label}</button>
                ))}
              </div>
            </div>
          )}

          {history.map((item, idx) => (
            <div key={item.id}>
              <div className="chat-block user" ref={idx === history.length - 1 ? lastUserRef : null}>
                <div className="chat-label">You</div>
                <div className="bubble user-bubble">{item.query}</div>
              </div>
              <div className="chat-block iris">
                <div className="chat-label">IRIS</div>
                <div className="bubble iris-bubble">
                  {item.response ? (
                    <PqResponse resp={item.response} dismissed={dismissed} isAdmin={isAdmin}
                      onOpen={open} onHide={hide} onDelete={del} onDeep={runDeepScan} onFlag={setFlagTarget} />
                  ) : item.error ? (
                    <p className="iris-msg" style={{ color: 'var(--bad)' }}>{item.error}</p>
                  ) : (
                    <span className="typing"><span /><span /><span /></span>
                  )}
                </div>
              </div>
            </div>
          ))}
          {opening && <div className="pq-loading"><Spinner size={16} /> Opening…</div>}
          <div style={{ height: 10 }} />
        </div>
      </div>

      <div className="input-area">
        <form className="input-inner" onSubmit={onSubmit} autoComplete="off">
          <div className="search-wrapper">
            {suggestions.length > 0 && (
              <div className="suggestions-box">
                {suggestions.map((s, i) => (s && s._pq ? (
                  <div className="suggestion-item clause-sugg" key={i} onMouseDown={(e) => { e.preventDefault(); setQuery(''); setSuggestions([]); open(s.id); }}>
                    <span className="clause-sugg-main"><span className="clause-id">Q{s.pq_no}</span> <span className="clause-snip">{s.subject || s.title}</span></span>
                    {s.house && <span className="badge badge-navy">{s.house}</span>}
                  </div>
                ) : (
                  <div className="suggestion-item" key={i} onMouseDown={(e) => { e.preventDefault(); pickTag(s); }}>
                    <span><i className="fas fa-tag" style={{ fontSize: 10, color: 'var(--muted)', marginRight: 7 }} />{s}</span>
                    <span className="badge badge-concept">Tag</span>
                  </div>
                )))}
              </div>
            )}
            <textarea className="search-input" placeholder="Search PQs — topic, tag, or / + number…" value={query}
              onChange={onInput} onKeyDown={onKeyDown} rows={1} autoFocus />
            <button className="btn btn-primary search-submit" type="submit" disabled={busy} aria-label="Search">
              {busy ? <Spinner size={16} color="#fff" /> : <i className="fas fa-magnifying-glass" />} <span className="btn-label">Search</span>
            </button>
          </div>
        </form>
      </div>

      {uploadOpen && <UploadModal onClose={() => setUploadOpen(false)} onDone={() => { setUploadOpen(false); loadTags(); }} />}
      {bulkOpen && <BulkUploadModal onClose={() => setBulkOpen(false)} onDone={() => { setBulkOpen(false); loadTags(); }} />}

      {flagTarget && (
        <FlagModal kind="pq"
          title={flagTarget._noresult ? 'Report a missing PQ' : 'Flag this reply'}
          reasons={flagTarget._noresult ? NO_RESULT_REASONS : undefined}
          target={flagTarget._noresult
            ? `No result · “${flagTarget.label}” · Parliamentary Q&A`
            : `${[flagTarget.house, flagTarget.pq_no ? `Q No. ${flagTarget.pq_no}` : ''].filter(Boolean).join(' · ')} · ${flagTarget.subject || flagTarget.title}`}
          detail={flagTarget._noresult ? `Searched: ${flagTarget.label}` : (flagTarget.subject || flagTarget.title)}
          onClose={() => setFlagTarget(null)} />
      )}
    </div>
  );
}

function EditMeta({ pq, onSaved }) {
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState('');
  const [depts, setDepts] = useState([]);
  const [busy, setBusy] = useState(false);

  async function save() {
    setBusy(true);
    try {
      const r = await api.post(`/pq/${pq.id}/update`, { title, tags, departments: depts });
      toast.success('Saved'); setEditing(false); onSaved(r.title, r.tags || [], r.departments || []);
    } catch (e) { toast.error(e.message || 'Could not save'); }
    finally { setBusy(false); }
  }

  return (
    <>
      <button className="btn btn-ghost btn-sm" onClick={() => { setTitle(pq.title || ''); setTags((pq.tags || []).join(', ')); setDepts(pq.departments || []); setEditing(true); }}><i className="fas fa-pen" /> Edit</button>
      {editing && (
        <Modal title="Edit PQ" width="540px" onClose={() => setEditing(false)}
          footer={<>
            <button className="btn btn-ghost btn-sm" onClick={() => setEditing(false)} disabled={busy}>Cancel</button>
            <button className="btn btn-primary btn-sm" onClick={save} disabled={busy}>{busy ? <Spinner size={13} color="#fff" /> : 'Save'}</button>
          </>}>
          <div className="field" style={{ marginBottom: 12 }}>
            <label>Title</label>
            <textarea className="input" rows={2} value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div className="field" style={{ marginBottom: 12 }}>
            <label>Department(s) <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(pick any that apply)</span></label>
            <DeptPicker value={depts} onChange={setDepts} />
          </div>
          <div className="field">
            <label>Tags <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(comma-separated)</span></label>
            <input className="input" value={tags} onChange={(e) => setTags(e.target.value)} placeholder="e.g. senior citizens, claims" />
          </div>
        </Modal>
      )}
    </>
  );
}

function UploadModal({ onClose, onDone }) {
  const toast = useToast();
  const [file, setFile] = useState(null);
  const [tags, setTags] = useState('');
  const [depts, setDepts] = useState([]);
  const [busy, setBusy] = useState(false);
  const [dup, setDup] = useState(null);   // existing PQ flagged as a likely duplicate

  async function submit(force = false) {
    if (!file) { toast.error('Choose a .docx file'); return; }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', file); fd.append('tags', tags); fd.append('departments', depts.join(','));
      if (force) fd.append('force', '1');
      const r = await api.post('/pq/upload', fd);
      toast.success(`Added: ${r.title?.slice(0, 50) || 'PQ'}`);
      onDone();
    } catch (e) {
      if (e.status === 409 && e.data?.duplicate) setDup(e.data.existing);
      else toast.error(e.message || 'Upload failed');
    }
    finally { setBusy(false); }
  }

  return (
    <Modal title="Upload Parliamentary Question" width="520px" onClose={onClose}
      footer={dup ? <>
        <button className="btn btn-ghost btn-sm" onClick={() => setDup(null)} disabled={busy}>Cancel</button>
        <button className="btn btn-primary btn-sm danger" onClick={() => submit(true)} disabled={busy}>
          {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-triangle-exclamation" />} Upload anyway
        </button>
      </> : <>
        <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
        <button className="btn btn-primary btn-sm" onClick={() => submit(false)} disabled={busy || !file}>
          {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-upload" />} Upload &amp; publish
        </button>
      </>}>
      {dup && (
        <div className="pq-dup-warn">
          <i className="fas fa-triangle-exclamation" />
          <div>
            <strong>Looks like this is already in IRIS.</strong>
            <p>A reply <b>{dup.house} No. {dup.pq_no}</b> already exists: “{dup.title?.slice(0, 90)}”. Upload again only if this is a different or corrected version.</p>
          </div>
        </div>
      )}
      <p className="guide-intro">Upload the approved reply as a Word file. IRIS renders it on screen (formatting + tables preserved), keeps the original for download, and makes it searchable here.</p>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Word document (.docx)</label>
        <input className="input" type="file" accept=".docx" onChange={(e) => { setFile(e.target.files?.[0] || null); setDup(null); }} />
      </div>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Department(s) <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(pick any that apply)</span></label>
        <DeptPicker value={depts} onChange={setDepts} />
      </div>
      <div className="field">
        <label>Tags <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(comma-separated — drives search)</span></label>
        <input className="input" value={tags} onChange={(e) => setTags(e.target.value)}
          placeholder="e.g. senior citizens, claim repudiation, grievance" />
      </div>
    </Modal>
  );
}

// Bulk: upload many .docx, then walk through each to confirm title + add tags.
function BulkUploadModal({ onClose, onDone }) {
  const toast = useToast();
  const [files, setFiles] = useState([]);
  const [phase, setPhase] = useState('select');   // select | uploading | review
  const [created, setCreated] = useState([]);
  const [idx, setIdx] = useState(0);
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState('');
  const [depts, setDepts] = useState([]);
  const [busy, setBusy] = useState(false);

  async function startUpload() {
    if (!files.length) { toast.error('Choose .docx files'); return; }
    setPhase('uploading');
    try {
      const fd = new FormData();
      [...files].forEach((f) => fd.append('files', f));
      const r = await api.post('/pq/bulk-upload', fd);
      const c = r.created || [];
      const dupes = r.duplicates || [];
      if (!c.length) {
        toast.error(dupes.length ? `All ${dupes.length} already exist in IRIS — nothing uploaded.` : 'No valid .docx files');
        setPhase('select'); return;
      }
      setCreated(c); setIdx(0); setTitle(c[0].title || ''); setTags(''); setDepts([]);
      setPhase('review');
      toast.success(`Uploaded ${c.length}${dupes.length ? `, skipped ${dupes.length} duplicate${dupes.length > 1 ? 's' : ''}` : ''} — now add tags`);
    } catch (e) { toast.error(e.message || 'Upload failed'); setPhase('select'); }
  }

  function goNext(n) {
    if (n >= created.length) { toast.success('Done'); onDone(); return; }
    setIdx(n); setTitle(created[n].title || ''); setTags(''); setDepts([]);
  }
  async function saveCurrent() {
    const pq = created[idx];
    setBusy(true);
    try { await api.post(`/pq/${pq.id}/update`, { title, tags, departments: depts }); }
    catch (e) { toast.error(e.message || 'Save failed'); setBusy(false); return; }
    setBusy(false); goNext(idx + 1);
  }

  if (phase === 'review') {
    const pq = created[idx];
    return (
      <Modal title={`Tag PQ ${idx + 1} of ${created.length}`} width="560px" onClose={onDone}
        footer={<>
          <button className="btn btn-ghost btn-sm" onClick={() => goNext(idx + 1)} disabled={busy}>Skip</button>
          <button className="btn btn-primary btn-sm" onClick={saveCurrent} disabled={busy}>
            {busy ? <Spinner size={13} color="#fff" /> : (idx + 1 < created.length ? 'Save & next' : 'Save & finish')}
          </button>
        </>}>
        <div className="pq-bulk-meta">{[pq.house, pq.pq_no ? `Q ${pq.pq_no}` : '', pq.filename].filter(Boolean).join(' · ')}</div>
        <div className="field" style={{ marginBottom: 12 }}>
          <label>Title</label>
          <textarea className="input" rows={2} value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        <div className="field" style={{ marginBottom: 12 }}>
          <label>Department(s) <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(pick any that apply)</span></label>
          <DeptPicker value={depts} onChange={setDepts} />
        </div>
        <div className="field">
          <label>Tags <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(comma-separated)</span></label>
          <input className="input" value={tags} onChange={(e) => setTags(e.target.value)} autoFocus placeholder="e.g. senior citizens, claims" />
        </div>
      </Modal>
    );
  }

  return (
    <Modal title="Bulk upload PQs" width="520px" onClose={onClose}
      footer={<>
        <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={phase === 'uploading'}>Cancel</button>
        <button className="btn btn-primary btn-sm" onClick={startUpload} disabled={phase === 'uploading' || !files.length}>
          {phase === 'uploading' ? <Spinner size={14} color="#fff" /> : <i className="fas fa-upload" />} Upload{files.length ? ` ${files.length}` : ''}
        </button>
      </>}>
      <p className="guide-intro">Select several approved replies (.docx). IRIS uploads them all, then walks you through each to confirm the title and add tags.</p>
      <div className="field">
        <label>Word documents (.docx)</label>
        <input className="input" type="file" accept=".docx" multiple onChange={(e) => setFiles(e.target.files)} />
      </div>
      {files.length > 0 && <div className="pq-bulk-list">{[...files].map((f, i) => <div key={i} className="pq-bulk-file"><i className="fas fa-file-word" /> {f.name}</div>)}</div>}
    </Modal>
  );
}
