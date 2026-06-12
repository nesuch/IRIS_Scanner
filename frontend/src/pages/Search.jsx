import { useEffect, useMemo, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import ClauseEditorModal from '../components/ClauseEditor.jsx';
import { api } from '../api.js';
import { TYPE_STYLES, ClauseBody, groupByType } from './search/clauseRender.jsx';
import PdfViewer from './search/PdfViewer.jsx';
import FlagModal from '../components/FlagModal.jsx';
import './search/search.css';

// Reasons tailored to a "no clauses found" outcome (the default flag reasons
// assume a clause exists and is wrong).
const NO_RESULT_REASONS = [
  'Relevant clause exists but was not found',
  'Content missing from the knowledge base',
  'Document not loaded for this module',
  'Searched the wrong department/module',
  'Other',
];

const MODULE_META = {
  universal: { title: 'Universal Search', icon: 'fa-magnifying-glass', scope: 'Unified Search (All Departments)' },
  health: { title: 'Health Department', icon: 'fa-heart-pulse', scope: 'Acts, Regulations & Master Circulars (Health)' },
  life: { title: 'Life Department', icon: 'fa-umbrella', scope: 'Regulatory Framework (Life Insurance)' },
  nonlife: { title: 'Non-Life Department', icon: 'fa-shield-halved', scope: 'Regulatory Framework (General Insurance)' },
};

// Dates as DD MMM YYYY (e.g. 29 May 2024).
function fmtClauseDate(d) {
  if (!d) return '';
  const dt = new Date(d);
  if (Number.isNaN(dt.getTime())) return d;
  return dt.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
}

// Clause body + admin controls (rich-edit + tags). Holds a local HTML override
// so an edit shows immediately without re-running the search.
function ClauseContent({ m, keywords }) {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin' || !!user?.is_admin;
  const isEditor = isAdmin || user?.role === 'editor';
  const [html, setHtml] = useState(m.html || '');
  const [editing, setEditing] = useState(false);
  return (
    <>
      {html
        ? <div className="clause-html" dangerouslySetInnerHTML={{ __html: html }} />
        : <ClauseBody text={m.raw_text} keywords={keywords} />}
      {isEditor && (
        <div className="clause-admin-row">
          <button className="clause-edit-btn" onClick={() => setEditing(true)}><i className="fas fa-pen-to-square" />Edit clause</button>
        </div>
      )}
      <ClauseTags m={m} />
      {editing && (
        <ClauseEditorModal clause={m} onClose={() => setEditing(false)}
          onSaved={(h) => { setHtml(h); setEditing(false); }} />
      )}
    </>
  );
}

// Admin-only inline tag editor on a clause card (drives tag search).
function ClauseTags({ m }) {
  const { user } = useAuth();
  const toast = useToast();
  const [tags, setTags] = useState(m.tags || []);
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState('');
  const [busy, setBusy] = useState(false);
  if (!(user?.role === 'admin' || user?.is_admin || user?.role === 'editor')) return null;

  async function save() {
    setBusy(true);
    try {
      const r = await api.post('/clause/retag', { id: m.id, source: m.source, tags: val });
      setTags(r.tags || []); setEditing(false); toast.success('Tags updated');
    } catch (e) { toast.error(e.message || 'Could not update'); }
    finally { setBusy(false); }
  }

  return (
    <div className="clause-tags-row">
      {!editing ? (
        <>
          <span className="clause-tags-label"><i className="fas fa-tags" /> Tags:</span>
          {tags.length ? tags.map((t) => <span key={t} className="clause-tag">{t}</span>)
            : <span className="clause-tag-empty">none</span>}
          <button className="clause-tag-edit" onClick={() => { setVal(tags.join(', ')); setEditing(true); }}>Edit</button>
        </>
      ) : (
        <span className="clause-retag">
          <input className="input" value={val} autoFocus onChange={(e) => setVal(e.target.value)} placeholder="comma-separated tags" />
          <button className="btn btn-primary btn-sm" onClick={save} disabled={busy}>{busy ? '…' : 'Save'}</button>
          <button className="btn btn-ghost btn-sm" onClick={() => setEditing(false)} disabled={busy}>Cancel</button>
        </span>
      )}
    </div>
  );
}

function ResultCards({ resp, onChip, onOpenPane, onFlag }) {
  const toast = useToast();
  const copy = async (text) => {
    try { await navigator.clipboard.writeText(text); toast.success('Clause text copied!'); }
    catch { toast.error('Could not copy'); }
  };

  // Lets users report a missing/incorrect "no results" outcome.
  const flagNoResult = onFlag && (
    <button className="flag-link" title="Report missing or wrong content"
      onClick={() => onFlag({ _noresult: true, header: resp.query_label || 'search', id: resp.module || '', raw_text: resp.note || 'No matches found' })}>
      <i className="fas fa-flag" /> Flag this
    </button>
  );

  if (resp.kind === 'greeting') {
    return (
      <p className="iris-msg">
        <strong>Hello!</strong> I am <strong>IRIS</strong>. Ask me anything related to IRDAI Acts,
        Regulations, Circulars, or Guidelines.
      </p>
    );
  }
  if (resp.kind === 'rejected') {
    return <p className="iris-msg">{resp.note} {flagNoResult}</p>;
  }

  const groups = groupByType(resp.matches || []);
  const hasMatches = (resp.matches || []).length > 0;

  return (
    <div>
      {resp.kind === 'deep_scan' && hasMatches && (
        <div className="iris-analysis">Deep Scan results in <strong>{resp.module.toUpperCase()}</strong>: <strong>{(resp.keywords || []).join(', ')}</strong></div>
      )}
      {resp.kind === 'tags' && hasMatches && (
        <div className="iris-foundvia">Found via <strong>Tags</strong>: {(resp.keywords || []).join(', ')}</div>
      )}
      {resp.note && !hasMatches && <p className="iris-msg">{resp.note} {flagNoResult}</p>}

      {groups.map((g, gi) => {
        const st = TYPE_STYLES[g.type] || TYPE_STYLES.UNKNOWN;
        return (
          <div key={gi}>
            <div className="type-band" style={{ background: st.bg, color: st.color, borderLeftColor: st.bar }}>{st.label}</div>
            {g.items.map((m, mi) => (
              <div className="clause-card" key={mi}>
                <div className="clause-meta">
                  <span className="clause-source">{m.source}</span>
                  <span className="sep">|</span>
                  <span className="clause-header">{m.header}</span>
                  <span className="sep">|</span>
                  <span className="clause-id">Clause: {m.id}</span>
                  {m.doc_status && (() => {
                    const rep = m.doc_status.toLowerCase() === 'repealed';
                    return (
                      <span className={`clause-status ${rep ? 'cs-repealed' : 'cs-active'}`}>
                        <i className={`fas ${rep ? 'fa-ban' : 'fa-circle-check'}`} />
                        {rep
                          ? `Repealed${m.repealed_on ? ` ${fmtClauseDate(m.repealed_on)}` : ''}`
                          : `Active${m.effective_date ? ` · from ${fmtClauseDate(m.effective_date)}` : ''}`}
                      </span>
                    );
                  })()}
                  <span className="clause-actions">
                    {m.pdf_url && (
                      <button className="pane-btn" title="Open PDF in side pane" onClick={() => onOpenPane(m)}><i className="fas fa-table-columns" /> Pane</button>
                    )}
                    {m.pdf_url && (
                      <a className="pdf-btn" href={m.pdf_url} target="_blank" rel="noreferrer"><i className="fas fa-file-pdf" /> PDF</a>
                    )}
                    <button className="copy-btn" title="Copy clause" onClick={() => copy(m.raw_text)}><i className="far fa-copy" /></button>
                    <button className="flag-btn" title="Flag this clause" onClick={() => onFlag(m)}><i className="fas fa-flag" /></button>
                  </span>
                </div>
                <ClauseContent m={m} keywords={resp.highlight || []} />
              </div>
            ))}
          </div>
        );
      })}

      {(resp.chips || []).length > 0 && (
        <div className="deep-chips">
          <div className="deep-chips-hint">Not finding what you need? <strong>Deep Scan specific terms:</strong></div>
          <div className="deep-chips-row">
            {resp.chips.map((c, ci) => (
              <button key={ci} className={`chip chip-${c.kind}`} onClick={() => onChip(c.payload)}>{c.label}</button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Search({ module }) {
  const meta = MODULE_META[module] || MODULE_META.universal;
  const toast = useToast();
  const [history, setHistory] = useState([]);
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [docGroups, setDocGroups] = useState([]);
  const [docTags, setDocTags] = useState({});
  const [selected, setSelected] = useState(() => new Set());
  const [docFilterOpen, setDocFilterOpen] = useState(false);
  const [pdfPane, setPdfPane] = useState(null); // { url, source }
  const [flagClause, setFlagClause] = useState(null); // clause being flagged
  const [paneWidth, setPaneWidth] = useState(46); // % width of the PDF pane
  const [dragging, setDragging] = useState(false);
  const chatRef = useRef(null);
  const lastUserRef = useRef(null);
  const docFilterRef = useRef(null);
  const searchMainRef = useRef(null);

  // Lock the document to the viewport (fixed bottom bar; only results scroll).
  useEffect(() => {
    document.documentElement.classList.add('app-fixed');
    return () => document.documentElement.classList.remove('app-fixed');
  }, []);

  // Reset chat when switching modules (matches per-page server history reset).
  useEffect(() => { setHistory([]); setQuery(''); setSuggestions([]); setDocFilterOpen(false); setPdfPane(null); }, [module]);

  // Load the documents available to this module (with their tags) and select all by default.
  useEffect(() => {
    api.get(`/docs?module=${module}`).then((d) => {
      const groups = d.groups || [];
      setDocGroups(groups);
      setDocTags(d.doc_tags || {});
      const all = new Set();
      groups.forEach((g) => g.docs.forEach((s) => all.add(s)));
      setSelected(all);
    }).catch(() => { setDocGroups([]); setDocTags({}); setSelected(new Set()); });
  }, [module]);

  // Autocomplete vocabulary = tags present only in the currently selected documents.
  const vocab = useMemo(() => {
    const set = new Set();
    selected.forEach((s) => (docTags[s] || []).forEach((t) => set.add(t)));
    return [...set];
  }, [docTags, selected]);

  // Close the doc-filter popover on outside click.
  useEffect(() => {
    if (!docFilterOpen) return undefined;
    const onDown = (e) => { if (docFilterRef.current && !docFilterRef.current.contains(e.target)) setDocFilterOpen(false); };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [docFilterOpen]);

  const allDocs = docGroups.flatMap((g) => g.docs);
  const allSelected = allDocs.length > 0 && selected.size === allDocs.length;
  const noneSelected = selected.size === 0;
  const toggleDoc = (src) => setSelected((s) => { const n = new Set(s); n.has(src) ? n.delete(src) : n.add(src); return n; });
  const toggleGroup = (g) => setSelected((s) => { const n = new Set(s); const allIn = g.docs.every((d) => n.has(d)); g.docs.forEach((d) => (allIn ? n.delete(d) : n.add(d))); return n; });
  const toggleAll = () => setSelected((s) => (s.size === allDocs.length ? new Set() : new Set(allDocs)));

  // Drag the divider to resize the results / PDF split.
  function startResize(e) {
    e.preventDefault();
    const rect = searchMainRef.current?.getBoundingClientRect();
    if (!rect) return;
    setDragging(true);
    document.body.style.userSelect = 'none';
    const onMove = (ev) => {
      const pct = ((rect.right - ev.clientX) / rect.width) * 100;
      setPaneWidth(Math.min(72, Math.max(28, pct)));
    };
    const onUp = () => {
      setDragging(false);
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }

  useEffect(() => {
    // Scroll so the latest question sits near the top (jumpToLatestQuestionStart).
    if (lastUserRef.current && chatRef.current) {
      const top = lastUserRef.current.offsetTop - 16;
      chatRef.current.scrollTop = Math.max(top, 0);
    }
  }, [history]);

  async function runSearch(q, displayLabel) {
    if (!q.trim() || busy) return;
    const entry = { id: Math.random().toString(36).slice(2), query: displayLabel ?? q, response: null };
    setHistory((h) => [...h, entry]);
    setBusy(true);
    try {
      const body = { module, query: q };
      const total = docGroups.reduce((n, g) => n + g.docs.length, 0);
      if (docGroups.length && selected.size < total) body.sources = [...selected]; // subset only; all => omit
      const resp = await api.post('/search', body);
      setHistory((h) => h.map((x) => (x.id === entry.id ? { ...x, response: resp } : x)));
    } catch (err) {
      setHistory((h) => h.map((x) => (x.id === entry.id ? { ...x, error: err.message } : x)));
      toast.error(err.message || 'Search failed');
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setSuggestions([]);
    setQuery('');
    runSearch(q);
  }

  function onChip(payload) {
    runSearch('__DEEP_SCAN__:' + payload, 'Deep Scan');
  }

  function onInput(e) {
    const val = e.target.value;
    setQuery(val);
    // Clause-number mode: "/" then a number (e.g. /64VB, /4 (1) (i)).
    if (val.trimStart().startsWith('/')) {
      const key = val.replace(/[^a-z0-9]/gi, '');
      if (key.length < 1) { setSuggestions([]); return; }
      const params = new URLSearchParams({ q: val.trim() });
      if (allDocs.length && selected.size < allDocs.length) [...selected].forEach((s) => params.append('source', s));
      api.get(`/clause-suggest?${params.toString()}`)
        .then((d) => setSuggestions((d.suggestions || []).map((s) => ({ ...s, clause: true }))))
        .catch(() => setSuggestions([]));
      return;
    }
    const words = val.toLowerCase().split(/[\s,]+/);
    const lastWord = words[words.length - 1];
    if (lastWord.length < 2) { setSuggestions([]); return; }
    const text = val.toLowerCase();
    const matches = vocab
      .filter((c) => c.toLowerCase().includes(lastWord) && !text.includes(c.toLowerCase()))
      .slice(0, 8)
      .map((c) => c.charAt(0).toUpperCase() + c.slice(1));
    setSuggestions(matches);
  }

  function selectSuggestion(value) {
    // Click-and-go: a clause suggestion runs its lookup; a tag suggestion completes
    // the query and searches immediately (no separate "press search" step).
    if (value && typeof value === 'object' && value.clause) {
      setSuggestions([]);
      setQuery('');
      runSearch(`/${value.id}`, value.id);
      return;
    }
    const val = query;
    const lastSpace = val.lastIndexOf(' ');
    const next = (lastSpace === -1 ? value : val.substring(0, lastSpace + 1) + value).trim();
    setSuggestions([]);
    setQuery('');
    runSearch(next);
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(e); }
  }

  const empty = history.length === 0;

  return (
    <div className="search-shell">
      <PageHeader fullForm="IRDAI's Regulatory Intelligence System" title={meta.title} scope={`Scope: ${meta.scope}`} />

      <div className={`search-main ${dragging ? 'dragging' : ''}`} ref={searchMainRef}>
      <div className={`chat-window ${empty ? 'is-empty' : ''}`} ref={chatRef}>
        {empty && (
          <div className="chat-empty anim-fade">
            <i className="fas fa-lightbulb" />
            <p><strong>Ready to assist.</strong><br />Try searching for “Migration”, “No Claim Bonus”, or “Free Look Period”.</p>
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
                {item.response ? <ResultCards resp={item.response} onChip={onChip} onOpenPane={(m) => setPdfPane({ url: m.pdf_url, source: m.source })} onFlag={(m) => setFlagClause(m)} />
                  : item.error ? <p className="iris-msg" style={{ color: 'var(--bad)' }}>{item.error}</p>
                  : <span className="typing"><span /><span /><span /></span>}
              </div>
            </div>
          </div>
        ))}
        <div style={{ height: 10 }} />
      </div>

        {pdfPane && <div className={`pane-resizer ${dragging ? 'dragging' : ''}`} onMouseDown={startResize} title="Drag to resize" />}
        {pdfPane && (
          <aside className="pdf-pane anim-fade" style={{ flexBasis: `${paneWidth}%` }}>
            <div className="pdf-pane-head">
              <span className="pdf-pane-title" title={pdfPane.source}><i className="fas fa-file-pdf" /> {pdfPane.source}</span>
              <span className="pdf-pane-actions">
                <a className="pdf-pane-btn" href={pdfPane.url} target="_blank" rel="noreferrer" title="Open in new tab"><i className="fas fa-arrow-up-right-from-square" /></a>
                <button className="pdf-pane-btn" onClick={() => setPdfPane(null)} title="Close pane" aria-label="Close PDF pane"><i className="fas fa-xmark" /></button>
              </span>
            </div>
            <PdfViewer url={pdfPane.url} />
          </aside>
        )}
      </div>

      <div className="input-area">
        {empty && !query && <div className="search-tip">Type a question — or <b>/</b> then a clause number (e.g. <b>/64</b>)</div>}
        <form className="input-inner" onSubmit={onSubmit} autoComplete="off">
          <div className="search-wrapper">
            <div className="doc-filter" ref={docFilterRef}>
              <button type="button" className={`doc-filter-btn ${!allSelected && allDocs.length ? 'is-active' : ''}`}
                onClick={() => setDocFilterOpen((o) => !o)} title="Filter source documents" aria-label="Filter source documents">
                <i className="fas fa-filter" />
                {!allSelected && allDocs.length > 0 && <span className="doc-filter-badge">{selected.size}</span>}
              </button>
              {docFilterOpen && (
                <div className="doc-filter-panel">
                  <div className="doc-filter-head">
                    <span>Search in documents</span>
                    <button type="button" className="doc-filter-close" onClick={() => setDocFilterOpen(false)} aria-label="Close">&times;</button>
                  </div>
                  <label className="doc-row doc-row-all">
                    <input type="checkbox" checked={allSelected}
                      ref={(el) => { if (el) el.indeterminate = !allSelected && !noneSelected; }} onChange={toggleAll} />
                    <span>Select all</span>
                    <span className="doc-group-count">{selected.size}/{allDocs.length}</span>
                  </label>
                  <div className="doc-filter-body">
                    {docGroups.length === 0 && <div className="doc-empty">No documents in this module.</div>}
                    {docGroups.map((g) => {
                      const sel = g.docs.filter((d) => selected.has(d)).length;
                      const groupAll = sel === g.docs.length;
                      const groupSome = sel > 0 && !groupAll;
                      return (
                        <div className="doc-group" key={g.type}>
                          <label className="doc-row doc-group-head">
                            <input type="checkbox" checked={groupAll}
                              ref={(el) => { if (el) el.indeterminate = groupSome; }} onChange={() => toggleGroup(g)} />
                            <span className="doc-group-label">{g.label}</span>
                            <span className="doc-group-count">{sel}/{g.docs.length}</span>
                          </label>
                          {g.docs.map((d) => (
                            <label className="doc-row doc-item" key={d}>
                              <input type="checkbox" checked={selected.has(d)} onChange={() => toggleDoc(d)} />
                              <span>{d}</span>
                            </label>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
            {suggestions.length > 0 && (
              <div className="suggestions-box">
                {suggestions.map((s, i) => (
                  s && typeof s === 'object' && s.clause ? (
                    <div className="suggestion-item clause-sugg" key={i} onMouseDown={(e) => { e.preventDefault(); selectSuggestion(s); }}>
                      <span className="clause-sugg-main"><span className="clause-id">{s.id}</span> <span className="clause-snip">{s.snippet}</span></span>
                      <span className="badge badge-navy">{s.source}</span>
                    </div>
                  ) : (
                    <div className="suggestion-item" key={i} onMouseDown={(e) => { e.preventDefault(); selectSuggestion(s); }}>
                      <span>{s}</span><span className="badge badge-concept">Concept</span>
                    </div>
                  )
                ))}
              </div>
            )}
            <textarea className="search-input" placeholder="Ask IRIS…" value={query}
              onChange={onInput} onKeyDown={onKeyDown} rows={1} autoFocus />
            <button className="btn btn-primary search-submit" type="submit" disabled={busy} aria-label="Search">
              <i className="fas fa-magnifying-glass" /> <span className="btn-label">Search</span>
            </button>
          </div>
        </form>
      </div>

      {flagClause && (
        <FlagModal kind="clause"
          title={flagClause._noresult ? 'Report a missing result' : undefined}
          reasons={flagClause._noresult ? NO_RESULT_REASONS : undefined}
          target={flagClause._noresult
            ? `No result · "${flagClause.header}" · ${flagClause.id || 'all'} module`
            : `${flagClause.source} · ${flagClause.header} · Clause ${flagClause.id}`}
          detail={flagClause.raw_text}
          onClose={() => setFlagClause(null)} />
      )}
    </div>
  );
}
