import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, EmptyState, PageLoading } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import FlagModal from '../components/FlagModal.jsx';
import ClauseEditorModal from '../components/ClauseEditor.jsx';
import PdfViewer from './search/PdfViewer.jsx';
import { TYPE_STYLES } from './search/clauseRender.jsx';
import { copyText, copyRich } from '../copy.js';
import { wrapClauseTables, inlineTableStyles, extractTableForCopy } from '../clauseHtml.js';
import { api } from '../api.js';
import './reader/reader.css';

function stripHtml(h) {
  const d = document.createElement('div');
  d.innerHTML = h || '';
  return (d.textContent || '').replace(/\s+/g, ' ').trim();
}

const TYPE_ORDER = ['ACT', 'REGULATION', 'MASTER_CIRCULAR', 'MASTER', 'CIRCULAR', 'GUIDELINE', 'UNKNOWN'];

function fmtDate(d) {
  if (!d) return null;
  try { return new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }); }
  catch { return d; }
}

// ---------------------------------------------------------------------------
// Document picker — shown at /read with no ?source.
// ---------------------------------------------------------------------------
function DocPicker() {
  const toast = useToast();
  const [docs, setDocs] = useState(null);
  const [q, setQ] = useState('');

  useEffect(() => {
    api.get('/clause/readable').then((d) => setDocs(d.docs || []))
      .catch((e) => { toast.error(e.message || 'Could not load documents'); setDocs([]); });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const groups = useMemo(() => {
    if (!docs) return [];
    const needle = q.trim().toLowerCase();
    const filtered = needle ? docs.filter((d) => d.source.toLowerCase().includes(needle)) : docs;
    const by = {};
    filtered.forEach((d) => { const t = (d.type || 'UNKNOWN').toUpperCase(); (by[t] ||= []).push(d); });
    const order = [...TYPE_ORDER.filter((t) => by[t]), ...Object.keys(by).filter((t) => !TYPE_ORDER.includes(t))];
    return order.map((t) => ({ type: t, style: TYPE_STYLES[t] || TYPE_STYLES.UNKNOWN, docs: by[t] }));
  }, [docs, q]);

  if (docs === null) return (<><PageHeader fullForm="Regulatory Library" title="Read Documents" scope="Read any regulation end-to-end" /><div className="page-body"><PageLoading /></div></>);

  return (
    <>
      <PageHeader fullForm="Regulatory Library" title="Read Documents" scope="Read any regulation, circular or guideline end-to-end" scopeDot={false} />
      <div className="page-body">
        <p className="reader-intro">Open a document to read it in full — a clause index on the left, the complete text on the right. Jump to any clause, or read straight through.</p>
        <div className="reader-pick-search">
          <i className="fas fa-magnifying-glass" />
          <input className="input" placeholder="Filter documents…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {groups.length === 0 ? <EmptyState icon="fa-folder-open">No documents to read.</EmptyState>
          : groups.map((g) => (
            <div className="reader-pick-group" key={g.type}>
              <div className="reader-pick-band" style={{ background: g.style.bg, color: g.style.color, borderLeftColor: g.style.bar }}>{g.style.label}</div>
              <div className="reader-pick-grid">
                {g.docs.map((d) => {
                  const repealed = (d.status || '').toLowerCase() === 'repealed';
                  return (
                    <Link key={d.source} className="reader-pick-card" to={`/read?source=${encodeURIComponent(d.source)}`}>
                      <span className="reader-pick-name">{d.source}</span>
                      <span className="reader-pick-meta">
                        <span>{d.clauses} clauses</span>
                        <span className={`reader-pick-st ${repealed ? 'is-repealed' : 'is-active'}`}>{repealed ? 'Repealed' : 'Active'}</span>
                      </span>
                      {d.effective_date && <span className="reader-pick-date">from {fmtDate(d.effective_date)}</span>}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Full-document reader — /read?source=…(&c=clauseId)
// ---------------------------------------------------------------------------
function DocReader({ source, target }) {
  const toast = useToast();
  const navigate = useNavigate();
  // Show a "Back" affordance only when we actually arrived from somewhere in-app
  // (e.g. the "Read in full" button on a search result) — not on a fresh deep link.
  const cameFromApp = typeof window !== 'undefined' && window.history.length > 1;
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin' || !!user?.is_admin;
  const [data, setData] = useState(null);
  const [active, setActive] = useState(null);
  const [tocQ, setTocQ] = useState('');
  // The index covers the content on a phone, so start it closed there.
  const [showToc, setShowToc] = useState(() => (typeof window === 'undefined' || window.innerWidth > 760));
  const [showPdf, setShowPdf] = useState(false);
  const [flagClause, setFlagClause] = useState(null);
  const [tocW, setTocW] = useState(300);   // index width, px (resizable)
  const [pdfW, setPdfW] = useState(42);     // pdf pane width, % (resizable)
  const [dragging, setDragging] = useState(false);
  const mainRef = useRef(null);
  const bodyRef = useRef(null);
  const secRefs = useRef({});      // clause id -> <section> element
  const tocRefs = useRef({});      // clause id -> toc <button>

  useEffect(() => {
    setData(null); setActive(null);
    api.get(`/clause/doc-read?source=${encodeURIComponent(source)}`)
      .then((d) => {
        const clauses = (d.clauses || []).map((c) => ({ ...c, _text: stripHtml(c.html) }));
        setData({ ...d, clauses }); setActive(clauses[0]?.id || null);
      })
      .catch((e) => { toast.error(e.message || 'Could not open document'); setData({ clauses: [] }); });
  }, [source]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Drag the index / PDF dividers, like the clause editor's resizable panes.
  function startResize(which) {
    return (e) => {
      e.preventDefault();
      const rect = bodyRef.current?.getBoundingClientRect(); if (!rect) return;
      setDragging(true);
      const onMove = (ev) => {
        if (which === 'toc') setTocW(Math.min(480, Math.max(180, ev.clientX - rect.left)));
        else setPdfW(Math.min(68, Math.max(22, ((rect.right - ev.clientX) / rect.width) * 100)));
      };
      const onUp = () => { setDragging(false); window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); document.body.style.userSelect = ''; };
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
    };
  }

  async function copyClause(c) {
    const text = c._text || stripHtml(c.html);
    // Copy rich HTML so tables/bold paste into Word/Docs; plain text is the fallback.
    const ok = c.html ? await copyRich(inlineTableStyles(c.html), text) : await copyText(text);
    if (ok) toast.success('Clause copied!');
    else toast.error('Could not copy');
  }

  // Scroll-spy: the clause nearest the top of the reading pane is "active".
  useEffect(() => {
    if (!data?.clauses?.length || !mainRef.current) return undefined;
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) setActive(e.target.dataset.cid); });
    }, { root: mainRef.current, rootMargin: '0px 0px -72% 0px', threshold: 0 });
    Object.values(secRefs.current).forEach((el) => el && obs.observe(el));
    return () => obs.disconnect();
  }, [data]);

  // Keep the active clause visible in the table of contents.
  useEffect(() => {
    const el = active && tocRefs.current[active];
    if (el) el.scrollIntoView({ block: 'nearest' });
  }, [active]);

  // Deep link (?c=clauseId): jump to and flash the target clause once loaded.
  useEffect(() => {
    if (!data?.clauses?.length || !target) return;
    const el = secRefs.current[target];
    if (el) {
      el.scrollIntoView({ block: 'start' });
      el.classList.add('reader-flash');
      setActive(target);
      const t = setTimeout(() => el.classList.remove('reader-flash'), 1800);
      return () => clearTimeout(t);
    }
    return undefined;
  }, [data, target]);

  const goto = (id) => {
    const el = secRefs.current[id];
    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); setActive(id); }
    if (typeof window !== 'undefined' && window.innerWidth <= 760) setShowToc(false);  // free the screen on mobile
  };

  if (!data) return (<div className="reader-shell"><PageHeader fullForm="Regulatory Library" title="Reader" scope={source} /><div className="page-body"><PageLoading /></div></div>);

  const clauses = data.clauses || [];
  const repealed = (data.status || '').toLowerCase() === 'repealed';
  const needle = tocQ.trim().toLowerCase();
  // Full-text: match the clause id, its title, or anywhere in the body.
  const tocClauses = needle
    ? clauses.filter((c) => c.id.toLowerCase().includes(needle)
        || (c.title || '').toLowerCase().includes(needle)
        || (c._text || '').toLowerCase().includes(needle))
    : clauses;
  const isEditor = isAdmin || user?.role === 'editor';
  const registerRef = (id, el) => { secRefs.current[id] = el; };

  return (
    <div className="reader-shell">
      <PageHeader fullForm="Regulatory Library" title="Reader" scope={source} scopeDot={false} showZoom>
        {cameFromApp && <button className="btn btn-ghost btn-sm" onClick={() => navigate(-1)} title="Return to where you came from (e.g. your search results)"><i className="fas fa-arrow-left" /> Back</button>}
        <Link className="btn btn-ghost btn-sm" to="/read"><i className="fas fa-folder-open" /> All documents</Link>
        <button className="btn btn-ghost btn-sm" onClick={() => setShowToc((s) => !s)}><i className={`fas ${showToc ? 'fa-list-ul' : 'fa-list'}`} /> {showToc ? 'Hide index' : 'Index'}</button>
        {data.pdf_url && <button className="btn btn-ghost btn-sm" onClick={() => setShowPdf((s) => !s)}><i className={`fas ${showPdf ? 'fa-eye-slash' : 'fa-file-pdf'}`} /> {showPdf ? 'Hide PDF' : 'Original PDF'}</button>}
      </PageHeader>

      <div className="reader-statusbar">
        <span className="reader-type-tag">{(data.type || 'Document').replace(/_/g, ' ')}</span>
        <span className={`reader-status ${repealed ? 'st-repealed' : 'st-active'}`}>
          <i className={`fas ${repealed ? 'fa-ban' : 'fa-circle-check'}`} /> {repealed ? 'Repealed' : 'Active'}
        </span>
        {data.effective_date && <span className="reader-eff">Effective {fmtDate(data.effective_date)}</span>}
        <span className="reader-count">{clauses.length} clauses</span>
      </div>

      <div className={`reader-body ${dragging ? 'dragging' : ''}`} ref={bodyRef}>
        {showToc && (
          <>
            <div className="reader-toc-backdrop" onClick={() => setShowToc(false)} />
            <aside className="reader-toc" style={{ flexBasis: tocW }}>
              <div className="reader-toc-search">
                <i className="fas fa-magnifying-glass" />
                <input placeholder="Search this document…" value={tocQ} onChange={(e) => setTocQ(e.target.value)} />
                <button className="reader-toc-close" onClick={() => setShowToc(false)} title="Close index" aria-label="Close index"><i className="fas fa-xmark" /></button>
              </div>
              <div className="reader-toc-list">
                {needle && <div className="reader-toc-hint">{tocClauses.length} clause{tocClauses.length === 1 ? '' : 's'} match “{tocQ.trim()}”</div>}
                {tocClauses.length === 0 ? <div className="reader-toc-empty">No matching clause.</div>
                  : tocClauses.map((c) => (
                    <button
                      key={c.id}
                      ref={(el) => { tocRefs.current[c.id] = el; }}
                      className={`reader-toc-item ${active === c.id ? 'is-active' : ''}`}
                      onClick={() => goto(c.id)}
                      title={c.title}
                    >
                      <span className="reader-toc-id">{c.id}</span>
                      {c.title && <span className="reader-toc-title">{c.title}</span>}
                    </button>
                  ))}
              </div>
            </aside>
            <div className="reader-resizer" onMouseDown={startResize('toc')} title="Drag to resize the index" />
          </>
        )}

        <main className="reader-main" ref={mainRef}>
          {clauses.length === 0 ? <EmptyState icon="fa-file-circle-xmark">This document has no readable clauses.</EmptyState>
            : (
              <article className="reader-doc">
                <h1 className="reader-doc-title">{source}</h1>
                {clauses.map((c) => (
                  <ReaderClause
                    key={c.id}
                    c={c}
                    source={source}
                    isActive={active === c.id}
                    isEditor={isEditor}
                    pdfUrl={data.pdf_url}
                    status={data.status}
                    effectiveDate={data.effective_date}
                    registerRef={registerRef}
                    onCopy={copyClause}
                    onFlag={(cl) => setFlagClause(cl)}
                  />
                ))}
              </article>
            )}
        </main>

        {showPdf && data.pdf_url && (
          <>
            <div className="reader-resizer" onMouseDown={startResize('pdf')} title="Drag to resize the PDF" />
            <aside className="reader-pdf" style={{ flexBasis: `${pdfW}%` }}>
              <button className="reader-pdf-close" onClick={() => setShowPdf(false)}><i className="fas fa-xmark" /> Close PDF</button>
              <PdfViewer url={data.pdf_url} />
            </aside>
          </>
        )}
      </div>

      {flagClause && (
        <FlagModal kind="clause"
          target={`${source} · ${flagClause.title || flagClause.id} · Clause ${flagClause.id}`}
          detail={flagClause._text}
          onClose={() => setFlagClause(null)} />
      )}
    </div>
  );
}

// A single clause in the reader: content + per-clause actions (copy, flag,
// edit tags for editors, open source PDF).
function ReaderClause({ c, source, isActive, isEditor, pdfUrl, status, effectiveDate, registerRef, onCopy, onFlag }) {
  const toast = useToast();
  const [tags, setTags] = useState(c.tags || []);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [html, setHtml] = useState(c.html);
  const [editClause, setEditClause] = useState(false);

  const startEdit = () => { setDraft(tags.join(', ')); setEditing(true); };
  async function saveTags() {
    const next = draft.split(',').map((t) => t.trim()).filter(Boolean);
    setSaving(true);
    try {
      const r = await api.post('/clause/retag', { id: c.id, source, tags: next });
      setTags(r.tags || next); setEditing(false);
      window.dispatchEvent(new CustomEvent('iris:tags-changed'));
      toast.success('Tags updated');
    } catch (e) { toast.error(e.message || 'Could not update tags'); }
    finally { setSaving(false); }
  }

  return (
    <section data-cid={c.id} ref={(el) => registerRef(c.id, el)} className={`reader-clause ${isActive ? 'is-active' : ''}`}>
      <div className="reader-clause-head">
        <span className="reader-clause-id">{c.id}</span>
        {(() => {
          const rep = (status || '').toLowerCase() === 'repealed';
          return (
            <span className={`reader-clause-status ${rep ? 'cs-repealed' : 'cs-active'}`}>
              <i className={`fas ${rep ? 'fa-ban' : 'fa-circle-check'}`} />
              {rep ? 'Repealed' : `Active${effectiveDate ? ` · from ${fmtDate(effectiveDate)}` : ''}`}
            </span>
          );
        })()}
        <span className="reader-clause-actions">
          {pdfUrl && <a className="rc-btn" href={pdfUrl} target="_blank" rel="noreferrer" title="Open source PDF"><i className="fas fa-file-pdf" /></a>}
          <button className="rc-btn" title="Copy clause" onClick={() => onCopy(c)}><i className="far fa-copy" /></button>
          {isEditor && <button className="rc-btn" title="Edit clause text" onClick={() => setEditClause(true)}><i className="fas fa-pen-to-square" /></button>}
          {isEditor && <button className="rc-btn" title="Edit tags" onClick={startEdit}><i className="fas fa-tags" /></button>}
          <button className="rc-btn" title="Flag this clause" onClick={() => onFlag(c)}><i className="fas fa-flag" /></button>
        </span>
      </div>
      <div className="clause-html" onClick={async (e) => {
        const t = extractTableForCopy(e.target);
        if (!t) return;
        const ok = await copyRich(t.html, t.text);
        toast[ok ? 'success' : 'error'](ok ? 'Table copied!' : 'Could not copy table');
      }} dangerouslySetInnerHTML={{ __html: wrapClauseTables(html) }} />
      {editing ? (
        <div className="reader-tag-edit">
          <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="comma, separated, tags"
            onKeyDown={(e) => { if (e.key === 'Enter') saveTags(); if (e.key === 'Escape') setEditing(false); }} />
          <button className="btn btn-primary btn-sm" disabled={saving} onClick={saveTags}>{saving ? <Spinner size={12} color="#fff" /> : 'Save'}</button>
          <button className="btn btn-ghost btn-sm" onClick={() => setEditing(false)}>Cancel</button>
        </div>
      ) : tags.length > 0 && (
        <div className="reader-clause-tags">{tags.map((t) => <span key={t} className="reader-tag">{t}</span>)}</div>
      )}
      {editClause && (
        <ClauseEditorModal
          clause={{ id: c.id, source }}
          onClose={() => setEditClause(false)}
          onSaved={(h) => { setHtml(h); c._text = stripHtml(h); setEditClause(false); }} />
      )}
    </section>
  );
}

export default function Reader() {
  const [params] = useSearchParams();
  const source = params.get('source');
  const target = params.get('c');
  return source ? <DocReader source={source} target={target} /> : <DocPicker />;
}
