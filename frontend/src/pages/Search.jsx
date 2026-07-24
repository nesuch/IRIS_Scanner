import { memo, startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import ClauseEditorModal from '../components/ClauseEditor.jsx';
import { api } from '../api.js';
import { copyText, copyRich } from '../copy.js';
import { wrapClauseTables, inlineTableStyles, extractTableForCopy } from '../clauseHtml.js';
import { TYPE_STYLES, ClauseBody, groupByType, highlightHtml } from './search/clauseRender.jsx';
import { CLAUSE_SLASH_COMMANDS, parseSlash as parseSlashWith } from './search/slash.js';
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

// Slash-command palette (Claude-style): typing "/" opens a menu of search modes.
const SLASH_COMMANDS = CLAUSE_SLASH_COMMANDS;
const parseSlash = (v) => parseSlashWith(v, SLASH_COMMANDS);

// Dates as DD MMM YYYY (e.g. 29 May 2024).
function fmtClauseDate(d) {
  if (!d) return '';
  const dt = new Date(d);
  if (Number.isNaN(dt.getTime())) return d;
  return dt.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
}

// Clause body + admin controls (rich-edit + tags). Holds a local HTML override
// so an edit shows immediately without re-running the search.
function ClauseContent({ m, keywords, phraseKeywords }) {
  const { user } = useAuth();
  const toast = useToast();
  const isAdmin = user?.role === 'admin' || !!user?.is_admin;
  const isEditor = isAdmin || user?.role === 'editor';
  const [html, setHtml] = useState(m.html || '');
  const [editing, setEditing] = useState(false);
  // highlightHtml() runs a DOMParser + tree-walks and wrapClauseTables() runs regex
  // passes — both expensive. Memoise so they run ONCE per clause, not on every
  // re-render (each card's ResizeObserver flips `overflowing` state, which would
  // otherwise re-run this for all ~50 cards and freeze the results view for seconds).
  const renderedHtml = useMemo(
    () => (html ? wrapClauseTables(highlightHtml(html, keywords, phraseKeywords)) : ''),
    [html, keywords, phraseKeywords],
  );
  async function onClauseAreaClick(e) {
    const t = extractTableForCopy(e.target);
    if (!t) return;
    const ok = await copyRich(t.html, t.text);
    toast[ok ? 'success' : 'error'](ok ? 'Table copied!' : 'Could not copy table');
  }
  return (
    <>
      {html
        ? <div className="clause-html" onClick={onClauseAreaClick} dangerouslySetInnerHTML={{ __html: renderedHtml }} />
        : <ClauseBody text={m.raw_text} keywords={keywords} phraseKeywords={phraseKeywords} />}
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
      window.dispatchEvent(new CustomEvent('iris:tags-changed'));   // refresh autocomplete vocab
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

// Rough pre-filter for whether to show the Expand-all/Collapse-all toolbar.
// Real collapsibility is decided per-card by measuring actual rendered height
// (below); this only governs the toolbar's visibility, so a coarse char check
// is fine.
const TOOLBAR_MIN_CHARS = 1400;

// One result card. The real, fully-formatted clause is rendered inside a
// height-clamped box (~20 lines) with a fade-out; if it actually overflows that
// height we offer Expand. Short clauses don't overflow, so they show whole with
// no control. A global expand/collapse-all signal (globalSeq bumps each click)
// overrides per-card state.
function ClauseCard({ m, keywords, phraseKeywords, copy, onOpenPane, onFlag, globalExpand, globalSeq }) {
  const st = TYPE_STYLES[m.type] || TYPE_STYLES.UNKNOWN;
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const cardRef = useRef(null);
  const clampRef = useRef(null);

  // Measure whether the clamped clause is taller than the clamp. Only meaningful
  // while collapsed (when expanded the clamp is lifted); once we've seen it
  // overflow, the collapse controls stay available. Re-checks on resize and when
  // the stored HTML (tables/images) finishes laying out.
  useEffect(() => {
    if (expanded) return undefined;
    const el = clampRef.current;
    if (!el) return undefined;
    // Measure overflow OFF the critical path: reading scrollHeight forces a
    // synchronous layout, and doing it for every card as it mounts (plus a
    // ResizeObserver per card) thrashes layout and freezes a large result list.
    // Defer the read to an idle/animation frame (batched, after paint) and use a
    // lightweight ResizeObserver whose callback is likewise rAF-debounced.
    let raf = 0;
    const check = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setOverflowing(el.scrollHeight > el.clientHeight + 4));
    };
    check();
    const ro = new ResizeObserver(check);
    ro.observe(el);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [expanded, m.id, m.html]);

  useEffect(() => {
    if (globalExpand !== null) setExpanded(globalExpand);
  }, [globalSeq]);   // eslint-disable-line react-hooks/exhaustive-deps

  const collapse = () => {
    setExpanded(false);
    // Bring the card's top back into view so collapsing from deep inside a long
    // clause doesn't leave you staring at blank space.
    requestAnimationFrame(() => cardRef.current?.scrollIntoView({ block: 'nearest' }));
  };

  return (
    <div className="clause-card" ref={cardRef}>
      <div className="clause-title" style={{ color: st.color }}>
        <i className="fas fa-file-lines" style={{ color: st.bar }} />
        <span className="clause-source-name">{m.source}</span>
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
      </div>
      <div className="clause-meta">
        <span className="clause-header">{m.header}</span>
        <span className="sep">|</span>
        <span className="clause-id">Clause: {m.id}</span>
        <span className="clause-actions">
          <Link className="pane-btn" title="Read this clause in the full document" to={`/read?source=${encodeURIComponent(m.source)}&c=${encodeURIComponent(m.id)}`}><i className="fas fa-book-open" /> Read in full</Link>
          {m.pdf_url && (
            <button className="pane-btn" title="Open PDF in side pane" onClick={() => onOpenPane(m)}><i className="fas fa-table-columns" /> Pane</button>
          )}
          {m.pdf_url && (
            <a className="pdf-btn" href={m.pdf_url} target="_blank" rel="noreferrer"><i className="fas fa-file-pdf" /> PDF</a>
          )}
          {/* Annexure bundle (e.g. the Cyber Security Guidelines 2026 forms, which
              IRDAI ships as a ZIP). Offered on EVERY clause of the document, so it
              is reachable from whichever clause the search landed on. */}
          {m.bundle && (
            <a className="pdf-btn" href={m.bundle.url} download
               title={`Download the annexures for this document (${m.bundle.name})`}>
              <i className="fas fa-file-zipper" /> Annexures
            </a>
          )}
          <button className="copy-btn" title="Copy clause" onClick={() => copy(m.html, m.raw_text)}><i className="far fa-copy" /></button>
          <button className="flag-btn" title="Flag this clause" onClick={() => onFlag(m)}><i className="fas fa-flag" /></button>
        </span>
      </div>
      <div className={`clause-clamp ${expanded ? 'open' : ''} ${!expanded && overflowing ? 'has-more' : ''}`} ref={clampRef}>
        {expanded && overflowing && (
          // Floating, sticky collapse control — reachable from anywhere inside a
          // long expanded clause without scrolling to the top or bottom.
          <div className="clause-collapse-sticky">
            <button className="clause-collapse-orb" onClick={collapse} title="Collapse this clause" aria-label="Collapse this clause">
              <i className="fas fa-chevron-up" />
            </button>
          </div>
        )}
        <ClauseContent m={m} keywords={keywords} phraseKeywords={phraseKeywords} />
      </div>
      {!expanded && overflowing && (
        <button className="clause-expand-btn" onClick={() => setExpanded(true)}>
          <i className="fas fa-chevron-down" /> Expand clause
        </button>
      )}
      {expanded && overflowing && (
        <button className="clause-expand-btn collapse" onClick={collapse}>
          <i className="fas fa-chevron-up" /> Collapse
        </button>
      )}
    </div>
  );
}

// Memoised: a settled result block (with its 30–60 highlight-heavy clause cards)
// must NOT re-render every time the user types the NEXT query in the search box.
// Its props (the stored `resp` + the stable callbacks below) don't change while
// typing, so React.memo skips the whole subtree — killing the per-keystroke lag.
const ResultCards = memo(function ResultCards({ resp, onChip, onOpenPane, onFlag }) {
  const toast = useToast();
  // Global expand/collapse-all: allState is the target (true/false) or null
  // (untouched — cards keep their size-based default); allSeq forces cards to
  // re-apply it even if the same button is clicked twice.
  const [allState, setAllState] = useState(null);
  const [allSeq, setAllSeq] = useState(0);
  const expandAll = (v) => { setAllState(v); setAllSeq((s) => s + 1); };
  const copy = async (html, text) => {
    // Copy rich HTML when available so tables/bold paste into Word/Docs.
    const ok = html ? await copyRich(inlineTableStyles(html), text) : await copyText(text);
    if (ok) toast.success('Clause copied!');
    else toast.error('Could not copy');
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

  const phraseGroups = groupByType(resp.phrase_matches || []);
  const hasPhrase = (resp.phrase_matches || []).length > 0;
  const groups = groupByType(resp.matches || []);
  const hasMatches = (resp.matches || []).length > 0;
  const contentGroups = groupByType(resp.content_matches || []);
  const hasContent = (resp.content_matches || []).length > 0;

  const card = (m, mi) => (
    <ClauseCard key={`${m.id}-${mi}`} m={m} keywords={resp.highlight || []}
      phraseKeywords={resp.highlight_phrase || []}
      copy={copy} onOpenPane={onOpenPane} onFlag={onFlag}
      globalExpand={allState} globalSeq={allSeq} />
  );

  // Only worth offering expand/collapse-all when something is likely collapsible
  // (coarse char check; each card still decides for itself by measured height).
  const hasCollapsible = [...(resp.phrase_matches || []), ...(resp.matches || []), ...(resp.content_matches || [])]
    .some((m) => String(m.raw_text || '').length > TOOLBAR_MIN_CHARS);

  return (
    <div>
      {resp.kind === 'deep_scan' && hasMatches && (
        <div className="iris-analysis">Deep Scan results in <strong>{resp.module.toUpperCase()}</strong>: <strong>{(resp.keywords || []).join(', ')}</strong></div>
      )}
      {resp.kind === 'tags' && hasMatches && (
        <div className="iris-foundvia">Found via <strong>Tags</strong>: {(resp.keywords || []).join(', ')}</div>
      )}
      {resp.note && !hasPhrase && !hasMatches && !hasContent && <p className="iris-msg">{resp.note} {flagNoResult}</p>}

      {hasCollapsible && (hasPhrase || hasMatches || hasContent) && (
        <div className="results-tools">
          <span className="results-tools-hint">Long clauses are previewed —</span>
          <button className="results-tool-btn" onClick={() => expandAll(true)}><i className="fas fa-angles-down" /> Expand all</button>
          <button className="results-tool-btn" onClick={() => expandAll(false)}><i className="fas fa-angles-up" /> Collapse all</button>
        </div>
      )}

      {/* Top tier: clauses that literally contain the typed phrase. */}
      {hasPhrase && (
        <div className="phrase-tier">
          <div className="phrase-tier-band">
            <i className="fas fa-bullseye" /> Best matches for “{resp.phrase || resp.query_label}” — {(resp.phrase_matches || []).length} clause{(resp.phrase_matches || []).length === 1 ? '' : 's'}
          </div>
          {phraseGroups.map((g, gi) => {
            const st = TYPE_STYLES[g.type] || TYPE_STYLES.UNKNOWN;
            return (
              <div key={gi}>
                <div className="type-band" style={{ background: st.bg, color: st.color, borderLeftColor: st.bar }}>{st.label}</div>
                {g.items.map(card)}
              </div>
            );
          })}
        </div>
      )}

      {hasPhrase && hasMatches && (
        <div className="phrase-tier-divider">Other concept / tag matches</div>
      )}
      {groups.map((g, gi) => {
        const st = TYPE_STYLES[g.type] || TYPE_STYLES.UNKNOWN;
        return (
          <div key={gi}>
            <div className="type-band" style={{ background: st.bg, color: st.color, borderLeftColor: st.bar }}>{st.label}</div>
            {g.items.map(card)}
          </div>
        );
      })}

      {/* Windows-style content tier: matched in the clause text, no manual deep scan. */}
      {hasContent && (
        <div className="content-tier">
          <div className="content-tier-band">
            <i className="fas fa-align-left" /> Also found in the text of {(resp.content_matches || []).length} clause{(resp.content_matches || []).length === 1 ? '' : 's'}
            {hasMatches && <span className="content-tier-sub"> (beyond the tagged matches above)</span>}
          </div>
          {contentGroups.map((g, gi) => {
            const st = TYPE_STYLES[g.type] || TYPE_STYLES.UNKNOWN;
            return (
              <div key={gi}>
                <div className="type-band" style={{ background: st.bg, color: st.color, borderLeftColor: st.bar }}>{st.label}</div>
                {g.items.map(card)}
              </div>
            );
          })}
        </div>
      )}

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
});

// Persist each module's search conversation for the tab session so navigating
// away (e.g. "Read in full" → Reader) and back doesn't wipe the results. Only
// settled blocks (with a response or error) are kept — never a pending live one.
const SESSION_KEY = (m) => `iris_search_session_${m}`;
function loadSession(m) {
  try { const r = sessionStorage.getItem(SESSION_KEY(m)); return r ? JSON.parse(r) : []; }
  catch { return []; }
}
function saveSession(m, hist) {
  try { sessionStorage.setItem(SESSION_KEY(m), JSON.stringify(hist.filter((x) => x.response || x.error))); }
  catch { /* quota / disabled — non-fatal */ }
}

export default function Search({ module }) {
  const toast = useToast();
  // Custom (admin-created) departments aren't in MODULE_META — resolve their
  // friendly label from the departments registry for the page title/scope.
  const [deptLabel, setDeptLabel] = useState('');
  useEffect(() => {
    if (MODULE_META[module]) { setDeptLabel(''); return; }
    api.get('/departments').then((d) => {
      const found = (d.departments || []).find((x) => x.key.toLowerCase() === module);
      setDeptLabel(found ? found.label : '');
    }).catch(() => {});
  }, [module]);
  const meta = MODULE_META[module] || {
    title: `${deptLabel || module} Department`,
    icon: 'fa-folder',
    scope: `Regulatory Framework (${deptLabel || module})`,
  };
  const [history, setHistory] = useState(() => loadSession(module));
  const mountedModuleRef = useRef(module);   // distinguishes initial mount from a real module switch
  const [query, setQuery] = useState('');
  const [cmd, setCmd] = useState(null);   // active slash-command ('deep' | 'clause'); query holds its argument
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [activeSugg, setActiveSugg] = useState(-1);   // keyboard-highlighted suggestion
  const [docGroups, setDocGroups] = useState([]);
  const [docTags, setDocTags] = useState({});
  const [tagClauses, setTagClauses] = useState({}); // normalised tag -> [{id, source}]
  const [selected, setSelected] = useState(() => new Set());
  const [docFilterOpen, setDocFilterOpen] = useState(false);
  const [docQuery, setDocQuery] = useState('');   // filter the doc-picker list by name
  const [pdfPane, setPdfPane] = useState(null); // { url, source }
  const [flagClause, setFlagClause] = useState(null); // clause being flagged
  const [paneWidth, setPaneWidth] = useState(46); // % width of the PDF pane
  const [dragging, setDragging] = useState(false);
  const chatRef = useRef(null);
  const lastUserRef = useRef(null);
  const docFilterRef = useRef(null);
  const searchMainRef = useRef(null);
  const inputAreaRef = useRef(null);
  const suggBoxRef = useRef(null);
  const liveIdRef = useRef(null);     // id of the current live (as-you-type) result block, or null
  const liveSeqRef = useRef(0);       // guards against out-of-order live responses
  const prevLenRef = useRef(0);       // history length, to scroll only on NEW blocks

  // Lock the document to the viewport (fixed bottom bar; only results scroll).
  useEffect(() => {
    document.documentElement.classList.add('app-fixed');
    return () => document.documentElement.classList.remove('app-fixed');
  }, []);

  // On a real module switch, restore that module's saved session (not the same
  // as the initial mount, where the lazy initializer already loaded it).
  useEffect(() => {
    if (mountedModuleRef.current === module) return;   // initial mount — keep restored session
    mountedModuleRef.current = module;
    liveIdRef.current = null; prevLenRef.current = 0;
    setHistory(loadSession(module)); setQuery(''); setSuggestions([]); setDocFilterOpen(false); setPdfPane(null);
  }, [module]);

  // Persist the conversation so it survives navigation away (e.g. to the Reader)
  // and back within the same browser tab.
  useEffect(() => { saveSession(module, history); }, [history, module]);

  // Re-clicking the active module in the sidebar clears the conversation.
  useEffect(() => {
    const onReclick = () => { liveIdRef.current = null; prevLenRef.current = 0; setHistory([]); saveSession(module, []); setQuery(''); setSuggestions([]); setPdfPane(null); };
    window.addEventListener('iris:reclick', onReclick);
    return () => window.removeEventListener('iris:reclick', onReclick);
  }, [module]);

  // Load the documents available to this module (with their tags) and select all by default.
  useEffect(() => {
    api.get(`/docs?module=${module}`).then((d) => {
      const groups = d.groups || [];
      setDocGroups(groups);
      setDocTags(d.doc_tags || {});
      setTagClauses(d.tag_clauses || {});
      const all = new Set();
      groups.forEach((g) => g.docs.forEach((s) => all.add(s)));
      setSelected(all);
    }).catch(() => { setDocGroups([]); setDocTags({}); setTagClauses({}); setSelected(new Set()); });
  }, [module]);

  // After a clause is re-tagged in-app, refresh the tag vocabulary (preserving the
  // current doc selection) so autocomplete reflects the edit without a page reload.
  useEffect(() => {
    const refresh = () => api.get(`/docs?module=${module}`)
      .then((d) => { setDocGroups(d.groups || []); setDocTags(d.doc_tags || {}); setTagClauses(d.tag_clauses || {}); })
      .catch(() => {});
    window.addEventListener('iris:tags-changed', refresh);
    return () => window.removeEventListener('iris:tags-changed', refresh);
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

  // Tapping anywhere outside the search bar (e.g. the sidebar toggle on mobile)
  // dismisses the suggestions dropdown so it never overlays the sidebar.
  useEffect(() => {
    if (!suggestions.length) return undefined;
    const onDown = (e) => {
      if (inputAreaRef.current && !inputAreaRef.current.contains(e.target)) setSuggestions([]);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('touchstart', onDown);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('touchstart', onDown); };
  }, [suggestions.length]);

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
    // Scroll so the latest question sits near the top — only when a NEW block is
    // added, not on every in-place live-result update (that would jump on each key).
    if (history.length > prevLenRef.current && lastUserRef.current && chatRef.current) {
      const top = lastUserRef.current.offsetTop - 16;
      chatRef.current.scrollTop = Math.max(top, 0);
    }
    prevLenRef.current = history.length;
  }, [history]);

  // live=true: as-you-type — replace the current live block in place (no chat
  // spam) and apply only the newest response. live=false: a committed search
  // (Enter / suggestion / deep-scan chip) appended as a normal block.
  async function runSearch(q, displayLabel, { live = false } = {}) {
    if (!q.trim()) return;
    if (!live && busy) return;
    let entryId;
    if (live && liveIdRef.current) {
      entryId = liveIdRef.current;
      setHistory((h) => h.map((x) => (x.id === entryId ? { ...x, query: displayLabel ?? q, response: null, error: null } : x)));
    } else {
      entryId = Math.random().toString(36).slice(2);
      setHistory((h) => [...h, { id: entryId, query: displayLabel ?? q, response: null }]);
      if (live) liveIdRef.current = entryId;
      else liveIdRef.current = null;   // a committed search ends any live session
    }
    const seq = live ? (liveSeqRef.current += 1) : null;
    if (!live) setBusy(true);
    try {
      const body = { module, query: q };
      const total = docGroups.reduce((n, g) => n + g.docs.length, 0);
      if (docGroups.length && selected.size < total) body.sources = [...selected]; // subset only; all => omit
      const resp = await api.post('/search', body);
      if (live && seq !== liveSeqRef.current) return;   // a newer keystroke superseded this
      // Committing a live response renders 30-70 clause cards at once. This runs in
      // a fetch .then() (not a React event), so React 18 flushes it SYNCHRONOUSLY
      // and blocks the main thread for the whole commit — which is why a key pressed
      // just as results land takes a beat to appear in the box. startTransition marks
      // this render as interruptible, so the urgent input update preempts it and the
      // letter shows immediately; the cards paint a frame later. Only the live path
      // needs this — a committed (Enter) search has nothing racing it.
      const applyResp = () =>
        setHistory((h) => h.map((x) => (x.id === entryId ? { ...x, response: resp } : x)));
      if (live) startTransition(applyResp);
      else applyResp();
    } catch (err) {
      if (live && seq !== liveSeqRef.current) return;
      setHistory((h) => h.map((x) => (x.id === entryId ? { ...x, error: err.message } : x)));
      if (!live) toast.error(err.message || 'Search failed');
    } finally {
      if (!live) setBusy(false);
    }
  }

  function onSubmit(e) {
    e.preventDefault();
    const q = query.trim();
    setSuggestions([]);
    // Command mode (coloured pill): the input holds the command's argument.
    if (cmd) {
      if (!q) return;
      setQuery(''); setCmd(null);
      if (cmd === 'deep') runSearch('__DEEP_SCAN__:' + q, 'Deep Scan');
      else if (cmd === 'clause') runSearch('/' + q.replace(/^\/+/, ''));
      return;
    }
    if (!q) return;
    // Live results are already on screen — Enter just commits the block & clears.
    if (liveIdRef.current) { liveIdRef.current = null; setQuery(''); return; }
    // Slash-command routing: /deep runs a deep scan, /clause does a clause lookup.
    const p = parseSlash(q);
    if (p && p.kind === 'cmd') {
      const arg = (p.arg || '').trim();
      if (!arg) return;
      setQuery('');
      if (p.cmd === 'deep') { runSearch('__DEEP_SCAN__:' + arg, 'Deep Scan'); return; }
      if (p.cmd === 'clause') { runSearch('/' + arg.replace(/^\/+/, '')); return; }
    }
    setQuery('');
    runSearch(q);
  }

  // As-you-type live search: debounced, replacing one live block in place (no
  // chat spam). Skips clause-number mode, the data module, and very short queries.
  useEffect(() => {
    const q = query.trim();
    if (q.length < 3 || q.startsWith('/') || module === 'data' || cmd) {
      if (liveIdRef.current) {   // query cleared/too short — drop the pending live block
        const id = liveIdRef.current; liveIdRef.current = null;
        setHistory((h) => h.filter((x) => x.id !== id));
      }
      return undefined;
    }
    const t = setTimeout(() => { runSearch(q, undefined, { live: true }); }, 500);
    return () => clearTimeout(t);
  }, [query, module, selected, cmd]);   // eslint-disable-line react-hooks/exhaustive-deps

  // Stable callbacks (identity preserved across renders) so the memoised
  // ResultCards blocks don't re-render while the user types the next query.
  // runSearch is read through a ref so onChip needn't depend on it.
  const runSearchRef = useRef(null);
  runSearchRef.current = runSearch;
  const onChip = useCallback((payload) => {
    runSearchRef.current('__DEEP_SCAN__:' + payload, 'Deep Scan');
  }, []);
  const openPane = useCallback((m) => setPdfPane({ url: m.pdf_url, source: m.source }), []);
  const openFlag = useCallback((m) => setFlagClause(m), []);

  function onInput(e) {
    const val = e.target.value;
    setQuery(val);
    setActiveSugg(-1);
    // In command mode the input holds only the argument (a coloured pill shows the
    // command). For /clause, suggest clauses as they type; /deep waits for Enter.
    if (cmd) {
      if (cmd === 'clause' && val.replace(/[^a-z0-9]/gi, '').length >= 1) {
        const params = new URLSearchParams({ q: '/' + val.trim(), module });
        if (allDocs.length && selected.size < allDocs.length) [...selected].forEach((s) => params.append('source', s));
        api.get(`/clause-suggest?${params.toString()}`)
          .then((d) => setSuggestions((d.suggestions || []).map((s) => ({ ...s, clause: true }))))
          .catch(() => setSuggestions([]));
      } else {
        setSuggestions([]);
      }
      return;
    }
    // Slash commands: "/" opens a mode menu; "/clause <id>" and "/<id>" do a clause
    // lookup; "/deep <terms>" waits for Enter to run a deep scan.
    if (val.trimStart().startsWith('/')) {
      const p = parseSlash(val);
      if (p && p.kind === 'menu') {
        setSuggestions(p.list.map((c) => ({ ...c, command: true })));
        return;
      }
      if (p && p.kind === 'cmd' && p.cmd === 'deep') { setSuggestions([]); return; }
      // clause lookup — either "/clause <id>" (strip the command) or a bare "/<id>"
      const lookup = p && p.kind === 'cmd' && p.cmd === 'clause'
        ? '/' + p.arg.trim() : val.trim();
      const key = lookup.replace(/[^a-z0-9]/gi, '');
      if (key.length < 1) { setSuggestions([]); return; }
      const params = new URLSearchParams({ q: lookup, module });
      if (allDocs.length && selected.size < allDocs.length) [...selected].forEach((s) => params.append('source', s));
      api.get(`/clause-suggest?${params.toString()}`)
        .then((d) => setSuggestions((d.suggestions || []).map((s) => ({ ...s, clause: true }))))
        .catch(() => setSuggestions([]));
      return;
    }
    // Once the query is long enough for the live as-you-type results to show
    // (>=3 chars), hand off to them — don't pop the tag dropdown over the results.
    if (val.trim().length >= 3) { setSuggestions([]); return; }
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
    // A slash-command chip enters command mode: a coloured pill shows the command
    // and the input now holds just its argument.
    if (value && typeof value === 'object' && value.command) {
      setSuggestions([]);
      setActiveSugg(-1);
      setCmd(value.cmd);
      setQuery('');
      return;
    }
    // Click-and-go: a clause suggestion runs its lookup; a tag suggestion completes
    // the query and searches immediately (no separate "press search" step).
    if (value && typeof value === 'object' && value.clause) {
      setSuggestions([]);
      setActiveSugg(-1);
      setQuery('');
      setCmd(null);
      runSearch(`/${value.id}`, value.id);
      return;
    }
    const val = query;
    const lastSpace = val.lastIndexOf(' ');
    const next = (lastSpace === -1 ? value : val.substring(0, lastSpace + 1) + value).trim();
    setSuggestions([]);
    setActiveSugg(-1);
    setQuery(next);   // live debounce searches the completed term; Enter commits it
  }

  function onKeyDown(e) {
    const open = suggestions.length > 0;
    // Backspace on an empty argument (or Escape) exits command mode — drops the pill.
    if (cmd && ((e.key === 'Backspace' && query === '') || (e.key === 'Escape' && !open))) {
      e.preventDefault(); setCmd(null); setSuggestions([]); setActiveSugg(-1); return;
    }
    if (open && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      e.preventDefault();
      setActiveSugg((i) => {
        const n = suggestions.length;
        // The menu opens ABOVE the input, so ArrowUp starts at the item NEAREST the
        // search bar (bottom of the list) and moves up; ArrowDown starts at the top.
        // Hard stop at the ends (no wrap-around).
        if (e.key === 'ArrowUp') return i === -1 ? n - 1 : Math.max(i - 1, 0);
        return i === -1 ? 0 : Math.min(i + 1, n - 1);
      });
      return;
    }
    if (e.key === 'Escape' && open) { e.preventDefault(); setSuggestions([]); setActiveSugg(-1); return; }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (open && activeSugg >= 0 && activeSugg < suggestions.length) { selectSuggestion(suggestions[activeSugg]); return; }
      onSubmit(e);
    }
  }

  // Keep the keyboard-highlighted suggestion scrolled into view in a long list.
  useEffect(() => {
    if (activeSugg < 0 || !suggBoxRef.current) return;
    const el = suggBoxRef.current.children[activeSugg];
    if (el) el.scrollIntoView({ block: 'nearest' });
  }, [activeSugg]);

  const empty = history.length === 0;

  return (
    <div className="search-shell">
      <PageHeader fullForm="IRDAI's Regulatory Intelligence System" title={meta.title} scope={`Scope: ${meta.scope}`} showZoom />

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
                {item.response ? <ResultCards resp={item.response} onChip={onChip} onOpenPane={openPane} onFlag={openFlag} />
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

      <div className="input-area" ref={inputAreaRef}>
        {empty && !query && <div className="search-tip">Type a question — or press <b>/</b> for search modes (clause lookup, deep scan)</div>}
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
                  <div className="doc-filter-search">
                    <i className="fas fa-magnifying-glass" />
                    <input type="text" placeholder="Filter documents…" value={docQuery}
                      onChange={(e) => setDocQuery(e.target.value)} autoFocus />
                    {docQuery && <button type="button" className="doc-filter-search-clear" onClick={() => setDocQuery('')} aria-label="Clear">&times;</button>}
                  </div>
                  <label className="doc-row doc-row-all">
                    <input type="checkbox" checked={allSelected}
                      ref={(el) => { if (el) el.indeterminate = !allSelected && !noneSelected; }} onChange={toggleAll} />
                    <span>Select all</span>
                    <span className="doc-group-count">{selected.size}/{allDocs.length}</span>
                  </label>
                  <div className="doc-filter-body">
                    {docGroups.length === 0 && <div className="doc-empty">No documents in this module.</div>}
                    {(() => {
                      const q = docQuery.trim().toLowerCase();
                      const groups = q
                        ? docGroups.map((g) => ({ ...g, docs: g.docs.filter((d) => d.toLowerCase().includes(q)) })).filter((g) => g.docs.length)
                        : docGroups;
                      if (q && groups.length === 0) return <div className="doc-empty">No documents match “{docQuery}”.</div>;
                      return groups.map((g) => {
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
                      });
                    })()}
                  </div>
                </div>
              )}
            </div>
            {suggestions.length > 0 && (
              <div className="suggestions-box" ref={suggBoxRef}>
                {suggestions.map((s, i) => (
                  s && typeof s === 'object' && s.command ? (
                    <div className={`suggestion-item slash-cmd ${activeSugg === i ? 'is-active' : ''}`} key={i} onMouseEnter={() => setActiveSugg(i)} onMouseDown={(e) => { e.preventDefault(); selectSuggestion(s); }}>
                      <i className={`fas ${s.icon} slash-cmd-ic`} />
                      <span className="slash-cmd-main">
                        <span className="slash-cmd-label">/{s.cmd}</span>
                        <span className="slash-cmd-desc">{s.desc}</span>
                      </span>
                    </div>
                  ) : s && typeof s === 'object' && s.clause ? (
                    <div className={`suggestion-item clause-sugg ${activeSugg === i ? 'is-active' : ''}`} key={i} onMouseEnter={() => setActiveSugg(i)} onMouseDown={(e) => { e.preventDefault(); selectSuggestion(s); }}>
                      <span className="clause-sugg-main"><span className="clause-id">{s.id}</span> <span className="clause-snip">{s.snippet}</span></span>
                      <span className="badge badge-navy">{s.source}</span>
                    </div>
                  ) : (() => {
                    const hits = (tagClauses[String(s).toLowerCase()] || []).filter((c) => selected.has(c.source));
                    return (
                      <div className={`suggestion-item ${activeSugg === i ? 'is-active' : ''}`} key={i} onMouseEnter={() => setActiveSugg(i)} onMouseDown={(e) => { e.preventDefault(); selectSuggestion(s); }}>
                        <span>{s}</span>
                        {hits.length === 1
                          ? <span className="badge badge-concept">{hits[0].id}</span>
                          : hits.length > 1
                            ? <span className="badge badge-concept">{hits.length} clauses</span>
                            : null}
                      </div>
                    );
                  })()
                ))}
              </div>
            )}
            {cmd && (
              <span className={`cmd-pill cmd-${cmd}`}>
                /{cmd}
                <button type="button" className="cmd-pill-x" onMouseDown={(e) => { e.preventDefault(); setCmd(null); setSuggestions([]); }} aria-label="Remove command">&times;</button>
              </span>
            )}
            <textarea className="search-input" value={query}
              placeholder={cmd === 'deep' ? 'Enter terms to deep-scan…' : cmd === 'clause' ? 'Enter a clause ID…' : 'Ask IRIS…'}
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
