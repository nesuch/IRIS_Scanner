import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import FlagModal from '../components/FlagModal.jsx';
import { api } from '../api.js';
import { ClauseSnippet, highlightHtml } from './search/clauseRender.jsx';
import { PQ_SLASH_COMMANDS, parseSlash } from './search/slash.js';
import './search/search.css';   // reuse the universal-search shell (bottom bar, suggestions)
import './pqs/pqs.css';

// Result groups, in order. A reply tagged for the query is a stronger answer than
// one that merely mentions it, so tags lead and bodies follow — never interleaved.
const TIERS = [
  ['tag', 'Tagged for this', 'fa-tag'],
  ['body', 'Found in reply text', 'fa-align-left'],
];

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
  const [filter, setFilter] = useState('');
  const [sort, setSort] = useState('relevance');
  const results = resp.items || [];
  const toggleDept = (code) => setDeptFilter((cur) => cur.includes(code) ? cur.filter((c) => c !== code) : [...cur, code]);
  // Highlight terms travel with the reply when it is opened, so the words lit up on
  // the card are still lit up in the full text.
  const hl = { words: resp.highlight || [], phrases: resp.highlight_phrase || [] };

  // Narrow-down box. Deliberately a literal substring match over the fields shown on
  // the card, NOT another relevance search: this is for "the one about pandemic
  // cover, I'll know it when I see it" over a long list, where a ranked re-query
  // would reshuffle the very thing the reader is scanning. Reply bodies are excluded
  // for the same reason — a filter that matches text you cannot see looks broken.
  const needle = filter.trim().toLowerCase();
  const matchesFilter = (p) => !needle || [
    p.subject, p.title, p.pq_no, p.house, ...(p.tags || []),
    ...(p.departments || []).map((c) => DEPT_LABEL[c]),
  ].some((f) => String(f || '').toLowerCase().includes(needle));

  let visible = results.filter((p) => !dismissed.has(p.id)
    && (deptFilter.length === 0 || (p.departments || []).some((c) => deptFilter.includes(c)))
    && matchesFilter(p));

  // Date sorts reorder the answer set the search already chose — they do not re-run
  // the search. An undated reply sorts last either way: an unparsed date is neither
  // the newest nor the oldest thing here, and guessing would put it at one extreme.
  if (sort !== 'relevance') {
    const dir = sort === 'oldest' ? 1 : -1;
    visible = visible.slice().sort((a, b) => {
      const x = a.date_iso || ''; const y = b.date_iso || '';
      if (!x && !y) return 0;
      if (!x) return 1;
      if (!y) return -1;
      return x < y ? dir : x > y ? -dir : 0;
    });
  }

  // A tag hit is an editor's judgement that the reply IS about this; a body hit is
  // a mention somewhere in the text. Worth telling apart, so the groups are labelled
  // — but only under relevance order. Grouping by tier while sorting by date would
  // silently break the date order the reader just asked for, into two runs.
  const tiered = results.some((p) => p.tier) && sort === 'relevance';

  // No matches at all — explain and offer Deep Scan (full reply bodies).
  if (results.length === 0) {
    return (
      <div className="pq-noresult">
        <p className="iris-msg">{resp.deep
          ? `I read every reply in full — none mention ${resp.label}.`
          : resp.browse ? 'There are no Parliamentary Questions in the database yet.'
          : `No reply is tagged for “${resp.label}” or mentions it.`}
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
            : <><i className="fas fa-magnifying-glass" /> Search · <strong>{resp.label}</strong></>}
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
      {results.length > 1 && (
        <div className="pq-toolbar">
          <div className="pq-filter">
            <i className="fas fa-filter" />
            <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)}
              placeholder={`Filter these ${results.length} replies — subject, Q number, tag…`}
              aria-label="Filter these results" />
            {filter && <button className="pq-filter-x" onClick={() => setFilter('')} aria-label="Clear filter">&times;</button>}
          </div>
          <div className="pq-sort" role="group" aria-label="Sort results">
            {[['relevance', 'Relevance'], ['newest', 'Newest'], ['oldest', 'Oldest']].map(([k, lbl]) => (
              <button key={k} type="button" className={sort === k ? 'on' : ''}
                onClick={() => setSort(k)}>{lbl}</button>
            ))}
          </div>
        </div>
      )}
      <div className="pq-result-count">{visible.length} {visible.length === 1 ? 'reply' : 'replies'}{resp.deep ? ` mentioning ${resp.label} · deep scan` : resp.browse ? ' in the database' : ` for “${resp.label}”`}{deptFilter.length > 0 ? ` · ${deptFilter.map((c) => DEPT_LABEL[c]).join(' / ')}` : ''}{needle ? ` · filtered by “${filter.trim()}”` : ''}{sort !== 'relevance' ? ` · ${sort} first` : ''}</div>
      {visible.length === 0 ? (
        <p className="iris-msg">
          {needle
            ? <>No reply here matches “{filter.trim()}”. <button className="flag-link" onClick={() => setFilter('')}>Clear the filter</button></>
            : <>No {deptFilter.map((c) => DEPT_LABEL[c]).join(' / ')} PQs in this set.</>}
        </p>
      ) : (
        <div className="pq-list">
          {/* Split into tiers only under relevance order. Under a date sort the list
              is one run, or the tier split would quietly re-break the ordering. */}
          {(tiered ? TIERS : [['all', null, null]]).map(([tier, head, icon]) => {
            const group = tiered ? visible.filter((p) => (p.tier || 'tag') === tier) : visible;
            if (group.length === 0) return null;
            return (
              <Fragment key={tier}>
                {tiered && <div className="pq-tier-head"><i className={`fas ${icon}`} /> {head}</div>}
                {group.map((p) => (
                  <div key={p.id} className="pq-card" onClick={() => onOpen(p.id, hl)} role="button" tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') onOpen(p.id, hl); }}>
                    <button className="pq-card-hide" title="Hide from results" onClick={(e) => { e.stopPropagation(); onHide(p.id); }}>&times;</button>
                    {onFlag && <button className="pq-card-flag" title="Flag this reply" onClick={(e) => { e.stopPropagation(); onFlag(p); }}><i className="fas fa-flag" /></button>}
                    {isAdmin && <button className="pq-card-del" title="Delete" onClick={(e) => { e.stopPropagation(); onDelete(p.id); }}><i className="fas fa-trash" /></button>}
                    <div className="pq-card-top">
                      {p.house && <span className="pq-house">{p.house}</span>}
                      {p.pq_no && <span className="pq-no">Q No. {p.pq_no}</span>}
                      {p.date && <span className="pq-date"><i className="far fa-calendar" /> {p.date}</span>}
                    </div>
                    <div className="pq-card-title">{p.subject || p.title}</div>
                    {p.snippet && (
                      <ClauseSnippet text={p.snippet} keywords={resp.highlight || []}
                        phraseKeywords={resp.highlight_phrase || []} radius={400} />
                    )}
                    {((p.departments || []).length > 0 || p.tags?.length > 0) && (
                      <div className="pq-card-meta-row">
                        {(p.departments || []).map((c) => <span key={c} className="pq-dept-badge">{DEPT_LABEL[c]}</span>)}
                        {(p.tags || []).slice(0, 5).map((t) => <span key={t} className="pq-tag">{t}</span>)}
                      </div>
                    )}
                    <span className="pq-card-open">Read full reply <i className="fas fa-arrow-right" /></span>
                  </div>
                ))}
              </Fragment>
            );
          })}
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
  // Highlight terms carried IN from the search that found this reply, so the words
  // that were lit up on the card stay lit up in the full text. Without this, opening
  // a hit dropped every mark and left the reader hunting through 25k characters for
  // the word they had just searched for.
  const [activeHl, setActiveHl] = useState(null);
  const [hitIdx, setHitIdx] = useState(0);
  const [opening, setOpening] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [bulkOpen, setBulkOpen] = useState(false);
  const [allTags, setAllTags] = useState([]);
  const [flagTarget, setFlagTarget] = useState(null);   // PQ (or no-result) being flagged
  const chatRef = useRef(null);
  const docRef = useRef(null);        // the rendered reply, for match navigation
  const lastUserRef = useRef(null);
  const inputAreaRef = useRef(null);
  const inputRef = useRef(null);      // picking a slash command hands focus back
  const liveIdRef = useRef(null);     // id of the current as-you-type block, or null
  const liveSeqRef = useRef(0);       // guards against out-of-order live responses

  // Mark the search terms inside the opened reply. highlightHtml parses and walks
  // text nodes only, so the stored markup is untouched — and it is memoised because
  // a PQ body is tens of thousands of characters and this would otherwise re-parse
  // on every keystroke elsewhere on the page.
  const activeHtml = useMemo(() => {
    if (!active?.html) return '';
    if (!activeHl?.words?.length && !activeHl?.phrases?.length) return active.html;
    return highlightHtml(active.html, activeHl.words, activeHl.phrases);
  }, [active, activeHl]);

  // Counted off the markup rather than the DOM so it is ready on first paint —
  // highlightHtml emits class="hl" and class="hl hl-phrase", both matched here.
  const hitCount = useMemo(
    () => (activeHtml.match(/<mark class="hl/g) || []).length, [activeHtml]);

  // Jump straight to the first match. The verbatim hit wins over a single-word one:
  // landing on an incidental cousin of one word, when the exact phrase sits further
  // down, is worse than not scrolling at all.
  useEffect(() => {
    if (!active || !docRef.current) return;
    const marks = docRef.current.querySelectorAll('mark.hl');
    if (!marks.length) { setHitIdx(0); return; }
    const first = docRef.current.querySelector('mark.hl-phrase') || marks[0];
    const i = [...marks].indexOf(first);
    setHitIdx(i);
    marks.forEach((el, n) => el.classList.toggle('is-cur', n === i));
    first.scrollIntoView({ block: 'center' });
  }, [active, activeHtml]);

  // Step through matches. Wraps, so it never dead-ends at the last hit.
  function gotoHit(delta) {
    const marks = docRef.current?.querySelectorAll('mark.hl');
    if (!marks?.length) return;
    const next = (hitIdx + delta + marks.length) % marks.length;
    setHitIdx(next);
    marks.forEach((el, n) => el.classList.toggle('is-cur', n === next));
    marks[next].scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  const loadTags = () => api.get('/pq/tags').then((d) => setAllTags(d.tags || [])).catch(() => setAllTags([]));
  useEffect(() => { loadTags(); }, []);
  useEffect(() => {
    document.documentElement.classList.add('app-fixed');
    return () => document.documentElement.classList.remove('app-fixed');
  }, []);

  function clearChat() { liveIdRef.current = null; setHistory([]); setQuery(''); setSuggestions([]); setActive(null); setDismissed(new Set()); }

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

  // Append a chat turn (You → IRIS) and resolve its response.
  // live=true: as-you-type — reuse one block in place (no chat spam) and apply
  // only the newest response. live=false: a committed search, appended as usual.
  async function ask(url, label, meta = {}, { live = false } = {}) {
    if (!live && busy) return;
    let id;
    if (live && liveIdRef.current) {
      id = liveIdRef.current;
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, query: label, response: null, error: null } : x)));
    } else {
      id = Math.random().toString(36).slice(2);
      setHistory((h) => [...h, { id, query: label, response: null }]);
      liveIdRef.current = live ? id : null;   // a committed search ends the live session
    }
    const seq = live ? (liveSeqRef.current += 1) : null;
    if (!live) setBusy(true);
    try {
      const d = await api.get(url);
      if (live && seq !== liveSeqRef.current) return;   // a newer keystroke superseded this
      const response = {
        items: d.items || [], chips: d.chips || [], deep: !!d.deep,
        highlight: d.highlight || [], highlight_phrase: d.highlight_phrase || [],
        label, ...meta,
      };
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (e) {
      if (live && seq !== liveSeqRef.current) return;
      setHistory((h) => h.map((x) => (x.id === id ? { ...x, error: e.message || 'Search failed' } : x)));
      if (!live) toast.error(e.message || 'Search failed');
    } finally { if (!live) setBusy(false); }
  }

  // mode: 'q' free text | 'tag' exact tag | 'num' PQ number
  function runSearch(value, mode = 'q', { live = false } = {}) {
    const v = value.trim(); if (!v) return;
    const key = mode === 'tag' ? 'tag' : mode === 'num' ? 'num' : 'q';
    const label = mode === 'num' ? `/${v}` : v;
    ask(`/pq?${key}=${encodeURIComponent(v)}`, label, { qText: mode === 'q' ? v : '' }, { live });
  }

  // As-you-type live search, mirroring the universal search: debounced, replacing
  // one block in place. Skips slash commands (those are committed on Enter) and
  // very short queries. This is what makes PQ search feel like the rest of IRIS —
  // without it a typed word does nothing until you submit, which reads as "there
  // is no full-text search" even when there is.
  useEffect(() => {
    const q = query.trim();
    if (q.length < 3 || q.startsWith('/')) {
      if (liveIdRef.current) {   // cleared or too short — drop the pending live block
        const id = liveIdRef.current; liveIdRef.current = null;
        setHistory((h) => h.filter((x) => x.id !== id));
      }
      return undefined;
    }
    const t = setTimeout(() => { runSearch(q, 'q', { live: true }); }, 400);
    return () => clearTimeout(t);
  }, [query]);   // eslint-disable-line react-hooks/exhaustive-deps

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
    const p = parseSlash(q, PQ_SLASH_COMMANDS);
    if (p && p.kind === 'menu') return;          // still choosing a command
    setQuery(''); setSuggestions([]);
    if (p && p.kind === 'cmd') {
      const arg = (p.arg || '').trim();
      if (!arg) return;
      if (p.cmd === 'deep') runDeepScan(arg, 'all', `“${arg}” (all words)`);
      else if (p.cmd === 'num') runSearch(arg.replace(/^\/+/, ''), 'num');
      return;
    }
    // A bare "/9000" isn't a command (no command starts with a digit) — keep the
    // long-standing shortcut working.
    if (q.startsWith('/')) { const d = q.slice(1).trim(); if (d) runSearch(d, 'num'); return; }
    // Live results for this query are already on screen — Enter just keeps them
    // and clears the box, rather than firing the identical search again.
    if (liveIdRef.current) { liveIdRef.current = null; return; }
    runSearch(q, 'q');
  }
  function onKeyDown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(e); } }

  function onInput(e) {
    const val = e.target.value;
    setQuery(val);
    const t = val.trim();
    const p = parseSlash(t, PQ_SLASH_COMMANDS);
    if (p && p.kind === 'menu') {             // "/" or "/de" -> offer the commands
      setSuggestions(p.list.map((c) => ({ ...c, _cmd: true })));
      return;
    }
    if (p && p.kind === 'cmd' && p.cmd !== 'num') { setSuggestions([]); return; }
    // "/num 9000" or a bare "/9000" -> suggest PQs by number
    if (t.startsWith('/')) {
      const arg = (p && p.kind === 'cmd' ? p.arg : t.slice(1)).trim();
      if (!arg) { setSuggestions([]); return; }
      api.get(`/pq?num=${encodeURIComponent(arg)}`)
        .then((d) => setSuggestions((d.items || []).slice(0, 30).map((x) => ({ ...x, _pq: true }))))
        .catch(() => setSuggestions([]));
      return;
    }
    const low = t.toLowerCase();
    if (low.length < 2) { setSuggestions([]); return; }
    // Tags only. The full-text results for what's typed are already updating live
    // behind this dropdown, so a "search for X" row here would offer what is
    // already on screen; a tag is the narrower, different thing you can pick.
    setSuggestions(allTags.filter((tag) => tag.toLowerCase().includes(low)).slice(0, 30));
  }

  function pickTag(tag) { setQuery(''); setSuggestions([]); runSearch(tag, 'tag'); }

  // `hl` is the highlight payload of the search that surfaced this reply; null when
  // opened from browse or a suggestion, where nothing was searched for.
  function open(id, hl = null) {
    setOpening(true);
    setActiveHl(hl);
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
        <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions" scopeDot={false} />
        <div className="pq-read-bar">
          <button className="btn btn-ghost btn-sm" onClick={() => setActive(null)}><i className="fas fa-arrow-left" /> Back to results</button>
          {hitCount > 0 && (
            <div className="pq-hitnav" title="Matches for your search, in this reply">
              <i className="fas fa-highlighter" />
              <span className="pq-hitnav-n"><strong>{hitIdx + 1}</strong> of {hitCount}</span>
              <button type="button" onClick={() => gotoHit(-1)} aria-label="Previous match"><i className="fas fa-chevron-up" /></button>
              <button type="button" onClick={() => gotoHit(1)} aria-label="Next match"><i className="fas fa-chevron-down" /></button>
            </div>
          )}
          <div className="pq-read-actions">
            {isEditor && <EditMeta pq={active}
              onSaved={(title, tags, departments, date) => { setActive({ ...active, title, tags, departments, date }); loadTags(); }} />}
            {isAdmin && <button className="btn btn-ghost btn-sm danger" onClick={() => del(active.id)}><i className="fas fa-trash" /> Delete</button>}
            {active.download_url && (() => {
              const isPdf = /\.pdf$/i.test(active.filename || '');
              return (
                <a className="btn btn-primary btn-sm" href={active.download_url} target="_blank" rel="noreferrer">
                  <i className={`fas ${isPdf ? 'fa-file-pdf' : 'fa-file-word'}`} /> Download original
                </a>
              );
            })()}
          </div>
        </div>
        <div className="pq-read-scroll">
          <div className="pq-doc card">
            <div className="pq-doc-head">
              {active.house && <span className="pq-house">{active.house}</span>}
              {active.pq_no && <span className="pq-no">Q No. {active.pq_no}</span>}
              {active.date && <span className="pq-date"><i className="far fa-calendar" /> {active.date}</span>}
            </div>
            <h2 className="pq-doc-title">{active.title}</h2>
            {active.departments?.length > 0 && (
              <div className="pq-doc-depts">{active.departments.map((c) => <span key={c} className="pq-dept-badge">{DEPT_LABEL[c]}</span>)}</div>
            )}
            {active.tags?.length > 0 && (
              <div className="pq-tags">{active.tags.map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
            )}
            <div className="pq-html" ref={docRef} dangerouslySetInnerHTML={{ __html: activeHtml }} />
          </div>
        </div>
      </div>
    );
  }

  const empty = history.length === 0;

  // ---- Chat view ----
  return (
    <div className="search-shell">
      <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="Search IRDAI replies to Parliamentary Questions" scopeDot={false}>
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

      <div className="input-area" ref={inputAreaRef}>
        <form className="input-inner" onSubmit={onSubmit} autoComplete="off">
          <div className="search-wrapper">
            {suggestions.length > 0 && (
              <div className="suggestions-box">
                {suggestions.map((s, i) => (s && s._cmd ? (
                  <div className="suggestion-item slash-cmd" key={i}
                    onMouseDown={(e) => { e.preventDefault(); setQuery(`/${s.cmd} `); setSuggestions([]); inputRef.current?.focus(); }}>
                    <span><i className={`fas ${s.icon}`} style={{ fontSize: 11, color: 'var(--muted)', marginRight: 8 }} />
                      <strong>/{s.cmd}</strong> <span className="clause-snip">{s.desc}</span></span>
                    <span className="badge badge-concept">{s.label}</span>
                  </div>
                ) : s && s._pq ? (
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
            <textarea className="search-input" ref={inputRef} placeholder="Search PQs — topic, tag, or / for commands…" value={query}
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
  const [date, setDate] = useState('');
  const [tags, setTags] = useState('');
  const [depts, setDepts] = useState([]);
  const [busy, setBusy] = useState(false);

  async function save() {
    setBusy(true);
    try {
      const r = await api.post(`/pq/${pq.id}/update`, { title, date, tags, departments: depts });
      toast.success('Saved'); setEditing(false); onSaved(r.title, r.tags || [], r.departments || [], r.date || '');
    } catch (e) { toast.error(e.message || 'Could not save'); }
    finally { setBusy(false); }
  }

  return (
    <>
      <button className="btn btn-ghost btn-sm" onClick={() => { setTitle(pq.title || ''); setDate(pq.date || ''); setTags((pq.tags || []).join(', ')); setDepts(pq.departments || []); setEditing(true); }}><i className="fas fa-pen" /> Edit</button>
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
            <label>Date <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(as it should display)</span></label>
            <input className="input" value={date} onChange={(e) => setDate(e.target.value)} placeholder="e.g. 12 March 2024" />
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
  const [date, setDate] = useState('');
  const [tags, setTags] = useState('');
  const [depts, setDepts] = useState([]);
  const [busy, setBusy] = useState(false);
  const [dup, setDup] = useState(null);   // existing PQ flagged as a likely duplicate

  async function submit(force = false) {
    if (!file) { toast.error('Choose a .docx or .pdf file'); return; }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', file); fd.append('tags', tags); fd.append('departments', depts.join(',')); fd.append('date', date);
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
      <p className="guide-intro">Upload the approved reply as a Word (.docx) or PDF file. IRIS renders it on screen, keeps the original for download, and makes it searchable here. Word preserves formatting and tables best; PDF is fully supported but its text-only rendering won’t carry bold/table styling.</p>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Reply document (.docx or .pdf)</label>
        <input className="input" type="file" accept=".docx,.pdf" onChange={(e) => { setFile(e.target.files?.[0] || null); setDup(null); }} />
      </div>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Date <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(optional — auto-detected from the file if left blank)</span></label>
        <input className="input" value={date} onChange={(e) => setDate(e.target.value)} placeholder="e.g. 12 March 2024" />
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
  const [date, setDate] = useState('');
  const [tags, setTags] = useState('');
  const [depts, setDepts] = useState([]);
  const [busy, setBusy] = useState(false);

  async function startUpload() {
    if (!files.length) { toast.error('Choose .docx or .pdf files'); return; }
    setPhase('uploading');
    try {
      const fd = new FormData();
      [...files].forEach((f) => fd.append('files', f));
      const r = await api.post('/pq/bulk-upload', fd);
      const c = r.created || [];
      const dupes = r.duplicates || [];
      if (!c.length) {
        toast.error(dupes.length ? `All ${dupes.length} already exist in IRIS — nothing uploaded.` : 'No valid .docx or .pdf files');
        setPhase('select'); return;
      }
      setCreated(c); setIdx(0); setTitle(c[0].title || ''); setDate(c[0].date || ''); setTags(''); setDepts([]);
      setPhase('review');
      toast.success(`Uploaded ${c.length}${dupes.length ? `, skipped ${dupes.length} duplicate${dupes.length > 1 ? 's' : ''}` : ''} — now add tags`);
    } catch (e) { toast.error(e.message || 'Upload failed'); setPhase('select'); }
  }

  function goNext(n) {
    if (n >= created.length) { toast.success('Done'); onDone(); return; }
    setIdx(n); setTitle(created[n].title || ''); setDate(created[n].date || ''); setTags(''); setDepts([]);
  }
  async function saveCurrent() {
    const pq = created[idx];
    setBusy(true);
    try { await api.post(`/pq/${pq.id}/update`, { title, date, tags, departments: depts }); }
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
          <label>Date</label>
          <input className="input" value={date} onChange={(e) => setDate(e.target.value)} placeholder="e.g. 12 March 2024" />
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
      <p className="guide-intro">Select several approved replies (.docx or .pdf). IRIS uploads them all, then walks you through each to confirm the title and add tags.</p>
      <div className="field">
        <label>Reply documents (.docx or .pdf)</label>
        <input className="input" type="file" accept=".docx,.pdf" multiple onChange={(e) => setFiles(e.target.files)} />
      </div>
      {files.length > 0 && <div className="pq-bulk-list">{[...files].map((f, i) => <div key={i} className="pq-bulk-file"><i className="fas fa-file-word" /> {f.name}</div>)}</div>}
    </Modal>
  );
}
