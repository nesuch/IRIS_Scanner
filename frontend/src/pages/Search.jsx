import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { useToast } from '../components/Toast.jsx';
import { api } from '../api.js';
import { TYPE_STYLES, ClauseBody, groupByType } from './search/clauseRender.jsx';
import './search/search.css';

const MODULE_META = {
  universal: { title: 'Universal Search', icon: 'fa-magnifying-glass', scope: 'Unified Search (All Departments)' },
  health: { title: 'Health Department', icon: 'fa-heart-pulse', scope: 'Acts, Regulations & Master Circulars (Health)' },
  life: { title: 'Life Department', icon: 'fa-umbrella', scope: 'Regulatory Framework (Life Insurance)' },
};

function ResultCards({ resp, onChip }) {
  const toast = useToast();
  const copy = async (text) => {
    try { await navigator.clipboard.writeText(text); toast.success('Clause text copied!'); }
    catch { toast.error('Could not copy'); }
  };

  if (resp.kind === 'greeting') {
    return (
      <p className="iris-msg">
        <strong>Hello!</strong> I am <strong>IRIS</strong>. Ask me anything related to IRDAI Acts,
        Regulations, Circulars, or Guidelines.
      </p>
    );
  }
  if (resp.kind === 'rejected') {
    return <p className="iris-msg">{resp.note}</p>;
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
      {resp.note && !hasMatches && <p className="iris-msg">{resp.note}</p>}

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
                  <span className="clause-actions">
                    {m.pdf_url && (
                      <a className="pdf-btn" href={m.pdf_url} target="_blank" rel="noreferrer"><i className="fas fa-file-pdf" /> PDF</a>
                    )}
                    <button className="copy-btn" title="Copy clause" onClick={() => copy(m.raw_text)}><i className="far fa-copy" /></button>
                  </span>
                </div>
                <ClauseBody text={m.raw_text} keywords={resp.highlight || []} />
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
  const [vocab, setVocab] = useState([]);
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const chatRef = useRef(null);
  const lastUserRef = useRef(null);

  // Reset chat when switching modules (matches per-page server history reset).
  useEffect(() => { setHistory([]); setQuery(''); setSuggestions([]); }, [module]);

  useEffect(() => {
    api.get('/vocab').then((d) => setVocab(d.CONCEPTS || [])).catch(() => {});
  }, []);

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
      const resp = await api.post('/search', { module, query: q });
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
    const val = query;
    const lastSpace = val.lastIndexOf(' ');
    const next = lastSpace === -1 ? value + ' ' : val.substring(0, lastSpace + 1) + value + ' ';
    setQuery(next);
    setSuggestions([]);
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(e); }
  }

  const empty = history.length === 0;

  return (
    <div className="search-shell">
      <PageHeader fullForm="IRDAI's Regulatory Intelligence System" title={meta.title} scope={`Scope: ${meta.scope}`} />

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
                {item.response ? <ResultCards resp={item.response} onChip={onChip} />
                  : item.error ? <p className="iris-msg" style={{ color: 'var(--bad)' }}>{item.error}</p>
                  : <span className="typing"><span /><span /><span /></span>}
              </div>
            </div>
          </div>
        ))}
        <div style={{ height: 10 }} />
      </div>

      <div className="input-area">
        <form className="input-inner" onSubmit={onSubmit} autoComplete="off">
          <div className="search-wrapper">
            {suggestions.length > 0 && (
              <div className="suggestions-box">
                {suggestions.map((s, i) => (
                  <div className="suggestion-item" key={i} onMouseDown={(e) => { e.preventDefault(); selectSuggestion(s); }}>
                    <span>{s}</span><span className="badge badge-concept">Concept</span>
                  </div>
                ))}
              </div>
            )}
            <textarea className="search-input" placeholder="Ask IRIS..." value={query}
              onChange={onInput} onKeyDown={onKeyDown} rows={1} autoFocus />
            <button className="btn btn-primary" type="submit" disabled={busy}>
              <i className="fas fa-magnifying-glass" /> Search
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
