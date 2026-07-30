import { Fragment, memo, useRef } from 'react';
import { useToast } from '../../components/Toast.jsx';
import { copyRich } from '../../copy.js';
import { inlineTableStyles } from '../../clauseHtml.js';

// A markdown-rendered table plus a "Copy table" button (matches the HTML clause
// path). Copies the rendered table as Word-ready HTML so it pastes with gridlines.
function CopyableTable({ children }) {
  const ref = useRef(null);
  const toast = useToast();
  const copy = async () => {
    const table = ref.current && ref.current.querySelector('table');
    if (!table) return;
    const ok = await copyRich(inlineTableStyles(table.outerHTML), table.innerText);
    toast[ok ? 'success' : 'error'](ok ? 'Table copied!' : 'Could not copy table');
  };
  return (
    <div className="clause-table-wrap" ref={ref}>
      <div className="clause-table-bar">
        <button type="button" className="clause-table-copy" title="Copy just this table" onClick={copy}>
          <i className="fas fa-copy" /> Copy table
        </button>
      </div>
      <div className="clause-table-scroll">{children}</div>
    </div>
  );
}

// Doc-type styles — ports app.py TYPE_STYLES, remapped onto the blue brand
// palette (distinct by hue/lightness, all on-brand). ACT keeps a warm-gold law
// accent for hierarchy clarity.
// Distinct hue per doc type so they don't blur together (Regulation/Circular were
// both blue, hard to tell apart and too close to the blue system UI).
export const TYPE_STYLES = {
  ACT:        { label: 'ACT (The Law)',   color: '#92400e', bg: 'rgba(245,158,11,0.14)', bar: '#d97706' }, // amber
  REGULATION: { label: 'REGULATION',      color: '#3730a3', bg: 'rgba(79,70,229,0.12)',  bar: '#4f46e5' }, // indigo
  MASTER:     { label: 'MASTER CIRCULAR', color: '#0f766e', bg: 'rgba(20,184,166,0.14)', bar: '#14b8a6' }, // teal
  CIRCULAR:   { label: 'CIRCULAR',        color: '#1d4ed8', bg: 'rgba(37,99,235,0.10)',  bar: '#2563eb' }, // blue
  GUIDELINE:  { label: 'GUIDELINE',       color: '#9f1239', bg: 'rgba(244,63,94,0.10)',  bar: '#e11d48' }, // rose
  UNKNOWN:    { label: 'DOCUMENT',        color: '#475569', bg: 'rgba(148,163,184,0.14)', bar: '#94a3b8' }, // slate
};

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// One word's pattern. Pure word stems prefix-match the whole family — Porter
// turns a trailing 'y' into 'i' (policy->polici), so drop it to recover the
// common prefix ("polic" covers policy/policies). Digit/symbol tokens ("10%")
// match literally.
function wordPattern(w) {
  const t = String(w).toLowerCase().trim();
  if (t.length < 1) return null;
  if (/^[a-z]+$/.test(t)) {
    const root = (t.length >= 4 && t.endsWith('i')) ? t.slice(0, -1) : t;
    return `${escapeRe(root)}[a-z]*`;
  }
  // Literal — trailing boundary only when it ends in a word char (so "10" ≠ "100").
  return `${escapeRe(t)}${/\w$/.test(t) ? '\\b' : ''}`;
}

// Build a case-insensitive matcher. The backend sends phrases (each a list of
// word stems); a multi-word phrase like ["free","look","period"] is matched as a
// consecutive unit (so standalone "period" is NOT highlighted), words separated
// by whitespace/hyphen. Single string terms are treated as one-word phrases.
function buildHighlightRegex(terms) {
  if (!terms || !terms.length) return null;
  const pats = [];
  const seen = new Set();
  for (const phrase of terms) {
    const words = Array.isArray(phrase) ? phrase : [phrase];
    const wps = words.map(wordPattern).filter(Boolean);
    if (!wps.length) continue;
    const pat = `\\b${wps.join('[\\s\\-]+')}`;
    if (seen.has(pat)) continue;
    seen.add(pat);
    pats.push(pat);
  }
  if (!pats.length) return null;
  pats.sort((a, b) => b.length - a.length);
  return new RegExp(`(${pats.join('|')})`, 'gi');
}

// One regex PER term, rather than the single combined matcher above. Used only to
// answer "does this line contain every query word?" — the tier-2 signal. A combined
// regex can't answer that: it tells you something matched, not that everything did.
// Returns null below two terms, where "all words present" is the same statement as
// "a word is present" and the wash would just be a second, redundant mark.
function buildTermRegexes(terms) {
  if (!terms || terms.length < 2) return null;
  const out = [];
  const seen = new Set();
  for (const phrase of terms) {
    const words = Array.isArray(phrase) ? phrase : [phrase];
    const wps = words.map(wordPattern).filter(Boolean);
    if (!wps.length) continue;
    const pat = `\\b${wps.join('[\\s\\-]+')}`;
    if (seen.has(pat)) continue;
    seen.add(pat);
    out.push(new RegExp(pat, 'i'));   // no /g: .test() only, so lastIndex can't drift
  }
  return out.length >= 2 ? out : null;
}

// True when every query term appears in this stretch of text.
function coversAll(termRes, text) {
  if (!termRes) return false;
  return termRes.every((re) => re.test(text));
}

// Highlight keyword matches inside pre-rendered clause HTML (edited / imported
// clauses that carry clause_html and render via dangerouslySetInnerHTML, so they
// otherwise miss the keyword highlighting that ClauseBody applies). Walks text
// nodes only — tags, attributes and existing <mark>s are left untouched.
function _highlightNodes(doc, regex, cls) {
  if (!regex) return;
  const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
  const targets = [];
  let n;
  // eslint-disable-next-line no-cond-assign
  while ((n = walker.nextNode())) {
    const tag = n.parentNode?.nodeName;
    if (tag === 'MARK' || tag === 'SCRIPT' || tag === 'STYLE') continue;  // don't re-mark
    regex.lastIndex = 0;
    if (regex.test(n.nodeValue)) targets.push(n);
  }
  targets.forEach((node) => {
    const text = node.nodeValue;
    const frag = doc.createDocumentFragment();
    let last = 0; let m;
    regex.lastIndex = 0;
    while ((m = regex.exec(text)) !== null) {
      if (m.index > last) frag.appendChild(doc.createTextNode(text.slice(last, m.index)));
      const mark = doc.createElement('mark');
      mark.className = cls;
      mark.textContent = m[0];
      frag.appendChild(mark);
      last = m.index + m[0].length;
      if (m.index === regex.lastIndex) regex.lastIndex++;
    }
    if (last < text.length) frag.appendChild(doc.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  });
}

// Two-pass highlight: the verbatim PHRASE first (its own colour, `hl-phrase`), then
// the individual words (`hl`). Doing the phrase first means the words inside it are
// already wrapped in a <mark>, so the word pass skips them — the phrase stays one
// contiguous highlight instead of being chopped into per-word pieces.
export function highlightHtml(html, keywords, phraseKeywords) {
  const wordRe = buildHighlightRegex(keywords);
  const phraseRe = buildHighlightRegex(phraseKeywords);
  if ((!wordRe && !phraseRe) || !html) return html;
  const doc = new DOMParser().parseFromString(`<body>${html}</body>`, 'text/html');
  _highlightNodes(doc, phraseRe, 'hl hl-phrase');
  _highlightNodes(doc, wordRe, 'hl');
  _markCoverBlocks(doc, buildTermRegexes(keywords));
  return doc.body.innerHTML;
}

// Tier 2 on the pre-rendered-HTML path. ClauseBody can wash a line because it owns
// the line divs; here the markup is the document's own, so the wash goes on whatever
// block element carries every query word.
const _COVER_BLOCKS = 'p,li,td,th,h1,h2,h3,h4,h5,h6,blockquote';

function _markCoverBlocks(doc, termRes) {
  if (!termRes) return;
  const blocks = [...doc.body.querySelectorAll(_COVER_BLOCKS)];
  const hits = blocks.filter((el) => coversAll(termRes, el.textContent || ''));
  // Innermost only. A <td> inside a covering <p> inside a covering <blockquote>
  // would otherwise wash three nested boxes and tint half the clause.
  hits.forEach((el) => {
    if (!hits.some((other) => other !== el && el.contains(other))) el.classList.add('hl-cover');
  });
}

// Highlight one line: the verbatim PHRASE (own colour) takes precedence, then the
// individual words fill the gaps it doesn't already cover.
function markLine(text, phraseRe, wordRe, keyPrefix) {
  const ranges = [];
  const scan = (re, cls, avoid) => {
    if (!re) return;
    re.lastIndex = 0; let m;
    // eslint-disable-next-line no-cond-assign
    while ((m = re.exec(text)) !== null) {
      const s = m.index; const e = s + m[0].length;
      if (!avoid || !avoid.some(([as, ae]) => s < ae && e > as)) ranges.push([s, e, cls]);
      if (m.index === re.lastIndex) re.lastIndex++;
    }
  };
  scan(phraseRe, 'hl hl-phrase', null);
  const phraseRanges = ranges.map(([s, e]) => [s, e]);
  scan(wordRe, 'hl', phraseRanges);
  if (!ranges.length) return text;
  ranges.sort((a, b) => a[0] - b[0]);
  const out = []; let last = 0; let i = 0;
  for (const [s, e, cls] of ranges) {
    if (s < last) continue;                       // drop overlaps
    if (s > last) out.push(text.slice(last, s));
    out.push(<mark className={cls} key={`${keyPrefix}-${i++}`}>{text.slice(s, e)}</mark>);
    last = e;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Render one clause body: markdown tables -> <table>, heading lines (ending
// with ':') bold (no highlight), other lines keyword-highlighted, pre-wrap.
export const ClauseBody = memo(function ClauseBody({ text, keywords, phraseKeywords }) {
  const regex = buildHighlightRegex(keywords);
  const phraseRe = buildHighlightRegex(phraseKeywords);
  const termRes = buildTermRegexes(keywords);
  const lines = String(text || '').split('\n');
  const blocks = [];
  let tableRows = [];

  const isSeparator = (r) => /^\|[\s\-:|]+\|$/.test(r.trim());

  const flushTable = (key) => {
    if (!tableRows.length) return;
    // Derive per-column alignment from the GFM separator row (the one with
    // dashes): `:---` left, `---:` right, `:--:` center. Pure Markdown — applied
    // as a CSS text-align (no HTML), so unaligned tables render exactly as before.
    const sep = tableRows.find((r) => isSeparator(r) && r.includes('-'));
    const aligns = sep
      ? sep.trim().replace(/^\||\|$/g, '').split('|').map((seg) => {
          const s = seg.trim();
          const l = s.startsWith(':'); const rt = s.endsWith(':');
          return l && rt ? 'center' : rt ? 'right' : l ? 'left' : null;
        })
      : [];
    const rows = tableRows.filter((r) => !isSeparator(r));
    blocks.push(
      <CopyableTable key={`t-${key}`}>
        <table className="clause-md-table">
          <tbody>
            {rows.map((row, ri) => {
              const cells = row.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
              const Tag = ri === 0 ? 'th' : 'td';
              return (
                <tr key={ri}>
                  {cells.map((c, ci) => (
                    <Tag key={ci} style={aligns[ci] ? { textAlign: aligns[ci] } : undefined}>{c}</Tag>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </CopyableTable>
    );
    tableRows = [];
  };

  lines.forEach((line, idx) => {
    const stripped = line.trim();
    if (stripped.startsWith('|') && stripped.endsWith('|')) {
      tableRows.push(stripped);
      return;
    }
    flushTable(idx);
    // Tier 2: wash the whole line when it carries every query word. Skipped on
    // heading lines — a bold heading is already emphasised, and tinting it too
    // stacks two signals on the least informative line.
    const cover = coversAll(termRes, line) ? ' hl-cover' : '';
    if (stripped.endsWith(':') && stripped.length) {
      blocks.push(<div className="clause-line clause-head" key={idx}><strong>{markLine(line, phraseRe, regex, idx)}</strong></div>);
    } else {
      blocks.push(<div className={`clause-line${cover}`} key={idx}>{markLine(line, phraseRe, regex, idx)}</div>);
    }
  });
  flushTable('end');

  return <div className="clause-body">{blocks}</div>;
});

// A short preview of a long body, centred on the first keyword match so the
// relevant line is visible without scrolling/expanding. Built from plain text
// (works uniformly for both render paths — HTML and markdown — and never slices a
// <table> mid-tag). Falls back to the start of the text when no keyword is present
// (e.g. a tag-only match). Used for PQ result cards, whose replies run to tens of
// thousands of characters; deliberately NOT ClauseBody, which would read a windowed
// row of a PQ data table as markdown and render a stray <table>.
export function ClauseSnippet({ text, keywords, phraseKeywords, radius = 170 }) {
  const raw = String(text || '').replace(/\s+/g, ' ').trim();
  const regex = buildHighlightRegex(keywords);
  const phraseRe = buildHighlightRegex(phraseKeywords);
  const termRes = buildTermRegexes(keywords);
  // Centre on the VERBATIM match when there is one, falling back to the first
  // single-word hit. Previously only the word regex steered the window, so a
  // snippet could open on an incidental cousin of one word while the exact phrase
  // sat outside the window entirely — showing the weakest evidence for the match.
  let start = 0;
  let anchor = -1;
  if (phraseRe) {
    phraseRe.lastIndex = 0;
    const pm = phraseRe.exec(raw);
    if (pm) anchor = pm.index;
  }
  if (anchor < 0 && regex) {
    regex.lastIndex = 0;
    const m = regex.exec(raw);
    if (m) anchor = m.index;
  }
  if (anchor > radius) start = anchor - radius;
  let end = Math.min(raw.length, start + radius * 2);
  // Snap the window edges to word boundaries so we never cut a word in half.
  if (start > 0) {
    const sp = raw.indexOf(' ', start);
    if (sp !== -1 && sp < start + 40) start = sp + 1;
  }
  if (end < raw.length) {
    const sp = raw.lastIndexOf(' ', end);
    if (sp > start + 40) end = sp;
  }
  const slice = raw.slice(start, end);
  return (
    <div className={`clause-snippet${coversAll(termRes, slice) ? ' hl-cover' : ''}`}>
      {start > 0 && '… '}
      {markLine(slice, phraseRe, regex, 'snip')}
      {end < raw.length && ' …'}
    </div>
  );
}

// Group consecutive matches by doc type (mirrors build_results_html ordering),
// inserting a type header when the type changes, a divider otherwise.
export function groupByType(matches) {
  const groups = [];
  let current = null;
  matches.forEach((m) => {
    const type = m.type || 'UNKNOWN';
    if (!current || current.type !== type) {
      current = { type, items: [m] };
      groups.push(current);
    } else {
      current.items.push(m);
    }
  });
  return groups;
}

export { Fragment };
