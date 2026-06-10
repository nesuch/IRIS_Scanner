import { Fragment } from 'react';

// Doc-type styles — ports app.py TYPE_STYLES, remapped onto the blue brand
// palette (distinct by hue/lightness, all on-brand). ACT keeps a warm-gold law
// accent for hierarchy clarity.
export const TYPE_STYLES = {
  ACT:        { label: 'ACT (The Law)',   color: '#92400e', bg: 'rgba(245,158,11,0.12)', bar: '#d97706' },
  REGULATION: { label: 'REGULATION',      color: '#1a237e', bg: 'rgba(26,35,126,0.10)',  bar: '#283593' },
  MASTER:     { label: 'MASTER CIRCULAR', color: '#0e7490', bg: 'rgba(6,182,212,0.12)',  bar: '#06b6d4' },
  CIRCULAR:   { label: 'CIRCULAR',        color: '#1d4ed8', bg: 'rgba(37,99,235,0.10)',  bar: '#2563eb' },
  GUIDELINE:  { label: 'GUIDELINE',       color: '#334155', bg: 'rgba(100,116,139,0.12)', bar: '#64748b' },
  UNKNOWN:    { label: 'DOCUMENT',        color: '#475569', bg: 'rgba(148,163,184,0.14)', bar: '#94a3b8' },
};

// Build a case-insensitive matcher for keywords + simple suffix variants,
// mirroring highlight_keywords() in app.py.
function buildHighlightRegex(keywords) {
  if (!keywords || !keywords.length) return null;
  const expanded = new Set();
  for (let k of keywords) {
    k = String(k).toLowerCase();
    if (k.length < 3) continue;
    expanded.add(k); expanded.add(k + 's'); expanded.add(k + 'ed'); expanded.add(k + 'ing');
  }
  const parts = [...expanded].sort((a, b) => b.length - a.length)
    .map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (!parts.length) return null;
  return new RegExp(`\\b(${parts.join('|')})\\b`, 'gi');
}

function highlightInto(text, regex, keyPrefix) {
  if (!regex) return text;
  const out = [];
  let last = 0; let m; let i = 0;
  regex.lastIndex = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(<mark className="hl" key={`${keyPrefix}-${i++}`}>{m[0]}</mark>);
    last = m.index + m[0].length;
    if (m.index === regex.lastIndex) regex.lastIndex++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Render one clause body: markdown tables -> <table>, heading lines (ending
// with ':') bold (no highlight), other lines keyword-highlighted, pre-wrap.
export function ClauseBody({ text, keywords }) {
  const regex = buildHighlightRegex(keywords);
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
      <table className="clause-md-table" key={`t-${key}`}>
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
    if (stripped.endsWith(':') && stripped.length) {
      blocks.push(<div className="clause-line clause-head" key={idx}><strong>{line}</strong></div>);
    } else {
      blocks.push(<div className="clause-line" key={idx}>{highlightInto(line, regex, idx)}</div>);
    }
  });
  flushTable('end');

  return <div className="clause-body">{blocks}</div>;
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
