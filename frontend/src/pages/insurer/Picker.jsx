// Searchable insurer picker.
//
// A native <select> with 87 options grouped into five classes is unusable — you
// scroll a wall of near-identical company names. This is a combobox: type to
// filter, arrow keys to move, Enter to choose, and the list stays short and
// scrollable.
import { useEffect, useMemo, useRef, useState } from 'react';

export default function Picker({ classes, order, labels, value, onChange, placeholder }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const [active, setActive] = useState(0);
  const boxRef = useRef(null);
  const listRef = useRef(null);

  const flat = useMemo(() => {
    const out = [];
    (order || []).forEach((c) => (classes[c] || []).forEach((i) => out.push({ ...i, cls: c })));
    return out;
  }, [classes, order]);

  const current = flat.find((x) => x.id === value);

  const hits = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return flat;
    // Match on name OR class, so "sahi" or "health" narrows to that segment.
    return flat.filter((x) => x.name.toLowerCase().includes(s)
      || (labels[x.cls] || x.cls).toLowerCase().includes(s));
  }, [flat, q, labels]);

  useEffect(() => { setActive(0); }, [q]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  // Keep the highlighted row in view when arrowing through a long list.
  useEffect(() => {
    const el = listRef.current?.querySelector('.pk-opt.on');
    if (el) el.scrollIntoView({ block: 'nearest' });
  }, [active, open]);

  const choose = (x) => { onChange(x.id); setOpen(false); setQ(''); };

  const onKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, hits.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (e.key === 'Enter') { e.preventDefault(); if (hits[active]) choose(hits[active]); }
    else if (e.key === 'Escape') { setOpen(false); }
  };

  return (
    <div className="pk" ref={boxRef}>
      <button type="button" className="pk-btn" onClick={() => { setOpen((o) => !o); setQ(''); }}>
        <span className={current ? '' : 'pk-ph'}>{current ? current.name : (placeholder || 'Select…')}</span>
        {current && <em className="pk-cls">{labels[current.cls] || current.cls}</em>}
        <i className="fas fa-chevron-down" />
      </button>

      {open && (
        <div className="pk-pop">
          <div className="pk-search">
            <i className="fas fa-magnifying-glass" />
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={onKey}
              placeholder="Search insurer or class…" aria-label="Search insurers" />
            {q && <button type="button" onClick={() => setQ('')} aria-label="Clear">&times;</button>}
          </div>
          <div className="pk-list" ref={listRef} role="listbox">
            {hits.length === 0 && <div className="pk-none">No match for “{q}”</div>}
            {hits.map((x, i) => (
              <button type="button" key={x.id} role="option" aria-selected={x.id === value}
                className={`pk-opt ${i === active ? 'on' : ''} ${x.id === value ? 'sel' : ''}`}
                onMouseEnter={() => setActive(i)} onClick={() => choose(x)}>
                <span className="pk-name">{x.name}</span>
                <span className="pk-tag">{labels[x.cls] || x.cls}</span>
              </button>
            ))}
          </div>
          <div className="pk-foot">{hits.length} of {flat.length}</div>
        </div>
      )}
    </div>
  );
}
