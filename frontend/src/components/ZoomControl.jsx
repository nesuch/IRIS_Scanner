import { useState, useEffect } from 'react';

// Text-size control for clause viewing. Scales ONLY the clause text (via the
// global --clause-zoom CSS variable applied to .clause-html / .clause-body),
// not the surrounding UI. One setting, persisted in localStorage, shared by the
// Search results and the Reader.
const KEY = 'iris_clause_zoom';
const MIN = 0.85;
const MAX = 1.6;
const STEP = 0.1;

const clamp = (v) => Math.min(MAX, Math.max(MIN, Math.round(v * 100) / 100));

export function getClauseZoom() {
  const r = parseFloat(localStorage.getItem(KEY));
  return Number.isFinite(r) ? clamp(r) : 1;
}

export function applyClauseZoom(v) {
  if (typeof document !== 'undefined') document.documentElement.style.setProperty('--clause-zoom', String(v));
}

// Apply the saved zoom as early as possible so clauses render at the right size
// even before a control mounts.
applyClauseZoom(getClauseZoom());

export default function ZoomControl({ className = '' }) {
  const [z, setZ] = useState(getClauseZoom);
  useEffect(() => { applyClauseZoom(z); try { localStorage.setItem(KEY, String(z)); } catch { /* ignore */ } }, [z]);

  return (
    <div className={`zoom-ctl ${className}`} role="group" aria-label="Clause text size">
      <button type="button" className="zoom-btn" title="Smaller text" aria-label="Smaller text"
        onClick={() => setZ((v) => clamp(v - STEP))} disabled={z <= MIN}>
        <i className="fas fa-magnifying-glass-minus" />
      </button>
      <button type="button" className="zoom-pct" title="Reset text size" aria-label="Reset text size"
        onClick={() => setZ(1)}>{Math.round(z * 100)}%</button>
      <button type="button" className="zoom-btn" title="Larger text" aria-label="Larger text"
        onClick={() => setZ((v) => clamp(v + STEP))} disabled={z >= MAX}>
        <i className="fas fa-magnifying-glass-plus" />
      </button>
    </div>
  );
}
