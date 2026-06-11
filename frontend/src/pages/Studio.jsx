import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, EmptyState, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import StudioEditor from '../components/StudioEditor.jsx';
import PdfViewer from './search/PdfViewer.jsx';
import { api } from '../api.js';
import './studio/studio.css';

let KEYSEQ = 1;
const nk = () => `k${KEYSEQ++}`;
const clone = (cs) => cs.map((c) => ({ ...c, tags: [...c.tags] }));

export default function Studio() {
  const toast = useToast();
  const [docs, setDocs] = useState(null);
  const [doc, setDoc] = useState(null);
  const [clauses, setClauses] = useState(null);       // [{key,id,html,tags}]
  const [activeKey, setActiveKey] = useState(null);
  const [rev, setRev] = useState(0);                   // bumps to remount the editor on structural change
  const [hist, setHist] = useState([]);
  const [future, setFuture] = useState([]);
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showPdf, setShowPdf] = useState(true);
  const [showRail, setShowRail] = useState(true);
  const [paneW, setPaneW] = useState(38);
  const [importOpen, setImportOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [dragKey, setDragKey] = useState(null);
  const editorApi = useRef(null);
  const splitRef = useRef(null);
  // Refs mirror state so the editor's (stale-closure) onChange always sees current values.
  const clausesRef = useRef(clauses); clausesRef.current = clauses;
  const activeKeyRef = useRef(activeKey); activeKeyRef.current = activeKey;
  const burstRef = useRef({ key: null, t: 0 });

  useEffect(() => { reloadDocs(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  function reloadDocs() { return api.get('/clause/docs').then((d) => setDocs(d.docs || [])).catch(() => setDocs([])); }

  function openDoc(d) {
    setDoc(d); setClauses(null); setActiveKey(null); setHist([]); setFuture([]); setDirty(false); setLoading(true);
    api.get(`/clause/doc-full?source=${encodeURIComponent(d.source)}`)
      .then((r) => {
        const cs = (r.clauses || []).map((c) => ({ key: nk(), id: c.id, html: c.html || '<p></p>', tags: c.tags || [], edited: !!c.edited, changed: false }));
        setClauses(cs); setActiveKey(cs[0]?.key || null); setRev((x) => x + 1);
      })
      .catch((e) => { toast.error(e.message || 'Could not open document'); setClauses([]); })
      .finally(() => setLoading(false));
  }
  function closeDoc() {
    if (dirty && !window.confirm('Discard unsaved changes?')) return;
    setDoc(null); setClauses(null); setActiveKey(null);
  }

  const idx = () => (clauses ? clauses.findIndex((c) => c.key === activeKey) : -1);
  const active = clauses ? clauses.find((c) => c.key === activeKey) : null;

  function pushHist() {
    setHist((h) => [...h.slice(-49), JSON.stringify({ c: clausesRef.current, a: activeKeyRef.current })]);
    setFuture([]);
  }
  // Coalesce a burst of edits (same target within 1s) into a single undo step.
  function noteChange(burstKey) {
    const now = Date.now(); const b = burstRef.current;
    if (!(b.key === burstKey && now - b.t < 1000)) pushHist();
    burstRef.current = { key: burstKey, t: now };
  }

  // Push current state to history before a structural change.
  function commit(next, nextActive) {
    pushHist();
    burstRef.current = { key: null, t: 0 };   // structural op ends any edit burst
    setClauses(next);
    if (nextActive !== undefined) setActiveKey(nextActive);
    setRev((x) => x + 1);
    setDirty(true);
  }
  function undo() {
    if (!hist.length) return;
    const prev = JSON.parse(hist[hist.length - 1]);
    setFuture((f) => [JSON.stringify({ c: clauses, a: activeKey }), ...f]);
    setHist((h) => h.slice(0, -1));
    setClauses(prev.c); setActiveKey(prev.a); setRev((x) => x + 1); setDirty(true);
  }
  function redo() {
    if (!future.length) return;
    const nx = JSON.parse(future[0]);
    setHist((h) => [...h, JSON.stringify({ c: clauses, a: activeKey })]);
    setFuture((f) => f.slice(1));
    setClauses(nx.c); setActiveKey(nx.a); setRev((x) => x + 1); setDirty(true);
  }

  // Continuous content edits from the editor — update active clause, mark dirty
  // (no history snapshot per keystroke; structural ops snapshot instead).
  function onEdit(html) {
    const key = activeKeyRef.current;
    noteChange('content:' + key);
    setClauses((cs) => cs.map((c) => (c.key === key ? { ...c, html, edited: true, changed: true } : c)));
    setDirty(true);
  }
  function setField(field, value) {
    const key = activeKeyRef.current;
    noteChange(field + ':' + key);
    setClauses((cs) => cs.map((c) => (c.key === key ? { ...c, [field]: value, edited: true, changed: true } : c)));
    setDirty(true);
  }

  function uniqueId(base, list) {
    let id = base; let n = 1;
    const ids = new Set(list.map((c) => c.id));
    while (ids.has(id)) { n += 1; id = `${base}-${n}`; }
    return id;
  }

  function addClause() {
    const i = idx(); if (i < 0) return;
    const c = [...clone(clauses)];
    const k = nk();
    c.splice(i + 1, 0, { key: k, id: uniqueId('NEW', clauses), html: '<p>New clause:</p>', tags: [], edited: true, changed: true });
    commit(c, k);
  }
  function delClause() {
    const i = idx(); if (i < 0 || clauses.length <= 1) { toast.error('Cannot delete the only clause'); return; }
    const c = clone(clauses).filter((_, j) => j !== i);
    commit(c, c[Math.min(i, c.length - 1)].key);
  }
  function mergeUp() {
    const i = idx(); if (i <= 0) { toast.error('No clause above'); return; }
    const c = clone(clauses);
    c[i - 1] = { ...c[i - 1], html: c[i - 1].html + c[i].html, edited: true, changed: true };
    const keep = c[i - 1].key; c.splice(i, 1); commit(c, keep);
  }
  function mergeDown() {
    const i = idx(); if (i < 0 || i >= clauses.length - 1) { toast.error('No clause below'); return; }
    const c = clone(clauses);
    c[i + 1] = { ...c[i + 1], html: c[i].html + c[i + 1].html, edited: true, changed: true };
    const keep = c[i + 1].key; c.splice(i, 1); commit(c, keep);
  }
  function applyRestore(v) {
    const i = idx(); if (i < 0) return;
    const c = clone(clauses);
    c[i] = { ...c[i], html: v.html, tags: [...(v.tags || [])], edited: true, changed: true };
    commit(c, activeKey);
    setHistoryOpen(false);
    toast.success('Version restored — Save to keep it');
  }

  function splitClause() {
    const i = idx(); if (i < 0 || !editorApi.current) return;
    const { before, after } = editorApi.current.split();
    const c = clone(clauses);
    c[i] = { ...c[i], html: before, edited: true, changed: true };
    const k = nk();
    c.splice(i + 1, 0, { key: k, id: uniqueId(`${c[i].id}-b`, clauses), html: after, tags: [], edited: true, changed: true });
    commit(c, c[i].key);
  }

  // Drag to reorder
  function onDrop(targetKey) {
    if (!dragKey || dragKey === targetKey) { setDragKey(null); return; }
    const c = clone(clauses);
    const from = c.findIndex((x) => x.key === dragKey);
    const to = c.findIndex((x) => x.key === targetKey);
    const [moved] = c.splice(from, 1);
    c.splice(to, 0, moved);
    setDragKey(null);
    commit(c, activeKey);
  }

  async function save() {
    setSaving(true);
    try {
      const payload = clauses.map((c) => ({ id: c.id, html: c.html, tags: c.tags, changed: !!c.changed }));
      const r = await api.post('/clause/doc-save', { source: doc.source, clauses: payload });
      toast.success(`Saved ${r.clauses} clauses`);
      setClauses((cs) => cs.map((c) => ({ ...c, changed: false })));
      setDirty(false); setHist([]); setFuture([]); reloadDocs();
    } catch (e) { toast.error(e.message || 'Save failed'); }
    finally { setSaving(false); }
  }

  async function delDoc(d) {
    if (!window.confirm(`Delete "${d.source}" and all ${d.clauses} clauses? This cannot be undone.`)) return;
    try { await api.post('/clause/doc-delete', { source: d.source }); toast.success('Document deleted'); if (doc?.source === d.source) closeDoc(); reloadDocs(); }
    catch (e) { toast.error(e.message || 'Could not delete'); }
  }

  function startResize(e) {
    e.preventDefault();
    const rect = splitRef.current?.getBoundingClientRect(); if (!rect) return;
    const onMove = (ev) => setPaneW(Math.min(68, Math.max(22, ((rect.right - ev.clientX) / rect.width) * 100)));
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); document.body.style.userSelect = ''; };
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
  }

  // ---------- Document list ----------
  if (!doc) {
    return (
      <div className="studio-shell">
        <PageHeader fullForm="Regulatory Library" title="Document Studio" scope="Edit clauses with the original document alongside">
          <button className="btn btn-primary btn-sm" onClick={() => setImportOpen(true)}><i className="fas fa-file-import" /> Import PDF</button>
        </PageHeader>
        <div className="page-body">
          {docs === null ? <div className="studio-loading"><Spinner size={15} /> Loading…</div>
            : docs.length === 0 ? <EmptyState icon="fa-folder-open">No documents.</EmptyState>
              : (
                <div className="studio-doc-grid">
                  {docs.map((d) => (
                    <div key={d.source} className="studio-doc" onClick={() => openDoc(d)} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') openDoc(d); }}>
                      <button className="studio-doc-del" title="Delete document" onClick={(e) => { e.stopPropagation(); delDoc(d); }}><i className="fas fa-trash" /></button>
                      <span className="studio-doc-name">{d.source}</span>
                      <span className="studio-doc-meta">{d.clauses} clauses · {d.edited} edited</span>
                    </div>
                  ))}
                </div>
              )}
        </div>
        {importOpen && <ImportModal onClose={() => setImportOpen(false)} onDone={() => { setImportOpen(false); reloadDocs(); }} />}
      </div>
    );
  }

  // ---------- Document editor ----------
  return (
    <div className="studio-shell">
      <PageHeader fullForm="Regulatory Library" title="Document Studio" scope={doc.source}>
        <button className="btn btn-ghost btn-sm" onClick={closeDoc}><i className="fas fa-arrow-left" /> Documents</button>
        <button className="btn btn-ghost btn-sm" onClick={() => setShowRail((s) => !s)}><i className={`fas ${showRail ? 'fa-list-ul' : 'fa-list'}`} /> {showRail ? 'Hide clauses' : 'Show clauses'}</button>
        {doc.pdf_url && <button className="btn btn-ghost btn-sm" onClick={() => setShowPdf((s) => !s)}><i className={`fas ${showPdf ? 'fa-eye-slash' : 'fa-file-pdf'}`} /> {showPdf ? 'Hide PDF' : 'Show PDF'}</button>}
        <button className="btn btn-primary btn-sm" onClick={save} disabled={saving || !dirty}>{saving ? <Spinner size={13} color="#fff" /> : <i className="fas fa-floppy-disk" />} Save{dirty ? ' *' : ''}</button>
      </PageHeader>

      {/* Action bar */}
      <div className="studio-actionbar">
        <button title="Undo" onClick={undo} disabled={!hist.length}><i className="fas fa-rotate-left" /></button>
        <button title="Redo" onClick={redo} disabled={!future.length}><i className="fas fa-rotate-right" /></button>
        <span className="ab-div" />
        <button title="Add clause below" onClick={addClause}><i className="fas fa-plus" /> Add</button>
        <button title="Split at cursor" onClick={splitClause}><i className="fas fa-scissors" /> Split</button>
        <button title="Merge into clause above" onClick={mergeUp}><i className="fas fa-up-long" /> Merge ↑</button>
        <button title="Merge into clause below" onClick={mergeDown}><i className="fas fa-down-long" /> Merge ↓</button>
        <button className="danger" title="Delete clause" onClick={delClause}><i className="fas fa-trash" /> Delete</button>
        <span className="ab-div" />
        <button title="Version history" onClick={() => setHistoryOpen(true)} disabled={!active}><i className="fas fa-clock-rotate-left" /> History</button>
        <span className="ab-spacer" />
        <span className="ab-hint">Drag clauses to reorder</span>
      </div>

      <div className="studio-body">
        {showRail && (
        <aside className="studio-rail">
          <div className="studio-rail-body">
            {clauses === null ? <div className="studio-loading"><Spinner size={15} /> Loading…</div>
              : clauses.map((c) => (
                <div key={c.key}
                  className={`studio-clause ${activeKey === c.key ? 'is-active' : ''} ${dragKey === c.key ? 'dragging' : ''}`}
                  draggable onDragStart={() => setDragKey(c.key)} onDragOver={(e) => e.preventDefault()} onDrop={() => onDrop(c.key)}
                  onClick={() => setActiveKey(c.key)} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') setActiveKey(c.key); }}>
                  <i className="fas fa-grip-vertical studio-grip" />
                  <span className="studio-clause-id">{c.id}</span>
                  {c.edited && <i className="fas fa-pen studio-edited" title="Edited" />}
                </div>
              ))}
          </div>
        </aside>
        )}

        <div className="studio-main" ref={splitRef}>
          <div className="studio-editor">
            {loading || clauses === null ? <div className="studio-loading"><Spinner size={16} /> Loading clauses…</div>
              : !active ? <div className="studio-placeholder"><i className="fas fa-pen-to-square" /><p>This document has no clauses.</p></div>
                : (
                  <>
                    <div className="studio-meta">
                      <label>
                        <span className="studio-meta-lbl">Clause ID</span>
                        <input className="input" value={active.id} onChange={(e) => setField('id', e.target.value)} />
                      </label>
                      <label>
                        <span className="studio-meta-lbl">Tags <span className="studio-meta-hint">(comma-separated)</span></span>
                        <input className="input" value={active.tags.join(', ')}
                          onChange={(e) => setField('tags', e.target.value.split(',').map((t) => t.trim()).filter(Boolean))} />
                      </label>
                    </div>
                    <StudioEditor key={`${active.key}:${rev}`} value={active.html} onChange={onEdit} apiRef={editorApi} />
                  </>
                )}
          </div>

          {showPdf && doc.pdf_url && (
            <>
              <div className="studio-resizer" onMouseDown={startResize} />
              <aside className="studio-pdf" style={{ flexBasis: `${paneW}%` }}>
                <div className="studio-pdf-head"><i className="fas fa-file-pdf" /> {doc.source}<span className="studio-pdf-tag">{doc.has_uploaded_pdf ? 'uploaded' : 'bundled'}</span></div>
                <PdfViewer key={doc.pdf_url} url={doc.pdf_url} />
              </aside>
            </>
          )}
        </div>
      </div>
      {historyOpen && active && <HistoryModal source={doc.source} id={active.id} onRestore={applyRestore} onClose={() => setHistoryOpen(false)} />}
    </div>
  );
}

function HistoryModal({ source, id, onRestore, onClose }) {
  const toast = useToast();
  const [versions, setVersions] = useState(null);
  useEffect(() => {
    api.get(`/clause/history?source=${encodeURIComponent(source)}&id=${encodeURIComponent(id)}`)
      .then((d) => setVersions(d.versions || []))
      .catch((e) => { toast.error(e.message || 'Could not load history'); onClose(); });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  const fmt = (s) => (s ? s.replace('T', ' ').slice(0, 16) : '');
  return (
    <Modal title={`History — ${id}`} width="560px" onClose={onClose}>
      {versions === null ? <div className="studio-loading"><Spinner size={15} /> Loading…</div>
        : versions.length === 0 ? <p className="guide-intro">No prior versions yet. Versions are recorded each time you save, merge, or delete.</p>
          : (
            <div className="hist-list">
              {versions.map((v) => (
                <div className="hist-row" key={v.version_id}>
                  <div className="hist-info">
                    <div className="hist-meta">{fmt(v.edited_at)} · {v.edited_by || '—'}</div>
                    <div className="hist-snip">{v.snippet || '(empty)'}</div>
                  </div>
                  <button className="btn btn-ghost btn-sm" onClick={() => onRestore(v)}>Restore</button>
                </div>
              ))}
            </div>
          )}
    </Modal>
  );
}

function ImportModal({ onClose, onDone }) {
  const toast = useToast();
  const [specs, setSpecs] = useState([]);
  const [file, setFile] = useState(null);
  const [specId, setSpecId] = useState('');
  const [source, setSource] = useState('');
  const [docType, setDocType] = useState('REGULATION');
  const [category, setCategory] = useState('GENERAL');
  const [busy, setBusy] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [detected, setDetected] = useState(null);

  useEffect(() => { api.get('/clause/specs').then((d) => setSpecs(d.specs || [])).catch(() => {}); }, []);

  async function onFile(e) {
    const f = e.target.files?.[0] || null;
    setFile(f); setDetected(null);
    if (!f) return;
    if (!source.trim()) setSource(f.name.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' '));
    setDetecting(true);
    try {
      const fd = new FormData(); fd.append('file', f);
      const d = await api.post('/clause/detect-spec', fd);
      const best = (d.ranked || [])[0];
      if (best) { setSpecId(best.spec_id); setDetected(best); }
    } catch { /* manual */ } finally { setDetecting(false); }
  }

  async function go() {
    if (!file || !specId || !source.trim()) { toast.error('PDF, spec and a document name are required'); return; }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', file); fd.append('spec_id', specId); fd.append('source', source.trim());
      fd.append('doc_type', docType); fd.append('category', category);
      const r = await api.post('/clause/import-pdf', fd);
      toast.success(`Imported ${r.clauses} clauses${r.orphans ? ` (${r.orphans} unmatched lines)` : ''}`);
      onDone();
    } catch (e) { toast.error(e.message || 'Import failed'); }
    finally { setBusy(false); }
  }

  return (
    <Modal title="Import document from PDF" width="560px" onClose={onClose}
      footer={(
        <>
          <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={go} disabled={busy || !file || !specId || !source.trim()}>
            {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-file-import" />} Import &amp; create
          </button>
        </>
      )}>
      <p className="guide-intro">Pick the matching document type — IRIS segments the PDF into editable clauses (deterministic, no AI) and keeps the PDF for reference/download.</p>
      <div className="field" style={{ marginBottom: 12 }}>
        <label>PDF file</label>
        <input className="input" type="file" accept="application/pdf" onChange={onFile} />
      </div>
      <div className="field" style={{ marginBottom: 12 }}>
        <label>Document type (spec)</label>
        <select className="input" value={specId} onChange={(e) => setSpecId(e.target.value)}>
          <option value="">— select the matching document —</option>
          {specs.map((s) => <option key={s.id} value={s.id}>{s.doc_id || s.id}</option>)}
        </select>
        {detecting && <div className="studio-detect"><Spinner size={12} /> Detecting best match…</div>}
        {!detecting && detected && (
          <div className="studio-detect">
            <i className="fas fa-wand-magic-sparkles" /> Auto-detected: <strong>{detected.doc_id}</strong>
            {' '}({detected.clauses} clauses{detected.orphans ? `, ${detected.orphans} unmatched lines` : ''}).
          </div>
        )}
      </div>
      <div className="field" style={{ marginBottom: 12 }}>
        <label>Document name (as shown in IRIS)</label>
        <input className="input" value={source} onChange={(e) => setSource(e.target.value)} placeholder="e.g. EoM Regulations 2024" />
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        <div className="field" style={{ flex: 1 }}>
          <label>Type band</label>
          <select className="input" value={docType} onChange={(e) => setDocType(e.target.value)}>
            <option value="ACT">Act</option><option value="REGULATION">Regulation</option>
            <option value="MASTER">Master Circular</option><option value="CIRCULAR">Circular</option><option value="GUIDELINE">Guideline</option>
          </select>
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label>Department</label>
          <select className="input" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="GENERAL">General</option><option value="HEALTH">Health</option>
            <option value="LIFE">Life</option><option value="NONLIFE">Non-Life</option>
          </select>
        </div>
      </div>
    </Modal>
  );
}
