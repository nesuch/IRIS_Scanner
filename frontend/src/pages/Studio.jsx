import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, EmptyState, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { ClauseEditorPanel } from '../components/ClauseEditor.jsx';
import PdfViewer from './search/PdfViewer.jsx';
import { api } from '../api.js';
import './studio/studio.css';

function AttachPdf({ doc, onAttached }) {
  const toast = useToast();
  const ref = useRef(null);
  const [busy, setBusy] = useState(false);
  async function onFile(e) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) { toast.error('Choose a PDF file'); return; }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('source', doc.source);
      fd.append('file', file);
      const r = await api.post('/clause/doc-pdf', fd);
      toast.success('Original PDF attached');
      onAttached(r.pdf_url);
    } catch (e) { toast.error(e.message || 'Upload failed'); }
    finally { setBusy(false); }
  }
  return (
    <>
      <input ref={ref} type="file" accept="application/pdf" hidden onChange={onFile} />
      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => ref.current?.click()}>
        {busy ? <Spinner size={13} /> : <i className="fas fa-file-arrow-up" />} {doc.pdf_url ? 'Replace PDF' : 'Attach PDF'}
      </button>
    </>
  );
}

export default function Studio() {
  const toast = useToast();
  const [docs, setDocs] = useState(null);
  const [doc, setDoc] = useState(null);          // selected document {source, pdf_url, ...}
  const [clauses, setClauses] = useState(null);
  const [active, setActive] = useState(null);    // { id, source } being edited
  const [initialHtml, setInitialHtml] = useState(null);
  const [loadingClause, setLoadingClause] = useState(false);
  const [showPdf, setShowPdf] = useState(true);
  const [paneW, setPaneW] = useState(40);        // % width of PDF pane
  const [importOpen, setImportOpen] = useState(false);
  const splitRef = useRef(null);

  function reloadDocs() {
    return api.get('/clause/docs').then((d) => setDocs(d.docs || [])).catch(() => {});
  }

  async function delDoc(d) {
    if (!window.confirm(`Delete "${d.source}" and all ${d.clauses} clauses? This cannot be undone.`)) return;
    try {
      const r = await api.post('/clause/doc-delete', { source: d.source });
      toast.success(`Deleted ${r.deleted} clauses`);
      if (doc?.source === d.source) { setDoc(null); setActive(null); }
      reloadDocs();
    } catch (e) { toast.error(e.message || 'Could not delete'); }
  }

  useEffect(() => {
    api.get('/clause/docs').then((d) => setDocs(d.docs || []))
      .catch((e) => { toast.error(e.message || 'Could not load documents'); setDocs([]); });
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  function openDoc(d) {
    setDoc(d); setActive(null); setInitialHtml(null); setClauses(null);
    api.get(`/clause/list?source=${encodeURIComponent(d.source)}`)
      .then((r) => setClauses(r.clauses || []))
      .catch((e) => { toast.error(e.message || 'Could not load clauses'); setClauses([]); });
  }

  function openClause(c) {
    setActive({ id: c.id, source: doc.source }); setInitialHtml(null); setLoadingClause(true);
    api.get(`/clause/edit?id=${encodeURIComponent(c.id)}&source=${encodeURIComponent(doc.source)}`)
      .then((d) => setInitialHtml(d.html || '<p></p>'))
      .catch((e) => { toast.error(e.message || 'Could not open clause'); setActive(null); })
      .finally(() => setLoadingClause(false));
  }

  function onSaved() {
    // mark the clause as edited in the list
    setClauses((cs) => cs.map((c) => (c.id === active.id ? { ...c, edited: true } : c)));
  }

  function reloadClauses() {
    if (!doc) return Promise.resolve();
    return api.get(`/clause/list?source=${encodeURIComponent(doc.source)}`)
      .then((r) => setClauses(r.clauses || [])).catch(() => {});
  }

  async function clauseOp(e, path, body, okMsg) {
    e?.stopPropagation();
    try { const r = await api.post(path, body); toast.success(okMsg); return r; }
    catch (err) { toast.error(err.message || 'Action failed'); return null; }
    finally { reloadClauses(); }
  }

  async function addBelow(e, c) {
    const r = await clauseOp(e, '/clause/add', { source: doc.source, after_id: c.id }, 'Clause added');
    if (r?.id) openClause({ id: r.id });
  }
  async function removeClause(e, c) {
    e.stopPropagation();
    if (!window.confirm(`Delete clause ${c.id}?`)) return;
    if (active?.id === c.id) setActive(null);
    clauseOp(null, '/clause/remove', { source: doc.source, id: c.id }, 'Clause deleted');
  }

  function startResize(e) {
    e.preventDefault();
    const rect = splitRef.current?.getBoundingClientRect();
    if (!rect) return;
    const onMove = (ev) => setPaneW(Math.min(70, Math.max(24, ((rect.right - ev.clientX) / rect.width) * 100)));
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); document.body.style.userSelect = ''; };
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
  }

  return (
    <div className="studio-shell">
      <PageHeader fullForm="Regulatory Library" title="Document Studio" scope="Edit clauses with the original document alongside">
        {!doc && <button className="btn btn-primary btn-sm" onClick={() => setImportOpen(true)}><i className="fas fa-file-import" /> Import PDF</button>}
        {doc && <AttachPdf doc={doc} onAttached={(url) => { setDoc((d) => ({ ...d, pdf_url: url, has_uploaded_pdf: true })); setShowPdf(true); }} />}
        {doc?.pdf_url && (
          <button className="btn btn-ghost btn-sm" onClick={() => setShowPdf((s) => !s)}>
            <i className={`fas ${showPdf ? 'fa-eye-slash' : 'fa-file-pdf'}`} /> {showPdf ? 'Hide PDF' : 'Show PDF'}
          </button>
        )}
      </PageHeader>

      <div className="studio-body">
        {/* Left rail: documents + clauses */}
        <aside className="studio-rail">
          <div className="studio-rail-head">{doc ? <button className="studio-back" onClick={() => { setDoc(null); setActive(null); }}><i className="fas fa-arrow-left" /> Documents</button> : 'Documents'}</div>
          <div className="studio-rail-body">
            {!doc ? (
              docs === null ? <div className="studio-loading"><Spinner size={15} /> Loading…</div>
                : docs.length === 0 ? <EmptyState icon="fa-folder-open">No documents.</EmptyState>
                  : docs.map((d) => (
                    <div key={d.source} className="studio-doc" onClick={() => openDoc(d)} role="button" tabIndex={0}
                      onKeyDown={(e) => { if (e.key === 'Enter') openDoc(d); }}>
                      <button className="studio-doc-del" title="Delete document" onClick={(e) => { e.stopPropagation(); delDoc(d); }}><i className="fas fa-trash" /></button>
                      <span className="studio-doc-name">{d.source}</span>
                      <span className="studio-doc-meta">{d.clauses} clauses · {d.edited} edited</span>
                    </div>
                  ))
            ) : (
              clauses === null ? <div className="studio-loading"><Spinner size={15} /> Loading…</div>
                : clauses.map((c) => (
                  <div key={c.id} className={`studio-clause ${active?.id === c.id ? 'is-active' : ''}`}>
                    <div className="studio-clause-main" onClick={() => openClause(c)} role="button" tabIndex={0}
                      onKeyDown={(e) => { if (e.key === 'Enter') openClause(c); }}>
                      <span className="studio-clause-id">{c.id}{c.edited && <i className="fas fa-pen studio-edited" title="Edited" />}</span>
                      <span className="studio-clause-prev">{c.preview}</span>
                    </div>
                    <div className="studio-clause-acts">
                      <button title="Move up" onClick={(e) => clauseOp(e, '/clause/move', { source: doc.source, id: c.id, direction: 'up' }, 'Moved up')}><i className="fas fa-arrow-up" /></button>
                      <button title="Move down" onClick={(e) => clauseOp(e, '/clause/move', { source: doc.source, id: c.id, direction: 'down' }, 'Moved down')}><i className="fas fa-arrow-down" /></button>
                      <button title="Add clause below" onClick={(e) => addBelow(e, c)}><i className="fas fa-plus" /></button>
                      <button title="Merge into clause above" onClick={(e) => clauseOp(e, '/clause/merge', { source: doc.source, id: c.id }, 'Merged up')}><i className="fas fa-up-long" /></button>
                      <button title="Delete clause" className="danger" onClick={(e) => removeClause(e, c)}><i className="fas fa-trash" /></button>
                    </div>
                  </div>
                ))
            )}
          </div>
        </aside>

        {/* Editor + PDF split */}
        <div className="studio-main" ref={splitRef}>
          <div className="studio-editor">
            {!active ? (
              <div className="studio-placeholder"><i className="fas fa-pen-to-square" /><p>{doc ? 'Pick a clause on the left to edit it.' : 'Pick a document to begin.'}</p></div>
            ) : loadingClause || initialHtml === null ? (
              <div className="studio-loading"><Spinner size={16} /> Loading clause…</div>
            ) : (
              <>
                <div className="studio-editing-id">Editing <strong>{active.id}</strong> · {active.source}</div>
                <ClauseEditorPanel key={active.id} clause={active} initialHtml={initialHtml} onSaved={onSaved} />
              </>
            )}
          </div>

          {showPdf && doc?.pdf_url && (
            <>
              <div className="studio-resizer" onMouseDown={startResize} title="Drag to resize" />
              <aside className="studio-pdf" style={{ flexBasis: `${paneW}%` }}>
                <div className="studio-pdf-head">
                  <i className="fas fa-file-pdf" /> {doc.source}
                  <span className="studio-pdf-tag">{doc.has_uploaded_pdf ? 'uploaded' : 'bundled'}</span>
                </div>
                <PdfViewer key={doc.pdf_url} url={doc.pdf_url} />
              </aside>
            </>
          )}
        </div>
      </div>
      {importOpen && <ImportModal onClose={() => setImportOpen(false)} onDone={() => { setImportOpen(false); reloadDocs(); }} />}
    </div>
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
    } catch { /* fall back to manual pick */ }
    finally { setDetecting(false); }
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
            {detected.orphans > 0 && ' Check it’s the right type, or pick another.'}
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
            <option value="ACT">Act</option>
            <option value="REGULATION">Regulation</option>
            <option value="MASTER">Master Circular</option>
            <option value="CIRCULAR">Circular</option>
            <option value="GUIDELINE">Guideline</option>
          </select>
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label>Department</label>
          <select className="input" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="GENERAL">General</option>
            <option value="HEALTH">Health</option>
            <option value="LIFE">Life</option>
            <option value="NONLIFE">Non-Life</option>
          </select>
        </div>
      </div>
    </Modal>
  );
}
