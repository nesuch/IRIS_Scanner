import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, EmptyState } from '../components/UI.jsx';
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
  const splitRef = useRef(null);

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
                    <button key={d.source} className="studio-doc" onClick={() => openDoc(d)}>
                      <span className="studio-doc-name">{d.source}</span>
                      <span className="studio-doc-meta">{d.clauses} clauses · {d.edited} edited</span>
                    </button>
                  ))
            ) : (
              clauses === null ? <div className="studio-loading"><Spinner size={15} /> Loading…</div>
                : clauses.map((c) => (
                  <button key={c.id} className={`studio-clause ${active?.id === c.id ? 'is-active' : ''}`} onClick={() => openClause(c)}>
                    <span className="studio-clause-id">{c.id}{c.edited && <i className="fas fa-pen studio-edited" title="Edited" />}</span>
                    <span className="studio-clause-prev">{c.preview}</span>
                  </button>
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
    </div>
  );
}
