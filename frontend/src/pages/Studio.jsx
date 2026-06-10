import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { Spinner, EmptyState } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { ClauseEditorPanel } from '../components/ClauseEditor.jsx';
import PdfViewer from './search/PdfViewer.jsx';
import { api } from '../api.js';
import './studio/studio.css';

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
                <div className="studio-pdf-head"><i className="fas fa-file-pdf" /> {doc.source}</div>
                <PdfViewer url={doc.pdf_url} />
              </aside>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
