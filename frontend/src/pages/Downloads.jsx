import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { EmptyState, PageLoading } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import PdfViewer from './search/PdfViewer.jsx';
import { api } from '../api.js';
import './downloads/downloads.css';

const TYPE_ICON = {
  Act: 'fa-scale-balanced', Regulation: 'fa-book', 'Master Circular': 'fa-file-lines',
  Circular: 'fa-file-circle-check', Guideline: 'fa-file-pen',
};

function fmtDate(d) {
  if (!d) return null;
  try { return new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }); }
  catch { return d; }
}

function DocNode({ node, depth, onView, onRemove }) {
  const repealed = (node.status || '').toLowerCase() === 'repealed';
  return (
    <div className="doc-node" style={{ marginLeft: depth ? 22 : 0 }}>
      <div className={`doc-node-row ${repealed ? 'is-repealed' : ''}`}>
        <i className={`fas ${TYPE_ICON[node.type] || 'fa-file'} doc-node-icon`} />
        <div className="doc-node-main">
          <div className="doc-node-title">
            {node.title}
            <span className={`doc-status ${repealed ? 'st-repealed' : 'st-active'}`}>{repealed ? 'Repealed' : 'Active'}</span>
          </div>
          <div className="doc-node-meta">
            <span className="doc-type-tag">{node.type}</span>
            {node.clauses > 0 && <span>{node.clauses} clauses</span>}
            {node.effective_date && <span>Effective {fmtDate(node.effective_date)}</span>}
            {repealed && node.repealed_on && <span className="meta-repeal">Repealed {fmtDate(node.repealed_on)}{node.repealed_by ? ` · by ${node.repealed_by}` : ''}</span>}
          </div>
        </div>
        <span className="doc-node-actions">
          {node.download_url ? (
            <>
              <button className="btn btn-ghost btn-sm doc-dl" onClick={() => onView(node)}><i className="fas fa-eye" /> View</button>
              <a className="btn btn-ghost btn-sm doc-dl" href={node.download_url} target="_blank" rel="noreferrer"><i className="fas fa-download" /> PDF</a>
            </>
          ) : <span className="doc-dl-missing">No file</span>}
          {onRemove && (
            <button className="btn btn-ghost btn-sm doc-dl doc-dl-remove" title="Remove this stray document (clauses + PDF)" onClick={() => onRemove(node)}>
              <i className="fas fa-trash" /> Remove
            </button>
          )}
        </span>
      </div>
      {node.children?.length > 0 && (
        <div className="doc-children">{node.children.map((c) => <DocNode key={c.id} node={c} depth={depth + 1} onView={onView} />)}</div>
      )}
    </div>
  );
}

// In-page PDF viewer overlay (so a doc can be read without leaving Downloads).
function DocViewer({ node, onClose }) {
  // Freeze the background page scroll while the viewer is open.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, []);
  return (
    <div className="dl-viewer-overlay" onMouseDown={onClose}>
      <div className="dl-viewer" onMouseDown={(e) => e.stopPropagation()}>
        <div className="dl-viewer-head">
          <span className="dl-viewer-title"><i className="fas fa-file-pdf" /> {node.title}</span>
          <span className="dl-viewer-actions">
            <a href={node.download_url} target="_blank" rel="noreferrer" title="Open in new tab"><i className="fas fa-arrow-up-right-from-square" /></a>
            <button onClick={onClose} title="Close" aria-label="Close"><i className="fas fa-xmark" /></button>
          </span>
        </div>
        <div className="dl-viewer-body"><PdfViewer url={node.download_url} /></div>
      </div>
    </div>
  );
}

export default function Downloads() {
  const toast = useToast();
  const { user } = useAuth();
  const isAdmin = !!(user?.is_admin);
  const [data, setData] = useState(null);
  const [viewing, setViewing] = useState(null);

  const load = () => api.get('/documents').then(setData)
    .catch((e) => { toast.error(e.message || 'Could not load documents'); setData({ tree: [], repealed: [] }); });

  useEffect(() => { load(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  async function removeStray(node) {
    if (!window.confirm(`Remove "${node.title}"? This deletes its clauses and any attached PDF. This is for stray/duplicate imports.`)) return;
    try {
      await api.post('/clause/doc-delete', { source: node.id });
      toast.success('Removed');
      load();
    } catch (e) { toast.error(e.message || 'Could not remove'); }
  }

  if (!data) return (<><PageHeader fullForm="Regulatory Library" title="Downloads" scope="Acts, Regulations & Circulars" /><div className="page-body"><PageLoading /></div></>);

  const { tree = [], repealed = [], imported = [] } = data;
  return (
    <>
      <PageHeader fullForm="Regulatory Library" title="Downloads" scope="Acts, Regulations & Circulars" />
      <div className="page-body">
        <p className="dl-intro">The regulatory hierarchy — circulars operationalise the regulations they sit under, which in turn flow from the Acts. View any document in-app or download it as PDF.</p>
        {tree.length === 0 ? <EmptyState icon="fa-folder-open">No documents available.</EmptyState>
          : <div className="doc-tree card pad">{tree.map((n) => <DocNode key={n.id} node={n} depth={0} onView={setViewing} />)}</div>}

        {imported.length > 0 && (
          <>
            <h3 className="dl-section-head"><i className="fas fa-file-import" /> Imported documents</h3>
            <div className="doc-tree card pad">{imported.map((n) => <DocNode key={n.id} node={n} depth={0} onView={setViewing} onRemove={isAdmin ? removeStray : undefined} />)}</div>
          </>
        )}

        {repealed.length > 0 && (
          <>
            <h3 className="dl-section-head"><i className="fas fa-ban" /> Repealed / Superseded</h3>
            <div className="doc-tree card pad repealed-tree">{repealed.map((n) => <DocNode key={n.id} node={n} depth={0} onView={setViewing} />)}</div>
          </>
        )}
      </div>
      {viewing && <DocViewer node={viewing} onClose={() => setViewing(null)} />}
    </>
  );
}
