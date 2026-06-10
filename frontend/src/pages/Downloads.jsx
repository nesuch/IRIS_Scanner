import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { EmptyState, PageLoading } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
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

function DocNode({ node, depth }) {
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
        {node.download_url
          ? <a className="btn btn-ghost btn-sm doc-dl" href={node.download_url} target="_blank" rel="noreferrer"><i className="fas fa-download" /> PDF</a>
          : <span className="doc-dl-missing">No file</span>}
      </div>
      {node.children?.length > 0 && (
        <div className="doc-children">{node.children.map((c) => <DocNode key={c.id} node={c} depth={depth + 1} />)}</div>
      )}
    </div>
  );
}

export default function Downloads() {
  const toast = useToast();
  const [data, setData] = useState(null);

  useEffect(() => {
    api.get('/documents').then(setData)
      .catch((e) => { toast.error(e.message || 'Could not load documents'); setData({ tree: [], repealed: [] }); });
  }, []);

  if (!data) return (<><PageHeader fullForm="Regulatory Library" title="Downloads" scope="Acts, Regulations & Circulars" /><div className="page-body"><PageLoading /></div></>);

  const { tree = [], repealed = [] } = data;
  return (
    <>
      <PageHeader fullForm="Regulatory Library" title="Downloads" scope="Acts, Regulations & Circulars" />
      <div className="page-body">
        <p className="dl-intro">The regulatory hierarchy — circulars operationalise the regulations they sit under, which in turn flow from the Acts. Download any document as PDF.</p>
        {tree.length === 0 ? <EmptyState icon="fa-folder-open">No documents available.</EmptyState>
          : <div className="doc-tree card pad">{tree.map((n) => <DocNode key={n.id} node={n} depth={0} />)}</div>}

        {repealed.length > 0 && (
          <>
            <h3 className="dl-section-head"><i className="fas fa-ban" /> Repealed / Superseded</h3>
            <div className="doc-tree card pad repealed-tree">{repealed.map((n) => <DocNode key={n.id} node={n} depth={0} />)}</div>
          </>
        )}
      </div>
    </>
  );
}
