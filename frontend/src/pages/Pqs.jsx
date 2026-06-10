import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { EmptyState, PageLoading, Spinner } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { api } from '../api.js';
import './pqs/pqs.css';

export default function Pqs() {
  const toast = useToast();
  const [list, setList] = useState(null);
  const [active, setActive] = useState(null);   // full PQ being read
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get('/pq').then((d) => setList(d.items || []))
      .catch((e) => { toast.error(e.message || 'Could not load PQs'); setList([]); });
  }, []);

  function open(id) {
    setLoading(true);
    api.get(`/pq/${id}`).then(setActive)
      .catch((e) => toast.error(e.message || 'Could not load PQ'))
      .finally(() => setLoading(false));
  }

  if (!list) return (<><PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="IRDAI replies to Parliamentary Questions" /><div className="page-body"><PageLoading /></div></>);

  // Reading view
  if (active) {
    return (
      <>
        <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="IRDAI replies to Parliamentary Questions" />
        <div className="page-body">
          <div className="pq-read-bar">
            <button className="btn btn-ghost btn-sm" onClick={() => setActive(null)}><i className="fas fa-arrow-left" /> Back to list</button>
            {active.download_url && (
              <a className="btn btn-primary btn-sm" href={active.download_url} target="_blank" rel="noreferrer"><i className="fas fa-file-word" /> Download original (.docx)</a>
            )}
          </div>
          <div className="pq-doc card">
            <div className="pq-doc-head">
              {active.house && <span className="pq-house">{active.house}</span>}
              {active.pq_no && <span className="pq-no">Q No. {active.pq_no}</span>}
              {active.date && <span className="pq-date">{active.date}</span>}
            </div>
            <h2 className="pq-doc-title">{active.title}</h2>
            {active.tags?.length > 0 && (
              <div className="pq-tags">{active.tags.map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
            )}
            {/* Rendered docx — bold/italic/underline/tables preserved by mammoth */}
            <div className="pq-html" dangerouslySetInnerHTML={{ __html: active.html }} />
          </div>
        </div>
      </>
    );
  }

  // List view
  return (
    <>
      <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="IRDAI replies to Parliamentary Questions" />
      <div className="page-body">
        {loading && <div className="pq-loading"><Spinner size={16} /> Opening…</div>}
        {list.length === 0 ? <EmptyState icon="fa-landmark">No Parliamentary Questions yet.</EmptyState>
          : (
            <div className="pq-list">
              {list.map((p) => (
                <button key={p.id} className="pq-card" onClick={() => open(p.id)}>
                  <div className="pq-card-top">
                    {p.house && <span className="pq-house">{p.house}</span>}
                    {p.pq_no && <span className="pq-no">Q No. {p.pq_no}</span>}
                    {p.date && <span className="pq-date">{p.date}</span>}
                  </div>
                  <div className="pq-card-title">{p.subject || p.title}</div>
                  {p.tags?.length > 0 && (
                    <div className="pq-tags">{p.tags.slice(0, 5).map((t) => <span key={t} className="pq-tag">{t}</span>)}</div>
                  )}
                  <span className="pq-card-open">Read <i className="fas fa-arrow-right" /></span>
                </button>
              ))}
            </div>
          )}
      </div>
    </>
  );
}
