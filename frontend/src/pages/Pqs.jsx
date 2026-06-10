import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import { EmptyState, PageLoading, Spinner, Modal } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';
import './pqs/pqs.css';

export default function Pqs() {
  const toast = useToast();
  const { user } = useAuth();
  const isAdmin = !!user?.is_admin;
  const [params, setParams] = useSearchParams();
  const [list, setList] = useState(null);
  const [active, setActive] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);

  function load() {
    return api.get('/pq').then((d) => setList(d.items || []))
      .catch((e) => { toast.error(e.message || 'Could not load PQs'); setList([]); });
  }
  useEffect(() => { load(); }, []);

  function open(id) {
    setLoading(true);
    api.get(`/pq/${id}`).then(setActive)
      .catch((e) => toast.error(e.message || 'Could not load PQ'))
      .finally(() => setLoading(false));
  }

  // Deep-link from search: /pqs?open=<id>
  useEffect(() => {
    const id = params.get('open');
    if (id) { open(id); setParams({}, { replace: true }); }
  }, [params]);  // eslint-disable-line react-hooks/exhaustive-deps

  if (!list) return (<><PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="IRDAI replies to Parliamentary Questions" /><div className="page-body"><PageLoading /></div></>);

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
            <div className="pq-html" dangerouslySetInnerHTML={{ __html: active.html }} />
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <PageHeader fullForm="Regulatory Library" title="Parliamentary Q&A" scope="IRDAI replies to Parliamentary Questions">
        {isAdmin && <button className="btn btn-primary btn-sm" onClick={() => setUploadOpen(true)}><i className="fas fa-upload" /> Upload PQ</button>}
      </PageHeader>
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
      {uploadOpen && <UploadModal onClose={() => setUploadOpen(false)} onDone={() => { setUploadOpen(false); load(); }} />}
    </>
  );
}

function UploadModal({ onClose, onDone }) {
  const toast = useToast();
  const [file, setFile] = useState(null);
  const [tags, setTags] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!file) { toast.error('Choose a .docx file'); return; }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('tags', tags);
      const r = await api.post('/pq/upload', fd);
      toast.success(`Added: ${r.title?.slice(0, 50) || 'PQ'}`);
      onDone();
    } catch (e) {
      toast.error(e.message || 'Upload failed');
    } finally { setBusy(false); }
  }

  return (
    <Modal title="Upload Parliamentary Question" width="520px" onClose={onClose}
      footer={(
        <>
          <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={submit} disabled={busy || !file}>
            {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-upload" />} Upload &amp; publish
          </button>
        </>
      )}>
      <p className="guide-intro">Upload the approved reply as a Word file. IRIS renders it on screen (formatting + tables preserved), keeps the original for download, and makes it searchable.</p>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Word document (.docx)</label>
        <input className="input" type="file" accept=".docx" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      </div>
      <div className="field">
        <label>Tags <span style={{ color: 'var(--faint)', fontWeight: 400 }}>(comma-separated — drives search)</span></label>
        <input className="input" value={tags} onChange={(e) => setTags(e.target.value)}
          placeholder="e.g. senior citizens, claim repudiation, grievance" />
      </div>
    </Modal>
  );
}
