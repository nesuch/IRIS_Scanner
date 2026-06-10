import { useState, useEffect } from 'react';
import { Modal, Spinner } from './UI.jsx';
import { useToast } from './Toast.jsx';
import { api } from '../api.js';

const REASONS = [
  'Wrong information',
  'Outdated / superseded',
  'Wrong document/source',
  'Formatting issue',
  'Other',
];

// Report an issue on a clause or a financial data row.
// props: kind ('clause'|'financial'), target (id/context), detail (snapshot),
//        reasons (optional list to override the default), title (optional), onClose
export default function FlagModal({ kind = 'clause', target, detail, onClose, reasons, title }) {
  const toast = useToast();
  const reasonList = reasons && reasons.length ? reasons : REASONS;
  const [reason, setReason] = useState(reasonList[0]);
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);
  const [shot, setShot] = useState(null);          // captured data URL
  const [attach, setAttach] = useState(true);
  const [capturing, setCapturing] = useState(true);

  // Capture the screen once on open, excluding this modal so the report itself
  // is what the admin sees. Best-effort — failures just disable the attachment.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { default: html2canvas } = await import('html2canvas');
        // Capture at the screen's real pixel density (cap at 2×) for a sharp image.
        const canvas = await html2canvas(document.body, {
          scale: Math.min(window.devicePixelRatio || 1, 2),
          logging: false, useCORS: true, backgroundColor: '#ffffff',
          windowWidth: document.documentElement.scrollWidth,
          windowHeight: document.documentElement.scrollHeight,
          ignoreElements: (el) => el.classList?.contains('modal-overlay'),
          // Force full opacity / no mid-flight animations so the capture isn't faint.
          onclone: (doc) => {
            const s = doc.createElement('style');
            s.textContent = '*{animation:none!important;transition:none!important;opacity:1!important;filter:none!important;}';
            doc.head.appendChild(s);
          },
        });
        if (alive) setShot(canvas.toDataURL('image/jpeg', 0.9));
      } catch {
        if (alive) setShot(null);
      } finally {
        if (alive) setCapturing(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  async function submit() {
    setBusy(true);
    try {
      await api.post('/flag', {
        kind, reason, description, target, detail,
        screenshot: attach && shot ? shot : undefined,
      });
      toast.success('Flag submitted — thank you.');
      onClose();
    } catch (e) {
      toast.error(e.message || 'Could not submit flag.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title={title || (kind === 'financial' ? 'Flag this data' : 'Flag this clause')} width="520px" onClose={onClose}
      footer={(
        <>
          <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={submit} disabled={busy}>
            {busy ? <Spinner size={14} color="#fff" /> : <i className="fas fa-flag" />} Submit flag
          </button>
        </>
      )}>
      {target && <div className="flag-target"><i className="fas fa-location-dot" /> {target}</div>}
      <div className="field" style={{ marginBottom: 14 }}>
        <label>What's wrong?</label>
        <select className="select" value={reason} onChange={(e) => setReason(e.target.value)}>
          {reasonList.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>
      <div className="field" style={{ marginBottom: 14 }}>
        <label>Description (optional)</label>
        <textarea className="input" rows={4} value={description} onChange={(e) => setDescription(e.target.value)}
          placeholder="Add any detail that helps us fix this…" />
      </div>
      <div className="flag-shot">
        <label className="flag-shot-toggle">
          <input type="checkbox" checked={attach && !!shot} disabled={!shot || capturing}
            onChange={(e) => setAttach(e.target.checked)} />
          {capturing
            ? <span className="flag-shot-status"><Spinner size={12} /> Capturing screenshot…</span>
            : shot
              ? 'Attach screenshot of this screen (helps the reviewer)'
              : <span className="flag-shot-status">Screenshot unavailable</span>}
        </label>
        {shot && attach && <img className="flag-shot-preview" src={shot} alt="screen preview" />}
      </div>
    </Modal>
  );
}
