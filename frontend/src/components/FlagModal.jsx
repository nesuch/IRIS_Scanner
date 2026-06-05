import { useState } from 'react';
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
// props: kind ('clause'|'financial'), target (id/context), detail (snapshot), onClose
export default function FlagModal({ kind = 'clause', target, detail, onClose }) {
  const toast = useToast();
  const [reason, setReason] = useState(REASONS[0]);
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    try {
      await api.post('/flag', { kind, reason, description, target, detail });
      toast.success('Flag submitted — thank you.');
      onClose();
    } catch (e) {
      toast.error(e.message || 'Could not submit flag.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title={kind === 'financial' ? 'Flag this data' : 'Flag this clause'} width="520px" onClose={onClose}
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
          {REASONS.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>
      <div className="field">
        <label>Description (optional)</label>
        <textarea className="input" rows={4} value={description} onChange={(e) => setDescription(e.target.value)}
          placeholder="Add any detail that helps us fix this…" />
      </div>
    </Modal>
  );
}
