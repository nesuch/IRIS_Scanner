import { useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { useToast } from '../components/Toast.jsx';
import { api } from '../api.js';

const CATEGORIES = ['Bug', 'Suggestion', 'UI Issue', 'Other (please specify)'];

export default function Feedback() {
  const toast = useToast();
  const [category, setCategory] = useState('Suggestion');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    try {
      const res = await api.post('/feedback', { category, message });
      toast.success(res.message || 'Feedback submitted');
      setMessage(''); setCategory('Suggestion'); setDone(true);
    } catch (err) {
      toast.error(err.message || 'Could not submit feedback');
    } finally { setBusy(false); }
  }

  return (
    <>
      <PageHeader fullForm="Help us improve" title="Feedback" scope="Share bugs, suggestions, and UI issues" />
      <div className="page-body">
        <form className="card pad anim-rise" onSubmit={submit} style={{ maxWidth: 700, margin: '0 auto' }}>
          {done && <div className="auth-ok" style={{ marginBottom: 16 }}><i className="fas fa-circle-check" /> Thanks! Your feedback has been submitted.</div>}
          <div className="field" style={{ marginBottom: 16 }}>
            <label>Category</label>
            <select className="select" value={category} onChange={(e) => setCategory(e.target.value)}>
              {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Message</label>
            <textarea className="input" required minLength={5} rows={6} value={message}
              onChange={(e) => { setMessage(e.target.value); setDone(false); }} placeholder="Tell us what's on your mind…" />
          </div>
          <button className="btn btn-primary" type="submit" disabled={busy} style={{ marginTop: 16 }}>
            <i className="fas fa-paper-plane" /> Submit Feedback
          </button>
        </form>
      </div>
    </>
  );
}
