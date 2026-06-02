import { useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import { Spinner } from '../components/UI.jsx';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [success, setSuccess] = useState(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setBusy(true);
    try {
      const data = await api.post('/forgot-password', { email });
      setSuccess(data.message);
    } catch (err) {
      // Endpoint always returns the generic message; show it regardless.
      setSuccess('If the account is eligible, a password reset link has been generated.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrap">
      <div className="auth-bg-glow g1" />
      <div className="auth-bg-glow g2" />
      <div className="auth-card">
        <div className="auth-brand">
          <img src="/static/iris_logo.png" alt="IRIS" />
          <span className="wm grad-text">IRIS</span>
          <span className="wm-sub">Reset your password</span>
        </div>

        {success && <div className="auth-ok"><i className="fas fa-circle-check" /> {success}</div>}

        <form className="auth-form" onSubmit={onSubmit}>
          <div className="field">
            <label>Official Email</label>
            <input className="input" type="email" placeholder="name@irdai.gov.in"
              value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
          </div>
          <button className="btn btn-primary" type="submit" disabled={busy} style={{ width: '100%' }}>
            {busy ? <Spinner size={16} color="#fff" /> : <i className="fas fa-paper-plane" />}
            {busy ? 'Sending…' : 'Send reset link'}
          </button>
        </form>

        <p className="auth-link" style={{ marginTop: 20 }}>
          <Link to="/login">Back to sign in</Link>
        </p>
      </div>
    </div>
  );
}
