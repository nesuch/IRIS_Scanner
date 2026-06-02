import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api.js';
import { Spinner } from '../components/UI.jsx';

export default function ResetPassword() {
  const { token } = useParams();
  const navigate = useNavigate();
  const [valid, setValid] = useState(null); // null=checking
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get(`/reset-password/${token}/valid`)
      .then((d) => setValid(!!d.valid))
      .catch(() => setValid(false));
  }, [token]);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await api.post(`/reset-password/${token}`, { new_password: newPassword, confirm_password: confirmPassword });
      navigate('/login', { replace: true });
    } catch (err) {
      setError(err.message || 'Could not reset password.');
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
          <span className="wm-sub">Choose a new password</span>
        </div>

        {valid === null && <div className="empty-state" style={{ padding: 20 }}><Spinner size={24} /></div>}
        {valid === false && (
          <>
            <div className="auth-error"><i className="fas fa-link-slash" /> Invalid or expired reset link.</div>
            <p className="auth-link" style={{ marginTop: 20 }}><Link to="/forgot-password">Request a new link</Link></p>
          </>
        )}
        {valid === true && (
          <>
            {error && <div className="auth-error"><i className="fas fa-circle-exclamation" /> {error}</div>}
            <form className="auth-form" onSubmit={onSubmit}>
              <div className="field">
                <label>New password</label>
                <input className="input" type="password" placeholder="At least 8 characters"
                  value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required autoFocus />
              </div>
              <div className="field">
                <label>Confirm password</label>
                <input className="input" type="password" placeholder="Re-enter password"
                  value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
              </div>
              <button className="btn btn-primary" type="submit" disabled={busy} style={{ width: '100%' }}>
                {busy ? <Spinner size={16} color="#fff" /> : <i className="fas fa-key" />}
                {busy ? 'Updating…' : 'Reset password'}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
