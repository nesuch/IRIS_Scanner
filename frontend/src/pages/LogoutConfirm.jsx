import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';
import { Spinner } from '../components/UI.jsx';

export default function LogoutConfirm() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);

  async function confirm() {
    setBusy(true);
    await logout();
    navigate('/login', { replace: true });
  }

  return (
    <div className="auth-wrap">
      <div className="auth-bg-glow g1" />
      <div className="auth-bg-glow g2" />
      <div className="auth-card" style={{ textAlign: 'center' }}>
        <div className="auth-brand">
          <img src="/static/iris_logo.png" alt="IRIS" />
          <span className="wm grad-text">Sign out</span>
        </div>
        <p style={{ color: 'var(--ink-soft)', marginBottom: 24 }}>
          {user ? <>You are signed in as <strong>{user.email}</strong>. End this session?</> : 'You are not signed in.'}
        </p>
        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => navigate(-1)} disabled={busy}>Cancel</button>
          <button className="btn btn-danger" style={{ flex: 1 }} onClick={confirm} disabled={busy}>
            {busy ? <Spinner size={16} color="var(--bad)" /> : <i className="fas fa-right-from-bracket" />} Sign out
          </button>
        </div>
      </div>
    </div>
  );
}
