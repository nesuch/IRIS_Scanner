import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';
import { Spinner } from '../components/UI.jsx';

export default function LogoutConfirm() {
  const { user, logout, setUser } = useAuth();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(null); // 'one' | 'all' | null

  async function signOut() {
    setBusy('one');
    await logout();
    navigate('/login', { replace: true });
  }

  async function signOutAll() {
    setBusy('all');
    try { await api.post('/logout-all'); } catch { /* ignore */ }
    setUser(null);
    navigate('/login', { replace: true });
  }

  const devices = user?.device_count || 0;

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
          {user ? <>You are signed in as <strong>{user.email}</strong>.</> : 'You are not signed in.'}
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <button className="btn btn-danger" onClick={signOut} disabled={!!busy}>
            {busy === 'one' ? <Spinner size={16} color="var(--bad)" /> : <i className="fas fa-right-from-bracket" />} Sign out of this device
          </button>
          <button className="btn btn-ghost" onClick={signOutAll} disabled={!!busy || !user}>
            {busy === 'all' ? <Spinner size={16} /> : <i className="fas fa-tower-broadcast" />}
            Sign out of all devices{devices > 1 ? ` (${devices})` : ''}
          </button>
          <button className="btn btn-ghost" onClick={() => navigate(-1)} disabled={!!busy}>Cancel</button>
        </div>
      </div>
    </div>
  );
}
