import { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';
import { Spinner } from '../components/UI.jsx';

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const next = params.get('next') || '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const data = await login(email, password, next);
      navigate(data.next || next, { replace: true });
    } catch (err) {
      setError(err.message || 'Invalid credentials.');
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
          <span className="wm-sub">IRDAI&rsquo;s Regulatory<br />Intelligence System</span>
        </div>

        {error && <div className="auth-error"><i className="fas fa-circle-exclamation" /> {error}</div>}

        <form className="auth-form" onSubmit={onSubmit}>
          <div className="field">
            <label>Official Email</label>
            <input className="input" type="email" autoComplete="username" placeholder="name@irdai.gov.in"
              value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
          </div>
          <div className="field">
            <label>Password</label>
            <input className="input" type="password" autoComplete="current-password" placeholder="••••••••"
              value={password} onChange={(e) => setPassword(e.target.value)} required />
          </div>
          <button className="btn btn-primary" type="submit" disabled={busy} style={{ width: '100%', marginTop: 4 }}>
            {busy ? <Spinner size={16} color="#fff" /> : <i className="fas fa-arrow-right-to-bracket" />}
            {busy ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <p className="auth-link" style={{ marginTop: 20 }}>
          <Link to="/forgot-password">Forgot password?</Link>
        </p>
      </div>
    </div>
  );
}
