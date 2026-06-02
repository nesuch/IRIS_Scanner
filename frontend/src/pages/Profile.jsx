import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import { PageLoading } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';

export default function Profile() {
  const toast = useToast();
  const navigate = useNavigate();
  const { setUser } = useAuth();
  const [profile, setProfile] = useState(null);
  const [pw, setPw] = useState({ current_password: '', new_password: '', confirm_password: '' });
  const [busy, setBusy] = useState(false);

  const load = () => api.get('/profile').then(setProfile).catch((e) => toast.error(e.message));
  useEffect(() => { load(); }, []);

  async function changePassword(e) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.post('/profile/password', pw);
      toast.success('Password changed. Please sign in again.');
      setUser(null);
      navigate('/login', { replace: true });
    } catch (err) {
      toast.error(err.message || 'Could not change password.');
    } finally { setBusy(false); }
  }

  async function killSession(id) {
    try {
      const res = await api.post(`/profile/session/${id}/logout`);
      if (res.logged_out) { setUser(null); navigate('/login', { replace: true }); return; }
      toast.success('Device signed out'); load();
    } catch (err) { toast.error(err.message); }
  }

  if (!profile) return (<><PageHeader fullForm="User Profile" title="Account Settings" scope="Password and device management" /><div className="page-body"><PageLoading /></div></>);

  return (
    <>
      <PageHeader fullForm="User Profile" title="Account Settings" scope="Password and device management" />
      <div className="page-body" style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 980 }}>
        <div className="card pad anim-rise">
          <h3 className="section-title"><i className="fas fa-lock" /> Change Password</h3>
          <form onSubmit={changePassword}>
            <div className="admin-form-grid">
              <div className="field"><label>Current password</label><input className="input" type="password" required value={pw.current_password} onChange={(e) => setPw({ ...pw, current_password: e.target.value })} /></div>
              <div className="field"><label>New password</label><input className="input" type="password" required minLength={8} value={pw.new_password} onChange={(e) => setPw({ ...pw, new_password: e.target.value })} /></div>
              <div className="field"><label>Confirm new password</label><input className="input" type="password" required minLength={8} value={pw.confirm_password} onChange={(e) => setPw({ ...pw, confirm_password: e.target.value })} /></div>
            </div>
            <button className="btn btn-primary" type="submit" disabled={busy} style={{ marginTop: 16 }}><i className="fas fa-key" /> Update Password</button>
          </form>
        </div>

        <div className="card pad anim-rise">
          <h3 className="section-title"><i className="fas fa-laptop" /> Active Devices</h3>
          <div className="table-wrap"><div className="table-scroll">
            <table className="data">
              <thead><tr><th>IP</th><th>User Agent</th><th>Login Time</th><th>Action</th></tr></thead>
              <tbody>
                {profile.sessions?.length ? profile.sessions.map((s) => (
                  <tr key={s.id}>
                    <td>{s.ip || '-'}</td>
                    <td style={{ maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis' }} title={s.user_agent}>{s.user_agent || '-'}</td>
                    <td>{s.created_at}</td>
                    <td><button className="btn btn-ghost btn-sm" onClick={() => killSession(s.id)}>Logout</button></td>
                  </tr>
                )) : <tr><td colSpan={4} style={{ color: 'var(--faint)', textAlign: 'center' }}>No active devices found.</td></tr>}
              </tbody>
            </table>
          </div></div>
        </div>
      </div>
    </>
  );
}
