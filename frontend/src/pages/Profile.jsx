import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import { PageLoading, Spinner } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';
import AvatarCropper from './profile/AvatarCropper.jsx';
import './profile/profile.css';

function Avatar({ src, name, email, size = 64 }) {
  const initials = (name || email || '?').trim().slice(0, 1).toUpperCase();
  return src
    ? <img className="avatar" src={src} alt="avatar" style={{ width: size, height: size }} />
    : <span className="avatar avatar-fallback" style={{ width: size, height: size, fontSize: size * 0.4 }}>{initials}</span>;
}

// User-facing status: "Ignored" reads as "No action required".
const STATUS = {
  Open: { label: 'Pending', cls: 'badge-warn' },
  Resolved: { label: 'Resolved', cls: 'badge-good' },
  Ignored: { label: 'No action required', cls: 'badge-grey' },
};
const stView = (s) => STATUS[s] || STATUS.Open;

export default function Profile() {
  const toast = useToast();
  const navigate = useNavigate();
  const { setUser, refresh } = useAuth();
  const [profile, setProfile] = useState(null);
  const [name, setName] = useState('');
  const [avatar, setAvatar] = useState(null);
  const [pw, setPw] = useState({ current_password: '', new_password: '', confirm_password: '' });
  const [busy, setBusy] = useState(false);
  const [savingId, setSavingId] = useState(false);
  const [cropFile, setCropFile] = useState(null);
  const fileRef = useRef(null);

  const load = () => api.get('/profile').then((p) => {
    setProfile(p); setName(p.display_name || ''); setAvatar(p.avatar || null);
  }).catch((e) => toast.error(e.message));
  useEffect(() => { load(); }, []);

  function pickPhoto(e) {
    const file = e.target.files?.[0];
    if (file) setCropFile(file);   // open the cropper to position/zoom
    e.target.value = '';            // allow re-picking the same file
  }

  function onCropped(dataUrl) {
    setCropFile(null);
    if (dataUrl.length > 240000) { toast.error('Image too large — try zooming out or a smaller photo.'); return; }
    setAvatar(dataUrl);
  }

  async function saveIdentity(e) {
    e.preventDefault();
    setSavingId(true);
    try {
      const res = await api.post('/profile', { display_name: name, avatar: avatar || '' });
      if (res.user) setUser(res.user); else refresh();
      toast.success('Profile updated');
    } catch (err) { toast.error(err.message || 'Could not save profile.'); }
    finally { setSavingId(false); }
  }

  async function changePassword(e) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.post('/profile/password', pw);
      toast.success('Password changed. Please sign in again.');
      setUser(null);
      navigate('/login', { replace: true });
    } catch (err) { toast.error(err.message || 'Could not change password.'); }
    finally { setBusy(false); }
  }

  async function killSession(id) {
    try {
      const res = await api.post(`/profile/session/${id}/logout`);
      if (res.logged_out) { setUser(null); navigate('/login', { replace: true }); return; }
      toast.success('Device signed out'); load();
    } catch (err) { toast.error(err.message); }
  }

  async function addComment(kind, id, value) {
    if (!value.trim()) return;
    try { await api.post(`/${kind}/${id}/comment`, { body: value.trim() }); load(); }
    catch (err) { toast.error(err.message || 'Could not post comment'); }
  }

  async function deleteFeedback(id) {
    if (!window.confirm('Delete this feedback? This cannot be undone.')) return;
    try { await api.post(`/feedback/${id}/delete`); toast.success('Feedback deleted'); load(); }
    catch (err) { toast.error(err.message || 'Could not delete'); }
  }

  async function deleteFlag(id) {
    if (!window.confirm('Retract this flag?')) return;
    try { await api.post(`/flag/${id}/delete`); toast.success('Flag retracted'); load(); }
    catch (err) { toast.error(err.message || 'Could not retract'); }
  }

  if (!profile) return (<><PageHeader fullForm="User Profile" title="Account Settings" scope="Your profile, devices and feedback" /><div className="page-body"><PageLoading /></div></>);

  // Feedback + flags merged into one chronological list.
  const reports = [
    ...(profile.feedback || []).map((f) => ({ ...f, _type: 'feedback' })),
    ...(profile.flags || []).map((f) => ({ ...f, _type: 'flag' })),
  ].sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')));

  return (
    <>
      <PageHeader fullForm="User Profile" title="Account Settings" scope="Your profile, devices and feedback" />
      <div className="page-body" style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 980 }}>
        {/* Identity */}
        <div className="card pad anim-rise">
          <h3 className="section-title"><i className="fas fa-id-badge" /> Your Profile</h3>
          {name && <div className="profile-greeting">Signed in as <strong>{name}</strong></div>}
          <form onSubmit={saveIdentity} className="profile-identity">
            <div className="profile-photo">
              <Avatar src={avatar} name={name} email={profile.email} size={96} />
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={pickPhoto} />
              <div className="profile-photo-actions">
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => fileRef.current?.click()}><i className="fas fa-camera" /> Change photo</button>
                {avatar && <button type="button" className="btn btn-ghost btn-sm" onClick={() => setAvatar(null)}>Remove</button>}
              </div>
            </div>
            <div className="profile-fields">
              <div className="field"><label>Display name</label><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" maxLength={120} /></div>
              <div className="field"><label>Email</label><input className="input" value={profile.email} disabled /></div>
              <button className="btn btn-primary" type="submit" disabled={savingId} style={{ marginTop: 4, alignSelf: 'flex-start' }}>
                {savingId ? <Spinner size={14} color="#fff" /> : <i className="fas fa-floppy-disk" />} Save Profile
              </button>
            </div>
          </form>
        </div>

        {/* Change password */}
        <div className="card pad anim-rise">
          <h3 className="section-title"><i className="fas fa-lock" /> Change Password</h3>
          <form onSubmit={changePassword} className="pw-form">
            <div className="admin-form-grid">
              <div className="field"><label>Current password</label><input className="input" type="password" required value={pw.current_password} onChange={(e) => setPw({ ...pw, current_password: e.target.value })} /></div>
              <div className="field"><label>New password</label><input className="input" type="password" required minLength={8} value={pw.new_password} onChange={(e) => setPw({ ...pw, new_password: e.target.value })} /></div>
              <div className="field"><label>Confirm new password</label><input className="input" type="password" required minLength={8} value={pw.confirm_password} onChange={(e) => setPw({ ...pw, confirm_password: e.target.value })} /></div>
            </div>
            <button className="btn btn-primary" type="submit" disabled={busy} style={{ marginTop: 16 }}><i className="fas fa-key" /> Update Password</button>
          </form>
        </div>

        {/* My reports & flags (feedback + flags in one place) */}
        <div className="card pad anim-rise">
          <h3 className="section-title"><i className="fas fa-comment-dots" /> My Reports &amp; Flags</h3>
          {reports.length ? (
            <div className="fb-list">
              {reports.map((r) => {
                const sv = stView(r.status);
                const isFb = r._type === 'feedback';
                return (
                  <div key={`${r._type}${r.id}`} className="fb-card">
                    <div className="fb-card-head">
                      <span className={`badge ${isFb ? 'badge-blue' : 'badge-bad'}`}>{isFb ? 'Feedback' : 'Flag'}</span>
                      <span className="badge badge-navy">{isFb ? r.category : (r.kind === 'financial' ? 'Financial' : 'Clause')}</span>
                      {!isFb && <span className="badge badge-grey">{r.reason}</span>}
                      <span className={`badge ${sv.cls}`}>{sv.label}</span>
                      <span className="fb-date">{r.created_at}</span>
                      <button className="fb-del" title={isFb ? 'Delete feedback' : 'Retract flag'}
                        onClick={() => (isFb ? deleteFeedback(r.id) : deleteFlag(r.id))}><i className="fas fa-trash-can" /></button>
                    </div>
                    {!isFb && r.target && <div className="fb-target"><i className="fas fa-location-dot" /> {r.target}</div>}
                    {(isFb ? r.message : r.description) && <div className="fb-msg">{isFb ? r.message : r.description}</div>}
                    {r.comments?.length > 0 && (
                      <div className="fb-thread">
                        {r.comments.map((c, i) => (
                          <div key={i} className={`fb-comment ${c.is_admin && c.author_email !== profile.email ? 'admin' : ''}`}>
                            <span className="fb-author">{c.author_email === profile.email ? 'You' : (c.is_admin ? 'IRIS Team' : c.author_email)}</span> {c.body}
                            <span className="fb-cdate">{c.created_at}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    <input className="input fb-reply" placeholder="Add a follow-up…"
                      onKeyDown={(e) => { if (e.key === 'Enter') { addComment(isFb ? 'feedback' : 'flag', r.id, e.target.value); e.target.value = ''; } }} />
                  </div>
                );
              })}
            </div>
          ) : <p style={{ color: 'var(--faint)' }}>You haven&rsquo;t submitted any feedback or flags yet.</p>}
        </div>

        {/* Active devices */}
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
      {cropFile && <AvatarCropper file={cropFile} onCancel={() => setCropFile(null)} onDone={onCropped} />}
    </>
  );
}
