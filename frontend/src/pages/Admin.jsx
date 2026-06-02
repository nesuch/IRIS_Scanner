import { useEffect, useRef, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { PageLoading } from '../components/UI.jsx';
import { useToast } from '../components/Toast.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';
import './admin/admin.css';

function Section({ icon, title, action, children }) {
  return (
    <div className="admin-card anim-rise">
      <div className="admin-card-head">
        <h3><i className={`fas ${icon}`} /> {title}</h3>
        {action}
      </div>
      {children}
    </div>
  );
}

function SyncCard({ initialState, onLog }) {
  const toast = useToast();
  const [status, setStatus] = useState(initialState?.status || 'idle');
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState([{ t: 'READY', m: 'System initialized. Waiting for command.' }]);
  const pollRef = useRef(null);
  const logEnd = useRef(null);

  const addLog = (m) => setLog((l) => [...l, { t: new Date().toLocaleTimeString('en-US', { hour12: false }), m }]);
  useEffect(() => { logEnd.current?.scrollIntoView({ block: 'nearest' }); }, [log]);
  useEffect(() => () => clearInterval(pollRef.current), []);

  useEffect(() => {
    if (initialState?.status === 'running' || initialState?.status === 'starting') beginPolling();
  }, []); // eslint-disable-line

  function beginPolling() {
    setRunning(true);
    if (pollRef.current) return;
    pollRef.current = setInterval(poll, 1500);
  }

  async function poll() {
    try {
      const data = await api.get('/admin/sync_status');
      setStatus(data.status);
      setLog((l) => (l[l.length - 1]?.m === data.message ? l : [...l, { t: new Date().toLocaleTimeString('en-US', { hour12: false }), m: data.message }]));
      if (data.status === 'running' || data.status === 'starting') {
        setProgress((w) => (w < 90 ? w + 5 : w));
      } else if (data.status === 'complete') {
        clearInterval(pollRef.current); pollRef.current = null;
        setProgress(100); setRunning(false);
        addLog('Job finished at ' + (data.timestamp || 'now'));
        onLog?.();
      } else if (data.status === 'error') {
        clearInterval(pollRef.current); pollRef.current = null;
        setProgress(100); setRunning(false);
      }
    } catch { /* keep polling */ }
  }

  async function start() {
    setRunning(true); setProgress(10); setStatus('starting');
    addLog('Command sent: Start Sync');
    try {
      const data = await api.post('/admin/sync_start');
      if (data.status === 'started') { addLog('Background thread started successfully.'); beginPolling(); }
      else { addLog('Error: ' + data.message); setRunning(false); toast.error(data.message); }
    } catch (e) { addLog('Connection Error: ' + e.message); setRunning(false); }
  }

  const barColor = status === 'complete' ? '#2e7d32' : status === 'error' ? '#c62828' : undefined;

  return (
    <Section icon="fa-database" title="Knowledge Base Sync"
      action={<span className={`status-indicator status-${status}`}>{status}</span>}>
      <p className="admin-help">
        <strong>Instructions:</strong><br />
        1. Financial files → <code>knowledge_base/raw_submissions/</code>.<br />
        2. Regulatory files → <code>knowledge_base/health/</code> or <code>knowledge_base/life/</code>.<br />
        3. Click “Start Data Sync” to sync both datasets in the background.
      </p>
      <button className="btn btn-primary" onClick={start} disabled={running}>
        <i className={`fas ${running ? 'fa-circle-notch spin' : 'fa-rotate'}`} /> {running ? 'Syncing…' : status === 'complete' ? 'Sync Complete' : 'Start Data Sync'}
      </button>
      {(running || progress > 0) && (
        <div className="progress-container"><div className={`progress-bar ${running ? 'active' : ''}`} style={{ width: progress + '%', background: barColor }} /></div>
      )}
      <div className="console-log">
        {log.map((l, i) => <span key={i} className="console-line"><span className="console-ts">[{l.t}]</span> {l.m}</span>)}
        <span ref={logEnd} />
      </div>
    </Section>
  );
}

function CreateUserForm({ onCreated }) {
  const toast = useToast();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.post('/admin/create-user', { email, password });
      toast.success('User created successfully');
      setEmail(''); setPassword(''); onCreated?.();
    } catch (err) { toast.error(err.message || 'Unable to create user.'); }
    finally { setBusy(false); }
  }

  return (
    <form onSubmit={submit}>
      <div className="admin-form-grid">
        <div className="field"><label>Email</label><input className="input" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} /></div>
        <div className="field"><label>Password</label><input className="input" type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} /></div>
      </div>
      <button className="btn btn-primary" type="submit" disabled={busy} style={{ marginTop: 14 }}><i className="fas fa-user-plus" /> Create User</button>
    </form>
  );
}

export default function Admin() {
  const toast = useToast();
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [feedbackFilter, setFeedbackFilter] = useState('ALL');
  const [loading, setLoading] = useState(true);

  const load = (filter = feedbackFilter) => {
    api.get(`/admin/overview?feedback_type=${encodeURIComponent(filter)}`)
      .then(setData).catch((e) => toast.error(e.message)).finally(() => setLoading(false));
  };
  useEffect(() => { load('ALL'); }, []);

  async function userAction(id, action, confirmMsg) {
    if (confirmMsg && !window.confirm(confirmMsg)) return;
    try {
      const res = await api.post(`/admin/user/${id}/${action}`);
      if (action === 'trigger-reset' && res.reset_link) toast.success('Reset link generated');
      else toast.success('Done');
      load();
    } catch (e) { toast.error(e.message || 'Action failed'); }
  }

  async function clearAudit(path, label) {
    if (!window.confirm(`Clear ${label}?`)) return;
    try { await api.post(path); toast.success(`${label} cleared`); load(); }
    catch (e) { toast.error(e.message); }
  }

  if (loading) return (<><PageHeader fullForm="System Administration" title="Admin Console" scope="Manage Data & Configuration" /><div className="page-body"><PageLoading /></div></>);

  return (
    <>
      <PageHeader fullForm="System Administration" title="Admin Console" scope="Manage Data & Configuration" />
      <div className="page-body admin-body">
        <SyncCard initialState={data.sync_state} onLog={load} />

        <Section icon="fa-users-gear" title="User Management">
          <p className="admin-help">Create user accounts and manage access. Your active devices: <strong>{user?.device_count}</strong></p>
          <CreateUserForm onCreated={load} />
          <div className="table-wrap" style={{ marginTop: 20 }}>
            <div className="table-scroll">
              <table className="data">
                <thead><tr><th>Email</th><th>Active</th><th>Admin</th><th>Devices</th><th>Created</th><th>Actions</th></tr></thead>
                <tbody>
                  {data.users.length ? data.users.map((u) => (
                    <tr key={u.id}>
                      <td>{u.email}</td>
                      <td>{u.is_active ? 'Yes' : 'No'}</td>
                      <td>{u.is_admin ? 'Yes' : 'No'}</td>
                      <td>{u.device_count}</td>
                      <td>{u.created_at || '-'}</td>
                      <td>
                        <div className="row-actions">
                          <button className="btn btn-ghost btn-sm" onClick={() => userAction(u.id, 'toggle-active')}>{u.is_active ? 'Deactivate' : 'Activate'}</button>
                          <button className="btn btn-ghost btn-sm" onClick={() => userAction(u.id, 'trigger-reset')}>Trigger Reset</button>
                          <button className="btn btn-ghost btn-sm" onClick={() => userAction(u.id, 'logout-all')}>Logout All</button>
                          {u.id !== undefined && u.email !== user?.email && (
                            <button className="btn btn-danger btn-sm" onClick={() => userAction(u.id, 'delete', `Delete ${u.email}?`)}>Delete</button>
                          )}
                        </div>
                      </td>
                    </tr>
                  )) : <tr><td colSpan={6} className="muted-cell">No users available.</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
        </Section>

        <Section icon="fa-user-clock" title="Usage Insights">
          <p className="admin-help">Logged-in user behavior across modules.</p>
          <div className="table-wrap"><div className="table-scroll">
            <table className="data">
              <thead><tr><th>User</th><th>Requests</th><th>Est. Time (min)</th><th>Top Module</th><th>Last Seen</th></tr></thead>
              <tbody>
                {data.usage_insights.users?.length ? data.usage_insights.users.map((r, i) => (
                  <tr key={i}><td>{r.email}</td><td>{r.total_requests}</td><td>{r.estimated_minutes}</td><td>{r.top_module}</td><td>{r.last_seen}</td></tr>
                )) : <tr><td colSpan={5} className="muted-cell">No module usage tracked yet.</td></tr>}
              </tbody>
            </table>
          </div></div>
        </Section>

        <Section icon="fa-key" title="Password Reset Activity"
          action={<button className="btn btn-ghost btn-sm" onClick={() => clearAudit('/admin/clear-reset-audit', 'Reset Logs')}>Clear Reset Logs</button>}>
          <div className="table-wrap log-scroll"><div className="table-scroll">
            <table className="data">
              <thead><tr><th>Email</th><th>Requested</th><th>Expires</th><th>Reset Link</th></tr></thead>
              <tbody>
                {data.reset_audit.length ? data.reset_audit.map((r, i) => (
                  <tr key={i}><td>{r.email}</td><td>{r.requested_at}</td><td>{r.expires_at}</td><td><a className="reset-link" href={r.reset_link} target="_blank" rel="noreferrer">{r.reset_link}</a></td></tr>
                )) : <tr><td colSpan={4} className="muted-cell">No password reset requests found.</td></tr>}
              </tbody>
            </table>
          </div></div>
        </Section>

        <Section icon="fa-clipboard-list" title="Audit Logs"
          action={<button className="btn btn-ghost btn-sm" onClick={() => clearAudit('/admin/clear-audit-logs', 'Audit Logs')}>Clear Audit Logs</button>}>
          <div className="table-wrap log-scroll"><div className="table-scroll">
            <table className="data">
              <thead><tr><th>Email</th><th>Action</th><th>Status</th><th>Timestamp</th></tr></thead>
              <tbody>
                {data.audit_logs.length ? data.audit_logs.map((r, i) => (
                  <tr key={i}><td>{r.email || '-'}</td><td>{r.action_type}</td><td>{r.status}</td><td>{r.timestamp}</td></tr>
                )) : <tr><td colSpan={4} className="muted-cell">No audit logs found.</td></tr>}
              </tbody>
            </table>
          </div></div>
        </Section>

        <Section icon="fa-comment-dots" title="User Feedback"
          action={(
            <select className="select" style={{ width: 'auto' }} value={feedbackFilter}
              onChange={(e) => { setFeedbackFilter(e.target.value); setLoading(false); load(e.target.value); }}>
              <option value="ALL">All</option><option value="Bug">Bug</option><option value="Suggestion">Suggestion</option>
              <option value="UI Issue">UI Issue</option><option value="Other (please specify)">Other</option>
            </select>
          )}>
          <div className="table-wrap log-scroll"><div className="table-scroll">
            <table className="data">
              <thead><tr><th>Time</th><th>User</th><th>Type</th><th>Message</th></tr></thead>
              <tbody>
                {data.feedback_rows.length ? data.feedback_rows.map((r, i) => (
                  <tr key={i}><td>{r.created_at || '-'}</td><td>{r.user_email}</td><td>{r.category}</td><td style={{ whiteSpace: 'normal', minWidth: 320 }}>{r.message}</td></tr>
                )) : <tr><td colSpan={4} className="muted-cell">No feedback submissions found.</td></tr>}
              </tbody>
            </table>
          </div></div>
        </Section>
      </div>
    </>
  );
}
