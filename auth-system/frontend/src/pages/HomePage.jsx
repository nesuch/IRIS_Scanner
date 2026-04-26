import { api } from '../api/client';

export default function HomePage({ user, setUser }) {
  const callRoute = async (path) => {
    try {
      const { data } = await api.get(path);
      alert(`${path}: ${data.message}`);
    } catch (err) {
      alert(`${path}: ${err?.response?.data?.message || 'request failed'}`);
    }
  };

  const logout = async () => {
    await api.post('/auth/logout');
    setUser(null);
  };

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 20 }}>
      <h2>Welcome {user?.email}</h2>
      <p>Role: <strong>{user?.role}</strong></p>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button onClick={() => callRoute('/dashboard')}>Dashboard</button>
        <button onClick={() => callRoute('/admin')}>Admin Console</button>
        <button onClick={() => callRoute('/system-analytics')}>System Analytics</button>
        <button onClick={() => callRoute('/data-explorer')}>Data Explorer</button>
        <button onClick={() => callRoute('/compliance-cockpit')}>Compliance Cockpit</button>
        <button onClick={() => callRoute('/universal-module')}>Universal Module</button>
      </div>

      <button onClick={logout} style={{ marginTop: 20 }}>Logout</button>
    </div>
  );
}
