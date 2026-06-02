import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { api } from '../api.js';

const AuthCtx = createContext(null);
export const useAuth = () => useContext(AuthCtx);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await api.get('/me');
      setUser(data.authenticated ? data.user : null);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Any API call that gets a 401 (expired/invalid session) clears auth, which
  // sends the user to the login screen via ProtectedRoute.
  useEffect(() => {
    const onUnauthorized = () => setUser(null);
    window.addEventListener('iris:unauthorized', onUnauthorized);
    return () => window.removeEventListener('iris:unauthorized', onUnauthorized);
  }, []);

  const login = useCallback(async (email, password, next) => {
    const data = await api.post('/login', { email, password, next });
    setUser(data.user);
    return data;
  }, []);

  const logout = useCallback(async () => {
    try { await api.post('/logout'); } catch { /* ignore */ }
    setUser(null);
  }, []);

  return (
    <AuthCtx.Provider value={{ user, loading, login, logout, refresh, setUser }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function ProtectedRoute({ children, adminOnly = false }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) {
    return (
      <div style={{ display: 'grid', placeItems: 'center', height: '100vh' }}>
        <div className="spin" style={{ fontSize: 28, color: 'var(--azure-600)' }}>
          <i className="fas fa-circle-notch" />
        </div>
      </div>
    );
  }
  if (!user) {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?next=${next}`} replace />;
  }
  if (adminOnly && !user.is_admin) return <Navigate to="/" replace />;
  return children;
}
