import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { api } from '../api.js';

const AuthCtx = createContext(null);
export const useAuth = () => useContext(AuthCtx);

// The Search page persists each module's conversation in sessionStorage so results
// survive navigating to the Reader and back. That must NOT bleed across auth
// boundaries — clear it on login/logout/expiry so a fresh session never shows the
// previous user's (or the previous session's) query.
function clearSearchSessions() {
  try {
    Object.keys(sessionStorage)
      .filter((k) => k.startsWith('iris_search_session_'))
      .forEach((k) => sessionStorage.removeItem(k));
  } catch { /* sessionStorage unavailable — non-fatal */ }
}

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
    const onUnauthorized = () => { clearSearchSessions(); setUser(null); };
    window.addEventListener('iris:unauthorized', onUnauthorized);
    return () => window.removeEventListener('iris:unauthorized', onUnauthorized);
  }, []);

  const login = useCallback(async (email, password, next) => {
    const data = await api.post('/login', { email, password, next });
    clearSearchSessions();   // fresh session — don't inherit a stale query screen
    setUser(data.user);
    return data;
  }, []);

  const logout = useCallback(async () => {
    try { await api.post('/logout'); } catch { /* ignore */ }
    clearSearchSessions();
    setUser(null);
  }, []);

  return (
    <AuthCtx.Provider value={{ user, loading, login, logout, refresh, setUser }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function ProtectedRoute({ children, adminOnly = false, editorOnly = false }) {
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
  // Editor gate. Admins outrank editors, so they always pass. This only hides the
  // route — the API enforces the same rule independently, since a client-side
  // redirect is a convenience, never a security boundary.
  if (editorOnly && !(user.is_admin || user.role === 'admin' || user.role === 'editor')) {
    return <Navigate to="/" replace />;
  }
  return children;
}
