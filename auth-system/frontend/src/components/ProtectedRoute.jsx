import { Navigate } from 'react-router-dom';

export default function ProtectedRoute({ user, allowedRoles, children }) {
  if (!user) return <Navigate to="/login" replace />;
  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return <div style={{ padding: 20 }}>Forbidden: insufficient role</div>;
  }
  return children;
}
