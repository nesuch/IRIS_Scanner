import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ProtectedRoute } from './auth/AuthContext.jsx';
import Layout from './components/Layout.jsx';
import { PageLoading } from './components/UI.jsx';

// Auth pages (eager — small, needed first)
import Login from './pages/Login.jsx';
import ForgotPassword from './pages/ForgotPassword.jsx';
import ResetPassword from './pages/ResetPassword.jsx';
import LogoutConfirm from './pages/LogoutConfirm.jsx';

// App pages (lazy — code-split per route)
const Search = lazy(() => import('./pages/Search.jsx'));
const DataExplorer = lazy(() => import('./pages/DataExplorer.jsx'));
const Compliance = lazy(() => import('./pages/Compliance.jsx'));
const Analytics = lazy(() => import('./pages/Analytics.jsx'));
const Admin = lazy(() => import('./pages/Admin.jsx'));
const Profile = lazy(() => import('./pages/Profile.jsx'));
const Feedback = lazy(() => import('./pages/Feedback.jsx'));

export default function App() {
  return (
    <Routes>
      {/* Public auth routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/reset-password/:token" element={<ResetPassword />} />
      <Route path="/logout" element={<LogoutConfirm />} />

      {/* Protected app shell */}
      <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<Suspense fallback={<PageLoading />}><Search module="universal" /></Suspense>} />
        <Route path="health" element={<Suspense fallback={<PageLoading />}><Search module="health" /></Suspense>} />
        <Route path="life" element={<Suspense fallback={<PageLoading />}><Search module="life" /></Suspense>} />
        <Route path="data" element={<Suspense fallback={<PageLoading />}><DataExplorer /></Suspense>} />
        <Route path="compliance" element={<Suspense fallback={<PageLoading />}><Compliance /></Suspense>} />
        <Route path="analytics" element={<Suspense fallback={<PageLoading />}><Analytics /></Suspense>} />
        <Route path="profile" element={<Suspense fallback={<PageLoading />}><Profile /></Suspense>} />
        <Route path="feedback" element={<Suspense fallback={<PageLoading />}><Feedback /></Suspense>} />
        <Route path="admin" element={<ProtectedRoute adminOnly><Suspense fallback={<PageLoading />}><Admin /></Suspense></ProtectedRoute>} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
