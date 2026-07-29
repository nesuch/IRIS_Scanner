import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate, useParams } from 'react-router-dom';
import { ProtectedRoute } from './auth/AuthContext.jsx';
import Layout from './components/Layout.jsx';
import { PageLoading } from './components/UI.jsx';

// Auth pages (eager — small, needed first)
import Login from './pages/Login.jsx';
import ForgotPassword from './pages/ForgotPassword.jsx';
import ResetPassword from './pages/ResetPassword.jsx';
import LogoutConfirm from './pages/LogoutConfirm.jsx';

// Retry a chunk import a few times before giving up. On a cold Cloud Run instance
// (min-instances 0) a burst of chunk requests can transiently get a 429 ("no
// available instance yet"); without a retry, React.lazy's rejected promise
// unmounts the whole tree → blank screen. A couple of spaced retries ride out the
// cold start (the instance is warm within a second or two).
function lazyRetry(factory, retries = 4, delay = 350) {
  return lazy(() => new Promise((resolve, reject) => {
    const attempt = (n) => factory().then(resolve).catch((err) => {
      if (n <= 0) reject(err);
      else setTimeout(() => attempt(n - 1), delay);
    });
    attempt(retries);
  }));
}

// App pages (lazy — code-split per route)
const Search = lazyRetry(() => import('./pages/Search.jsx'));
const DataExplorer = lazyRetry(() => import('./pages/DataExplorer.jsx'));
const Insurer360 = lazyRetry(() => import('./pages/Insurer360.jsx'));
const Analytics = lazyRetry(() => import('./pages/Analytics.jsx'));
const Admin = lazyRetry(() => import('./pages/Admin.jsx'));
const Profile = lazyRetry(() => import('./pages/Profile.jsx'));
const Feedback = lazyRetry(() => import('./pages/Feedback.jsx'));
const Downloads = lazyRetry(() => import('./pages/Downloads.jsx'));
const Pqs = lazyRetry(() => import('./pages/Pqs.jsx'));
const Studio = lazyRetry(() => import('./pages/Studio.jsx'));
const Reader = lazyRetry(() => import('./pages/Reader.jsx'));

// A department module is just Search scoped to a Doc_Category. The slug in the
// URL (/dept/hr) is the lowercased category key the backend filters on.
function DeptSearch() {
  const { dept } = useParams();
  return <Suspense fallback={<PageLoading />}><Search module={(dept || '').toLowerCase()} /></Suspense>;
}

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
        <Route path="nonlife" element={<Suspense fallback={<PageLoading />}><Search module="nonlife" /></Suspense>} />
        <Route path="dept/:dept" element={<DeptSearch />} />
        <Route path="data" element={<Suspense fallback={<PageLoading />}><DataExplorer /></Suspense>} />
        <Route path="insurer" element={<Suspense fallback={<PageLoading />}><Insurer360 /></Suspense>} />
        <Route path="downloads" element={<Suspense fallback={<PageLoading />}><Downloads /></Suspense>} />
        <Route path="read" element={<Suspense fallback={<PageLoading />}><Reader /></Suspense>} />
        <Route path="pqs" element={<ProtectedRoute editorOnly><Suspense fallback={<PageLoading />}><Pqs /></Suspense></ProtectedRoute>} />
        <Route path="studio" element={<Suspense fallback={<PageLoading />}><Studio /></Suspense>} />
        <Route path="analytics" element={<Suspense fallback={<PageLoading />}><Analytics /></Suspense>} />
        <Route path="profile" element={<Suspense fallback={<PageLoading />}><Profile /></Suspense>} />
        <Route path="feedback" element={<Suspense fallback={<PageLoading />}><Feedback /></Suspense>} />
        <Route path="admin" element={<ProtectedRoute adminOnly><Suspense fallback={<PageLoading />}><Admin /></Suspense></ProtectedRoute>} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
