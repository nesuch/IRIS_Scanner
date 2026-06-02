import { useEffect, useRef, useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { useToast } from './Toast.jsx';

// Inactivity window before automatic sign-out (minutes). Overridable for testing.
const IDLE_MINUTES = 30;
const IDLE_MS = IDLE_MINUTES * 60 * 1000;

// Shared app chrome: collapsible sidebar + sticky header. Pages render a
// <PageHeader> via context-free props by using the exported Header component.
export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { logout } = useAuth();
  const toast = useToast();
  const idleRef = useRef(null);
  const isMobile = () => window.matchMedia('(max-width: 900px)').matches;

  // Close the mobile drawer on resize back to desktop.
  useEffect(() => {
    const onResize = () => { if (!isMobile()) setMobileOpen(false); };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Auto sign-out after a period of inactivity (security for shared terminals).
  useEffect(() => {
    const reset = () => {
      clearTimeout(idleRef.current);
      idleRef.current = setTimeout(async () => {
        await logout();
        toast.error('Signed out due to inactivity.');
      }, IDLE_MS);
    };
    const events = ['mousemove', 'mousedown', 'keydown', 'scroll', 'touchstart', 'wheel'];
    events.forEach((e) => window.addEventListener(e, reset, { passive: true }));
    reset();
    return () => {
      clearTimeout(idleRef.current);
      events.forEach((e) => window.removeEventListener(e, reset));
    };
  }, [logout, toast]);

  const toggle = () => { if (isMobile()) setMobileOpen((o) => !o); else setCollapsed((c) => !c); };

  return (
    <div className={`app-shell ${collapsed ? 'collapsed' : ''}`}>
      <Sidebar open={mobileOpen} onNavigate={() => setMobileOpen(false)} />
      {mobileOpen && <div className="scrim" onClick={() => setMobileOpen(false)} />}
      <main className="main">
        <Outlet context={{ toggleSidebar: toggle }} />
      </main>
    </div>
  );
}
