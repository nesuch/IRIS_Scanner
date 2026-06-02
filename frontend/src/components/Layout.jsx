import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar.jsx';

// Shared app chrome: collapsible sidebar + sticky header. Pages render a
// <PageHeader> via context-free props by using the exported Header component.
export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const isMobile = () => window.matchMedia('(max-width: 900px)').matches;

  // Close the mobile drawer on resize back to desktop.
  useEffect(() => {
    const onResize = () => { if (!isMobile()) setMobileOpen(false); };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

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
