import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';
import { api } from '../api.js';

const STORE_KEY = 'iris.sidebar.collapsed';

// Department modules are admin-managed, so the Knowledge Base section is built
// at render time from the fetched list (with sensible defaults as a fallback).
const DEFAULT_DEPTS = [
  { key: 'HEALTH', label: 'Health', icon: 'fa-heart-pulse' },
  { key: 'LIFE', label: 'Life', icon: 'fa-umbrella' },
  { key: 'NONLIFE', label: 'Non-Life', icon: 'fa-shield-halved' },
];

function buildSections(depts) {
  return [
  {
    label: 'Knowledge Base',
    items: [
      { to: '/', icon: 'fa-magnifying-glass', text: 'Universal Search', end: true },
      ...depts.map((d) => ({ to: `/dept/${d.key.toLowerCase()}`, icon: d.icon || 'fa-folder', text: `${d.label} Dept` })),
      { to: '/pqs', icon: 'fa-landmark', text: 'Parliamentary Q&A' },
      { to: '/read', icon: 'fa-book-open', text: 'Read Documents' },
    ],
  },
  {
    label: 'Data Intelligence',
    items: [
      { to: '/data', icon: 'fa-chart-line', text: 'Data Explorer' },
      { to: '/compliance', icon: 'fa-gavel', text: 'Compliance Cockpit' },
      { to: '/downloads', icon: 'fa-folder-tree', text: 'Downloads' },
      { to: '/studio', icon: 'fa-pen-ruler', text: 'Document Studio', editorOnly: true },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/analytics', icon: 'fa-server', text: 'System Analytics' },
      { to: '/admin', icon: 'fa-gear', text: 'Admin Panel', adminOnly: true },
      { to: '/profile', icon: 'fa-circle-user', text: 'Profile' },
      { to: '/feedback', icon: 'fa-comment-dots', text: 'Feedback' },
    ],
  },
  ];
}

export default function Sidebar({ open, onNavigate }) {
  const { user } = useAuth();
  const location = useLocation();
  const isAdmin = user?.role === 'admin' || user?.is_admin;
  const isEditor = isAdmin || user?.role === 'editor';
  const [depts, setDepts] = useState(DEFAULT_DEPTS);
  useEffect(() => {
    const load = () => api.get('/departments')
      .then((d) => { if (d.departments?.length) setDepts(d.departments); })
      .catch(() => {});
    load();
    window.addEventListener('iris:departments-changed', load);
    return () => window.removeEventListener('iris:departments-changed', load);
  }, []);
  const SECTIONS = buildSections(depts);
  // Clicking the module you're already in resets that page (clears the chat).
  const handleNav = (to) => {
    if (location.pathname === to) {
      window.dispatchEvent(new CustomEvent('iris:reclick', { detail: to }));
    }
    onNavigate?.();
  };
  const [collapsed, setCollapsed] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
  });
  const toggle = (label) => setCollapsed((c) => {
    const next = { ...c, [label]: !c[label] };
    try { localStorage.setItem(STORE_KEY, JSON.stringify(next)); } catch { /* ignore */ }
    return next;
  });
  return (
    <aside className={`sidebar ${open ? 'open' : ''}`}>
      <div className="sidebar-brand">
        <img className="sidebar-logo" src="/static/iris_logo.png" alt="IRIS" />
        <span className="wm">IRIS</span>
      </div>
      <div className="sidebar-tagline">Exact regulation.<br />Zero hallucination.</div>

      {SECTIONS.map((sec) => {
        const isCol = !!collapsed[sec.label];
        return (
          <div className={`nav-group ${isCol ? 'is-collapsed' : ''}`} key={sec.label}>
            <button type="button" className="nav-label nav-label-btn" onClick={() => toggle(sec.label)}>
              <span>{sec.label}</span>
              <i className="fas fa-chevron-down nav-chevron" />
            </button>
            {!isCol && sec.items
              .filter((it) => (!it.adminOnly || isAdmin) && (!it.editorOnly || isEditor))
              .map((it) => (
                <NavLink
                  key={it.to}
                  to={it.to}
                  end={it.end}
                  onClick={() => handleNav(it.to)}
                  className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                >
                  <i className={`fas ${it.icon}`} />
                  <span>{it.text}</span>
                </NavLink>
              ))}
          </div>
        );
      })}

      <div className="nav-spacer" />
      <NavLink to="/logout" onClick={onNavigate} className="nav-item">
        <i className="fas fa-right-from-bracket" />
        <span>Logout</span>
        {user && <span className="nav-badge">{user.device_count}</span>}
      </NavLink>
    </aside>
  );
}
