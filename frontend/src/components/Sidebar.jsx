import { NavLink } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';

const SECTIONS = [
  {
    label: 'Knowledge Base',
    items: [
      { to: '/', icon: 'fa-magnifying-glass', text: 'Universal Search', end: true },
      { to: '/health', icon: 'fa-heart-pulse', text: 'Health Dept' },
      { to: '/life', icon: 'fa-umbrella', text: 'Life Dept' },
      { to: '/nonlife', icon: 'fa-shield-halved', text: 'Non-Life Dept' },
    ],
  },
  {
    label: 'Data Intelligence',
    items: [
      { to: '/data', icon: 'fa-chart-line', text: 'Data Explorer' },
      { to: '/compliance', icon: 'fa-gavel', text: 'Compliance Cockpit' },
      { to: '/downloads', icon: 'fa-folder-tree', text: 'Downloads' },
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

export default function Sidebar({ open, onNavigate }) {
  const { user } = useAuth();
  return (
    <aside className={`sidebar ${open ? 'open' : ''}`}>
      <div className="sidebar-brand">
        <img className="sidebar-logo" src="/static/iris_logo.png" alt="IRIS" />
        <span className="wm">IRIS</span>
      </div>
      <div className="sidebar-tagline">Exact regulation.<br />Zero hallucination.</div>

      {SECTIONS.map((sec) => (
        <div key={sec.label}>
          <div className="nav-label">{sec.label}</div>
          {sec.items
            .filter((it) => !it.adminOnly || (user && user.is_admin))
            .map((it) => (
              <NavLink
                key={it.to}
                to={it.to}
                end={it.end}
                onClick={onNavigate}
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              >
                <i className={`fas ${it.icon}`} />
                <span>{it.text}</span>
              </NavLink>
            ))}
        </div>
      ))}

      <div className="nav-spacer" />
      <NavLink to="/logout" onClick={onNavigate} className="nav-item">
        <i className="fas fa-right-from-bracket" />
        <span>Logout</span>
        {user && <span className="nav-badge">{user.device_count}</span>}
      </NavLink>
    </aside>
  );
}
