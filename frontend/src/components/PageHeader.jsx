import { useOutletContext, Link } from 'react-router-dom';
import Notifications from './Notifications.jsx';
import ZoomControl from './ZoomControl.jsx';
import { useAuth } from '../auth/AuthContext.jsx';

// Sticky page header with the sidebar toggle, title, optional full-form label,
// scope indicator, the notifications bell, the user avatar, and an actions slot.
// `showZoom` adds the clause text-size control to the icon cluster — only on the
// pages that actually render clause text (Search, Reader), since it's a no-op
// elsewhere.
export default function PageHeader({ fullForm, title, scope, scopeDot = true, showZoom = false, children }) {
  const ctx = useOutletContext();
  const { user } = useAuth();
  const initials = (user?.display_name || user?.email || '?').trim().slice(0, 1).toUpperCase();
  return (
    <header className="page-header">
      <button className="collapse-btn" onClick={ctx?.toggleSidebar} title="Toggle sidebar" aria-label="Toggle sidebar">
        <i className="fas fa-bars" />
      </button>
      <div className="page-header-titles">
        {fullForm && <span className="full-form">{fullForm}</span>}
        <h1>{title}</h1>
        {scope && <span className="scope">{scopeDot && <span className="scope-dot" />}<span className="scope-text">{scope}</span></span>}
      </div>
      <div className="header-actions">
        {children && <div className="header-actions-scroll">{children}</div>}
        {showZoom && <ZoomControl className="header-zoom" />}
        <Link to="/downloads" className="header-icon-btn" title="Downloads — Acts, Regulations & Circulars" aria-label="Downloads">
          <i className="fas fa-folder-tree" />
        </Link>
        <Notifications />
        <Link to="/profile" className="header-avatar" title={user?.display_name || user?.email || 'Profile'}>
          {user?.avatar
            ? <img src={user.avatar} alt="profile" />
            : <span className="header-avatar-fallback">{initials}</span>}
        </Link>
      </div>
    </header>
  );
}
