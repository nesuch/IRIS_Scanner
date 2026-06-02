import { useOutletContext } from 'react-router-dom';

// Sticky page header with the sidebar toggle, title, optional full-form label,
// scope indicator, and right-aligned actions slot.
export default function PageHeader({ fullForm, title, scope, children }) {
  const ctx = useOutletContext();
  return (
    <header className="page-header">
      <button className="collapse-btn" onClick={ctx?.toggleSidebar} title="Toggle sidebar" aria-label="Toggle sidebar">
        <i className="fas fa-bars" />
      </button>
      <div className="page-header-titles">
        {fullForm && <span className="full-form">{fullForm}</span>}
        <h1>{title}</h1>
        {scope && <span className="scope"><span className="scope-dot" />{scope}</span>}
      </div>
      {children && <div className="header-actions">{children}</div>}
    </header>
  );
}
