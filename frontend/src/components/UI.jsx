// Small reusable design-system primitives.

export function Spinner({ size = 18, color = 'var(--azure-600)' }) {
  return <span className="spin" style={{ fontSize: size, color }}><i className="fas fa-circle-notch" /></span>;
}

export function PageLoading({ label = 'Loading…' }) {
  return (
    <div className="empty-state anim-fade">
      <Spinner size={34} />
      <p style={{ marginTop: 16 }}>{label}</p>
    </div>
  );
}

export function EmptyState({ icon = 'fa-inbox', children }) {
  return (
    <div className="empty-state anim-fade">
      <i className={`fas ${icon}`} />
      <p>{children}</p>
    </div>
  );
}

export function StatCard({ label, value, icon, tone }) {
  return (
    <div className={`stat-card anim-rise ${tone || ''}`}>
      {icon && <i className={`stat-icon fas ${icon}`} />}
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

export function Badge({ tone = 'blue', children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function TypingDots() {
  return <span className="typing"><span /><span /><span /></span>;
}

export function Modal({ title, onClose, children, footer, width }) {
  return (
    <div className="modal-overlay" onMouseDown={onClose}>
      <div className="modal" style={width ? { width } : undefined} onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose} aria-label="Close">&times;</button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>
  );
}
