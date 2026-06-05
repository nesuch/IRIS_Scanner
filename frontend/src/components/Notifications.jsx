import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';

// Header notification bell — shows team announcements/communications.
// Unread count is derived from the highest announcement id the user has seen
// (persisted in localStorage), so it clears once the menu is opened.
const SEEN_KEY = 'iris:lastSeenAnnouncement';
const DISMISSED_KEY = 'iris:dismissedAnnouncements';
const LEVEL_ICON = { info: 'fa-circle-info', success: 'fa-circle-check', warning: 'fa-triangle-exclamation' };

const loadDismissed = () => {
  try { return new Set(JSON.parse(localStorage.getItem(DISMISSED_KEY) || '[]')); }
  catch { return new Set(); }
};

export default function Notifications() {
  const [all, setAll] = useState([]);
  const [open, setOpen] = useState(false);
  const [lastSeen, setLastSeen] = useState(() => Number(localStorage.getItem(SEEN_KEY) || 0));
  const [dismissed, setDismissed] = useState(loadDismissed);
  const ref = useRef(null);

  // Visible = active announcements the user hasn't cleared locally.
  const items = all.filter((a) => !dismissed.has(a.id));

  const persistDismissed = (set) => {
    localStorage.setItem(DISMISSED_KEY, JSON.stringify([...set]));
    setDismissed(new Set(set));
  };
  const dismissOne = (id) => { const s = new Set(dismissed); s.add(id); persistDismissed(s); };
  const clearAll = () => { const s = new Set(dismissed); items.forEach((a) => s.add(a.id)); persistDismissed(s); };

  const load = () => api.get('/announcements')
    .then((d) => setAll(d.announcements || []))
    .catch(() => { /* silent */ });

  useEffect(() => {
    load();
    const t = setInterval(load, 120000); // refresh every 2 min
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const maxId = items.reduce((mx, a) => Math.max(mx, a.id), 0);
  const unread = items.filter((a) => a.id > lastSeen).length;

  function toggle() {
    const next = !open;
    setOpen(next);
    if (next && maxId > lastSeen) {
      localStorage.setItem(SEEN_KEY, String(maxId));
      setLastSeen(maxId);
    }
  }

  return (
    <div className="notif" ref={ref}>
      <button type="button" className="notif-btn" onClick={toggle} title="Notifications" aria-label="Notifications">
        <i className="fas fa-bell" />
        {unread > 0 && <span className="notif-dot">{unread > 9 ? '9+' : unread}</span>}
      </button>
      {open && (
        <div className="notif-panel">
          <div className="notif-head">
            <span>Updates &amp; Notices</span>
            {items.length > 0 && <button type="button" className="notif-clear" onClick={clearAll}>Clear all</button>}
          </div>
          <div className="notif-body">
            {items.length === 0 ? (
              <div className="notif-empty"><i className="far fa-bell-slash" /> No notifications.</div>
            ) : items.map((a) => (
              <div className={`notif-item lvl-${a.level || 'info'}`} key={a.id}>
                <i className={`fas ${LEVEL_ICON[a.level] || LEVEL_ICON.info} notif-ic`} />
                <div className="notif-text">
                  <div className="notif-title">{a.title}</div>
                  <div className="notif-msg">{a.body}</div>
                  <div className="notif-time">{a.created_at}</div>
                </div>
                <button type="button" className="notif-dismiss" onClick={() => dismissOne(a.id)} title="Dismiss" aria-label="Dismiss">&times;</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
