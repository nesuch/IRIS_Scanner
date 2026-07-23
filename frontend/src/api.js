// Thin fetch wrapper around the Flask /api layer. Session-cookie auth, so every
// request includes credentials. A 401 means the session is gone/invalid.

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

// Endpoints where a 401 is a normal, locally-handled outcome (not an expired
// session) — these must NOT trigger the global "bounce to login" behaviour.
function isAuthProbe(path) {
  return path === '/me' || path === '/login' || path === '/logout'
    || path === '/forgot-password' || path.startsWith('/reset-password');
}

// On any unexpected 401, notify the app so it can drop to the login screen
// (handled in AuthContext) instead of surfacing a raw "auth_required" error.
function handleUnauthorized(path) {
  if (!isAuthProbe(path) && typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('iris:unauthorized'));
  }
}

async function request(method, path, body, opts = {}) {
  const headers = {};
  let payload;
  if (body instanceof FormData) {
    payload = body;
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }
  const res = await fetch('/api' + path, {
    method,
    credentials: 'include',
    headers,
    body: payload,
    ...opts,
  });
  if (opts.raw) return res;
  let data = null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) data = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 401) handleUnauthorized(path);
    const msg = (data && (data.message || data.error)) || res.statusText;
    throw new ApiError(msg, res.status, data);
  }
  return data;
}

export const api = {
  get: (p, opts) => request('GET', p, undefined, opts),
  post: (p, body, opts) => request('POST', p, body, opts),
  del: (p, opts) => request('DELETE', p, undefined, opts),
  // Trigger a file download from a POST endpoint (xlsx export).
  async download(p, body, filename) {
    const res = await fetch('/api' + p, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      if (res.status === 401) handleUnauthorized(p);
      const data = await res.json().catch(() => null);
      throw new ApiError((data && data.message) || 'Download failed', res.status, data);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'download.xlsx';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },
};
