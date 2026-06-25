// Copy rich content: writes BOTH text/html and text/plain to the clipboard so
// apps like Word / Google Docs paste the formatting + tables, while plain
// editors still get the text. Falls back to a contenteditable + execCommand
// selection (which also carries rich HTML, and works on plain-HTTP LAN where the
// async Clipboard API is unavailable), then to plain text as a last resort.
export async function copyRich(html, text) {
  try {
    if (navigator.clipboard && window.ClipboardItem && window.isSecureContext) {
      const item = new window.ClipboardItem({
        'text/html': new Blob([html], { type: 'text/html' }),
        'text/plain': new Blob([text], { type: 'text/plain' }),
      });
      await navigator.clipboard.write([item]);
      return true;
    }
  } catch { /* fall through to the legacy rich-selection path */ }
  try {
    const div = document.createElement('div');
    div.setAttribute('contenteditable', 'true');
    div.innerHTML = html;
    div.style.position = 'fixed';
    div.style.left = '-9999px';
    div.style.top = '0';
    document.body.appendChild(div);
    const range = document.createRange();
    range.selectNodeContents(div);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    const ok = document.execCommand('copy');
    sel.removeAllRanges();
    div.remove();
    if (ok) return true;
  } catch { /* fall through to plain text */ }
  return copyText(text);
}

// Copy plain text to the clipboard. Uses the async Clipboard API in secure
// contexts (HTTPS / localhost); falls back to the legacy execCommand path
// otherwise — e.g. when the app is opened over a plain-HTTP LAN address like
// http://192.168.x.x:8080, where navigator.clipboard is unavailable.
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the legacy path */ }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.left = '0';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}
