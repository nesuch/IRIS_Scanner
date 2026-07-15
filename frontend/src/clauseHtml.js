// Shared helpers for rendering stored clause HTML in read views.

// Wrap each <table> in a horizontally-scrollable container so a wide table
// scrolls *within* the clause instead of overflowing the page. A <table> cannot
// be shrunk below its min-content width via CSS (max-width/min-width are clamped
// by the table layout algorithm), so a scroll wrapper is the only robust fix.
// Regulatory clause tables are never nested, so a simple tag wrap is safe.
export function wrapClauseTables(html) {
  if (!html) return html;
  // Collapse long runs of non-breaking spaces to a single normal space. PDF forms
  // (certificates, signature blocks) pad their columns with dozens of &nbsp;, which
  // never wrap and so stretch the clause far past its width and overflow the page.
  // A normal space wraps, so the form reflows within the clause instead.
  html = html.replace(/(?:&nbsp;|&#160;|&#xa0;| )(?:\s|&nbsp;|&#160;|&#xa0;| )+/gi, ' ');
  if (html.indexOf('<table') === -1) return html;
  // Each table gets a small toolbar (with a "Copy table" button) above a
  // horizontal-scroll area. The button has no React handler — clicks are caught
  // by a delegated listener on the clause container (see extractTableForCopy).
  return html
    .replace(/<table(\s|>)/gi,
      '<div class="clause-table-wrap">'
      + '<div class="clause-table-bar"><button type="button" class="clause-table-copy" title="Copy just this table">'
      + '<i class="fas fa-copy"></i> Copy table</button></div>'
      + '<div class="clause-table-scroll"><table$1')
    .replace(/<\/table>/gi, '</table></div></div>');
}

// Delegated click target: given a click's target inside a rendered clause, if it
// landed on a "Copy table" button, return that table's Word-ready HTML + text.
export function extractTableForCopy(target) {
  const btn = target && target.closest && target.closest('.clause-table-copy');
  if (!btn) return null;
  const wrap = btn.closest('.clause-table-wrap');
  const table = wrap && wrap.querySelector('table');
  if (!table) return null;
  return { html: inlineTableStyles(table.outerHTML), text: table.innerText };
}

// Bake table borders/padding into inline styles so a clause pasted into Word /
// Google Docs shows gridlines (those apps get the HTML, not our CSS). Also sets
// the legacy border="1" attribute, which Word honours. Used only for clipboard
// export, not for on-screen rendering.
export function inlineTableStyles(html) {
  if (!html || html.indexOf('<table') === -1) return html;
  try {
    const doc = new DOMParser().parseFromString(`<body>${html}</body>`, 'text/html');
    doc.querySelectorAll('table').forEach((t) => {
      t.style.borderCollapse = 'collapse';
      t.setAttribute('border', '1');
      t.setAttribute('cellspacing', '0');
    });
    doc.querySelectorAll('td, th').forEach((c) => {
      c.style.border = '1px solid #333333';
      c.style.padding = '4px 8px';
      c.style.verticalAlign = c.style.verticalAlign || 'top';
    });
    doc.querySelectorAll('th').forEach((c) => { c.style.fontWeight = '700'; });
    return doc.body.innerHTML;
  } catch {
    return html;
  }
}
