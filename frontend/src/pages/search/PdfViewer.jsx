import { useCallback, useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import './pdfViewer.css';

// Render PDFs in-app (pages drawn to canvas) so they always display inline,
// regardless of the browser's "download PDFs" setting. Pages render lazily as
// they scroll into view; fit-to-width is the 100% baseline, adjustable by zoom.
pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export default function PdfViewer({ url }) {
  const boxRef = useRef(null);
  const docRef = useRef(null);
  const obsRef = useRef(null);
  const fitRef = useRef(1);
  const zoomRef = useRef(100);
  const [zoom, setZoom] = useState(100);
  const [status, setStatus] = useState('loading'); // loading | ready | error

  const scale = () => (fitRef.current * zoomRef.current) / 100;

  const renderInto = useCallback(async (wrap) => {
    const pdf = docRef.current;
    if (!pdf || wrap.dataset.rendered === '1') return;
    wrap.dataset.rendered = '1';
    try {
      const page = await pdf.getPage(+wrap.dataset.page);
      const dpr = window.devicePixelRatio || 1;
      const vp = page.getViewport({ scale: scale() });
      wrap.style.width = `${vp.width}px`;
      wrap.style.height = `${vp.height}px`;
      wrap.innerHTML = '';
      const canvas = document.createElement('canvas');
      canvas.width = Math.floor(vp.width * dpr);
      canvas.height = Math.floor(vp.height * dpr);
      canvas.style.width = `${vp.width}px`;
      canvas.style.height = `${vp.height}px`;
      wrap.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      await page.render({ canvasContext: ctx, viewport: vp }).promise;
      // Selectable/copyable text layer aligned over the canvas.
      try {
        const textContent = await page.getTextContent();
        const tlDiv = document.createElement('div');
        tlDiv.className = 'textLayer';
        tlDiv.style.setProperty('--scale-factor', String(scale()));
        tlDiv.style.width = `${vp.width}px`;
        tlDiv.style.height = `${vp.height}px`;
        wrap.appendChild(tlDiv);
        const layer = new pdfjsLib.TextLayer({ textContentSource: textContent, container: tlDiv, viewport: vp });
        await layer.render();
      } catch { /* text layer optional */ }
    } catch { /* page render failure is non-fatal */ }
  }, []);

  // preserve=true keeps the reader on the same page across a rebuild (zoom, or a
  // width change when a pane opens/closes) instead of snapping back to page 1.
  const paint = useCallback(async (preserve = false) => {
    const box = boxRef.current;
    const pdf = docRef.current;
    if (!box || !pdf) return;
    // Capture which page (and how far into it) is at the top, before we wipe.
    let anchorIdx = 0; let anchorFrac = 0;
    if (preserve && box.children.length) {
      const st = box.scrollTop;
      const kids = box.children;
      for (let i = 0; i < kids.length; i += 1) {
        const el = kids[i];
        if (el.offsetTop + el.offsetHeight > st) {
          anchorIdx = i; anchorFrac = (st - el.offsetTop) / (el.offsetHeight || 1); break;
        }
      }
    }
    if (obsRef.current) { obsRef.current.disconnect(); obsRef.current = null; }
    box.innerHTML = '';
    const first = await pdf.getPage(1);
    const vp1 = first.getViewport({ scale: scale() });
    const placeholders = [];
    for (let i = 1; i <= pdf.numPages; i += 1) {
      const wrap = document.createElement('div');
      wrap.className = 'pdfv-page';
      wrap.style.width = `${vp1.width}px`;
      wrap.style.height = `${vp1.height}px`;
      wrap.dataset.page = i;
      wrap.dataset.rendered = '0';
      box.appendChild(wrap);
      placeholders.push(wrap);
    }
    // Restore the reading position relative to the rebuilt (rescaled) pages.
    if (preserve && placeholders[anchorIdx]) {
      const ph = placeholders[anchorIdx];
      box.scrollTop = ph.offsetTop + anchorFrac * ph.offsetHeight;
    }
    obsRef.current = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) renderInto(e.target); });
    }, { root: box, rootMargin: '700px 0px' });
    placeholders.forEach((w) => obsRef.current.observe(w));
    // Render the pages currently in view (not just page 1) so a preserved jump
    // doesn't show blank placeholders.
    const top = box.scrollTop; const h = box.clientHeight;
    placeholders.forEach((w) => {
      if (w.offsetTop + w.offsetHeight >= top - 200 && w.offsetTop <= top + h + 200) renderInto(w);
    });
  }, [renderInto]);

  const computeFit = useCallback(async () => {
    const box = boxRef.current;
    const pdf = docRef.current;
    if (!box || !pdf) return;
    const page = await pdf.getPage(1);
    const vn = page.getViewport({ scale: 1 });
    const avail = (box.clientWidth || 600) - 28;
    if (avail > 80) fitRef.current = Math.max(0.2, avail / vn.width);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    zoomRef.current = 100;
    setZoom(100);
    // withCredentials so cookie-authed PDF endpoints (e.g. attached document
    // PDFs at /api/doc-pdf/...) load instead of 401-ing.
    const task = pdfjsLib.getDocument({ url, withCredentials: true });
    task.promise.then(async (pdf) => {
      if (cancelled) { try { pdf.destroy(); } catch { /* noop */ } return; }
      docRef.current = pdf;
      await new Promise((r) => requestAnimationFrame(() => r())); // let the pane lay out
      if (cancelled) return;
      await computeFit();
      await paint();
      setStatus('ready');
    }).catch(() => { if (!cancelled) setStatus('error'); });
    return () => {
      cancelled = true;
      if (obsRef.current) { obsRef.current.disconnect(); obsRef.current = null; }
      try { task.destroy(); } catch { /* noop */ }
      docRef.current = null;
    };
  }, [url, computeFit, paint]);

  // Re-fit and repaint when the pane width changes (open, resize, sidebar toggle).
  useEffect(() => {
    const box = boxRef.current;
    if (!box || typeof ResizeObserver === 'undefined') return undefined;
    let last = box.clientWidth;
    const ro = new ResizeObserver(() => {
      const w = box.clientWidth;
      if (docRef.current && w > 0 && Math.abs(w - last) > 4) { last = w; computeFit().then(() => paint(true)); }
    });
    ro.observe(box);
    return () => ro.disconnect();
  }, [computeFit, paint]);

  const changeZoom = (delta) => {
    const next = Math.min(300, Math.max(50, zoomRef.current + delta));
    if (next === zoomRef.current) return;
    zoomRef.current = next;
    setZoom(next);
    paint(true);
  };

  return (
    <div className="pdfv">
      <div className="pdfv-toolbar">
        <button type="button" className="pdfv-zoom" onClick={() => changeZoom(-10)} aria-label="Zoom out">−</button>
        <span className="pdfv-zoom-lbl">{zoom}%</span>
        <button type="button" className="pdfv-zoom" onClick={() => changeZoom(10)} aria-label="Zoom in">+</button>
      </div>
      <div className="pdfv-scroll" ref={boxRef} />
      {status !== 'ready' && (
        <div className="pdfv-msg">
          {status === 'loading'
            ? <><span className="spin"><i className="fas fa-circle-notch" /></span> Loading PDF…</>
            : <><i className="fas fa-triangle-exclamation" /> Could not render this PDF.</>}
        </div>
      )}
    </div>
  );
}
