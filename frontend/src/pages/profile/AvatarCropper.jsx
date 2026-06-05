import { useEffect, useRef, useState } from 'react';

// Lightweight avatar cropper: a fixed square viewport over a draggable +
// zoomable image. The user positions their face inside the circle; on confirm
// the visible square is drawn to a small JPEG data-URL. No external library.
const VIEW = 264; // on-screen crop viewport (square, px)
const OUT = 224;  // output avatar size (px)

export default function AvatarCropper({ file, onCancel, onDone }) {
  const imgRef = useRef(null);
  const [nat, setNat] = useState(null); // natural { w, h }
  const [zoom, setZoom] = useState(1);
  const [off, setOff] = useState({ x: 0, y: 0 }); // image top-left in viewport coords
  const drag = useRef(null);

  const base = nat ? VIEW / Math.min(nat.w, nat.h) : 1; // cover the viewport
  const scale = base * zoom;
  const dispW = nat ? nat.w * scale : VIEW;
  const dispH = nat ? nat.h * scale : VIEW;

  const clamp = (o) => ({
    x: Math.min(0, Math.max(VIEW - dispW, o.x)),
    y: Math.min(0, Math.max(VIEW - dispH, o.y)),
  });
  const c = clamp(off);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setNat({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Center the image once it loads.
  useEffect(() => {
    if (!nat) return;
    setOff({ x: (VIEW - nat.w * base) / 2, y: (VIEW - nat.h * base) / 2 });
  }, [nat]); // eslint-disable-line

  const point = (e) => (e.touches ? e.touches[0] : e);
  function start(e) {
    const p = point(e);
    drag.current = { px: p.clientX, py: p.clientY, ox: c.x, oy: c.y };
  }
  function move(e) {
    if (!drag.current) return;
    const p = point(e);
    setOff(clamp({ x: drag.current.ox + (p.clientX - drag.current.px), y: drag.current.oy + (p.clientY - drag.current.py) }));
  }
  function end() { drag.current = null; }

  function confirm() {
    const img = imgRef.current;
    if (!img) return;
    const canvas = document.createElement('canvas');
    canvas.width = OUT; canvas.height = OUT;
    const ctx = canvas.getContext('2d');
    const sx = (-c.x) / scale;
    const sy = (-c.y) / scale;
    const sSize = VIEW / scale; // natural px visible in the viewport
    ctx.drawImage(img, sx, sy, sSize, sSize, 0, 0, OUT, OUT);
    onDone(canvas.toDataURL('image/jpeg', 0.85));
  }

  return (
    <div className="modal-overlay" onMouseUp={end} onMouseLeave={end} onTouchEnd={end}>
      <div className="modal" style={{ width: 'min(360px, 94vw)' }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head"><h3>Position your photo</h3><button className="modal-close" onClick={onCancel}>&times;</button></div>
        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
          <div
            className="cropper-view"
            style={{ width: VIEW, height: VIEW }}
            onMouseDown={start} onMouseMove={move}
            onTouchStart={start} onTouchMove={move}
          >
            {nat && (
              <img
                src={imgRef.current?.src} alt="crop" draggable={false}
                style={{ position: 'absolute', left: c.x, top: c.y, width: dispW, height: dispH, userSelect: 'none', touchAction: 'none' }}
              />
            )}
            <div className="cropper-ring" />
          </div>
          <div className="cropper-zoom">
            <i className="fas fa-image" style={{ fontSize: 12 }} />
            <input type="range" min="1" max="3" step="0.01" value={zoom} onChange={(e) => setZoom(parseFloat(e.target.value))} />
            <i className="fas fa-image" style={{ fontSize: 16 }} />
          </div>
          <p className="cropper-hint">Drag to reposition · slide to zoom</p>
        </div>
        <div className="modal-foot">
          <button className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={confirm}><i className="fas fa-check" /> Use photo</button>
        </div>
      </div>
    </div>
  );
}
