// Build a transparent-background favicon from the logo at runtime: load the
// PNG, knock out near-white pixels, and set it as the tab icon. Avoids shipping
// a separate edited asset and needs no image tooling on the backend.
export function setTransparentFavicon(src = '/static/iris_logo.png') {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const px = data.data;
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] > 238 && px[i + 1] > 238 && px[i + 2] > 238) px[i + 3] = 0; // near-white -> transparent
      }
      ctx.putImageData(data, 0, 0);
      const href = canvas.toDataURL('image/png');
      let link = document.querySelector("link[rel='icon']");
      if (!link) { link = document.createElement('link'); link.rel = 'icon'; document.head.appendChild(link); }
      link.type = 'image/png';
      link.href = href;
    } catch { /* tainted canvas or unsupported — keep the original favicon */ }
  };
  img.src = src;
}
