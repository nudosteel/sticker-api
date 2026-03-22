<script>
/* VERSION 31.3.1 - Frontend preview fix */

(function(){
  const API_URL = 'https://sticker-api-production-6a3d.up.railway.app/process-sticker';

  const uploadInput = document.getElementById('stk-image-upload');
  const designImg = document.getElementById('stk-design-preview');
  const borderLayer = document.getElementById('stk-border-layer');
  const borderShadowLayer = document.getElementById('stk-border-layer-shadow');
  const placeholder = document.getElementById('stk-text-placeholder');
  const matSelector = document.getElementById('stk-material-selector');
  const overlay = document.getElementById('stk-material-overlay');
  const widthInput = document.getElementById('stk-width');
  const heightInput = document.getElementById('stk-height');
  const qtyInput = document.getElementById('stk-qty');
  const priceEl = document.getElementById('stk-price');
  const priceNote = document.getElementById('stk-price-note');

  let hasUploadedImage = false;

  const MATERIALS = {
    vinyl:       { label:'Vinilo',       multiplier:1.00 },
    holographic: { label:'Holográfico',  multiplier:1.25 },
    transparent: { label:'Transparente', multiplier:1.15 },
    glitter:     { label:'Glitter',      multiplier:1.30 },
    mirror:      { label:'Espejo',       multiplier:1.35 },
    reflective:  { label:'Reflectivo',   multiplier:1.40 }
  };

  function formatPrice(value){
    return new Intl.NumberFormat('es-ES', {
      style:'currency',
      currency:'EUR'
    }).format(value);
  }

  function getAreaPrice(width, height){
    return Math.max(2.50, width * height * 0.12);
  }

  function getQtyDiscountMultiplier(qty){
    if (qty >= 500) return 0.72;
    if (qty >= 250) return 0.80;
    if (qty >= 100) return 0.88;
    if (qty >= 50) return 0.94;
    return 1;
  }

  function updatePrice(){
    const w = parseFloat(widthInput?.value) || 0;
    const h = parseFloat(heightInput?.value) || 0;
    const qty = parseInt(qtyInput?.value) || 1;
    const material = MATERIALS[matSelector?.value] || MATERIALS.vinyl;

    if (!hasUploadedImage) {
      if (priceEl) priceEl.textContent = formatPrice(0);
      if (priceNote) priceNote.textContent = 'Sube un diseño para empezar';
      return;
    }

    const unitBase = getAreaPrice(w, h);
    const unitMaterial = unitBase * material.multiplier;
    const unitFinal = unitMaterial * getQtyDiscountMultiplier(qty);
    const total = unitFinal * qty;

    if (priceEl) priceEl.textContent = formatPrice(total);
    if (priceNote) priceNote.textContent = `${material.label} · ${w}×${h} cm · ${qty} uds`;
  }

  async function processStickerFile(file, material) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('material', material);

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });

    let data = null;
    try {
      data = await response.json();
    } catch (e) {
      alert('La API no devolvió JSON válido');
      console.log('JSON parse error:', e);
      return null;
    }

    if (!response.ok || !data.ok) {
      console.log('Sticker API error:', data);
      alert(data?.error || data?.trace || 'Error desconocido');
      return null;
    }

    console.log('Sticker API response:', data);
    return data;
  }

  function showPreview(data){
    if (!data || !data.final_preview_png) return;

    if (placeholder) placeholder.style.display = 'none';

    if (designImg) {
      designImg.src = data.final_preview_png;
      designImg.style.display = 'block';
      designImg.style.opacity = '1';
      designImg.style.filter = 'none';
      designImg.style.zIndex = '3';
    }

    if (borderLayer) {
      borderLayer.removeAttribute('src');
      borderLayer.style.display = 'none';
    }

    if (borderShadowLayer) {
      borderShadowLayer.src = data.debug_sticker_mask_png || data.final_preview_png;
      borderShadowLayer.style.display = 'block';
      borderShadowLayer.style.opacity = '0.22';
      borderShadowLayer.style.zIndex = '1';
    }

    if (overlay) {
      overlay.style.display = 'none';
    }
  }

  async function refreshPreviewFromAPI() {
    const file = uploadInput?.files && uploadInput.files[0];
    if (!file) return;

    const material = matSelector?.value || 'vinyl';
    if (priceNote) priceNote.textContent = 'Procesando imagen...';

    const data = await processStickerFile(file, material);
    if (!data) return;

    showPreview(data);
    hasUploadedImage = true;
    updatePrice();

    console.log('debug_alpha_mask_png', data.debug_alpha_mask_png);
    console.log('debug_base_mask_png', data.debug_base_mask_png);
    console.log('debug_sticker_mask_png', data.debug_sticker_mask_png);
  }

  uploadInput?.addEventListener('change', refreshPreviewFromAPI);

  matSelector?.addEventListener('change', async function () {
    if (uploadInput?.files && uploadInput.files[0]) {
      await refreshPreviewFromAPI();
    } else {
      updatePrice();
    }
  });

  widthInput?.addEventListener('input', updatePrice);
  heightInput?.addEventListener('input', updatePrice);
  qtyInput?.addEventListener('input', updatePrice);

  updatePrice();
})();
</script>
