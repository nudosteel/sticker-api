<!-- VERSION 14 - Sticker Wizard completo -->

<div class="stk-wizard" id="stk-wizard">
  <div id="stk-preview-box">
    <div class="stk-scaler-container">
      <img id="stk-border-layer-shadow" src="" style="display:none;" />
      <img id="stk-final-preview" src="" style="display:none;" />
    </div>
    <div id="stk-text-placeholder">Sube tu diseño</div>
  </div>

  <div class="stk-controls">
    <label class="stk-label">1. Sube tu diseño:</label>
    <input type="file" id="stk-image-upload" accept=".png,.jpg,.jpeg,.webp" />

    <label class="stk-label">2. Material:</label>
    <select id="stk-material-selector">
      <option value="vinyl">Vinilo</option>
      <option value="holographic">Holográfico</option>
      <option value="transparent">Transparente</option>
      <option value="glitter">Glitter</option>
      <option value="mirror">Espejo</option>
      <option value="reflective">Reflectivo</option>
    </select>

    <div class="stk-row">
      <div>
        <label class="stk-label">3. Ancho (cm)</label>
        <input type="number" id="stk-width" min="3" max="100" step="0.1" value="8" />
      </div>
      <div>
        <label class="stk-label">4. Alto (cm)</label>
        <input type="number" id="stk-height" min="3" max="100" step="0.1" value="8" />
      </div>
    </div>

    <label class="stk-label">5. Cantidad</label>
    <input type="number" id="stk-qty" min="1" step="1" value="50" />

    <div class="stk-price-box">
      <div class="stk-price-label">Precio estimado</div>
      <div class="stk-price" id="stk-price">€0.00</div>
      <div class="stk-price-note" id="stk-price-note">Sube un diseño para empezar</div>
    </div>
  </div>
</div>

<style>
  .stk-wizard{
    font-family:Arial,sans-serif;
    max-width:460px;
    margin:0 auto;
    background:#fff;
    padding:20px;
    border-radius:16px;
    box-shadow:0 8px 30px rgba(0,0,0,.08);
  }

  #stk-preview-box{
    position:relative;
    width:100%;
    height:360px;
    background:radial-gradient(circle at 30% 20%, #f6f6f6 0%, #ebebeb 35%, #dddddd 100%);
    border:2px dashed #a5a5a5;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-bottom:18px;
    overflow:hidden;
  }

  #stk-text-placeholder{
    color:#555;
    font-weight:700;
    background:rgba(255,255,255,.80);
    padding:6px 12px;
    border-radius:8px;
    z-index:20;
  }

  .stk-scaler-container{
    position:relative;
    width:76%;
    height:76%;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:visible;
  }

  #stk-border-layer-shadow,
  #stk-final-preview{
    position:absolute;
    width:100%;
    height:100%;
    object-fit:contain;
    transition:all .25s ease;
  }

  #stk-border-layer-shadow{
    z-index:1;
    filter:brightness(0) blur(14px);
    opacity:0.22;
    transform:translateY(22px) scale(1.02);
    pointer-events:none;
  }

  #stk-final-preview{
    z-index:2;
  }

  .stk-controls{
    display:flex;
    flex-direction:column;
    gap:10px;
  }

  .stk-row{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
  }

  .stk-label{
    font-size:13px;
    font-weight:700;
    color:#444;
    margin-bottom:4px;
    display:block;
  }

  .stk-wizard select,
  .stk-wizard input[type="file"],
  .stk-wizard input[type="number"]{
    width:100%;
    padding:10px;
    border-radius:8px;
    border:1px solid #d0d0d0;
    background:#fff;
    box-sizing:border-box;
  }

  .stk-price-box{
    margin-top:6px;
    background:#f7f7f7;
    border:1px solid #e3e3e3;
    border-radius:12px;
    padding:14px;
  }

  .stk-price-label{
    font-size:12px;
    color:#666;
    margin-bottom:4px;
  }

  .stk-price{
    font-size:28px;
    font-weight:800;
    color:#111;
    line-height:1;
  }

  .stk-price-note{
    margin-top:6px;
    font-size:12px;
    color:#666;
  }
</style>

<script>
/* VERSION 14 - Frontend final composed preview */

(function () {
  const uploadInput = document.getElementById('stk-image-upload');
  const finalPreview = document.getElementById('stk-final-preview');
  const borderShadowLayer = document.getElementById('stk-border-layer-shadow');
  const placeholder = document.getElementById('stk-text-placeholder');
  const matSelector = document.getElementById('stk-material-selector');
  const widthInput = document.getElementById('stk-width');
  const heightInput = document.getElementById('stk-height');
  const qtyInput = document.getElementById('stk-qty');
  const priceEl = document.getElementById('stk-price');
  const priceNote = document.getElementById('stk-price-note');

  let hasUploadedImage = false;
  let currentFile = null;
  let currentContour = null;

  const API_URL = 'https://sticker-api-production-6a3d.up.railway.app/process-sticker';

  const MATERIALS = {
    vinyl:       { label: 'Vinilo',       multiplier: 1.00 },
    holographic: { label: 'Holográfico',  multiplier: 1.25 },
    transparent: { label: 'Transparente', multiplier: 1.15 },
    glitter:     { label: 'Glitter',      multiplier: 1.30 },
    mirror:      { label: 'Espejo',       multiplier: 1.35 },
    reflective:  { label: 'Reflectivo',   multiplier: 1.40 }
  };

  function formatPrice(value) {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'EUR'
    }).format(value);
  }

  function getAreaPrice(width, height) {
    return Math.max(2.50, width * height * 0.12);
  }

  function getQtyDiscountMultiplier(qty) {
    if (qty >= 500) return 0.72;
    if (qty >= 250) return 0.80;
    if (qty >= 100) return 0.88;
    if (qty >= 50) return 0.94;
    return 1;
  }

  function updatePrice() {
    const w = parseFloat(widthInput.value) || 0;
    const h = parseFloat(heightInput.value) || 0;
    const qty = parseInt(qtyInput.value) || 1;
    const material = MATERIALS[matSelector.value];

    if (!hasUploadedImage) {
      priceEl.textContent = formatPrice(0);
      priceNote.textContent = 'Sube un diseño para empezar';
      return;
    }

    const unitBase = getAreaPrice(w, h);
    const unitMaterial = unitBase * material.multiplier;
    const unitFinal = unitMaterial * getQtyDiscountMultiplier(qty);
    const total = unitFinal * qty;

    priceEl.textContent = formatPrice(total);
    priceNote.textContent = `${material.label} · ${w}×${h} cm · ${qty} uds`;
  }

  function showProcessedImages(finalPreviewSrc, contourSrc) {
    placeholder.style.display = 'none';

    finalPreview.src = finalPreviewSrc;
    borderShadowLayer.src = contourSrc;
    currentContour = contourSrc;

    finalPreview.style.display = 'block';
    borderShadowLayer.style.display = 'block';

    hasUploadedImage = true;
    updatePrice();
  }

  async function processWithAPI(file, material) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('material', material);

    const response = await fetch(API_URL, {
      method: 'POST',
      mode: 'cors',
      body: formData
    });

    let data = null;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error('La API no devolvió JSON válido');
    }

    console.log('Sticker API response:', data);

    if (!response.ok || !data.ok) {
      throw new Error(data?.error || `Error HTTP ${response.status}`);
    }

    if (!data.final_preview_png || !data.contour_png) {
      throw new Error('La API no devolvió el preview final');
    }

    return data;
  }

  async function renderSticker(file, material) {
    priceNote.textContent = 'Procesando imagen...';
    const data = await processWithAPI(file, material);
    showProcessedImages(data.final_preview_png, data.contour_png);
  }

  uploadInput.addEventListener('change', async function (e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    currentFile = file;

    try {
      await renderSticker(file, matSelector.value);
    } catch (error) {
      console.error('Sticker API error:', error);
      priceNote.textContent = 'Error al procesar la imagen';
      alert('No se pudo procesar la imagen. Revisa la API de Railway.');
    }
  });

  matSelector.addEventListener('change', async function () {
    if (!currentFile) {
      updatePrice();
      return;
    }

    try {
      await renderSticker(currentFile, matSelector.value);
    } catch (error) {
      console.error('Sticker API error:', error);
      priceNote.textContent = 'Error al cambiar material';
    }
  });

  widthInput.addEventListener('input', updatePrice);
  heightInput.addEventListener('input', updatePrice);
  qtyInput.addEventListener('input', updatePrice);

  updatePrice();
})();
</script>
