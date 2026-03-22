# VERSION 27 - Alpha-driven contour + dirty edge cleanup

from io import BytesIO
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


def to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_rgba_from_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGBA")


def trim_transparent(img: Image.Image, padding_ratio: float = 0.08) -> Image.Image:
    arr = np.array(img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    pad = int(max(w, h) * padding_ratio)

    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(arr.shape[1], x2 + pad + 1)
    bottom = min(arr.shape[0], y2 + pad + 1)

    cropped = arr[top:bottom, left:right]
    return Image.fromarray(cropped, "RGBA")


def make_ellipse_kernel(size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_alpha_mask(design_img: Image.Image) -> np.ndarray:
    """
    Usa SOLO alpha.
    Así da igual si el arte es negro, naranja o cualquier color.
    """
    arr = np.array(design_img)
    alpha = arr[:, :, 3].astype(np.uint8)

    # Umbral suave para conservar bordes más naturales
    mask = np.where(alpha >= 8, 255, 0).astype(np.uint8)

    # Cerrar huecos internos pequeños del alpha
    inv = 255 - mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    h, w = mask.shape
    border = set()
    border.update(np.unique(labels[0, :]).tolist())
    border.update(np.unique(labels[h - 1, :]).tolist())
    border.update(np.unique(labels[:, 0]).tolist())
    border.update(np.unique(labels[:, w - 1]).tolist())

    for i in range(1, num):
        if i in border:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= 10000:
            mask[labels == i] = 255

    return mask


def fill_small_inner_holes(mask: np.ndarray, max_hole_area: int = 4200) -> np.ndarray:
    solid = mask.copy()
    inv = 255 - solid

    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    h, w = solid.shape
    border = set()
    border.update(np.unique(labels[0, :]).tolist())
    border.update(np.unique(labels[h - 1, :]).tolist())
    border.update(np.unique(labels[:, 0]).tolist())
    border.update(np.unique(labels[:, w - 1]).tolist())

    for i in range(1, num):
        if i in border:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
            solid[labels == i] = 255

    return solid


def sanitize_design_rgba(design_trimmed: Image.Image) -> Image.Image:
    """
    Limpia halos o tonos raros en bordes semitransparentes.
    """
    arr = np.array(design_trimmed).astype(np.uint8)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)

    # Muy bajo alpha: limpiar totalmente
    very_low = arr[:, :, 3] < 12
    rgb[very_low] = 0

    # Semitransparencias: neutralizar suciedad de color
    semi = (arr[:, :, 3] >= 12) & (arr[:, :, 3] < 245)
    if np.any(semi):
        a = alpha[semi][:, None]
        rgb[semi] = rgb[semi] * a + 255.0 * (1.0 - a)

    arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def merge_nearby_components(mask: np.ndarray) -> np.ndarray:
    """
    Une piezas cercanas sin depender del color.
    """
    h, w = mask.shape
    max_dim = max(h, w)

    bridge_px = max(7, int(max_dim * 0.02))
    bridge_kernel = make_ellipse_kernel(bridge_px)

    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    return merged


def make_sticker_mask(alpha_mask: np.ndarray) -> np.ndarray:
    """
    Contorno final basado en alpha.
    """
    h, w = alpha_mask.shape

    merged = merge_nearby_components(alpha_mask)

    # GROSOR FIJO QUE QUIERES MANTENER
    border_px = max(18, int(max(h, w) * 0.06))
    border_kernel = make_ellipse_kernel(border_px)
    dilated = cv2.dilate(merged, border_kernel, iterations=1)

    round_size = max(7, border_px // 2)
    round_kernel = make_ellipse_kernel(round_size)

    shaped = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, round_kernel, iterations=1)
    shaped = cv2.morphologyEx(shaped, cv2.MORPH_OPEN, round_kernel, iterations=1)

    shaped = fill_small_inner_holes(shaped, max_hole_area=4200)

    contours, _ = cv2.findContours(shaped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(shaped)
    cv2.drawContours(final_mask, contours, -1, 255, thickness=cv2.FILLED)

    return final_mask


def make_rgba_from_alpha(alpha: np.ndarray, rgb=(255, 255, 255)) -> Image.Image:
    h, w = alpha.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = rgb[0]
    out[:, :, 1] = rgb[1]
    out[:, :, 2] = rgb[2]
    out[:, :, 3] = alpha
    return Image.fromarray(out, "RGBA")


def find_texture(filename: str) -> Path | None:
    exact = BASE_DIR / "textures" / filename
    if exact.exists():
        return exact

    railway_path = Path("/app/textures") / filename
    if railway_path.exists():
        return railway_path

    for p in BASE_DIR.rglob(filename):
        if p.is_file():
            return p

    return None


def load_texture(material: str, size: tuple[int, int]):
    if material == "holographic":
        path = find_texture("holographic.png")
    else:
        path = None

    if path is None:
        return None, None

    tex = Image.open(path).convert("RGBA")
    tex = tex.resize(size, Image.Resampling.LANCZOS)
    return tex, str(path)


def apply_alpha_mask(img: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    arr[:, :, 3] = alpha_mask
    return Image.fromarray(arr, "RGBA")


def compose_final_preview(design_img: Image.Image, sticker_alpha: np.ndarray, material: str):
    texture_path = None
    canvas = Image.new("RGBA", design_img.size, (0, 0, 0, 0))

    if material == "holographic":
        texture, texture_path = load_texture(material, design_img.size)
        if texture is not None:
            holo_base = apply_alpha_mask(texture, sticker_alpha)
            canvas.alpha_composite(holo_base)
        else:
            white_base = make_rgba_from_alpha(sticker_alpha, (255, 255, 255))
            canvas.alpha_composite(white_base)
    else:
        white_base = make_rgba_from_alpha(sticker_alpha, (255, 255, 255))
        canvas.alpha_composite(white_base)

    canvas.alpha_composite(design_img)
    return canvas, texture_path


@app.get("/")
def root():
    return {"ok": True, "version": 27}


@app.post("/process-sticker")
async def process_sticker(
    file: UploadFile = File(...),
    material: str = Form("vinyl")
):
    try:
        data = await file.read()
        raw_img = load_rgba_from_bytes(data)
        design_trimmed = trim_transparent(raw_img, padding_ratio=0.08)

        clean_design = sanitize_design_rgba(design_trimmed)
        alpha_mask = build_alpha_mask(clean_design)

        sticker_alpha = make_sticker_mask(alpha_mask)

        contour_img = make_rgba_from_alpha(sticker_alpha, (255, 255, 255))
        final_preview, texture_path = compose_final_preview(clean_design, sticker_alpha, material)

        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "contour_png": to_base64(contour_img),
            "debug_material": material,
            "debug_texture_found": texture_path is not None,
            "debug_texture_path": texture_path,
            "debug_version": 27
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
