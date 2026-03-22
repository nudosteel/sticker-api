# VERSION 24 - Smart unified silhouette + fixed border

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


def clean_design_alpha(design_img: Image.Image, max_hole_area: int = 10000) -> np.ndarray:
    arr = np.array(design_img)
    alpha = arr[:, :, 3]
    solid = np.where(alpha > 0, 255, 0).astype(np.uint8)

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


def make_ellipse_kernel(size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def smooth_mask(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    blur_size = max(3, int(blur_size))
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    _, th = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return th


def merge_nearby_components(design_alpha: np.ndarray) -> np.ndarray:
    """
    Une elementos cercanos como icono + texto sin cambiar demasiado la forma.
    """
    h, w = design_alpha.shape
    max_dim = max(h, w)

    bridge_px = max(9, int(max_dim * 0.028))
    bridge_kernel = make_ellipse_kernel(bridge_px)

    merged = cv2.morphologyEx(design_alpha, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    merged = smooth_mask(merged, blur_size=3)
    return merged


def simplify_outer_shape(mask: np.ndarray) -> np.ndarray:
    """
    Simplifica ligeramente el contorno exterior para quitar dientes pequeños.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        epsilon = max(1.2, 0.0035 * peri)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(out, [approx], -1, 255, thickness=cv2.FILLED)

    return out


def make_sticker_mask(design_alpha: np.ndarray) -> np.ndarray:
    """
    VERSION 24:
    - une componentes cercanos
    - genera borde fijo preferido
    - redondea ligeramente
    - devuelve silueta exterior limpia
    """
    h, w = design_alpha.shape
    max_dim = max(h, w)

    merged_design = merge_nearby_components(design_alpha)

    # AJUSTE FIJO DEL BORDE
    border_px = max(18, int(max_dim * 0.06))
    border_kernel = make_ellipse_kernel(border_px)
    dilated = cv2.dilate(merged_design, border_kernel, iterations=1)

    close_kernel = make_ellipse_kernel(max(9, border_px // 2))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    simplified = simplify_outer_shape(closed)

    round_kernel = make_ellipse_kernel(max(5, border_px // 3))
    rounded = cv2.morphologyEx(simplified, cv2.MORPH_OPEN, round_kernel, iterations=1)
    rounded = cv2.morphologyEx(rounded, cv2.MORPH_CLOSE, round_kernel, iterations=1)

    contours, _ = cv2.findContours(rounded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(rounded)
    cv2.drawContours(final_mask, contours, -1, 255, thickness=cv2.FILLED)

    final_mask = smooth_mask(final_mask, blur_size=5)
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
    return {"ok": True, "version": 24}


@app.post("/process-sticker")
async def process_sticker(
    file: UploadFile = File(...),
    material: str = Form("vinyl")
):
    try:
        data = await file.read()
        raw_img = load_rgba_from_bytes(data)
        design_trimmed = trim_transparent(raw_img, padding_ratio=0.08)

        design_alpha = clean_design_alpha(design_trimmed, max_hole_area=10000)

        design_arr = np.array(design_trimmed)
        design_arr[:, :, 3] = design_alpha
        design_img = Image.fromarray(design_arr, "RGBA")

        sticker_alpha = make_sticker_mask(design_alpha)

        contour_img = make_rgba_from_alpha(sticker_alpha, (255, 255, 255))
        final_preview, texture_path = compose_final_preview(design_img, sticker_alpha, material)

        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "contour_png": to_base64(contour_img),
            "debug_material": material,
            "debug_texture_found": texture_path is not None,
            "debug_texture_path": texture_path,
            "debug_version": 24
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
