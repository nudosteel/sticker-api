# VERSION 13 - Holographic on full white sticker area

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


def trim_transparent(img: Image.Image, padding_ratio: float = 0.10) -> Image.Image:
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


def fill_small_holes(img: Image.Image, max_hole_area: int = 10000) -> Image.Image:
    arr = np.array(img)
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

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 0:3] = 255
    out[:, :, 3] = solid
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
    if material != "holographic":
        return None, None

    path = find_texture("holographic.png")
    if path is None:
        return None, None

    tex = Image.open(path).convert("RGBA")
    tex = tex.resize(size, Image.Resampling.LANCZOS)
    return tex, str(path)


def make_white_area_mask(contour_img: Image.Image, design_img: Image.Image, keepout_px: int = 4) -> Image.Image:
    """
    Máscara del material holográfico sobre toda la zona blanca:
    contorno - diseño expandido ligeramente
    """
    contour_alpha = np.array(contour_img)[:, :, 3]
    design_alpha = np.array(design_img)[:, :, 3]

    design_solid = np.where(design_alpha > 0, 255, 0).astype(np.uint8)

    if keepout_px > 0:
        kernel = np.ones((keepout_px, keepout_px), np.uint8)
        design_solid = cv2.dilate(design_solid, kernel, iterations=1)

    white_area = cv2.subtract(contour_alpha, design_solid)

    out = np.zeros((white_area.shape[0], white_area.shape[1], 4), dtype=np.uint8)
    out[:, :, 0:3] = 255
    out[:, :, 3] = white_area
    return Image.fromarray(out, "RGBA")


def apply_mask_to_texture(texture: Image.Image, mask_img: Image.Image) -> Image.Image:
    tex = texture.copy().convert("RGBA")
    mask_alpha = np.array(mask_img)[:, :, 3]
    tex_arr = np.array(tex)
    tex_arr[:, :, 3] = mask_alpha
    return Image.fromarray(tex_arr, "RGBA")


def make_material_preview(contour_img: Image.Image, design_img: Image.Image, material: str):
    texture, texture_path = load_texture(material, contour_img.size)

    if texture is None:
        return None, texture_path

    material_mask = make_white_area_mask(contour_img, design_img, keepout_px=4)
    preview = apply_mask_to_texture(texture, material_mask)
    return preview, texture_path


@app.get("/")
def root():
    return {"ok": True, "version": 13}


@app.post("/process-sticker")
async def process_sticker(
    file: UploadFile = File(...),
    material: str = Form("vinyl")
):
    try:
        data = await file.read()
        img = load_rgba_from_bytes(data)

        design = trim_transparent(img, padding_ratio=0.10)
        contour = fill_small_holes(design, max_hole_area=10000)
        material_preview, texture_path = make_material_preview(contour, design, material)

        return JSONResponse({
            "ok": True,
            "design_png": to_base64(design),
            "contour_png": to_base64(contour),
            "preview_png": to_base64(material_preview) if material_preview else None,
            "debug_material": material,
            "debug_texture_found": material_preview is not None,
            "debug_texture_path": texture_path,
            "debug_version": 13
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
