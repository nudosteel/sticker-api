# Sticker Wizard - Version 6

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
TEXTURES_DIR = BASE_DIR / "textures"


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


def make_inner_mask(contour_img: Image.Image, inset_px: int = 18) -> Image.Image:
    arr = np.array(contour_img)
    alpha = arr[:, :, 3]
    kernel = np.ones((inset_px, inset_px), np.uint8)
    eroded = cv2.erode(alpha, kernel, iterations=1)

    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[:, :, 0:3] = 255
    out[:, :, 3] = eroded
    return Image.fromarray(out, "RGBA")


def make_white_sticker_base(contour_img: Image.Image) -> Image.Image:
    alpha = np.array(contour_img)[:, :, 3]
    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[:, :, 0:3] = 255
    out[:, :, 3] = alpha
    return Image.fromarray(out, "RGBA")


def load_texture(material: str, size: tuple[int, int]) -> Image.Image | None:
    if material == "holographic":
        path = TEXTURES_DIR / "holographic.png"
        if not path.exists():
            return None

        tex = Image.open(path).convert("RGBA")
        tex = tex.resize(size, Image.Resampling.LANCZOS)
        return tex

    return None


def apply_mask_to_texture(texture: Image.Image, mask_img: Image.Image) -> Image.Image:
    tex = texture.copy().convert("RGBA")
    mask_alpha = np.array(mask_img)[:, :, 3]
    tex_arr = np.array(tex)
    tex_arr[:, :, 3] = mask_alpha
    return Image.fromarray(tex_arr, "RGBA")


def compose_preview(design_img: Image.Image, contour_img: Image.Image, material: str) -> Image.Image:
    white_base = make_white_sticker_base(contour_img)
    inner_mask = make_inner_mask(contour_img, inset_px=18)

    canvas = Image.new("RGBA", design_img.size, (0, 0, 0, 0))
    canvas.alpha_composite(white_base)

    texture = load_texture(material, design_img.size)
    if texture is not None:
        textured = apply_mask_to_texture(texture, inner_mask)
        canvas.alpha_composite(textured)

    canvas.alpha_composite(design_img)
    return canvas


@app.get("/")
def root():
    return {"ok": True}


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
        preview = compose_preview(design, contour, material)

        return JSONResponse({
            "ok": True,
            "design_png": to_base64(design),
            "contour_png": to_base64(contour),
            "preview_png": to_base64(preview)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
