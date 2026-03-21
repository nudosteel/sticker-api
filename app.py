# VERSION 9 - Holographic FIXED

from io import BytesIO
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
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


def to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_rgba(data):
    return Image.open(BytesIO(data)).convert("RGBA")


def trim(img):
    arr = np.array(img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0:
        return img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad = int(max(x2 - x1, y2 - y1) * 0.1)

    cropped = arr[
        max(0, y1 - pad): y2 + pad,
        max(0, x1 - pad): x2 + pad
    ]

    return Image.fromarray(cropped, "RGBA")


def contour(img):
    arr = np.array(img)
    alpha = arr[:, :, 3]

    solid = np.where(alpha > 0, 255, 0).astype(np.uint8)

    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[:, :, :3] = 255
    out[:, :, 3] = solid

    return Image.fromarray(out, "RGBA")


def inner_mask(img):
    alpha = np.array(img)[:, :, 3]
    kernel = np.ones((18, 18), np.uint8)
    eroded = cv2.erode(alpha, kernel, iterations=1)

    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[:, :, 3] = eroded

    return Image.fromarray(out, "RGBA")


def find_texture():
    path = BASE_DIR / "textures" / "holographic.png"

    if path.exists():
        return path

    # fallback search
    for p in BASE_DIR.rglob("holographic.png"):
        return p

    return None


@app.post("/process-sticker")
async def process_sticker(file: UploadFile = File(...), material: str = Form("vinyl")):
    try:
        data = await file.read()
        img = load_rgba(data)

        design = trim(img)
        cont = contour(design)

        preview = None
        texture_path = None

        if material == "holographic":
            texture_path = find_texture()

            if texture_path:
                tex = Image.open(texture_path).convert("RGBA")
                tex = tex.resize(cont.size)

                mask = inner_mask(cont)

                tex_arr = np.array(tex)
                mask_alpha = np.array(mask)[:, :, 3]

                tex_arr[:, :, 3] = mask_alpha

                preview = Image.fromarray(tex_arr, "RGBA")

        return {
            "ok": True,
            "design_png": to_base64(design),
            "contour_png": to_base64(cont),
            "preview_png": to_base64(preview) if preview else None,
            "debug_texture": str(texture_path)
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
