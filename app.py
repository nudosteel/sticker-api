# VERSION 25 - Intelligent component clustering + fixed border

from io import BytesIO
import base64
from pathlib import Path
import math

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


def get_components(mask: np.ndarray, min_area: int = 20):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comps.append({
            "id": i,
            "x1": x,
            "y1": y,
            "x2": x + w - 1,
            "y2": y + h - 1,
            "w": w,
            "h": h,
            "area": area
        })
    return comps, labels


def bbox_gap(a, b):
    dx = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
    dy = max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
    return math.hypot(dx, dy), dx, dy


def overlap_ratio_1d(a1, a2, b1, b2):
    inter = max(0, min(a2, b2) - max(a1, b1))
    span = max(1, min(a2 - a1, b2 - b1))
    return inter / span


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_components(components, max_dim):
    if not components:
        return []

    dsu = DSU(len(components))

    base_gap = max(10, int(max_dim * 0.03))
    strong_gap = max(14, int(max_dim * 0.045))

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            a = components[i]
            b = components[j]

            gap, dx, dy = bbox_gap(a, b)
            y_overlap = overlap_ratio_1d(a["y1"], a["y2"], b["y1"], b["y2"])
            x_overlap = overlap_ratio_1d(a["x1"], a["x2"], b["x1"], b["x2"])

            # unión inteligente:
            # - elementos cercanos en la misma línea (icono + texto)
            # - letras cercanas entre sí
            # - piezas casi tocándose
            should_join = (
                gap <= base_gap or
                (dx <= strong_gap and y_overlap > 0.20) or
                (dy <= strong_gap and x_overlap > 0.20)
            )

            if should_join:
                dsu.union(i, j)

    groups = {}
    for idx, comp in enumerate(components):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(comp)

    return list(groups.values())


def merge_component_cluster(labels, cluster, max_dim):
    mask = np.zeros(labels.shape, dtype=np.uint8)
    ids = [c["id"] for c in cluster]
    ids_set = set(ids)
    mask[np.isin(labels, list(ids_set))] = 255

    # une piezas cercanas dentro del cluster
    bridge_px = max(7, int(max_dim * 0.02))
    bridge_kernel = make_ellipse_kernel(bridge_px)
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)

    return merged


def make_sticker_mask(design_alpha: np.ndarray) -> np.ndarray:
    """
    VERSION 25:
    - detecta componentes
    - une solo componentes cercanos
    - genera borde fijo
    - conserva una silueta limpia y estable
    """
    h, w = design_alpha.shape
    max_dim = max(h, w)

    # limpia ruido mínimo
    open_kernel = make_ellipse_kernel(max(3, int(max_dim * 0.006)))
    cleaned = cv2.morphologyEx(design_alpha, cv2.MORPH_OPEN, open_kernel, iterations=1)

    components, labels = get_components(cleaned, min_area=max(20, int(max_dim * 0.002)))
    clusters = cluster_components(components, max_dim)

    merged_global = np.zeros_like(cleaned)
    for cluster in clusters:
        cluster_mask = merge_component_cluster(labels, cluster, max_dim)
        merged_global = cv2.bitwise_or(merged_global, cluster_mask)

    # AJUSTE FIJO DEL BORDE QUE QUIERES MANTENER
    border_px = max(18, int(max_dim * 0.06))
    border_kernel = make_ellipse_kernel(border_px)
    dilated = cv2.dilate(merged_global, border_kernel, iterations=1)

    # suavizado estructural, no blur destructivo
    close_size = max(5, border_px // 3)
    close_kernel = make_ellipse_kernel(close_size)
    shaped = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    shaped = cv2.morphologyEx(shaped, cv2.MORPH_OPEN, close_kernel, iterations=1)

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
    return {"ok": True, "version": 25}


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

        # limpia RGB basura en píxeles casi transparentes
        low_alpha = design_arr[:, :, 3] < 20
        design_arr[low_alpha, 0] = 0
        design_arr[low_alpha, 1] = 0
        design_arr[low_alpha, 2] = 0

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
            "debug_version": 25
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
