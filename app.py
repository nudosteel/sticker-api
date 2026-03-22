# VERSION 29 - Vector offset cutline with rounded joins (pyclipper)

from io import BytesIO
import base64
from pathlib import Path
import math

import cv2
import numpy as np
import pyclipper
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
CLIPPER_SCALE = 100.0


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

    return Image.fromarray(arr[top:bottom, left:right], "RGBA")


def sanitize_design_rgba(design_trimmed: Image.Image) -> Image.Image:
    """
    Limpia halos / color basura en píxeles semitransparentes.
    """
    arr = np.array(design_trimmed).astype(np.uint8)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)

    very_low = arr[:, :, 3] < 12
    rgb[very_low] = 0

    semi = (arr[:, :, 3] >= 12) & (arr[:, :, 3] < 245)
    if np.any(semi):
        a = alpha[semi][:, None]
        rgb[semi] = rgb[semi] * a + 255.0 * (1.0 - a)

    arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def build_alpha_mask(design_img: Image.Image) -> np.ndarray:
    """
    Usa solo alpha.
    """
    arr = np.array(design_img)
    alpha = arr[:, :, 3].astype(np.uint8)
    mask = np.where(alpha >= 8, 255, 0).astype(np.uint8)

    # cerrar huecos internos pequeños del arte
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


def get_components(mask: np.ndarray, min_area: int = 16):
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
            "area": area,
        })
    return comps, labels


def bbox_gap(a, b):
    dx = max(0, max(a["x1"], b["x1"]) - min(a["x2"], b["x2"]))
    dy = max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))
    return dx, dy, math.hypot(dx, dy)


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
    """
    Une piezas cercanas (icono + texto, 2 líneas, letras muy próximas).
    """
    if not components:
        return []

    dsu = DSU(len(components))

    horizontal_gap = max(10, int(max_dim * 0.04))
    vertical_gap = max(10, int(max_dim * 0.04))
    diag_gap = max(12, int(max_dim * 0.05))

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            a = components[i]
            b = components[j]
            dx, dy, dist = bbox_gap(a, b)

            if dy <= vertical_gap or dx <= horizontal_gap or dist <= diag_gap:
                dsu.union(i, j)

    groups = {}
    for idx, comp in enumerate(components):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(comp)
    return list(groups.values())


def cluster_mask_from_labels(labels: np.ndarray, cluster) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=np.uint8)
    ids = [c["id"] for c in cluster]
    mask[np.isin(labels, ids)] = 255
    return mask


def contour_to_path(cnt: np.ndarray):
    return [(int(p[0][0] * CLIPPER_SCALE), int(p[0][1] * CLIPPER_SCALE)) for p in cnt]


def simplify_contour(cnt: np.ndarray):
    peri = cv2.arcLength(cnt, True)
    epsilon = max(1.0, 0.0035 * peri)
    return cv2.approxPolyDP(cnt, epsilon, True)


def offset_paths_round(paths, delta_px: float):
    if not paths:
        return []

    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 2.0
    for path in paths:
        if len(path) >= 3:
            pco.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    solution = pco.Execute(delta_px * CLIPPER_SCALE)
    return solution


def paths_to_mask(paths, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    polys = []
    for path in paths:
        if len(path) < 3:
            continue
        pts = np.array(
            [[int(round(x / CLIPPER_SCALE)), int(round(y / CLIPPER_SCALE))] for x, y in path],
            dtype=np.int32
        )
        polys.append(pts)

    if polys:
        cv2.fillPoly(mask, polys, 255)
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


def make_cutline_base(alpha_mask: np.ndarray) -> np.ndarray:
    """
    Base geométrica antes del offset:
    - agrupa componentes cercanos
    - usa solo contornos exteriores
    - simplifica cada contorno
    """
    h, w = alpha_mask.shape
    max_dim = max(h, w)

    comps, labels = get_components(alpha_mask, min_area=max(16, int(max_dim * 0.001)))
    clusters = cluster_components(comps, max_dim)

    if not clusters:
        return alpha_mask

    base = np.zeros_like(alpha_mask)

    for cluster in clusters:
        cluster_mask = cluster_mask_from_labels(labels, cluster)

        # pequeño cierre para unir microseparaciones dentro del cluster
        bridge = max(3, int(max_dim * 0.008))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge | 1, bridge | 1))
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            simp = simplify_contour(cnt)
            cv2.drawContours(base, [simp], -1, 255, thickness=cv2.FILLED)

    return base


def make_sticker_mask(alpha_mask: np.ndarray) -> np.ndarray:
    """
    Cutline final:
    - base geométrica
    - offset vectorial redondo
    - limpieza mínima
    """
    h, w = alpha_mask.shape
    max_dim = max(h, w)

    cutline_base = make_cutline_base(alpha_mask)

    contours, _ = cv2.findContours(cutline_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths = [contour_to_path(cnt) for cnt in contours if len(cnt) >= 3]

    # tu grosor fijo preferido
    border_px = max(18, int(max_dim * 0.06))
    offsetted = offset_paths_round(paths, border_px)

    mask = paths_to_mask(offsetted, alpha_mask.shape)

    # limpieza ligera, no destructiva
    small = max(3, int(border_px * 0.22))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small | 1, small | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = fill_small_inner_holes(mask, max_hole_area=4200)

    return mask


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
    return {"ok": True, "version": 29}


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
            "debug_version": 29
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
