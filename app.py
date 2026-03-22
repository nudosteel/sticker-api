# VERSION 31 - True Mathematical Circular Offset (DIST_MASK_PRECISE)

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


def sanitize_design_rgba(design_trimmed: Image.Image) -> Image.Image:
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
    arr = np.array(design_img)
    alpha = arr[:, :, 3].astype(np.uint8)

    mask = np.where(alpha >= 8, 255, 0).astype(np.uint8)

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
            "cx": x + w / 2.0,
            "cy": y + h / 2.0,
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
    if not components:
        return []

    dsu = DSU(len(components))

    horizontal_gap = max(12, int(max_dim * 0.05))
    vertical_gap = max(10, int(max_dim * 0.04))
    diag_gap = max(14, int(max_dim * 0.055))

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            a = components[i]
            b = components[j]

            dx, dy, dist = bbox_gap(a, b)

            same_row_like = dy <= vertical_gap
            stacked_like = dx <= horizontal_gap
            near_enough = dist <= diag_gap

            if same_row_like or stacked_like or near_enough:
                dsu.union(i, j)

    groups = {}
    for idx, comp in enumerate(components):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(comp)

    return list(groups.values())


def merge_cluster_mask(labels: np.ndarray, cluster) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=np.uint8)
    ids = [c["id"] for c in cluster]
    mask[np.isin(labels, ids)] = 255
    return mask


def exact_circular_dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Expansión MATEMÁTICA perfecta.
    El uso de cv2.DIST_MASK_PRECISE obliga a calcular el círculo exacto
    sin aproximaciones poligonales rápidas.
    """
    if radius <= 0:
        return mask
    inv = 255 - mask
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(dist <= radius, 255, 0).astype(np.uint8)


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
    h, w = alpha_mask.shape
    max_dim = max(h, w)

    comps, labels = get_components(alpha_mask, min_area=max(16, int(max_dim * 0.001)))
    clusters = cluster_components(comps, max_dim)

    base = np.zeros_like(alpha_mask)

    if not clusters:
        return alpha_mask

    bridge_radius = max(3, int(max_dim * 0.008))

    for cluster in clusters:
        cluster_mask = merge_cluster_mask(labels, cluster)

        if bridge_radius > 0:
            expanded = exact_circular_dilate(cluster_mask, bridge_radius)
            # Aquí también forzamos la precisión absoluta
            dist_in = cv2.distanceTransform(expanded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            closed = np.where(dist_in > bridge_radius, 255, 0).astype(np.uint8)
        else:
            closed = cluster_mask

        # CHAIN_APPROX_NONE para no perder resolución en las curvas
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, contours, -1, 255, thickness=cv2.FILLED)

    return base


def make_sticker_mask(alpha_mask: np.ndarray) -> np.ndarray:
    h, w = alpha_mask.shape
    max_dim = max(h, w)

    cutline_base = make_cutline_base(alpha_mask)

    # Grosor del borde de corte
    border_px = max(18, int(max_dim * 0.06))
    
    # Dilatación matemática estricta
    dilated = exact_circular_dilate(cutline_base, border_px)

    # Suavizado final ligero (anti-aliasing)
    blur_size = max(5, int(border_px * 0.2))
    if blur_size % 2 == 0:
        blur_size += 1
        
    blurred = cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
    _, shaped = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    shaped = fill_small_inner_holes(shaped, max_hole_area=6000)

    # CHAIN_APPROX_NONE evita que opencv convierta las curvas suaves en polígonos
    contours, _ = cv2.findContours(shaped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    return {"ok": True, "version": 31}


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
            "debug_version": 31
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
