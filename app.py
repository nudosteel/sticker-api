# VERSION 34.6 - Stable upload version without rembg

from io import BytesIO
import base64
from pathlib import Path
import math
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter

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


def has_useful_alpha(img: Image.Image) -> bool:
    alpha = np.array(img.convert("RGBA"))[:, :, 3]
    return bool(np.any(alpha < 250))


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


def add_canvas_padding(img: Image.Image, padding_ratio: float = 0.14, min_px: int = 70) -> Image.Image:
    w, h = img.size
    pad = max(min_px, int(max(w, h) * padding_ratio))
    canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    canvas.alpha_composite(img, dest=(pad, pad))
    return canvas


def make_ellipse_kernel(w: int, h: int = None) -> np.ndarray:
    if h is None:
        h = w
    w = max(3, int(w))
    h = max(3, int(h))
    if w % 2 == 0:
        w += 1
    if h % 2 == 0:
        h += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))


def sanitize_design_rgba(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.uint8)
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


def antialias_rgba_edges(img: Image.Image, sigma: float = 0.9) -> Image.Image:
    arr = np.array(img.convert("RGBA")).copy()
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def build_alpha_mask(img: Image.Image) -> np.ndarray:
    alpha = np.array(img)[:, :, 3].astype(np.uint8)
    return np.where(alpha >= 8, 255, 0).astype(np.uint8)


def extract_logo_from_light_background(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    fg = ((val < 245) | (sat > 25)).astype(np.uint8) * 255

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, make_ellipse_kernel(3), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, make_ellipse_kernel(5), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    clean = np.zeros_like(fg)
    h, w = fg.shape
    min_area = max(40, int((h * w) * 0.0005))

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    alpha = clean.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2, sigmaY=1.2)
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    out = np.array(rgba).copy()
    out[:, :, 3] = alpha
    return Image.fromarray(out, "RGBA")


def prepare_input_image(img: Image.Image) -> tuple[Image.Image, str]:
    img = img.convert("RGBA")

    if has_useful_alpha(img):
        return img, "alpha"

    light_bg = extract_logo_from_light_background(img)
    light_alpha = np.array(light_bg)[:, :, 3]
    coverage = float(np.count_nonzero(light_alpha)) / float(light_alpha.size)

    if 0.001 < coverage < 0.55:
        return light_bg, "light_bg"

    return img, "none"


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
            "x1": x, "y1": y, "x2": x + w - 1, "y2": y + h - 1,
            "w": w, "h": h, "area": area
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
    gap_x = max(10, int(max_dim * 0.035))
    gap_y = max(10, int(max_dim * 0.04))
    diag_gap = max(12, int(max_dim * 0.045))

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            a = components[i]
            b = components[j]
            dx, dy, dist = bbox_gap(a, b)

            same_row = dy <= gap_y and dx <= gap_x * 2
            stacked = dx <= gap_x and dy <= gap_y * 2
            near = dist <= diag_gap

            if same_row or stacked or near:
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


def merge_cluster_shape(cluster_mask: np.ndarray, max_dim: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cluster_mask, 8)
    if num <= 2:
        return cluster_mask

    comps = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        comps.append((x, y, w, h))

    total_w = max(x + w for x, y, w, h in comps) - min(x for x, y, w, h in comps)
    total_h = max(y + h for x, y, w, h in comps) - min(y for x, y, w, h in comps)

    if total_w >= total_h * 1.4:
        kernel = make_ellipse_kernel(max(5, int(max_dim * 0.012)), max(3, int(max_dim * 0.006)))
    else:
        kernel = make_ellipse_kernel(max(5, int(max_dim * 0.010)), max(7, int(max_dim * 0.016)))

    return cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def metaball_outline(mask: np.ndarray, border_px: int) -> np.ndarray:
    inv = 255 - mask
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    expanded = np.where(dist <= border_px, 255, 0).astype(np.uint8)

    blur = max(5, int(border_px * 0.55))
    if blur % 2 == 0:
        blur += 1

    blurred = cv2.GaussianBlur(expanded, (blur, blur), 0)
    _, smooth = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    clean_k = max(3, int(border_px * 0.18))
    kernel = make_ellipse_kernel(clean_k)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel, iterations=1)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel, iterations=1)
    return smooth


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


def make_sticker_mask(alpha_mask: np.ndarray) -> np.ndarray:
    h, w = alpha_mask.shape
    max_dim = max(h, w)

    comps, labels = get_components(alpha_mask, min_area=max(16, int(max_dim * 0.001)))
    clusters = cluster_components(comps, max_dim)

    if not clusters:
        base = alpha_mask.copy()
    else:
        base = np.zeros_like(alpha_mask)
        for cluster in clusters:
            cluster_mask = cluster_mask_from_labels(labels, cluster)
            cluster_mask = merge_cluster_shape(cluster_mask, max_dim)
            base = cv2.bitwise_or(base, cluster_mask)

    border_px = max(16, int(max_dim * 0.05))
    sticker = metaball_outline(base, border_px)
    sticker = fill_small_inner_holes(sticker, max_hole_area=4200)

    contours, _ = cv2.findContours(sticker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(sticker)
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


def load_texture(material: str, size):
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


def create_shadow_from_mask(mask: np.ndarray, blur_radius: int = 18, opacity: int = 15, offset=(0, 7)) -> Image.Image:
    h, w = mask.shape
    alpha = Image.fromarray(mask, "L")

    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow_rgba = Image.new("RGBA", (w, h), (0, 0, 0, opacity))
    shadow_rgba.putalpha(shadow_alpha)

    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.alpha_composite(shadow_rgba, dest=offset)
    return base


def compose_final_preview(design_img: Image.Image, sticker_alpha: np.ndarray, material: str):
    texture_path = None
    w, h = design_img.size

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow = create_shadow_from_mask(sticker_alpha, blur_radius=15, opacity=15, offset=(0, 11))
    canvas.alpha_composite(shadow)

    if material == "holographic":
        texture, texture_path = load_texture(material, design_img.size)
        if texture is not None:
            holo_base = apply_alpha_mask(texture, sticker_alpha)
            canvas.alpha_composite(holo_base)
        else:
            canvas.alpha_composite(make_rgba_from_alpha(sticker_alpha, (255, 255, 255)))
    else:
        canvas.alpha_composite(make_rgba_from_alpha(sticker_alpha, (255, 255, 255)))

    canvas.alpha_composite(design_img)
    return canvas, texture_path


@app.get("/")
def root():
    return {"ok": True, "version": "34.6"}


@app.post("/process-sticker")
async def process_sticker(file: UploadFile = File(...), material: str = Form("vinyl")):
    try:
        data = await file.read()
        raw_img = load_rgba_from_bytes(data)

        prepared_img, bg_method = prepare_input_image(raw_img)
        design_trimmed = trim_transparent(prepared_img, padding_ratio=0.08)
        clean_design = sanitize_design_rgba(design_trimmed)
        clean_design = antialias_rgba_edges(clean_design, sigma=0.9)
        padded_design = add_canvas_padding(clean_design, padding_ratio=0.14, min_px=70)

        alpha_mask = build_alpha_mask(padded_design)
        sticker_alpha = make_sticker_mask(alpha_mask)
        final_preview, texture_path = compose_final_preview(padded_design, sticker_alpha, material)

        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "debug_version": "34.6",
            "debug_texture_found": texture_path is not None,
            "debug_texture_path": texture_path,
            "debug_bg_method": bg_method
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()}
        )
