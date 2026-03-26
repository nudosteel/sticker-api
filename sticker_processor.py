# VERSION 36.3 - PDF upload support + geometric shapes + any-color bg removal
# Changes from v36.1:
#   - NEW: PDF file upload support via PyMuPDF (fitz)
#   - convert_pdf_to_rgba() converts first page of PDF to RGBA at 216 DPI
#   - load_rgba_from_bytes() auto-detects PDF by magic bytes
#   - VALID_FORMATS updated to include application/pdf
# Previous changes (v36.1):
#   - apply_geometric_shape() for circle, oval, square, rounded, contour-cut
#   - /save-design, /webhook/order-paid endpoints
#   - reportlab PDF generation for plotter-ready output

from io import BytesIO
import base64
import hashlib
import time
import threading
from pathlib import Path
import math
import traceback
import os
import smtplib
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
try:
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# EMAIL CONFIG (set via environment variables on Railway)
# ─────────────────────────────────────────────
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER     = os.environ.get("SMTP_USER", "")
SMTP_PASS     = os.environ.get("SMTP_PASS", "")
OWNER_EMAIL   = os.environ.get("OWNER_EMAIL", "jeanmarco@gmail.com")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB = 18
MAX_DIMENSION_PX = 5000
PREVIEW_MAX_DIM = 1200
CACHE_TTL_SECONDS = 600
CACHE_MAX_ENTRIES = 50

VALID_FORMATS = {"image/png", "image/jpeg", "image/webp", "image/tiff", "image/bmp", "application/pdf"}
VALID_MATERIALS = {"vinyl", "matte", "clear", "holographic", "kraft", "glitter", "mirror", "transparent", "reflective"}
VALID_SHAPES = {"contour-cut", "square", "circle", "oval", "rounded"}

# ─────────────────────────────────────────────
# CACHE (thread-safe, LRU + TTL)
# ─────────────────────────────────────────────
class StickerCache:
    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl: int = CACHE_TTL_SECONDS):
        self._lock = threading.Lock()
        self._store: dict[str, dict] = {}
        self._max = max_entries
        self._ttl = ttl

    def _make_key(self, file_hash: str, border_ratio: float) -> str:
        return f"{file_hash}:{border_ratio:.4f}"

    def get(self, file_hash: str, border_ratio: float) -> Optional[dict]:
        key = self._make_key(file_hash, border_ratio)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry["ts"] > self._ttl:
                del self._store[key]
                return None
            entry["ts"] = time.time()
            return entry["data"]

    def put(self, file_hash: str, border_ratio: float, data: dict):
        key = self._make_key(file_hash, border_ratio)
        with self._lock:
            if len(self._store) >= self._max and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k]["ts"])
                del self._store[oldest_key]
            self._store[key] = {"data": data, "ts": time.time()}

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._store), "max": self._max, "ttl": self._ttl}


cache = StickerCache()

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def convert_pdf_to_rgba(data: bytes) -> Image.Image:
    """Convert first page of PDF to RGBA image at high resolution.
    Renders WITHOUT alpha so background is white (not transparent).
    This lets the pipeline detect and remove the white background
    just like it does for JPG/PNG images."""
    if not HAS_PYMUPDF:
        raise HTTPException(400, "PDF no soportado (PyMuPDF no instalado)")
    doc = fitz.open(stream=data, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(3.0, 3.0)  # 3x zoom = ~216 DPI
    pix = page.get_pixmap(matrix=mat, alpha=False)  # white background, no alpha
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img.convert("RGBA")  # convert to RGBA with all alpha=255


def load_rgba_from_bytes(data: bytes) -> Image.Image:
    if data[:5] == b'%PDF-':
        return convert_pdf_to_rgba(data)
    return Image.open(BytesIO(data)).convert("RGBA")


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


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


def add_canvas_padding(img: Image.Image, padding_ratio: float = 0.08, min_px: int = 40) -> Image.Image:
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


def sanitize_design_rgba(img: Image.Image, material: str = "vinyl") -> Image.Image:
    arr = np.array(img).astype(np.uint8)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    very_low = arr[:, :, 3] < 12
    arr[:, :, :3][very_low] = 0
    if material not in ("clear", "holographic", "transparent"):
        rgb = arr[:, :, :3].astype(np.float32)
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


# ─────────────────────────────────────────────
# BACKGROUND REMOVAL
# ─────────────────────────────────────────────
def detect_background_color(img_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = img_rgb.shape[:2]
    border_px = max(5, min(h, w) // 15)
    top = img_rgb[:border_px, :].reshape(-1, 3)
    bottom = img_rgb[h - border_px:, :].reshape(-1, 3)
    left = img_rgb[:, :border_px].reshape(-1, 3)
    right = img_rgb[:, w - border_px:].reshape(-1, 3)
    border_pixels = np.vstack([top, bottom, left, right])
    median_color = np.median(border_pixels, axis=0).astype(np.uint8)
    diffs = np.abs(border_pixels.astype(np.float32) - median_color.astype(np.float32))
    mean_diff = np.mean(diffs)
    is_solid = mean_diff < 30
    return median_color, is_solid


def remove_solid_background(img: Image.Image, bg_color: np.ndarray, tolerance: int = 40) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.float32)
    bg = bg_color.astype(np.float32)
    diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    inner = tolerance * 0.6
    outer = tolerance * 1.2
    alpha_f = np.clip((diff - inner) / max(1, outer - inner), 0, 1)
    fg_alpha = (alpha_f * 255).astype(np.uint8)
    fg_binary = (fg_alpha > 30).astype(np.uint8) * 255
    fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, make_ellipse_kernel(3), iterations=1)
    fg_binary = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, make_ellipse_kernel(7), iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg_binary, 8)
    h, w = fg_binary.shape
    min_area = max(40, int((h * w) * 0.0005))
    clean = np.zeros_like(fg_binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    alpha_final = np.minimum(fg_alpha, clean)
    alpha_final = cv2.GaussianBlur(alpha_final.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    alpha_final = np.clip(alpha_final, 0, 255).astype(np.uint8)
    out = arr.copy()
    out[:, :, 3] = alpha_final
    return Image.fromarray(out, "RGBA")


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
    rgb = np.array(img)[:, :, :3]
    bg_color, is_solid = detect_background_color(rgb)
    if is_solid:
        removed = remove_solid_background(img, bg_color, tolerance=40)
        rm_alpha = np.array(removed)[:, :, 3]
        coverage = float(np.count_nonzero(rm_alpha)) / float(rm_alpha.size)
        if 0.001 < coverage < 0.85:
            return removed, "solid_bg"
    light_bg = extract_logo_from_light_background(img)
    light_alpha = np.array(light_bg)[:, :, 3]
    coverage = float(np.count_nonzero(light_alpha)) / float(light_alpha.size)
    if 0.001 < coverage < 0.55:
        return light_bg, "light_bg"
    return img, "none"


# ─────────────────────────────────────────────
# MASK GENERATION
# ─────────────────────────────────────────────
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
        kernel = make_ellipse_kernel(max(3, int(max_dim * 0.008)), max(3, int(max_dim * 0.004)))
    else:
        kernel = make_ellipse_kernel(max(3, int(max_dim * 0.006)), max(3, int(max_dim * 0.010)))
    return cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def chaikin_smooth(pts: np.ndarray, iterations: int = 3) -> np.ndarray:
    for _ in range(iterations):
        n = len(pts)
        if n < 3:
            return pts
        q = np.empty((n * 2, 2), dtype=pts.dtype)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q[2 * i] = 0.75 * p0 + 0.25 * p1
            q[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = q
    return pts


def smooth_contour_spline(contour: np.ndarray, num_points: int = 300, smoothing: float = 0.0) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 6:
        return contour
    if HAS_SCIPY:
        try:
            pts_closed = np.vstack([pts, pts[0:1]])
            x, y = pts_closed[:, 0], pts_closed[:, 1]
            tck, u = splprep([x, y], s=smoothing, per=True, k=3)
            u_new = np.linspace(0, 1, num_points)
            sx, sy = splev(u_new, tck)
            smooth_pts = np.column_stack([sx, sy]).astype(np.int32)
            return smooth_pts.reshape(-1, 1, 2)
        except (ValueError, TypeError):
            pass
    smooth_pts = chaikin_smooth(pts, iterations=5).astype(np.int32)
    return smooth_pts.reshape(-1, 1, 2)


def metaball_outline(mask: np.ndarray, border_px: int) -> np.ndarray:
    inv = 255 - mask
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    expanded = np.where(dist <= border_px, 255, 0).astype(np.uint8)
    blur = max(3, int(border_px * 0.30))
    if blur % 2 == 0:
        blur += 1
    blurred = cv2.GaussianBlur(expanded, (blur, blur), 0)
    _, smooth = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    clean_k = max(3, int(border_px * 0.12))
    kernel = make_ellipse_kernel(clean_k)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel, iterations=1)
    aa_blur = max(5, int(border_px * 0.40))
    if aa_blur % 2 == 0:
        aa_blur += 1
    aa = cv2.GaussianBlur(smooth, (aa_blur, aa_blur), 0)
    _, smooth = cv2.threshold(aa, 127, 255, cv2.THRESH_BINARY)
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


def make_sticker_mask(alpha_mask: np.ndarray, border_ratio: float = 0.028) -> np.ndarray:
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

    border_px = max(10, int(max_dim * border_ratio))
    sticker = metaball_outline(base, border_px)
    sticker = fill_small_inner_holes(sticker, max_hole_area=4200)

    contours, _ = cv2.findContours(sticker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_mask = np.zeros_like(sticker)
    for cnt in contours:
        if len(cnt) < 6:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        num_pts = max(100, min(800, int(perimeter * 0.4)))
        spline_s = float(border_px) * 1.5
        smooth_cnt = smooth_contour_spline(cnt, num_points=num_pts, smoothing=spline_s)
        cv2.drawContours(final_mask, [smooth_cnt], -1, 255, thickness=cv2.FILLED)

    return final_mask


def apply_geometric_shape(
    design_img: Image.Image,
    alpha_mask: np.ndarray,
    shape: str,
    border_ratio: float = 0.028,
) -> tuple[Image.Image, np.ndarray]:
    if shape == "contour-cut":
        sticker_alpha = make_sticker_mask(alpha_mask, border_ratio)
        return design_img, sticker_alpha

    orig_w, orig_h = design_img.size

    ys, xs = np.where(alpha_mask > 0)
    if len(xs) == 0:
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        return design_img, mask

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    dw = x2 - x1 + 1
    dh = y2 - y1 + 1

    design_diag = max(dw, dh)
    border_px = max(10, int(design_diag * border_ratio))

    if shape == "circle":
        radius = int((design_diag / 2) / 0.55 + border_px)
        sq_size = radius * 2 + border_px * 4
    elif shape == "oval":
        ax_w = int((dw / 2) / 0.55 + border_px)
        ax_h = int((dh / 2) / 0.55 + border_px)
        min_ax = max(ax_w, ax_h)
        ax_w = max(ax_w, int(min_ax * 0.65))
        ax_h = max(ax_h, int(min_ax * 0.65))
        sq_size = max(ax_w, ax_h) * 2 + border_px * 4
    elif shape == "square":
        half = int((design_diag / 2) / 0.55 + border_px)
        sq_size = half * 2 + border_px * 4
    elif shape == "rounded":
        half_w = int((dw / 2) / 0.55 + border_px)
        half_h = int((dh / 2) / 0.55 + border_px)
        sq_size = max(half_w, half_h) * 2 + border_px * 4
    else:
        sticker_alpha = make_sticker_mask(alpha_mask, border_ratio)
        return design_img, sticker_alpha

    sq_size = max(sq_size, max(orig_w, orig_h))
    if sq_size % 2 != 0:
        sq_size += 1

    cx = sq_size // 2
    cy = sq_size // 2

    design_arr = np.array(design_img)
    cropped = Image.fromarray(design_arr[y1:y2+1, x1:x2+1], "RGBA")

    target_area = sq_size * 0.55
    scale = min(target_area / max(1, dw), target_area / max(1, dh))
    scale = min(scale, 1.0)

    new_dw = max(1, int(dw * scale))
    new_dh = max(1, int(dh * scale))
    cropped_scaled = cropped.resize((new_dw, new_dh), Image.Resampling.LANCZOS)

    new_design = Image.new("RGBA", (sq_size, sq_size), (0, 0, 0, 0))
    paste_x = (sq_size - new_dw) // 2
    paste_y = (sq_size - new_dh) // 2
    new_design.alpha_composite(cropped_scaled, dest=(paste_x, paste_y))

    mask = np.zeros((sq_size, sq_size), dtype=np.uint8)

    if shape == "circle":
        radius = int(max(new_dw, new_dh) / 2 / 0.75 + border_px)
        radius = min(radius, sq_size // 2 - 2)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    elif shape == "oval":
        ax_w = int((new_dw / 2) / 0.75 + border_px)
        ax_h = int((new_dh / 2) / 0.75 + border_px)
        min_ax = max(ax_w, ax_h)
        ax_w = max(ax_w, int(min_ax * 0.65))
        ax_h = max(ax_h, int(min_ax * 0.65))
        ax_w = min(ax_w, sq_size // 2 - 2)
        ax_h = min(ax_h, sq_size // 2 - 2)
        cv2.ellipse(mask, (cx, cy), (ax_w, ax_h), 0, 0, 360, 255, -1)

    elif shape == "square":
        half = int(max(new_dw, new_dh) / 2 / 0.75 + border_px)
        half = min(half, sq_size // 2 - 2)
        pt1 = (cx - half, cy - half)
        pt2 = (cx + half, cy + half)
        cv2.rectangle(mask, pt1, pt2, 255, -1)

    elif shape == "rounded":
        half_w = int((new_dw / 2) / 0.75 + border_px)
        half_h = int((new_dh / 2) / 0.75 + border_px)
        half_w = min(half_w, sq_size // 2 - 2)
        half_h = min(half_h, sq_size // 2 - 2)
        corner_r = max(8, int(min(half_w, half_h) * 0.25))
        rx1 = cx - half_w
        ry1 = cy - half_h
        rx2 = cx + half_w
        ry2 = cy + half_h
        cv2.rectangle(mask, (rx1 + corner_r, ry1), (rx2 - corner_r, ry2), 255, -1)
        cv2.rectangle(mask, (rx1, ry1 + corner_r), (rx2, ry2 - corner_r), 255, -1)
        cv2.circle(mask, (rx1 + corner_r, ry1 + corner_r), corner_r, 255, -1)
        cv2.circle(mask, (rx2 - corner_r, ry1 + corner_r), corner_r, 255, -1)
        cv2.circle(mask, (rx1 + corner_r, ry2 - corner_r), corner_r, 255, -1)
        cv2.circle(mask, (rx2 - corner_r, ry2 - corner_r), corner_r, 255, -1)

    aa_blur = max(3, int(border_px * 0.15))
    if aa_blur % 2 == 0:
        aa_blur += 1
    mask = cv2.GaussianBlur(mask, (aa_blur, aa_blur), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return new_design, mask


# ─────────────────────────────────────────────
# COMPOSITION (light step — no heavy CV)
# ─────────────────────────────────────────────
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


TEXTURE_MAP = {
    "holographic": "holographic.png",
    "glitter":     "glitter.png",
    "kraft":       "kraft.png",
    "mirror":      "mirror.png",
    "reflective":  "reflective.png",
}

MATERIAL_BASE_COLOR = {
    "vinyl":       (255, 255, 255),
    "matte":       (245, 245, 245),
    "clear":       (255, 255, 255),
    "holographic": (255, 255, 255),
    "kraft":       (194, 164, 120),
    "glitter":     (255, 255, 255),
    "mirror":      (220, 220, 230),
    "transparent": (255, 255, 255),
    "reflective":  (200, 200, 210),
}


def load_texture(material: str, size) -> tuple[Optional[Image.Image], Optional[str]]:
    filename = TEXTURE_MAP.get(material)
    if filename is None:
        return None, None
    path = find_texture(filename)
    if path is None:
        return None, None
    tex = Image.open(path).convert("RGBA")
    tex = tex.resize(size, Image.Resampling.LANCZOS)
    return tex, str(path)


def apply_alpha_mask(img: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    arr[:, :, 3] = alpha_mask
    return Image.fromarray(arr, "RGBA")


def create_shadow_from_mask(
    mask: np.ndarray,
    blur_radius: int = 28,
    opacity: int = 55,
    offset: tuple[int, int] = (4, 8),
) -> Image.Image:
    h, w = mask.shape
    pad_x = abs(offset[0]) + blur_radius
    pad_y = abs(offset[1]) + blur_radius
    canvas_w = w + pad_x * 2
    canvas_h = h + pad_y * 2
    padded_mask = Image.new("L", (canvas_w, canvas_h), 0)
    padded_mask.paste(Image.fromarray(mask, "L"), (pad_x, pad_y))
    shadow_alpha = padded_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    sa = np.array(shadow_alpha).astype(np.float32)
    sa = sa * (opacity / 255.0)
    shadow_alpha = Image.fromarray(np.clip(sa, 0, 255).astype(np.uint8), "L")
    shadow_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow_color = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    shadow_color.putalpha(shadow_alpha)
    shadow_rgba.alpha_composite(shadow_color, dest=offset)
    result = shadow_rgba.crop((pad_x, pad_y, pad_x + w, pad_y + h))
    return result


def compose_final_preview(
    design_img: Image.Image,
    sticker_alpha: np.ndarray,
    material: str,
) -> tuple[Image.Image, Optional[str]]:
    texture_path = None
    w, h = design_img.size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    shadow = create_shadow_from_mask(sticker_alpha, blur_radius=28, opacity=55, offset=(4, 8))
    canvas.alpha_composite(shadow)

    texture, texture_path = load_texture(material, design_img.size)
    if texture is not None:
        mat_base = apply_alpha_mask(texture, sticker_alpha)
        canvas.alpha_composite(mat_base)
    elif material in ("clear", "transparent"):
        clear_alpha = (sticker_alpha.astype(np.float32) * 0.35).astype(np.uint8)
        canvas.alpha_composite(make_rgba_from_alpha(clear_alpha, (255, 255, 255)))
    else:
        color = MATERIAL_BASE_COLOR.get(material, (255, 255, 255))
        canvas.alpha_composite(make_rgba_from_alpha(sticker_alpha, color))

    canvas.alpha_composite(design_img)
    return canvas, texture_path


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────
def validate_upload(data: bytes, content_type: str) -> None:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
    if content_type and content_type not in VALID_FORMATS:
        raise HTTPException(400, f"Unsupported format: {content_type}. Use PNG, JPEG, WebP, TIFF, BMP, or PDF.")


def validate_dimensions(img: Image.Image) -> None:
    w, h = img.size
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise HTTPException(400, f"Image too large: {w}x{h}px (max {MAX_DIMENSION_PX}px per side).")


def downscale_for_preview(img: Image.Image, max_dim: int = PREVIEW_MAX_DIM) -> tuple[Image.Image, float]:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, 1.0
    ratio = max_dim / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS), ratio


# ─────────────────────────────────────────────
# HEAVY PIPELINE (cached)
# ─────────────────────────────────────────────
def run_heavy_pipeline(
    raw_data: bytes,
    border_ratio: float = 0.028,
    for_preview: bool = True,
) -> dict:
    fhash = file_hash(raw_data)

    cached = cache.get(fhash, border_ratio)
    if cached is not None:
        return {**cached, "cache_hit": True}

    raw_img = load_rgba_from_bytes(raw_data)
    validate_dimensions(raw_img)

    prepared_img, bg_method = prepare_input_image(raw_img)
    design_trimmed = trim_transparent(prepared_img, padding_ratio=0.08)

    if for_preview:
        design_trimmed, scale = downscale_for_preview(design_trimmed)

    clean_design = sanitize_design_rgba(design_trimmed, material="vinyl")
    clean_design = antialias_rgba_edges(clean_design, sigma=0.9)
    padded_design = add_canvas_padding(clean_design, padding_ratio=0.08, min_px=40)

    alpha_mask = build_alpha_mask(padded_design)
    sticker_alpha = make_sticker_mask(alpha_mask, border_ratio=border_ratio)

    result = {
        "padded_design": padded_design,
        "alpha_mask": alpha_mask,
        "sticker_alpha": sticker_alpha,
        "bg_method": bg_method,
        "fhash": fhash,
        "cache_hit": False,
    }

    cache.put(fhash, border_ratio, result)
    return result


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "version": "36.3"}


@app.get("/cache-stats")
def cache_stats():
    return cache.stats()


@app.post("/process-sticker")
async def process_sticker(
    file: UploadFile = File(...),
    material: str = Form("vinyl"),
    shape: str = Form("contour-cut"),
    border_ratio: float = Form(0.028),
):
    if material not in VALID_MATERIALS:
        raise HTTPException(400, f"Unknown material: {material}")
    if shape not in VALID_SHAPES:
        shape = "contour-cut"

    border_ratio = max(0.02, min(0.12, border_ratio))

    try:
        data = await file.read()
        validate_upload(data, file.content_type)

        heavy = run_heavy_pipeline(data, border_ratio=border_ratio, for_preview=True)

        design_for_compose, sticker_alpha = apply_geometric_shape(
            heavy["padded_design"], heavy["alpha_mask"], shape, border_ratio
        )

        final_preview, texture_path = compose_final_preview(
            design_for_compose, sticker_alpha, material
        )

        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "cache_key": heavy["fhash"],
            "border_ratio": border_ratio,
            "material": material,
            "shape": shape,
            "debug_version": "36.3",
            "debug_cache_hit": heavy["cache_hit"],
            "debug_texture_found": texture_path is not None,
            "debug_bg_method": heavy["bg_method"],
        })
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()},
        )


@app.post("/recompose")
async def recompose(
    cache_key: str = Form(...),
    material: str = Form("vinyl"),
    shape: str = Form("contour-cut"),
    border_ratio: float = Form(0.028),
):
    if material not in VALID_MATERIALS:
        raise HTTPException(400, f"Unknown material: {material}")
    if shape not in VALID_SHAPES:
        shape = "contour-cut"

    border_ratio = max(0.02, min(0.12, border_ratio))

    cached = cache.get(cache_key, border_ratio)
    if cached is None:
        raise HTTPException(404, "Cache miss — please re-upload via /process-sticker.")

    try:
        design_for_compose, sticker_alpha = apply_geometric_shape(
            cached["padded_design"], cached["alpha_mask"], shape, border_ratio
        )

        final_preview, texture_path = compose_final_preview(
            design_for_compose, sticker_alpha, material
        )
        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "cache_key": cache_key,
            "border_ratio": border_ratio,
            "material": material,
            "shape": shape,
            "debug_version": "36.3",
            "debug_cache_hit": True,
            "debug_texture_found": texture_path is not None,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()},
        )


# ─────────────────────────────────────────────
# PERSISTENT DESIGN STORAGE (for post-payment processing)
# ─────────────────────────────────────────────
DESIGNS_DIR = BASE_DIR / "saved_designs"
DESIGNS_DIR.mkdir(exist_ok=True)


def save_design_to_disk(order_token: str, file_bytes: bytes, meta: dict):
    folder = DESIGNS_DIR / order_token
    folder.mkdir(exist_ok=True)
    (folder / "design.png").write_bytes(file_bytes)
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_design_from_disk(order_token: str) -> tuple[bytes, dict] | tuple[None, None]:
    folder = DESIGNS_DIR / order_token
    design_path = folder / "design.png"
    meta_path = folder / "meta.json"
    if not design_path.exists() or not meta_path.exists():
        return None, None
    return design_path.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))


@app.post("/save-design")
async def save_design(
    file: UploadFile = File(...),
    material: str = Form("vinyl"),
    shape: str = Form("contour-cut"),
    finish: str = Form("glossy"),
    width_cm: float = Form(8),
    height_cm: float = Form(8),
    quantity: int = Form(50),
    total_price: str = Form("0,00 EUR"),
    comment: str = Form(""),
    border_ratio: float = Form(0.028),
):
    try:
        data = await file.read()
        validate_upload(data, file.content_type)

        fhash = file_hash(data)
        token = f"{fhash}_{int(time.time())}"

        meta = {
            "material": material, "shape": shape, "finish": finish,
            "width_cm": width_cm, "height_cm": height_cm,
            "quantity": quantity, "total_price": total_price,
            "comment": comment, "border_ratio": border_ratio,
            "created_at": datetime.now().isoformat(),
        }
        save_design_to_disk(token, data, meta)

        return JSONResponse({"ok": True, "order_token": token, "debug_version": "36.3"})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


# ─────────────────────────────────────────────
# PDF GENERATION (for plotter)
# ─────────────────────────────────────────────
def generate_cut_contour(sticker_alpha: np.ndarray) -> list:
    contours, _ = cv2.findContours(sticker_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    h, w = sticker_alpha.shape
    pts = biggest.reshape(-1, 2)
    return [(float(p[0]) / w, float(p[1]) / h) for p in pts]


def generate_print_pdf(
    design_img, sticker_alpha, width_cm, height_cm, quantity,
    shape, material, finish,
    customer_name, customer_email, customer_phone,
    comment, total_price, wc_order_id="",
) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed")

    buf = BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    # PAGE 1: ORDER SUMMARY
    c.setFont("Helvetica-Bold", 22)
    c.drawString(2 * cm, page_h - 3 * cm, "Orden de Stickers")
    c.setFont("Helvetica", 10)
    label = f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    if wc_order_id:
        label += f"  |  Pedido WC #{wc_order_id}"
    c.drawString(2 * cm, page_h - 3.6 * cm, label)

    y = page_h - 5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Datos del Cliente")
    y -= 0.7 * cm
    c.setFont("Helvetica", 11)
    for lbl, val in [("Nombre", customer_name), ("Email", customer_email), ("Telefono", customer_phone)]:
        if val:
            c.drawString(2.5 * cm, y, f"{lbl}: {val}")
            y -= 0.55 * cm

    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Especificaciones")
    y -= 0.7 * cm
    c.setFont("Helvetica", 11)
    for lbl, val in [
        ("Forma", shape), ("Material", material), ("Acabado", finish),
        ("Tamano", f"{width_cm} x {height_cm} cm"),
        ("Cantidad", str(quantity)), ("Total", total_price),
    ]:
        c.drawString(2.5 * cm, y, f"{lbl}: {val}")
        y -= 0.55 * cm

    if comment:
        y -= 0.5 * cm
        c.setFont("Helvetica-Bold", 13)
        c.drawString(2 * cm, y, "Comentario")
        y -= 0.7 * cm
        c.setFont("Helvetica", 11)
        c.drawString(2.5 * cm, y, comment[:200])

    try:
        thumb = design_img.copy()
        thumb.thumbnail((300, 300), Image.Resampling.LANCZOS)
        y -= 1.5 * cm
        c.drawImage(ImageReader(thumb), 2.5 * cm, y - thumb.size[1] * 0.24,
                     thumb.size[0] * 0.24, thumb.size[1] * 0.24, mask='auto')
    except Exception:
        pass

    c.showPage()

    # PAGE 2: PRINT SHEET (real size)
    sticker_w_pts = width_cm * cm
    sticker_h_pts = height_cm * cm
    margin = 1.5 * cm
    mark_len = 0.4 * cm
    x_off = (page_w - sticker_w_pts) / 2
    y_off = (page_h - sticker_h_pts) / 2

    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.3)
    for cx_pt, cy_pt in [
        (x_off - margin, y_off - margin), (x_off + sticker_w_pts + margin, y_off - margin),
        (x_off - margin, y_off + sticker_h_pts + margin), (x_off + sticker_w_pts + margin, y_off + sticker_h_pts + margin),
    ]:
        c.line(cx_pt - mark_len, cy_pt, cx_pt + mark_len, cy_pt)
        c.line(cx_pt, cy_pt - mark_len, cx_pt, cy_pt + mark_len)

    try:
        c.drawImage(ImageReader(design_img), x_off, y_off, sticker_w_pts, sticker_h_pts, mask='auto')
    except Exception:
        pass

    contour_pts = generate_cut_contour(sticker_alpha)
    if contour_pts:
        c.setStrokeColorRGB(1, 0, 0)
        c.setLineWidth(0.5)
        c.setDash(2, 1)
        path = c.beginPath()
        first = contour_pts[0]
        path.moveTo(x_off + first[0] * sticker_w_pts, y_off + (1 - first[1]) * sticker_h_pts)
        for pt in contour_pts[1:]:
            path.lineTo(x_off + pt[0] * sticker_w_pts, y_off + (1 - pt[1]) * sticker_h_pts)
        path.close()
        c.drawPath(path, stroke=1, fill=0)

    c.setDash()
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(x_off, y_off - 1 * cm, f"{width_cm} x {height_cm} cm | {material} | {shape} | Qty: {quantity}")
    c.drawString(x_off, y_off - 1.4 * cm, "Linea roja = linea de corte para plotter")

    c.showPage()
    c.save()
    return buf.getvalue()


# ─────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────
def send_order_email(pdf_bytes, customer_name, customer_email, customer_phone,
                     specs_summary, total_price, comment, wc_order_id=""):
    if not SMTP_USER or not SMTP_PASS:
        print("[EMAIL] SMTP not configured")
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = OWNER_EMAIL
    msg["Subject"] = f"PAGO CONFIRMADO - Stickers #{wc_order_id} - {customer_name}"

    body = f"""
    <h2>Pago Confirmado - Orden #{wc_order_id}</h2>
    <h3>Cliente</h3>
    <p><b>Nombre:</b> {customer_name}<br>
    <b>Email:</b> {customer_email}<br>
    <b>Telefono:</b> {customer_phone}</p>
    <h3>Pedido</h3>
    <p>{specs_summary}</p>
    <p><b>Total:</b> {total_price}</p>
    {f'<h3>Comentario</h3><p>{comment}</p>' if comment else ''}
    <hr>
    <p><b>PDF de impresion adjunto listo para plotter.</b></p>
    """
    msg.attach(MIMEText(body, "html"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    att = MIMEApplication(pdf_bytes, _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename=f"sticker_order_{wc_order_id}_{ts}.pdf")
    msg.attach(att)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"[EMAIL] Sent order #{wc_order_id} to {OWNER_EMAIL}")
        return True
    except Exception as e:
        print(f"[EMAIL] Failed: {e}")
        return False


# ─────────────────────────────────────────────
# WEBHOOK: WooCommerce order paid
# ─────────────────────────────────────────────
def verify_wc_webhook(body: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True
    import hmac as hmac_mod
    expected = base64.b64encode(
        hmac_mod.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).digest()
    ).decode()
    return hmac_mod.compare_digest(expected, signature)


@app.post("/webhook/order-paid")
async def webhook_order_paid(request: Request):
    body = await request.body()

    signature = request.headers.get("x-wc-webhook-signature", "")
    if not verify_wc_webhook(body, signature):
        raise HTTPException(401, "Invalid webhook signature")

    try:
        order = json.loads(body)
        wc_order_id = str(order.get("id", ""))
        status = order.get("status", "")

        if status not in ("processing", "completed"):
            return JSONResponse({"ok": True, "skipped": True, "reason": f"status={status}"})

        billing = order.get("billing", {})
        customer_name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()
        customer_email = billing.get("email", "")
        customer_phone = billing.get("phone", "")

        order_token = None
        stk_meta = {}
        for item in order.get("line_items", []):
            for meta in item.get("meta_data", []):
                key = meta.get("key", "")
                val = meta.get("value", "")
                if key == "stk_order_token":
                    order_token = val
                elif key.startswith("stk_"):
                    stk_meta[key] = val

        if not order_token:
            return JSONResponse({"ok": True, "skipped": True, "reason": "no sticker token"})

        file_bytes, saved_meta = load_design_from_disk(order_token)
        if file_bytes is None:
            return JSONResponse(status_code=404, content={"ok": False, "error": f"Design not found: {order_token}"})

        meta = saved_meta
        material = meta.get("material", "vinyl")
        shape = meta.get("shape", "contour-cut")
        finish = meta.get("finish", "glossy")
        width_cm = float(meta.get("width_cm", 8))
        height_cm = float(meta.get("height_cm", 8))
        quantity = int(meta.get("quantity", 50))
        total_price = meta.get("total_price", "")
        comment = meta.get("comment", "")
        border_ratio = float(meta.get("border_ratio", 0.028))

        heavy = run_heavy_pipeline(file_bytes, border_ratio=border_ratio, for_preview=True)

        design_for_compose, sticker_alpha = apply_geometric_shape(
            heavy["padded_design"], heavy["alpha_mask"], shape, border_ratio
        )
        final_preview, _ = compose_final_preview(design_for_compose, sticker_alpha, material)

        pdf_bytes = None
        if HAS_REPORTLAB:
            pdf_bytes = generate_print_pdf(
                final_preview, sticker_alpha, width_cm, height_cm, quantity,
                shape, material, finish,
                customer_name, customer_email, customer_phone,
                comment, total_price, wc_order_id,
            )

        specs = f"Forma: {shape} | Material: {material} | Acabado: {finish} | Tamano: {width_cm}x{height_cm} cm | Cantidad: {quantity}"
        email_sent = False
        if pdf_bytes:
            email_sent = send_order_email(
                pdf_bytes, customer_name, customer_email, customer_phone,
                specs, total_price, comment, wc_order_id,
            )

        print(f"[WEBHOOK] Order #{wc_order_id} — PDF: {pdf_bytes is not None}, Email: {email_sent}")

        return JSONResponse({"ok": True, "order_id": wc_order_id, "pdf_generated": pdf_bytes is not None, "email_sent": email_sent})

    except HTTPException:
        raise
    except Exception as e:
        print(f"[WEBHOOK] Error: {e}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
