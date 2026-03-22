# VERSION 35.3 - Order PDF + Email
# Changes from v35.2:
#   - New /generate-order endpoint: generates print-ready PDF + sends email
#   - PDF includes: sticker at real size, cut contour lines, registration marks, order info
#   - Email via SMTP with order specs + PDF attachment
#   - All v35.2 features preserved

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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageDraw

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

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
# EMAIL CONFIG (set via environment variables)
# ─────────────────────────────────────────────
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER     = os.environ.get("SMTP_USER", "")        # tu email
SMTP_PASS     = os.environ.get("SMTP_PASS", "")        # app password
OWNER_EMAIL   = os.environ.get("OWNER_EMAIL", "jeanmarco@gmail.com")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB = 15
MAX_DIMENSION_PX = 5000
PREVIEW_MAX_DIM = 1200          # downscale for preview generation
CACHE_TTL_SECONDS = 600         # 10 min
CACHE_MAX_ENTRIES = 50

VALID_FORMATS = {"image/png", "image/jpeg", "image/webp", "image/tiff", "image/bmp"}
VALID_MATERIALS = {"vinyl", "matte", "clear", "holographic", "kraft", "glitter", "mirror", "transparent", "reflective"}

# ─────────────────────────────────────────────
# CACHE (thread-safe, LRU + TTL)
# ─────────────────────────────────────────────
class StickerCache:
    """
    Stores the HEAVY part of the pipeline (padded design + sticker alpha mask)
    so that changing material/border only requires the LIGHT recompose step.
    """
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
            entry["ts"] = time.time()  # refresh on access
            return entry["data"]

    def put(self, file_hash: str, border_ratio: float, data: dict):
        key = self._make_key(file_hash, border_ratio)
        with self._lock:
            # evict oldest if full
            if len(self._store) >= self._max and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k]["ts"])
                del self._store[oldest_key]
            self._store[key] = {"data": data, "ts": time.time()}

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._store), "max": self._max, "ttl": self._ttl}


cache = StickerCache()

# ─────────────────────────────────────────────
# UTILITIES (unchanged from v34.6)
# ─────────────────────────────────────────────
def to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_rgba_from_bytes(data: bytes) -> Image.Image:
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
    """
    FIX from v34.6: now material-aware.
    For clear/transparent materials, we preserve semi-transparent pixels.
    For opaque materials (vinyl, matte, kraft), we composite against white.
    """
    arr = np.array(img).astype(np.uint8)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0

    # always kill near-invisible pixels
    very_low = arr[:, :, 3] < 12
    arr[:, :, :3][very_low] = 0

    # only flatten semi-transparent to white for opaque materials
    if material not in ("clear", "holographic"):
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
# BACKGROUND REMOVAL (unchanged)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# MASK GENERATION (unchanged core, parameterized border)
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
    """
    v35.1: Tighter kernels for merging nearby components.
    This prevents excessive white fill between letters like B-U-S-I-N-E-S-S.
    """
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
    # smaller kernels → components merge but don't inflate
    if total_w >= total_h * 1.4:
        kernel = make_ellipse_kernel(max(3, int(max_dim * 0.008)), max(3, int(max_dim * 0.004)))
    else:
        kernel = make_ellipse_kernel(max(3, int(max_dim * 0.006)), max(3, int(max_dim * 0.010)))
    return cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def chaikin_smooth(pts: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Chaikin's corner cutting algorithm — pure numpy, no scipy needed.
    Each iteration replaces each segment with two points at 25%/75%,
    progressively rounding corners into smooth curves.
    Works beautifully for sticker outlines.
    """
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
    """
    Takes a raw OpenCV contour and returns a smoothed version.
    Uses scipy cubic splines if available, otherwise falls back to
    Chaikin's corner cutting (pure numpy, same visual quality).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 6:
        return contour

    # --- Method 1: scipy cubic spline (best quality) ---
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
            pass  # fall through to Chaikin

    # --- Method 2: Chaikin corner cutting (no dependencies) ---
    # 5 iterations for extra smooth curves (each iteration doubles points)
    smooth_pts = chaikin_smooth(pts, iterations=5).astype(np.int32)
    return smooth_pts.reshape(-1, 1, 2)


def metaball_outline(mask: np.ndarray, border_px: int) -> np.ndarray:
    """
    v35.2: Tighter outline + anti-alias pre-smoothing.
    - Distance transform expansion for uniform border width
    - Two-pass Gaussian: first a tight pass for shape (0.30x), then a
      final anti-alias pass to eliminate pixel staircase before contour extraction
    - Lower threshold preserves concavities between letters
    """
    inv = 255 - mask
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    expanded = np.where(dist <= border_px, 255, 0).astype(np.uint8)

    # shape blur — tight, preserves concavities
    blur = max(3, int(border_px * 0.30))
    if blur % 2 == 0:
        blur += 1
    blurred = cv2.GaussianBlur(expanded, (blur, blur), 0)
    _, smooth = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)

    # close tiny gaps
    clean_k = max(3, int(border_px * 0.12))
    kernel = make_ellipse_kernel(clean_k)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ANTI-ALIAS PASS: blur the binary mask edge and re-threshold.
    # This converts the pixel staircase into a smooth gradient edge,
    # so findContours extracts a clean curve instead of jagged pixels.
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


VALID_SHAPES = {"contour-cut", "square", "circle", "oval", "rounded"}


def make_sticker_mask(alpha_mask: np.ndarray, border_ratio: float = 0.028) -> np.ndarray:
    """
    Contour-cut mask: organic border following the design shape.
    """
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


def make_shape_mask(alpha_mask: np.ndarray, shape: str, border_ratio: float = 0.028) -> np.ndarray:
    """
    Generates a geometric sticker mask (circle, square, oval, rounded).
    The shape is sized to fit the design with a uniform white border.
    For contour-cut, delegates to make_sticker_mask().
    """
    if shape == "contour-cut":
        return make_sticker_mask(alpha_mask, border_ratio)

    h, w = alpha_mask.shape
    max_dim = max(h, w)
    border_px = max(10, int(max_dim * border_ratio))
    mask = np.zeros((h, w), dtype=np.uint8)

    # find bounding box of the design
    ys, xs = np.where(alpha_mask > 0)
    if len(xs) == 0:
        return mask

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    dw = x2 - x1
    dh = y2 - y1

    if shape == "circle":
        # circle that fits design + border
        radius = int(max(dw, dh) / 2 + border_px)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    elif shape == "oval":
        # ellipse with border
        ax_w = int(dw / 2 + border_px)
        ax_h = int(dh / 2 + border_px)
        cv2.ellipse(mask, (cx, cy), (ax_w, ax_h), 0, 0, 360, 255, -1)

    elif shape == "square":
        # square centered on design + border
        half = int(max(dw, dh) / 2 + border_px)
        pt1 = (max(0, cx - half), max(0, cy - half))
        pt2 = (min(w - 1, cx + half), min(h - 1, cy + half))
        cv2.rectangle(mask, pt1, pt2, 255, -1)

    elif shape == "rounded":
        # rounded rectangle
        half_w = int(dw / 2 + border_px)
        half_h = int(dh / 2 + border_px)
        corner_r = max(8, int(min(half_w, half_h) * 0.25))
        pt1 = (max(0, cx - half_w), max(0, cy - half_h))
        pt2 = (min(w - 1, cx + half_w), min(h - 1, cy + half_h))
        # OpenCV doesn't have rounded rect, build with rectangles + circles
        rx1, ry1 = pt1
        rx2, ry2 = pt2
        # fill body
        cv2.rectangle(mask, (rx1 + corner_r, ry1), (rx2 - corner_r, ry2), 255, -1)
        cv2.rectangle(mask, (rx1, ry1 + corner_r), (rx2, ry2 - corner_r), 255, -1)
        # fill corners
        cv2.circle(mask, (rx1 + corner_r, ry1 + corner_r), corner_r, 255, -1)
        cv2.circle(mask, (rx2 - corner_r, ry1 + corner_r), corner_r, 255, -1)
        cv2.circle(mask, (rx1 + corner_r, ry2 - corner_r), corner_r, 255, -1)
        cv2.circle(mask, (rx2 - corner_r, ry2 - corner_r), corner_r, 255, -1)

    else:
        return make_sticker_mask(alpha_mask, border_ratio)

    # anti-alias the geometric shape edges
    aa_blur = max(3, int(border_px * 0.15))
    if aa_blur % 2 == 0:
        aa_blur += 1
    mask = cv2.GaussianBlur(mask, (aa_blur, aa_blur), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


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
    "holographic":  "holographic.png",
    "glitter":      "glitter.png",
    "kraft":        "kraft.png",
    "mirror":       "mirror.png",
    "reflective":   "reflective.png",
}

MATERIAL_BASE_COLOR = {
    "vinyl":       (255, 255, 255),
    "matte":       (245, 245, 245),
    "clear":       (255, 255, 255),  # handled specially with lower opacity
    "holographic": (255, 255, 255),
    "kraft":       (194, 164, 120),
    "glitter":     (255, 255, 255),
    "mirror":       (220, 220, 230),
    "transparent":  (255, 255, 255),  # handled like "clear"
    "reflective":   (200, 200, 210),
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
    """
    v35.2: StickerApp-style shadow.
    - Higher opacity (55 vs 15) for visible depth
    - Larger blur radius (28 vs 15) for soft diffuse spread
    - Offset slightly right + down (4, 8) matching SA's lighting angle
    - Uses the mask's actual alpha as shadow shape (not flat opacity)
    """
    h, w = mask.shape

    # expand canvas to accommodate shadow offset without clipping
    pad_x = abs(offset[0]) + blur_radius
    pad_y = abs(offset[1]) + blur_radius
    canvas_w = w + pad_x * 2
    canvas_h = h + pad_y * 2

    # place mask centered in padded canvas
    padded_mask = Image.new("L", (canvas_w, canvas_h), 0)
    padded_mask.paste(Image.fromarray(mask, "L"), (pad_x, pad_y))

    # blur to create soft shadow spread
    shadow_alpha = padded_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # scale alpha to target opacity
    sa = np.array(shadow_alpha).astype(np.float32)
    sa = sa * (opacity / 255.0)
    shadow_alpha = Image.fromarray(np.clip(sa, 0, 255).astype(np.uint8), "L")

    # build shadow RGBA
    shadow_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow_color = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    shadow_color.putalpha(shadow_alpha)
    shadow_rgba.alpha_composite(shadow_color, dest=offset)

    # crop back to original size
    result = shadow_rgba.crop((pad_x, pad_y, pad_x + w, pad_y + h))
    return result


def compose_final_preview(
    design_img: Image.Image,
    sticker_alpha: np.ndarray,
    material: str,
) -> tuple[Image.Image, Optional[str]]:
    """LIGHT step: composites design onto sticker shape with material appearance."""
    texture_path = None
    w, h = design_img.size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # shadow — SA-style: soft, visible, offset down-right
    shadow = create_shadow_from_mask(sticker_alpha, blur_radius=28, opacity=55, offset=(4, 8))
    canvas.alpha_composite(shadow)

    # material base
    texture, texture_path = load_texture(material, design_img.size)
    if texture is not None:
        mat_base = apply_alpha_mask(texture, sticker_alpha)
        canvas.alpha_composite(mat_base)
    elif material in ("clear", "transparent"):
        # simulate transparent vinyl: semi-transparent white
        clear_alpha = (sticker_alpha.astype(np.float32) * 0.35).astype(np.uint8)
        canvas.alpha_composite(make_rgba_from_alpha(clear_alpha, (255, 255, 255)))
    else:
        color = MATERIAL_BASE_COLOR.get(material, (255, 255, 255))
        canvas.alpha_composite(make_rgba_from_alpha(sticker_alpha, color))

    # design on top
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
        raise HTTPException(400, f"Unsupported format: {content_type}. Use PNG, JPEG, WebP, TIFF, or BMP.")


def validate_dimensions(img: Image.Image) -> None:
    w, h = img.size
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise HTTPException(
            400,
            f"Image too large: {w}x{h}px (max {MAX_DIMENSION_PX}px per side). "
            f"Consider resizing before upload."
        )


def downscale_for_preview(img: Image.Image, max_dim: int = PREVIEW_MAX_DIM) -> tuple[Image.Image, float]:
    """Returns (scaled_image, scale_factor). scale_factor=1.0 if no resize needed."""
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
    """
    Runs the expensive part: load → prepare → trim → pad → mask.
    Returns dict with padded_design (PIL), sticker_alpha (np), bg_method, fhash.
    Results are cached by file hash + border_ratio.
    """
    fhash = file_hash(raw_data)

    # check cache
    cached = cache.get(fhash, border_ratio)
    if cached is not None:
        return {**cached, "cache_hit": True}

    # full pipeline
    raw_img = load_rgba_from_bytes(raw_data)
    validate_dimensions(raw_img)

    prepared_img, bg_method = prepare_input_image(raw_img)
    design_trimmed = trim_transparent(prepared_img, padding_ratio=0.08)

    # downscale for preview if image is very large
    if for_preview:
        design_trimmed, scale = downscale_for_preview(design_trimmed)

    clean_design = sanitize_design_rgba(design_trimmed, material="vinyl")  # initial sanitize
    clean_design = antialias_rgba_edges(clean_design, sigma=0.9)
    padded_design = add_canvas_padding(clean_design, padding_ratio=0.08, min_px=40)

    alpha_mask = build_alpha_mask(padded_design)
    # default contour-cut mask for initial preview
    sticker_alpha = make_sticker_mask(alpha_mask, border_ratio=border_ratio)

    result = {
        "padded_design": padded_design,
        "alpha_mask": alpha_mask,          # raw design mask — needed to regenerate any shape
        "sticker_alpha": sticker_alpha,    # contour-cut by default
        "bg_method": bg_method,
        "fhash": fhash,
        "cache_hit": False,
    }

    # cache it
    cache.put(fhash, border_ratio, result)
    return result


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "version": "35.3"}


@app.get("/cache-stats")
def cache_stats():
    """Monitor cache usage."""
    return cache.stats()


@app.post("/process-sticker")
async def process_sticker(
    file: UploadFile = File(...),
    material: str = Form("vinyl"),
    shape: str = Form("contour-cut"),
    border_ratio: float = Form(0.028),
):
    """
    Full pipeline: upload → mask (cached) → shape → compose.
    Returns preview + cache key for fast recompose.
    """
    if material not in VALID_MATERIALS:
        raise HTTPException(400, f"Unknown material: {material}")
    if shape not in VALID_SHAPES:
        shape = "contour-cut"

    border_ratio = max(0.02, min(0.12, border_ratio))

    try:
        data = await file.read()
        validate_upload(data, file.content_type)

        # HEAVY (cached) — always generates contour-cut + stores raw alpha_mask
        heavy = run_heavy_pipeline(data, border_ratio=border_ratio, for_preview=True)

        # generate shape-specific mask if not contour-cut
        if shape == "contour-cut":
            sticker_alpha = heavy["sticker_alpha"]
        else:
            sticker_alpha = make_shape_mask(heavy["alpha_mask"], shape, border_ratio)

        # LIGHT compose
        final_preview, texture_path = compose_final_preview(
            heavy["padded_design"], sticker_alpha, material
        )

        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "cache_key": heavy["fhash"],
            "border_ratio": border_ratio,
            "material": material,
            "shape": shape,
            "debug_version": "35.3",
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
    """
    FAST endpoint: recomposes from cached data.
    Supports changing material AND shape without re-uploading.
    Shape changes regenerate the mask from cached alpha_mask (~20ms).
    """
    if material not in VALID_MATERIALS:
        raise HTTPException(400, f"Unknown material: {material}")
    if shape not in VALID_SHAPES:
        shape = "contour-cut"

    border_ratio = max(0.02, min(0.12, border_ratio))

    cached = cache.get(cache_key, border_ratio)
    if cached is None:
        raise HTTPException(
            404,
            "Cache miss — design not found. Please re-upload via /process-sticker."
        )

    try:
        # generate shape mask
        if shape == "contour-cut":
            sticker_alpha = cached["sticker_alpha"]
        else:
            sticker_alpha = make_shape_mask(cached["alpha_mask"], shape, border_ratio)

        final_preview, texture_path = compose_final_preview(
            cached["padded_design"], sticker_alpha, material
        )
        return JSONResponse({
            "ok": True,
            "final_preview_png": to_base64(final_preview),
            "cache_key": cache_key,
            "border_ratio": border_ratio,
            "material": material,
            "shape": shape,
            "debug_version": "35.3",
            "debug_cache_hit": True,
            "debug_texture_found": texture_path is not None,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()},
        )


# ─────────────────────────────────────────────
# PERSISTENT DESIGN STORAGE
# Saves uploaded designs so they survive past the
# 10-min in-memory cache. When WooCommerce confirms
# payment, the webhook uses the order_token to
# retrieve the design and generate the PDF.
# ─────────────────────────────────────────────
DESIGNS_DIR = BASE_DIR / "saved_designs"
DESIGNS_DIR.mkdir(exist_ok=True)


def save_design_to_disk(order_token: str, file_bytes: bytes, meta: dict):
    """Saves design image + metadata JSON to disk."""
    folder = DESIGNS_DIR / order_token
    folder.mkdir(exist_ok=True)
    (folder / "design.png").write_bytes(file_bytes)
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_design_from_disk(order_token: str) -> tuple[bytes, dict] | tuple[None, None]:
    """Loads design image bytes + metadata from disk."""
    folder = DESIGNS_DIR / order_token
    design_path = folder / "design.png"
    meta_path = folder / "meta.json"
    if not design_path.exists() or not meta_path.exists():
        return None, None
    file_bytes = design_path.read_bytes()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return file_bytes, meta


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
    """
    Called by frontend when customer clicks 'Add to Cart'.
    Saves the design file + all specs to disk with a unique token.
    The token gets stored in WooCommerce order meta.
    NO PDF, NO email — that happens after payment.
    """
    try:
        data = await file.read()
        validate_upload(data, file.content_type)

        # generate unique token
        fhash = file_hash(data)
        token = f"{fhash}_{int(time.time())}"

        # save to disk
        meta = {
            "material": material,
            "shape": shape,
            "finish": finish,
            "width_cm": width_cm,
            "height_cm": height_cm,
            "quantity": quantity,
            "total_price": total_price,
            "comment": comment,
            "border_ratio": border_ratio,
            "created_at": datetime.now().isoformat(),
        }
        save_design_to_disk(token, data, meta)

        return JSONResponse({
            "ok": True,
            "order_token": token,
            "debug_version": "35.3",
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()},
        )


# ─────────────────────────────────────────────
# PRINT-READY PDF GENERATION
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
    design_img: Image.Image,
    sticker_alpha: np.ndarray,
    width_cm: float,
    height_cm: float,
    quantity: int,
    shape: str,
    material: str,
    finish: str,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    comment: str,
    total_price: str,
    wc_order_id: str = "",
) -> bytes:
    """
    PDF listo para plotter:
    - Pagina 1: Resumen del pedido (cliente + specs)
    - Pagina 2: Sticker a tamano real + linea de corte roja + marcas de registro
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed")

    buf = BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    # ── PAGE 1: ORDER SUMMARY ──
    c.setFont("Helvetica-Bold", 22)
    c.drawString(2 * cm, page_h - 3 * cm, "Orden de Stickers")
    c.setFont("Helvetica", 10)
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    order_label = f"Fecha: {date_str}"
    if wc_order_id:
        order_label += f"  |  Pedido WC #{wc_order_id}"
    c.drawString(2 * cm, page_h - 3.6 * cm, order_label)

    y = page_h - 5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Datos del Cliente")
    y -= 0.7 * cm
    c.setFont("Helvetica", 11)
    for label, val in [("Nombre", customer_name), ("Email", customer_email), ("Telefono", customer_phone)]:
        if val:
            c.drawString(2.5 * cm, y, f"{label}: {val}")
            y -= 0.55 * cm

    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2 * cm, y, "Especificaciones")
    y -= 0.7 * cm
    c.setFont("Helvetica", 11)

    shape_names = {"contour-cut": "Contour Cut", "square": "Cuadrado", "circle": "Circulo", "oval": "Ovalo", "rounded": "Esquinas redondeadas"}
    material_names = {"vinyl": "Vinilo", "holographic": "Holografico", "transparent": "Transparente", "glitter": "Glitter", "mirror": "Espejo", "reflective": "Reflectivo", "matte": "Mate", "clear": "Clear", "kraft": "Kraft"}
    finish_names = {"glossy": "Laminado Brillante", "matte": "Laminado Mate"}

    for label, val in [
        ("Forma", shape_names.get(shape, shape)),
        ("Material", material_names.get(material, material)),
        ("Acabado", finish_names.get(finish, finish)),
        ("Tamano", f"{width_cm} x {height_cm} cm"),
        ("Cantidad", str(quantity)),
        ("Total", total_price),
    ]:
        c.drawString(2.5 * cm, y, f"{label}: {val}")
        y -= 0.55 * cm

    if comment:
        y -= 0.5 * cm
        c.setFont("Helvetica-Bold", 13)
        c.drawString(2 * cm, y, "Comentario")
        y -= 0.7 * cm
        c.setFont("Helvetica", 11)
        words = comment.split()
        line = ""
        for word in words:
            if c.stringWidth(line + " " + word, "Helvetica", 11) > 15 * cm:
                c.drawString(2.5 * cm, y, line)
                y -= 0.5 * cm
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            c.drawString(2.5 * cm, y, line)

    # preview thumbnail
    y -= 1.5 * cm
    try:
        thumb = design_img.copy()
        thumb.thumbnail((300, 300), Image.Resampling.LANCZOS)
        c.drawImage(ImageReader(thumb), 2.5 * cm, y - thumb.size[1] * 0.24,
                     thumb.size[0] * 0.24, thumb.size[1] * 0.24, mask='auto')
    except Exception:
        pass

    c.showPage()

    # ── PAGE 2: PRINT SHEET (real size for plotter) ──
    sticker_w_pts = width_cm * cm
    sticker_h_pts = height_cm * cm
    margin = 1.5 * cm
    mark_len = 0.4 * cm
    x_off = (page_w - sticker_w_pts) / 2
    y_off = (page_h - sticker_h_pts) / 2

    # registration marks
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.3)
    for cx_pt, cy_pt in [
        (x_off - margin, y_off - margin),
        (x_off + sticker_w_pts + margin, y_off - margin),
        (x_off - margin, y_off + sticker_h_pts + margin),
        (x_off + sticker_w_pts + margin, y_off + sticker_h_pts + margin),
    ]:
        c.line(cx_pt - mark_len, cy_pt, cx_pt + mark_len, cy_pt)
        c.line(cx_pt, cy_pt - mark_len, cx_pt, cy_pt + mark_len)

    # sticker at real size
    try:
        c.drawImage(ImageReader(design_img), x_off, y_off, sticker_w_pts, sticker_h_pts, mask='auto')
    except Exception:
        pass

    # cut contour (red dashed)
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
def send_order_email(
    pdf_bytes: bytes,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    specs_summary: str,
    total_price: str,
    comment: str,
    wc_order_id: str = "",
):
    if not SMTP_USER or not SMTP_PASS:
        print("[EMAIL] SMTP not configured — skipping")
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    att = MIMEApplication(pdf_bytes, _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename=f"sticker_order_{wc_order_id}_{timestamp}.pdf")
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
# WEBHOOK: WooCommerce order.paid
# ─────────────────────────────────────────────
# Configure in WooCommerce > Settings > Advanced > Webhooks:
#   Topic: Order updated (status: processing)
#   Delivery URL: https://your-api.up.railway.app/webhook/order-paid
#   Secret: your WEBHOOK_SECRET env var
# ─────────────────────────────────────────────
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def verify_wc_webhook(body: bytes, signature: str) -> bool:
    """Verify WooCommerce webhook HMAC-SHA256 signature."""
    if not WEBHOOK_SECRET:
        return True  # no secret = dev mode, accept all
    import hmac as hmac_mod
    expected = base64.b64encode(
        hmac_mod.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).digest()
    ).decode()
    return hmac_mod.compare_digest(expected, signature)


from fastapi import Request


@app.post("/webhook/order-paid")
async def webhook_order_paid(request: Request):
    """
    Called by WooCommerce AFTER payment is confirmed.
    Flow:
    1. Verifies webhook signature
    2. Extracts order_token from order meta
    3. Loads saved design from disk
    4. Runs sticker pipeline → generates PDF
    5. Sends email to owner with PDF attachment
    """
    body = await request.body()

    # verify signature
    signature = request.headers.get("x-wc-webhook-signature", "")
    if not verify_wc_webhook(body, signature):
        raise HTTPException(401, "Invalid webhook signature")

    try:
        order = json.loads(body)

        # WooCommerce sends the full order object
        wc_order_id = str(order.get("id", ""))
        status = order.get("status", "")

        # only process paid/processing orders
        if status not in ("processing", "completed"):
            return JSONResponse({"ok": True, "skipped": True, "reason": f"status={status}"})

        # extract customer data from WooCommerce order
        billing = order.get("billing", {})
        customer_name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()
        customer_email = billing.get("email", "")
        customer_phone = billing.get("phone", "")

        # find order_token in line items meta
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
            return JSONResponse({"ok": True, "skipped": True, "reason": "no sticker order_token"})

        # load design from disk
        file_bytes, saved_meta = load_design_from_disk(order_token)
        if file_bytes is None:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": f"Design not found for token {order_token}"}
            )

        # use saved meta (from /save-design) with WC meta as fallback
        meta = saved_meta
        material = meta.get("material", stk_meta.get("stk_material", "vinyl"))
        shape = meta.get("shape", stk_meta.get("stk_shape", "contour-cut"))
        finish = meta.get("finish", stk_meta.get("stk_finish", "glossy"))
        width_cm = float(meta.get("width_cm", stk_meta.get("stk_width", 8)))
        height_cm = float(meta.get("height_cm", stk_meta.get("stk_height", 8)))
        quantity = int(meta.get("quantity", stk_meta.get("stk_quantity", 50)))
        total_price = meta.get("total_price", stk_meta.get("stk_total", ""))
        comment = meta.get("comment", stk_meta.get("stk_comment", ""))
        border_ratio = float(meta.get("border_ratio", 0.028))

        # run sticker pipeline
        heavy = run_heavy_pipeline(file_bytes, border_ratio=border_ratio, for_preview=True)
        final_preview, _ = compose_final_preview(
            heavy["padded_design"], heavy["sticker_alpha"], material
        )

        # generate PDF
        pdf_bytes = None
        if HAS_REPORTLAB:
            pdf_bytes = generate_print_pdf(
                design_img=final_preview,
                sticker_alpha=heavy["sticker_alpha"],
                width_cm=width_cm, height_cm=height_cm,
                quantity=quantity, shape=shape,
                material=material, finish=finish,
                customer_name=customer_name,
                customer_email=customer_email,
                customer_phone=customer_phone,
                comment=comment, total_price=total_price,
                wc_order_id=wc_order_id,
            )

        # send email
        specs_summary = (
            f"Forma: {shape} | Material: {material} | Acabado: {finish} | "
            f"Tamano: {width_cm} x {height_cm} cm | Cantidad: {quantity}"
        )
        email_sent = False
        if pdf_bytes:
            email_sent = send_order_email(
                pdf_bytes=pdf_bytes,
                customer_name=customer_name,
                customer_email=customer_email,
                customer_phone=customer_phone,
                specs_summary=specs_summary,
                total_price=total_price,
                comment=comment,
                wc_order_id=wc_order_id,
            )

        print(f"[WEBHOOK] Order #{wc_order_id} processed — PDF: {pdf_bytes is not None}, Email: {email_sent}")

        return JSONResponse({
            "ok": True,
            "order_id": wc_order_id,
            "pdf_generated": pdf_bytes is not None,
            "email_sent": email_sent,
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[WEBHOOK] Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "trace": traceback.format_exc()},
        )
