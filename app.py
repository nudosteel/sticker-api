# Sticker Wizard - Version 2

from io import BytesIO
import base64
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

def to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

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

def fill_holes(img):
    arr = np.array(img)
    alpha = arr[:, :, 3]

    solid = np.where(alpha > 0, 255, 0).astype(np.uint8)
    inv = 255 - solid

    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    h, w = solid.shape

    border = set()
    border.update(np.unique(labels[0, :]))
    border.update(np.unique(labels[h-1, :]))
    border.update(np.unique(labels[:, 0]))
    border.update(np.unique(labels[:, w-1]))

    for i in range(1, num):
        if i in border:
            continue

        if stats[i, cv2.CC_STAT_AREA] < 10000:
            solid[labels == i] = 255

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = 255
    out[:, :, 3] = solid

    return Image.fromarray(out, "RGBA")

@app.get("/")
def root():
    return {"ok": True}

@app.post("/process-sticker")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(BytesIO(data)).convert("RGBA")

    design = trim(img)
    contour = fill_holes(design)

    return JSONResponse({
        "ok": True,
        "design_png": to_base64(design),
        "contour_png": to_base64(contour)
    })
