from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from ultralytics import YOLO

import os
os.environ["ULTRALYTICS_HEIF"] = "0"

try:
    import cv2
except Exception:
    cv2 = None

# -----------------------------
# Config
# -----------------------------

DET_MODEL_PATH = "models/screw_detector.pt"
CLS_MODEL_PATH = "models/screw_classifier.pt"

DEVICE = "mps"  # (Mac için mps)

# -----------------------------
# App + Models (startup'ta yükle)
# -----------------------------
app = FastAPI(title="Screw Detect + Classify API", version="1.0.0")

det_model: Optional[YOLO] = None
cls_model: Optional[YOLO] = None

@app.on_event("startup")
def _load_models():
    global det_model, cls_model
    det_model = YOLO(DET_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "det_model": DET_MODEL_PATH,
        "cls_model": CLS_MODEL_PATH,
    }

# -----------------------------
# Helpers
# -----------------------------
def read_image(file_bytes: bytes) -> Image.Image:

    img = Image.open(io.BytesIO(file_bytes))

    # EXIF bilgisine göre resmi döndür (Fix Orientation)
    img = ImageOps.exif_transpose(img)
    
    return img.convert("RGB")

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)  # RGB uint8

def clip_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def crop_rgb(np_img: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = np_img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
    return np_img[y1:y2, x1:x2, :]

def head_square_crop(np_img: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = np_img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)

    bw = x2 - x1
    bh = y2 - y1

    side = int(max(bw, bh))
    if side < 2:
        return crop_rgb(np_img, (x1, y1, x2, y2))

    cx = int((x1 + x2) / 2)
    cy = int(y1 + bh * 0.45)

    pad = int(side * 0.05)
    side2 = side + 2 * pad

    nx1 = cx - side2
    ny1 = cy - side2
    nx2 = nx1 + side2
    ny2 = ny1 + side2

    nx1, ny1, nx2, ny2 = clip_box(nx1, ny1, nx2, ny2, w, h)
    return np_img[ny1:ny2, nx1:nx2, :]

def enhance_for_cls(crop: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return crop

    # 1. RGB -> LAB Dönüşümü
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 2. Sadece L (Işık) Kanalına CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 3. Kanalları Birleştir ve Geri RGB Yap
    limg = cv2.merge((cl, a, b))
    out = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return out

def classify_crop(
    crop: np.ndarray,
    imgsz: int = 1024,
    topk: int = 4,
) -> Dict[str, Any]:

    
    assert cls_model is not None

    res = cls_model.predict(
        source=crop,
        device=DEVICE,
        imgsz=imgsz,
        verbose=False,
    )[0]

    probs = res.probs
    if probs is None:
        return {"label": None, "confidence": None, "topk": []}

    p = probs.data.cpu().numpy().astype(float)
    idxs = np.argsort(-p)[:topk]
    names = res.names

    top_list = [
        {"label": names[int(i)], "confidence": float(p[int(i)])}
        for i in idxs
    ]
    best = top_list[0] if top_list else {"label": None, "confidence": None}

    return {"label": best["label"], "confidence": best["confidence"], "topk": top_list}

# -----------------------------
# Main endpoint
# -----------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    det_conf: float = Query(0.20, ge=0.0, le=1.0),
    det_iou: float = Query(0.70, ge=0.0, le=1.0),
    cls_imgsz: int = Query(1024, ge=64, le=1024),
    cls_topk: int = Query(3, ge=1, le=10),
    max_dets: int = Query(50, ge=1, le=500),
):


    global det_model, cls_model
    if det_model is None or cls_model is None:
        return JSONResponse({"ok": False, "error": "Models not loaded"}, status_code=500)
    t0 = time.time()

    raw = await file.read()
    pil_img = read_image(raw)
    
    debug_path = "debug_gelen_resim.jpg"
    pil_img.save(debug_path)

    print("PIL size:", pil_img.width, pil_img.height)
    
    np_img = pil_to_np(pil_img)

    # -------- Detection --------
    det_res = det_model.predict(
        source=debug_path,
        device=DEVICE,
        imgsz=1024,
        conf=det_conf,
        iou=det_iou,
        max_det=max_dets,
        verbose=False,
    )[0]
    
    print("YOLO orig_shape:", det_res.orig_shape)  # (h, w)
    if det_res.boxes is not None and len(det_res.boxes) > 0:
        print("First box xyxy:", det_res.boxes.xyxy[0].tolist(), "conf:", float(det_res.boxes.conf[0]))

    boxes = det_res.boxes
    out: List[Dict[str, Any]] = []

    if boxes is None or len(boxes) == 0:
        return {
            "ok": True,
            "filename": file.filename,
            "image_size": {"w": pil_img.width, "h": pil_img.height},
            "detections": [],
            "timing_ms": {"total": int((time.time() - t0) * 1000)},
        }

    names = det_res.names
    
    im = det_res.plot()
    im = im[..., ::-1]
    Image.fromarray(im).save("debug_pred.jpg")
    print("saved debug_pred.jpg")

    # For each detection crop + classify
    for j in range(len(boxes)):
    
        b = boxes[j]
        xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
        conf = float(b.conf[0].detach().cpu().numpy())
        cls_id = int(b.cls[0].detach().cpu().numpy()) if b.cls is not None else 0
        det_label = names.get(cls_id, str(cls_id))

        crop = head_square_crop(np_img, tuple(xyxy))
        crop = enhance_for_cls(crop)

        cls_out = classify_crop(crop, imgsz=cls_imgsz, topk=cls_topk)

        out.append(
            {
                "det": {
                    "label": det_label,
                    "confidence": conf,
                    "bbox_xyxy": [float(x) for x in xyxy],
                },
                "cls": cls_out,
            }
        )
    return {
        "ok": True,
        "filename": file.filename,
        "image_size": {"w": pil_img.width, "h": pil_img.height},
        "detections": out,
        "timing_ms": {"total": int((time.time() - t0) * 1000)},
    }
