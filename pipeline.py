# pipeline.py - PowderyVision7 simplificado
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from segment_mildew import segmentar_lesiones

CLS_MODEL_PATH = "models/yolo11m-cls_best.pt"
SEG_MODEL_PATH = "models/yolo11n_leaf_seg_last.pt"

CLASS_INFO = {
    "pea_healthy": {
        "species": "pea", "crop_name_es": "Arveja",
        "status": "healthy", "status_es": "sana",
        "disease": None, "disease_es": None
    },
    "pea_powderymildew": {
        "species": "pea", "crop_name_es": "Arveja",
        "status": "diseased", "status_es": "enferma",
        "disease": "powdery mildew", "disease_es": "mildiu polvoso"
    },
    "tomato_healthy": {
        "species": "tomato", "crop_name_es": "Tomate",
        "status": "healthy", "status_es": "sana",
        "disease": None, "disease_es": None
    },
    "tomato_powderymildew": {
        "species": "tomato", "crop_name_es": "Tomate",
        "status": "diseased", "status_es": "enferma",
        "disease": "powdery mildew", "disease_es": "mildiu polvoso"
    },
    "background": {
        "species": None, "crop_name_es": None,
        "status": "background", "status_es": "sin_hoja"
    }
}

CLASES_SANAS = {"pea_healthy", "tomato_healthy"}
CLASE_BACKGROUND = "background"
CONF_MIN_LEAF = 0.50

# Cargar modelos una sola vez
cls_model = YOLO(CLS_MODEL_PATH)
seg_model = YOLO(SEG_MODEL_PATH)

def run_full_pipeline(bgr_image, save_root="static/results", prefix="img"):
    os.makedirs(save_root, exist_ok=True)
    h, w = bgr_image.shape[:2]

    # Guardar original
    orig_path = os.path.join(save_root, f"{prefix}_orig.png")
    cv2.imwrite(orig_path, bgr_image)

    # ----- 1) CLASIFICACIÓN -----
    t0 = time.time()
    cls_results = cls_model(bgr_image, verbose=False)
    t1 = time.time()
    pred_cls = cls_results[0]

    top_idx = int(pred_cls.probs.top1)
    top_conf = float(pred_cls.probs.top1conf)
    cls_name = cls_model.names[top_idx]
    cls_time_ms = (t1 - t0) * 1000

    info = CLASS_INFO.get(cls_name, {})
    salida = {
        "ok": True,
        "cls_name": cls_name,
        "cls_conf": top_conf,
        "cls_time_ms": cls_time_ms,
        "species": info.get("species"),
        "crop_name_es": info.get("crop_name_es"),
        "status": info.get("status"),
        "status_es": info.get("status_es"),
        "disease": info.get("disease"),
        "disease_es": info.get("disease_es"),
        "orig_img_path": "/" + orig_path.replace("\\", "/"),
        "crop_img_path": None,
        "seg_vis_path": None,
        "panel_img_path": None,
        "seg_ok": False,
        "leaf_conf": 0.0,
        "seg_time_ms": 0.0,
        "severidad_pct": 0.0,
        "is_healthy": False,
    }

    if cls_name == CLASE_BACKGROUND:
        salida["ok"] = False
        salida["msg"] = "No se detectó una hoja en la imagen (clase background)."
        return salida

    if cls_name in CLASES_SANAS:
        salida["is_healthy"] = True
        return salida

    # ----- 2) SEGMENTACIÓN HOJA -----
    t2 = time.time()
    seg_results = seg_model(bgr_image, verbose=False)
    t3 = time.time()
    salida["seg_time_ms"] = (t3 - t2) * 1000

    pred = seg_results[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        salida["ok"] = False
        salida["msg"] = "No se pudo detectar la hoja para hacer el crop."
        return salida

    confs = pred.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_box = pred.boxes[best_idx]
    leaf_conf = float(best_box.conf.cpu().numpy())
    salida["leaf_conf"] = leaf_conf

    if leaf_conf < CONF_MIN_LEAF:
        salida["ok"] = False
        salida["msg"] = "La hoja fue detectada con baja confianza."
        return salida

    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop_bgr = bgr_image[y1:y2, x1:x2]
    crop_path = os.path.join(save_root, f"{prefix}_crop.png")
    cv2.imwrite(crop_path, crop_bgr)
    salida["crop_img_path"] = "/" + crop_path.replace("\\", "/")
    salida["seg_ok"] = True

    # ----- 3) SEGMENTACIÓN MILDIU EN CROP -----
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    seg_data = segmentar_lesiones(crop_rgb)
    panel = seg_data["panel"]
    severidad = float(seg_data["severidad"])

    # guardar panel y visualización
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    panel_path = os.path.join(save_root, f"{prefix}_panel.png")
    cv2.imwrite(panel_path, panel_bgr)
    salida["panel_img_path"] = "/" + panel_path.replace("\\", "/")

    # también podemos generar una imagen de la hoja con contornos (ya viene en panel)
    salida["seg_vis_path"] = salida["panel_img_path"]
    salida["severidad_pct"] = severidad
    salida["is_healthy"] = False

    return salida
