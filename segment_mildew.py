# -*- coding: utf-8 -*-
"""
segment_mildew.py

Segmentación de mildiu polvoso sobre un CROP de hoja (RGB).
Devuelve un panel con varias vistas y el porcentaje de severidad.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.filters import rank
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from skimage.segmentation import morphological_chan_vese
from skimage.color import deltaE_ciede2000

AREA_MIN_HOJA_PX   = 3000
ENTROPY_RADIO      = 5
ENTROPY_PCT        = 45
CV_ITERS           = 150
MIN_OBJ_PX         = 80
FILL_HOLES_PX      = 120
WHITE_L_MIN = 190
WHITE_S_MAX = 60
D_PCT       = 70

def hoja_mask_from_green(bgr):
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  lower = np.array([25, 40, 40], dtype=np.uint8)
  upper = np.array([95, 255, 255], dtype=np.uint8)
  mask = cv2.inRange(hsv, lower, upper)
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
  return mask

def normalize_u8(img):
  return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def entropy_map(gray_u8):
  return rank.entropy(gray_u8, disk(ENTROPY_RADIO))

def retinex_like_L(l_chan):
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  return clahe.apply(l_chan)

def limpiar_mask(mask, min_obj=MIN_OBJ_PX, fill_holes_area=FILL_HOLES_PX):
  from skimage.util import img_as_bool
  m = img_as_bool(mask > 0)
  m = remove_small_objects(m, min_size=min_obj)
  m = remove_small_holes(m, area_threshold=fill_holes_area)
  return (m.astype(np.uint8) * 255)

def chan_vese_refine(gray_u8, init_mask):
  imgf = (gray_u8.astype(np.float32) / 255.0)
  init = (init_mask > 0)
  try:
    seg = morphological_chan_vese(
      imgf, num_iter=CV_ITERS, init_level_set=init,
      smoothing=1, lambda1=1, lambda2=1
    )
  except TypeError:
    seg = morphological_chan_vese(
      imgf, iterations=CV_ITERS, init_level_set=init,
      smoothing=1, lambda1=1, lambda2=1
    )
  return (seg.astype(np.uint8) * 255)

def deltaE2000_map(lab_img, mask):
  lab_ref = np.median(lab_img[mask == 255].reshape(-1, 3), axis=0)
  ref_img = np.full_like(lab_img, lab_ref)
  deltaE = deltaE_ciede2000(lab_img, ref_img)
  return normalize_u8(deltaE)

def _panel_to_array(rgb, Lr_u, ent_u, deltaE, seg, vis, porcentaje):
  fig = plt.figure(figsize=(10, 6))
  plt.subplot(2, 3, 1); plt.imshow(rgb);      plt.title("Hoja (verde)");          plt.axis("off")
  plt.subplot(2, 3, 2); plt.imshow(Lr_u, cmap="gray");     plt.title("Luminancia CLAHE");    plt.axis("off")
  plt.subplot(2, 3, 3); plt.imshow(ent_u, cmap="gray");    plt.title("Textura (Entropía)");  plt.axis("off")
  plt.subplot(2, 3, 4); plt.imshow(deltaE, cmap="inferno");plt.title("ΔE2000");             plt.axis("off")
  plt.subplot(2, 3, 5); plt.imshow(seg, cmap="gray");      plt.title("Máscara final");       plt.axis("off")
  plt.subplot(2, 3, 6); plt.imshow(vis);      plt.title(f"Resultado ({porcentaje:.2f}%)");   plt.axis("off")
  plt.tight_layout()

  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  panel = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  panel = panel.reshape(h, w, 3)
  plt.close(fig)
  return panel

def segmentar_lesiones(crop_rgb):
  crop_rgb = np.asarray(crop_rgb)
  if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
    raise ValueError("segmentar_lesiones espera una imagen RGB (H, W, 3).")

  bgr_full = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

  h, w = bgr_full.shape[:2]
  MAX_CROP_DIM = 800
  if max(h, w) > MAX_CROP_DIM:
    scale = MAX_CROP_DIM / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    bgr_full = cv2.resize(bgr_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

  leaf_mask = hoja_mask_from_green(bgr_full)

  if cv2.countNonZero(leaf_mask) < AREA_MIN_HOJA_PX:
    rgb_small = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2RGB)
    panel = _panel_to_array(rgb_small,
                            np.zeros_like(leaf_mask),
                            np.zeros_like(leaf_mask),
                            np.zeros_like(leaf_mask),
                            np.zeros_like(leaf_mask),
                            rgb_small,
                            0.0)
    return {"panel": panel, "severidad": 0.0}

  bgr = cv2.bitwise_and(bgr_full, bgr_full, mask=leaf_mask)

  lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
  L, A, B = cv2.split(lab)
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(hsv)

  Lr = retinex_like_L(L)
  Lr_u = normalize_u8(Lr)

  ent = entropy_map(Lr_u)
  ent_u = normalize_u8(ent)

  deltaE = deltaE2000_map(lab, leaf_mask)

  L_vals = Lr_u[leaf_mask == 255]
  E_vals = ent_u[leaf_mask == 255]
  D_vals = deltaE[leaf_mask == 255]

  if len(L_vals) == 0 or len(E_vals) == 0 or len(D_vals) == 0:
    rgb_small = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    panel = _panel_to_array(rgb_small, Lr_u, ent_u, deltaE,
                            np.zeros_like(leaf_mask), rgb_small, 0.0)
    return {"panel": panel, "severidad": 0.0}

  L_thr = np.percentile(L_vals, 80)
  E_thr = np.percentile(E_vals, ENTROPY_PCT)
  D_thr = np.percentile(D_vals, D_PCT)

  seeds = np.zeros_like(Lr_u, dtype=np.uint8)
  whiteness = (
      (Lr_u >= WHITE_L_MIN) &
      (S    <= WHITE_S_MAX) &
      (deltaE >= D_thr) &
      (leaf_mask == 255)
  )
  seeds[whiteness] = 255

  seg_raw = chan_vese_refine(Lr_u, seeds)
  seg = limpiar_mask(seg_raw)
  seg = cv2.bitwise_and(seg, leaf_mask)
  seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)

  area_hoja = cv2.countNonZero(leaf_mask)
  area_lesion = cv2.countNonZero(seg)
  porcentaje = (area_lesion / area_hoja * 100.0) if area_hoja > 0 else 0.0

  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  vis = rgb.copy()
  cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)

  panel = _panel_to_array(rgb, Lr_u, ent_u, deltaE, seg, vis, porcentaje)

  return {
      "panel": panel,
      "severidad": float(porcentaje)
  }
