import os
import time
from typing import Dict, Tuple, List, Any, Optional

import cv2
import numpy as np

from AI_Engine import run_inference

# Utility helpers (self-contained to avoid circular imports)

# --------------------------- Environment handling ---------------------------

_ENV_CACHE: Optional[Dict[str, str]] = None


def _load_env_params(env_path: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '=' in s:
                    k, v = s.split('=', 1)
                    params[k.strip()] = v.strip()
    except Exception:
        # Silently ignore if no env file
        pass
    return params


def _get_env() -> Dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        env_path = os.path.join(os.path.dirname(__file__), 'parameters.env')
        _ENV_CACHE = _load_env_params(env_path)
    return _ENV_CACHE


def _env_get(key: str, default: str) -> str:
    return _get_env().get(key, default)


def _env_get_int(key: str, default: int) -> int:
    try:
        return int(_env_get(key, str(default)))
    except Exception:
        return int(default)


def _env_get_float(key: str, default: float) -> float:
    try:
        return float(_env_get(key, str(default)))
    except Exception:
        return float(default)

def open_video_capture(src: str):
    cap = cv2.VideoCapture(src)
    return cap, cap.isOpened()


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]


def write_yolo_txt(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def calculate_euclidean_distance(x1, y1, x2, y2):
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def prepare_inference_crop(crop: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Only resize if larger than target; if smaller, do not upscale, just pad to target.
    target = DEFAULT_INFERENCE_WIDTH = _env_get_int('DEFAULT_INFERENCE_WIDTH', 640)
    orig_h, orig_w = crop.shape[:2]

    scaled = False
    if orig_h > target or orig_w > target:
        scale = min(target / float(orig_w), target / float(orig_h))
        new_w = max(1, min(target, int(round(orig_w * scale))))
        new_h = max(1, min(target, int(round(orig_h * scale))))
        prepared = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled = True
    else:
        new_w, new_h = orig_w, orig_h
        prepared = crop

    left = (target - new_w) // 2 if new_w < target else 0
    right = target - new_w - left if new_w < target else 0
    top = (target - new_h) // 2 if new_h < target else 0
    bottom = target - new_h - top if new_h < target else 0
    if left or right or top or bottom:
        prepared = cv2.copyMakeBorder(prepared, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    ph, pw = prepared.shape[:2]
    if ph != target or pw != target:
        extra_bottom = max(0, target - ph)
        extra_right = max(0, target - pw)
        prepared = cv2.copyMakeBorder(prepared, 0, extra_bottom, 0, extra_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    scale_x = (new_w / float(orig_w)) if scaled and orig_w > 0 else 1.0
    scale_y = (new_h / float(orig_h)) if scaled and orig_h > 0 else 1.0

    padded = (left != 0) or (right != 0) or (top != 0) or (bottom != 0) or (ph != target) or (pw != target)
    pad_info: Dict[str, Any] = {
        'orig_w': orig_w,
        'orig_h': orig_h,
        'scaled_w': new_w,
        'scaled_h': new_h,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'left_pad': left,
        'top_pad': top,
        'resized': bool(scaled or padded),
        'letterboxed': bool(padded),
    }
    return prepared, pad_info


def prepare_inference_pad(crop: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Resize to fit within target (640x640 by default), preserving aspect ratio.
    # Upscale when smaller and downscale when larger; then letterbox to target.
    target = DEFAULT_INFERENCE_WIDTH = _env_get_int('DEFAULT_INFERENCE_WIDTH', 640)
    orig_h, orig_w = crop.shape[:2]

    scaled = False
    if orig_w > 0 and orig_h > 0:
        scale = min(target / float(orig_w), target / float(orig_h))
    else:
        scale = 1.0

    if scale != 1.0 or orig_h != target or orig_w != target:
        new_w = max(1, min(target, int(round(orig_w * scale))))
        new_h = max(1, min(target, int(round(orig_h * scale))))
        interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        prepared = cv2.resize(crop, (new_w, new_h), interpolation=interp)
        scaled = True
    else:
        new_w, new_h = orig_w, orig_h
        prepared = crop

    left = (target - new_w) // 2 if new_w < target else 0
    right = target - new_w - left if new_w < target else 0
    top = (target - new_h) // 2 if new_h < target else 0
    bottom = target - new_h - top if new_h < target else 0
    if left or right or top or bottom:
        prepared = cv2.copyMakeBorder(prepared, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    ph, pw = prepared.shape[:2]
    if ph != target or pw != target:
        extra_bottom = max(0, target - ph)
        extra_right = max(0, target - pw)
        prepared = cv2.copyMakeBorder(prepared, 0, extra_bottom, 0, extra_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    scale_x = (new_w / float(orig_w)) if scaled and orig_w > 0 else 1.0
    scale_y = (new_h / float(orig_h)) if scaled and orig_h > 0 else 1.0

    padded = (left != 0) or (right != 0) or (top != 0) or (bottom != 0) or (ph != target) or (pw != target)
    pad_info: Dict[str, Any] = {
        'orig_w': orig_w,
        'orig_h': orig_h,
        'scaled_w': new_w,
        'scaled_h': new_h,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'left_pad': left,
        'top_pad': top,
        'resized': bool(scaled or padded),
        'letterboxed': bool(padded),
    }
  
    return prepared, pad_info


def remap_bbox_from_detection_space(
    bbox: Tuple[float, float, float, float],
    pad_info: Dict[str, Any],
    origin_x: float,
    origin_y: float
) -> Tuple[float, float, float, float]:
    xx1, yy1, xx2, yy2 = bbox
    scale_x = pad_info.get('scale_x', 1.0)
    scale_y = pad_info.get('scale_y', 1.0)
    left_pad = pad_info.get('left_pad', 0)
    top_pad = pad_info.get('top_pad', 0)
    is_letterboxed = bool(pad_info.get('letterboxed', False))
    is_resized = bool(pad_info.get('resized', False))

    sx1, sx2, sy1, sy2 = float(xx1), float(xx2), float(yy1), float(yy2)
    if is_letterboxed:
        sx1 -= left_pad; sx2 -= left_pad
        sy1 -= top_pad;  sy2 -= top_pad
    if is_resized:
        local_x1 = sx1 / scale_x; local_x2 = sx2 / scale_x
        local_y1 = sy1 / scale_y; local_y2 = sy2 / scale_y
    else:
        local_x1, local_x2 = sx1, sx2
        local_y1, local_y2 = sy1, sy2

    ow = float(pad_info.get('orig_w', 0))
    oh = float(pad_info.get('orig_h', 0))
    local_x1 = max(0.0, min(ow - 1.0, local_x1))
    local_x2 = max(0.0, min(ow - 1.0, local_x2))
    local_y1 = max(0.0, min(oh - 1.0, local_y1))
    local_y2 = max(0.0, min(oh - 1.0, local_y2))

    global_x1 = origin_x + local_x1
    global_y1 = origin_y + local_y1
    global_x2 = origin_x + local_x2
    global_y2 = origin_y + local_y2
    return global_x1, global_y1, global_x2, global_y2


# ---------------------- Contour merging (NMS-like) ----------------------

def _rect_iou(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _rect_center_distance(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    cx1 = x1 + w1 / 2.0
    cy1 = y1 + h1 / 2.0
    cx2 = x2 + w2 / 2.0
    cy2 = y2 + h2 / 2.0
    return calculate_euclidean_distance(cx1, cy1, cx2, cy2)


def merge_nearby_contours(
    contours: List[np.ndarray],
    max_merge_distance: float = _env_get_float('DUP_MIN_DISTANCE', 200.0),
    iou_threshold: float = 0.1,
    min_group_area: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """Merge nearby/overlapping contours into larger bounding rectangles.

    - Builds bounding rects for each input contour.
    - Connects rects if center-distance <= max_merge_distance OR IoU >= iou_threshold.
    - Returns one merged rect per connected component by taking the union
      bounding rectangle.

    Parameters:
      contours: list of cv2 contours (each shape (N,1,2)).
      max_merge_distance: max center-to-center distance to consider neighbors.
      iou_threshold: minimum IoU to consider neighbors.
      min_group_area: optional minimum merged area; groups below are discarded.

    Returns:
      List of merged rectangles as (x, y, w, h).
    """
    if not contours:
        return []
    
    rects: List[Tuple[int, int, int, int]] = [cv2.boundingRect(c) for c in contours]
    n = len(rects)
    if n == 1:
        x, y, w, h = rects[0]
        if min_group_area is not None and (w * h) < int(min_group_area):
            return []
        return [rects[0]]

    # Build adjacency graph
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _rect_center_distance(rects[i], rects[j])
            iou = _rect_iou(rects[i], rects[j])
            if (d <= max_merge_distance) or (iou >= iou_threshold):
                adj[i].append(j)
                adj[j].append(i)

    # Connected components via Breadth-First Search
    seen = [False] * n
    merged: List[Tuple[int, int, int, int]] = []
    #TODO switch to deque for performance? Sounds easy but this is working
    #TODO consider Depth-First Search if performance is an issue
    for i in range(n):
        if seen[i]:
            continue
        queue = [i]
        seen[i] = True
        comp_indices = [i]
        while queue:
            k = queue.pop(0)
            for nb in adj[k]:
                if not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
                    comp_indices.append(nb)
        # Merge component rects
        xs1 = []
        ys1 = []
        xs2 = []
        ys2 = []
        for idx in comp_indices:
            x, y, w, h = rects[idx]
            xs1.append(x)
            ys1.append(y)
            xs2.append(x + w)
            ys2.append(y + h)
        mx1 = min(xs1)
        my1 = min(ys1)
        mx2 = max(xs2)
        my2 = max(ys2)
        mw = max(0, mx2 - mx1)
        mh = max(0, my2 - my1)
        if min_group_area is not None and (mw * mh) < int(min_group_area):
            continue
        merged.append((mx1, my1, mw, mh))

    return merged


# ----------------------------- Inference methods -----------------------------


def infer_full(src: str, out_dir: str, eng, conf_threshold: Optional[float] = None) -> Tuple[int, float]:
    if conf_threshold is None:
        conf_threshold = _env_get_float('CONFIDENCE_THRESHOLD', 0.60)
    cap, opened = open_video_capture(src)
    if not opened:
        raise RuntimeError(f"Could not open video source: {src}")
    start_t = time.time()
    frames = 0
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            frames += 1
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            base = os.path.join(out_dir, str(frame_number))
            frame_for_full = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            dets = run_inference(frame_for_full, conf_threshold=conf_threshold, engine=eng)
            lines: List[str] = []
            h_img, w_img = frame_for_full.shape[:2]
            for d in dets or []:
                bbox = d.get('bbox'); cid = d.get('class_id')
                if bbox:
                    x1, y1, x2, y2 = bbox
                    xc, yc, ww, hh = pascal_voc_to_yolo(x1, y1, x2, y2, w_img, h_img)
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            write_yolo_txt(base + ".txt", lines)
    finally:
        cap.release()
    return frames, max(1e-9, time.time() - start_t)


def infer_full_pad(src: str, out_dir: str, eng, conf_threshold: Optional[float] = None) -> Tuple[int, float]:
    if conf_threshold is None:
        conf_threshold = _env_get_float('CONFIDENCE_THRESHOLD', 0.60)
    cap, opened = open_video_capture(src)
    if not opened:
        raise RuntimeError(f"Could not open video source: {src}")
    start_t = time.time()
    frames = 0
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            frames += 1
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            base = os.path.join(out_dir, str(frame_number))
            frame_for_full, pad_info = prepare_inference_pad(frame)
            kernel = np.array([[-1, -1, -1],
                   [-1,  13, -1],
                   [-1, -1, -1]])
            frame_for_full = cv2.filter2D(frame_for_full, -1, kernel) 
            dets = run_inference(frame_for_full, conf_threshold=conf_threshold, engine=eng)
            lines: List[str] = []
            h_orig, w_orig = frame.shape[:2]
            for d in dets or []:
                bbox = d.get('bbox'); cid = d.get('class_id')
                if not bbox:
                    continue
                xx1, yy1, xx2, yy2 = bbox
                gx1, gy1, gx2, gy2 = remap_bbox_from_detection_space(
                    (xx1, yy1, xx2, yy2), pad_info, origin_x=0, origin_y=0
                )
                xc, yc, ww, hh = pascal_voc_to_yolo(gx1, gy1, gx2, gy2, w_orig, h_orig)
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            write_yolo_txt(base + ".txt", lines)
    finally:
        cap.release()
    return frames, max(1e-9, time.time() - start_t)


def infer_tiled(
    src: str,
    out_dir: str,
    eng,
    stride: Optional[int] = None,
    conf_threshold: Optional[float] = None,
    dup_min_distance: Optional[float] = None,
) -> Tuple[int, float]:
    if stride is None:
        stride = _env_get_int('TILE_STRIDE', 320)
    if conf_threshold is None:
        conf_threshold = _env_get_float('CONFIDENCE_THRESHOLD', 0.60)
    if dup_min_distance is None:
        dup_min_distance = _env_get_float('DUP_MIN_DISTANCE', 50.0)
    cap, opened = open_video_capture(src)
    if not opened:
        raise RuntimeError(f"Could not open video source: {src}")
    start_t = time.time()
    frames = 0
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            frames += 1
            h, w = frame.shape[:2]
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            base = os.path.join(out_dir, str(frame_number))
            lines: List[str] = []
            accepted: List[Tuple[int, float, float]] = []
            for y in range(0, h + stride, stride):
                for x in range(0, w + stride, stride):
                    xe = min(x + 640, w)
                    ye = min(y + 640, h)
                    original_tile = frame[y:ye, x:xe]
                    if original_tile is None or original_tile.size == 0:
                        continue
                    inf_tile, pad_info = prepare_inference_pad(original_tile)
                    dets = run_inference(inf_tile, conf_threshold=conf_threshold, engine=eng)
                    for d in dets or []:
                        cid = d.get('class_id'); bbox = d.get('bbox')
                        if not bbox:
                            continue
                        xx1, yy1, xx2, yy2 = bbox
                        global_x1, global_y1, global_x2, global_y2 = remap_bbox_from_detection_space(
                            (xx1, yy1, xx2, yy2), pad_info, origin_x=x, origin_y=y
                        )
                        candidate_cx = (global_x1 + global_x2) / 2.0
                        candidate_cy = (global_y1 + global_y2) / 2.0
                        too_close = False
                        for ac_cid, ac_cx, ac_cy in accepted:
                            if ac_cid != cid:
                                continue
                            dist = calculate_euclidean_distance(candidate_cx, candidate_cy, ac_cx, ac_cy)
                            if dist <= dup_min_distance:
                                too_close = True
                                break
                        if not too_close:
                            accepted.append((cid, candidate_cx, candidate_cy))
                            xc, yc, ww, hh = pascal_voc_to_yolo(global_x1, global_y1, global_x2, global_y2, w, h)
                            lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            write_yolo_txt(base + ".txt", lines)
    finally:
        cap.release()
    return frames, max(1e-9, time.time() - start_t)


def infer_motion(
    src: str,
    out_dir: str,
    eng,
    conf_threshold: Optional[float] = None,
    gaussian_blur_kernel: Optional[int] = None,
    motion_diff_threshold: Optional[int] = None,
    motion_min_contour_area: Optional[int] = None,
    dup_min_distance: Optional[float] = None,
) -> Tuple[int, float]:
    if conf_threshold is None:
        conf_threshold = _env_get_float('CONFIDENCE_THRESHOLD', 0.60)
    if gaussian_blur_kernel is None:
        gaussian_blur_kernel = _env_get_int('GAUSSIAN_BLUR_KERNEL', 21)
    if motion_diff_threshold is None:
        motion_diff_threshold = _env_get_int('MOTION_DIFF_THRESHOLD', 50)
    if motion_min_contour_area is None:
        motion_min_contour_area = _env_get_int('MOTION_MIN_CONTOUR_AREA', 500)
    if dup_min_distance is None:
        dup_min_distance = _env_get_float('DUP_MIN_DISTANCE', 50.0)
    cap, opened = open_video_capture(src)
    if not opened:
        raise RuntimeError(f"Could not open video source: {src}")
    start_t = time.time()
    frames = 0
    k = int(gaussian_blur_kernel)
    if k < 1:
        k = 1
    if (k % 2) == 0:
        k += 1

    try:
        frame1 = read_frame(cap)
        if frame1 is None:
            return 0, 0.0
        frames += 1
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (k, k), 0)
        while True:
            frame2 = read_frame(cap)
            if frame2 is None:
                break
            frames += 1
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            base = os.path.join(out_dir, str(frame_number))
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (k, k), 0)
            delta = cv2.absdiff(gray1, gray2)
            thresh = cv2.threshold(delta, int(motion_diff_threshold), 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, (k, k), iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines: List[str] = []
            accepted: List[Tuple[int, float, float]] = []
            merged_rects = merge_nearby_contours(
                contours=contours,
                max_merge_distance= dup_min_distance,
                iou_threshold=0.1,
                min_group_area=motion_min_contour_area,
            )
            for (x, y, w, h) in merged_rects:
                if (w * h) <= int(motion_min_contour_area):
                    continue
                fr = frame2.copy()
                if w >= 640 and h >= 640:
                    x1 = max(x, 0); y1 = max(y, 0)
                    x2 = min(x + w, fr.shape[1]); y2 = min(y + h, fr.shape[0])
                else:
                    cx = x + w // 2; cy = y + h // 2
                    half = 320
                    x1 = max(cx - half, 0); y1 = max(cy - half, 0)
                    x2 = min(cx + half, fr.shape[1]); y2 = min(cy + half, fr.shape[0])
                original_crop = fr[y1:y2, x1:x2]
                if original_crop is None or original_crop.size == 0:
                    continue
                inf_crop, pad_info = prepare_inference_pad(original_crop)

                dets = run_inference(inf_crop, conf_threshold=conf_threshold, engine=eng)
                frame_h, frame_w = frame2.shape[:2]
                for d in dets or []:
                    cid = d.get('class_id'); bbox = d.get('bbox')
                    if not bbox:
                        continue
                    xx1, yy1, xx2, yy2 = bbox
                    global_x1, global_y1, global_x2, global_y2 = remap_bbox_from_detection_space(
                        (xx1, yy1, xx2, yy2), pad_info, origin_x=x1, origin_y=y1
                    )
                    candidate_cx = (global_x1 + global_x2) / 2.0
                    candidate_cy = (global_y1 + global_y2) / 2.0
                    too_close = False
                    for ac_cid, ac_cx, ac_cy in accepted:
                        if ac_cid != cid:
                            continue
                        dist = calculate_euclidean_distance(candidate_cx, candidate_cy, ac_cx, ac_cy)
                        if dist <= dup_min_distance:
                            too_close = True
                            break
                    if not too_close:
                        accepted.append((cid, candidate_cx, candidate_cy))
                        xc, yc, ww, hh = pascal_voc_to_yolo(global_x1, global_y1, global_x2, global_y2, frame_w, frame_h)
                        lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            write_yolo_txt(base + ".txt", lines)
    finally:
        cap.release()
    return frames, max(1e-9, time.time() - start_t)


def infer_motion_MOG2(
    src: str,
    out_dir: str,
    eng,
    conf_threshold: Optional[float] = None,
    mog2_history: Optional[int] = None,
    mog2_var_threshold: Optional[int] = None,
    mog2_kernel_size: Optional[int] = None,
    motion_diff_threshold: Optional[int] = None,
    motion_min_contour_area: Optional[int] = None,
    dup_min_distance: Optional[float] = None,
) -> Tuple[int, float]:
    if conf_threshold is None:
        conf_threshold = _env_get_float('CONFIDENCE_THRESHOLD', 0.60)
    if mog2_history is None:
        mog2_history = _env_get_int('MOG2_HISTORY', 1)
    if mog2_var_threshold is None:
        mog2_var_threshold = _env_get_int('MOG2_VAR_THRESHOLD', _env_get_int('MOTION_DIFF_THRESHOLD', 50))
    if mog2_kernel_size is None:
        mog2_kernel_size = _env_get_int('MOG2_KERNEL_SIZE', 7)
    if motion_diff_threshold is None:
        motion_diff_threshold = _env_get_int('MOTION_DIFF_THRESHOLD', 50)
    if motion_min_contour_area is None:
        motion_min_contour_area = _env_get_int('MOTION_MIN_CONTOUR_AREA', 500)
    if dup_min_distance is None:
        dup_min_distance = _env_get_float('DUP_MIN_DISTANCE', 50.0)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=int(mog2_history), varThreshold=int(mog2_var_threshold), detectShadows=False
    )
    cap, opened = open_video_capture(src)
    if not opened:
        raise RuntimeError(f"Could not open video source: {src}")
    start_t = time.time()
    frames = 0
    k = int(mog2_kernel_size)
    if k < 1:
        k = 1
    if (k % 2) == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            frames += 1
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            base = os.path.join(out_dir, str(frame_number))

            fgmask = subtractor.apply(frame)
            _, thresh = cv2.threshold(fgmask, int(motion_diff_threshold), 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lines: List[str] = []
            accepted: List[Tuple[int, float, float]] = []
            frame_h, frame_w = frame.shape[:2]

            merged_rects = merge_nearby_contours(
                contours=contours,
                max_merge_distance=50,#_env_get_float('DUP_MIN_DISTANCE', 50.0),
                iou_threshold=0.3,
                min_group_area=motion_min_contour_area,
            )

            for (x, y, w, h) in merged_rects:
                if (w * h) <= int(motion_min_contour_area):
                    continue
                fr = frame
                if w >= 640 and h >= 640:
                    x1 = max(x, 0); y1 = max(y, 0)
                    x2 = min(x + w, fr.shape[1]); y2 = min(y + h, fr.shape[0])
                else:
                    cx = x + w // 2; cy = y + h // 2
                    half = 320
                    x1 = max(cx - half, 0); y1 = max(cy - half, 0)
                    x2 = min(cx + half, fr.shape[1]); y2 = min(cy + half, fr.shape[0])
                original_crop = fr[y1:y2, x1:x2]
                if original_crop is None or original_crop.size == 0:
                    continue
                inf_crop, pad_info = prepare_inference_pad(original_crop)

                dets = run_inference(inf_crop, conf_threshold=conf_threshold, engine=eng)
                for d in dets or []:
                    cid = d.get('class_id'); bbox = d.get('bbox')
                    if not bbox:
                        continue
                    xx1, yy1, xx2, yy2 = bbox
                    global_x1, global_y1, global_x2, global_y2 = remap_bbox_from_detection_space(
                        (xx1, yy1, xx2, yy2), pad_info, origin_x=x1, origin_y=y1
                    )
                    candidate_cx = (global_x1 + global_x2) / 2.0
                    candidate_cy = (global_y1 + global_y2) / 2.0
                    too_close = False
                    for ac_cid, ac_cx, ac_cy in accepted:
                        if ac_cid != cid:
                            continue
                        if calculate_euclidean_distance(candidate_cx, candidate_cy, ac_cx, ac_cy) <= dup_min_distance:
                            too_close = True
                            break
                    if not too_close:
                        accepted.append((cid, candidate_cx, candidate_cy))
                        xc, yc, ww, hh = pascal_voc_to_yolo(global_x1, global_y1, global_x2, global_y2, frame_w, frame_h)
                        lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            write_yolo_txt(base + ".txt", lines)
    finally:
        cap.release()
    return frames, max(1e-9, time.time() - start_t)
