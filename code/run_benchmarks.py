"""
Run Five slicing methods over a single input (local MPG/MP4 or RTSP URL),
collect detections with YOLO, measure device utilization and FPS via
'hailortcli monitor' (if running in Hailo environment), and compute Precision/Recall/F1 against a ground truth


Methods:
- full: resize whole frame to 640x640 and infer
- pad: resize whole frame to fit within 640x640 with aspect ratio preserved and letterbox padding
- tiled: tile frame into 640x640 windows with stride (default 320x320)
- motion: motion-based 640x640 crops using frame differencing
- motion_MOG2: motion-based 640x640 crops using MOG2 background subtraction

Outputs (in output_root):
- full/ *.txt  
- pad/ *.txt 
- tiled/ *.txt  
- motion/ *.txt 
- MOG2/ *.txt 
- metrics summary printed to console
- metrics summary saved to text file in root output folder
"""

import os
import sys
import time
import signal
import subprocess
from typing import Dict, Tuple, List, Set, Any
from collections import Counter
import math
try:
    import cv2
    # Report which OpenCV library was loaded to confirm environment
    try:
        print(f"[INFO] Using OpenCV from: {getattr(cv2, '__file__', 'unknown path')}")
    except Exception:
        pass
except ImportError:
    # Helpful diagnostics to detect environment mismatch
    print("[ERROR] OpenCV (cv2) not found. Install with: pip install opencv-contrib-python")
    print(f"[DIAG] Python executable: {sys.executable}")
    print(f"[DIAG] sys.path (first 5): {sys.path[:5]}")
    print("[DIAG] If you have multiple Python versions, ensure you installed into the one running this script:")
    print(f"       {sys.executable} -m pip install opencv-contrib-python")
    sys.exit(1)
import numpy as np
# ------------------------------- Hailo Engine -------------------------------
from AI_Engine import init_engine, run_inference, HailoInferenceEngine  # use existing engine module
from AI_Engine import initialize  # noqa: E402
from datetime import datetime, timezone
import csv
import infer_utils


# ================================ CONSTANTS ==================================
# Optional: load overrides from parameters.env
def _load_env_params(env_path: str) -> dict:
    params = {}
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
        pass
    return params

# This script is meant for comparing the five slicing methods on the same input and ground truth.
# There are additional steps needed to use this with an RTSP stream.
# That approach is not implemented here, but is used in other scripts where each method
# is implemented separately.
# TODO: Pass the class Names for the hef in use if different from default COCO classes.


_ENV = _load_env_params(os.path.join(os.path.dirname(__file__), 'parameters.env'))
MODEL_PATH: str = _ENV.get('MODEL_PATH', _ENV.get('HEF_PATH', "yolo11m.pt"))
INPUT_SRC: str = _ENV.get('INPUT_SRC', "your path here")
GT_FOLDER: str = _ENV.get('GT_FOLDER', "your path here")
OUTPUT_ROOT: str = _ENV.get('OUTPUT_ROOT', "your path here")
DEFAULT_INFERENCE_SIZE: Tuple[int, int] = (
    int(_ENV.get('DEFAULT_INFERENCE_WIDTH', '640')),
    int(_ENV.get('DEFAULT_INFERENCE_HEIGHT', '640'))
)
CONFIDENCE_THRESHOLD: float = float(_ENV.get('CONFIDENCE_THRESHOLD', '0.60'))
TILE_STRIDE: int = int(_ENV.get('TILE_STRIDE', '320'))
DUP_MIN_DISTANCE: float = float(_ENV.get('DUP_MIN_DISTANCE', '50.0'))
MOTION_MIN_CONTOUR_AREA: int = int(_ENV.get('MOTION_MIN_CONTOUR_AREA', '500'))
MOTION_DIFF_THRESHOLD: int = int(_ENV.get('MOTION_DIFF_THRESHOLD', '50'))
GAUSSIAN_BLUR_KERNEL: int = int(_ENV.get('GAUSSIAN_BLUR_KERNEL', '21'))
MOG2_KERNEL_SIZE: int = int(_ENV.get('MOG2_KERNEL_SIZE', '7'))
MOG2_HISTORY: int = int(_ENV.get('MOG2_HISTORY', '10'))
MOG2_VAR_THRESHOLD: int = int(_ENV.get('MOG2_VAR_THRESHOLD', _ENV.get('MOTION_DIFF_THRESHOLD', '5000')))

# Hailo monitor header signature from HailoRTCLI output
MONITOR_HEADER: str = (
    "Model                                                       Utilization (%)          FPS            PID            "
)





# -------------------------- Hailo monitor Setup --------------------------

def start_hailo_monitor(log_path: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    # Start hailortcli monitor; requires proper permissions
    proc = subprocess.Popen(
        ["hailortcli", "monitor"], stdout=f, stderr=subprocess.STDOUT, text=True
    )
    return proc

def stop_hailo_monitor(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass

    #Parse hailortcli monitor log and return (avg_utilization, avg_fps).
def parse_hailo_monitor_file(path: str) -> Tuple[float, float]:


    utils: List[float] = []
    fps_list: List[float] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            it = iter(f)
            for line in it:
                if MONITOR_HEADER in line:
                    _ = next(it, None)  # skip separator line
                    ln2 = next(it, None)  # data line
                    if not ln2:
                        continue
                    toks = ln2.strip().split()
                    if len(toks) >= 3:
                        try:
                            utils.append(float(toks[1]))
                            fps_list.append(float(toks[2]))
                        except (ValueError, IndexError):
                            pass
    except FileNotFoundError:
        pass
    avg_util = sum(utils)/len(utils) if utils else 0.0
    avg_fps = sum(fps_list)/len(fps_list) if fps_list else 0.0
    return avg_util, avg_fps

# ----------------------------- Slicing methods -----------------------------

def infer_full(src: str, out_dir: str, eng: HailoInferenceEngine):
    # Delegate to shared utils; defaults come from parameters.env
    return infer_utils.infer_full(src, out_dir, eng)
def infer_full_pad(src: str, out_dir: str, eng: HailoInferenceEngine):
    return infer_utils.infer_full_pad(src, out_dir, eng)

def infer_tiled(src: str, out_dir: str, eng: HailoInferenceEngine, stride: int = TILE_STRIDE):
    # Pass stride override; other params default via env
    return infer_utils.infer_tiled(src, out_dir, eng, stride)

def infer_motion(src: str, out_dir: str, eng: HailoInferenceEngine):
    # All parameters default from env in infer_utils
    return infer_utils.infer_motion(src, out_dir, eng)

def infer_motion_MOG2(src: str, out_dir: str, eng: HailoInferenceEngine):
    # All parameters default from env in infer_utils
    return infer_utils.infer_motion_MOG2(src, out_dir, eng)



#These are the metrics functions that compute and save precision, recall, and F1 score

def list_txt_files(folder: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in os.listdir(folder):
        if not name.lower().endswith('.txt'):
            continue
        base = os.path.splitext(name)[0]
        out[base] = os.path.abspath(os.path.join(folder, name))
    return out

def read_class_set(path: str) -> Set[str]:
    classes: Set[str] = set()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            first = s.split(None, 1)[0]
            classes.add(first)
    return classes

def _read_class_counts(path: str) -> Counter:
    """Read a YOLO detection/ground-truth txt file and count multiplicity of class IDs."""
    counts: Counter = Counter()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if not parts:
                continue
            cid = parts[0]
            counts[cid] += 1
    return counts

def compute_metrics(det_folder: str, gt_folder: str) -> Tuple[int, int, int, float, float, float, int]:
        """Compute metrics by exact filename match (intersection of frame bases).

        Assumes detection and ground-truth folders contain the same base filenames.
        Returns: tp_total, fp_total, fn_total, precision, recall, f1, shared_frame_count
        """
        map_det = list_txt_files(det_folder)
        map_gt = list_txt_files(gt_folder)
        keys_det = set(map_det.keys())
        keys_gt = set(map_gt.keys())
        shared = keys_det & keys_gt

        # Diagnostic if mismatch
        if keys_det != keys_gt:
            missing_in_det = list(sorted(keys_gt - keys_det))[:10]
            missing_in_gt = list(sorted(keys_det - keys_gt))[:10]
            if missing_in_det:
                print(f"[WARN] GT-only frames detected (sample): {missing_in_det}")
            if missing_in_gt:
                print(f"[WARN] Det-only frames detected (sample): {missing_in_gt}")

        tp_total = fp_total = fn_total = 0
        for base in shared:
            det_counts = _read_class_counts(map_det[base])
            gt_counts = _read_class_counts(map_gt[base])
            all_classes = set(det_counts.keys()) | set(gt_counts.keys())
            for c in all_classes:
                dc = det_counts.get(c, 0)
                gc = gt_counts.get(c, 0)
                tp_c = min(dc, gc)
                tp_total += tp_c
                fp_total += max(0, dc - gc)
                fn_total += max(0, gc - dc)

        prec_den = tp_total + fp_total
        rec_den = tp_total + fn_total
        precision = (tp_total / prec_den) if prec_den > 0 else 0.0
        recall = (tp_total / rec_den) if rec_den > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return tp_total, fp_total, fn_total, precision, recall, f1, len(shared)



def _count_total_gt_detections(gt_folder: str) -> int:
    """Sum the total number of labeled objects across all ground-truth .txt files."""
    total = 0
    try:
        for name in os.listdir(gt_folder):
            if not name.lower().endswith('.txt'):
                continue
            path = os.path.join(gt_folder, name)
            counts = _read_class_counts(path)
            total += sum(counts.values())
    except Exception:
        # If folder missing or read error, return 0
        pass
    return total

def save_metrics_report(out_root: str, entries: List[Dict[str, Any]], model_name: str,
                        inference_size: Tuple[int, int] = DEFAULT_INFERENCE_SIZE,
                        conf_threshold: float = CONFIDENCE_THRESHOLD,
                        total_gt_detections: int = 0):
    """Save aggregated metrics for each slicing method to a text file in out_root.
    Includes model name, inference resolution (W,H) and other constants that affect the run.
    TODO: Make the Gaussian, Median, and Bilateral filter parameters configurable and add them here.
    This is to make the runs reproducible and comparable.
    """
    path = os.path.join(out_root, "metrics_summary.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("Benchmark Metrics Summary\n")
            f.write("=================================\n\n")
            f.write(f"Model: {model_name}\n")
            # Add input filepath and filename
            f.write(f"Input Source Path: {INPUT_SRC}\n")
            f.write(f"Input Source Name: {os.path.basename(INPUT_SRC)}\n")
            f.write(f"Inference Resolution: {inference_size[0]}x{inference_size[1]}\n")
            f.write(f"Confidence Threshold: {conf_threshold:.2f}\n\n")
            f.write(f"TILE_STRIDE: {TILE_STRIDE}\n")
            f.write(f"MOTION_MIN_CONTOUR_AREA: {MOTION_MIN_CONTOUR_AREA}\n")
            f.write(f"MOTION_DIFF_THRESHOLD: {MOTION_DIFF_THRESHOLD}\n")
            f.write(f"Duplicate Min Distance (tiling & motion): {DUP_MIN_DISTANCE}\n")
            f.write(f"Total Ground-Truth Detections: {total_gt_detections}\n\n")
            f.write(f"GAUSSIAN_BLUR_KERNEL: {GAUSSIAN_BLUR_KERNEL}\n")
            try:
                f.write(f"MOG2_HISTORY: {MOG2_HISTORY}\n")
            except NameError:
                pass
            try:
                f.write(f"MOG2_VAR_THRESHOLD: {MOG2_VAR_THRESHOLD}\n")
            except NameError:
                pass
            f.write(f"MOG2_KERNEL_SIZE: {MOG2_KERNEL_SIZE}\n\n")

            # Tab-separated metrics table
            # f.write("Method\tFiles\tTP\tFP\tFN\tPrec\tRecall\tF1\tUtil\tHailo FPS\tProc FPS\n")
            f.write("Method\tFiles\tTP\tFP\tFN\tPrec\tRecall\tF1\tUtil\tHailo FPS\tProc FPS\n")
            for e in entries:
                f.write(
                    f"{e['name']}\t{e['shared']}\t{e['tp']}\t{e['fp']}\t{e['fn']}\t"
                    f"{e['precision']:.4f}\t{e['recall']:.4f}\t{e['f1']:.4f}\t"
                    f"{e['utilization']:.2f}\t{e['hailo_fps']:.2f}\t\t{e['proc_fps']:.2f}\n"
                )
    except Exception as exc:
        print(f"Failed to write metrics summary file: {exc}")
    else:
        print(f"Metrics summary saved to: {path}")



def _collect_constants_for_csv() -> Dict[str, Any]:
    """Gather constants from parameters.env for CSV logging."""
    return {
        "MODEL_PATH": MODEL_PATH,
        "INPUT_SRC": INPUT_SRC,
        "GT_FOLDER": GT_FOLDER,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "TILE_STRIDE": TILE_STRIDE,
        "DUP_MIN_DISTANCE": DUP_MIN_DISTANCE,
        "MOTION_MIN_CONTOUR_AREA": MOTION_MIN_CONTOUR_AREA,
        "MOTION_DIFF_THRESHOLD": MOTION_DIFF_THRESHOLD,
        "GAUSSIAN_BLUR_KERNEL": GAUSSIAN_BLUR_KERNEL,
        "MOG2_KERNEL_SIZE": MOG2_KERNEL_SIZE,
        "MOG2_HISTORY": globals().get("MOG2_HISTORY", None),
        "MOG2_VAR_THRESHOLD": globals().get("MOG2_VAR_THRESHOLD", None),
    }

def append_metrics_csv(csv_path: str, constants: Dict[str, Any], metrics_entries: List[Dict[str, Any]], timestamp: str) -> None:
    """
    Append one CSV row per method containing:
    timestamp + selected constants + precision/recall/F1 and other metrics.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    const_cols = [
        "MODEL_PATH","INPUT_SRC","GT_FOLDER","OUTPUT_ROOT",
        "CONFIDENCE_THRESHOLD","TILE_STRIDE","DUP_MIN_DISTANCE",
        "MOTION_MIN_CONTOUR_AREA","MOTION_DIFF_THRESHOLD",
        "GAUSSIAN_BLUR_KERNEL","MOG2_KERNEL_SIZE","MOG2_HISTORY","MOG2_VAR_THRESHOLD",
    ]
    run_cols = ["method","files_counted","tp","fp","fn","precision","recall","f1","proc_fps"]
    header = ["timestamp"] + const_cols + run_cols

    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for e in metrics_entries:
            row = {"timestamp": timestamp}
            row.update({k: constants.get(k, "") for k in const_cols})
            row.update({
                "method": e.get("name",""),
                "files_counted": e.get("shared", 0),
                "tp": e.get("tp", 0),
                "fp": e.get("fp", 0),
                "fn": e.get("fn", 0),
                "precision": f'{e.get("precision", 0.0):.6f}',
                "recall": f'{e.get("recall", 0.0):.6f}',
                "f1": f'{e.get("f1", 0.0):.6f}',
                "proc_fps": f'{e.get("proc_fps", 0.0):.6f}',
            })
            w.writerow(row)

def run_with_monitor(run_fn, src: str, out_dir: str, eng: HailoInferenceEngine) -> Tuple[float, float, float]:
    os.makedirs(out_dir, exist_ok=True)
    is_hailo = isinstance(eng, HailoInferenceEngine)
    monitor_log = os.path.join(out_dir, "hailo_monitor.txt")
    proc = None

    if is_hailo:
        proc = start_hailo_monitor(monitor_log)

    try:
        frames, elapsed = run_fn(src, out_dir, eng)
    finally:
        if proc is not None:
            stop_hailo_monitor(proc)

    if is_hailo:
        util, hailo_fps = parse_hailo_monitor_file(monitor_log)
    else:
        util, hailo_fps = 0.0, 0.0

    processing_fps = (frames / elapsed) if elapsed > 0 else 0.0
    return util, hailo_fps, processing_fps

def main() -> int:

    # Start total runtime timer
    t0 = time.perf_counter()

    if not os.path.exists(INPUT_SRC):
        print(f"Input source does not exist or is not reachable: {INPUT_SRC}")
        return 2
    if not os.path.isdir(GT_FOLDER):
        print(f"Ground truth folder not found: {GT_FOLDER}")
        return 2

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = os.path.join(OUTPUT_ROOT, timestamp)
    out_full = os.path.join(out_root, "full")
    out_full_pad = os.path.join(out_root, "full_pad")
    out_tiled = os.path.join(out_root, "tiled")
    out_motion = os.path.join(out_root, "motion")
    out_mog2 = os.path.join(out_root, "MOG2")
    csv_path = os.path.join(OUTPUT_ROOT, f"benchmark_runs.csv")
    # Initialize engine via probe to detect backend and handle errors
    status = initialize(hef_path=MODEL_PATH)

    eng = status.get("engine")
    if eng is None:
        print("[ERROR] No inference engine available. Provide a valid .onnx or .pt/.pth model when Hailo is unavailable.")
        return 3

    if not eng.wait_ready(timeout=15.0, poll=0.2, do_warmup=True):
        print("Inference engine not ready within timeout.")
        return 3

    print("Running motion-crop method...")
    util_motion, hailo_motion_fps, proc_motion_fps = run_with_monitor(infer_motion, INPUT_SRC, out_motion, eng)

    print("Running motion MOG2 method...")
    util_mog2, hailo_mog2_fps, proc_mog2_fps = run_with_monitor(infer_motion_MOG2, INPUT_SRC, out_mog2, eng)

    print("Running full-frame method...")
    util_full, hailo_full_fps, proc_full_fps = run_with_monitor(infer_full, INPUT_SRC, out_full, eng)

    print("Running full-frame method with padding to preserve aspect ratio...")
    util_full_pad, hailo_full_pad_fps, proc_full_pad_fps = run_with_monitor(infer_full_pad, INPUT_SRC, out_full_pad, eng)

    print("Running tiled method...")
    util_tiled, hailo_tiled_fps, proc_tiled_fps = run_with_monitor(lambda s, d, e: infer_tiled(s, d, e, stride=320), INPUT_SRC, out_tiled, eng)

    # Compute metrics against ground truth
    print("\nMetrics vs ground truth (instance counts per class):")
    # Console TSV header to match file output
    print("Method\tFiles Counted\tTP\tFP\tFN\tPrecision\tRecall\tF1\tUtilization\tHailo FPS\tProcessing FPS")
    metrics_entries: List[Dict[str, Any]] = []
    for name, folder, util, hailo_fps, proc_fps in [
        ("full", out_full, util_full, hailo_full_fps, proc_full_fps),
        ("pad", out_full_pad, util_full_pad, hailo_full_pad_fps, proc_full_pad_fps),
        ("tiled", out_tiled, util_tiled, hailo_tiled_fps, proc_tiled_fps),
        ("mot", out_motion, util_motion, hailo_motion_fps, proc_motion_fps),
        ("MOG2", out_mog2, util_mog2, hailo_mog2_fps, proc_mog2_fps),
    ]:
        tp, fp, fn, prec, rec, f1, shared = compute_metrics(folder, GT_FOLDER)
        # Console TSV row
        print(f"{name}\t{shared}\t{tp}\t{fp}\t{fn}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{util:.2f}\t{hailo_fps:.2f}\t{proc_fps:.2f}")
        metrics_entries.append({
            "name": name, "shared": shared, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "utilization": util, "hailo_fps": hailo_fps, "proc_fps": proc_fps,
        })

    # Save report a report with all of the metrics we have collected
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    total_gt = _count_total_gt_detections(GT_FOLDER)

    save_metrics_report(out_root, metrics_entries, model_name,
                        inference_size=DEFAULT_INFERENCE_SIZE,
                        conf_threshold=CONFIDENCE_THRESHOLD,
                        total_gt_detections=total_gt)

    # Append per-method metrics to CSV
    constants = _collect_constants_for_csv()

 
    # Strict ISO 8601 with timezone offset (e.g., 2025-01-02T03:04:05+00:00)
    end_ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    append_metrics_csv(csv_path, constants, metrics_entries, timestamp=end_ts)

    eng.close()
    print(f"\nOutputs written to: {out_root}")

    # Stop timer and report total runtime
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss), {elapsed:.2f} seconds")

    return 0

# Using remap_bbox_from_detection_space from infer_utils

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
