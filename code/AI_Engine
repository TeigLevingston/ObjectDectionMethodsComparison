import time
import atexit
import os
from typing import List, Dict, Any, Optional

try:
    import cv2  # provided by opencv-contrib-python
except ImportError:
    cv2 = None
    print("[WARN] OpenCV not found. Install with: pip install opencv-contrib-python")
import numpy as np
try:
    import hailo_platform as hpf
except ImportError:
    hpf = None
    print("[WARN] Hailo Platform SDK not found. Ensure it is installed and accessible.")

MODEL_PATH_DEFAULT = "yolo11m.pt"
# Rename COCO80 to CLASS_NAMES
CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


def parse_hailo_yolo_output(raw_out, frame_shape, class_names: List[str], conf_threshold: float = 0.60) -> List[Dict[str, Any]]:
    """
    Normalize Hailo YOLO postprocess outputs into a list of detections:
      {class_id, class_name, score, bbox=[x1,y1,x2,y2]}
    """
    H_img, W_img = frame_shape[:2]

    # Model output: raw_out[0] is a list of 80 COCO classes; each item is (5, K)
    out = raw_out[0]

    dets: List[Dict[str, Any]] = []
    for cls_id, bucket in enumerate(out):
        if bucket is None:
            continue
        if not (hasattr(bucket, "ndim") and bucket.ndim == 2):
            continue

        # Expect shape (5, K) after transpose
        aT = bucket.T
        y1, x1, y2, x2, score = aT
        keep = (score >= conf_threshold) & (x2 > x1) & (y2 > y1)
        if not np.any(keep):
            continue

        y1 = y1[keep]; x1 = x1[keep]; y2 = y2[keep]; x2 = x2[keep]; score = score[keep]

        # Hailo YOLO outputs normalized boxes in [0,1], convert to pixel coords
        if max(float(np.max(x2, initial=0)), float(np.max(y2, initial=0))) <= 1.5:
            x1 = (x1 * W_img).astype(int); x2 = (x2 * W_img).astype(int)
            y1 = (y1 * H_img).astype(int); y2 = (y2 * H_img).astype(int)
        else:
            x1 = x1.astype(int); x2 = x2.astype(int); y1 = y1.astype(int); y2 = y2.astype(int)

        for xi1, yi1, xi2, yi2, sc in zip(x1, y1, x2, y2, score):
            dets.append({
                "class_id": int(cls_id),
                "class_name": class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id),
                "score": float(sc),
                "bbox": [int(xi1), int(yi1), int(xi2), int(yi2)]
            })
    return dets


# New: check HAILO architecture availability
def is_hailo_available() -> bool:
    """Return True if Hailo platform is available and a VDevice can be created."""
    if hpf is None:
        print("[INFO] Hailo SDK not imported. Using non-Hailo backend if a valid model is provided.")
        return False
    try:
        # Basic sanity: create a VDevice and release it
        vdev = hpf.VDevice()
        vdev.release()
        return True
    except Exception:
        return False


# New: CUDA availability via PyTorch
def is_cuda_available() -> bool:
    """Return True if CUDA is available via PyTorch."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _select_cuda_device() -> None:
    """
    If CUDA is available, select device 0 and print its name.
    Safe no-op on CPU-only environments.
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[INFO] Using CUDA device 0: {gpu_name}")
            except Exception:
                print("[INFO] Using CUDA device 0")
        else:
            print("[WARN] CUDA not available. Running on CPU.")
    except Exception:
        print("[WARN] PyTorch not available. Running on CPU.")


# New: Model format validation for non-Hailo backends
def _is_supported_non_hailo_model(path_or_id: str) -> bool:
    """
    Accept:
      - PyTorch: .pt, .pth
      - ONNX: .onnx
      - Hugging Face: repo ids like 'org/model' or local dir with config/model files.
    Minimal check: extension or 'org/model' pattern.
    """
    lower = path_or_id.lower()
    if lower.endswith((".pt", ".pth", ".onnx")):
        return True
    # Heuristic: huggingface repo id "org/model"
    if "/" in path_or_id and not lower.endswith(".hef"):
        return True
    # Local directory with HF-like files (heuristic)
    try:
        if os.path.isdir(path_or_id):
            entries = set(name.lower() for name in os.listdir(path_or_id))
            if any(k in entries for k in ("config.json", "model.safetensors", "pytorch_model.bin")):
                return True
    except Exception:
        pass
    return False


# --------------------- Non-Hailo helpers (postprocessing) ---------------------

def _parse_generic_yolo_output(pred: np.ndarray, img_shape: tuple, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Parse common YOLO-style output tensors into detections with class_id and bbox.
    Assumptions:
    - pred shape is (N, A) or (1, N, A), where A >= 5 + num_classes.
    - Columns are [x, y, w, h, conf, class_scores...].
    - x,y,w,h may be normalized in [0,1] or pixel units; detect heuristically.
    Returns: [{class_id:int, score:float, bbox:[x1,y1,x2,y2]}]
    """
    H, W = img_shape[:2]
    # Squeeze batch dimension if present
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[1] < 6:
        return []

    x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    conf = pred[:, 4]

    # Class scores (softmax/logits); pick argmax
    cls_scores = pred[:, 5:]
    if cls_scores.size == 0:
        # No class dimension; treat all as class 0
        cls_id = np.zeros((pred.shape[0],), dtype=np.int64)
        best_score = conf
    else:
        cls_id = np.argmax(cls_scores, axis=1)
        best_score = conf * np.max(cls_scores, axis=1)

    # Heuristic: normalized vs pixel boxes
    max_coord = float(np.max([np.max(x), np.max(y), np.max(w), np.max(h)]) if pred.size else 0.0)
    if max_coord <= 1.5:
        # normalized xywh -> pixel
        x *= W; y *= H; w *= W; h *= H

    # xywh to xyxy
    x1 = (x - w / 2).astype(np.float32)
    y1 = (y - h / 2).astype(np.float32)
    x2 = (x + w / 2).astype(np.float32)
    y2 = (y + h / 2).astype(np.float32)

    # Clip to image bounds
    x1 = np.clip(x1, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    x2 = np.clip(x2, 0, W - 1)
    y2 = np.clip(y2, 0, H - 1)

    # Confidence filter
    keep = best_score >= conf_threshold
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    scores = best_score[keep]
    cls_id = cls_id[keep]

    if x1.size == 0:
        return []

    # NMS per-class using OpenCV if available; otherwise simple numpy NMS
    dets: List[Dict[str, Any]] = []
    if cv2 is not None and hasattr(cv2.dnn, "NMSBoxes"):
        boxes = [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        # NMSBoxes expects score list of floats
        idxs = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf_threshold, nms_threshold=0.45)
        idxs = np.array(idxs).reshape(-1) if len(idxs) > 0 else np.array([], dtype=np.int64)
        for i in idxs:
            a, b, c, d = x1[i], y1[i], x2[i], y2[i]
            dets.append({
                "class_id": int(cls_id[i]),
                "score": float(scores[i]),
                "bbox": [int(a), int(b), int(c), int(d)]
            })
    else:
        # Simple class-agnostic NMS
        order = np.argsort(-scores)
        x1o, y1o, x2o, y2o, so = x1[order], y1[order], x2[order], y2[order], scores[order]
        cls_o = cls_id[order]
        keep_idxs = []
        while len(order) > 0:
            i = order[0]
            keep_idxs.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w_int = np.maximum(0.0, xx2 - xx1)
            h_int = np.maximum(0.0, yy2 - yy1)
            inter = w_int * h_int
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            iou = inter / (area_i + area_j - inter + 1e-9)
            remain = np.where(iou <= 0.45)[0] + 1
            order = order[remain]
        for i in keep_idxs:
            dets.append({
                "class_id": int(cls_id[i]),
                "score": float(scores[i]),
                "bbox": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
            })
    return dets

# --------------------- Non-Hailo fallback (CPU/GPU) ----------------------

class NonHailoInferenceEngine:
    """
    Non-Hailo inference engine:
      - ONNX via OpenCV DNN (CUDA if available, else CPU)
      - PyTorch .pt/.pth via TorchScript or Ultralytics YOLO (CUDA if available, else CPU)
    Produces detections with class_id, score, and bbox via YOLO-style parsing.
    """
    def __init__(self, model_path: str, class_names: Optional[List[str]] = None):
        self.model_path = model_path
        self.class_names = class_names or CLASS_NAMES
        self.backend = None
        self.net = None           # OpenCV DNN network for ONNX
        self.torch_model = None   # TorchScript model
        self.ultra_model = None   # Ultralytics YOLO model

        lower = model_path.lower()
        is_onnx = lower.endswith(".onnx")
        is_pt = lower.endswith(".pt") or lower.endswith(".pth")

        if not os.path.isfile(model_path) or not (is_onnx or is_pt):
            raise RuntimeError("Non-Hailo backend requires a valid ONNX (.onnx) or PyTorch (.pt/.pth) model file.")

        # Ensure CUDA device selection happens prior to any GPU model init
        _select_cuda_device()

        if is_onnx:
            if cv2 is None:
                raise RuntimeError("OpenCV (cv2) is required to run ONNX models.")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            # Prefer CUDA for OpenCV DNN if compiled with CUDA support
            try:
                has_cuda_backend = hasattr(cv2.dnn, "DNN_BACKEND_CUDA")
                print(f"[DEBUG] OpenCV DNN CUDA backend available: {has_cuda_backend}")
                if is_cuda_available() and has_cuda_backend:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    # Use full CUDA or FP16 depending on support; FP16 often faster on 4090
                    target = getattr(cv2.dnn, "DNN_TARGET_CUDA_FP16", cv2.dnn.DNN_TARGET_CUDA)
                    self.net.setPreferableTarget(target)
                    print("[INFO] Using CUDA for OpenCV DNN (ONNX).")
                    self.backend = "ONNX-CUDA"
                else:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("[INFO] Using CPU for OpenCV DNN (ONNX).")
                    self.backend = "ONNX-CPU"
            except Exception as e:
                print(f"[WARN] OpenCV DNN backend selection failed: {e}. Falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "ONNX-CPU"
        elif is_pt:
            import torch  # type: ignore
            device = "cuda" if is_cuda_available() else "cpu"
            self.torch_device = torch.device(device)
            print(f"[DEBUG] Torch device selected: {self.torch_device}")

            # 1) Try TorchScript first
            try:
                self.torch_model = torch.jit.load(model_path, map_location=self.torch_device)
                if hasattr(self.torch_model, "eval"):
                    self.torch_model.eval()
                print(f"[INFO] Loaded TorchScript model on {device.upper()}.")
                self.backend = f"PYTORCH-{device.upper()}"
            except Exception:
                self.torch_model = None

            # 2) If not TorchScript, try Ultralytics YOLO loader
            if self.torch_model is None:
                try:
                    from ultralytics import YOLO  # type: ignore
                    self.ultra_model = YOLO(model_path)
                    print(f"[INFO] Loaded Ultralytics YOLO model.")
                    # Ultralytics manages device internally; we will pass device explicitly on predict()
                    self.backend = f"ULTRALYTICS-{device.upper()}"
                except Exception as e:
                    raise RuntimeError(
                        "Failed to load PyTorch .pt/.pth. Use a TorchScript export (torch.jit.save) "
                        "or ensure 'pip install ultralytics' is installed.\n"
                        f"Underlying error: {e}"
                    )

    def infer(self, frame_bgr: np.ndarray, conf_threshold: float = 0.60):
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        # Preprocess to model size (assume 640x640 for typical YOLO exports)
        img = frame_bgr
        if img.shape[0] != 640 or img.shape[1] != 640:
            if cv2 is not None:
                img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            else:
                img = np.resize(img, (640, 640, img.shape[2]))

        # ONNX via OpenCV DNN
        if self.net is not None and cv2 is not None:
            blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(640, 640), swapRB=True)
            self.net.setInput(blob)
            out = self.net.forward()
            pred = out[0] if isinstance(out, (list, tuple)) else out
            return _parse_generic_yolo_output(np.asarray(pred), img.shape, conf_threshold=conf_threshold)

        # TorchScript
        if getattr(self, "torch_model", None) is not None:
            try:
                import torch  # type: ignore
                rgb = img[:, :, ::-1]
                x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
                x = x.unsqueeze(0).to(self.torch_device, non_blocking=True)
                with torch.no_grad():
                    out = self.torch_model(x)
                pred = out[0] if isinstance(out, (list, tuple)) else out
                pred_np = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
                return _parse_generic_yolo_output(pred_np, img.shape, conf_threshold=conf_threshold)
            except Exception as e:
                print(f"[WARN] TorchScript inference failed: {e}")
                return []

        # Ultralytics YOLO
        if getattr(self, "ultra_model", None) is not None:
            try:
                use_device = 0 if is_cuda_available() else None
                results = self.ultra_model.predict(
                    img, imgsz=640, conf=conf_threshold, device=use_device, verbose=False
                )
                dets: List[Dict[str, Any]] = []
                if not results:
                    return dets
                res = results[0]
                if hasattr(res, "boxes") and res.boxes is not None:
                    xyxy = res.boxes.xyxy.detach().cpu().numpy()
                    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
                    conf = res.boxes.conf.detach().cpu().numpy()
                    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
                        dets.append({
                            "class_id": int(c),
                            "score": float(s),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                return dets
            except Exception as e:
                print(f"[WARN] Ultralytics inference failed: {e}")
                return []

        # No backend initialized
        raise RuntimeError("Non-Hailo backend not initialized.")

    # Ensure API parity with HailoInferenceEngine
    def wait_ready(self, timeout: float = 5.0, poll: float = 0.1, do_warmup: bool = False) -> bool:
        # Non-Hailo engines initialize synchronously; return True
        return True

    def close(self):
        # Release references to backend models
        self.net = None
        self.torch_model = None
        # If using Ultralytics, clear it as well (if present)
        if hasattr(self, "ultra_model"):
            self.ultra_model = None

# -------------------------- Hailo engine (default) --------------------------

class HailoInferenceEngine:
    """Persistent Hailo YOLO inference engine.
    Loads HEF, configures device and opens InferVStreams once. Call infer(frame) to get detections.
    """
    def __init__(self, hef_path: Optional[str] = None, model_path: Optional[str] = None, class_names: Optional[List[str]] = None):
        if hpf is None:
            raise RuntimeError("Hailo SDK not available; cannot construct HailoInferenceEngine.")
        if cv2 is None:
            raise RuntimeError("OpenCV not available; cannot construct HailoInferenceEngine.")
        # Backward compatible: prefer explicit model_path, else hef_path, else default
        self.hef_path = model_path or hef_path or MODEL_PATH_DEFAULT
        self.class_names = class_names or CLASS_NAMES
        self.hef = hpf.HEF(self.hef_path)
        self.ready = False

        # Create device and configure with HEF
        self.vdev = hpf.VDevice()
        cfg = hpf.ConfigureParams.create_from_hef(self.hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_group = self.vdev.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()

        # IO metadata
        in_info = self.hef.get_input_vstream_infos()[0]
        out_info = self.hef.get_output_vstream_infos()[0]
        self.in_name, self.out_name = in_info.name, out_info.name
        self.in_shape = tuple(in_info.shape)  # (H,W,C) if NHWC
        H, W, C = self.in_shape

        # VStream params
        self.in_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.UINT8
        )
        self.out_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.FLOAT32
        )

        # Activate + open infer streams
        self.activation = self.network_group.activate(self.ng_params)
        self.activation.__enter__()
        self.infer_pipe = hpf.InferVStreams(self.network_group, self.in_params, self.out_params)
        self.infer_pipe.__enter__()
        self.ready = True

    def infer(self, frame_bgr: np.ndarray, conf_threshold: float = 0.60):
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        H, W, C = self.in_shape
        if frame_bgr.shape[0] != H or frame_bgr.shape[1] != W:
            frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        inp_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
        inp = np.expand_dims(inp_rgb, axis=0)
        inp = np.ascontiguousarray(inp)
        expected_shape = (1, H, W, C)
        if inp.shape != expected_shape:
            print(f"Bad input shape {inp.shape}, expected {expected_shape}")
            return []
        results = self.infer_pipe.infer({self.in_name: inp})
        raw_out = results[self.out_name]
        return parse_hailo_yolo_output(raw_out, frame_bgr.shape, class_names=self.class_names, conf_threshold=conf_threshold)

    def wait_ready(self, timeout: float = 10.0, poll: float = 0.1, do_warmup: bool = True) -> bool:
        deadline = time.time() + timeout
        if self.ready and not do_warmup:
            return True
        while time.time() < deadline:
            if do_warmup:
                try:
                    H, W, C = self.in_shape
                    dummy = np.zeros((H, W, 3), dtype=np.uint8)
                    _ = self.infer(dummy, conf_threshold=0.99)
                    return True
                except Exception:
                    time.sleep(poll)
                    continue
            else:
                if self.ready:
                    return True
                time.sleep(poll)
        return False

    def close(self):
        if hasattr(self, 'infer_pipe') and self.infer_pipe:
            self.infer_pipe.__exit__(None, None, None)
        if hasattr(self, 'activation') and self.activation:
            self.activation.__exit__(None, None, None)
        if hasattr(self, 'vdev') and self.vdev:
            self.vdev.release()
        self.ready = False


_engine: Optional[HailoInferenceEngine] = None

def init_engine(model_path: Optional[str] = None, hef_path: Optional[str] = None, class_names: Optional[List[str]] = None):
    global _engine
    if _engine is None:
        resolved_path = model_path or hef_path or MODEL_PATH_DEFAULT
        if is_hailo_available():
            _engine = HailoInferenceEngine(model_path=resolved_path, class_names=class_names)
            atexit.register(_engine.close)
            return _engine

        # Hailo not available: require supported non-Hailo model and load it
        if not _is_supported_non_hailo_model(resolved_path):
            raise RuntimeError("Hailo unavailable and model path not recognized (.pt/.pth/.onnx or HF repo/dir). Please provide a valid non-Hailo model.")
        # Select CUDA device before non-Hailo backend init
        _select_cuda_device()
        _engine = NonHailoInferenceEngine(model_path=resolved_path, class_names=class_names)
        atexit.register(_engine.close)
    return _engine

def initialize(model_path: Optional[str] = None, hef_path: Optional[str] = None, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Probe the environment and select a backend.
    - Prefer Hailo if available; return a ready Hailo engine.
    - Otherwise, fall back to CUDA if available; otherwise CPU, using ONNX/PT model.
    """
    status: Dict[str, Any] = {"backend": None, "hailo_available": False, "cuda_available": False, "engine": None}

    resolved_path = model_path or hef_path or MODEL_PATH_DEFAULT
    if is_hailo_available():
        print("[INFO] Hailo platform detected. Initializing Hailo engine.")
        try:
            eng = init_engine(model_path=resolved_path, class_names=class_names)
            status.update({"backend": "HAILO", "hailo_available": True, "engine": eng})
            return status
        except Exception as e:
            print(f"[WARN] Failed to initialize Hailo engine: {e}")

    cuda = is_cuda_available()
    status["cuda_available"] = cuda
    backend = "CUDA" if cuda else "CPU"
    status["backend"] = backend

    if not _is_supported_non_hailo_model(resolved_path):
        print("[ERROR] Non-Hailo model path not recognized (.pt/.pth/.onnx or HF). Provide a valid model.")
        return status

    try:
        # Select CUDA device before non-Hailo backend init
        _select_cuda_device()
        eng = NonHailoInferenceEngine(model_path=resolved_path, class_names=class_names)
        status["engine"] = eng
        print(f"[INFO] Using non-Hailo backend: {backend} with model '{resolved_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to initialize non-Hailo backend: {e}")

    return status

def run_inference(img: np.ndarray, conf_threshold: float = 0.60, engine: Optional[HailoInferenceEngine] = None, class_names: Optional[List[str]] = None):
    eng = engine or init_engine(class_names=class_names)
    return eng.infer(img, conf_threshold=conf_threshold)
