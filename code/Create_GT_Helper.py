import os
import time
import cv2
import numpy as np
from AI_Engine import init_engine, run_inference
# Saves a png of each cropped frame where motion is detected 
# along with a txt file of detections in YOLO format for each contour
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo11m.pt")

# --- Local video file config ------------------------------------------------
# Provide path to an MP4 (or AVI, MKV) file to process instead of RTSP stream.
#Todo fix this so it uses Windows or Linux Path separators, manually adjust this based on your OS

#Linux path
#VIDEO_PATH = "/home/.../day1.mp4" 
#Windows path
VIDEO_PATH = "C:\\...\\Test_Video.mp4" # <- change to your file
# ---------------------------------------------------------------------------


def open_video_capture(path: str):
    cap = cv2.VideoCapture(path)
    return cap, cap.isOpened()


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame
def pad_image(img):
    """Pad or resize an image to 640x640 with black borders, preserving BGR order."""
    target = 640
    h, w = img.shape[:2]
    top = max((target - h) // 2, 0)
    bottom = max(target - h - top, 0)
    left = max((target - w) // 2, 0)
    right = max(target - w - left, 0)
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded = img
    if padded.shape[0] != target or padded.shape[1] != target:
        padded = cv2.resize(padded, (target, target), interpolation=cv2.INTER_NEAREST)
    return padded

    
# This function sopports saving the bounding box in YOLO format. I use that in YOLOLabel, which is a great tool for creating training data.
# I will be trainng a model later with the data I collect.    
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

# I need the inference engine because of how I am doing motion detection and cropping. Most examples implement this a local
# and then send a stream to it. That lets the model load once and works. I need the persistant instance
# so can pass multiple frames to it without getting the hailo_platform.pyhailort.pyhailort.hailortstatus exception: 8
# This is based on Hailo SDK examples and my own experiments.
# If you are interested in Hailo SDK programming, I hope this is helpful.
                        

# From here down is the motion detection and main loop code. I am pulling frames from a recorded rtsp stream, 
# Using OpenCV for motion detection and cropping. 
# Then passing the cropped frames to the AI InferenceEngine instance for detection.
# This lets me keep the model loaded once and avoid the hailortstatus exception: 8


# Add duplicate filtering threshold (pixels) like benchmark script
DUP_MIN_DISTANCE: float = 50.0

def calculate_euclidean_distance(x1, y1, x2, y2):
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

def prepare_inference_crop(crop: np.ndarray):
    """Pad-only if smaller than 640 or resize-only if larger; ensure exact 640x640.
    Returns (prepared_image, pad_info). pad_info keys:
      orig_w, orig_h, left_pad, top_pad, resized
    """
    target = 640
    orig_h, orig_w = crop.shape[:2]
    left = 0; top = 0
    resized = False
    prepared = crop
    if orig_h < target or orig_w < target:
        top = max((target - orig_h) // 2, 0)
        bottom = max(target - orig_h - top, 0)
        left = max((target - orig_w) // 2, 0)
        right = max(target - orig_w - left, 0)
        prepared = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ph, pw = prepared.shape[:2]
        extra_top = 0; extra_left = 0
        if ph != target or pw != target:
            extra_bottom = max(target - ph, 0)
            extra_right = max(target - pw, 0)
            prepared = cv2.copyMakeBorder(prepared, extra_top, extra_bottom, extra_left, extra_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        total_left_pad = left + extra_left
        total_top_pad = top + extra_top
    else:
        total_left_pad = 0
        total_top_pad = 0
    if orig_h > target or orig_w > target:
        prepared = cv2.resize(crop, (target, target), interpolation=cv2.INTER_NEAREST)
        total_left_pad = 0
        total_top_pad = 0
        resized = True
    pad_info = {
        'orig_w': orig_w,
        'orig_h': orig_h,
        'left_pad': float(total_left_pad),
        'top_pad': float(total_top_pad),
        'resized': resized,
    }
    return prepared, pad_info

def detect_motion():

    # Open local video file once
    cap, opened = open_video_capture(VIDEO_PATH)
    if not opened:
        raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")
    # Ensure the inference engine is initialized and ready before processing frames
    eng = init_engine(model_path= MODEL_PATH)
    if not eng.wait_ready(timeout=15.0, poll=0.2, do_warmup=True):
        print("Hailo inference engine not ready within timeout.")
        return
    try:
        frame1 = read_frame(cap)
        if frame1 is None:
            # End of file
            print("End of video.")


        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame1 = cv2.GaussianBlur(gray_frame1, (5, 5), 0)       
        while True:


            frame2 = read_frame(cap)
            if frame2 is None:
                print("End of video.")
                break
            base = f"images/{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}"
            print(base)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.GaussianBlur(gray_frame2, (5,5), 0)
            frame_delta = cv2.absdiff(gray_frame1, gray_frame2)
          # the integer 25 in the next line is a constant for testing in the run bencmarks script. Values less than 25 caused excessive contours to be detected in my testing.
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=1)

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            os.makedirs("images", exist_ok=True)


            yolo_lines = []
            counter = 0
            frame_for_full = cv2.resize(frame2, (640, 640), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(base + ".png", frame_for_full,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            fulldetections = run_inference(frame_for_full, conf_threshold=0.60, engine=eng)

            # Prepare unified coordinate system (original frame dimensions)
            frame_h, frame_w = frame2.shape[:2]
            # Duplicate filtering list across entire frame (full + motion)
            accepted = []  # (cid, global_centroid_x, global_centroid_y)

            if fulldetections:
                # Map full-frame detections (in 640x640 space) back to original frame coordinates
                scale_x_full = frame_w / 640.0
                scale_y_full = frame_h / 640.0
                for d in fulldetections:
                    bbox = d.get('bbox'); cid = d.get('class_id')
                    if not bbox:
                        continue
                    x1b, y1b, x2b, y2b = bbox  # in 640x640 resized full-frame space
                    # Map to original frame space
                    gx1 = x1b * scale_x_full; gy1 = y1b * scale_y_full
                    gx2 = x2b * scale_x_full; gy2 = y2b * scale_y_full
                    # Record centroid for duplicate filtering against motion crops
                    cxc = (gx1 + gx2) / 2.0
                    cyc = (gy1 + gy2) / 2.0
                    accepted.append((cid, cxc, cyc))
                    # Write YOLO normalized by original frame dimensions (unified with motion crops)
                    xc, yc, ww, hh = pascal_voc_to_yolo(gx1, gy1, gx2, gy2, frame_w, frame_h)
                    yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

            # Motion crops; duplicate filtering against 'accepted' suppresses overlaps with full-frame detections
            for contour in contours:
                if cv2.contourArea(contour) <= 100:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                frame_for_crop = frame2.copy()
                if w >= 640 and h >= 640:
                    x1 = max(x, 0); y1 = max(y, 0)
                    x2 = min(x + w, frame_for_crop.shape[1]); y2 = min(y + h, frame_for_crop.shape[0])
                else:
                    cx = x + w // 2; cy = y + h // 2
                    half = 320
                    x1 = max(cx - half, 0); y1 = max(cy - half, 0)
                    x2 = min(cx + half, frame_for_crop.shape[1]); y2 = min(cy + half, frame_for_crop.shape[0])

                original_crop = frame_for_crop[y1:y2, x1:x2]
                if original_crop is None or original_crop.size == 0:
                    continue

                inf_crop, pad_info = prepare_inference_crop(original_crop)
                detections = run_inference(inf_crop, conf_threshold=0.60, engine=eng)

                if detections:
                    for d in detections:
                        bbox = d.get('bbox'); cid = d.get('class_id')
                        if not bbox:
                            continue
                        xx1, yy1, xx2, yy2 = bbox  # in 640x640 inference space
                        if pad_info['resized']:
                            scale_x = pad_info['orig_w'] / 640.0
                            scale_y = pad_info['orig_h'] / 640.0
                            local_x1 = xx1 * scale_x; local_x2 = xx2 * scale_x
                            local_y1 = yy1 * scale_y; local_y2 = yy2 * scale_y
                        else:
                            local_x1 = xx1 - pad_info['left_pad']; local_x2 = xx2 - pad_info['left_pad']
                            local_y1 = yy1 - pad_info['top_pad'];  local_y2 = yy2 - pad_info['top_pad']
                        local_x1 = max(0.0, min(pad_info['orig_w'] - 1.0, local_x1))
                        local_x2 = max(0.0, min(pad_info['orig_w'] - 1.0, local_x2))
                        local_y1 = max(0.0, min(pad_info['orig_h'] - 1.0, local_y1))
                        local_y2 = max(0.0, min(pad_info['orig_h'] - 1.0, local_y2))
                        global_x1 = x1 + local_x1; global_y1 = y1 + local_y1
                        global_x2 = x1 + local_x2; global_y2 = y1 + local_y2

                        candidate_cx = (global_x1 + global_x2) / 2.0
                        candidate_cy = (global_y1 + global_y2) / 2.0

                        too_close = False
                        for ac_cid, ex_cx, ex_cy in accepted:
                            if ac_cid != cid:
                                continue
                            if calculate_euclidean_distance(candidate_cx, candidate_cy, ex_cx, ex_cy) <= DUP_MIN_DISTANCE:
                                too_close = True
                                break
                        if too_close:
                            continue

                        accepted.append((cid, candidate_cx, candidate_cy))
                        xc, yc, ww, hh = pascal_voc_to_yolo(global_x1, global_y1, global_x2, global_y2, frame_w, frame_h)
                        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
                    counter += 1

            with open(base + f".txt", "w") as f:
                for line in yolo_lines:
                    f.write(line + "\n")


# the first frame of the stream is used for motion detection comparison
# it does not get infered.
        #frame1 = frame2
        #gray_frame1 = gray_frame2   

    except Exception as e:
        print(f"Error in detect_motion loop: {e}")
    finally:
        try:
            cap.release()
        except Exception:
            pass





def main():


    detect_motion()
       
if __name__ == "__main__":
    main()
