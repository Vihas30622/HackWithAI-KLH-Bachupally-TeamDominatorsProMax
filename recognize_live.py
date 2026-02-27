"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: recognize_live.py
=============================================================
PURPOSE
-------
Phase-2 live-camera recognition.  Opens the webcam, processes
each frame to detect faces, compares against stored embeddings,
and overlays a real-time result (green = granted, red = denied).

USAGE
-----
    python recognize_live.py
    python recognize_live.py --camera 0      # camera index
    python recognize_live.py --threshold 0.38
    python recognize_live.py --fps 15        # max processed FPS

KEY BINDINGS (while window is open)
------------------------------------
    q  → Quit
    s  → Save current frame as screenshot
    +  → Increase threshold (more permissive)
    -  → Decrease threshold (stricter)
    r  → Reset threshold to default
    d  → Toggle debug mode (show all member distances)

HOW FRAMES ARE PROCESSED
--------------------------
To keep the UI responsive, face recognition runs on every
PROC_EVERY n-th frame (configurable), while the overlay from
the last recognition is shown on all intermediate frames.
"""

from __future__ import annotations
import os
import sys
import csv
import pickle
import time
import logging
import argparse
import threading
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace

from config import (
    EMBEDDINGS_FILE, MODEL_NAME, DETECTOR_BACKEND,
    DISTANCE_THRESHOLD, LOGS_DIR, LOG_FILE, LOG_HEADERS,
    COLOR_GRANTED, COLOR_DENIED, COLOR_UNKNOWN,
    FONT_SCALE, THICKNESS, FORCE_CPU,
)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── Constants ─────────────────────────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_DUPLEX
SMALL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_TITLE    = "Premium Lounge  |  Face Recognition System - [Koushik Test]"
THRESHOLD_STEP  = 0.01

# ── Shared State for Threading ────────────────────────────────────────────────
shared_state = {
    "current_frame": None,
    # Now stores a LIST of result dicts, one for each face in the frame
    "last_results": [],
    "running": True
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


def load_embeddings() -> dict:
    if not os.path.exists(EMBEDDINGS_FILE):
        log.error("No embeddings file found. Run register_members.py first.")
        sys.exit(1)
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    log.info("Loaded %d member embedding(s).", len(data))
    return data


def get_frame_embeddings(frame_bgr: np.ndarray, model: str, detector: str) -> list[tuple[np.ndarray, dict]]:
    """
    Extract face embeddings from an OpenCV BGR frame for ALL detected faces.
    Returns a list of tuples: [(embedding_array, facial_area_dict), ...]
    Returns empty list if no faces found.
    """
    try:
        # DeepFace.represent returns a list of dicts, one for each face
        # Setting enforce_detection=False reduces deepface's internal overhead since we 
        # already pass it an image that contains faces (guaranteed by our 60FPS tracker)
        results = DeepFace.represent(
            img_path=frame_bgr,
            model_name=model,
            detector_backend=detector,
            enforce_detection=False,
        )
        if not results:
            return []
            
        faces = []
        for face_data in results:
            emb = np.array(face_data["embedding"], dtype=np.float32)
            area = face_data["facial_area"]
            faces.append((emb, area))
            
        return faces
        
    except Exception:
        # Exceptions (like "Face could not be detected") mean 0 faces
        return []


def match_against_database(
    probe_emb: np.ndarray,
    embeddings: dict,
    threshold: float
) -> tuple:
    best_id       = None
    best_name     = None
    best_level    = None
    best_distance = float("inf")
    all_scores    = {}

    for mid, data in embeddings.items():
        dist = cosine_distance(probe_emb, data["embedding"])
        all_scores[mid] = (data["name"], dist)
        if dist < best_distance:
            best_distance = dist
            best_id       = mid
            best_name     = data["name"]
            best_level    = data["level"]

    if best_distance > threshold:
        return None, None, None, best_distance, all_scores

    return best_id, best_name, best_level, best_distance, all_scores


def log_entry(member_id, name, level, distance, decision):
    os.makedirs(LOGS_DIR, exist_ok=True)
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LOG_HEADERS)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            member_id or "UNKNOWN",
            name or "UNKNOWN",
            level or "N/A",
            f"{distance:.4f}",
            decision,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, threshold: float, fps: float,
             debug: bool, frame_count: int) -> np.ndarray:
    """Draw the HUD bar at the top of the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 30), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    cv2.putText(frame, "PREMIUM LOUNGE  |  FACE RECOGNITION",
                (12, 30), FONT, 0.65, (200, 200, 255), 1, cv2.LINE_AA)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"{ts}  |  THR:{threshold:.2f}  |  {fps:.0f}fps",
                (w - 340, 30), SMALL_FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    if debug:
        cv2.putText(frame, "[DEBUG ON]", (w - 100, 48 + 18),
                    SMALL_FONT, 0.45, (0, 220, 220), 1, cv2.LINE_AA)
    return frame


def draw_face_box(frame: np.ndarray, area: dict,
                  color: tuple, label_lines: list[str]) -> np.ndarray:
    """Draw a bounding box and label lines around a detected face."""
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]

    # Glow effect — two rectangles
    cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3),
                  tuple(max(c - 120, 0) for c in color), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, THICKNESS)

    # Corner tick marks
    tick = 18
    for sx, sy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1),
                             (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        cv2.line(frame, (sx, sy), (sx + dx * tick, sy), color, 3)
        cv2.line(frame, (sx, sy), (sx, sy + dy * tick), color, 3)

    # Label background + text
    line_h    = 26
    box_h     = line_h * len(label_lines) + 8
    label_top = y - box_h - 4
    if label_top < 55:
        label_top = y + h + 4
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x, label_top),
                  (x + w, label_top + box_h),
                  (10, 10, 20), -1)
    frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)
    for i, line in enumerate(label_lines):
        cv2.putText(frame, line,
                    (x + 6, label_top + line_h * (i + 1) - 4),
                    SMALL_FONT, 0.55, color, 1, cv2.LINE_AA)
    return frame


def draw_no_face(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.putText(frame, "No face detected",
                (w // 2 - 120, h // 2),
                FONT, 0.9, COLOR_UNKNOWN, 2, cv2.LINE_AA)
    return frame


def draw_debug_scores(frame: np.ndarray, scores: dict) -> np.ndarray:
    """Show a small leaderboard of member distances."""
    y_start = 70
    cv2.putText(frame, "── Score Board ──",
                (10, y_start), SMALL_FONT, 0.45, (200, 200, 200), 1)
    for i, (mid, (name, dist)) in enumerate(
            sorted(scores.items(), key=lambda kv: kv[1][1])):
        col = COLOR_GRANTED if i == 0 else (180, 180, 180)
        cv2.putText(frame,
                    f"{name[:16]:<16} {dist:.3f}",
                    (10, y_start + 20 + i * 18),
                    SMALL_FONT, 0.42, col, 1)
    return frame


def draw_keymap(frame: np.ndarray) -> np.ndarray:
    """Bottom key-binding reference."""
    h, w = frame.shape[:2]
    keys = " q:Quit  s:Save  +:Permissive  -:Strict  r:Reset  d:Debug "
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 26), (w, h), (10, 10, 30), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.putText(frame, keys, (8, h - 8),
                SMALL_FONT, 0.42, (160, 160, 200), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Security placeholders
# ─────────────────────────────────────────────────────────────────────────────

def liveness_check(frame_bgr: np.ndarray) -> bool:
    """
    PLACEHOLDER — Liveness / anti-spoofing detection.
    Integrate Silent-Face or FasNet here for production.
    Currently always returns True (pass-through).
    """
    # TODO: Implement blink detection, texture analysis, etc.
    return True


def mask_compatibility_check(frame_bgr: np.ndarray) -> bool:
    """
    PLACEHOLDER — Mask detector.
    Could use a YOLO-based mask model.
    Currently always returns True.
    """
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-2 Live Camera Face Recognition Entry System (60 FPS Multi-Face)."
    )
    parser.add_argument("--camera",    default="0",
                        help="Camera index (e.g. 0) or IP camera URL (e.g. http://192.168.1.5:8080/video).")
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD,
                        help=f"Cosine distance threshold (default {DISTANCE_THRESHOLD}).")
    parser.add_argument("--model",     default=MODEL_NAME,
                        help=f"DeepFace model (default {MODEL_NAME}).")
    parser.add_argument("--debug",     action="store_true",
                        help="Show per-member distance scores.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Background Recognition Thread (Multi-Face)
# ─────────────────────────────────────────────────────────────────────────────

def recognition_worker(args, embeddings):
    """
    Runs continuously in the background. Grabs the latest frame from shared_state,
    runs DeepFace inference for ALL faces, and updates shared_state["last_results"].
    """
    last_logged_ids = set()
    log.info("Background multi-face recognition thread started.")
    
    while shared_state["running"]:
        frame = shared_state["current_frame"]
        if frame is None:
            time.sleep(0.005) # Super fast poll for 60fps responsiveness
            continue
            
        # Heavy inference block - returns data for EVERY face in the frame
        faces_data = get_frame_embeddings(frame, args.model, DETECTOR_BACKEND)
        
        new_results = []
        current_frame_ids = set()
        
        if not faces_data:
            pass # new_results remains empty (no faces drawn)
        else:
            is_live = liveness_check(frame)
            
            for probe_emb, area in faces_data:
                if not is_live:
                    new_results.append({
                        "member_id": None, "name": "SPOOF?", "level": None,
                        "distance":  1.0,  "area": area,
                        "decision": "DENIED", "scores": {},
                    })
                else:
                    mid, name, level, dist, scores = match_against_database(
                        probe_emb, embeddings, args.threshold
                    )
                    decision = "GRANTED" if mid else "DENIED"
                    
                    new_results.append({
                        "member_id": mid,   "name":  name,
                        "level":     level, "distance": dist,
                        "area":      area,  "decision": decision,
                        "scores":    scores,
                    })

                    # Log only once per session per person
                    if mid and mid not in last_logged_ids:
                        log_entry(mid, name, level, dist, decision)
                        last_logged_ids.add(mid)
                        log.info("  %s | %s | Dist=%.4f (Multi-Face)", 
                                 decision, name, dist)
                                 
        # Atomically update the display state with all faces
        shared_state["last_results"] = new_results
        
        # Micro-sleep to prevent 100% CPU lock on the thread
        time.sleep(0.005)

# ─────────────────────────────────────────────────────────────────────────────
# Main camera loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    threshold   = args.threshold
    debug_mode  = args.debug

    embeddings = load_embeddings()

    # Superior 60 FPS Real-time Tracker using OpenCV's DNN Face Detector (ResNet-10)
    # This tracks perfectly at angles/dark lighting where Haar totally fails.
    dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    cam_src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        log.error("Could not open camera '%s'.", args.camera)
        sys.exit(1)

    # For RTSP/HTTP streams, lowering buffer size helps reduce lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    log.info("Camera '%s' opened. Press q to quit.", args.camera)

    # Start the background recognition thread
    recog_thread = threading.Thread(
        target=recognition_worker, 
        args=(args, embeddings),
        daemon=True
    )
    recog_thread.start()

    frame_count    = 0
    fps_timer      = time.time()
    fps            = 0.0
    screenshot_n   = 0

    # Stores identity buffer for trackers so name doesn't frantically flicker "Scanning..." 
    # when you briefly look away or move very fast
    tracked_identities = []

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame capture failed — retrying …")
            time.sleep(0.05)
            continue
            
        # Update shared frame for background thread (use a copy to prevent thread tearing)
        shared_state["current_frame"] = frame.copy()

        frame_count += 1

        # ── FPS calculation ────────────────────────────────────────────────
        now = time.time()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer   = now

        # ── Render overlay ────────────────────────────────────────────────
        frame = draw_hud(frame, threshold, fps, debug_mode, frame_count)

        # ── Fast Real-time Box Tracking (DNN Module) ─────────────────────────────────
        # Uses Deep Neural Net tracker; completely robust to odd angles/blur/lighting.
        (h_f, w_f) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        
        fast_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Lowered tracking confidence to 0.15 to find faces further away 
            if confidence > 0.15: 
                box = detections[0, 0, i, 3:7] * np.array([w_f, h_f, w_f, h_f])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX, startY = max(0, startX), max(0, startY)
                endX, endY     = min(w_f, endX), min(h_f, endY)
                
                # Allow slightly smaller faces from a distance
                if endX - startX > 30 and endY - startY > 30: 
                    fast_faces.append((startX, startY, endX - startX, endY - startY))

        current_faces = list(shared_state["last_results"])
        
        if len(fast_faces) == 0:
            frame = draw_no_face(frame)
            tracked_identities = [] # Clear memory if no face is visible
        else:
            new_tracked_identities = []
            
            # Draw a box for every face detected instantly
            for (x, y, w, h) in fast_faces:
                area = {"x": x, "y": y, "w": w, "h": h}
                center_fast = (x + w/2, y + h/2)
                
                # 1. Look for a fresh AI match from the background thread
                best_match = None
                best_dist  = float('inf')
                for face_res in current_faces:
                    if face_res["area"] is None: continue
                    cx = face_res["area"]["x"] + face_res["area"]["w"]/2
                    cy = face_res["area"]["y"] + face_res["area"]["h"]/2
                    dist = ((cx - center_fast[0])**2 + (cy - center_fast[1])**2) ** 0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_match = face_res
                
                # 2. Check if we just have a recent "Sticky Identity" to carry over 
                # (helps immensely when turning or moving fast)
                memory_match = None
                if not (best_match and best_dist < 100):
                    m_dist = float('inf')
                    for past_id in tracked_identities:
                        dist = ((past_id["center"][0] - center_fast[0])**2 + (past_id["center"][1] - center_fast[1])**2) ** 0.5
                        if dist < 120 and dist < m_dist:
                            m_dist = dist
                            memory_match = past_id
                
                
                decision = "UNKNOWN"
                labels = ["Scanning Face..."]
                color = COLOR_UNKNOWN

                # --- PROXIMITY CHECK ---
                # Deny access natively if the person's face is extremely small (further away)
                if w < 90 or h < 90:
                    decision = "DENIED"
                    color = COLOR_DENIED
                    labels = ["[X] DENIED: TOO FAR", "Please step closer"]
                    new_tracked_identities.append({ "center": center_fast, "decision": decision, "labels": labels, "color": color, "age": 0 })
                else:
                    if best_match and best_dist < 100:
                        decision = best_match["decision"]
                        similarity = (1.0 - best_match["distance"]) * 100
                        if decision == "GRANTED":
                            color = COLOR_GRANTED
                            labels = [f"[OK] GRANTED  {similarity:.1f}%", f"{best_match['name']}", f"{best_match['level']}"]
                        else:
                            color = COLOR_DENIED
                            labels = [f"[X] DENIED  {similarity:.1f}%", "Unknown"]
                            
                        new_tracked_identities.append({ "center": center_fast, "decision": decision, "labels": labels, "color": color, "age": 0 })

                    elif memory_match and memory_match["age"] < 15: # Remember face for ~0.5 seconds
                        color = memory_match["color"]
                        labels = memory_match["labels"]
                        new_tracked_identities.append({
                            "center": center_fast, "decision": memory_match["decision"], 
                            "labels": labels, "color": color, "age": memory_match["age"] + 1
                        })

                frame = draw_face_box(frame, area, color, labels)

            tracked_identities = new_tracked_identities

            # Debug scores: just show the primary face's scores to avoid clutter
            if debug_mode and current_faces and current_faces[0]["scores"]:
                frame = draw_debug_scores(frame, current_faces[0]["scores"])

        frame = draw_keymap(frame)

        cv2.imshow(WINDOW_TITLE, frame)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log.info("Quit requested.")
            shared_state["running"] = False # Stop the background thread
            break
        elif key == ord("s"):
            screenshot_n += 1
            fname = os.path.join(LOGS_DIR,
                                 f"screenshot_{datetime.now():%Y%m%d_%H%M%S}_{screenshot_n}.jpg")
            cv2.imwrite(fname, frame)
            log.info("Screenshot saved -> %s", fname)
        elif key == ord("+"):
            args.threshold = round(min(args.threshold + THRESHOLD_STEP, 1.0), 3)
            threshold = args.threshold
            log.info("Threshold increased -> %.3f", threshold)
        elif key == ord("-"):
            args.threshold = round(max(args.threshold - THRESHOLD_STEP, 0.01), 3)
            threshold = args.threshold
            log.info("Threshold decreased -> %.3f", threshold)
        elif key == ord("r"):
            args.threshold = DISTANCE_THRESHOLD
            threshold = args.threshold
            log.info("Threshold reset -> %.3f", threshold)
        elif key == ord("d"):
            debug_mode = not debug_mode
            log.info("Debug mode: %s", "ON" if debug_mode else "OFF")

    # Give thread a moment to shut down cleanly
    recog_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    log.info("Session ended. Logs -> %s", LOG_FILE)


if __name__ == "__main__":
    main()
