"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: app.py  —  Flask Web API
=============================================================
Endpoints:
  GET  /                       — Frontend UI
  POST /api/register           — Register member (upload photo)
  GET  /api/members            — List all members
  DELETE /api/delete/<id>      — Delete a member  (note: /api/delete/ to avoid static conflict)
  GET  /api/member_image/<id>  — Serve member photo
  GET  /api/video_feed         — MJPEG camera stream
  GET  /api/status             — Health check
  POST /api/start_camera       — Start camera
  POST /api/stop_camera        — Stop camera
  GET  /api/logs               — Entry log
"""

from __future__ import annotations
import os
import csv
import pickle
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS
from deepface import DeepFace

from config import (
    DATABASE_DIR, EMBEDDINGS_FILE, MODEL_NAME, DETECTOR_BACKEND,
    DISTANCE_THRESHOLD, LOGS_DIR, LOG_FILE, LOG_HEADERS,
    COLOR_GRANTED, COLOR_DENIED, COLOR_UNKNOWN,
    FONT_SCALE, THICKNESS, FORCE_CPU,
)
from db_utils import create_schema, insert_member, get_all_members, delete_member

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Serve the frontend directory as static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "DELETE", "OPTIONS"])

# Ensure dirs exist
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
create_schema()

# ── Shared camera/recognition state ──────────────────────────────────────────
camera_state = {
    "running": False,
    "cap": None,
    "output_frame": None,
    "lock": threading.Lock(),
    "embeddings": {},
}

_recog_state = {
    "current_frame": None,
    "last_results": [],
    "running": False,
}

DNN_NET = None
DNN_NET_LOCK = threading.Lock()

FONT       = cv2.FONT_HERSHEY_DUPLEX
SMALL_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


def load_embeddings() -> dict:
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_embeddings(embeddings: dict) -> None:
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)


def generate_embedding(image_path: str) -> np.ndarray | None:
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
        if not result:
            return None
        best = max(result, key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
        return np.array(best["embedding"], dtype=np.float32)
    except Exception as exc:
        log.error("Embedding error: %s", exc)
        return None


def match_against_database(probe_emb: np.ndarray, embeddings: dict, threshold: float) -> tuple:
    best_id = best_name = best_level = None
    best_distance = float("inf")
    all_scores = {}
    for mid, data in embeddings.items():
        dist = cosine_distance(probe_emb, data["embedding"])
        all_scores[mid] = {"name": data["name"], "distance": dist}
        if dist < best_distance:
            best_distance = dist
            best_id   = mid
            best_name = data["name"]
            best_level = data["level"]
    if best_distance > threshold:
        return None, None, None, best_distance, all_scores
    return best_id, best_name, best_level, best_distance, all_scores


def log_entry(member_id, name, level, distance, decision):
    os.makedirs(LOGS_DIR, exist_ok=True)
    write_header = not os.path.exists(LOG_FILE)
    try:
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
    except Exception as exc:
        log.error("Log write error: %s", exc)


def get_dnn_net():
    global DNN_NET
    with DNN_NET_LOCK:
        if DNN_NET is None:
            prototxt    = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
            caffemodel  = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                DNN_NET = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            else:
                log.warning("DNN model files not found — using Haar fallback.")
        return DNN_NET


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (8, 8, 24), -1)
    frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)
    cv2.putText(frame, "PREMIUM LOUNGE  |  FACE RECOGNITION",
                (12, 32), FONT, 0.65, (180, 160, 255), 1, cv2.LINE_AA)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"{ts}  |  {fps:.0f} FPS",
                (w - 220, 32), SMALL_FONT, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return frame


def draw_face_box(frame: np.ndarray, area: dict, color: tuple, label_lines: list) -> np.ndarray:
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3),
                  tuple(max(c - 120, 0) for c in color), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, THICKNESS)
    tick = 18
    for sx, sy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1),
                             (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        cv2.line(frame, (sx, sy), (sx + dx * tick, sy), color, 3)
        cv2.line(frame, (sx, sy), (sx, sy + dy * tick), color, 3)

    line_h    = 26
    box_h     = line_h * len(label_lines) + 8
    label_top = y - box_h - 4
    if label_top < 58:
        label_top = y + h + 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, label_top), (x + w, label_top + box_h), (8, 8, 20), -1)
    frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)
    for i, line in enumerate(label_lines):
        cv2.putText(frame, line,
                    (x + 6, label_top + line_h * (i + 1) - 4),
                    SMALL_FONT, 0.55, color, 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Background recognition thread
# ─────────────────────────────────────────────────────────────────────────────

def recognition_worker():
    last_logged_ids = set()
    log.info("Recognition worker started.")
    while _recog_state["running"]:
        try:
            frame = _recog_state["current_frame"]
            if frame is None:
                time.sleep(0.02)
                continue

            embeddings = camera_state["embeddings"]
            if not embeddings:
                time.sleep(0.1)
                continue

            try:
                results = DeepFace.represent(
                    img_path=frame.copy(),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                )
            except Exception as exc:
                log.debug("DeepFace error (non-fatal): %s", exc)
                results = []

            if results and len(results) > 1:
                results.sort(key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0), reverse=True)
                results = [results[0]]

            new_results = []
            for face_data in (results or []):
                try:
                    emb  = np.array(face_data["embedding"], dtype=np.float32)
                    area = face_data["facial_area"]
                    mid, name, level, dist, scores = match_against_database(
                        emb, embeddings, DISTANCE_THRESHOLD
                    )
                    decision = "GRANTED" if mid else "DENIED"
                    new_results.append({
                        "member_id": mid, "name": name, "level": level,
                        "distance": dist, "area": area, "decision": decision,
                    })
                    if mid and mid not in last_logged_ids:
                        log_entry(mid, name, level, dist, decision)
                        last_logged_ids.add(mid)
                        log.info("[%s] %s | dist=%.4f", decision, name, dist)
                except Exception as exc:
                    log.debug("Face result parse error: %s", exc)

            _recog_state["last_results"] = new_results

        except Exception as exc:
            log.error("Recognition worker unhandled error: %s", exc)
            time.sleep(0.5)

        time.sleep(0.05)   # ~20 recognitions/sec max

    log.info("Recognition worker stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture + rendering loop
# ─────────────────────────────────────────────────────────────────────────────

def _make_blank_frame(text: str = "Initialising camera...") -> bytes:
    """Return a JPEG bytes for a dark placeholder frame."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (12, 12, 30)
    cv2.putText(img, text, (60, 240), FONT, 0.9, (180, 160, 255), 1, cv2.LINE_AA)
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()


def camera_loop():
    cap = camera_state["cap"]
    dnn = get_dnn_net()

    fps_timer   = time.time()
    frame_count = 0
    fps         = 0.0

    # Send a placeholder immediately so the MJPEG stream isn't empty at startup
    with camera_state["lock"]:
        camera_state["output_frame"] = _make_blank_frame()

    while camera_state["running"]:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            _recog_state["current_frame"] = frame.copy()
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now

            frame = draw_hud(frame, fps)
            h_f, w_f = frame.shape[:2]

            # ── Fast DNN face tracking ─────────────────────────────────────
            fast_faces = []
            if dnn is not None:
                try:
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    dnn.setInput(blob)
                    detections = dnn.forward()
                    for i in range(detections.shape[2]):
                        conf = detections[0, 0, i, 2]
                        if conf > 0.15:
                            box = detections[0, 0, i, 3:7] * np.array([w_f, h_f, w_f, h_f])
                            sx, sy, ex, ey = box.astype("int")
                            sx, sy = max(0, sx), max(0, sy)
                            ex, ey = min(w_f, ex), min(h_f, ey)
                            if ex - sx > 30 and ey - sy > 30:
                                fast_faces.append((sx, sy, ex - sx, ey - sy))
                except Exception:
                    pass
            else:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    for (x, y, w, h) in cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40)):
                        fast_faces.append((x, y, w, h))
                except Exception:
                    pass

            if fast_faces:
                fast_faces.sort(key=lambda f: f[2] * f[3], reverse=True)
                fast_faces = [fast_faces[0]]

            current_faces = list(_recog_state.get("last_results", []))

            if not fast_faces:
                cv2.putText(frame, "No face detected",
                            (w_f // 2 - 130, h_f // 2),
                            FONT, 0.9, COLOR_UNKNOWN, 2, cv2.LINE_AA)
            else:
                for (x, y, w, h) in fast_faces:
                    area  = {"x": x, "y": y, "w": w, "h": h}
                    cx, cy = x + w / 2, y + h / 2
                    color  = COLOR_UNKNOWN
                    labels = ["Scanning..."]

                    if w < 90 or h < 90:
                        color  = COLOR_DENIED
                        labels = ["[X] DENIED: TOO FAR", "Step closer"]
                    else:
                        best_match, best_dist = None, float("inf")
                        for res in current_faces:
                            if not res.get("area"):
                                continue
                            rx = res["area"]["x"] + res["area"]["w"] / 2
                            ry = res["area"]["y"] + res["area"]["h"] / 2
                            d  = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
                            if d < best_dist:
                                best_dist  = d
                                best_match = res
                        if best_match and best_dist < 100:
                            decision = best_match["decision"]
                            sim = (1.0 - best_match["distance"]) * 100
                            if decision == "GRANTED":
                                color  = COLOR_GRANTED
                                labels = [f"[OK] GRANTED  {sim:.1f}%",
                                          str(best_match["name"]),
                                          str(best_match["level"])]
                            else:
                                color  = COLOR_DENIED
                                labels = [f"[X] DENIED  {sim:.1f}%", "Unknown"]

                    frame = draw_face_box(frame, area, color, labels)

            # Encode to JPEG and push into shared state
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with camera_state["lock"]:
                    camera_state["output_frame"] = jpeg.tobytes()

        except Exception as exc:
            log.error("Camera loop error: %s", exc)
            time.sleep(0.1)

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    with camera_state["lock"]:
        camera_state["output_frame"] = None
    log.info("Camera loop stopped.")


def generate_mjpeg():
    """
    MJPEG generator — yields JPEG frames as a multipart HTTP response.
    Exits cleanly when the camera stops (no stale open connections).
    """
    idle_ticks = 0
    while True:
        # Stop the generator when camera is off AND there's no more frame
        if not camera_state["running"]:
            with camera_state["lock"]:
                frame_bytes = camera_state.get("output_frame")
            if frame_bytes is None:
                return   # closed stream – no zombie connection
        else:
            with camera_state["lock"]:
                frame_bytes = camera_state.get("output_frame")

        if frame_bytes is None:
            idle_ticks += 1
            if idle_ticks > 200:   # ~10 s without a frame → give up
                return
            time.sleep(0.05)
            continue

        idle_ticks = 0
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
            + frame_bytes + b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps


# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))


@app.route("/api/status")
def api_status():
    embeddings = load_embeddings()
    members    = get_all_members()
    return jsonify({
        "status":            "ok",
        "model":             MODEL_NAME,
        "detector":          DETECTOR_BACKEND,
        "threshold":         DISTANCE_THRESHOLD,
        "members_count":     len(members),
        "embeddings_count":  len(embeddings),
        "camera_active":     camera_state["running"],
        "timestamp":         datetime.now().isoformat(),
    })


@app.route("/api/members", methods=["GET"])
def api_get_members():
    members = get_all_members()
    for m in members:
        img_path = m.get("image_path", "")
        m["has_image"] = os.path.isfile(img_path)
        m["image_url"] = f"/api/member_image/{m['member_id']}" if m["has_image"] else None
    return jsonify(members)


@app.route("/api/member_image/<member_id>")
def api_member_image(member_id: str):
    for m in get_all_members():
        if m["member_id"] == member_id:
            img_path = m.get("image_path", "")
            if os.path.isfile(img_path):
                return send_file(img_path)
    return jsonify({"error": "Image not found"}), 404


@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Expects multipart/form-data:
      name             (str, required)
      membership_level (str, optional – default 'Premium')
      photo            (file, required)
    """
    name  = request.form.get("name", "").strip()
    level = request.form.get("membership_level", "Premium").strip()
    photo = request.files.get("photo")

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not photo or photo.filename == "":
        return jsonify({"error": "Photo file is required"}), 400

    # Auto-increment member ID
    existing_ids = [m["member_id"] for m in get_all_members()]
    num = 1
    while f"M{num:03d}" in existing_ids:
        num += 1
    member_id = f"M{num:03d}"

    ext      = Path(photo.filename).suffix.lower() or ".jpg"
    filename = f"{member_id}_{name.replace(' ', '_')}{ext}"
    save_path = os.path.join(DATABASE_DIR, filename)
    photo.save(save_path)

    embedding = generate_embedding(save_path)
    if embedding is None:
        try:
            os.remove(save_path)
        except Exception:
            pass
        return jsonify({"error": "No face detected. Use a clear, well-lit photo."}), 400

    embeddings = load_embeddings()
    embeddings[member_id] = {
        "embedding":  embedding,
        "name":       name,
        "level":      level,
        "image_path": save_path,
    }
    save_embeddings(embeddings)
    insert_member(member_id, name, level, save_path)

    if camera_state["running"]:
        camera_state["embeddings"] = embeddings

    log.info("[REGISTER] %s | %s | %s", member_id, name, level)
    return jsonify({
        "success":   True,
        "member_id": member_id,
        "name":      name,
        "level":     level,
        "message":   f"'{name}' registered as {member_id}!",
    })


# ── DELETE uses /api/delete/<id> (avoids Flask static file route conflict) ───
@app.route("/api/delete/<member_id>", methods=["DELETE", "POST"])
def api_delete_member(member_id: str):
    """
    Deletes a member from SQLite + embeddings.pkl + image file.
    Supports both DELETE and POST for wider browser/CORS compatibility.
    """
    try:
        delete_member(member_id)
    except Exception as exc:
        log.error("SQLite delete error for %s: %s", member_id, exc)
        return jsonify({"error": f"DB delete failed: {exc}"}), 500

    try:
        embeddings = load_embeddings()
        if member_id in embeddings:
            img_path = embeddings[member_id].get("image_path", "")
            del embeddings[member_id]
            save_embeddings(embeddings)
            if img_path and os.path.isfile(img_path):
                os.remove(img_path)
            if camera_state["running"]:
                camera_state["embeddings"] = embeddings
        else:
            embeddings = load_embeddings()   # still reload so camera is in sync
    except Exception as exc:
        log.error("Embeddings delete error for %s: %s", member_id, exc)
        # Member was deleted from DB so still return success
        return jsonify({
            "success": True,
            "warning": f"Removed from DB but embeddings cleanup failed: {exc}",
        })

    log.info("[DELETE] %s removed.", member_id)
    return jsonify({"success": True, "message": f"Member {member_id} deleted."})


@app.route("/api/start_camera", methods=["POST"])
def api_start_camera():
    if camera_state["running"]:
        return jsonify({"success": True, "message": "Camera already running."})

    try:
        body      = request.get_json(silent=True) or {}
        cam_index = int(body.get("camera_index", 0))
    except (ValueError, TypeError):
        cam_index = 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return jsonify({"error": f"Could not open camera index {cam_index}. "
                                  "Is another app using the camera?"}), 500

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camera_state["cap"]        = cap
    camera_state["running"]    = True
    camera_state["embeddings"] = load_embeddings()

    _recog_state["running"]       = True
    _recog_state["current_frame"] = None
    _recog_state["last_results"]  = []

    threading.Thread(target=recognition_worker, daemon=True).start()
    threading.Thread(target=camera_loop,        daemon=True).start()

    log.info("Camera %d started.", cam_index)
    return jsonify({"success": True, "message": f"Camera {cam_index} started."})


@app.route("/api/stop_camera", methods=["POST"])
def api_stop_camera():
    camera_state["running"]       = False
    _recog_state["running"]       = False
    _recog_state["current_frame"] = None
    return jsonify({"success": True, "message": "Camera stopped."})


@app.route("/api/video_feed")
def api_video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":             "no-cache, no-store, must-revalidate",
            "Pragma":                    "no-cache",
            "Expires":                   "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/logs")
def api_logs():
    rows = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            rows.reverse()
        except Exception:
            pass
    return jsonify(rows[:100])


@app.route("/api/logs/clear", methods=["POST"])
def api_clear_logs():
    """
    Removes all log entries that belong to deleted/UNKNOWN members.
    Keeps only rows whose MemberID is in the current registered members list.
    """
    try:
        valid_ids = {m["member_id"] for m in get_all_members()}
        if not os.path.exists(LOG_FILE):
            return jsonify({"success": True, "removed": 0, "kept": 0})

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            fieldnames = reader.fieldnames or LOG_HEADERS

        kept    = [r for r in all_rows if r.get("MemberID") in valid_ids]
        removed = len(all_rows) - len(kept)

        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

        log.info("[LOG] Cleared %d stale entries. %d kept.", removed, len(kept))
        return jsonify({"success": True, "removed": removed, "kept": len(kept)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting FaceVault API  →  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
