"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: recognize_photo.py
=============================================================
PURPOSE
-------
Phase-1 prototype.  Accepts a single test image (test/guest.jpg),
generates its face embedding, compares it against all stored
member embeddings using cosine distance, and prints an
"Access Granted" or "Access Denied" result.

USAGE
-----
    python recognize_photo.py
    # or supply a custom image:
    python recognize_photo.py --image path/to/photo.jpg

OPTIONS
-------
    --image   PATH    Override default test/guest.jpg
    --threshold 0.4   Override distance threshold from config.py
    --model   NAME    Override model from config.py
"""

from __future__ import annotations
import os
import sys
import csv
import pickle
import argparse
import logging
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace

from config import (TEST_DIR, EMBEDDINGS_FILE, MODEL_NAME,
                    DETECTOR_BACKEND, DISTANCE_THRESHOLD,
                    LOGS_DIR, LOG_FILE, LOG_HEADERS, FORCE_CPU,
                    COLOR_GRANTED, COLOR_DENIED, COLOR_UNKNOWN)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance in [0, 1].
    0 → identical direction, 1 → orthogonal.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


def cosine_similarity_pct(distance: float) -> float:
    """Convert cosine distance → similarity percentage (0–100)."""
    return round((1.0 - distance) * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Core recognition
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings() -> dict:
    """Load stored member embeddings from disk."""
    if not os.path.exists(EMBEDDINGS_FILE):
        log.error("No embeddings file found at %s.\n"
                  "Run register_members.py first.", EMBEDDINGS_FILE)
        sys.exit(1)
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    log.info("Loaded %d member embedding(s).", len(data))
    return data


def extract_embeddings(image_path: str, model_name: str, detector: str) -> list[tuple[np.ndarray, dict]]:
    """
    Extract face embeddings from an image for ALL detected faces.
    Returns a list of tuples: [(embedding_array, facial_area_dict), ...]
    Returns empty list if no faces found.
    """
    if not os.path.isfile(image_path):
        log.error("Test image not found: %s", image_path)
        return []

    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector,
            enforce_detection=True
        )
        if not results:
            log.warning("No face detected in %s.", image_path)
            return []

        faces = []
        for face_data in results:
            emb = np.array(face_data["embedding"], dtype=np.float32)
            # Use 'facial_area' instead of deprecated 'region' if possible
            area = face_data.get("facial_area", face_data.get("region"))
            faces.append((emb, area))

        return faces

    except Exception as e:
        log.error("DeepFace extraction failed: %s", e)
        return []


def match_against_database(
    probe_emb: np.ndarray,
    embeddings: dict,
    threshold: float
) -> tuple[str | None, str | None, str | None, float]:
    """
    Compare probe embedding against all stored members.

    Returns:
        (member_id, name, level, best_distance)
        member_id is None if no match found below threshold.
    """
    best_id       = None
    best_name     = None
    best_level    = None
    best_distance = float("inf")

    for member_id, data in embeddings.items():
        dist = cosine_distance(probe_emb, data["embedding"])
        log.debug("  %s | %s → distance %.4f", member_id, data["name"], dist)
        if dist < best_distance:
            best_distance = dist
            best_id       = member_id
            best_name     = data["name"]
            best_level    = data["level"]

    if best_distance > threshold:
        return None, None, None, best_distance

    return best_id, best_name, best_level, best_distance


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_entry(member_id: str | None, name: str | None,
              level: str | None, distance: float, decision: str) -> None:
    """Append an entry record to the CSV log file."""
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
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  [*] PREMIUM LOUNGE -- BIOMETRIC ENTRY SYSTEM")
    print("=" * 60)


def print_result(decision: str, name: str | None,
                 level: str | None, distance: float) -> None:
    similarity = cosine_similarity_pct(distance)
    sep = "-" * 60
    if decision == "GRANTED":
        print(f"\n  {'[ACCESS GRANTED]':^56}")
        print(sep)
        print(f"  Welcome, {name}")
        print(f"  Membership Level : {level}")
        print(f"  Match Confidence : {similarity:.1f}%")
        print(f"  Cosine Distance  : {distance:.4f}")
    else:
        print(f"\n  {'[ACCESS DENIED]':^56}")
        print(sep)
        print(f"  No registered member matched this face.")
        print(f"  Best Confidence  : {similarity:.1f}%")
        print(f"  Cosine Distance  : {distance:.4f}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-1 photo-based face recognition entry system."
    )
    parser.add_argument("--image",     default=os.path.join(TEST_DIR, "guest.jpg"),
                        help="Path to the probe / guest image.")
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD,
                        help=f"Cosine distance threshold (default {DISTANCE_THRESHOLD}).")
    parser.add_argument("--model",     default=MODEL_NAME,
                        help=f"DeepFace model (default {MODEL_NAME}).")
    parser.add_argument("--verbose",   action="store_true",
                        help="Show per-member distances.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()
    log.info("Probe image  : %s", args.image)
    log.info("Model        : %s", args.model)
    log.info("Threshold    : %.2f (cosine distance)", args.threshold)

    # 1. Load stored embeddings
    embeddings = load_embeddings()

    # 2. Extract embeddings for all faces in image
    log.info("Generating probe embedding(s) ...")
    faces_data = extract_embeddings(args.image, args.model, DETECTOR_BACKEND)

    if not faces_data:
        log.warning("No face detected in the image.")
        sys.exit(0)
    
    log.info("Detected %d face(s) in %s.", len(faces_data), os.path.basename(args.image))

    # Read the image to draw bounding boxes
    frame = cv2.imread(args.image)
    if frame is None:
        log.error("Failed to cv2.imread %s", args.image)
        sys.exit(1)

    for i, (probe_emb, area) in enumerate(faces_data):
        log.info("--- Processing Face %d ---", i + 1)
        member_id, name, level, distance = match_against_database(probe_emb, embeddings, args.threshold)
        
        decision = "GRANTED" if member_id else "DENIED"
        print_result(decision, name, level, distance)
        log_entry(member_id, name, level, distance, decision)
        
        # Determine Box Color
        color = COLOR_GRANTED if decision == "GRANTED" else COLOR_DENIED
        similarity = (1.0 - distance) * 100
        
        # Build text label
        if decision == "GRANTED":
            labels = [f"[OK] GRANTED  {similarity:.1f}%", f"{name}", f"{level}"]
        else:
            labels = [f"[X] DENIED  {similarity:.1f}%", "Unknown"]
            
        # Draw Box (if region was provided by detector backend)
        if area:
            # Handle deepface mapping
            x = area.get('x', 0)
            y = area.get('y', 0)
            w = area.get('w', 200)
            h = area.get('h', 200)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            for j, line in enumerate(labels):
                cv2.putText(frame, line, (x + 5, y - 5 - (len(labels)-j-1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    log.info("Matching complete. Entry logged -> %s", LOG_FILE)
    
    cv2.imshow("Photo Recognition Result - Press any key", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
