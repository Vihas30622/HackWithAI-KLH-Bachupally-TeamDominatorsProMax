"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: test_system.py
=============================================================
PURPOSE
-------
Quick sanity-test:
  1. Validates that all dependencies can be imported.
  2. Checks that config paths are consistent.
  3. Verifies embeddings.pkl integrity.
  4. Runs a self-match test: each member image should match itself.
  5. Reports pass / fail for each check.

USAGE
-----
    python test_system.py
"""

import os
import sys
import pickle
import logging

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"

results = []   # (label, passed: bool)


def check(label: str, cond: bool, warn: str = "") -> None:
    symbol = PASS if cond else FAIL
    print(f"{symbol}  {label}")
    if not cond and warn:
        print(f"         → {warn}")
    results.append((label, cond))


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  PREMIUM FACE ENTRY SYSTEM — SYSTEM TEST")
print("═" * 60)

# 1. Dependency imports
print("\n[1] Checking imports …")
try:
    import cv2
    check("opencv-python", True)
except ImportError as e:
    check("opencv-python", False, str(e))

try:
    from deepface import DeepFace
    check("deepface", True)
except ImportError as e:
    check("deepface", False, str(e))

try:
    import numpy
    check("numpy", True)
except ImportError as e:
    check("numpy", False, str(e))

try:
    import sqlite3
    check("sqlite3 (stdlib)", True)
except ImportError as e:
    check("sqlite3 (stdlib)", False, str(e))

# 2. Config paths
print("\n[2] Checking config …")
from config import (DATABASE_DIR, TEST_DIR, EMBEDDINGS_FILE,
                    DB_FILE, LOG_FILE, LOGS_DIR, MODEL_NAME,
                    DETECTOR_BACKEND, DISTANCE_THRESHOLD)

check("DATABASE_DIR exists",    os.path.isdir(DATABASE_DIR),
      f"Create: {DATABASE_DIR}")
check("TEST_DIR exists",        os.path.isdir(TEST_DIR),
      f"Create: {TEST_DIR}")
check("MODEL_NAME set",         bool(MODEL_NAME))
check("DETECTOR_BACKEND set",   bool(DETECTOR_BACKEND))
check("DISTANCE_THRESHOLD > 0", DISTANCE_THRESHOLD > 0)

# 3. Photo-in-database check
print("\n[3] Checking database images …")
img_exts  = (".jpg", ".jpeg", ".png", ".bmp")
db_images = [f for f in os.listdir(DATABASE_DIR)
             if f.lower().endswith(img_exts)] if os.path.isdir(DATABASE_DIR) else []
check(f"At least 1 image in /database/ (found {len(db_images)})",
      len(db_images) >= 1,
      "Run download_demo_images.py to get sample images.")

guest_img = os.path.join(TEST_DIR, "guest.jpg")
check("test/guest.jpg exists",  os.path.isfile(guest_img),
      "Run download_demo_images.py or copy a face image here.")

# 4. Embeddings integrity
print("\n[4] Checking embeddings.pkl …")
if not os.path.exists(EMBEDDINGS_FILE):
    print(f"{SKIP}  embeddings.pkl not found — run register_members.py first.")
    results.append(("embeddings.pkl exists", False))
else:
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            emb_data = pickle.load(f)
        check("embeddings.pkl is valid pickle", True)
        check(f"Contains ≥1 member (found {len(emb_data)})", len(emb_data) >= 1)

        # Validate each entry
        for mid, data in emb_data.items():
            has_emb  = ("embedding" in data and
                        isinstance(data["embedding"], np.ndarray))
            has_meta = all(k in data for k in ("name", "level", "image_path"))
            check(f"Member {mid}: valid embedding", has_emb)
            check(f"Member {mid}: valid metadata", has_meta)

    except Exception as exc:
        check("embeddings.pkl is valid pickle", False, str(exc))

# 5. SQLite check
print("\n[5] Checking SQLite database …")
if not os.path.exists(DB_FILE):
    print(f"{SKIP}  database.db not found — run register_members.py first.")
else:
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM members")
        count = cursor.fetchone()[0]
        conn.close()
        check(f"SQLite members table accessible (rows: {count})", True)
    except Exception as exc:
        check("SQLite database accessible", False, str(exc))

# 6. Self-match test
print("\n[6] Self-match test (each member should match itself) …")
if not os.path.exists(EMBEDDINGS_FILE):
    print(f"{SKIP}  No embeddings — skipping self-match test.")
else:
    from recognize_photo import cosine_distance, load_embeddings

    embeddings = load_embeddings()
    all_pass   = True
    for mid, data in embeddings.items():
        emb  = data["embedding"]
        dist = cosine_distance(emb, emb)   # must be ~0
        ok   = dist < 0.001
        check(f"Self-match {mid} ({data['name']}): dist={dist:.6f}", ok,
              "Embedding may be corrupted if this fails.")
        all_pass = all_pass and ok

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
pass_count = sum(1 for _, ok in results if ok)
fail_count = len(results) - pass_count
print(f"  Result: {pass_count} passed, {fail_count} failed out of {len(results)} checks")
print("═" * 60 + "\n")

if fail_count > 0:
    sys.exit(1)
