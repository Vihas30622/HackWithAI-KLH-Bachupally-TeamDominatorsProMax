"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: register_members.py
=============================================================
PURPOSE
-------
Scans the /database/ folder for .jpg / .png images, detects
faces in each image using DeepFace, generates face embeddings,
stores them in embeddings.pkl, and records member details in
the SQLite database.

USAGE
-----
1. Place member photos in /database/  (named member1.jpg, etc.)
2. Edit the MEMBER_CATALOG list below with member metadata.
3. Run:  python register_members.py

NOTE: Re-running overwrites existing embeddings for the same ID.
"""

from __future__ import annotations
import os
import sys
import pickle
import logging
from datetime import datetime

import numpy as np
from deepface import DeepFace

from config import (DATABASE_DIR, EMBEDDINGS_FILE, MODEL_NAME,
                    DETECTOR_BACKEND, FORCE_CPU)
from db_utils import create_schema, insert_member

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Optional: force CPU ──────────────────────────────────────────────────────
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── Member catalog ────────────────────────────────────────────────────────────
# Add an entry for every image in /database/.
# Format: (member_id, name, membership_level, filename_in_database_dir)
MEMBER_CATALOG = [
    ("M001", "Alice Johnson",  "Premium", "member1.jpg"),
    ("M002", "Bob Williams",   "Premium", "member2.jpg"),
    ("M003", "Carol Davis",    "Premium", "member3.jpg"),
    ("M004", "David Martinez", "Premium", "member4.jpg"),
    ("M005", "Eva Chen",       "Premium", "member5.jpg"),
    ("M006", "Koushik",        "Premium", "koushik.jpg"),   # Real member
    ("M007", "Vihas",          "Premium", "vihas.jpg"),     # Real member
    ("M008", "Manoj",          "Premium", "manoj.jpg"),     # Real member
    ("M009", "Lanja",          "Premium", "lanja.jpg"),     # Real member
]


# ─────────────────────────────────────────────────────────────────────────────
def generate_embedding(image_path: str) -> np.ndarray | None:
    """
    Detect a face in image_path and return its embedding vector.

    Returns:
        np.ndarray of shape (embedding_dim,) or None on failure.

    Raises handled:
        - No face detected
        - Multiple faces detected (uses the first one)
        - Corrupted / unreadable image
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,   # raises if no face found
        )
        # DeepFace returns a list of dicts; use first detected face
        if len(result) == 0:
            log.warning("No embedding returned for %s", image_path)
            return None
        if len(result) > 1:
            log.warning("Multiple faces in %s — using the largest region.",
                        image_path)
        # Pick the face with the largest bounding-box area
        best = max(result,
                   key=lambda r: r["facial_area"]["w"] * r["facial_area"]["h"])
        return np.array(best["embedding"], dtype=np.float32)

    except ValueError as exc:
        log.error("Face not detected in %s: %s", image_path, exc)
        return None
    except Exception as exc:
        log.error("Error processing %s: %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
def load_existing_embeddings() -> dict:
    """Load embeddings.pkl if it exists, else return empty dict."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_embeddings(embeddings: dict) -> None:
    """Persist the embeddings dictionary to disk."""
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    log.info("Embeddings saved -> %s", EMBEDDINGS_FILE)


# ─────────────────────────────────────────────────────────────────────────────
def register_all_members() -> None:
    """
    Main registration routine.
    Iterates MEMBER_CATALOG, generates embeddings, updates SQLite + pkl.
    """
    create_schema()
    embeddings = load_existing_embeddings()

    success_count = 0
    fail_count    = 0

    for member_id, name, level, filename in MEMBER_CATALOG:
        image_path = os.path.join(DATABASE_DIR, filename)

        if not os.path.isfile(image_path):
            log.warning("Image not found: %s  — skipping %s.", image_path, name)
            fail_count += 1
            continue

        log.info("Processing %s (%s) …", name, member_id)
        embedding = generate_embedding(image_path)

        if embedding is None:
            log.error("  [FAIL] Registration FAILED for %s.", name)
            fail_count += 1
            continue

        # Store in pkl dict
        embeddings[member_id] = {
            "embedding": embedding,
            "name":      name,
            "level":     level,
            "image_path": image_path,
        }

        # Store in SQLite
        insert_member(member_id, name, level, image_path)

        log.info("  [OK] Registered %s | %s | %s", member_id, name, level)
        success_count += 1

    save_embeddings(embeddings)

    print("\n" + "=" * 55)
    print(f"  Registration complete: {success_count} success, {fail_count} failed")
    print(f"  Model : {MODEL_NAME}")
    print(f"  DB    : {EMBEDDINGS_FILE}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    register_all_members()
