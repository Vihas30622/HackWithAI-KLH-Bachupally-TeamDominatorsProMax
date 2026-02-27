"""
=============================================================
  Premium Guest Face-Recognition Entry System — Config
=============================================================
Central configuration for all scripts.
"""

import os

# ── Paths ─────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR    = os.path.join(BASE_DIR, "database")
TEST_DIR        = os.path.join(BASE_DIR, "test")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.pkl")
DB_FILE         = os.path.join(BASE_DIR, "database.db")
LOG_FILE        = os.path.join(LOGS_DIR, "entry_log.csv")

# ── Face-Recognition Model ────────────────────────────────
# Options: "Facenet", "Facenet512", "VGG-Face", "ArcFace", "DeepFace"
MODEL_NAME      = "ArcFace"      # InsightFace model: Absolute state-of-the-art for preventing false-matches

# ── Face Detector ─────────────────────────────────────────
# Options: "opencv", "mtcnn", "retinaface", "mediapipe", "ssd"
DETECTOR_BACKEND = "retinaface"       # Slower but most accurate for live feeds

# ── Similarity / Distance ─────────────────────────────────
# cosine distance: 0 = identical, 1 = completely different
# A lower threshold = stricter matching
DISTANCE_METRIC  = "cosine"
DISTANCE_THRESHOLD = 0.75           # ArcFace: 0.75 allows movement/light changes safely

# ── Display ───────────────────────────────────────────────
FONT_SCALE      = 0.8
THICKNESS       = 2
COLOR_GRANTED   = (0, 255, 100)     # Green
COLOR_DENIED    = (0, 0, 255)       # Red
COLOR_UNKNOWN   = (0, 165, 255)     # Orange

# ── Logging ───────────────────────────────────────────────
LOG_HEADERS = ["Timestamp", "MemberID", "Name", "MembershipLevel",
               "Distance", "Decision"]

# ── GPU / Performance ─────────────────────────────────────
# DeepFace will auto-detect CUDA (RTX 4060) if TF is GPU-compiled.
# Set to True to force CPU if GPU causes errors.
FORCE_CPU = False
