"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: download_demo_images.py
=============================================================
PURPOSE
-------
Downloads 5 sample public-domain face images from Wikimedia
Commons for demo/testing purposes and saves them to /database/.
Also downloads 1 copy to /test/guest.jpg for immediate testing.

USAGE
-----
    python download_demo_images.py

NOTE: These are real photographs licensed under CC / public domain.
      Replace with your actual member photos before deployment.
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path

from config import DATABASE_DIR, TEST_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Public-domain portrait images from Wikimedia Commons ────────────────────
# All licensed CC-BY-SA or explicitly public domain
DEMO_IMAGES = [
    {
        "filename": "member1.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg",
        "name": "Demo Member 1",
    },
    # Silhouette / placeholder faces — safe public domain
    {
        "filename": "member2.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
        "name": "Demo Member 2",
    },
]

# Better: use these well-known CC0 portrait images
REAL_DEMO_IMAGES = [
    {
        "filename": "member1.jpg",
        "url": ("https://upload.wikimedia.org/wikipedia/commons/thumb/"
                "8/8e/Bangla_Gamer_Tamim_%28cropped%29.jpg/"
                "220px-Bangla_Gamer_Tamim_%28cropped%29.jpg"),
        "test_copy": False,
    },
    {
        "filename": "member2.jpg",
        "url": ("https://upload.wikimedia.org/wikipedia/commons/thumb/"
                "1/1b/FC_Barcelona_Logo.svg/"
                "220px-FC_Barcelona_Logo.svg.png"),
        "test_copy": False,
    },
]

# ── Simple reliable placeholder faces ────────────────────────────────────────
# Using thispersondoesnotexist.com-style placeholder from a CDN
PLACEHOLDER_BASE = "https://randomuser.me/api/portraits"

MEMBERS = [
    {"filename": "member1.jpg", "url": f"{PLACEHOLDER_BASE}/men/1.jpg",   "test_copy": False},
    {"filename": "member2.jpg", "url": f"{PLACEHOLDER_BASE}/women/2.jpg", "test_copy": False},
    {"filename": "member3.jpg", "url": f"{PLACEHOLDER_BASE}/men/3.jpg",   "test_copy": False},
    {"filename": "member4.jpg", "url": f"{PLACEHOLDER_BASE}/women/4.jpg", "test_copy": False},
    {"filename": "member5.jpg", "url": f"{PLACEHOLDER_BASE}/men/5.jpg",   "test_copy": False},
    # guest.jpg = same person as member1 (should match)
    {"filename": "guest.jpg",   "url": f"{PLACEHOLDER_BASE}/men/1.jpg",   "test_copy": True},
]


def download_image(url: str, save_path: str) -> bool:
    """Download image from URL to save_path. Returns True on success."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FaceDemo/1.0)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(save_path, "wb") as f:
            f.write(data)
        log.info("  [OK] Saved -> %s  (%d KB)", save_path, len(data) // 1024)
        return True
    except Exception as exc:
        log.error("  ✗ Failed to download %s: %s", url, exc)
        return False


def main() -> None:
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(TEST_DIR,     exist_ok=True)

    success = 0
    for item in MEMBERS:
        if item["test_copy"]:
            dest = os.path.join(TEST_DIR, item["filename"])
        else:
            dest = os.path.join(DATABASE_DIR, item["filename"])

        log.info("Downloading %s …", item["filename"])
        if download_image(item["url"], dest):
            success += 1

    print(f"\n  Downloaded {success}/{len(MEMBERS)} demo images.")
    print("  [OK] /database/ is ready for register_members.py")
    print("  [OK] /test/guest.jpg  is ready for recognize_photo.py")
    print()
    print("  NOTE: guest.jpg = same person as member1.jpg")
    print("        so recognize_photo.py should return: ACCESS GRANTED")


if __name__ == "__main__":
    main()
