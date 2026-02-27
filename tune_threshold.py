"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: tune_threshold.py
=============================================================
PURPOSE
-------
Helper utility to find the optimal cosine-distance threshold
for your specific set of member faces.

HOW IT WORKS
------------
1. Loads all stored embeddings.
2. Computes a pairwise distance matrix (genuine pairs + impostor pairs).
3. Sweeps threshold from 0.20 to 0.70 and reports FAR/FRR at each step.
4. Suggests the EER (Equal Error Rate) threshold as a starting point.
5. Optionally plots a ROC-like curve (requires matplotlib).

USAGE
-----
    python tune_threshold.py
    python tune_threshold.py --plot   # show matplotlib graph
"""

from __future__ import annotations
import os
import sys
import pickle
import argparse
import itertools
import logging
from typing import Optional

import numpy as np

from config import EMBEDDINGS_FILE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    if a_n == 0 or b_n == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_n * b_n))


def load_embeddings() -> dict:
    if not os.path.exists(EMBEDDINGS_FILE):
        log.error("No embeddings.pkl found. Run register_members.py first.")
        sys.exit(1)
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def compute_pairwise_distances(embeddings: dict) -> tuple[list, list]:
    """
    Returns:
        genuine_dists  – distances between same-label pairs (if any duplicates).
        impostor_dists – distances between different-member pairs.

    NOTE: With only 1 image per member there are NO genuine pairs.
          In that case genuine_dists will be empty and we can only
          inspect the impostor distribution to pick a safe threshold.
    """
    ids   = list(embeddings.keys())
    embs  = {k: v["embedding"] for k, v in embeddings.items()}
    names = {k: v["name"]      for k, v in embeddings.items()}

    genuine_dists  = []
    impostor_dists = []

    for id_a, id_b in itertools.combinations(ids, 2):
        dist = cosine_distance(embs[id_a], embs[id_b])
        # With 1 image per member, all pairs are impostors
        impostor_dists.append((id_a, id_b, dist))

    return genuine_dists, impostor_dists


def threshold_sweep(
    genuine_dists: list,
    impostor_dists: list,
    steps: int = 50
) -> tuple[float, list]:
    """
    Sweep threshold and return (eer_threshold, stats_list).
    stats_list: [(threshold, FAR, FRR), ...]
    """
    imp_values = [d for _, _, d in impostor_dists]
    gen_values = [d for _, _, d in genuine_dists] if genuine_dists else []

    thresholds = np.linspace(0.20, 0.70, steps)
    stats      = []

    for thr in thresholds:
        # FAR: fraction of impostors ACCEPTED (dist < thr → accepted)
        fa    = sum(1 for d in imp_values if d < thr)
        far   = fa / len(imp_values) if imp_values else 0.0

        # FRR: fraction of genuines REJECTED (dist >= thr → rejected)
        fr    = sum(1 for d in gen_values if d >= thr)
        frr   = fr / len(gen_values) if gen_values else float("nan")

        stats.append((round(float(thr), 3), round(far, 4),
                      round(frr, 4) if gen_values else float("nan")))

    # EER: where FAR ≈ FRR
    eer_thr = 0.40  # sensible default
    if gen_values:
        best_diff = float("inf")
        for thr, far, frr in stats:
            diff = abs(far - frr)
            if diff < best_diff:
                best_diff = diff
                eer_thr   = thr

    return eer_thr, stats


def print_stats(stats: list, eer_thr: float, has_genuine: bool) -> None:
    print("\n" + "═" * 60)
    print(f"  THRESHOLD SWEEP  (EER ≈ {eer_thr:.3f})")
    print("═" * 60)
    print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}")
    print("─" * 60)
    for thr, far, frr in stats[::5]:    # Print every 5th row
        frr_str = f"{frr:.4f}" if not (isinstance(frr, float) and np.isnan(frr)) else " N/A  "
        flag    = " ← EER" if abs(thr - eer_thr) < 0.001 else ""
        print(f"  {thr:>10.3f}  {far:>8.4f}  {frr_str:>8}{flag}")
    print("─" * 60)
    if not has_genuine:
        print("  NOTE: No genuine pairs (only 1 image/member).")
        print("  FRR could not be computed.  FAR shows impostor distribution.")
        # Recommend a threshold just below the minimum impostor distance
    print("═" * 60 + "\n")


def suggest_threshold(impostor_dists: list) -> float:
    """
    With only impostor pairs available, suggest a threshold
    slightly below the minimum impostor distance for strict security.
    """
    if not impostor_dists:
        return 0.40
    min_dist = min(d for _, _, d in impostor_dists)
    return round(max(min_dist - 0.02, 0.20), 3)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold tuning utility for the face-recognition system."
    )
    parser.add_argument("--plot", action="store_true",
                        help="Show ROC-like matplotlib plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embeddings = load_embeddings()
    log.info("Loaded %d member(s).", len(embeddings))

    genuine_dists, impostor_dists = compute_pairwise_distances(embeddings)
    has_genuine = bool(genuine_dists)

    log.info("Impostor pairs: %d | Genuine pairs: %d",
             len(impostor_dists), len(genuine_dists))

    eer_thr, stats = threshold_sweep(genuine_dists, impostor_dists)

    # Print impostor distance distribution
    imp_vals = [d for _, _, d in impostor_dists]
    if imp_vals:
        print("\n  Impostor Distance Distribution:")
        print(f"    Min : {min(imp_vals):.4f}")
        print(f"    Max : {max(imp_vals):.4f}")
        print(f"    Mean: {np.mean(imp_vals):.4f}")
        print(f"    Std : {np.std(imp_vals):.4f}")

    print_stats(stats, eer_thr, has_genuine)

    suggested = suggest_threshold(impostor_dists)
    print(f"  ✔  Recommended threshold for your database: {suggested:.3f}")
    print(f"     Update DISTANCE_THRESHOLD in config.py to: {suggested}")
    print()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            thrs = [s[0] for s in stats]
            fars = [s[1] for s in stats]
            plt.figure(figsize=(8, 5))
            plt.plot(thrs, fars, "r-o", markersize=4, label="FAR (Impostor)")
            if has_genuine:
                frrs = [s[2] for s in stats]
                plt.plot(thrs, frrs, "b-o", markersize=4, label="FRR (Genuine)")
            plt.axvline(suggested, color="green", linestyle="--",
                        label=f"Suggested threshold ({suggested})")
            plt.xlabel("Threshold")
            plt.ylabel("Rate")
            plt.title("FAR / FRR vs Threshold")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()
        except ImportError:
            log.warning("matplotlib not installed. Skipping plot.")


if __name__ == "__main__":
    main()
