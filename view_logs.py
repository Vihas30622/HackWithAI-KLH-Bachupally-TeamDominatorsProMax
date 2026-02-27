"""
=============================================================
  Premium Guest Face-Recognition Entry System
  Script: view_logs.py
=============================================================
PURPOSE
-------
Pretty-prints the entry log CSV in the terminal.
Optionally filter by decision, date, or member ID.

USAGE
-----
    python view_logs.py
    python view_logs.py --decision GRANTED
    python view_logs.py --id M001
    python view_logs.py --date 2026-02-27
    python view_logs.py --tail 20
"""

import os
import csv
import argparse
from datetime import datetime

from config import LOG_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View entry logs.")
    parser.add_argument("--decision", choices=["GRANTED", "DENIED"],
                        help="Filter by decision.")
    parser.add_argument("--id",   dest="member_id",
                        help="Filter by member ID.")
    parser.add_argument("--date", help="Filter by date (YYYY-MM-DD).")
    parser.add_argument("--tail", type=int,
                        help="Show only the last N entries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(LOG_FILE):
        print("[LOG] No log file found yet. Run a recognition script first.")
        return

    rows = []
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.decision and row["Decision"] != args.decision:
                continue
            if args.member_id and row["MemberID"] != args.member_id:
                continue
            if args.date and not row["Timestamp"].startswith(args.date):
                continue
            rows.append(row)

    if args.tail:
        rows = rows[-args.tail:]

    if not rows:
        print("[LOG] No matching entries found.")
        return

    # Column widths
    col_w = [19, 8, 20, 15, 10, 8]
    headers = ["Timestamp", "MemberID", "Name", "MembershipLevel",
               "Distance", "Decision"]

    sep   = "─" * (sum(col_w) + len(col_w) * 3 + 1)
    fmt   = lambda vals: " | ".join(
        str(v)[:w].ljust(w) for v, w in zip(vals, col_w)
    )

    print("\n" + sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        vals = [row[h] for h in headers]
        line = fmt(vals)
        if row["Decision"] == "GRANTED":
            print(f"\033[92m{line}\033[0m")   # Green
        else:
            print(f"\033[91m{line}\033[0m")   # Red
    print(sep)
    print(f"\n  Total entries shown: {len(rows)}\n")


if __name__ == "__main__":
    main()
