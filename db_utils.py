"""
=============================================================
  Premium Guest Face-Recognition Entry System — DB Utils
=============================================================
SQLite helper: create schema, insert members, query members.
"""

from __future__ import annotations
import sqlite3
import os
from config import DB_FILE


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def create_schema() -> None:
    """Create the members table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS members (
            member_id        TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            membership_level TEXT NOT NULL,
            image_path       TEXT NOT NULL,
            registered_on    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Schema ready.")


def insert_member(member_id: str, name: str, level: str, image_path: str) -> None:
    """
    Insert or replace a member record.

    Args:
        member_id:  Unique ID string (e.g. 'M001').
        name:       Full name.
        level:      Membership tier (e.g. 'Platinum', 'Gold').
        image_path: Absolute path to the member's photo.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO members
            (member_id, name, membership_level, image_path)
        VALUES (?, ?, ?, ?)
    """, (member_id, name, level, image_path))
    conn.commit()
    conn.close()


def get_all_members() -> list[dict]:
    """Return all registered members as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM members ORDER BY registered_on DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_member_by_id(member_id: str) -> dict | None:
    """Return a single member dict or None."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM members WHERE member_id = ?", (member_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_member(member_id: str) -> None:
    """Remove a member from the database and embeddings."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM members WHERE member_id = ?", (member_id,))
    conn.commit()
    conn.close()
    print(f"[DB] Member {member_id} deleted.")


if __name__ == "__main__":
    create_schema()
    all_members = get_all_members()
    print(f"[DB] {len(all_members)} member(s) registered.")
    for m in all_members:
        print(f"  {m['member_id']} | {m['name']} | {m['membership_level']}")
