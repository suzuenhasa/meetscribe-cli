#!/usr/bin/env python3
"""Re-key the profile store from filenames to meeting ids.

  migrate_ids.py            what would change, touching nothing
  migrate_ids.py --apply    do it, after backing the store up

speakers.db used to record which meeting a voice was enrolled from, and which
cluster a decision was about, by the SANITISED FILENAME:

    prototypes.meeting = "One-Trust-Network-to-Rule-them-All:G02"
    decisions.meeting  = "One-Trust-Network-to-Rule-them-All"

which made the filename the identity. Rename a recording and every decision ever
made about it is orphaned; two recordings sanitising to one name share a history
that belongs to neither. Meetings carry a short random id now, so the store keys
on that instead and none of it can happen.

Reversible: the store is copied to speakers.db.pre-ids first, and a legacy_name
column keeps what each row used to say. Rows that cannot be matched to a meeting
in the library are LEFT ALONE and reported -- a decision about a recording you
no longer have is still a fact about a person, and dropping it silently to make
a migration look clean would be the wrong trade.
"""
import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import library as LIB
import speakers as S


def index_by_legacy_name(lib=None):
    """{sanitised name: meeting} for everything in the library.

    The old key was the transcript's filename run through the same sanitiser
    ./transcribe used, so that is what has to be reconstructed to match rows
    against meetings."""
    out = {}
    for m in LIB.all_meetings(lib):
        d = m.read()
        for cand in (d.get("source", ""), d.get("title", ""), m.stem):
            if not cand:
                continue
            stem = Path(cand).stem
            safe = "".join(c if c.isalnum() or c in "._-" else "-"
                           for c in stem).strip("-")
            if safe:
                out.setdefault(safe, m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default=None)
    ap.add_argument("--library", default=None)
    a = ap.parse_args()

    db = Path(a.db or S.DB)
    if not db.exists():
        print(f"no store at {db} — nothing to migrate")
        return 0
    known = index_by_legacy_name(a.library)
    if not known:
        print("no meetings in the library to match against; run transcribe first")
        return 0

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    plan, unmatched = [], []
    for table, col in (("prototypes", "meeting"), ("decisions", "meeting")):
        try:
            rows = conn.execute(f"SELECT rowid AS rid, {col} FROM {table}").fetchall()
        except sqlite3.Error:
            continue
        for r in rows:
            old = r[col] or ""
            name, _, cluster = old.partition(":")
            m = known.get(name)
            if m is None:
                if old:
                    unmatched.append((table, old))
                continue
            new = f"{m.id}:{cluster}" if cluster else m.id
            if new != old:
                plan.append((table, r["rid"], old, new))

    print(f"store:   {db}")
    print(f"library: {len(known)} name(s) resolve to meetings")
    print(f"rows to re-key: {len(plan)}")
    for t, _, old, new in plan[:8]:
        print(f"    {t:11} {old[:44]:44} -> {new}")
    if len(plan) > 8:
        print(f"    ... and {len(plan) - 8} more")
    if unmatched:
        print(f"left alone, no meeting in the library: {len(unmatched)}")
        for t, old in unmatched[:5]:
            print(f"    {t:11} {old[:60]}")

    if not a.apply:
        print("\nnothing changed. Re-run with --apply to do it.")
        return 0
    if not plan:
        print("\nnothing to do.")
        return 0

    backup = db.with_suffix(db.suffix + ".pre-ids")
    shutil.copy2(db, backup)
    print(f"\nbacked up to {backup}")
    for table in ("prototypes", "decisions"):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN legacy_name TEXT")
        except sqlite3.Error:
            pass                      # already there; re-running is allowed
    for table, rowid, old, new in plan:
        conn.execute(f"UPDATE {table} SET meeting=?, legacy_name=COALESCE(legacy_name, ?)"
                     f" WHERE rowid=?", (new, old, rowid))
    conn.commit()
    print(f"re-keyed {len(plan)} rows. The old value of each is in legacy_name,"
          f"\nand {backup.name} is the store as it was.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
