#!/usr/bin/env python3
"""Apply what the profile store knows to every meeting already in the library.

  relabel.py                 what would change, touching nothing
  relabel.py --apply         re-identify and re-render

Identification happens once, when a recording is processed, against whoever was
enrolled at that moment. Name someone afterwards and every meeting they are
already in keeps calling them Speaker 3 -- the store knows who they are, the
transcripts do not, and nothing reconciles the two.

That is the wrong shape for the whole point of this: a voice named ONCE should be
recognised everywhere, including backwards. Enrolling is when the gallery
changes, so this is what to run after it.

Cheap enough not to think about: identification is cosine arithmetic over
centroids that are already on disk. No GPU, no re-transcription, no audio -- a
library of a few hundred meetings is seconds.
"""
import argparse
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import library as LIB
import postproc

PIPE = os.path.dirname(os.path.abspath(__file__))


def names_in(m):
    """{cluster: name} as the transcript currently claims."""
    try:
        return json.loads(m.file("names", "json").read_text())
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--library", default=None)
    ap.add_argument("--roster", default="")
    ap.add_argument("meetings", nargs="*",
                    help="only these; default is every meeting in the library")
    a = ap.parse_args()

    postproc.pool_init(PIPE)
    if a.meetings:
        ms = [m for m in (LIB.find(r, a.library) for r in a.meetings) if m]
    else:
        ms = LIB.all_meetings(a.library)
    if not ms:
        print("no meetings in the library")
        return 0

    changed, skipped = [], []
    tmpdir = tempfile.mkdtemp(prefix="ms-relabel-")
    for m in ms:
        if not m.file("clusters", "npz").exists():
            skipped.append((m, "no clusters — transcribe it again"))
            continue
        before = names_in(m)
        # Identify into a SCRATCH names file. A dry run that rewrote the real one
        # would leave nothing for --apply to notice, so `apply` then `apply
        # --apply` did nothing at all and every transcript kept its old labels
        # while the store insisted they were fine.
        scratch = Path(tmpdir) / f"{m.id}.json"
        rc, out = postproc.run_module((m.id, f"{PIPE}/identify.py", [
            f"{PIPE}/identify.py",
            "--clusters", str(m.file("clusters", "npz")),
            "--meeting", m.id, "--roster", a.roster,
            "--names", str(scratch),
        ]))[1:]
        if rc != 0:
            skipped.append((m, "identify failed"))
            continue
        try:
            after = json.loads(scratch.read_text())
        except (OSError, ValueError):
            after = {}
        if after != before:
            changed.append((m, before, after, scratch))

    for m, before, after, _ in changed:
        gained = {k: v for k, v in after.items() if before.get(k) != v}
        who = ", ".join(f"{k} -> {v}" for k, v in sorted(gained.items()))
        print(f"  {m.path.name[:44]:44} {who}")
    if skipped:
        for m, why in skipped:
            print(f"  {m.path.name[:44]:44} skipped: {why}")
    if not changed:
        print("nothing to change: every transcript already says what the store knows")
        return 0

    if not a.apply:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\n{len(changed)} transcript(s) would change. Re-run with --apply.")
        return 0

    for m, _, _, scratch in changed:
        shutil.copy2(scratch, m.file("names", "json"))
        rc, out = postproc.run_module((m.id, f"{PIPE}/mktxt.py", [
            f"{PIPE}/mktxt.py",
            str(m.file("transcript", "json")), str(m.file("raw", "json")),
            str(m.file("transcript", "txt")), m.title,
            str(m.file("names", "json")),
        ]))[1:]
        if rc != 0:
            print(f"  !! could not re-render {m.path.name}: {out.strip()[:120]}")
    print(f"\nre-rendered {len(changed)} transcript(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
