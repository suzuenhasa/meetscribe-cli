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
import speakers as SPK

PIPE = os.path.dirname(os.path.abspath(__file__))


def names_in(m):
    """{cluster: name} as the transcript currently claims."""
    try:
        return json.loads(m.file("names", "json").read_text())
    except (OSError, ValueError):
        return {}


def names_from_groups(conn, m):
    """{cluster: name} for one meeting, from the linked groups. -> dict

    A cluster is named only if linking put it in a group and a human named that
    group. Anything else stays unnamed, which is the honest answer: the group
    system already refused to guess, and re-deriving here would overturn that.
    """
    rows = conn.execute(
        "SELECT c.cluster, s.name FROM clusters c "
        "JOIN groups g ON g.id = c.group_id "
        "JOIN speakers s ON s.id = g.speaker_id "
        "WHERE c.meeting = ? AND c.embed_model = ?",
        (m.id, SPK.EMBED_MODEL)).fetchall()
    return {cl: nm for cl, nm in rows if nm}


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

    conn = SPK.db()
    n_groups = conn.execute(
        "SELECT COUNT(*) FROM groups WHERE speaker_id IS NOT NULL").fetchone()[0]
    if not n_groups:
        print("no named groups in the store -- run: speakers.py link --apply, "
              "then speakers.py name <group> \"Name\"")
        return 0
    changed, skipped = [], []
    tmpdir = tempfile.mkdtemp(prefix="ms-relabel-")
    for m in ms:
        if not m.file("clusters", "npz").exists():
            skipped.append((m, "no clusters — transcribe it again"))
            continue
        before = names_in(m)
        # RENDER what linking decided; do not decide again.
        #
        # This used to run identify.py per meeting, re-scoring every cluster
        # against the gallery at ACCEPT (0.55). That is a second, looser opinion
        # competing with speakers.py link, which groups clusters from the audio
        # and only names a group at LINK_ACCEPT (0.75). The two disagreed and
        # this one won, because it is what writes the transcript: measured on a
        # SCOTUS argument, linking put a cluster in its own unnamed group -- 0.911
        # to its own members, 0.586 to the nearest named one, correctly declining
        # -- and relabel wrote that name in anyway because 0.586 clears 0.55.
        # A wrong name on a transcript is worse than no name, and it was being
        # applied over a correct abstention.
        #
        # Naming is a decision made once, in one place. Here we only write it out.
        scratch = Path(tmpdir) / f"{m.id}.json"
        after = names_from_groups(conn, m)
        scratch.write_text(json.dumps(after, indent=1))
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
