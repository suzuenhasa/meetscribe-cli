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
import match_speakers as MS
import postproc
import speakers as SPK

PIPE = os.path.dirname(os.path.abspath(__file__))


def names_in(m):
    """{cluster: name} as the transcript currently claims."""
    try:
        return json.loads(m.file("names", "json").read_text())
    except (OSError, ValueError):
        return {}


def relabel_by_matching(m, bank, condition=None):
    """Re-label one meeting against the store, from MOSS's own labels. -> dict

    The group path below can only rename CLUSTERS -- whatever the clustering
    already decided, right or wrong -- and it compares averaged cluster
    centroids. This goes back to the atoms: one vector per (window, local
    label), which is the purest unit available, 93% of them >=90% one speaker.
    Measured, that is the difference between 2.63% wrong names and 0.18%.

    Rewrites the linked transcript as well as the name map, because matching can
    conclude that two things the clustering separated are one person, and that
    has to reach the file a reader opens.
    """
    import numpy as np

    raw = json.loads(m.file("raw", "json").read_text())
    z = np.load(str(m.file("embeddings", "npz")), allow_pickle=True)
    atoms = MS.atoms_from(raw["segments"], z["emb"], z["seg_idx"])
    if not atoms:
        return {}
    names, prov, sim = MS.assign(atoms, bank)

    order, out, tag_of = {}, {}, {}
    for i, a in enumerate(atoms):
        tag = names[i] or prov[i]
        if tag is None:
            continue
        if tag not in order:
            order[tag] = "G%02d" % len(order)
            if names[i]:
                out[order[tag]] = names[i]
        tag_of[a["key"]] = order[tag]
    for seg in raw["segments"]:
        g = tag_of.get((int(seg["window"]), seg["local_speaker"]))
        seg["global"] = g
        if g and g in out:
            seg["speaker_name"] = out[g]
        else:
            seg.pop("speaker_name", None)
    m.file("transcript", "json").write_text(json.dumps(raw))

    # Matching renumbers `global`, so the clusters table -- written at link time
    # -- now names different things by the same ids. Anything joining the two
    # afterwards silently reads one meeting's G07 as another's: it put a
    # justice's entire 15 hours under a colleague's name before this line
    # existed. Rewrite the sidecar so re-indexing puts the store back in step.
    cents, secs, ids = [], [], []
    for tag, gid in sorted(order.items(), key=lambda kv: kv[1]):
        mine = [i for i, a in enumerate(atoms)
                if tag_of.get(a["key"]) == gid]
        if not mine:
            continue
        w = np.array([atoms[i]["sec"] for i in mine], dtype=np.float32)
        V = np.stack([atoms[i]["v"] for i in mine])
        cents.append(MS.unit((V * w[:, None]).sum(axis=0)))
        secs.append(float(w.sum()))
        ids.append(gid)
    if cents:
        np.savez(str(m.file("clusters", "npz")),
                 centroid=np.stack(cents).astype(np.float32),
                 cluster=np.array(ids), secs=np.array(secs, dtype=np.float32),
                 meeting=np.array(m.id))
    return out


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
    ap.add_argument("--roster-file", default=None, dest="roster_file",
                    help="JSON mapping each meeting to who could be in it: "
                         '{\"<meeting id or title>\": [\"Ada\", \"Bo\"]}. '
                         "One global --roster cannot describe a library where "
                         "every recording has different people in it, which is "
                         "every real library. A calendar integration produces "
                         "exactly this shape.")
    ap.add_argument("--condition", default=None,
                    help="circumstance to record for anyone recognised here")
    ap.add_argument("--from-groups", action="store_true",
                    help="rename clusters from the linked groups instead of "
                         "re-matching MOSS's labels against the store. What this "
                         "did before there was anything to match against.")
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
    # The store's named voices, as sub-profiles. Empty, or --from-groups, and
    # this falls back to renaming clusters from the linked groups.
    # --roster was declared here and never used. Restricting the gallery to who
    # could actually be in the recording is the cheapest accuracy there is: it
    # removes impostor trials that were never going to be right.
    _names = [x.strip() for x in a.roster.split(",") if x.strip()]
    _per_meeting = {}
    if a.roster_file:
        _per_meeting = json.loads(Path(a.roster_file).read_text())
    # One bank per DISTINCT roster, not per meeting: loading it is a query and
    # a library of 300 recordings sharing a dozen rosters should pay for a dozen.
    _banks = {}

    def bank_for(m):
        if a.from_groups:
            return None
        who = _per_meeting.get(m.id) or _per_meeting.get(m.title) or _names
        key = tuple(sorted(who)) if who else None
        if key not in _banks:
            _banks[key] = MS.load_bank(conn, SPK.EMBED_MODEL,
                                       names=list(key) if key else None)
        return _banks[key]

    bank = None if a.from_groups else MS.load_bank(conn, SPK.EMBED_MODEL,
                                                   names=_names or None)
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
        _b = bank_for(m)
        if _b is not None and len(_b) and m.file("embeddings", "npz").exists():
            after = relabel_by_matching(m, _b, a.condition)
        else:
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
    # Matching renumbered `global`, and the clusters table was indexed when
    # link ran. Leaving them disagreeing is how a name lands on the wrong person:
    # it once filed a justice's fifteen hours under a colleague, and it silently
    # blanks the samples `review` shows you. Re-index what we just rewrote so the
    # store is consistent when this returns, rather than until someone
    # remembers to re-link.
    try:
        n = SPK.index_clusters(conn)
        # Re-indexing correctly clears group_id wherever the vector changed --
        # an identity must follow the voice, not the label. But matching just
        # decided who each of these clusters IS, and throwing that away leaves
        # named people with no clusters at all: `review` then reports nothing
        # waiting while `profiles` shows sub-profiles backed by no recordings.
        # Write the decision down instead.
        back = 0
        for m in ms:
            after = json.loads(m.file("names", "json").read_text() or "{}") \
                if m.file("names", "json").exists() else {}
            for cluster, who in after.items():
                row = conn.execute(
                    "SELECT g.id FROM groups g JOIN speakers s ON s.id ="
                    " g.speaker_id WHERE s.name=? AND g.embed_model=?",
                    (who, SPK.EMBED_MODEL)).fetchone()
                if not row:
                    continue
                back += conn.execute(
                    "UPDATE clusters SET group_id=? WHERE meeting=? AND"
                    " cluster=? AND embed_model=? AND group_id IS NULL",
                    (row[0], m.id, cluster, SPK.EMBED_MODEL)).rowcount
        conn.commit()
        print(f"re-indexed {n} clusters, {back} re-attached to the person "
              f"matching named them.")
    except Exception as e:
        print(f"  !! could not re-index the store: {e}")
        print("     run `speakers.py link --apply` before naming anyone.")
    print(f"\nre-rendered {len(changed)} transcript(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
