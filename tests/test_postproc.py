"""The data/state layer: the profile store and the script that rewrites it.

  RELABEL DRY RUN (b405cab). `relabel.py` identified into the REAL names.json to
  work out what would change, so the dry run WAS the change: `apply` followed by
  `apply --apply` then found nothing to do and re-rendered nothing, while
  reporting that every transcript already agreed with the store. It did not.

  CENTROIDS. What the linker's npz is allowed to become a voiceprint of, which
  is the one thing every later naming decision is built on.

This file was three times the size and covered two more things: identify.py's
0.55/0.10/0.40 decision rules, and migrate_ids.py re-keying the store from
filenames onto meeting ids. Both scripts are deleted, along with the
`prototypes` and `decisions` tables they were about.

Vectors here are exact by construction (see at_cosine in conftest). Nothing in
here can reach the real speakers.db or library.
"""
import json
import time
from pathlib import Path

import pytest

from conftest import REF, at_cosine, basis, make_meeting, snapshot, write_clusters_npz

import speakers as S


# =====================================================================
# 1. RELABEL DRY RUN -- commit b405cab
# =====================================================================

@pytest.fixture
def two_meetings(lib):
    """A two-meeting library with one voice, G00, in both.

    The first meeting carries an EMPTY names.json -- what a recording processed
    against an empty store looks like -- and the second has no names.json at all,
    which is the same situation one release earlier. Both shapes have to survive
    a dry run byte for byte.
    """
    a = make_meeting(lib, "Design Review Q3", "Design Review Q3.mp3", "aaa11111",
                     {"G00": (REF, 600.0), "G01": (at_cosine(0.20, 2), 300.0)},
                     names={})
    b = make_meeting(lib, "Roadmap Sync", "Roadmap Sync.mp3", "bbb22222",
                     {"G00": (REF, 400.0), "G01": (at_cosine(0.15, 3), 100.0)})
    return a, b


@pytest.fixture
def alice(conn):
    """One named group spanning G00 of both meetings.

    This is the state relabel.py exists for: a voice named AFTER the meetings it
    appears in were processed, so the store knows and the transcripts do not.

    A row in `speakers` is deliberately not enough. relabel renders what LINKING
    decided, ever since it stopped re-deciding at a looser threshold of its own
    and overturning correct abstentions -- so the GROUP is what has to exist for
    a name to reach a transcript.
    """
    sid = conn.execute("INSERT INTO speakers(name, created_at) VALUES(?,?)",
                       ("Alice Anderson", time.time())).lastrowid
    gid = conn.execute("INSERT INTO groups(speaker_id, embed_model, linked_at)"
                       " VALUES(?,?,?)",
                       (sid, S.EMBED_MODEL, time.time())).lastrowid
    v = at_cosine(0.98, 1).astype("float32")
    for meeting in ("aaa11111", "bbb22222"):
        conn.execute(
            "INSERT INTO clusters(meeting, cluster, emb, dim, embed_model,"
            " seconds, group_id, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (meeting, "G00", v.tobytes(), len(v), S.EMBED_MODEL, 90.0, gid,
             time.time()))
    conn.commit()
    return "Alice Anderson"


def test_dry_run_does_not_touch_the_library(lib, two_meetings, alice, run_pipe):
    a, b = two_meetings
    before = snapshot(lib)

    p = run_pipe("relabel.py", "--library", lib)

    assert "would change" in p.stdout
    assert snapshot(lib) == before, "a dry run wrote into the library"
    # Said twice, because equality of a snapshot is easy to read past: the
    # meeting that had no names.json must still have none, and the one that had
    # an empty one must still be empty.
    assert not b.file("names", "json").exists()
    assert json.loads(a.file("names", "json").read_text()) == {}


def test_two_dry_runs_report_the_same_pending_changes(lib, two_meetings, alice,
                                                      run_pipe):
    first = run_pipe("relabel.py", "--library", lib)
    second = run_pipe("relabel.py", "--library", lib)

    assert first.stdout == second.stdout
    assert "G00 -> Alice Anderson" in first.stdout
    assert first.stdout.count("G00 -> Alice Anderson") == 2
    assert "2 transcript(s) would change" in first.stdout


def test_apply_after_a_dry_run_still_rewrites_the_names(lib, two_meetings, alice,
                                                        run_pipe):
    """The regression itself: the dry run must not consume its own finding."""
    a, b = two_meetings
    dry = run_pipe("relabel.py", "--library", lib)
    assert "2 transcript(s) would change" in dry.stdout

    applied = run_pipe("relabel.py", "--library", lib, "--apply")

    assert "re-rendered 2 transcript(s)." in applied.stdout
    assert "nothing to change" not in applied.stdout
    assert json.loads(a.file("names", "json").read_text()) == {"G00": "Alice Anderson"}
    assert json.loads(b.file("names", "json").read_text()) == {"G00": "Alice Anderson"}


def test_apply_re_renders_the_transcript_a_person_reads(lib, two_meetings, alice,
                                                        run_pipe):
    a, _ = two_meetings
    run_pipe("relabel.py", "--library", lib, "--apply")

    txt = a.file("transcript", "txt").read_text()
    assert "Alice Anderson" in txt
    # G00 spoke most, so it was "Speaker 1" before and must not still be.
    assert "Speaker 1" not in txt
    # ...while the cluster nobody recognised keeps its number.
    assert "Speaker 2" in txt
    assert "2 speakers, 1 named" in txt


def test_apply_is_idempotent(lib, two_meetings, alice, run_pipe):
    run_pipe("relabel.py", "--library", lib, "--apply")
    settled = snapshot(lib)

    again = run_pipe("relabel.py", "--library", lib)
    assert "nothing to change" in again.stdout

    once_more = run_pipe("relabel.py", "--library", lib, "--apply")
    assert "nothing to change" in once_more.stdout
    assert snapshot(lib) == settled


def test_only_the_meetings_named_on_the_command_line_change(lib, two_meetings,
                                                            alice, run_pipe):
    a, b = two_meetings
    p = run_pipe("relabel.py", "--library", lib, "--apply", b.id)

    assert "re-rendered 1 transcript(s)." in p.stdout
    assert json.loads(b.file("names", "json").read_text()) == {"G00": "Alice Anderson"}
    assert json.loads(a.file("names", "json").read_text()) == {}


def test_a_meeting_with_no_clusters_is_skipped_not_failed(lib, two_meetings,
                                                          alice, run_pipe):
    """Half a library missing its npz must not stop the other half."""
    stale = make_meeting(lib, "Old Import", "Old Import.mp3", "ccc33333",
                         clusters=None)

    p = run_pipe("relabel.py", "--library", lib, "--apply")

    assert "skipped:" in p.stdout and "no clusters" in p.stdout
    assert stale.path.name[:44] in p.stdout
    assert "re-rendered 2 transcript(s)." in p.stdout
    assert not stale.file("names", "json").exists()


# =====================================================================
# 2. WHAT COUNTS AS A VOICE -- centroids_from_npz
# =====================================================================

def clusters_file(tmp_path, spec, name="run_clusters.npz"):
    p = tmp_path / name
    write_clusters_npz(p, spec, meeting="m1")
    return p


def test_the_linkers_leftover_bucket_is_never_a_person(tmp_path):
    """G- clusters are segments too short to cluster, not a speaker."""
    p = clusters_file(tmp_path, {"G00": (REF, 300.0),
                                 "G-1": (at_cosine(0.9, 1), 12.0)})
    assert sorted(S.centroids_from_npz(p)) == ["G00"]


def test_a_zero_length_centroid_is_dropped_rather_than_dividing_by_zero(tmp_path):
    p = clusters_file(tmp_path, {"G00": (REF, 300.0),
                                 "G01": (basis(0) * 0.0, 8.0)})
    assert sorted(S.centroids_from_npz(p)) == ["G00"]




# =====================================================================
# 3. FORGETTING A PERSON -- the foreign keys the store declared and never
#    enabled
# =====================================================================
#
# `exemplars` and `groups` have declared FOREIGN KEY(speaker_id) since they
# existed, and sqlite ignores every one of them unless the connection asks for
# enforcement. Reproduced before the fix: after `forget 1`, exemplars auto-1 and
# auto-2 still pointed at speaker 1 and group 2 still claimed it, so `groups`
# printed the person as "-- unnamed --" while `review` skipped them as already
# named -- and naming the next person minted id 1 again and handed them both
# voiceprints. What forget deletes is now written out in cmd_forget rather than
# left to ON DELETE CASCADE: CREATE TABLE IF NOT EXISTS will not retrofit that
# clause onto the store already on disk, and a cascade that only happens on a
# store made this week is the divergence, not the fix.

@pytest.fixture
def dana(conn):
    """One named speaker, two voiceprints, two groups claiming her."""
    sid = conn.execute("INSERT INTO speakers(name, created_at) VALUES(?,?)",
                       ("Dana Whitfield", time.time())).lastrowid
    v = at_cosine(0.99, 1).astype("float32")
    for cond, secs in (("auto-1", 120.0), ("auto-2", 90.0)):
        conn.execute(
            "INSERT INTO exemplars(speaker_id, condition, emb, dim, embed_model,"
            " seconds, created_at) VALUES(?,?,?,?,?,?,?)",
            (sid, cond, v.tobytes(), len(v), S.EMBED_MODEL, secs, time.time()))
    for _ in range(2):
        conn.execute("INSERT INTO groups(speaker_id, embed_model, linked_at)"
                     " VALUES(?,?,?)", (sid, S.EMBED_MODEL, time.time()))
    conn.commit()
    return sid


def test_the_store_enforces_its_own_foreign_keys(conn):
    """PRAGMA foreign_keys is per CONNECTION, so db() has to issue it."""
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(Exception) as e:
        conn.execute(
            "INSERT INTO exemplars(speaker_id, condition, emb, dim, embed_model,"
            " seconds, created_at) VALUES(999,'x',?,4,?,1.0,0)",
            (b"\0\0\0\0", S.EMBED_MODEL))
    assert "FOREIGN KEY" in str(e.value)


def test_forget_deletes_the_voiceprints_the_matcher_reads(conn, dana, run_pipe):
    run_pipe("speakers.py", "forget", str(dana))
    assert conn.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0] == 0


def test_forget_returns_the_groups_to_the_review_queue(conn, dana, run_pipe):
    """The group is a voice the linker found; only the human's answer is withdrawn."""
    p = run_pipe("speakers.py", "forget", str(dana))
    rows = conn.execute("SELECT speaker_id FROM groups").fetchall()
    assert len(rows) == 2 and all(r[0] is None for r in rows)
    assert "2 groups back in the review queue" in p.stdout


def test_forget_leaves_nothing_pointing_at_the_dead_id(conn, dana, run_pipe):
    """The transplant: `id INTEGER PRIMARY KEY` reuses a freed rowid, so the
    next person named takes the number -- and used to inherit the orphans with
    it."""
    run_pipe("speakers.py", "forget", str(dana))
    reborn = conn.execute("INSERT INTO speakers(name, created_at) VALUES(?,?)",
                          ("Someone Else", time.time())).lastrowid
    conn.commit()
    assert reborn == dana, "rowid was not reused; this test no longer tests it"
    assert conn.execute("SELECT COUNT(*) FROM exemplars WHERE speaker_id=?",
                        (reborn,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM groups WHERE speaker_id=?",
                        (reborn,)).fetchone()[0] == 0


def test_forgetting_someone_who_is_not_there_says_so(conn, run_pipe):
    p = run_pipe("speakers.py", "forget", "42", expect_rc=1)
    assert "no speaker with id 42" in p.stderr
