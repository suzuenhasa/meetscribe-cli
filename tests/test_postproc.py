"""The data/state layer: the profile store and the two scripts that rewrite it.

Three groups, each anchored to a defect this repo actually had:

  RELABEL DRY RUN (b405cab). `relabel.py` identified into the REAL names.json to
  work out what would change, so the dry run WAS the change: `apply` followed by
  `apply --apply` then found nothing to do and re-rendered nothing, while
  reporting that every transcript already agreed with the store. It did not.

  ID MIGRATION (019f335). speakers.db keyed on the sanitised filename, which made
  the filename the identity. migrate_ids.py re-keys onto meeting ids, and the
  whole value of it is in what it refuses to do: rows it cannot match are LEFT
  ALONE, the old value is kept, and the store is copied first.

  THE PROFILE STORE. 0.55 to accept, a 0.10 margin over the runner-up, one person
  to one cluster -- and the production failure those three produce together, when
  one human is enrolled twice under two spellings.

Scores here are exact by construction (see at_cosine in conftest), so a test
about the 0.55 threshold is about the threshold and not about where a random
fixture landed. Nothing in here can reach the real speakers.db or library.
"""
import json
import time
from pathlib import Path

import pytest

from conftest import REF, at_cosine, basis, make_meeting, rows_of, snapshot, write_clusters_npz

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
    """One profile in the store, scoring 0.98 against G00 of both meetings.

    This is the state relabel.py exists for: a voice named AFTER the meetings it
    appears in were processed, so the store knows and the transcripts do not.
    """
    S.enroll_centroid(conn, "Alice Anderson", at_cosine(0.98, 1), 90.0,
                      "aaa11111", "G00")
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
# 2. ID MIGRATION -- migrate_ids.py, commit 019f335
# =====================================================================

#: What the store used to say, before ids. The key was the transcript filename
#: run through ./transcribe's sanitiser, and prototypes appended ":<cluster>".
LEGACY_PROTO = "Design-Review-Q3:G02"
LEGACY_DECISION = "Design-Review-Q3"
ORPHAN = "Some-Recording-I-Deleted"


@pytest.fixture
def legacy_store(conn, lib, store):
    """A filename-keyed store beside a library of two meetings.

    Deliberately mixed: rows that match a meeting, rows that match nothing
    because the recording is gone, and one row with no meeting at all.
    """
    make_meeting(lib, "Design Review Q3", "Design Review Q3.mp3", "aaa11111")
    make_meeting(lib, "Roadmap Sync", "Roadmap Sync.mp3", "bbb22222")

    S.enroll_centroid(conn, "Alice Anderson", REF, 90.0, "Design-Review-Q3", "G02")
    S.enroll_centroid(conn, "Bob Brown", REF, 90.0, ORPHAN, "G01")
    for meeting in (LEGACY_DECISION, "Roadmap-Sync", ORPHAN, ""):
        conn.execute(
            "INSERT INTO decisions(meeting, cluster, speaker_id, score, second,"
            " threshold, level, roster, outcome, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (meeting, "G00", 1, 0.9, 0.1, S.ACCEPT, "centroid", None,
             "accept", time.time()))
    conn.commit()
    return store


def migrate(run_pipe, store, lib, *extra):
    return run_pipe("migrate_ids.py", "--db", store, "--library", lib, *extra)


def test_migration_dry_run_changes_nothing(run_pipe, legacy_store, lib):
    before = {t: rows_of(legacy_store, t) for t in ("prototypes", "decisions")}

    p = migrate(run_pipe, legacy_store, lib)

    assert "rows to re-key: 3" in p.stdout
    assert "nothing changed" in p.stdout
    assert {t: rows_of(legacy_store, t) for t in ("prototypes", "decisions")} == before
    assert not Path(str(legacy_store) + ".pre-ids").exists()
    # Not even the schema: a dry run that added legacy_name would already have
    # written to the one file in the project that cannot be rebuilt.
    assert "legacy_name" not in before["prototypes"][0]


def test_apply_rekeys_the_rows_that_match_a_meeting(run_pipe, legacy_store, lib):
    migrate(run_pipe, legacy_store, lib, "--apply")

    protos = rows_of(legacy_store, "prototypes")
    decisions = rows_of(legacy_store, "decisions")
    assert protos[0]["meeting"] == "aaa11111:G02", "cluster suffix must survive"
    assert decisions[0]["meeting"] == "aaa11111"
    assert decisions[1]["meeting"] == "bbb22222"


def test_rows_with_no_meeting_left_are_left_alone_and_reported(run_pipe,
                                                               legacy_store, lib):
    """Dropping them would make the migration look tidy and lose a real fact."""
    p = migrate(run_pipe, legacy_store, lib, "--apply")

    assert "left alone, no meeting in the library: 2" in p.stdout
    assert ORPHAN in p.stdout
    protos = rows_of(legacy_store, "prototypes")
    decisions = rows_of(legacy_store, "decisions")
    assert protos[1]["meeting"] == ORPHAN + ":G01"
    assert protos[1]["legacy_name"] is None, "untouched rows get no legacy_name"
    assert decisions[2]["meeting"] == ORPHAN


def test_a_row_with_no_meeting_at_all_is_ignored_silently(run_pipe, legacy_store,
                                                          lib):
    p = migrate(run_pipe, legacy_store, lib, "--apply")

    # Four decisions rows, one of them blank. The blank one is neither re-keyed
    # nor counted as an orphan -- it has nothing to say about any recording.
    assert "left alone, no meeting in the library: 2" in p.stdout
    assert rows_of(legacy_store, "decisions")[3]["meeting"] == ""


def test_legacy_name_keeps_what_each_row_used_to_say(run_pipe, legacy_store, lib):
    migrate(run_pipe, legacy_store, lib, "--apply")

    assert rows_of(legacy_store, "prototypes")[0]["legacy_name"] == LEGACY_PROTO
    assert rows_of(legacy_store, "decisions")[0]["legacy_name"] == LEGACY_DECISION


def test_the_store_is_backed_up_before_it_is_rewritten(run_pipe, legacy_store, lib):
    backup = Path(str(legacy_store) + ".pre-ids")

    p = migrate(run_pipe, legacy_store, lib, "--apply")

    assert backup.exists(), p.stdout
    assert [r["meeting"] for r in rows_of(backup, "prototypes")] == [
        LEGACY_PROTO, ORPHAN + ":G01"]
    assert [r["meeting"] for r in rows_of(backup, "decisions")] == [
        LEGACY_DECISION, "Roadmap-Sync", ORPHAN, ""]


def test_running_the_migration_twice_is_safe(run_pipe, legacy_store, lib):
    migrate(run_pipe, legacy_store, lib, "--apply")
    after_first = {t: rows_of(legacy_store, t) for t in ("prototypes", "decisions")}
    backup_first = Path(str(legacy_store) + ".pre-ids").read_bytes()

    p = migrate(run_pipe, legacy_store, lib, "--apply")

    assert "rows to re-key: 0" in p.stdout
    assert "nothing to do" in p.stdout
    assert {t: rows_of(legacy_store, t) for t in ("prototypes", "decisions")} == after_first
    # In particular legacy_name is not overwritten with the id it now holds, and
    # the backup is still the store as it was BEFORE the first run.
    assert after_first["prototypes"][0]["legacy_name"] == LEGACY_PROTO
    assert Path(str(legacy_store) + ".pre-ids").read_bytes() == backup_first


# FIXED (was xfail):
# A second run reports every already-migrated row under 'left alone, no meeting in the
# library', which reads as data about to be lost -- for rows pointing at meetings that
# ARE in the library. The rows are in fact fine; only the report is wrong. To pass,
# main() in pipeline/migrate_ids.py would have to recognise a row already keyed on a
# live meeting (build {m.id for m in LIB.all_meetings(...)} alongside `known`, and
# when `name` is in that set treat the row as done rather than appending it to
# `unmatched`).
def test_a_second_run_does_not_call_migrated_rows_orphans(run_pipe, legacy_store,
                                                          lib):
    migrate(run_pipe, legacy_store, lib, "--apply")

    p = migrate(run_pipe, legacy_store, lib)

    assert "rows to re-key: 0" in p.stdout
    # The one genuine orphan, and not the three rows that were just migrated.
    assert "left alone, no meeting in the library: 2" in p.stdout
    assert "aaa11111" not in p.stdout.split("left alone")[1]


# =====================================================================
# 3. THE PROFILE STORE -- enrol, match, thresholds, one-to-one
# =====================================================================

def clusters_file(tmp_path, spec, name="run_clusters.npz"):
    p = tmp_path / name
    write_clusters_npz(p, spec, meeting="m1")
    return p


def identify(run_pipe, tmp_path, spec, roster="", enroll=(), meeting="m1"):
    """Run identify.py over one synthetic meeting. -> (process, names map)."""
    names = tmp_path / ("%s-names.json" % meeting)
    args = ["identify.py", "--clusters", clusters_file(tmp_path, spec),
            "--meeting", meeting, "--roster", roster, "--names", names]
    if enroll:
        args += ["--enroll"] + list(enroll)
    p = run_pipe(*args)
    return p, json.loads(names.read_text())


def enrol(conn, *people):
    """enrol(conn, ("Alice", at_cosine(0.9, 1)), ...)"""
    for name, vec in people:
        S.enroll_centroid(conn, name, vec, 90.0, "m0", "G00")
    return conn


# -- the store itself ---------------------------------------------------

def test_enrolling_puts_one_unit_centroid_in_the_gallery(conn):
    S.enroll_centroid(conn, "Alice Anderson", at_cosine(0.80, 1), 42.0, "m0", "G03")

    G = S.gallery(conn)
    assert len(G) == 1
    _sid, name, vec = G[0]
    assert name == "Alice Anderson"
    assert abs(float(vec @ REF) - 0.80) < 1e-6
    assert abs(float(vec @ vec) - 1.0) < 1e-6

    proto = conn.execute("SELECT level, seconds, meeting, dim FROM prototypes").fetchone()
    assert proto[0] == "centroid", "the level is what makes 0.55 meaningful"
    assert proto[1] == 42.0
    assert proto[2] == "m0:G03"
    assert proto[3] == len(REF)


def test_enrolling_one_person_twice_leaves_one_candidate(conn):
    """Two sessions of one voice average into one profile, not two rivals.

    This is the case the margin rule survives, and the contrast that makes the
    two-spellings failure below a bug rather than the rule working as designed.
    """
    enrol(conn, ("Alice Anderson", at_cosine(0.999, 1)),
                ("Alice Anderson", at_cosine(0.998, 2)))

    assert conn.execute("SELECT COUNT(*) FROM prototypes").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0] == 1
    assert len(S.gallery(conn)) == 1


def test_a_roster_restricts_the_candidate_pool(conn):
    enrol(conn, ("Alice Anderson", at_cosine(0.9, 1)),
                ("Bob Brown", at_cosine(0.8, 2)))

    assert sorted(n for _, n, _ in S.gallery(conn)) == ["Alice Anderson", "Bob Brown"]
    assert [n for _, n, _ in S.gallery(conn, ["Alice Anderson"])] == ["Alice Anderson"]


def test_the_linkers_leftover_bucket_is_never_a_person(tmp_path):
    """G- clusters are segments too short to cluster, not a speaker."""
    p = clusters_file(tmp_path, {"G00": (REF, 300.0),
                                 "G-1": (at_cosine(0.9, 1), 12.0)})
    assert sorted(S.centroids_from_npz(p)) == ["G00"]


def test_a_zero_length_centroid_is_dropped_rather_than_dividing_by_zero(tmp_path):
    p = clusters_file(tmp_path, {"G00": (REF, 300.0),
                                 "G01": (basis(0) * 0.0, 8.0)})
    assert sorted(S.centroids_from_npz(p)) == ["G00"]


# -- the decision rules -------------------------------------------------

def test_a_score_above_the_accept_threshold_names_the_cluster(conn, tmp_path,
                                                              run_pipe, decisions):
    enrol(conn, ("Alice Anderson", at_cosine(0.60, 1)))

    _p, names = identify(run_pipe, tmp_path, {"G00": (REF, 120.0)})

    assert names == {"G00": "Alice Anderson"}
    assert decisions()[("m1", "G00")]["outcome"] == "accept"


def test_a_score_below_the_accept_threshold_is_review_not_a_name(conn, tmp_path,
                                                                 run_pipe, decisions):
    """0.54 against a 0.55 threshold, with nothing else in the gallery."""
    enrol(conn, ("Alice Anderson", at_cosine(0.54, 1)))

    p, names = identify(run_pipe, tmp_path, {"G00": (REF, 120.0)})

    assert names == {}
    d = decisions()[("m1", "G00")]
    assert d["outcome"] == "review"
    assert abs(d["score"] - 0.54) < 1e-6
    assert "? G00" in p.stdout, "a review has to be visible to the person reading"


def test_a_score_below_the_review_floor_is_unknown(conn, tmp_path, run_pipe,
                                                   decisions):
    enrol(conn, ("Alice Anderson", at_cosine(0.20, 1)))

    _p, names = identify(run_pipe, tmp_path, {"G00": (REF, 120.0)})

    assert names == {}
    assert decisions()[("m1", "G00")]["outcome"] == "unknown"


def test_the_margin_blocks_a_name_when_the_runner_up_is_close(conn, tmp_path,
                                                              run_pipe, decisions):
    """0.70 is well above 0.55, and still not enough at 0.05 over second."""
    enrol(conn, ("Alice Anderson", at_cosine(0.70, 1)),
                ("Bob Brown", at_cosine(0.65, 2)))

    _p, names = identify(run_pipe, tmp_path, {"G00": (REF, 120.0)})

    assert names == {}
    d = decisions()[("m1", "G00")]
    assert d["outcome"] == "review"
    assert d["score"] > S.ACCEPT and (d["score"] - d["second"]) < S.MARGIN


def test_the_margin_is_satisfied_when_the_runner_up_is_far_enough_back(
        conn, tmp_path, run_pipe, decisions):
    """Same 0.70 best as above; only the runner-up moved, and now it names."""
    enrol(conn, ("Alice Anderson", at_cosine(0.70, 1)),
                ("Bob Brown", at_cosine(0.55, 2)))

    _p, names = identify(run_pipe, tmp_path, {"G00": (REF, 120.0)})

    assert names == {"G00": "Alice Anderson"}
    d = decisions()[("m1", "G00")]
    assert (d["score"] - d["second"]) >= S.MARGIN


def test_one_person_cannot_be_two_clusters_in_one_meeting(conn, tmp_path,
                                                          run_pipe, decisions):
    """The cluster that spoke most takes the name; the other needs a human."""
    enrol(conn, ("Alice Anderson", at_cosine(0.95, 1)))

    _p, names = identify(run_pipe, tmp_path,
                         {"G00": (REF, 600.0), "G01": (REF, 60.0)})

    assert names == {"G00": "Alice Anderson"}
    got = decisions()
    assert got[("m1", "G00")]["outcome"] == "accept"
    assert got[("m1", "G01")]["outcome"] == "review"
    assert got[("m1", "G01")]["score"] == got[("m1", "G00")]["score"]


def test_enrolling_below_the_measured_knee_warns_and_still_enrols(conn, tmp_path,
                                                                  run_pipe):
    p, names = identify(run_pipe, tmp_path, {"G00": (REF, 5.0)},
                        enroll=['G00=Alice Anderson'])

    assert "5s of speech" in p.stdout and "Enrolling anyway" in p.stdout
    # Enrolled in time to be matched by the same run.
    assert names == {"G00": "Alice Anderson"}
    assert conn.execute("SELECT name FROM speakers").fetchall() == [("Alice Anderson",)]


# -- the production failure ---------------------------------------------
#
# One human, enrolled twice under two spellings, is two speakers rows and so two
# gallery entries. Both score ~1.000 against the same voice, the runner-up is
# always within 0.10 of the best, and the margin rule can therefore never be
# satisfied for that person again -- in any meeting, however much they speak,
# however many times relabel.py is re-run. gallery() folds multiple prototypes of
# ONE speaker id into one candidate, so this only happens across names.

DUPES = [("Sreeram Kannan", at_cosine(0.999, 1)),
         ("Sreeram (laptop mic)", at_cosine(0.998, 2))]


def test_two_profiles_for_one_person_leave_everyone_unnamed(conn, tmp_path,
                                                            run_pipe, decisions):
    enrol(conn, *DUPES)

    _p, names = identify(run_pipe, tmp_path,
                         {"G00": (REF, 600.0), "G01": (at_cosine(0.20, 3), 120.0)})

    assert names == {}, "the main speaker of the meeting comes out unnamed"
    d = decisions()[("m1", "G00")]
    assert d["outcome"] == "review"
    # Not a weak match -- a near-perfect one, blocked only by its own duplicate.
    assert d["score"] > 0.99 and d["second"] > 0.99
    assert (d["score"] - d["second"]) < S.MARGIN


def test_a_roster_is_the_way_out_of_the_duplicate_profile_trap(conn, tmp_path,
                                                               run_pipe):
    """Naming the attendee restricts the gallery to one, so nothing rivals it."""
    enrol(conn, *DUPES)

    _p, names = identify(run_pipe, tmp_path, {"G00": (REF, 600.0)},
                         roster="Sreeram Kannan")

    assert names == {"G00": "Sreeram Kannan"}


# FIXED (was xfail):
# When the margin is what blocked a 0.999 match, identify.py prints the runner-up's
# SCORE but never its NAME, and then explains the 0.40-0.55 review band -- which is
# not the band this cluster was in. The person reading has no way to see that two
# profiles are one human and that forgetting one fixes it, so the meeting stays
# unnamed forever. To pass, main() in pipeline/identify.py would have to carry the
# runner-up's name into `report` (it keeps `second[0]` only, not `second[2]`) and
# print it on a margin-blocked line.",
def test_a_margin_blocked_cluster_says_which_profile_blocked_it(conn, tmp_path,
                                                                run_pipe):
    enrol(conn, *DUPES)

    p, _names = identify(run_pipe, tmp_path, {"G00": (REF, 600.0)})

    assert "laptop mic" in p.stdout
