"""assemble() in pipeline/transcribe_meeting.py -- the correctness guards.

Every test here runs THE REAL FUNCTION out of the real file (see the loader in
conftest.py: the module imports vllm at module level, so its dependencies are
stubbed for the length of the import). Nothing in this file re-implements the
logic under test; a copy would drift and then assert nothing.

Each test corresponds to a defect the pipeline actually shipped:

  * bc42e1c / aff6307  the repetition guard dropped the line BEFORE the loop
  * 72b7319            speech both windows thought the other one owned
  * 03e61f3            coverage checked on the path that does not run, and a
                       diagnostic that raised NameError when it fired
"""
import ast
import builtins
import re

import pytest


def texts(segments):
    return [s["text"] for s in segments]


def starts(segments):
    return [s["start"] for s in segments]


def test_the_module_under_test_is_the_shipped_file(tm, repo):
    """Guard the guard: if this ever loads a copy, every test below is theatre."""
    assert tm.__file__ == str(repo / "pipeline" / "transcribe_meeting.py")
    assert callable(tm.assemble)


# ---------------------------------------------------------------------------
# 1. The repetition guard (bc42e1c, aff6307).
#
# MOSS loops on degenerate input and has none of Whisper's decoder guards, so
# identical lines in a row are dropped. Two bugs: the run was seeded with
# idx - 1 -- the preceding and DIFFERENT segment -- so the guard deleted a real
# line ahead of every loop and kept a loop member in its place; and that
# negative index sat in the drop set matching nothing, so the printed count was
# one higher than the number actually removed.
#
# All of these use ONE window, because the seam trimmer only fires across a
# window boundary and would otherwise be a second variable.
# ---------------------------------------------------------------------------
def one_window(win, lines, t0=0.0):
    """One window whose segments are `lines`, one second each, all in core."""
    segs = [(t0 + i, t0 + i + 0.9, txt) for i, txt in enumerate(lines)]
    return [win(0.0, (0.0, 100.0), segs)]


def test_repetition_guard_keeps_the_line_before_the_loop(assemble, win, capsys):
    segments, _, _, _ = assemble(
        one_window(win, ["hello", "yeah", "yeah", "yeah", "yeah"]))

    # "hello" is real speech and must survive; one "yeah" survives because
    # somebody usually did say it once before the decoder latched on.
    assert texts(segments) == ["hello", "yeah"]
    # And it is the FIRST "yeah", not a later loop member.
    assert segments[1]["start"] == pytest.approx(1.0)
    assert "repetition guard: dropped 3 looped segments" in capsys.readouterr().out


def test_repetition_guard_catches_a_run_starting_at_index_zero(assemble, win, capsys):
    """The window opened straight into the loop -- no earlier line to lean on."""
    segments, _, _, _ = assemble(one_window(win, ["x", "x", "x", "x"]))

    assert texts(segments) == ["x"]
    assert segments[0]["start"] == pytest.approx(0.0)
    assert "repetition guard: dropped 3 looped segments" in capsys.readouterr().out


def test_repetition_guard_reports_the_number_it_actually_dropped(assemble, win, capsys):
    """The old seed put -1 in the drop set: it matched no segment, so the log
    line claimed one more removal than it had made."""
    lines = ["x", "x", "x", "x", "x"]
    segments, _, _, _ = assemble(one_window(win, lines))
    out = capsys.readouterr().out

    m = re.search(r"repetition guard: dropped (\d+) looped segments", out)
    assert m, "the guard fired but printed nothing: %r" % out
    assert int(m.group(1)) == len(lines) - len(segments) == 4


def test_a_run_of_two_survives_untouched(assemble, win, capsys):
    """People repeat themselves. Two in a row is not a decoder loop."""
    segments, _, _, _ = assemble(one_window(win, ["a", "b", "b", "c"]))

    assert texts(segments) == ["a", "b", "b", "c"]
    assert "repetition guard" not in capsys.readouterr().out


def test_a_run_of_three_survives_and_four_does_not(assemble, win):
    """The threshold, pinned: the guard wants 3+ REPEATS, i.e. a 4th line."""
    three, _, _, _ = assemble(one_window(win, ["ok", "ok", "ok"]))
    four, _, _, _ = assemble(one_window(win, ["ok", "ok", "ok", "ok"]))

    assert texts(three) == ["ok", "ok", "ok"]
    assert texts(four) == ["ok"]


def test_alternating_lines_are_not_a_run(assemble, win, capsys):
    segments, _, _, _ = assemble(
        one_window(win, ["a", "b", "a", "b", "a", "b"]))

    assert texts(segments) == ["a", "b", "a", "b", "a", "b"]
    assert "repetition guard" not in capsys.readouterr().out


def test_a_loop_is_matched_case_and_whitespace_insensitively(assemble, win):
    """The decoder does not re-punctuate a line it is stuck on, but it does
    re-case it. Comparison is on text.strip().lower()."""
    segments, _, _, _ = assemble(
        one_window(win, ["Right.", "right.", " RIGHT. ", "right.", "moving on"]))

    assert texts(segments) == ["Right.", "moving on"]


# ---------------------------------------------------------------------------
# 2. Seam recovery (72b7319).
#
# A segment decoded in a window's context padding is dropped because the
# NEIGHBOURING window owns it as core. That holds only if both windows cut the
# seam the same way, and they do not -- they are independent generations over
# different context. So window k can push a segment's midpoint just past its
# core while k+1 pulls its version just before ITS core, and the speech is
# simply gone. Orphans are now reinstated, but ONLY where nothing else covers
# their span.
#
# Layout throughout: 10s cores with 5s of context each side, so window 1's
# audio starts at meeting time 5.0 and its core is local (5, 15) = global
# (10, 20).
# ---------------------------------------------------------------------------
def test_an_orphan_in_a_genuine_hole_is_recovered(assemble, win, capsys):
    segments, _, _, _ = assemble([
        win(0.0, (0.0, 10.0), [(2.0, 3.0, "opening remark"),
                               (10.2, 11.0, "so who are you exactly")]),
        win(5.0, (5.0, 15.0), [(12.0, 13.0, "later point")]),
    ])
    out = capsys.readouterr().out

    assert texts(segments) == ["opening remark", "so who are you exactly",
                               "later point"]
    assert segments[1]["start"] == pytest.approx(10.2)
    assert segments[1]["end"] == pytest.approx(11.0)
    assert starts(segments) == sorted(starts(segments))
    assert "recovered 1 segment(s)" in out


def test_an_orphan_the_neighbour_did_emit_is_not_recovered(assemble, win, capsys):
    """The neighbour owns it as core, so keeping the orphan would duplicate
    the speech -- which is the reason the drop existed in the first place."""
    segments, _, _, _ = assemble([
        win(0.0, (0.0, 10.0), [(2.0, 3.0, "opening remark"),
                               (10.2, 11.0, "so who are you exactly")]),
        win(5.0, (5.0, 15.0), [(5.2, 6.0, "so who are you exactly"),
                               (12.0, 13.0, "later point")]),
    ])
    out = capsys.readouterr().out

    assert texts(segments) == ["opening remark", "so who are you exactly",
                               "later point"]
    # Exactly one copy, and it is the neighbour's core version.
    assert [s for s in segments if s["text"] == "so who are you exactly"][0]["window"] == 1
    assert "recovered" not in out


def test_an_orphan_only_partly_covered_is_not_recovered(assemble, win, capsys):
    """Overlap at all means the neighbour did decode this speech, however it
    cut it. Reinstating would restate part of what is already there."""
    segments, _, _, _ = assemble([
        win(0.0, (0.0, 10.0), [(2.0, 3.0, "opening remark"),
                               (10.2, 11.0, "so who are you exactly")]),
        win(5.0, (5.0, 15.0), [(5.5, 7.0, "who are you and what do you do")]),
    ])
    out = capsys.readouterr().out

    assert texts(segments) == ["opening remark", "who are you and what do you do"]
    assert "recovered" not in out


def test_two_orphans_competing_for_one_hole_yield_one(assemble, win, capsys):
    """This is the exact failure: BOTH windows disowned their own version of
    the same speech, w0 into its right padding and w1 into its left. Filling
    the hole with both would duplicate it, so the earlier one wins alone."""
    segments, _, _, _ = assemble([
        # midpoint 10.6, past w0's core -> orphan
        win(0.0, (0.0, 10.0), [(2.0, 3.0, "opening remark"),
                               (10.2, 11.0, "w0 cut of the seam line")]),
        # midpoint local 4.95, before w1's core -> orphan at global 9.5-10.4
        win(5.0, (5.0, 15.0), [(4.5, 5.4, "w1 cut of the seam line"),
                               (12.0, 13.0, "later point")]),
    ])
    out = capsys.readouterr().out

    assert texts(segments) == ["opening remark", "w1 cut of the seam line",
                               "later point"]
    assert segments[1]["start"] == pytest.approx(9.5)
    assert "recovered 1 segment(s)" in out


def test_orphans_before_and_after_everything_are_recovered(assemble, win):
    """Nothing covers a hole at either end, so both come back -- and the
    reinstated segments must land in the right place in the list."""
    segments, _, _, _ = assemble([
        win(20.0, (5.0, 15.0), [(0.5, 1.5, "before the first core"),
                                (6.0, 7.0, "alpha")]),
        win(35.0, (5.0, 15.0), [(6.0, 8.0, "beta"),
                                (15.5, 16.5, "after the last core")]),
    ])

    assert texts(segments) == ["before the first core", "alpha", "beta",
                               "after the last core"]
    assert starts(segments) == sorted(starts(segments))
    assert segments[0]["start"] == pytest.approx(20.5)
    assert segments[-1]["start"] == pytest.approx(50.5)


def test_an_orphan_inside_a_long_segment_is_not_recovered(assemble, win, capsys):
    """The hole test is coverage of the SPAN, not a gap between neighbours: a
    long kept segment swallowing the orphan's span still counts as covered."""
    segments, _, _, _ = assemble([
        win(0.0, (0.0, 10.0), [(2.0, 3.0, "opening remark"),
                               (12.0, 12.5, "mm hmm")]),
        win(5.0, (5.0, 15.0), [(5.5, 14.0, "a long uninterrupted answer")]),
    ])
    out = capsys.readouterr().out

    assert texts(segments) == ["opening remark", "a long uninterrupted answer"]
    assert "recovered" not in out


def test_recovery_keeps_the_output_sorted_by_start(assemble, win):
    """Recovered orphans are merged, not appended."""
    segments, _, _, _ = assemble([
        win(0.0, (0.0, 10.0), [(10.2, 11.0, "orphan in the hole"),
                               (2.0, 3.0, "opening remark")]),
        win(5.0, (5.0, 15.0), [(12.0, 13.0, "later point"),
                               (16.0, 17.0, "orphan past the core")]),
    ])

    assert starts(segments) == sorted(starts(segments))
    assert texts(segments) == ["opening remark", "orphan in the hole",
                               "later point", "orphan past the core"]


# ---------------------------------------------------------------------------
# 3. Coverage (03e61f3).
#
# assemble() returns FOUR values now. The fourth exists because the single-file
# guard's own diagnostic named speech_end, which was local to assemble(): the
# moment the check detected the failure it exists to detect, it raised
# NameError while building its message. The batch path -- the one every run
# takes -- did not check at all.
# ---------------------------------------------------------------------------
def test_assemble_returns_four_values_in_a_known_order(assemble, win):
    result = assemble([
        win(0.0, (0.0, 10.0), [(1.0, 2.0, "one")]),
        win(10.0, (0.0, 10.0), [(1.0, 2.0, "two")], finish_reason="length"),
    ], dur=30.0)

    assert len(result) == 4
    segments, cov, capped, speech_end = result
    assert [s["text"] for s in segments] == ["one", "two"]
    assert 0.0 <= cov <= 1.0
    assert capped == [1]                    # window 1 hit the token cap
    assert isinstance(speech_end, float)


def test_coverage_min_is_095(tm):
    assert tm.COVERAGE_MIN == 0.95


def test_both_callers_unpack_four_values(assemble_call_arities):
    """The grep-level check that would have caught the NameError's sibling:
    assemble() grew a fourth return value and a caller kept unpacking three."""
    for path, arities in assemble_call_arities.items():
        assert arities, "no `= assemble(...)` call found in %s" % path
        assert all(n == 4 for n in arities), \
            "%s unpacks %r values from assemble()" % (path, arities)


def test_main_uses_only_names_it_has_bound(repo):
    """The NameError itself, statically.

    main()'s coverage diagnostic referred to speech_end, which was a local of
    assemble(). Any name main() loads must be bound in main, a module global,
    or a builtin -- otherwise the code path that uses it dies when it runs,
    which is exactly what happened to the guard.
    """
    src = (repo / "pipeline" / "transcribe_meeting.py").read_text()
    tree = ast.parse(src)

    module_names = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                module_names.add(a.asname or a.name.split(".")[0])
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                module_names |= {n.id for n in ast.walk(t)
                                 if isinstance(n, ast.Name)}

    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    bound = {n.id for n in ast.walk(main)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {a.arg for a in main.args.args}
    loaded = {n.id for n in ast.walk(main)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    undefined = loaded - bound - module_names - set(dir(builtins))
    assert not undefined, "main() reads names nothing binds: %s" % sorted(undefined)
    assert "speech_end" in bound and "speech_end" in loaded, \
        "the coverage diagnostic must be able to name where speech ends"


def test_the_coverage_check_is_not_an_assert(repo):
    """An assert is compiled out under -O, and this is an operational check on
    the model's output rather than a claim about the code."""
    src = (repo / "pipeline" / "transcribe_meeting.py").read_text()
    main = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "main")

    for node in ast.walk(main):
        if isinstance(node, ast.Assert):
            dumped = ast.dump(node)
            assert "cov" not in dumped and "COVERAGE_MIN" not in dumped, \
                "coverage is being enforced with an assert again"


def test_coverage_is_measured_against_where_speech_ends(assemble, win, tm,
                                                        wav_factory):
    """Not against file duration. A recording whose last real turn is well
    before the end of the file is not a truncated transcript, and judging it
    against file length made the guard fire on correct output."""
    wav = wav_factory(60.0, speech_until=30.0)
    segments, cov, _, speech_end = assemble(
        [win(0.0, (0.0, 60.0), [(0.5, 10.0, "first half"),
                                (20.0, 30.0, "up to the end of the speech")])],
        dur=60.0, wav=wav)

    assert speech_end == pytest.approx(29.99, abs=0.05)
    assert cov == pytest.approx(1.0)
    assert cov >= tm.COVERAGE_MIN
    assert len(segments) == 2


def test_a_transcript_that_stops_early_falls_below_coverage_min(assemble, win, tm,
                                                                wav_factory):
    """The failure the guard exists for: MOSS silently dropped 38% of a
    meeting and reported success."""
    wav = wav_factory(60.0, speech_until=30.0)
    _, cov, _, speech_end = assemble(
        [win(0.0, (0.0, 60.0), [(0.5, 15.0, "and then it stopped")])],
        dur=60.0, wav=wav)

    assert speech_end == pytest.approx(29.99, abs=0.05)
    assert cov == pytest.approx(15.0 / 29.99, rel=1e-3)
    assert cov < tm.COVERAGE_MIN


def test_coverage_uses_the_furthest_end_not_the_last_segment_by_start(assemble, win,
                                                                      wav_factory):
    """Segments are sorted by START, and overlapping speech nests: a short
    late-starting backchannel can be last in the list while ending long before
    the real final segment."""
    wav = wav_factory(60.0, speech_until=40.0)
    segments, cov, _, _ = assemble(
        [win(0.0, (0.0, 60.0), [(5.0, 40.0, "a long answer"),
                                (10.0, 11.0, "mm hmm")])],
        dur=60.0, wav=wav)

    assert segments[-1]["end"] == pytest.approx(11.0)    # last BY START
    assert cov == pytest.approx(1.0)                     # but coverage is full
