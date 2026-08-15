"""The bash entry points: ./engine and ./transcribe, run as subprocesses.

Every test here corresponds to a defect this repo actually shipped. The named
commit in each section is the fix; the test is there so it stays fixed.

The interpreter is stubbed (see conftest), so nothing loads a model, touches a
GPU, or ssh's anywhere -- but the scripts themselves are the real, shipped
files, running their real control flow. That matters: three of the four defects
below are `set -e` interactions, which only exist when bash is actually bash.
"""

import re

import pytest

from conftest import flag_value


# =====================================================================
# 1. `engine status` exit codes                         (commit 7cadd9b)
#
# engined.py --status answers with 0 (up), 2 (up, but running code older than
# pipeline/ on disk) or 1 (nothing there). ./engine has to translate all three.
#
# The bug: `OUT="$(...)"; RC=$?` takes the assignment's exit status FROM the
# substitution, so under `set -e` the assignment itself ended the script and the
# case that prints the answer never ran. `status` printed NOTHING and exited
# with the raw code -- including on rc 1, "no engine", which is the answer most
# of the time. The stale-code note on rc 2 was unreachable code.
#
# Hence: every case asserts the output is non-empty. A silent status is the
# regression, whatever the exit code says.
# =====================================================================

STALE_NOTE = "this engine is running code older than pipeline/ on disk"


@pytest.mark.parametrize("engined_rc,want_exit,want_text", [
    (0, 0, "running"),          # up
    (2, 0, "running"),          # up, holding stale code -- still UP
    (1, 1, "not running"),      # nothing there
])
def test_status_answers_whatever_engined_says(tree, engined_rc, want_exit, want_text):
    p = tree.run("engine", "status", stub_rc=engined_rc,
                 stub_out="engine up\n" + STALE_NOTE)

    # The regression itself: status that says nothing at all.
    assert p.stdout.strip(), (
        "engine status printed nothing for engined.py rc=%d (rc=%d, stderr=%r)"
        % (engined_rc, p.returncode, p.stderr))
    assert p.returncode == want_exit, (
        "rc=%d, expected %d\nstdout=%r stderr=%r"
        % (p.returncode, want_exit, p.stdout, p.stderr))

    first = p.stdout.strip().split("\n")[0]
    if want_text == "running":
        assert "not running" not in first, (
            "rc=%d means the engine is UP; calling it down invites a second "
            "engine onto an occupied card. Got: %r" % (engined_rc, first))
    assert want_text in first


def test_status_rc0_names_the_socket_and_says_nothing_else(tree):
    p = tree.run("engine", "status", stub_rc=0, stub_out="engine up")
    assert p.returncode == 0
    lines = [l for l in p.stdout.strip().split("\n") if l]
    assert len(lines) == 1, "a healthy engine gets one line, got %r" % (lines,)
    assert "running" in lines[0] and "not running" not in lines[0]
    assert str(tree.path / "run" / "engine.sock") in lines[0]


def test_status_rc2_prints_the_stale_code_note(tree):
    """rc 2 is 'up, but not the code you think you deployed' -- and it has to SAY so."""
    p = tree.run("engine", "status", stub_rc=2,
                 stub_out="engine up\n" + STALE_NOTE)

    assert p.returncode == 0, (
        "rc 2 from engined.py is a warning about an engine that IS running; "
        "status must not report it as a failure (rc=%d)" % p.returncode)
    lines = [l for l in p.stdout.strip().split("\n") if l]
    assert "running" in lines[0] and "not running" not in lines[0]
    assert STALE_NOTE in p.stdout, (
        "the stale-code note engined.py printed never reached the user: %r"
        % (p.stdout,))


def test_status_rc1_is_the_common_case_and_still_answers(tree):
    """No engine is not an error condition to hide -- it is the usual answer."""
    p = tree.run("engine", "status", stub_rc=1, stub_out="no engine on that socket")
    assert p.stdout.strip() == "not running", (
        "expected a plain 'not running', got stdout=%r stderr=%r"
        % (p.stdout, p.stderr))
    assert p.returncode == 1


# =====================================================================
# 2. The provisioning guard                             (commit 87596bb)
#
# vast marks an instance ready when the CONTAINER boots, minutes before
# provisioning finishes. ./transcribe and ./engine start refuse while
# $WORK/.provisioning exists, because running against a half-built tree fails in
# ways that read as bugs in the script rather than as "wait".
#
# The bug it caused: provisioning RUNS both of those itself -- transcribe for
# the warm-up that fills the compile cache, engine start when supervisor did not
# -- so the guard blocked the two steps that make the box ready, and presented
# as three unrelated faults none of which named the guard. MS_PROVISIONING=1 is
# the exemption, and it is only useful if provision.sh exports it BEFORE it runs
# either script (test at the end of this section).
#
# `engine status` is deliberately NOT guarded: it is how you check.
# =====================================================================

EX_TEMPFAIL = 75


def test_transcribe_refuses_while_the_box_is_still_provisioning(tree):
    tree.audio()
    tree.mark_provisioning("2026-08-15T09:00:00Z")

    p = tree.run("transcribe", "meeting.mp3")

    assert p.returncode == EX_TEMPFAIL, (
        "expected EX_TEMPFAIL (75) = try again shortly, got %d" % p.returncode)
    assert "still being set up" in p.stderr
    assert "2026-08-15T09:00:00Z" in p.stderr, (
        "the marker holds the start time; the message should quote it: %r" % p.stderr)
    assert not tree.invocations, (
        "the guard must refuse BEFORE reaching the python layer, got %r"
        % (tree.invocations,))


def test_engine_start_refuses_while_the_box_is_still_provisioning(tree):
    tree.mark_provisioning()

    p = tree.run("engine", "start")

    assert p.returncode == EX_TEMPFAIL, (
        "expected 75, got %d (stdout=%r stderr=%r)"
        % (p.returncode, p.stdout, p.stderr))
    assert "still being set up" in p.stderr
    assert not tree.invocations, (
        "a guarded start must not reach engined.py at all, got %r" % (tree.invocations,))


@pytest.mark.parametrize("script,args", [
    ("transcribe", ("meeting.mp3",)),
    ("engine", ("start",)),
])
def test_the_exemption_lets_provisionings_own_children_through(tree, script, args):
    """MS_PROVISIONING=1 is how provision.sh runs the steps that make the box ready."""
    tree.audio()
    tree.mark_provisioning()

    p = tree.run(script, *args, mklib=True, env={"MS_PROVISIONING": "1"})

    assert p.returncode != EX_TEMPFAIL, (
        "MS_PROVISIONING=1 must exempt this run; it was refused anyway. "
        "stderr=%r" % p.stderr)
    assert "still being set up" not in p.stderr


@pytest.mark.parametrize("script,args", [
    ("transcribe", ("meeting.mp3",)),
    ("engine", ("start",)),
])
def test_a_plain_install_has_no_marker_and_is_never_blocked(tree, script, args):
    """./setup.sh never writes the marker, so nothing here should ever be refused."""
    tree.audio()
    assert not (tree.path / ".provisioning").exists()

    p = tree.run(script, *args, mklib=True)

    assert p.returncode != EX_TEMPFAIL
    assert "still being set up" not in p.stderr


def test_engine_status_stays_answerable_while_provisioning(tree):
    """status is how you check whether the wait is over, so it is not guarded.

    NOT asserting rc 0: with no engine running -- the normal state on a box that
    is still provisioning -- the right answer is 'not running' and rc 1. The
    thing being asserted is that it ANSWERS rather than refusing with 75.
    """
    tree.mark_provisioning()

    p = tree.run("engine", "status", stub_rc=1)

    assert p.returncode != EX_TEMPFAIL, "status must not be refused by the guard"
    assert p.stdout.strip(), "status must answer while provisioning, said nothing"
    assert "still being set up" not in p.stderr
    assert "not running" in p.stdout


# $MS_WORK/transcribe or $MS_WORK/engine being executed.
_RUNS_A_SCRIPT = re.compile(r"\$\{?MS_WORK\}?/(transcribe|engine)\b")
_EXPORTS_EXEMPTION = re.compile(r"^\s*export\s+MS_PROVISIONING=1\s*(#.*)?$")


def test_provision_sh_exports_the_exemption_before_it_runs_either_script(repo):
    """The exemption is useless unless it is in the environment first.

    Static, because the failure is an ordering one: provision.sh writes the
    marker, then runs ./transcribe for the warm-up and ./engine start for the
    fallback. If MS_PROVISIONING is exported after either of those -- or merely
    assigned rather than exported, so it never reaches a child -- the guard
    blocks provisioning's own steps and the box never becomes ready.
    """
    lines = (repo / "vast" / "provision.sh").read_text().split("\n")

    exports = [n for n, l in enumerate(lines, 1) if _EXPORTS_EXEMPTION.match(l)]
    assert exports, (
        "vast/provision.sh must `export MS_PROVISIONING=1`; without it the "
        "guard refuses the warm-up and the fallback engine start")
    first_export = min(exports)

    runs = [(n, l.strip()) for n, l in enumerate(lines, 1)
            if _RUNS_A_SCRIPT.search(l) and not l.lstrip().startswith("#")]
    assert runs, (
        "expected provision.sh to invoke $MS_WORK/transcribe or $MS_WORK/engine; "
        "if the invocations moved, this test needs to follow them")

    too_early = [(n, l) for n, l in runs if n < first_export]
    assert not too_early, (
        "these run before `export MS_PROVISIONING=1` (line %d) and would be "
        "refused with 75: %r" % (first_export, too_early))

    unset = [n for n, l in enumerate(lines, 1)
             if re.match(r"\s*unset\s+([\w\s]*\s)?MS_PROVISIONING\b", l)]
    assert not unset, "MS_PROVISIONING is unset at line(s) %r" % (unset,)


# =====================================================================
# 3. Argument pass-through                   (commits c3772ee, 7b83a77)
#
# --replace existed in batch.py and ./transcribe had never heard of it, so it
# fell through the argument loop's default case -- "this must be the filename" --
# and printed usage at you. Every flag below is asserted to arrive at the python
# layer with its value, on BOTH paths: a single file and a directory build their
# command lines separately, and --replace was missing from both.
#
# --overlap defaults to 0 (commit 7b83a77): the four defaults across
# transcribe/batch.py/engined.py/transcribe_meeting.py must move together,
# because engined.py refuses a job whose window geometry differs from the engine
# it built -- so a stale default here makes every default run miss the resident
# engine and load a second one onto a card with no room for it.
# =====================================================================

def source_for(tree, kind):
    """A single recording, or a directory holding one -- the two code paths."""
    if kind == "file":
        tree.audio("meeting.mp3")
        return "meeting.mp3"
    d = tree.path / "recordings"
    d.mkdir()
    (d / "standup.mp3").write_bytes(b"\x00" * 64)
    return "recordings"


BOTH_PATHS = pytest.mark.parametrize("kind", ["file", "dir"])


@BOTH_PATHS
def test_every_flag_reaches_the_python_layer(tree, kind):
    src = source_for(tree, kind)

    p = tree.run("transcribe", src,
                 "--replace", "9ajq9",
                 "--window", "60",
                 "--overlap", "5",
                 "--thr", "0.70",
                 "--roster", "Bob Smith,Jane Doe",
                 "--glossary", "Dana Whitfield,Northwind",
                 mklib=True)

    assert p.returncode == 0, "stdout=%r stderr=%r" % (p.stdout, p.stderr)
    argv = tree.invocation_for("engined.py")
    for flag, value in [("--replace", "9ajq9"),
                        ("--window", "60"),
                        ("--overlap", "5"),
                        ("--thr", "0.70"),
                        ("--roster", "Bob Smith,Jane Doe"),
                        ("--glossary", "Dana Whitfield,Northwind")]:
        got = flag_value(argv, flag)
        assert got == value, (
            "%s reached the python layer as %r, expected %r (%s path)\nargv=%r"
            % (flag, got, value, kind, argv))


@BOTH_PATHS
def test_replace_is_not_invented_when_it_was_not_asked_for(tree, kind):
    """`--replace` redoes a meeting IN PLACE; sending it unasked would be destructive."""
    src = source_for(tree, kind)
    p = tree.run("transcribe", src, mklib=True)

    assert p.returncode == 0, "stdout=%r stderr=%r" % (p.stdout, p.stderr)
    argv = tree.invocation_for("engined.py")
    assert "--replace" not in argv, "unasked-for --replace in %r" % (argv,)


@BOTH_PATHS
def test_overlap_defaults_to_zero(tree, kind):
    """The default is 0. It was 5, and 5 decodes 40s of audio to keep 30."""
    src = source_for(tree, kind)
    p = tree.run("transcribe", src, mklib=True)

    assert p.returncode == 0, "stdout=%r stderr=%r" % (p.stdout, p.stderr)
    argv = tree.invocation_for("engined.py")
    assert flag_value(argv, "--overlap") == "0", (
        "--overlap defaulted to %r, not 0. engined.py refuses a job whose window "
        "geometry differs from the engine it built, so this default has to match "
        "the one in engined.py/batch.py.\nargv=%r"
        % (flag_value(argv, "--overlap"), argv))


@BOTH_PATHS
def test_window_and_thr_have_their_documented_defaults(tree, kind):
    src = source_for(tree, kind)
    p = tree.run("transcribe", src, mklib=True)

    argv = tree.invocation_for("engined.py")
    assert flag_value(argv, "--window") == "30"
    assert flag_value(argv, "--thr") == "auto", (
        "--thr defaults to self-calibration, not a hardcoded cut: guessing wrong "
        "silently merges every speaker into one")


def test_glossary_takes_a_file_of_terms(tree):
    """--glossary crypto.txt used to send the literal string 'crypto.txt' as a TERM.

    Silently: the model was told to listen out for the words "crypto dot txt"
    and nothing ever said the file had not been read.
    """
    tree.audio()
    (tree.path / "crypto.txt").write_text(
        "# subject vocabulary\n"
        "Dana Whitfield\n"
        "\n"
        "Northwind\n")

    p = tree.run("transcribe", "meeting.mp3", "--glossary", "crypto.txt", mklib=True)

    assert p.returncode == 0, "stdout=%r stderr=%r" % (p.stdout, p.stderr)
    got = flag_value(tree.invocation_for("engined.py"), "--glossary")
    assert got == "Dana Whitfield,Northwind", (
        "expected the file's terms, got %r -- comments and blank lines are "
        "dropped and the rest joined with commas" % (got,))


def test_a_glossary_path_that_is_not_there_is_an_error_not_a_term(tree):
    tree.audio()
    p = tree.run("transcribe", "meeting.mp3", "--glossary", "missing.txt")

    assert p.returncode == 2, (
        "a path-like --glossary with no file behind it must fail loudly, "
        "not become a term (rc=%d)" % p.returncode)
    assert "no glossary file at missing.txt" in p.stderr
    assert not tree.invocations


# =====================================================================
# 4. --out                                   (commits b6d69c1, 18863d4)
#
# b6d69c1: --out set where the closing "N meetings in ..." line POINTED and
# nothing else -- the invocation hardcoded --library "$REMOTE/library", and on a
# local run that string is executed right here. So the meetings landed in the
# install's own library while the closing line confirmed the directory the
# caller asked for, by counting a directory the run had just created and left
# empty. Silent data in a place the caller explicitly said not to put it.
#
# 18863d4: an unwritable --out warns and falls back. Refusing was the wrong
# trade -- nothing expensive has happened yet, but the same rule applied after a
# REMOTE run would strand a finished batch on a box about to be destroyed.
#
# The stub writes a meeting into whatever --library it is handed (MS_STUB_MKLIB),
# so these ask where the meetings actually WENT, not merely which flag was sent.
# =====================================================================

@BOTH_PATHS
def test_out_is_where_the_meetings_are_written(tree, kind):
    src = source_for(tree, kind)
    out = tree.path.parent / "somewhere else"

    p = tree.run("transcribe", src, "--out", str(out), mklib=True)

    assert p.returncode == 0, "stdout=%r stderr=%r" % (p.stdout, p.stderr)
    assert flag_value(tree.invocation_for("engined.py"), "--library") == str(out), (
        "the job was told to write somewhere other than --out")
    assert tree.meetings_in(out), (
        "no meeting under --out %s; the run reported: %r" % (out, p.stdout))
    assert not tree.meetings_in(tree.path / "library"), (
        "meetings landed in the install's own library despite --out -- this is "
        "the b6d69c1 defect: %r" % tree.meetings_in(tree.path / "library"))
    assert str(out) in p.stdout, "the closing line should name where they went"


@BOTH_PATHS
def test_without_out_the_meetings_go_to_the_default_library(tree, kind):
    src = source_for(tree, kind)

    p = tree.run("transcribe", src, mklib=True)

    assert p.returncode == 0, "stdout=%r stderr=%r" % (p.stdout, p.stderr)
    assert tree.meetings_in(tree.path / "library"), (
        "expected the meeting in %s, stdout=%r" % (tree.path / "library", p.stdout))


@BOTH_PATHS
def test_an_unwritable_out_warns_and_falls_back_to_the_default_library(tree, kind):
    """Warn and carry on. Refusing after a remote run would strand finished work."""
    src = source_for(tree, kind)
    # A regular file where a directory would have to be: mkdir -p cannot
    # succeed against ENOTDIR for any user, including root.
    blocker = tree.path.parent / "blocker"
    blocker.write_text("not a directory\n")
    out = blocker / "library"

    p = tree.run("transcribe", src, "--out", str(out), mklib=True)

    assert p.returncode == 0, (
        "an unwritable --out is a warning, not a stop -- the expensive part has "
        "not happened yet and refusing costs the caller a retype, while the same "
        "rule after a remote run strands the batch. rc=%d stderr=%r"
        % (p.returncode, p.stderr))
    assert str(out) in p.stderr and "cannot write" in p.stderr, (
        "the fallback has to be said out loud: %r" % p.stderr)

    default = tree.path / "library"
    assert str(default) in p.stderr, (
        "the warning should name where the meetings went instead: %r" % p.stderr)
    assert flag_value(tree.invocation_for("engined.py"), "--library") == str(default), (
        "the job must be re-pointed at the default library, not merely counted there")
    assert tree.meetings_in(default), (
        "nothing landed in the fallback library, so the work was lost silently")


# =====================================================================
# 5. Still open: the same set -e defect as (1), in ./transcribe
# =====================================================================

@pytest.mark.xfail(strict=True, reason=(
    "./transcribe dies at the final `M=\"$(ls -dt \"$LIBRARY\"/*/ ... | head -1)\"` "
    "when the run produced no meeting: the glob matches nothing, ls exits 2, "
    "`set -o pipefail` propagates that through `| head -1`, and `set -e` kills "
    "the script AT THE ASSIGNMENT -- so the `!! nothing arrived in $LIBRARY` "
    "branch below it is unreachable and the run exits 2 in silence. Same shape "
    "as the status bug fixed in 7cadd9b. FIX: make the assignment unable to "
    "abort, e.g. `M=\"$(ls -dt \"$LIBRARY\"/*/ 2>/dev/null | head -1)\" || M=\"\"`."))
def test_a_single_run_that_produces_nothing_says_so(tree):
    """The library is empty afterwards, and the user should be told which one."""
    tree.audio()

    # mklib is OFF: the python layer succeeds but writes no meeting, which is
    # what a fallen-over remote rsync or a recording with no speech looks like.
    p = tree.run("transcribe", "meeting.mp3")

    assert "nothing arrived" in p.stderr, (
        "the run left the library empty and said nothing about it "
        "(rc=%d, stdout=%r, stderr=%r)" % (p.returncode, p.stdout, p.stderr))
