"""The window geometry defaults, which have to agree across four files.

commit 7b83a77 -- "Stop giving each window 5s of context it does not use".

engined.py refuses a job whose geometry differs from the engine it already
built (engined.py: "it was built for window {res.window}s overlap {res.overlap}s,
and this job wants ..."), so a default that moves in one file and not the others
makes every DEFAULT run miss the resident engine and try to load a second one
onto a card with no room for it. test_shell.py states that invariant in a
comment and then checks only ./transcribe's end of it; this file checks the rest.

Static, via AST: batch.py and engined.py import vllm at module level and cannot
be imported on a CPU box, but the defaults are in the source text either way.
Every lookup below asserts it FOUND the argument first -- a test that silently
matched nothing would pin nothing.
"""
import ast
import re

import pytest

#: The python components that parse --window/--overlap off a command line. All
#: three build or accept a job's geometry, so all three have to agree.
PY_SOURCES = ("pipeline/batch.py",
              "pipeline/engined.py",
              "pipeline/transcribe_meeting.py")


def argparse_default(path, flag):
    """The `default=` on `add_argument("<flag>", ...)` in `path`.

    -> the literal default. Raises if the flag is absent or declared twice, so
    this cannot quietly stop testing anything when an argument is renamed.
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != flag:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        assert "default" in kw, "%s declares %s with no default" % (path.name, flag)
        found.append(ast.literal_eval(kw["default"]))
    assert len(found) == 1, (
        "expected exactly one `add_argument(%r)` in %s, found %d. If the "
        "argument moved, this test has to follow it rather than pass."
        % (flag, path.name, len(found)))
    return found[0]


def shell_var(repo, name):
    """The literal `NAME="..."` default assigned in ./transcribe."""
    src = (repo / "transcribe").read_text()
    hits = re.findall(r'^%s="([^"]*)"' % name, src, re.M)
    assert len(hits) == 1, (
        "expected one %s= default in ./transcribe, found %r" % (name, hits))
    return hits[0]


@pytest.mark.parametrize("rel", PY_SOURCES)
def test_overlap_defaults_to_zero_in_every_python_component(repo, rel):
    """0, not 5: 5 decodes 40s of audio to keep 30 and doubles the engine's KV."""
    assert argparse_default(repo / rel, "--overlap") == 0.0


@pytest.mark.parametrize("rel", PY_SOURCES)
def test_window_defaults_to_thirty_in_every_python_component(repo, rel):
    assert argparse_default(repo / rel, "--window") == 30.0


def test_the_shell_defaults_match_the_python_ones(repo):
    """./transcribe splices these into the command line it builds, so its
    literals are the fourth copy of the same two numbers."""
    assert float(shell_var(repo, "OVERLAP")) == 0.0
    assert float(shell_var(repo, "WINDOW")) == 30.0
    assert shell_var(repo, "THR") == argparse_default(
        repo / "pipeline" / "batch.py", "--thr") == "auto"


# FIXED (was xfail):
# build_engine() in pipeline/transcribe_meeting.py still carries `overlap=5.0` in its
# signature -- the pre-7b83a77 default, left behind when the four CLI defaults moved
# to 0. It is not live today only because all three callers (engined.py serve(),
# batch.py main(), transcribe_meeting.py main()) pass overlap explicitly. A fourth
# caller that omits it builds an engine sized for 40s of audio to keep 30, and since
# engined.py refuses any job whose geometry differs from the engine it built, every
# default job would then be refused by the resident engine it was supposed to reuse.
# FIX: change the signature default to 0.0 so the number is stated the same way in all
# five places.
def test_build_engine_carries_the_same_overlap_default(repo):
    src = (repo / "pipeline" / "transcribe_meeting.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "build_engine")
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                        [ast.literal_eval(d) for d in fn.args.defaults]))
    assert "overlap" in defaults, "build_engine no longer defaults overlap"
    assert defaults["overlap"] == 0.0
