"""Shared fixtures for the meetscribe test suite.

The bash entry points (./transcribe, ./engine) are tested for real -- as
subprocesses -- but with everything expensive replaced:

  * MS_PY points at a stub "interpreter" that records its argv, optionally
    prints a canned stdout, and exits with a code the test chooses. Both
    scripts resolve the interpreter once (`. env.sh; ${MS_PY:-python3}`) and
    splice that literal path into every command they run, so a stub there
    intercepts every call into the python layer without touching production
    code.
  * The scripts are COPIED into a temp tree next to a fake pipeline/, because
    both derive HERE from BASH_SOURCE and hang WORK/REMOTE/LIBRARY off it. Run
    in place they would write into the real checkout and, worse, find the real
    pipeline.

env.sh is gitignored and absent from a clean checkout; `. env.sh` failing is
expected and MS_PY falls back to the environment. The tree deliberately does
not create one.
"""

import os
import shutil
import stat
import subprocess

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Entry points copied into every sandbox tree.
SCRIPTS = ("transcribe", "engine")

# Modules the scripts reference by path. Contents are irrelevant -- MS_PY never
# actually interprets them -- but ./transcribe decides local-vs-remote by
# testing for pipeline/transcribe_meeting.py, so the files must EXIST.
PIPELINE_FILES = (
    "pipeline/engined.py",
    "pipeline/batch.py",
    "pipeline/transcribe_meeting.py",
    "pipeline/identify.py",
    "pipeline/mktxt.py",
    "pipeline/link/embed_batched.py",
    "pipeline/link/link.py",
)

# A stand-in for the python interpreter. Records one line per argv element,
# framed by markers so an empty argument (e.g. --roster '') stays visible.
STUB_PY = r"""#!/usr/bin/env bash
# Test stub: pretends to be python3, records argv, exits MS_STUB_RC.
set -u
{
  printf '=== INVOCATION\n'
  for a in "$@"; do printf '%s\n' "$a"; done
  printf '=== END\n'
} >> "$MS_STUB_LOG"

# Simulate the python layer writing a meeting into whatever --library it was
# handed, so a test can ask where the meetings actually landed rather than
# only which flag was passed.
if [ -n "${MS_STUB_MKLIB:-}" ]; then
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--library" ] && [ -n "$a" ]; then
      mkdir -p "$a/mtg-test"
      printf '{"id": "mtg-test"}\n' > "$a/mtg-test/meeting.json"
    fi
    prev="$a"
  done
fi

if [ -n "${MS_STUB_OUT:-}" ]; then printf '%s\n' "$MS_STUB_OUT"; fi
exit "${MS_STUB_RC:-0}"
"""


def parse_invocations(log_path):
    """Read the stub's log into a list of argv lists (one per invocation)."""
    log = Path(log_path)
    if not log.exists():
        return []
    invocations, current = [], None
    for line in log.read_text().split("\n")[:-1]:
        if line == "=== INVOCATION":
            current = []
        elif line == "=== END":
            invocations.append(current)
            current = None
        elif current is not None:
            current.append(line)
    return invocations


def flag_value(argv, flag):
    """The argument following `flag`, or None if the flag is absent."""
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else ""
    return None


class Tree:
    """A throwaway checkout: the real scripts, a fake pipeline, a stub python."""

    def __init__(self, path):
        self.path = Path(path)
        self.log = self.path / "_stub_argv.log"
        self.stub = self.path / "_stub_python"

    # -- running -----------------------------------------------------------
    def run(self, script, *args, stub_rc=0, stub_out=None, mklib=False,
            env=None, cwd=None, timeout=60):
        """Run ./<script> in the tree. Returns CompletedProcess (text mode)."""
        e = dict(os.environ)
        # Nothing from the developer's own shell may steer the scripts: MS_HOST
        # alone would turn every local test into an ssh attempt.
        for k in list(e):
            if k.startswith("MS_"):
                del e[k]
        e["MS_PY"] = str(self.stub)
        e["MS_STUB_LOG"] = str(self.log)
        e["MS_STUB_RC"] = str(stub_rc)
        if stub_out is not None:
            e["MS_STUB_OUT"] = stub_out
        if mklib:
            e["MS_STUB_MKLIB"] = "1"
        e.update(env or {})
        return subprocess.run(
            [str(self.path / script)] + [str(a) for a in args],
            cwd=str(cwd or self.path),
            env=e,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )

    # -- inspecting --------------------------------------------------------
    @property
    def invocations(self):
        return parse_invocations(self.log)

    def only_invocation(self):
        got = self.invocations
        assert len(got) == 1, "expected one call into the python layer, got %d: %r" % (
            len(got), got,
        )
        return got[0]

    def invocation_for(self, module):
        """The single argv whose first path-like element ends in `module`."""
        hits = [a for a in self.invocations if any(x.endswith(module) for x in a)]
        assert len(hits) == 1, "expected one %s call, got %d: %r" % (
            module, len(hits), self.invocations,
        )
        return hits[0]

    # -- fixtures for the tree itself --------------------------------------
    def mark_provisioning(self, when="2026-08-15T00:00:00Z"):
        (self.path / ".provisioning").write_text(when + "\n")

    def audio(self, name="meeting.mp3"):
        p = self.path / name
        p.write_bytes(b"\x00" * 64)
        return p

    def meetings_in(self, directory):
        d = Path(directory)
        if not d.exists():
            return []
        return sorted(str(p) for p in d.glob("*/meeting.json"))


@pytest.fixture
def tree(tmp_path):
    """A sandbox copy of the entry points with a stub interpreter."""
    root = tmp_path / "meetscribe"
    root.mkdir()
    for name in SCRIPTS:
        shutil.copy2(REPO / name, root / name)
        (root / name).chmod(0o755)
    for rel in PIPELINE_FILES:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# test placeholder -- MS_PY never interprets this\n")

    t = Tree(root)
    t.stub.write_text(STUB_PY)
    t.stub.chmod(t.stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return t


@pytest.fixture(scope="session")
def repo():
    """The real checkout, for static assertions about shipped files."""
    return REPO


# =====================================================================
# The data/state layer: the profile store, the library, and the scripts
# that rewrite them (test_postproc.py).
#
# Three rules everything below follows:
#
# * NOTHING TOUCHES THE REAL STORE OR LIBRARY. Every subprocess gets
#   MS_SPEAKER_DB and MS_WORK pointed inside tmp_path, and every meeting is
#   built in a tmp library. speakers.py resolves DB once at import time from
#   the environment, so the environment has to be right before the interpreter
#   starts -- which is why these scripts are run as subprocesses and not
#   in-process.
# * VOICEPRINTS ARE EXACT, NOT RANDOM. Every fixture cluster centroid is e0,
#   and every enrolled profile is at_cosine(c, axis): a unit vector whose
#   cosine with e0 is exactly c. A test that wants "just under the 0.55 accept
#   threshold" asks for 0.54 and gets 0.54, so a threshold test is about the
#   threshold rather than about where the fixture happened to land.
# * MEETINGS ARE BUILT THROUGH library.create(). Folder name, id, file stems
#   and meeting.json all come from production code, so a test cannot pass by
#   agreeing with a hand-rolled copy of the layout that has since drifted.
# =====================================================================
import json          # noqa: E402
import sqlite3       # noqa: E402
import sys           # noqa: E402

import numpy as np   # noqa: E402

PIPE = REPO / "pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

import library as LIB       # noqa: E402  (needs PIPE on sys.path)
import speakers as S        # noqa: E402

DIM = 16


# ------------------------------------------------------------------ vectors
def basis(i, dim=DIM):
    v = np.zeros(dim, dtype=np.float64)
    v[i] = 1.0
    return v


#: What every fixture meeting's cluster centroid is, so a profile's score
#: against it is whatever we constructed that profile to be.
REF = basis(0)


def at_cosine(c, axis):
    """A unit vector whose cosine with REF is exactly `c`.

    `axis` picks the orthogonal direction it leans into, so two profiles at the
    same cosine are still different vectors -- which is what makes the margin
    rule and the one-to-one rule testable at all.
    """
    assert axis != 0, "axis 0 is REF itself"
    v = c * basis(0) + np.sqrt(max(0.0, 1.0 - c * c)) * basis(axis)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ library
def write_clusters_npz(path, spec, meeting="fixture"):
    """The _clusters.npz link.py writes: one centroid per speaker, plus seconds.

    spec is {cluster_id: (centroid, seconds)}.
    """
    ids = list(spec)
    np.savez(str(path),
             centroid=np.array([spec[g][0] for g in ids], dtype=np.float32),
             cluster=np.array(ids),
             secs=np.array([spec[g][1] for g in ids], dtype=np.float32),
             meeting=np.array(meeting))


def make_meeting(lib, title, source, mid, clusters=None, names=None):
    """One meeting directory holding everything the post-GPU steps read.

    clusters is {cluster_id: (centroid, seconds)}; pass None for a meeting with
    no _clusters.npz, which is the "transcribe it again" case. names is the
    {cluster: name} map the transcript currently claims -- pass None to leave
    the file absent, which is what a meeting processed against an empty store
    looks like.
    """
    m = LIB.create(title, source, lib=str(lib), mid=mid)
    clusters = clusters or {}
    if clusters:
        write_clusters_npz(m.file("clusters", "npz"), clusters, meeting=mid)

    segs, t = [], 0.0
    for g, (_v, secs) in clusters.items():
        half = secs / 2.0
        for k in (1, 2):
            segs.append({"start": round(t, 2), "end": round(t + half, 2),
                         "window": 0, "local_speaker": "S01",
                         "speaker": "w000_%s" % g, "global": g,
                         "text": "%s line %d." % (g, k)})
            t += half
    body = {"audio": source, "duration_s": round(t, 2), "window_s": 30.0,
            "n_windows": 1, "coverage": 1.0, "segments": segs}
    m.file("transcript", "json").write_text(json.dumps(body))
    m.file("raw", "json").write_text(json.dumps(body))
    if names is not None:
        m.file("names", "json").write_text(json.dumps(names))
    m.write(duration_s=body["duration_s"], n_segments=len(segs))
    return m


def snapshot(root):
    """{relative path: bytes} for every file under root.

    A dry run is only a dry run if this is identical afterwards. names.json is
    the file that matters, but a run that quietly re-rendered a transcript or
    dropped a stray scratch file would be just as wrong.
    """
    root = Path(root)
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def rows_of(db_path, table):
    """Every row of `table` as a list of dicts, ordered by rowid."""
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in c.execute(
            "SELECT * FROM %s ORDER BY rowid" % table)]
    finally:
        c.close()


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def store(tmp_path):
    """Path to an empty profile store carrying the production schema.

    Opened through speakers.db() rather than hand-written DDL, so the fixture
    cannot drift from the one definition of that schema.
    """
    p = tmp_path / "speakers.db"
    S.db(str(p)).close()
    return p


@pytest.fixture
def conn(store):
    c = S.db(str(store))
    yield c
    c.close()


@pytest.fixture
def lib(tmp_path):
    d = tmp_path / "library"
    d.mkdir()
    return d


@pytest.fixture
def run_pipe(store, tmp_path):
    """Run a pipeline/ script as a subprocess against the fixture store.

    A subprocess, because speakers.py reads MS_SPEAKER_DB once at import: an
    in-process call would either hit the developer's real speakers.db or depend
    on which test imported it first. MS_WORK is redirected as well, so even a
    path that ignores MS_SPEAKER_DB cannot reach the checkout's own store or
    library.
    """
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)

    def run(script, *args, **kw):
        expect_rc = kw.pop("expect_rc", 0)
        assert not kw, "unexpected kwargs: %r" % (kw,)
        e = dict(os.environ)
        for k in list(e):
            if k.startswith("MS_"):
                del e[k]
        e["MS_SPEAKER_DB"] = str(store)
        e["MS_WORK"] = str(work)
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        p = subprocess.run(
            [sys.executable, str(PIPE / script)] + [str(a) for a in args],
            cwd=str(REPO), env=e,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=120)
        if expect_rc is not None:
            assert p.returncode == expect_rc, (
                "%s exited %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (script, p.returncode, p.stdout, p.stderr))
        return p

    return run


@pytest.fixture
def decisions(store):
    """-> {(meeting, cluster): {outcome, score, second}} from the store.

    identify.py records why it did what it did, which is a steadier assertion
    surface than the human-readable table it prints.
    """
    def read():
        c = sqlite3.connect(str(store))
        try:
            rows = c.execute("SELECT meeting, cluster, outcome, score, second"
                             " FROM decisions ORDER BY id").fetchall()
        finally:
            c.close()
        return {(m, cl): {"outcome": o, "score": sc, "second": se}
                for m, cl, o, sc, se in rows}

    return read


# =====================================================================
# assemble() and its guards, in pipeline/transcribe_meeting.py
# (test_assemble.py).
#
# That module imports vllm, transformers and moss_transcribe_diarize at module
# level and none of those exist on a CPU box. The tests still have to run
# against THE REAL FILE -- a copy of assemble() pasted into a test drifts the
# moment production changes and then tests nothing -- so the missing modules
# are stubbed into sys.modules for exactly as long as the import takes, the
# real source is exec'd from its real path, and the stubs are then removed.
# test_assemble.py asserts tm.__file__ IS the production path.
#
# The only stub whose behaviour matters is parse_transcript(): it is the seam
# between the model's raw text and assemble()'s logic. Here it reads a JSON
# list of {start, end, speaker, text}, so a test can state exactly what a
# window decoded without the test owning a transcript parser.
# =====================================================================
import ast                # noqa: E402
import importlib.util     # noqa: E402
import types              # noqa: E402
from collections import namedtuple    # noqa: E402

TM_PATH = REPO / "pipeline" / "transcribe_meeting.py"
BATCH_PATH = REPO / "pipeline" / "batch.py"
SR = 16000

#: What moss_transcribe_diarize.parse_transcript yields. assemble() reads
#: .start, .end, .speaker and .text off it and nothing else.
Seg = namedtuple("Seg", "start end speaker text")


def _parse_transcript(text):
    """Stub for moss_transcribe_diarize.parse_transcript.

    Times are LOCAL to the window, exactly as the model emits them; rebasing
    them onto the meeting clock is assemble()'s job and so is under test.
    """
    if not text or not text.strip():
        return []
    return [Seg(float(d["start"]), float(d["end"]), d.get("speaker", 0), d["text"])
            for d in json.loads(text)]


def _tm_stub_modules():
    """The module-level imports transcribe_meeting.py cannot satisfy here."""
    vllm = types.ModuleType("vllm")

    class _LLM:                                    # never constructed in tests
        def __init__(self, *a, **k):
            raise RuntimeError("no GPU in tests")

    class _SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    vllm.LLM = _LLM
    vllm.SamplingParams = _SamplingParams

    transformers = types.ModuleType("transformers")

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(*a, **k):
            raise RuntimeError("no model weights in tests")

    transformers.AutoProcessor = _AutoProcessor

    moss = types.ModuleType("moss_transcribe_diarize")
    moss.parse_transcript = _parse_transcript
    iu = types.ModuleType("moss_transcribe_diarize.inference_utils")
    iu.DEFAULT_PROMPT = "TEST_PROMPT"

    def _load_audio_item(*a, **k):
        raise RuntimeError("no audio decoder in tests")

    iu.load_audio_item = _load_audio_item
    moss.inference_utils = iu

    return {"vllm": vllm,
            "transformers": transformers,
            "moss_transcribe_diarize": moss,
            "moss_transcribe_diarize.inference_utils": iu}


def _load_transcribe_meeting():
    stubs = _tm_stub_modules()
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "meetscribe_transcribe_meeting_under_test", str(TM_PATH))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return mod


@pytest.fixture(scope="session")
def tm():
    """The real pipeline/transcribe_meeting.py, imported with stubbed deps."""
    return _load_transcribe_meeting()


@pytest.fixture(scope="session")
def assemble_call_arities():
    """-> {path: [n, ...]}: how many names each `= assemble(...)` unpacks.

    AST, not import: batch.py imports vllm at module level and cannot be
    imported here at all, but parsing needs only the text. This is the
    grep-level check that would have caught assemble() growing a fourth
    return value while a caller still unpacked three.
    """
    def arities(path):
        found = []
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "assemble":
                continue
            target = node.targets[0]
            found.append(len(target.elts) if isinstance(target, ast.Tuple) else 1)
        return found

    return {TM_PATH: arities(TM_PATH), BATCH_PATH: arities(BATCH_PATH)}


def make_wav(dur_s, speech_until=None, amp=0.5, seed=0):
    """Loud audio of dur_s seconds, exact silence after speech_until.

    Loud enough to clear SILENCE_GATE_DB (-70 dB) by a wide margin; the silent
    tail is exact zeros, which reads as -120 dB, so speech_end lands where the
    test put it rather than at the end of the file.
    """
    n = int(round(dur_s * SR))
    rng = np.random.default_rng(seed)
    wav = (rng.standard_normal(n).astype(np.float32) * 0.05 + amp).astype(np.float32)
    if speech_until is not None:
        wav[int(round(speech_until * SR)):] = 0.0
    return wav


@pytest.fixture
def wav_factory():
    """-> make_wav(dur_s, speech_until=None): audio for the coverage guard."""
    return make_wav


class _Completion:
    def __init__(self, text, finish_reason):
        self.text = text
        self.finish_reason = finish_reason


class _Output:
    """Shaped like a vLLM RequestOutput: .outputs[0].text / .finish_reason."""

    def __init__(self, text, finish_reason):
        self.outputs = [_Completion(text, finish_reason)]


@pytest.fixture
def win():
    """Build one window's worth of input for assemble().

    offset: where this window's audio starts in the meeting, seconds.
    core:   (lo, hi) in WINDOW-LOCAL seconds. A segment whose midpoint lands
            outside it was decoded in context padding and is a neighbour's to
            own -- an orphan.
    segs:   (start, end, text) or (start, end, text, speaker), window-local.
    """
    def _win(offset, core, segs, finish_reason="stop"):
        return {"offset": float(offset), "core": (float(core[0]), float(core[1])),
                "segs": list(segs), "finish_reason": finish_reason}

    return _win


@pytest.fixture
def assemble(tm):
    """Run the real assemble() over windows built by `win`.

    -> (segments, coverage, capped, speech_end)

    The silence gate is off by default: these tests are about the seam,
    repetition and coverage guards, and a gate drop would mask them.
    """
    def _assemble(windows, dur=None, wav=None, no_silence_gate=True):
        outs, offsets, cores = [], [], []
        for w in windows:
            payload = [{"start": float(s[0]), "end": float(s[1]), "text": s[2],
                        "speaker": s[3] if len(s) > 3 else 0}
                       for s in w["segs"]]
            outs.append(_Output(json.dumps(payload), w["finish_reason"]))
            offsets.append(w["offset"])
            cores.append(w["core"])
        if dur is None:
            ends = [w["offset"] + s[1] for w in windows for s in w["segs"]]
            ends.append(max((w["offset"] + w["core"][1] for w in windows), default=1.0))
            dur = max(ends) + 5.0
        if wav is None:
            wav = make_wav(dur)
        return tm.assemble(outs, offsets, cores, wav, dur, no_silence_gate)

    return _assemble


# =====================================================================
# Constrained speaker clustering, in pipeline/cluster_speakers.py
# (test_clustering.py).
#
# That module imports numpy and stdlib only, so it is imported for real -- no
# stubbing, no exec of source text. PIPE is already on sys.path above.
#
# Two rules the fixtures below follow:
#
# * COSINES ARE CONSTRUCTED, NOT SAMPLED. Every fixture embedding is built from
#   orthogonal axes via axis_vec/speaker_vec, so "these two aggregates sit at
#   0.95 and cosine wants to merge them" is an exact property of the fixture
#   rather than whatever a random draw happened to produce. A clustering test
#   whose premise is approximate is testing the draw.
# * THE POSTCONDITION IS STATED ONCE. violating_pairs() is the only definition
#   of "a cannot-link pair ended up in one cluster" the tests use, and it is
#   the same expression cluster() asserts on. Every stage is checked against
#   it, not against a speaker COUNT -- the count is exactly what let the bug in
#   6506bd8 live on real recordings.
# =====================================================================
import cluster_speakers as CS_MOD    # noqa: E402  (needs PIPE on sys.path)


@pytest.fixture
def cs():
    """The real pipeline/cluster_speakers.py."""
    return CS_MOD


# ------------------------------------------------- embeddings with exact cosines
def l2norm(v):
    """L2-normalise, the way link.py's aggregate() does before clustering."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def axis_vec(components, dim=DIM):
    """Unit vector from {axis_index: coefficient}.

    Distinct axes are orthogonal, so the cosine between two vectors built this
    way is fixed by the coefficients they share and by nothing else.
    """
    v = np.zeros(dim, dtype=np.float64)
    for i, c in components.items():
        v[i] = c
    return l2norm(v)


def speaker_vec(speaker_axis, residual_axis, within_cos, dim=DIM):
    """A member of a speaker whose members sit at exactly `within_cos` apart.

    Two members share `speaker_axis` with coefficient sqrt(within_cos) and lean
    into different residual axes, so their cosine is within_cos exactly.
    """
    return axis_vec({speaker_axis: np.sqrt(within_cos),
                     residual_axis: np.sqrt(1.0 - within_cos)}, dim)


# ------------------------------------------------- reference implementations
def naive_constrained_linkage(S, cannot):
    """Greedy constrained average linkage, written the O(n^3)-interpreted way.

    An independent statement of what 2851014's live-similarity-matrix rewrite is
    supposed to compute: rescan every surviving pair on every merge. Tie-break
    is the first maximum in (i, j) order, which is what np.argmax over a
    row-major upper triangle gives. -> (heights, order)
    """
    n = len(S)
    Ssum = np.asarray(S, dtype=np.float64).copy()
    cnt = np.ones((n, n))
    CL = np.asarray(cannot).astype(bool).copy()
    alive = [True] * n
    heights, order = [], []
    while True:
        best, bi, bj = None, -1, -1
        for i in range(n):
            if not alive[i]:
                continue
            for j in range(i + 1, n):
                if not alive[j] or CL[i, j]:
                    continue
                v = Ssum[i, j] / cnt[i, j]
                if best is None or v > best:
                    best, bi, bj = v, i, j
        if best is None:
            break
        heights.append(float(best))
        order.append((bi, bj))
        for o in range(n):
            if not alive[o] or o == bi or o == bj:
                continue
            Ssum[bi, o] += Ssum[bj, o]
            Ssum[o, bi] = Ssum[bi, o]
            cnt[bi, o] += cnt[bj, o]
            cnt[o, bi] = cnt[bi, o]
            CL[bi, o] |= CL[bj, o]
            CL[o, bi] = CL[bi, o]
        alive[bj] = False
    return heights, order


def brute_cluster_cannot_link(lab, cannot, k):
    """The k x k block relation, computed one member pair at a time."""
    out = np.zeros((k, k), dtype=bool)
    if cannot is None or not len(cannot):
        return out
    lab = np.asarray(lab)
    for a in range(len(lab)):
        for b in range(len(lab)):
            if cannot[a, b]:
                out[lab[a], lab[b]] = True
    return out


def violating_pairs(lab_core, cannot):
    """Forbidden pairs sharing a cluster -- THE postcondition, stated once.

    Deliberately the same expression cluster() asserts on, so a stage-by-stage
    test and the production check cannot disagree about what a violation is.
    """
    lab = np.asarray(lab_core)
    if cannot is None or not len(cannot):
        return np.empty((0, 2), dtype=int)
    return np.argwhere(np.triu(np.asarray(cannot, dtype=bool), 1)
                       & (lab[:, None] == lab[None, :]))


# ------------------------------------------------- partitions
def partition_of(lab):
    """A labelling as a set of frozensets, so labellings compare by grouping."""
    lab = np.asarray(lab)
    return {frozenset(np.where(lab == c)[0].tolist()) for c in set(lab.tolist())}


def is_coarsening(coarse, fine):
    """True if every block of `fine` sits inside one block of `coarse`."""
    return all(any(b <= B for B in coarse) for b in fine)


# ------------------------------------------------- random inputs
def random_similarity(n, rng, dim=24):
    """Gram matrix of n random unit vectors: symmetric, unit diagonal, in [-1, 1]."""
    X = rng.normal(size=(n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def random_cannot(n, rng, p=0.25):
    """A symmetric hollow boolean relation, roughly `p` dense."""
    C = rng.random((n, n)) < p
    C = np.triu(C, 1)
    return C | C.T


def random_meeting(seed, n_windows=6, n_speakers=3, dim=DIM, scale=0.6,
                   sec_lo=3.0, sec_hi=25.0):
    """A synthetic meeting shaped like link.py's aggregate() output.

    Every speaker talks in every window, so every window carries a full set of
    cannot-links -- the regime where honouring them AFTER the cut is what
    decides the answer. -> (A, secs, keys) ready for cluster_speakers.cluster().
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_speakers, dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    keys, secs, rows = [], [], []
    for w in range(n_windows):
        for li in range(n_speakers):
            keys.append((w, "S%02d" % li))
            secs.append(float(rng.uniform(sec_lo, sec_hi)))
            rows.append(l2norm(base[li] + rng.normal(scale=scale, size=dim)))
    return np.vstack(rows), np.asarray(secs, dtype=float), keys
