"""The flag census: what the python modules define, against what the wrappers forward.

Every other test in this suite exercises the modules -- which is exactly where a
dropped flag WORKS. `python3 pipeline/batch.py rec.mp3 --no-clips` has always
honoured `--no-clips`; `./transcribe rec.mp3 --no-clips` did not, because the
bash wrapper in front of it had never heard of the flag and fell through to its
"this must be a filename" arm. Defects of that exact shape kept shipping, and 654
module-level tests caught none of them. Four that this file reproduces against
the commit before each fix, and fails on:

  * `--replace`       c3772ee -- ./transcribe -> batch.py. Printed usage at you.
  * `--no-clips`      3d7bdc4 -- ./transcribe -> batch.py. Clips were cut
                                 whatever you asked for.
  * `--speaker-db`    95bfbca -- batch.py -> link.py. The bank inside link.py
                                 was always empty, so every recording was a
                                 cold start: 2.63% of speech under a wrong name
                                 against 0.18%.
  * multi-file argv   4fb1262 -- ./transcribe -> batch.py. `SRC="$1"` kept the
                                 LAST file of a glob and transcribed one
                                 recording of three, reporting success;
                                 batch.py had taken nargs="+" all along.

`--speaker-db` was dropped by a python module building another module's argv, not
by a wrapper, so this file checks that edge too -- `argv_link` in batch.py is a
command line in front of link.py exactly as ./transcribe is one in front of
batch.py.

Two more are named in the same audit and are NOT reproducible as forwarding
drops, which is worth knowing before trusting this file too far. `--roster`
(ba9df86) and `--manifest` (5bbf87e) were each added to the module and to its
caller in one commit, so there was never a revision where the flag existed and
was not passed. Their cost was real; a census cannot see a flag nobody had
written yet.

So this file does not run anything. It reads both sides out of the shipped
source and reconciles them: argparse and argv-building lists via `ast` (batch.py
imports vllm and cannot be imported on a CPU box), the wrappers via their `case`
arms and the literal text of the commands they build.

WHAT THIS PROVES, AND WHAT IT DOES NOT. It answers one narrow question per flag:
does the thing in front of the module name it at all. Two things it cannot see:

  * Whether the flag reaches EVERY implementation behind the wrapper.
    ./transcribe builds two argument arrays -- one for a batch, one for a
    single file -- and a flag can be spliced into one and missing from the
    other, which is what `--replace` was. It used to build three: the third was
    the five-script chain the single-file branch ran when there was no daemon,
    and collapsing it (plans/ARCHITECTURE.md section 6.1) is what took the count
    down. test_shell.py runs the wrapper on both remaining paths, with and
    without a daemon; this file cannot see either.
  * Whether the module does anything with the flag once it arrives. The other
    half of 4fb1262 was `--thr`: link.py parsed it, computed `fixed_thr` from it
    and then used it only inside `if args.legacy_cluster`, so `--thr 0.8` ran at
    0.62 and printed "thr=0.8000" while doing nothing of the kind. Forwarding is
    all that is asserted here.
"""

import ast
import re


# ---------------------------------------------------------------------------
# Reading the python side
# ---------------------------------------------------------------------------

def _argparse_spec(path):
    """{subcommand or None: {"--flag": lineno}} plus {subcommand: {positional: nargs}}.

    Static, because batch.py imports vllm at module level: importing it here, or
    shelling out to `--help`, needs a GPU. `ast` needs nothing.
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    # `lk = sub.add_parser("link")` -- the variable is how add_argument calls
    # further down say which subcommand they belong to.
    owner = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "add_parser":
            continue
        args = node.value.args
        if args and isinstance(args[0], ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    owner[target.id] = args[0].value

    flags, positionals = {}, {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        sub = owner.get(fn.value.id) if isinstance(fn.value, ast.Name) else None
        if name.startswith("-"):
            flags.setdefault(sub, {}).setdefault(name, node.lineno)
        else:
            nargs = next((kw.value.value for kw in node.keywords
                          if kw.arg == "nargs" and isinstance(kw.value, ast.Constant)),
                         None)
            positionals.setdefault(sub, {})[name] = (nargs, node.lineno)
    return flags, positionals


def _argv_builder_flags(path, target):
    """{"--flag"} for the function in `path` that builds an argv list running
    `target`, or None if there is no such function.

    batch.py hands link.py a python list rather than a shell string, so the
    same drop looks like a missing `argv += ["--flag", value]` instead of a
    missing case arm. Every string constant in that function starting with `--`
    counts as forwarded; the function builds one command line and nothing else.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = [n.value for n in ast.walk(fn)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if any(target in v for v in names):
            hits.append((fn, {v for v in names if v.startswith("--")}))
    # The INNERMOST match. `argv_link` is nested inside `run_job`, and run_job
    # also builds mktxt.py's and clips.py's command lines -- taking the outer
    # one would pool every flag in the queue and the check would pass on flags
    # link.py never sees.
    inner = [(fn, flags) for fn, flags in hits
             if not any(o is not fn and fn.lineno <= o.lineno
                        and o.end_lineno <= fn.end_lineno for o, _ in hits)]
    if not inner:
        return None
    fn, flags = inner[0]
    return flags, fn.name, fn.lineno


# ---------------------------------------------------------------------------
# Reading the bash side
# ---------------------------------------------------------------------------

# A case arm whose patterns are all flags: `--thr)`, `-h|--help)`,
# `--per-speaker|--min-core|--refine)`. Anchored to a line start, to `in ` (the
# one-line `case "$a" in --host) ... esac` in ./speakers) or to a preceding `;;`,
# so the `)` of a $(...) or of a function definition cannot match.
_FLAG_ARM = re.compile(
    r"(?:^[ \t]*|\bin[ \t]+|;;[ \t]+)"
    r"(-{1,2}[A-Za-z][\w-]*(?:\|-{1,2}[A-Za-z][\w-]*)*)\)", re.M)

# Any arm at all, flags or command words -- used to work out which `case "$CMD"`
# branch an invocation sits in, since ./speakers runs `speakers.py "$CMD" "$@"`.
_ANY_ARM = re.compile(r"(?m)^[ \t]*([A-Za-z*][\w|.*-]*)\)")

_FLAG_TOKEN = re.compile(r"--[A-Za-z][\w-]*")


def _case_flags(text):
    """{"--flag": lineno} for every flag the wrapper's case arms accept."""
    out = {}
    for m in _FLAG_ARM.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        for tok in m.group(1).split("|"):
            out.setdefault(tok, line)
    return out


def _mentions(text):
    """Every --flag the wrapper names anywhere: accepted in a case arm, or
    spliced into a command line the wrapper builds itself (--library, --titles,
    --move-audio). Both count as reached: the user does not have to type a flag
    the wrapper always sends."""
    return set(_FLAG_TOKEN.findall(text))


def _invocations(text, module):
    """Every command line in the wrapper that runs `module`, joined across
    backslash continuations. Returns (subcommands, forwards_user_args, lineno)."""
    out = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not re.search(r"[/'\"]%s\b" % re.escape(module), line):
            continue
        joined, j = line, i
        while joined.rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            joined = joined.rstrip()[:-1] + " " + lines[j]
        tail = joined.split(module, 1)[1]
        # The first bare word after the path, if there is one, names the
        # subcommand. `"$CMD"` means whichever case arm we are standing in --
        # ./speakers runs `speakers.py "$CMD" "$@"` for the three profile-*
        # commands, and the arm is the only place their names appear.
        words = tail.lstrip("'\" ").split()
        first = words[0].strip("'\"") if words else ""
        subs = {None}
        if first == "$CMD":
            arms = [m for m in _ANY_ARM.finditer(text)
                    if text.count("\n", 0, m.start()) < i]
            subs = set(arms[-1].group(1).split("|")) if arms else {None}
        elif first and not first.startswith(("-", "$", "|", "2")):
            subs = {first}
        # `"$@"` is the user's own arguments, whatever they were. An array the
        # wrapper built itself (${ARGS[@]}) is not -- its contents are the
        # literal flags above, and _mentions already sees those.
        out.append((subs, '"$@"' in tail, i + 1))
    return out


def _forwarding(text, module):
    """{subcommand or None} for the branches that hand the user's own args through."""
    live = set()
    for subs, fwd, _ in _invocations(text, module):
        if fwd:
            live |= subs
    return live


def _reached(text, module):
    """(mentions, forwarded-subcommands) -- the two ways a flag can be live."""
    return _mentions(text), _forwarding(text, module)


# ---------------------------------------------------------------------------
# The wrappers, and the modules each one runs
# ---------------------------------------------------------------------------

# Derived, not declared: whichever of the four modules a wrapper names is a
# wrapper/module pair this test owns. Move a module and the pair moves with it.
MODULES = ("pipeline/batch.py", "pipeline/speakers.py",
           "pipeline/relabel.py", "pipeline/link/link.py")
WRAPPERS = ("transcribe", "speakers")


def _pairs(repo):
    for w in WRAPPERS:
        text = (repo / w).read_text()
        for mod in MODULES:
            if re.search(r"[/'\"]%s\b" % re.escape(mod.split("/")[-1]), text):
                yield w, mod, text


def _inner_edges(repo):
    """(builder, target, forwarded, function, lineno) for every module in this
    set that builds another one's command line."""
    for builder in MODULES:
        for target in MODULES:
            if builder == target:
                continue
            # On a path boundary. Bare "speakers.py" also matches the
            # "cluster_speakers.py" in link.py's --thr help text, which made a
            # help string look like a call.
            found = _argv_builder_flags(repo / builder, "/" + target.split("/")[-1])
            if found:
                yield (builder, target) + found


# ---------------------------------------------------------------------------
# Deliberate omissions. Each says why the wrapper SHOULD not carry the flag.
# ---------------------------------------------------------------------------

ALLOWED = {
    # --- batch.py knobs that only make a run worse or slower -----------------
    ("transcribe", "pipeline/batch.py", "--no-convert"):
        "a benchmark switch; measured at roughly half the throughput on a 5090",
    ("transcribe", "pipeline/batch.py", "--no-overlap-embed"):
        "a benchmark switch; it only makes the run slower",
    ("transcribe", "pipeline/batch.py", "--embed-ckpt"):
        "vectors from another model do not compare against the store, and a "
        "resident daemon builds the Embedder without it anyway (engined.py:247)",
    ("transcribe", "pipeline/batch.py", "--embed-config"):
        "meaningless without --embed-ckpt, which is not exposed either",

    # --- batch.py -> link.py: the bench harness ------------------------------
    # ./transcribe had six entries of its own here, for the link.py the
    # five-script chain ran directly. That chain is gone and batch.py is the
    # only caller left, so this is the only edge into link.py there is.
    ("pipeline/batch.py", "pipeline/link/link.py", "--legacy-cluster"):
        "the comparison arm: 13.8% of speech on the wrong person against 0.18%",
    ("pipeline/batch.py", "pipeline/link/link.py", "--sweep"):
        "bench/ threshold sweep, not something a transcription run does",
    ("pipeline/batch.py", "pipeline/link/link.py", "--tag"):
        "labels a --sweep row in the bench output; meaningless outside one",
    ("pipeline/batch.py", "pipeline/link/link.py", "--ref"):
        "scores against a reference RTTM; bench only",

    # --- batch.py -> link.py: the five clustering knobs ----------------------
    # These were forwarded, faithfully, until the pass that added these five
    # entries -- and forwarding is what made them look alive. Every one of them
    # prints into link.py's AGG or CLUSTER line, so `--durable 9` came back in
    # the log as durable=9 while cluster_speakers.cluster(), the only code that
    # reads it, had not run. That is the whole reason not to forward a flag to a
    # branch nothing takes: the log says it was honoured.
    ("pipeline/batch.py", "pipeline/link/link.py", "--min-core"):
        "the clustering core cut. Read by cluster_speakers.cluster() under "
        "--legacy-cluster (link.py:259) and by --sweep (:244, :253); the "
        "matching path aggregates every label and cuts nothing, and the "
        "core(>=2.0s) figure in the AGG line at :240 is printed, not applied",
    ("pipeline/batch.py", "pipeline/link/link.py", "--refine"):
        "leave-one-out reassignment passes. Only cluster() has centroids to "
        "reassign against -- link.py:259 and the --sweep loop at :253. Matching "
        "compares each aggregate against the store once and never moves it",
    ("pipeline/batch.py", "pipeline/link/link.py", "--durable"):
        "how much speech a cannot-link claim needs to bind. Read only at "
        "link.py:260, inside --legacy-cluster. RUNBOOK's 37.3%% -> 22.1%% cpCER "
        "table is a measurement of THAT path; nothing merges on this one, so "
        "there is no merge for a claim to forbid",
    ("pipeline/batch.py", "pipeline/link/link.py", "--guard"):
        "window span a cannot-link claim is trusted across; the other half of "
        "--durable and read on the same line (link.py:260), so it is dead for "
        "the same reason -- no merges to constrain",
    ("pipeline/batch.py", "pipeline/link/link.py", "--min-cluster-sec"):
        "below this a cluster is absorbed into its nearest neighbour "
        "(link.py:261). Absorption is a merge; the matching path has none, and "
        "a short aggregate is either recognised or left unnamed",

    # --- speakers.py: dead CLI -----------------------------------------------
    ("speakers", "pipeline/speakers.py", "--run-dir"):
        "accepted and ignored: index_clusters takes run_dir and never reads it "
        "(speakers.py:211)",
}

# Flags the wrapper owns outright: it consumes them and nothing downstream has
# ever heard of them.
WRAPPER_OWNED = {
    ("transcribe", "--host"): "names the box to ssh to; the wrapper owns the transport",
    ("transcribe", "--out"): "the wrapper's name for batch.py's --library, resolved here first",
    ("transcribe", "--name"): "kept only to refuse it: enrolling during a run "
                              "was identify.py --enroll on the deleted chain, "
                              "and it named one recording's cluster rather "
                              "than the person. Both are now deleted; the arm "
                              "prints ./speakers name and exits 2",
    ("transcribe", "--help"): "the wrapper's own usage",
    ("speakers", "--host"): "names the box to ssh to; the wrapper owns the transport",
    ("speakers", "--help"): "the wrapper's own usage",
}

# ---------------------------------------------------------------------------
# Defects still on the floor. Strict: fixing one and leaving it listed FAILS,
# which is the signal to delete the line rather than the test.
# ---------------------------------------------------------------------------

# Two entries left here when this list was written -- link.py's --speaker-db
# and --condition -- and neither was fixed by forwarding a flag. ./transcribe
# stopped running link.py at all: the five-script chain that did is deleted, and
# batch.py, which passes both, is now the only way in. That is the third way an
# entry goes stale, and the check below covers it.
KNOWN_DROPS = {
    ("transcribe", "pipeline/batch.py", "--condition"):
        "a person is stored once per circumstance -- phone, far-field, that "
        "conference room -- and ./transcribe cannot say which, so every voice "
        "it files lands under the default one. ./speakers name already takes "
        "--condition; the transcribe side is the half that is missing.",
    ("transcribe", "pipeline/batch.py", "--snap"):
        "window geometry. Lands in SRCS and the run dies with `not a file: "
        "--snap`, exit 2.",
    ("transcribe", "pipeline/batch.py", "--slide"):
        "window geometry, and the cheaper half of --overlap: 1.2x the audio in "
        "native-sized requests against 1.33x in oversized ones. Same death.",
}


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def test_the_parsers_still_find_both_sides(repo):
    """A census that parses to nothing passes every assertion below it.

    Both sides are read out of source with regexes and `ast`, so a reformat can
    silently empty either one. This pinned a COUNT under each parser -- 25 flags
    out of batch.py, 20 case arms out of ./transcribe, 10 on the link.py edge,
    8 subparsers out of speakers.py -- and deleting five dead tuning knobs
    (--min-core, --refine, --durable, --guard, --min-cluster-sec) moved three of
    the four in one pass, with the fourth already sitting on its floor. A number
    you re-set to whatever the last deletion left is not a floor, and it fails
    with "the ast walk has lost the parser" when the parser is fine.

    The count was never the property. A parser that breaks here returns nothing,
    or near it -- a regex that stops matching case arms matches none of them, an
    `ast` walk that loses `add_argument` loses every one. So name what must be
    found. Deleting one of these flags then has to be done twice, which is the
    right price for deleting the thing another test is standing on.
    """
    flags, _ = _argparse_spec(repo / "pipeline" / "batch.py")
    # --replace and --no-clips are two of the four defects this file was written
    # for; the rest are what every run passes.
    missing = ({"--replace", "--no-clips", "--thr", "--library", "--window",
                "--titles"} - set(flags.get(None, {})))
    assert not missing, (
        "batch.py's argparse block parsed to %d flags and %r is not among them; "
        "the ast walk has lost the parser"
        % (len(flags.get(None, {})), sorted(missing)))

    sub_flags, _ = _argparse_spec(repo / "pipeline" / "speakers.py")
    missing = {"link", "name", "review", "suggest", "profiles"} - set(sub_flags)
    assert not missing, (
        "speakers.py parsed to subparsers %r, without %r; add_parser assignments "
        "are no longer being followed" % (sorted(map(str, sub_flags)), sorted(missing)))

    case = _case_flags((repo / "transcribe").read_text())
    missing = {"--host", "--thr", "--window", "--replace", "--no-clips",
               "--glossary"} - set(case)
    assert not missing, (
        "./transcribe's argument loop parsed to %r, without %r"
        % (sorted(case), sorted(missing)))

    stext = (repo / "speakers").read_text()
    assert "--host" in _case_flags(stext), (
        "./speakers takes --host from anywhere in its arguments; the arm that "
        "does it is no longer being found")
    assert _forwarding(stext, "speakers.py") >= {"link", "review", "name"}, (
        "./speakers hands the user's own arguments to speakers.py per "
        "subcommand; that is no longer being seen: %r"
        % (_forwarding(stext, "speakers.py"),))

    edge = _argv_builder_flags(repo / "pipeline" / "batch.py", "/link.py")
    assert edge, ("batch.py builds no link.py command line; the argv-building "
                  "function is no longer being found")
    # --speaker-db is the one 95bfbca lost. The other four are the arguments
    # without which link.py does not start.
    missing = {"--run", "--npz", "--out", "--speaker-db"} - edge[0]
    assert not missing, (
        "batch.py's link.py command line parsed to %r, without %r"
        % (sorted(edge[0]), sorted(missing)))

    assert _forwarding(stext, "relabel.py") == {None}, (
        "./speakers apply forwards \"$@\" straight to relabel.py; if it stopped "
        "doing that, every relabel flag below became unreachable")


def test_every_module_flag_reaches_its_wrapper(repo):
    """A flag the module defines that the wrapper in front of it never names."""
    drops = []
    for wrapper, module, text in _pairs(repo):
        flags, _ = _argparse_spec(repo / module)
        mentions, forwarded = _reached(text, module.split("/")[-1])
        for sub, defined in flags.items():
            for flag, lineno in sorted(defined.items()):
                if flag in mentions or sub in forwarded:
                    continue
                # Qualified by subcommand where there is one, so allowing a
                # flag on `review` does not silence the same name on `suggest`.
                key = (wrapper, module, "%s:%s" % (sub, flag) if sub else flag)
                if key in ALLOWED or key in KNOWN_DROPS:
                    continue
                where = "%s (%s:%d)" % (flag, module, lineno)
                if sub:
                    where += " on the `%s` subcommand" % sub
                drops.append(
                    "  %s defines %s -- ./%s never passes it.\n"
                    "      Either forward it, or add "
                    "%r to ALLOWED with the reason." % (module, where, wrapper, key))
    assert not drops, (
        "flags that exist in the python layer and cannot be reached through "
        "the wrapper:\n" + "\n".join(drops))


def test_every_module_flag_reaches_the_module_that_calls_it(repo):
    """The same defect one level down: batch.py builds link.py's command line.

    That is where --speaker-db (95bfbca) and --roster (ba9df86) were lost, and
    it is the more expensive of the two levels -- a flag missing here is missing
    on EVERY path, wrapper or not, because batch.run_job is the one
    implementation all four executions share.
    """
    edges = list(_inner_edges(repo))
    assert edges, (
        "no module in %r builds another one's argv any more; if that call moved "
        "to an import, this test needs to follow it" % (MODULES,))

    drops = []
    for builder, target, forwarded, fn_name, fn_line in edges:
        flags, _ = _argparse_spec(repo / target)
        for sub, defined in flags.items():
            for flag, lineno in sorted(defined.items()):
                if flag in forwarded:
                    continue
                key = (builder, target, "%s:%s" % (sub, flag) if sub else flag)
                if key in ALLOWED or key in KNOWN_DROPS:
                    continue
                drops.append(
                    "  %s defines %s (%s:%d) -- %s:%d (%s) builds its command "
                    "line and never adds it.\n      Either forward it, or add "
                    "%r to ALLOWED with the reason."
                    % (target, flag, target, lineno, builder, fn_line, fn_name, key))
    assert not drops, (
        "flags one module defines that the module calling it never passes:\n"
        + "\n".join(drops))


def test_no_wrapper_flag_is_unknown_downstream(repo):
    """The other direction: a flag the wrapper accepts that nothing defines.

    This is what a rename looks like from the wrapper's side -- the case arm
    keeps consuming the argument and the value goes nowhere.
    """
    orphans = []
    for wrapper in WRAPPERS:
        text = (repo / wrapper).read_text()
        downstream = set()
        for w, module, _ in _pairs(repo):
            if w != wrapper:
                continue
            flags, _ = _argparse_spec(repo / module)
            for defined in flags.values():
                downstream |= set(defined)
        for flag, lineno in sorted(_case_flags(text).items()):
            if flag in downstream or not flag.startswith("--"):
                continue
            if (wrapper, flag) in WRAPPER_OWNED:
                continue
            orphans.append(
                "  ./%s:%d accepts %s and no module downstream defines it.\n"
                "      Either it was renamed in the python layer, or it belongs "
                "in WRAPPER_OWNED with the reason." % (wrapper, lineno, flag))
    assert not orphans, (
        "flags the wrapper consumes that go nowhere:\n" + "\n".join(orphans))


def test_multi_file_positionals_survive_the_argument_loop(repo):
    """batch.py has taken nargs="+" since it existed; ./transcribe kept one file.

    `SRC="$1"` in the fallback arm meant every glob -- which is how anyone
    writes a batch -- transcribed only the LAST file, and said it had succeeded
    (4fb1262). Accumulation is the property; assert it rather than the variable.
    """
    flags, positionals = _argparse_spec(repo / "pipeline" / "batch.py")
    audio = positionals.get(None, {}).get("audio")
    assert audio is not None, "batch.py no longer takes a positional named audio"
    nargs, lineno = audio
    assert nargs == "+", (
        "batch.py:%d takes audio nargs=%r; this test pins the wrapper against "
        "'more than one'" % (lineno, nargs))

    text = (repo / "transcribe").read_text()
    fallback = re.search(r"(?m)^[ \t]*\*\)(.*(?:\n(?![ \t]*(?:-|\*|esac)).*)*)", text)
    assert fallback, "./transcribe's argument loop has no default `*)` arm"
    body = fallback.group(1)
    assert re.search(r"\w+\+=\(", body), (
        "./transcribe's `*)` arm does not accumulate -- batch.py takes "
        "nargs=\"+\" and this arm keeps one file:\n%s" % body)


def test_known_drops_are_still_dropped(repo):
    """Strict, the way the xfails in this suite are: a fixed defect that is
    still listed here fails, and the fix is to delete the line.

    Three ways an entry goes stale, and all three have to fail. The flag gets
    forwarded; the module stops defining it; or the caller stops calling that
    module at all -- which is how the two link.py entries went: collapsing the
    five-script chain into batch.py (ARCHITECTURE section 6.1) left ./transcribe
    with no link.py invocation to be missing a flag from. Silently keeping a
    resolved entry is how an allowlist becomes a place defects go to be
    forgotten.
    """
    stale = []
    for (caller, module, qualified), reason in sorted(KNOWN_DROPS.items()):
        flag = qualified.split(":")[-1]
        base = module.split("/")[-1]
        flags, _ = _argparse_spec(repo / module)
        subs = [s for s, d in flags.items() if flag in d]
        if not subs:
            stale.append("  %s no longer defines %s -- delete the KNOWN_DROPS "
                         "entry" % (module, flag))
            continue
        text = (repo / caller).read_text()
        if not re.search(r"[/'\"]%s\b" % re.escape(base), text):
            stale.append("  %s no longer runs %s at all -- the drop is resolved "
                         "by removal; delete the KNOWN_DROPS entry"
                         % (caller, module))
            continue
        mentions, forwarded = _reached(text, base)
        if flag in mentions or any(s in forwarded for s in subs):
            stale.append("  %s now passes %s to %s -- delete the KNOWN_DROPS "
                         "entry, the guard covers it from here"
                         % (caller, flag, module))
    assert not stale, "\n".join(stale)


def test_every_exception_carries_a_reason():
    """An allowlist without reasons is a list of flags nobody dares to remove.

    Both lists shrink the guard, so each entry has to be arguable on its own.
    """
    bare = ["  %r" % (k,) for k, v in
            list(ALLOWED.items()) + list(KNOWN_DROPS.items())
            if not v or len(v) < 30]
    assert not bare, ("listed with no usable reason -- say what went wrong, or "
                      "what would have to change:\n" + "\n".join(bare))
