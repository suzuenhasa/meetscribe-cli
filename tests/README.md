# tests

    cd /path/to/meetscribe-cli && python3 -m pytest tests/ -q

## No GPU, no venv, no network

Nothing here loads a model, allocates a card, or ssh's anywhere. It runs on a
plain CPU box with `pytest` and `numpy` and nothing else.

The two modules that cannot be imported here -- `pipeline/transcribe_meeting.py`
and `pipeline/batch.py`, which `import vllm` at module level -- are still tested
against **the real shipped file**, never a copy:

* `transcribe_meeting.py` is exec'd from its real path with `vllm`,
  `transformers` and `moss_transcribe_diarize` stubbed into `sys.modules` for
  the length of the import (see the `tm` fixture in `conftest.py`).
  `test_assemble.py` asserts `tm.__file__` is the production path -- if that
  loader ever picks up a copy, every test in the file is theatre.
* `batch.py` is read as text and parsed with `ast` where a claim can be made
  statically (does every caller still unpack four values from `assemble()`).
* `cluster_speakers.py` and `library.py` import only numpy and stdlib, so they
  are imported for real.
* `./transcribe` and `./engine` are run as **real subprocesses** against a
  stub interpreter (`MS_PY`) in a temp tree, so their `set -e` behaviour is
  bash's own. Most of the shell defects pinned here are `set -e` interactions
  and do not exist unless bash is actually bash. The stub's exit code is per
  module (`rc_for={"engined.py": 97}`), because "the daemon refused and the
  fallback ran it" is two calls with two codes and one code for the whole run
  cannot say it.

Nothing touches the checkout's own `speakers.db`, `library/`, or `inbox/`:
every subprocess gets `MS_SPEAKER_DB` and `MS_WORK` redirected into `tmp_path`,
and every `MS_*` variable inherited from your shell is stripped first.

## The convention: a test names the commit whose bug it pins

Every test here exists because the pipeline actually shipped that defect. Each
file, or each section inside it, names the commit that fixed it -- e.g.
`bc42e1c` (the repetition guard deleted the line *before* the loop), `72b7319`
(speech both windows thought the other one owned), `6506bd8` (cannot-link
honoured while building the tree but not after the cut). Read the commit before
changing the test: the assertion is usually narrower or wider than it looks for
a reason the commit message explains.

When you add a test, name the commit it pins in the same way.

## xfail means a defect that is still there

A strict `xfail` here is **not** a disabled test. It is a defect that survives
at HEAD, with a `reason=` stating what production code would have to change for
it to pass. Because they are strict, each one flips to a *failure* the moment
the defect is fixed -- which is the signal to delete the marker, not the test.

Run `python3 -m pytest tests/ -q -rx` to print them.

Do not "fix" production code from inside this directory to turn one green.

## test_wrapper_flags.py: the census, and its two lists

Everything else here tests the modules, which is exactly where a dropped flag
*works*. `test_wrapper_flags.py` reads argparse out of `batch.py`,
`speakers.py`, `relabel.py` and `link/link.py` with `ast`, reads `./transcribe`
and `./speakers` as text, and asserts the two sides reconcile -- plus the same
edge one level down, where `batch.py` builds `link.py`'s argv and where
`--speaker-db` was lost.

It carries two lists, and which one a flag belongs in is the whole judgement:

* `ALLOWED` -- deliberate. The wrapper *should* not carry this flag, and the
  entry says why (a bench switch, a comparison arm that exists to lose, an
  intermediate path the caller chooses).
* `KNOWN_DROPS` -- a defect still on the floor, strict in the same sense as the
  xfails above. Forward the flag and the entry becomes a failure telling you to
  delete the line.

A census that parses to nothing passes silently, so
`test_the_parsers_still_find_both_sides` names flags and subcommands that must
still be found on each side. It used to pin a *count* -- 25 flags out of
batch.py, 20 case arms out of ./transcribe -- and deleting five dead tuning
knobs moved three of those numbers at once, failing with "the ast walk has lost
the parser" while the parser was fine. A floor you re-set to whatever the last
deletion left is not a floor. If one of the named flags is deliberately deleted,
delete it here too and say so; if all of them vanish at once, fix the parser.
