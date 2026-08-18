"""Run the after-the-GPU steps: link, render.

This module exists to be CHEAP TO IMPORT, and that is its whole reason for
being here rather than in batch.py.

batch.py runs these in a multiprocessing pool, and `spawn` -- which is required,
since the parent holds a CUDA context and forking one is unsafe -- starts each
worker as a fresh interpreter that must import the module holding its target
function. When that function lived in batch.py, every worker imported batch.py,
which imports transcribe_meeting, which imports vLLM and torch. Measured on a
3090: 8.7s to import batch against 0.5s for the numpy and scipy these scripts
actually use, paid twice per job for the two pools. On a 3-minute recording that
was 17.4s of a 23s run -- more than the transcription, the embedding and the
engine startup put together, spent loading an inference stack to run a script
that clusters some vectors.

Nothing here may import torch, vLLM or transcribe_meeting, directly or
otherwise. The scripts it runs import what they need themselves.
"""
import contextlib
import io
import runpy
import sys
import traceback


def pool_init(pipe_dir):
    """Put the pipeline on the path so workers can import its modules."""
    for p in (pipe_dir, f"{pipe_dir}/link"):
        if p not in sys.path:
            sys.path.insert(0, p)


def run_module(spec):
    """Run one pipeline script in-process. -> (key, returncode, captured output)

    spec is (key, script_path, argv). Called either in a pool worker or directly
    in the parent, and it pays each import once and then handles many recordings
    -- the point of the exercise, since these scripts cost far more to start than
    to run. Measured per recording: link.py 0.13s of which ~0.10 is importing
    numpy, mktxt.py 0.036s against 0.034s for a bare interpreter. Launching them
    per file was almost entirely launch.

    runpy rather than importing and calling main(), because mktxt.py has no
    main() -- it is top-level script code reading sys.argv. Re-executing a module
    body is cheap and its imports still hit sys.modules from the previous call,
    which is where the saving actually comes from.

    stdout is captured and returned rather than printed, so a pool cannot
    interleave two recordings' diagnostics; the caller replays them in order.
    Exceptions become a nonzero code, which is what the subprocess version gave
    and what the artifact checking downstream expects.
    """
    key, path, argv = spec
    buf = io.StringIO()
    old = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            runpy.run_path(path, run_name="__main__")
        return key, 0, buf.getvalue()
    except SystemExit as e:
        return key, int(e.code or 0), buf.getvalue()
    except Exception:
        return key, 1, buf.getvalue() + traceback.format_exc()
    finally:
        sys.argv = old
