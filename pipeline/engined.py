#!/usr/bin/env python3
"""Hold the engine between runs, so a job does not pay to load it.

  engined.py --serve                     the daemon; supervisor runs this
  engined.py --submit -- <batch.py args> the client; ./transcribe runs this
  engined.py --status                    is one up, and what will it serve?

Loading the engine costs ~70s with a warm compile cache and 240-400s without,
and it is paid per PROCESS. batch.py already amortises it across a queue, which
is the whole reason it exists -- but the cost lands again on the next queue, and
in full on a single file. Transcribing one voice memo is 70s of loading and 3s
of work.

So: load it once at boot and keep it. The daemon owns the engine and runs jobs
against it; the client is a stdlib-only pipe that forwards argv and streams the
output back, so it starts instantly and never imports vLLM.

NONE OF THIS IS REQUIRED. ./setup.sh does not install a daemon, does not start
one, and does not know this file exists; only ./engine start and the vast
provisioning script ever run one. A normal install has no daemon and behaves
exactly as it did before this file was written.

FALLING BACK IS THEREFORE THE DEFAULT, not an error path. The client exits
NO_DAEMON when there is no daemon, when the socket is stale, when the daemon
died mid-job, or when its engine was built for a different window/overlap -- and
./transcribe then runs batch.py directly, paying the load as it always did. The
first of those is silent, because on most installs it is simply the truth and
not news; the others each print what happened, because each means something was
there and went wrong.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

NO_DAEMON = 97          # "I could not use the daemon" -- distinct from any rc
                        # batch.py returns, so the caller can tell the two apart
                        # and only retry locally for this one.
IDLE_SLEEP_S = float(os.environ.get("MS_ENGINE_IDLE_SLEEP", "900"))


def default_sock():
    work = os.environ.get("MS_WORK") or str(Path(__file__).resolve().parent.parent)
    return str(Path(work) / "run" / "engine.sock")


# --------------------------------------------------------------- the wire
# Newline-delimited JSON both ways. The client sends one request frame; the
# daemon streams {"out": ...} frames as the job prints and ends with exactly one
# {"rc": n} or {"reject": why}.
def send(f, obj):
    f.write((json.dumps(obj) + "\n").encode())
    f.flush()


class Stream:
    """A file-like that forwards everything printed to the client.

    Deliberately swallows write errors. If the client hangs up mid-job -- ^C,
    a dropped ssh -- the job is most of the way through work that is expensive
    and already paid for, and its outputs go to disk regardless of who is
    listening. Letting a broken pipe raise here would throw that away."""

    def __init__(self, f):
        self.f, self.live = f, True

    def write(self, s):
        if s and self.live:
            try:
                send(self.f, {"out": s})
            except OSError:
                self.live = False
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


# --------------------------------------------------------------- the client
def submit(sock_path, argv):
    if not Path(sock_path).exists():
        # SILENTLY. No socket is the normal state of an install that never runs
        # a daemon, which is most of them -- ./setup.sh does not create one and
        # nothing but ./engine start and the vast provisioning script ever will.
        # Saying "no resident engine" on every run would be a message about a
        # feature the user is not using, in the default path, forever. The cases
        # below are different: each one means something was there and went
        # wrong, which is worth a line.
        return NO_DAEMON
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Only the CONNECT is bounded. Once the daemon has the job, the read
        # side must block indefinitely -- a queue of meetings legitimately takes
        # hours, and a read timeout would abandon a running job and then start a
        # second engine alongside it.
        s.settimeout(10)
        s.connect(sock_path)
        s.settimeout(None)
    except ConnectionRefusedError:
        # Nothing is bound to it. On a unix socket that is unambiguous -- a
        # daemon that is still starting has not created the file yet, because
        # serve() unlinks before it binds -- so this file is litter from one
        # that died. Remove it, or the "engine died" line below would greet
        # every run from now on, on a box whose owner may have stopped it on
        # purpose. Say it once, on the run that cleans up, and never again.
        try:
            Path(sock_path).unlink()
            print("==> cleared the socket of an engine that is no longer running",
                  file=sys.stderr)
        except OSError:
            print(f"==> stale engine socket at {sock_path} that I cannot remove; "
                  f"loading an engine instead", file=sys.stderr)
        return NO_DAEMON
    except OSError as e:
        print(f"==> resident engine not answering ({e}); loading one instead",
              file=sys.stderr)
        return NO_DAEMON
    f = s.makefile("rwb")
    send(f, {"argv": list(argv), "cwd": os.getcwd(),
             "work": os.environ.get("MS_WORK", "")})
    for line in f:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "out" in msg:
            sys.stdout.write(msg["out"])
            sys.stdout.flush()
        elif "reject" in msg:
            print(f"==> resident engine cannot take this job: {msg['reject']}",
                  file=sys.stderr)
            return NO_DAEMON
        elif "rc" in msg:
            return int(msg["rc"])
    # The daemon died holding the job. Its output is already on the user's
    # screen, but nothing said whether the job finished, so treat it as a
    # fallback rather than reporting a success we cannot vouch for.
    print("==> resident engine went away mid-job; running it here instead",
          file=sys.stderr)
    return NO_DAEMON


# --------------------------------------------------------------- the daemon
def reap_orphans():
    """Kill any VLLM::EngineCore left behind by a dead parent.

    vLLM runs the model in a SEPARATE process. When the parent dies without
    stopping it -- SIGKILL, the OOM killer, a lost ssh session, supervisor
    restarting a wedged daemon -- that child survives and keeps the whole KV
    cache: 17.6 GiB of a 24 GiB card, measured here. Nothing can ever reach it
    again, because the sockets that addressed it died with its parent. It is
    purely lost memory.

    The symptom is bad out of proportion to the cause: the next start finds too
    little VRAM and dies inside the allocator with a stack trace about engine
    initialisation that names no cause at all, on a card nvidia-smi shows as
    two-thirds full with no obvious owner.

    An EngineCore whose parent is pid 1 has been orphaned, by definition and
    with no other explanation, so it is safe to take. One that is still doing
    work has a live parent and is not touched."""
    try:
        ps = subprocess.run(["ps", "-eo", "pid,ppid,args"],
                            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return
    reaped = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "1":
            continue
        if not parts[2].startswith("VLLM::EngineCore"):
            continue
        try:
            os.kill(int(parts[0]), signal.SIGKILL)
            reaped.append(parts[0])
        except (OSError, ValueError):
            pass
    if reaped:
        print(f"reaped orphaned VLLM::EngineCore {', '.join(reaped)} — a previous "
              f"engine died without releasing the card", flush=True)
        time.sleep(3)       # let the driver actually hand the memory back


def serve(sock_path, window, overlap, gpu_frac):
    reap_orphans()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import contextlib
    import transcribe_meeting as TM
    import batch

    work = os.environ.get("MS_WORK", "")
    t0 = time.time()
    split = TM.plan_gpu_split()
    frac = gpu_frac if gpu_frac is not None else split.frac
    llm = TM.build_engine(frac, window=window, overlap=overlap, releasable=True)
    res = batch.Resident(llm, frac, window, overlap)
    print(f"engine resident after {time.time()-t0:.1f}s — "
          f"window {window}s overlap {overlap}s, gpu-frac {frac:.2f}", flush=True)

    p = Path(sock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A socket file outliving its daemon is the normal case after a crash, and
    # bind() fails on an existing path whether or not anything is behind it.
    p.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)      # the library is the user's meetings
    srv.listen(16)                  # jobs queue here; see the accept loop below
    print(f"listening on {sock_path}", flush=True)

    # Take the socket file with us when told to stop. supervisorctl stop sends
    # SIGTERM, whose default action is to die without running anything -- which
    # left the file behind, so every later run found a socket with nothing
    # listening and reported a dead engine on a box whose owner had stopped it
    # deliberately. os._exit rather than a clean unwind: there is no state here
    # worth flushing, and stopping mid-job is what stopping means.
    def _bye(signum, _frame):
        p.unlink(missing_ok=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    parser = batch.build_parser()
    while True:
        # The timeout is what makes idle sleep possible, nothing else. One job at
        # a time, on purpose: two batches sharing one engine would each get half
        # the card and finish in more than twice the time, and the second would
        # race the first for the embedder's headroom.
        srv.settimeout(IDLE_SLEEP_S if IDLE_SLEEP_S > 0 and not res.asleep else None)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            # Nothing for a while: hand the VRAM back. It costs ~1s to wake
            # against ~70s to load, so this is nearly free, and it means a
            # resident engine does not lock other work off the card all day.
            try:
                llm.sleep(level=1)
                res.asleep = True
                print(f"idle {IDLE_SLEEP_S:.0f}s — released the card, "
                      f"still resident", flush=True)
            except Exception as e:
                print(f"!! could not release while idle ({type(e).__name__}: {e})",
                      flush=True)
                srv.settimeout(None)
            continue

        with conn, conn.makefile("rwb") as f:
            try:
                req = json.loads(f.readline() or "{}")
            except ValueError:
                continue
            try:
                a = parser.parse_args(req.get("argv", []))
            except SystemExit:
                send(f, {"reject": "arguments this engine does not understand"})
                continue

            # Three ways a resident engine is the wrong engine for a job. Each
            # sends the client to its own engine, which is slow and correct.
            why = None
            if req.get("work") and work and req["work"] != work:
                # speakers.db is per-install and is the one file that cannot be
                # rebuilt from the audio. Serving a job for a different install
                # would write this box's voice profiles into that job's meetings.
                why = (f"it belongs to the install at {work}, and this job is "
                       f"for {req['work']}")
            elif not res.serves(a):
                why = (f"it was built for window {res.window}s overlap "
                       f"{res.overlap}s, and this job wants {a.window}/{a.overlap}")
            elif a.gpu_frac is not None and abs(a.gpu_frac - res.gpu_frac) > 1e-6:
                why = (f"it holds gpu-frac {res.gpu_frac:.2f} and this job asked "
                       f"for {a.gpu_frac:.2f}")
            if why:
                send(f, {"reject": why})
                continue

            if req.get("cwd") and Path(req["cwd"]).is_dir():
                os.chdir(req["cwd"])        # safe: one job at a time
            out = Stream(f)
            print(f"job: {len(a.audio)} file(s) -> {a.out_dir}", flush=True)
            t = time.time()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    rc = batch.run_job(a, resident=res) or 0
            except Exception as e:
                import traceback
                send(f, {"out": traceback.format_exc()})
                rc = 1
                print(f"!! job raised {type(e).__name__}: {e}", flush=True)
            send(f, {"rc": rc})
            print(f"job done in {time.time()-t:.1f}s, rc {rc}", flush=True)

            if res.asleep:
                # run_job released the engine to fit the embedder on a small
                # card. Wake it now rather than on the next job, so the cost
                # lands in idle time instead of in someone's wall clock.
                try:
                    res.wake()
                except Exception as e:
                    # An engine that will not wake is no longer an engine. Exit
                    # and let supervisor build a clean one; clients fall back by
                    # themselves in the meantime.
                    print(f"!! could not wake the engine ({type(e).__name__}: {e})"
                          f" — exiting so it gets rebuilt", flush=True)
                    p.unlink(missing_ok=True)
                    return 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--sock", default=None)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=5.0)
    ap.add_argument("--gpu-frac", type=float, default=None)
    a, rest = ap.parse_known_args()
    sock_path = a.sock or default_sock()

    if a.status:
        if not Path(sock_path).exists():
            print(f"no resident engine ({sock_path} does not exist)")
            return 1
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5)
        try:
            s.connect(sock_path)
        except OSError as e:
            print(f"stale socket at {sock_path}: {e}")
            return 1
        # An empty request is rejected by design; the rejection proves it is
        # alive and answering, which is the whole question being asked.
        f = s.makefile("rwb")
        send(f, {"argv": [], "cwd": os.getcwd(), "work": os.environ.get("MS_WORK", "")})
        f.readline()
        print(f"resident engine up at {sock_path}")
        return 0

    if a.submit:
        if rest and rest[0] == "--":
            rest = rest[1:]
        return submit(sock_path, rest)
    return serve(sock_path, a.window, a.overlap, a.gpu_frac) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
