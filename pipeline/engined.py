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


def code_stamp():
    """Newest mtime among the modules the daemon holds in memory.

    A daemon imports batch, transcribe_meeting and postproc ONCE, at startup, and
    keeps them for its lifetime. Editing those files afterwards changes nothing
    until it restarts -- the running engine goes on executing the code it loaded.
    That is the correct behaviour for a long-lived process and a genuinely
    confusing one to debug: a fix is deployed, the file on disk contains it, and
    the run behaves as though it does not. It cost a full round of "why did my
    change do nothing" here.

    So record what was loaded and let anyone ask. Not a reload -- swapping code
    under a running queue is worse than the confusion it would save."""
    newest = 0.0
    d = Path(__file__).resolve().parent
    for f in list(d.glob("*.py")) + list((d / "link").glob("*.py")):
        try:
            newest = max(newest, f.stat().st_mtime)
        except OSError:
            pass
    return newest


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

    # BOTH models, and the second one started FIRST. Transcribing needs the vLLM
    # engine; working out who spoke needs WeSpeaker, a separate ~5s load that
    # every single run used to pay. Holding only the engine left that half of the
    # problem exactly where it was -- and on a short recording it is most of what
    # remains once the engine is free.
    #
    # It costs nothing because it loads in its OWN process while this one spends
    # ~70s building the engine. Started before that call rather than after, so
    # the whole of it lands inside a window we are already paying for.
    #
    # Only when the card can hold both at once, which plan_gpu_split already
    # decides: on a 6 GiB card it cannot, and jobs there transcribe first and
    # embed afterwards with the engine released -- a resident embedder would sit
    # in exactly the memory that scheme frees up.
    emb = None
    if split.concurrent:
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": work}
        env.pop("CUDA_VISIBLE_DEVICES", None)   # vLLM rewrites this for its workers
        try:
            emb = batch.Embedder(split.embed_batch, env,
                                 Path(sock_path).parent / "embed.log")
        except Exception as e:
            print(f"!! could not start the speaker model ({type(e).__name__}: {e})"
                  f" — jobs will load their own", flush=True)

    llm = TM.build_engine(frac, window=window, overlap=overlap, releasable=True)

    if emb is not None and not emb.alive():
        print("!! the speaker model died while loading (see run/embed.log) — "
              "jobs will load their own", flush=True)
        emb = None
    res = batch.Resident(llm, frac, window, overlap, embedder=emb)
    print(f"engine resident after {time.time()-t0:.1f}s — "
          f"window {window}s overlap {overlap}s, gpu-frac {frac:.2f}"
          + (f", speaker model held (--batch {emb.batch})" if emb
             else ", speaker model NOT held"), flush=True)

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
    loaded_at, started = code_stamp(), time.time()

    def handle(conn):
        """Serve one connection. Returns True if the engine must shut down."""
        with conn, conn.makefile("rwb") as f:
            line = f.readline()
            if not line.strip():
                # A probe: `engine status` connects and closes without sending
                # anything, so that asking whether the engine is up does not
                # queue behind a running job. There is nothing to answer.
                return False
            try:
                req = json.loads(line)
            except ValueError:
                return False

            if req.get("cmd") == "stamp":
                send(f, {"stamp": loaded_at, "started": started})
                return False

            if req.get("cmd") == "release":
                # Someone else needs the card. Anything that cannot go through
                # this engine has to load its OWN, and a resident engine holding
                # 17.6 of 24 GiB does not leave room for one -- so a caller about
                # to fall back asks us to step aside first. Costs ~1s to wake
                # again on the next job, against the run it would otherwise
                # break.
                try:
                    if not res.asleep:
                        llm.sleep(level=1)
                        res.asleep = True
                    send(f, {"rc": 0})
                    print("released the card on request", flush=True)
                except Exception as e:
                    send(f, {"reject": f"could not release: {type(e).__name__}: {e}"})
                return False

            try:
                a = parser.parse_args(req.get("argv", []))
            except SystemExit:
                send(f, {"reject": "arguments this engine does not understand"})
                return False

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
                return False

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
                    # An engine that will not wake is no longer an engine. Say
                    # so and shut down; supervisor builds a clean one, and
                    # clients fall back by themselves meanwhile.
                    print(f"!! could not wake the engine ({type(e).__name__}: {e})"
                          f" — exiting so it gets rebuilt", flush=True)
                    return True
            return False

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

        # NOTHING a client does may kill the engine. It costs 70-400s to replace
        # and may have a queue behind it, so the bar for dying is far higher than
        # for any single connection failing.
        #
        # This is not hypothetical. `engine status` connects and closes without
        # reading, and the daemon used to answer a client that was already gone,
        # get BrokenPipeError out of send(), and let it propagate out of this
        # loop. Every status check killed the engine it was checking, about a
        # second after reporting it healthy -- and everything downstream looked
        # like the cause instead: engines "dying on ssh logout", stale sockets,
        # jobs falling back for no reason, two engines racing for the card.
        try:
            if handle(conn):
                p.unlink(missing_ok=True)
                return 1
        except Exception as e:
            print(f"!! a client connection failed ({type(e).__name__}: {e}) — "
                  f"still listening", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--release", action="store_true")
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
        # CONNECTING is the whole test, and nothing is sent or read. On a unix
        # socket a successful connect means a listener is bound; a file with
        # nothing behind it gives ECONNREFUSED above. An earlier version sent a
        # request and waited for the reply, which meant status HUNG -- and then,
        # on the 5s timeout, reported "not running" -- for the entire duration of
        # any job, because the daemon serves one at a time and does not read the
        # next connection until the current one is done. Status claiming the
        # engine is down while it is busy transcribing is worse than useless:
        # ./engine start reads it, and would have started a second engine.
        # Ask what code it is holding. A daemon imports batch, transcribe_meeting
        # and postproc once at startup and keeps them, so an edit since then is
        # on disk and NOT in the running engine -- which looks exactly like the
        # edit having no effect.
        stale = None
        try:
            f = s.makefile("rwb")
            send(f, {"cmd": "stamp"})
            m = json.loads(f.readline() or "{}")
            if m.get("stamp") and code_stamp() > m["stamp"] + 1:
                stale = code_stamp() - m["stamp"]
        except Exception:
            pass
        s.close()
        print(f"resident engine up at {sock_path}")
        if stale:
            print(f"!! it is running code from before your last edit "
                  f"({stale/60:.0f} min older than pipeline/ on disk).")
            print(f"   restart it to pick that up:  engine restart")
            return 2
        return 0

    if a.release:
        # Best effort by design. The caller is about to load its own engine and
        # only wants the card free first; no daemon, or a daemon that will not
        # answer, both mean there is nothing holding it.
        if not Path(sock_path).exists():
            return 0
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(60)        # sleep(level=1) is quick, but it is real work
        try:
            s.connect(sock_path)
            f = s.makefile("rwb")
            send(f, {"cmd": "release"})
            msg = json.loads(f.readline() or "{}")
        except (OSError, ValueError) as e:
            print(f"==> could not ask the resident engine to release the card ({e})",
                  file=sys.stderr)
            return 1
        if "reject" in msg:
            print(f"==> resident engine could not release the card: {msg['reject']}",
                  file=sys.stderr)
            return 1
        print("==> asked the resident engine to release the card", file=sys.stderr)
        return 0

    if a.submit:
        if rest and rest[0] == "--":
            rest = rest[1:]
        return submit(sock_path, rest)
    return serve(sock_path, a.window, a.overlap, a.gpu_frac) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
