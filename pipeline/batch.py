#!/usr/bin/env python3
"""Transcribe a queue of meetings with the engine loaded once.

  batch.py a.mp3 b.mp3 c.wav --out-dir out/ [--glossary "..."] [--roster "..."]

Two reasons this exists rather than calling transcribe_meeting.py per file:

  the engine costs ~66s to load and that is paid ONCE here, not per meeting. On
  ten short meetings that alone is the difference between 11 minutes and 1.

  embedding meeting N runs concurrently with meeting N+1's transcription. The
  two workloads bottleneck on different things -- vLLM decoding is memory
  bandwidth and launch overhead, the speaker ResNet is compute -- so the overlap
  was measured to cost vLLM about 2% (4.8s -> 4.9s GPU) and is very nearly free.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transcribe_meeting as TM
from moss_transcribe_diarize.inference_utils import load_audio_item

def _default_work():
    """The checkout this file lives in. Everything -- venv, weights, profile
    store, work directories -- stays inside it, so nothing is written outside
    and two checkouts never share state. MS_WORK overrides."""
    import os
    if os.environ.get("MS_WORK"):
        return os.environ["MS_WORK"]
    here = os.path.dirname(os.path.abspath(__file__))
    # pipeline/x.py -> repo root; pipeline/link/x.py -> repo root
    while os.path.basename(here) in ("link", "pipeline"):
        here = os.path.dirname(here)
    return here


WORK = _default_work()
PIPE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def safe(name):
    s = "".join(c if c.isalnum() or c in "._-" else "-" for c in name).strip("-")
    return s or "meeting"


class Embedder:
    """One resident embedding worker for the whole queue.

    This used to be a subprocess per recording, spawned right after each
    transcription and reaped only after the loop finished -- so the number of
    embedders in flight was the number of files. That holds while the files are
    hour-long meetings, because each embedder finishes long before the next
    transcription does. It does not hold otherwise: with short recordings and
    sub-second transcriptions they arrive faster than they can initialise, and a
    122-file run died after 16 transcripts and 15 embeddings, taking the
    inference engine down with it. Each one carries its own CUDA context, so the
    card fills with processes that have barely started.

    A single worker fixes that structurally rather than by picking a limit: the
    queue can be any length and the GPU still holds exactly one embedder. It also
    stops reloading WeSpeaker per file -- ~5s of torch import, checkpoint read
    and CUDA init against ~2s of actual work, which is the dominant cost once the
    recordings are short.

    Overlap is preserved, which was the point of the subprocess: submitting is a
    pipe write, so meeting N embeds while N+1 transcribes. Jobs beyond that queue
    in the pipe rather than spawning anything.
    """

    def __init__(self, batch, env, log_path):
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [PY, f"{PIPE}/link/embed_batched.py", "--serve", "--batch", str(batch)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env=env, text=True, bufsize=1)
        self.acks = {}
        self.eof = False
        self.cond = threading.Condition()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        """Collect acks off the worker's stdout.

        On its own thread so a slow embedder never blocks transcription, and so
        the pipe cannot fill and deadlock the worker mid-write.
        """
        for line in self.proc.stdout:
            try:
                ack = json.loads(line)
            except ValueError:
                continue                      # not an ack; the log has the detail
            with self.cond:
                self.acks[ack["out"]] = ack
                self.cond.notify_all()
        with self.cond:
            self.eof = True                   # worker exited, expect no more acks
            self.cond.notify_all()

    def submit(self, run, wav, out):
        """Queue one recording. Returns False if the worker has already died."""
        try:
            self.proc.stdin.write(json.dumps(
                {"run": str(run), "wav": str(wav), "out": str(out)}) + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, ValueError):
            return False

    def wait_for(self, out):
        """Block until `out` is embedded. Returns False if it failed or the
        worker died first -- never waits forever on a dead worker."""
        out = str(out)
        with self.cond:
            while out not in self.acks and not self.eof:
                self.cond.wait()
            return bool(self.acks.get(out, {}).get("ok"))

    def close(self):
        """Finish the queue and shut the worker down. -> {out_path: ack}"""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait()
        self.reader.join(timeout=30)
        self.log.close()
        return dict(self.acks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out-dir", default=f"{WORK}/out")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=5.0)
    ap.add_argument("--glossary", default="")
    ap.add_argument("--roster", default="")
    ap.add_argument("--titles", default="",
                    help="JSON {safe_name: original title}. Filenames are "
                         "sanitised for the trip over ssh; this restores what "
                         "the human actually called the meeting.")
    ap.add_argument("--thr", default="auto")
    ap.add_argument("--gpu-frac", type=float, default=None,
                    help="vLLM's share of the card; the rest is headroom for the "
                         "concurrent embedder. Derived from the card's size by "
                         "default -- the engine's cost is a constant, so no single "
                         "fraction is right for both a 12 and a 24 GiB card. Pass "
                         "a value only to override that.")
    ap.add_argument("--no-overlap-embed", action="store_true",
                    help="wait for each embed instead of overlapping it")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in a.audio if Path(f).is_file()]
    if not files:
        raise SystemExit("no readable audio files")
    titles = {}
    if a.titles and Path(a.titles).is_file():
        titles = json.load(open(a.titles))

    t_start = time.time()
    ptxt = TM.build_prompt(a.glossary)
    auto_frac, embed_batch = TM.plan_gpu_split()
    gpu_frac = a.gpu_frac if a.gpu_frac is not None else auto_frac
    if a.gpu_frac is None:
        print(f"gpu split: vLLM {gpu_frac:.2f}, embedder --batch {embed_batch}",
              flush=True)
    llm = TM.build_engine(gpu_frac)
    startup = time.time() - t_start
    print(f"engine resident after {startup:.1f}s — {len(files)} meetings queued\n", flush=True)

    env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
    env.pop("CUDA_VISIBLE_DEVICES", None)     # vLLM rewrites this for its workers
    emb = Embedder(embed_batch, env, out / "_embed.log")

    pending, unreadable = [], []
    t_queue = time.time()
    for f in files:
        name = safe(f.stem)
        try:
            wav = load_audio_item(str(f), sampling_rate=TM.SR)
            dur = len(wav) / TM.SR

            t0 = time.time()
            reqs, offsets, cores = TM.plan_windows(wav, ptxt, a.window, a.overlap)
            mt = int((a.window + 2 * a.overlap) * 20)
            outs = llm.generate(reqs, TM.SamplingParams(temperature=0.0, max_tokens=mt))
            segs, cov, _ = TM.assemble(outs, offsets, cores, wav, dur)
            raw = out / f"{name}_raw.json"
            json.dump({"audio": str(f), "duration_s": round(dur, 2), "window_s": a.window,
                       "n_windows": len(reqs), "coverage": round(cov, 4),
                       "segments": segs}, open(raw, "w"))
            t_tr = time.time() - t0
        except Exception as e:
            # One unreadable recording must not cost the whole queue. A truncated
            # mp3 used to raise straight out of the loop, taking the engine with
            # it and losing every meeting behind it -- the worst possible failure
            # for a batch, since the engine load is the expensive part and it is
            # already paid. Report it and keep going.
            unreadable.append(name)
            print(f"  {f.name[:44]:44} SKIPPED — {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:80]}", flush=True)
            continue

        # Queue the embedding and move straight to the next transcription. Never
        # silence the worker: an early version sent stderr to DEVNULL and every
        # embed failed invisibly, leaving a "successful" run with no vectors.
        npz = out / f"{name}_emb.npz"
        if not emb.submit(raw, f, npz):
            print(f"\n!! embedding worker died — see {out}/_embed.log", flush=True)
        if a.no_overlap_embed:
            emb.wait_for(npz)
        pending.append((name, f))
        print(f"  {f.name[:44]:44} {dur/60:5.1f} min  transcribed {t_tr:5.1f}s  "
              f"{len(segs):4d} segs  coverage {cov:.0%}", flush=True)

    acks = emb.close()
    failed = [name for name, f in pending
              if not acks.get(str(out / f"{name}_emb.npz"), {}).get("ok")
              or not (out / f"{name}_emb.npz").exists()]
    t_gpu = time.time() - t_queue
    if failed:
        print(f"\n!! embedding FAILED for {', '.join(failed)} — see "
              f"{out}/_embed.log", flush=True)
    if unreadable:
        print(f"\n!! SKIPPED {len(unreadable)} unreadable file(s): "
              f"{', '.join(unreadable)}", flush=True)

    print(flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
    for name, f in pending:
        if name in failed:
            continue
        # Inherit stdout/stderr rather than discarding them. link.py is where
        # CLUSTER, LOW-SEPARATION and FLOOR-VIOLATION are reported, and the
        # troubleshooting docs tell people to read exactly those -- while the
        # docs also recommend folder mode, which is this path. Discarding them
        # here also hid link.py crashing outright.
        subprocess.run([PY, f"{PIPE}/link/link.py", "--run", str(out / f"{name}_raw.json"),
                        "--npz", str(out / f"{name}_emb.npz"), "--thr", a.thr,
                        "--out", str(out / f"{name}_linked.json")], env=env)
        subprocess.run([PY, f"{PIPE}/identify.py",
                        "--clusters", str(out / f"{name}_linked_clusters.npz"),
                        "--meeting", name, "--roster", a.roster,
                        "--names", str(out / f"{name}_names.json")], env=env)
        subprocess.run([PY, f"{PIPE}/mktxt.py", str(out / f"{name}_linked.json"),
                        str(out / f"{name}_raw.json"), str(out / f"{name}.txt"),
                        titles.get(name, f.stem),
                        str(out / f"{name}_names.json")], env=env)
        print(flush=True)

    total = time.time() - t_start
    mins = sum(json.load(open(out / f"{n}_raw.json"))["duration_s"]
               for n, _ in pending if n not in failed) / 60
    print(f"{len(pending) - len(failed)} meetings, {mins:.0f} min of audio")
    print(f"  startup          {startup:6.1f}s  (once, not per meeting)")
    print(f"  transcribe+embed {t_gpu:6.1f}s"
          + ("  [sequential]" if a.no_overlap_embed else "  [overlapped]"))
    print(f"  total            {total:6.1f}s  ->  {mins*60/max(t_gpu,0.01):.0f}x realtime "
          f"excluding startup")


if __name__ == "__main__":
    main()
