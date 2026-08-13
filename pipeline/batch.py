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
        # append, not truncate: a second worker may run after the first to
        # retry what did not fit, and the first attempt's diagnostics are
        # exactly what explains why.
        self.log = open(log_path, "a")
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
    # plan_gpu_split is a TARGET, not a promise: it only has to be good enough to
    # boot and to leave a plausible share behind. Built releasable regardless, so
    # the fallback below is available whatever the measurement turns out to be.
    split = TM.plan_gpu_split()
    gpu_frac = a.gpu_frac if a.gpu_frac is not None else split.frac
    # window/overlap are not optional here: they set the audio-length hint the
    # engine reserves against. See build_engine's limit_mm_per_prompt comment.
    llm = TM.build_engine(gpu_frac, window=a.window, overlap=a.overlap,
                          releasable=True)

    # Now ASK the card rather than trusting the estimate. gpu_memory_utilization
    # bounds only what vLLM's profiler measured, and it allocates on top of that
    # at run time, so what is left is never the predicted figure. Choosing here
    # makes an optimistic target cost throughput instead of a failed run, and
    # surfaces it a second after startup rather than after transcribing the lot.
    #
    # When it does not fit, transcribe everything first and embed afterwards with
    # the engine RELEASED in between. Separating them in time alone achieves
    # nothing -- vLLM holds its pool for the life of the process, so a merely
    # sequential embedder finds the same memory it would have found running
    # concurrently: on a 6 GiB card, 45 MiB when it needed 66. llm.sleep() hands
    # back 3.54 of 3.90 GiB there, which is the whole problem solved.
    free = TM.free_vram_gib()
    embed_batch, concurrent = TM.choose_embed_strategy(free)
    deferred = not concurrent
    print(f"gpu split: vLLM {gpu_frac:.2f} target, {free:.1f} GiB free after load"
          f" -> embedder --batch {embed_batch}"
          + ("" if concurrent else ", run after the engine is released"),
          flush=True)
    startup = time.time() - t_start
    print(f"engine resident after {startup:.1f}s — {len(files)} meetings queued\n", flush=True)

    env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
    env.pop("CUDA_VISIBLE_DEVICES", None)     # vLLM rewrites this for its workers
    emb = None if deferred else Embedder(embed_batch, env, out / "_embed.log")

    pending, unreadable, empty, queued = [], [], [], []
    sampling = TM.SamplingParams(temperature=0.0,
                                 max_tokens=int((a.window + 2 * a.overlap) * 20))

    # Pool windows ACROSS recordings before decoding. One generate() per file
    # makes the batch as big as that file happens to be, which is fine for an
    # hour-long meeting -- 149 windows fills the card -- and terrible for short
    # ones: a 20 s clip is a single window, so the GPU decodes a batch of one and
    # idles between Python round trips. In an external evaluation the same audio
    # ran at 13x as 122 short files and 145x concatenated into one, and this is
    # most of that gap. Windows are independent, so there is nothing to preserve
    # by keeping them apart.
    #
    # The target is max_num_seqs, which is also roughly what the KV cache holds
    # at this request length (a 5090 reports 192,320 tokens against ~800 per
    # window). vLLM queues internally beyond that, so overshooting is harmless;
    # the reason to bound it at all is the second limit below.
    target = TM._max_num_seqs()
    # Audio for a file has to stay resident until every one of its windows has
    # come back, so pooling holds wavs. Flush on this as well or a long queue
    # would accumulate them without limit.
    max_pooled_samples = int(4 * 3600 * TM.SR)

    inflight = {}          # name -> per-file state awaiting its windows
    pool = []              # (name, window_index, request)
    pooled_samples = 0
    t_queue = time.time()

    def finish(name):
        """Assemble one recording once all of its windows have come back."""
        st = inflight.pop(name)
        f, wav, dur = st["f"], st["wav"], st["dur"]
        segs, cov, _ = TM.assemble(st["outs"], st["offsets"], st["cores"], wav, dur)
        raw = out / f"{name}_raw.json"
        json.dump({"audio": str(f), "duration_s": round(dur, 2), "window_s": a.window,
                   "n_windows": len(st["outs"]), "coverage": round(cov, 4),
                   "segments": segs}, open(raw, "w"))
        if not segs:
            # MOSS returned nothing at all for this recording. Ten of forty
            # accented-parliament clips did exactly this in one evaluation and
            # were reported as successes carrying empty transcripts. There is
            # nothing to embed or cluster, so say so and move on.
            empty.append(name)
            print(f"  {f.name[:44]:44} {dur/60:5.1f} min  NO SPEECH FOUND — "
                  f"empty transcript", flush=True)
            return
        # Queue the embedding and move straight on. Never silence the worker: an
        # early version sent stderr to DEVNULL and every embed failed invisibly,
        # leaving a "successful" run with no vectors.
        npz = out / f"{name}_emb.npz"
        if deferred:
            queued.append((raw, f, npz))
        else:
            if not emb.submit(raw, f, npz):
                print(f"\n!! embedding worker died — see {out}/_embed.log", flush=True)
            if a.no_overlap_embed:
                emb.wait_for(npz)
        pending.append((name, f))
        print(f"  {f.name[:44]:44} {dur/60:5.1f} min  {len(st['outs']):4d} win  "
              f"{len(segs):4d} segs  coverage {cov:.0%}", flush=True)

    def flush():
        """Decode everything pooled, then assemble whichever files are complete."""
        nonlocal pooled_samples
        if not pool:
            return
        outs = llm.generate([r for _, _, r in pool], sampling)
        for (name, wi, _), o in zip(pool, outs):
            inflight[name]["outs"][wi] = o
        pool.clear()
        for name in [n for n, st in inflight.items()
                     if all(o is not None for o in st["outs"])]:
            pooled_samples -= len(inflight[name]["wav"])
            finish(name)

    for f in files:
        name = safe(f.stem)
        try:
            wav = load_audio_item(str(f), sampling_rate=TM.SR)
            reqs, offsets, cores = TM.plan_windows(wav, ptxt, a.window, a.overlap)
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

        inflight[name] = {"f": f, "wav": wav, "dur": len(wav) / TM.SR,
                          "offsets": offsets, "cores": cores,
                          "outs": [None] * len(reqs)}
        pool.extend((name, i, r) for i, r in enumerate(reqs))
        pooled_samples += len(wav)
        # `if`, not `while`: flush() decodes the entire pool, so one pass always
        # empties it. A loop could spin forever if pooled_samples stayed above
        # the bound with nothing left to decode.
        if len(pool) >= target or pooled_samples >= max_pooled_samples:
            flush()
    flush()

    def release_engine():
        """Hand the card back. level=1 offloads the weights and drops the KV
        cache; level=2 on top of it raises "CUDA Error: invalid argument" in the
        cumem allocator, so it is one call, once, and the engine is not used
        again afterwards. Measured on a 6 GiB card: 3.54 of 3.90 GiB returned."""
        try:
            llm.sleep(level=1)
            return True
        except Exception as e:
            print(f"!! could not release the engine ({type(e).__name__}: {e})",
                  flush=True)
            return False

    def embed_all(jobs, why):
        """Run a fresh worker over `jobs`. -> acks"""
        print(f"\n{why}", flush=True)
        worker = Embedder(embed_batch, env, out / "_embed.log")
        for raw, f, npz in jobs:
            if not worker.submit(raw, f, npz):
                print(f"!! embedding worker died — see {out}/_embed.log", flush=True)
        return worker.close()

    if deferred:
        release_engine()
        acks = embed_all(queued, "releasing the engine before embedding")
    else:
        acks = emb.close()

    def missing(names_and_files):
        return [(n, f) for n, f in names_and_files
                if not acks.get(str(out / f"{n}_emb.npz"), {}).get("ok")
                or not (out / f"{n}_emb.npz").exists()]

    # RECOVERY. Neither predicting the split nor measuring free VRAM after load
    # is sufficient: the engine's footprint grows once requests actually flow, so
    # room that existed at startup can be gone by the time the embedder runs.
    # Rather than guess more precisely, react -- the transcripts are already on
    # disk and only the vectors are missing, so release the engine and try again
    # with the whole card. This is what makes the estimate advisory: being wrong
    # costs one retry instead of the run.
    lost = missing(pending)
    if lost and not deferred:
        again = [(out / f"{n}_raw.json", f, out / f"{n}_emb.npz") for n, f in lost]
        if release_engine():
            acks.update(embed_all(
                again, f"embedding did not fit alongside the engine for "
                       f"{len(lost)} meeting(s) — releasing it and retrying"))

    failed = [n for n, _ in missing(pending)]
    t_gpu = time.time() - t_queue
    if failed:
        print(f"\n!! embedding FAILED for {', '.join(failed)} — see "
              f"{out}/_embed.log", flush=True)
    if unreadable:
        print(f"\n!! SKIPPED {len(unreadable)} unreadable file(s): "
              f"{', '.join(unreadable)}", flush=True)

    print(flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
    broken, broken_names = [], set()
    for name, f in pending:
        if name in failed:
            continue
        # Inherit stdout/stderr rather than discarding them. link.py is where
        # CLUSTER, LOW-SEPARATION and FLOOR-VIOLATION are reported, and the
        # troubleshooting docs tell people to read exactly those -- while the
        # docs also recommend folder mode, which is this path. Discarding them
        # here also hid link.py crashing outright.
        #
        # And CHECK the return codes. These ran unchecked, so when link.py died
        # on one recording the run still printed its meeting count and exited 0,
        # with that transcript simply absent -- 122 embeddings and 121 texts,
        # which nothing reported. A missing output is a failure, not a quiet
        # difference in the file count.
        steps = [
            ("link", [PY, f"{PIPE}/link/link.py", "--run", str(out / f"{name}_raw.json"),
                      "--npz", str(out / f"{name}_emb.npz"), "--thr", a.thr,
                      "--out", str(out / f"{name}_linked.json")]),
            ("identify", [PY, f"{PIPE}/identify.py",
                          "--clusters", str(out / f"{name}_linked_clusters.npz"),
                          "--meeting", name, "--roster", a.roster,
                          "--names", str(out / f"{name}_names.json")]),
            ("render", [PY, f"{PIPE}/mktxt.py", str(out / f"{name}_linked.json"),
                        str(out / f"{name}_raw.json"), str(out / f"{name}.txt"),
                        titles.get(name, f.stem),
                        str(out / f"{name}_names.json")]),
        ]
        # link and render are load-bearing. identify only decorates the result
        # with names it recognises, and mktxt is explicitly written to run
        # without it -- so aborting the chain when identify fails destroys a
        # transcript that would have rendered fine as "Speaker N". That is the
        # same missing-output failure this loop exists to catch, caused by the
        # catching.
        bad = None
        for step, argv in steps:
            rc = subprocess.run(argv, env=env).returncode
            if rc == 0:
                continue
            if step == "identify":
                print(f"  !! identify failed ({rc}) for {name} — speakers stay "
                      f"numbered in this transcript", flush=True)
                continue
            bad = f"{name} ({step} exited {rc})"
            break
        if bad is None:
            # Every step claimed success; confirm it actually left the artifacts
            # behind, since a step can exit 0 and still write nothing.
            missing = [p.name for p in (out / f"{name}_linked.json", out / f"{name}.txt")
                       if not p.exists() or p.stat().st_size == 0]
            if missing:
                bad = f"{name} (missing {', '.join(missing)})"
        if bad:
            broken.append(bad)
            broken_names.add(name)
        print(flush=True)

    total = time.time() - t_start
    # Exclude post-processing failures too, not just embedding ones, or the
    # headline still counts meetings whose transcript does not exist -- the
    # exact "122 embeddings, 121 texts" miscount this set out to remove.
    done = [n for n, _ in pending if n not in failed and n not in broken_names]
    mins = sum(json.load(open(out / f"{n}_raw.json"))["duration_s"]
               for n in done) / 60
    print(f"{len(done)} meetings, {mins:.0f} min of audio")
    print(f"  startup          {startup:6.1f}s  (once, not per meeting)")
    print(f"  transcribe+embed {t_gpu:6.1f}s"
          + ("  [sequential]" if a.no_overlap_embed else "  [overlapped]"))
    print(f"  total            {total:6.1f}s  ->  {mins*60/max(t_gpu,0.01):.0f}x realtime "
          f"excluding startup")

    # Exit nonzero if ANY recording did not come out whole. A batch that reports
    # success while a transcript is missing is worse than one that fails loudly:
    # the caller has no reason to look.
    if broken:
        print(f"\n!! POST-PROCESSING FAILED for {len(broken)}: {'; '.join(broken)}",
              flush=True)
    if empty:
        print(f"\n!! NO SPEECH FOUND in {len(empty)}: {', '.join(empty)}\n"
              f"   These produced an empty transcript. Check the audio is speech "
              f"and is not silent or corrupt.", flush=True)
    n_bad = len(broken) + len(failed) + len(unreadable) + len(empty)
    if n_bad:
        print(f"\n{n_bad} of {len(files)} file(s) did not complete.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
