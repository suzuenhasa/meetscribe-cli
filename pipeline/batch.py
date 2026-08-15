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
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transcribe_meeting as TM
import clips as CLIPS
import library as LIB
import postproc
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
        self.batch = batch
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

    def alive(self):
        return self.proc.poll() is None

    def drain(self, outs):
        """Wait for exactly these outputs. -> {out_path: ack}

        close() below ends the worker, which is right when it was built for one
        run and wrong when it is held across many by the daemon -- shutting it
        down there would reload WeSpeaker on the next job, which is the entire
        thing being avoided. Same waiting, no shutdown."""
        got = {}
        with self.cond:
            for o in [str(x) for x in outs]:
                while o not in self.acks and not self.eof:
                    self.cond.wait()
                if o in self.acks:
                    got[o] = self.acks[o]
        return got

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


# Below this many recordings, post-processing runs in the parent instead of in a
# pool. Set by what a worker costs to start (see run_all): a pool has to save
# more than its own startup, and per-file work here is milliseconds.
POST_POOL_MIN = int(os.environ.get("MS_POST_POOL_MIN", "4"))


def post_workers():
    """How many post-processing workers to run. Scales with the box.

    These tasks are ~10-30ms of work each once imports are paid, so a handful of
    workers saturates them; the cap keeps a 256-core host from spawning 256
    interpreters to do a few seconds of work.
    """
    return max(1, min(os.cpu_count() or 2, 8))


def to_16k_mono(files, scratch):
    """Decode everything that is not already 16 kHz mono, in parallel.

    -> ([Path to feed the model], {that path: the original it came from})

    The mapping matters: the converted wav is a decode CACHE, 8x the size of the
    source and reproducible from it in seconds, so it is what the model reads and
    the original is what the library keeps. Losing track of which was which
    parked a 73 MB wav beside the transcript and left the mp3 where it was.

    The model wants 16 kHz mono, so anything else is decoded and resampled
    before it sees it. That work has to happen; the question is where. Doing it
    in the file loop below means one file at a time with the GPU idle -- measured
    on a 5090 host, 6.7s for a 74-minute mp3 and 20s for a 100-minute opus,
    single-threaded, because a codec bitstream is a dependency chain and does not
    thread within one stream.

    Across FILES it is embarrassingly parallel, so do it here, all at once,
    before the engine is asked for anything. Same 379 minutes of audio on the
    same box: 115.2s as mp3, 60.1s as wav, for 6s of parallel ffmpeg.

    Files already 16 kHz mono are passed through untouched. Anything ffmpeg
    cannot read is passed through as well and left for the loader to fail on
    properly, with the filename in the message.
    """
    def already_ok(f):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=sample_rate,channels,codec_name", "-of", "csv=p=0", str(f)],
                capture_output=True, text=True, timeout=30).stdout.strip()
            codec, rate, ch = (out.split(",") + ["", "", ""])[:3]
            return codec == "pcm_s16le" and rate == "16000" and ch == "1"
        except Exception:
            return False

    todo = [f for f in files if not already_ok(f)]
    if not todo:
        return list(files), {}
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"converting {len(todo)} file(s) to 16 kHz mono, {min(8, len(todo))} at a time",
          flush=True)
    t0 = time.time()

    def one(f):
        dst = scratch / f"{f.stem}.wav"
        n = 1
        while dst.exists():
            dst = scratch / f"{f.stem}-{n}.wav"
            n += 1
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(f),
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
                           capture_output=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 44:
            return f, dst
        print(f"  !! could not convert {f.name[:50]} — using it as it is", flush=True)
        return f, f

    swap = {}
    with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
        for src, dst in pool.map(one, todo):
            swap[src] = dst
    print(f"  converted in {time.time()-t0:.1f}s", flush=True)
    out = [swap.get(f, f) for f in files]
    return out, {swap[k]: k for k in swap if swap[k] != k}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out-dir", default=f"{WORK}/out",
                    help="scratch for intermediates")
    ap.add_argument("--library", default=None,
                    help="where finished meetings are kept, one directory each: "
                         "library/<slug>-<id>/")
    ap.add_argument("--no-clips", action="store_true",
                    help="do not cut per-speaker clips. They are what lets you "
                         "name a voice after the source audio is gone.")
    ap.add_argument("--move-audio", action="store_true",
                    help="move the source into the meeting directory rather "
                         "than copying it. What the inbox does, since a worklist "
                         "should empty as work completes.")
    ap.add_argument("--replace", default=None,
                    help="overwrite this meeting id in place rather than making "
                         "a new one, so everything decided about it survives")
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
    ap.add_argument("--no-convert", action="store_true",
                    help="do not pre-decode to 16 kHz mono. The decode still "
                         "happens, just one file at a time inside the run with "
                         "the GPU waiting -- measured at roughly half the "
                         "throughput on a 5090.")
    return ap


class Resident:
    """An engine kept alive between jobs by engined.py.

    Loading the engine costs ~70s even with a warm compile cache, and that is
    paid per PROCESS. Batching a queue amortises it; a single file cannot, so a
    20 s voice memo spends 70 s starting and 3 s working. Holding the engine in
    a daemon moves that cost to boot, once, for every job the box ever runs.

    Two things make a resident engine different from a fresh one:

    the window/overlap it was built with are BAKED IN -- they set the audio
    length hint the engine reserves against (see build_engine), and a job asking
    for longer windows than the hint would be silently truncated. `serves` is
    that check; the client falls back to its own engine when it fails, which is
    correct but slow, so the daemon is built with whatever the box actually uses.

    and it can be asleep. On a card too small to hold engine and embedder at
    once, run_job releases the engine mid-job and never wakes it -- fine when
    the process was about to exit, wrong when another job is coming. `wake` is
    paired with that release and costs about a second, against the ~70s of a
    real load.
    """

    def __init__(self, llm, gpu_frac, window, overlap, embedder=None):
        self.llm, self.gpu_frac = llm, gpu_frac
        self.window, self.overlap = window, overlap
        self.asleep = False
        # The SECOND model. Holding the engine and not this one only solves half
        # the problem: WeSpeaker is a separate ~5s load that every run paid, and
        # on a short recording that is a large share of what is left once the
        # engine is free. Held for the daemon's lifetime, not the job's.
        self.embedder = embedder

    def embedder_if_alive(self):
        if self.embedder is not None and self.embedder.alive():
            return self.embedder
        return None

    def serves(self, a):
        return a.window == self.window and a.overlap == self.overlap

    def wake(self):
        if not self.asleep:
            return
        self.llm.wake_up()
        self.asleep = False


def run_job(a, resident=None):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in a.audio if Path(f).is_file()]
    if not files:
        raise SystemExit("no readable audio files")
    titles = {}
    if a.titles and Path(a.titles).is_file():
        titles = json.load(open(a.titles))

    # Before anything touches the GPU. Names are keyed off the stem, which the
    # conversion preserves, so nothing downstream can tell the difference except
    # by being faster.
    scratch, origin = None, {}
    if not a.no_convert:
        scratch = out / "_wav"
        files, origin = to_16k_mono(files, scratch)

    t_start = time.time()
    ptxt = TM.build_prompt(a.glossary)
    if resident is not None:
        # Already loaded, and possibly asleep from a previous job on a small card.
        resident.wake()
        llm, gpu_frac = resident.llm, resident.gpu_frac
    else:
        # plan_gpu_split is a TARGET, not a promise: it only has to be good enough
        # to boot and to leave a plausible share behind. Built releasable
        # regardless, so the fallback below is available whatever the measurement
        # turns out to be.
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
    held = resident.embedder_if_alive() if resident is not None else None
    if held is not None:
        # Nothing to decide. The embedder is already loaded and already holding
        # its share of the card, so measuring free VRAM here would see the memory
        # it is using and conclude it does not fit -- deferring an embedder that
        # is sitting right there, ready.
        embed_batch, deferred = held.batch, False
        print(f"gpu split: vLLM {gpu_frac:.2f}, embedder --batch {embed_batch} "
              f"(already loaded)", flush=True)
    else:
        free = TM.free_vram_gib()
        embed_batch, concurrent = TM.choose_embed_strategy(free)
        deferred = not concurrent
        print(f"gpu split: vLLM {gpu_frac:.2f} target, {free:.1f} GiB free after load"
              f" -> embedder --batch {embed_batch}"
              + ("" if concurrent else ", run after the engine is released"),
              flush=True)
    startup = time.time() - t_start
    print(f"engine resident after {startup:.1f}s"
          + (" (already loaded)" if resident is not None else "")
          + f" — {len(files)} meetings queued\n", flush=True)

    env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
    env.pop("CUDA_VISIBLE_DEVICES", None)     # vLLM rewrites this for its workers
    # Reuse the daemon's worker when there is one; otherwise this run owns its
    # own, exactly as it did before daemons existed.
    emb = None if deferred else (held or Embedder(embed_batch, env, out / "_embed.log"))

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

    libdir = Path(a.library) if a.library else LIB.library_dir(WORK)
    meetings = {}          # folder name -> Meeting
    inflight = {}          # name -> per-file state awaiting its windows
    pool = []              # (name, window_index, request)
    pooled_samples = 0
    t_queue = time.time()

    def finish(name):
        """Assemble one recording once all of its windows have come back."""
        st = inflight.pop(name)
        f, wav, dur = st["f"], st["wav"], st["dur"]
        segs, cov, _ = TM.assemble(st["outs"], st["offsets"], st["cores"], wav, dur)
        raw = meetings[name].file("raw", "json")
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
        npz = meetings[name].file("embeddings", "npz")
        if deferred:
            queued.append((raw, f, npz))
        else:
            if not emb.submit(raw, f, npz):
                print(f"\n!! embedding worker died — see {out}/_embed.log", flush=True)
            if a.no_overlap_embed:
                emb.wait_for(npz)
        meetings[name].write(duration_s=round(dur, 2), n_segments=len(segs),
                             coverage=round(cov, 4), window_s=a.window,
                             overlap_s=a.overlap, transcribed_at=time.time())
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
        # A directory per meeting, made before anything is decoded, so a meeting
        # has an identity from the moment it exists rather than acquiring one if
        # it happens to succeed.
        #
        # This is also what ends the name-collision class of bug structurally
        # instead of by suffixing: the folder is unique by construction, and the
        # files inside only have to be unique WITHIN it. "A Guide.wav" and
        # "A-Guide.wav" get two directories and cannot reach each other.
        if a.replace:
            m = LIB.find(a.replace, lib=libdir)
            if m is None:
                raise SystemExit(f"no meeting {a.replace!r} in {libdir}")
        else:
            src = origin.get(f, f)
            m = LIB.create(titles.get(safe(src.stem), src.stem), src.name, lib=libdir)
        name = m.path.name
        meetings[name] = m
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
        again afterwards. Measured on a 6 GiB card: 3.54 of 3.90 GiB returned.

        "Not used again" holds within a job. A resident engine gets another job
        after this one, so record that it is asleep -- run_job wakes it at the
        top rather than leaving the next caller to find an engine with no
        weights in it."""
        try:
            llm.sleep(level=1)
            if resident is not None:
                resident.asleep = True
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
    elif emb is held:
        # Wait for this run's work, and leave the worker up for the next one.
        acks = emb.drain(meetings[n].file("embeddings", "npz") for n, _ in pending)
    else:
        acks = emb.close()

    def missing(names_and_files):
        return [(n, f) for n, f in names_and_files
                if not acks.get(str(meetings[n].file("embeddings", "npz")), {}).get("ok")
                or not meetings[n].file("embeddings", "npz").exists()]

    # RECOVERY. Neither predicting the split nor measuring free VRAM after load
    # is sufficient: the engine's footprint grows once requests actually flow, so
    # room that existed at startup can be gone by the time the embedder runs.
    # Rather than guess more precisely, react -- the transcripts are already on
    # disk and only the vectors are missing, so release the engine and try again
    # with the whole card. This is what makes the estimate advisory: being wrong
    # costs one retry instead of the run.
    lost = missing(pending)
    if lost and not deferred:
        again = [(meetings[n].file("raw", "json"), f,
                  meetings[n].file("embeddings", "npz")) for n, f in lost]
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
    todo = [(name, f) for name, f in pending if name not in failed]

    # Post-processing runs IN-PROCESS in a worker pool, in three phases, rather
    # than as three subprocesses per recording. These scripts cost far more to
    # start than to run, so 120 recordings meant 360 interpreter launches for a
    # few seconds of actual work -- once the GPU work was pooled, this became the
    # bottleneck. A worker imports numpy, scipy and sqlite3 once and then handles
    # many files.
    #
    # Phased rather than per-file end to end, because the steps have a dependency
    # chain (link writes the npz identify reads, identify writes the names mktxt
    # reads) and because identify WRITES to speakers.db on every run -- a
    # decisions row per cluster, not only when enrolling. Running it serially in
    # the parent keeps sqlite single-writer. It can afford to be serial: its
    # 0.116s was 0.103s of import, so the work itself is milliseconds.
    #
    # spawn, not fork: this process holds a CUDA context and forking one is
    # unsafe. spawn also gives each worker the clean import we want anyway.
    def argv_link(name):
        m = meetings[name]
        return [f"{PIPE}/link/link.py", "--run", str(m.file("raw", "json")),
                "--npz", str(m.file("embeddings", "npz")), "--thr", a.thr,
                "--out", str(m.file("transcript", "json")),
                "--clusters-out", str(m.file("clusters", "npz"))]

    def argv_identify(name):
        m = meetings[name]
        # --meeting is the ID, not the filename. That is what lets a folder be
        # renamed, or a meeting re-run with --replace, without orphaning every
        # decision ever recorded about it.
        return [f"{PIPE}/identify.py",
                "--clusters", str(m.file("clusters", "npz")),
                "--meeting", m.id, "--roster", a.roster,
                "--names", str(m.file("names", "json"))]

    def argv_render(name, f):
        m = meetings[name]
        return [f"{PIPE}/mktxt.py", str(m.file("transcript", "json")),
                str(m.file("raw", "json")), str(m.file("transcript", "txt")),
                m.title, str(m.file("names", "json"))]

    os.environ["MS_WORK"] = WORK
    rc_link, rc_render = {}, {}
    if todo:
        # A pool is worth having for a queue and is pure loss for a handful of
        # files. Starting one means `spawn`, and spawn re-runs THIS module's top
        # level in every worker -- which imports transcribe_meeting, which
        # imports vLLM and torch. 8.7s per worker against 0.5s for the numpy and
        # scipy these scripts actually use, and two pools per run. On one
        # 3-minute recording that was 17.4s of a 23s job, spent loading an
        # inference stack to cluster a few hundred vectors.
        #
        # Below the threshold, run them right here: the parent has numpy loaded
        # already and each script is milliseconds of real work, so there is
        # nothing left to parallelise away. Above it, per-file work dominates
        # again and the pool earns its startup back.
        postproc.pool_init(PIPE)
        parallel = len(todo) > POST_POOL_MIN

        def run_all(specs):
            if not parallel:
                return {k: (rc, o) for k, rc, o in map(postproc.run_module, specs)}
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(min(post_workers(), len(specs)),
                          initializer=postproc.pool_init, initargs=(PIPE,)) as pool:
                return {k: (rc, o) for k, rc, o in
                        pool.imap_unordered(postproc.run_module, specs)}

        rc_link = run_all([(n, f"{PIPE}/link/link.py", argv_link(n)) for n, _ in todo])
        # identify in the parent: single sqlite writer, and cheap once imported
        rc_ident = {}
        for name, _ in todo:
            if rc_link.get(name, (1, ""))[0] == 0:
                rc_ident[name] = postproc.run_module((name, f"{PIPE}/identify.py", argv_identify(name)))[1:]
        ready = [(n, f) for n, f in todo if rc_link.get(n, (1, ""))[0] == 0]
        rc_render = run_all([(n, f"{PIPE}/mktxt.py", argv_render(n, f)) for n, f in ready])
    else:
        rc_ident = {}

    # Replay each recording's diagnostics together and in queue order -- link.py
    # is where CLUSTER, LOW-SEPARATION and FLOOR-VIOLATION are reported, and the
    # troubleshooting docs tell people to read exactly those.
    for name, f in todo:
        for step, table in (("link", rc_link), ("identify", rc_ident),
                            ("render", rc_render)):
            rc, output = table.get(name, (0, ""))
            if output.strip():
                print(output.rstrip(), flush=True)
        # link and render are load-bearing. identify only decorates the result
        # with names it recognises, and mktxt is explicitly written to run
        # without it -- so treating an identify failure as fatal would destroy a
        # transcript that would have rendered fine as "Speaker N".
        bad = None
        for step, table in (("link", rc_link), ("render", rc_render)):
            rc = table.get(name, (1, ""))[0]
            if rc != 0:
                bad = f"{name} ({step} exited {rc})"
                break
        if rc_ident.get(name, (0, ""))[0] != 0:
            print(f"  !! identify failed for {name} — speakers stay numbered",
                  flush=True)
        if bad is None:
            # Every step claimed success; confirm it actually left the artifacts
            # behind, since a step can exit 0 and still write nothing.
            gone = [p.name for p in (meetings[name].file("transcript", "json"),
                                     meetings[name].file("transcript", "txt"))
                    if not p.exists() or p.stat().st_size == 0]
            if gone:
                bad = f"{name} (missing {', '.join(gone)})"
        if bad:
            broken.append(bad)
            broken_names.add(name)
        print(flush=True)

    # The source audio belongs with everything derived from it. Moved when it
    # came from the inbox -- a worklist empties as work completes -- and copied
    # when it came from a path the caller nominated, because emptying someone's
    # directory unasked is not ours to do.
    for name, f in pending:
        if name in broken_names:
            continue
        m = meetings[name]
        if m.audio() is not None:
            continue
        src = origin.get(f, f)          # the original, never the decode cache
        dest = m.path / f"{m.stem}-audio{src.suffix.lower()}"
        try:
            if a.move_audio:
                shutil.move(str(src), dest)
            else:
                shutil.copy2(src, dest)
        except OSError as e:
            print(f"  !! could not put {f.name[:40]} beside its transcript "
                  f"({type(e).__name__}) — it is still at {f}", flush=True)

    # A few seconds of each speaker, so the library is still usable for naming
    # people after the source audio is archived or deleted. Cheap, and the only
    # thing that makes "keep the transcripts, drop the audio" a real option.
    if not a.no_clips:
        n_clips = 0
        for name, _ in pending:
            if name in broken_names:
                continue
            m = meetings[name]
            aud = m.audio()
            if aud:
                n_clips += CLIPS.cut(aud, m.file("transcript", "json"), m.clips_dir)
        if n_clips:
            print(f"cut {n_clips} clips for naming voices", flush=True)

    if scratch is not None and scratch.is_dir():
        # These are a decode cache, not an artifact -- 8x the size of the source
        # and reproducible from it. Leaving them behind fills the disk on a box
        # doing a library.
        shutil.rmtree(scratch, ignore_errors=True)

    total = time.time() - t_start
    # Exclude post-processing failures too, not just embedding ones, or the
    # headline still counts meetings whose transcript does not exist -- the
    # exact "122 embeddings, 121 texts" miscount this set out to remove.
    done = [n for n, _ in pending if n not in failed and n not in broken_names]
    mins = sum(json.load(open(meetings[n].file("raw", "json")))["duration_s"]
               for n in done) / 60
    # Say which number is which. "3 meetings, 32 min of audio" followed by a
    # figure in seconds reads as though the run took 32 minutes -- the audio's
    # length and the time it took are the two quantities here and they were not
    # labelled apart.
    n = len(done)
    print(f"{n} meeting{'' if n == 1 else 's'} transcribed")
    print(f"  audio in         {mins:6.0f} min")
    print(f"  startup          {startup:6.1f}s   (once, not per meeting)")
    print(f"  transcribe+embed {t_gpu:6.1f}s"
          + ("   [sequential]" if a.no_overlap_embed else "   [overlapped]"))
    print(f"  time taken       {total:6.1f}s   {mins*60/max(t_gpu,0.01):.0f}x faster than "
          f"real time, excluding startup")

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


def main(argv=None):
    return run_job(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main() or 0)
