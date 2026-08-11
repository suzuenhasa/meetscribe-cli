#!/usr/bin/env python3
"""Transcribe a queue of meetings with the engine loaded once.

  batch.py a.mp3 b.mp3 c.wav --out-dir /workspace/out [--glossary "..."] [--roster "..."]

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
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transcribe_meeting as TM
from moss_transcribe_diarize.inference_utils import load_audio_item

WORK = os.environ.get("MS_WORK", "/workspace")
PY = sys.executable


def safe(name):
    s = "".join(c if c.isalnum() or c in "._-" else "-" for c in name).strip("-")
    return s or "meeting"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out-dir", default=f"{WORK}/out")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=5.0)
    ap.add_argument("--glossary", default="")
    ap.add_argument("--roster", default="")
    ap.add_argument("--thr", default="auto")
    ap.add_argument("--gpu-frac", type=float, default=0.72,
                    help="vLLM's pool. The rest is headroom for the concurrent "
                         "embedder: vLLM pre-reserves its whole pool, so at 0.85 it "
                         "took 22 of 23.5 GiB and every embed subprocess OOM'd.")
    ap.add_argument("--no-overlap-embed", action="store_true",
                    help="wait for each embed instead of overlapping it")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in a.audio if Path(f).is_file()]
    if not files:
        raise SystemExit("no readable audio files")

    t_start = time.time()
    ptxt = TM.build_prompt(a.glossary)
    llm = TM.build_engine(a.gpu_frac)
    startup = time.time() - t_start
    print(f"engine resident after {startup:.1f}s — {len(files)} meetings queued\n", flush=True)

    pending = []
    t_queue = time.time()
    for f in files:
        name = safe(f.stem)
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

        # Hand embedding to a subprocess and move to the next transcription.
        # Never silence it: an early version sent stderr to DEVNULL and every
        # embed failed invisibly, leaving a "successful" run with no vectors.
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
        env.pop("CUDA_VISIBLE_DEVICES", None)     # vLLM rewrites this for its workers
        log = open(out / f"{name}_embed.log", "w")
        p = subprocess.Popen([PY, f"{WORK}/link/embed_batched.py", "--run", str(raw),
                              "--wav", str(f), "--out", str(out / f"{name}_emb.npz")],
                             stdout=log, stderr=subprocess.STDOUT, env=env)
        if a.no_overlap_embed:
            p.wait()
        pending.append((name, f, p))
        print(f"  {f.name[:44]:44} {dur/60:5.1f} min  transcribed {t_tr:5.1f}s  "
              f"{len(segs):4d} segs  coverage {cov:.0%}", flush=True)

    failed = []
    for name, f, p in pending:
        if p.wait() != 0 or not (out / f"{name}_emb.npz").exists():
            failed.append(name)
    t_gpu = time.time() - t_queue
    if failed:
        print(f"\n!! embedding FAILED for {', '.join(failed)} — see "
              f"{out}/<name>_embed.log", flush=True)

    print(flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MS_WORK": WORK}
    for name, f, _ in pending:
        if name in failed:
            continue
        subprocess.run([PY, f"{WORK}/link/link.py", "--run", str(out / f"{name}_raw.json"),
                        "--npz", str(out / f"{name}_emb.npz"), "--thr", a.thr,
                        "--out", str(out / f"{name}_linked.json")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        subprocess.run([PY, f"{WORK}/identify.py",
                        "--clusters", str(out / f"{name}_linked_clusters.npz"),
                        "--meeting", name, "--roster", a.roster,
                        "--names", str(out / f"{name}_names.json")], env=env)
        subprocess.run([PY, f"{WORK}/mktxt.py", str(out / f"{name}_linked.json"),
                        str(out / f"{name}_raw.json"), str(out / f"{name}.txt"),
                        f.stem, str(out / f"{name}_names.json")], env=env)
        print(flush=True)

    total = time.time() - t_start
    mins = sum(json.load(open(out / f"{n}_raw.json"))["duration_s"]
               for n, _, _ in pending if n not in failed) / 60
    print(f"{len(pending) - len(failed)} meetings, {mins:.0f} min of audio")
    print(f"  startup          {startup:6.1f}s  (once, not per meeting)")
    print(f"  transcribe+embed {t_gpu:6.1f}s"
          + ("  [sequential]" if a.no_overlap_embed else "  [overlapped]"))
    print(f"  total            {total:6.1f}s  ->  {mins*60/max(t_gpu,0.01):.0f}x realtime "
          f"excluding startup")


if __name__ == "__main__":
    main()
