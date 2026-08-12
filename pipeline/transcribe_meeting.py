"""Full-meeting windowed transcription via vLLM, with timestamps rebased to meeting time.

  python3 transcribe_meeting.py meeting.mp3 --out out.json
"""
import argparse, json, time
from collections import namedtuple

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, load_audio_item

import os

# MS_MODEL lets a fine-tuned checkpoint be swapped in without touching the pipeline.
MODEL = os.environ.get("MS_MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
SR = 16000
SILENCE_GATE_DB = -70.0


def build_prompt(glossary=""):
    """The chat-template prompt, with an optional proper-noun glossary appended.

    Names the model has never seen come out as the nearest familiar English --
    "I'm Sreeram" was transcribed "I'm sure I'm a, I'm". Post-hoc correction
    cannot fix that safely because the output is valid English, so the
    vocabulary has to reach the decoder.
    """
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    prompt = DEFAULT_PROMPT
    terms = [t.strip() for t in (glossary or "").split(",") if t.strip()]
    if terms:
        prompt += ("\n专有名词表（音频中若出现这些名称，请使用以下拼写）/ "
                   "Proper nouns that may occur; use exactly these spellings: "
                   + ", ".join(terms) + ".")
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": "x"},
                                         {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def engine_dtype():
    """bfloat16 needs compute capability 8.0 (Ampere). Turing and older -- a 2070,
    a 1080 Ti -- top out at 7.5, and vLLM refuses to start rather than falling
    back, so pick float16 there. Accuracy is unaffected at this model size; only
    the numeric range differs, and nothing here needs bf16's exponent."""
    try:
        import torch
        major, _ = torch.cuda.get_device_capability()
        return "bfloat16" if major >= 8 else "float16"
    except Exception:
        return "float16"


# The engine's fixed cost. It is a CONSTANT, not a share of the card, which is
# the whole reason one --gpu-frac cannot serve two card sizes:
#
#     available_kv = gpu_frac * total_gib - ENGINE_OVERHEAD_GIB
#
# vLLM reports the split at DEBUG, hidden by env.sh's VLLM_LOGGING_LEVEL=WARNING:
#
#     weights                              1.72 GiB   real, resident
#     CUDA context + persistent allocator  0.32 GiB   real, resident
#     transient activation headroom        6.26 GiB   reserved against a request
#                                                     nothing here can send
#
# That third line was 76% of the cost and was pure profiling artifact -- see
# build_engine's limit_mm_per_prompt comment for the mechanism and the fix.
# Declaring the real audio length drops the total from 8.30 to 2.23-2.50 GiB at
# no cost in throughput.
#
# It also explains three knobs that looked like they should have moved the old
# figure and did not. All three are the same dummy request: max_num_seqs 256->16
# only ever freed the sampler's logits buffer (256 x 151936 x 4 B = 0.145 GiB),
# max_model_len 8192->4096 made it WORSE by pushing the profiler from one encoder
# item to two (360 x 1500 x 4096 x 2 B = 4.12 GiB, exactly the allocation it died
# on), and enforce_eager freed nothing because graphs are captured after the
# profiling run, out of the slack rather than out of KV.
ENGINE_OVERHEAD_GIB = 2.6     # 1.72 weights + 0.32 context + margin over 2.50
KV_MIN_GIB = 0.9              # vLLM will not start without one max_model_len seq

# Headroom vLLM does NOT know it needs. gpu_memory_utilization bounds what the
# PROFILER measured, and vLLM then sizes KV to fill whatever the profile left
# over -- so removing the phantom 6.26 GiB did not just free memory, it handed
# that memory to KV and left the real forward passes nowhere to run. Measured:
# the engine overran its own budget by 2.64 GiB on a 3090 and 1.07 on a 2060,
# and the embedder was OOM-killed in the 294 MiB that remained. The phantom had
# been acting as an accidental runtime buffer. This is the deliberate version.
def _runtime_headroom_gib(total_gib):
    """Slack for activation vLLM does not reserve. Scales with the card.

    gpu_memory_utilization bounds what the PROFILER measured, and the profiler
    builds a dummy request with ONE audio item while the scheduler runs up to
    max_num_seqs of them -- so once the 90-minute phantom is gone, concurrent
    encoder activation is structurally under-reserved. A bigger card holds more
    KV, runs more windows at once, and overruns by more, so a flat constant is
    wrong at both ends.

    Measured overruns past budget: 1.07 GiB on a 6 GiB 2060, 2.64 on a 24 GiB
    3090, 3.2 on a 32 GiB 5090 fed 16 kHz wav (which, having no mp3 decode to
    throttle it, keeps the engine fuller than mp3 does). That is close enough to
    linear in card size to size it that way, with a floor that keeps small cards
    startable and a ceiling so a very large card does not hand over half of
    itself.
    """
    # The floor is 1.6, not something smaller: a 6 GiB 2060 overran its budget by
    # 1.07 GiB, and 1.6 is the value measured to hold there. An earlier version of
    # this function used a 0.8 floor and fixed a 5090 by starving that 2060 --
    # every meeting failed with vLLM itself out of memory before the embedder ever
    # ran.
    return min(max(0.13 * total_gib, 1.6), 4.5)

# Concurrent sequences, which is also concurrent AUDIO items -- one per prompt --
# so this bounds how many encoder forwards run at once, and therefore how much of
# the headroom above can be consumed at peak.
#
# Measured on a 3090, six meetings, only this varying: 16 -> 128x, 64 -> 160x,
# 256 -> 173x, 512 -> 219x*, 1024 -> 218x*. Flat past 256, and most of the gain
# is in by 64 -- the card saturates on arithmetic long before it runs out of KV.
# (*the 512/1024 pair was run on 16 kHz wav, hence the different absolute level;
# what matters is that they do not beat 256.)
#
# So 256 costs nothing on a card with room, and capping it lower is a real 35%
# loss -- which is exactly the mistake this constant existed to make. It is only
# lowered on cards too small to absorb the runtime activation of a wide batch.
def _max_num_seqs(total_gib=None):
    if os.environ.get("MS_MAX_SEQS"):
        return int(os.environ["MS_MAX_SEQS"])
    if total_gib is None:
        try:
            import torch
            total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        except Exception:
            return 64
    return 256 if total_gib >= 10.0 else 64

# WeSpeaker's peak. It runs in a SEPARATE process, so it comes out of what vLLM
# does not reserve. Measured by pinning vLLM's share and running the embedder
# against the remainder: at 1.49 GiB free --batch 32 OOMs and --batch 8 survives;
# at 0.91 GiB free both OOM. On an empty card it takes 4.2 GiB, but that is the
# caching allocator growing into free space rather than a requirement.
EMBED_RESERVE_GIB = {32: 4.3, 8: 1.6}
EMBED_SEQUENTIAL_GIB = 0.3    # not resident alongside the engine, so only slack

# A namedtuple rather than a bare tuple: this grew a third field, and the last
# time a tuple here changed width two call sites kept unpacking the old one and
# only failed at runtime.
GpuSplit = namedtuple("GpuSplit", "frac embed_batch concurrent")


def plan_gpu_split(total_gib=None):
    """Divide the card between vLLM and the embedder. -> GpuSplit

    Works in absolute GiB and converts to a fraction only at the end, because
    every quantity involved is absolute. Tries three tiers in order of speed:
    a large concurrent embedder batch, a small one, and finally embedding
    sequentially -- which gives up the overlap but reclaims 1.3 GiB, and is the
    difference between a 4 GiB card refusing to start and running slowly.

    Exits naming the actual numbers when even that does not fit. vLLM's own error
    arrives after the weights have loaded and blames gpu_memory_utilization,
    which sends you tuning the one thing that cannot help.
    """
    if total_gib is None:
        import torch
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30

    for batch, concurrent in ((32, True), (8, True), (8, False)):
        reserve = EMBED_RESERVE_GIB[batch] if concurrent else EMBED_SEQUENTIAL_GIB
        reserve += _runtime_headroom_gib(total_gib)
        if total_gib - reserve - ENGINE_OVERHEAD_GIB >= KV_MIN_GIB:
            return GpuSplit(round((total_gib - reserve) / total_gib, 3),
                            batch, concurrent)

    floor = (ENGINE_OVERHEAD_GIB + KV_MIN_GIB + EMBED_SEQUENTIAL_GIB
             + _runtime_headroom_gib(total_gib))
    raise SystemExit(
        f"this GPU has {total_gib:.1f} GiB; the pipeline needs {floor:.1f} GiB:\n"
        f"  {ENGINE_OVERHEAD_GIB:.1f}  model weights, CUDA context and activation\n"
        f"  {KV_MIN_GIB:.1f}  KV cache for a single window\n"
        f"  {EMBED_SEQUENTIAL_GIB:.1f}  the speaker embedder, run sequentially\n"
        f"  {_runtime_headroom_gib(total_gib):.1f}  activation headroom the engine does not reserve\n"
        f"Weights are the bulk of the first figure and cannot be reduced from "
        f"here.")


def free_vram_gib():
    """What is ACTUALLY free right now, rather than what we predicted."""
    import torch
    return torch.cuda.mem_get_info()[0] / 2**30


def choose_embed_strategy(free_gib):
    """Decide how to embed from a MEASUREMENT. -> (embed_batch, concurrent)

    Call this once the engine is up. Everything that made the prediction
    unreliable has resolved by then: real weights, real KV, real graphs, this
    card, this vLLM build.

    This exists because gpu_memory_utilization is not a cap. vLLM sizes KV from a
    profiling estimate and then allocates on top of it at run time, so the memory
    it actually occupies is never the number it was given -- measured overruns of
    1.07 GiB on a 2060, 2.64 on a 3090 and 3.2 on a 5090 fed 16 kHz wav. Every
    attempt to predict that gap with a constant broke on the card it was not
    derived from: 0.72 left a 12 GiB card with no KV at all, a flat 1.6 GiB of
    headroom failed on a 5090, and scaling it starved the 2060 it was supposed to
    protect. There is also a feedback loop -- more headroom means less KV, which
    means fewer concurrent windows, which means less activation to reserve
    against, so the quantity depends on the reservation.

    Asking the driver ends the argument. plan_gpu_split's answer is now a target
    that only has to be good enough to boot; if it was optimistic the cost is
    throughput, never a failed run, and it is known within a second of startup
    rather than after twenty minutes of transcription.
    """
    for batch in (32, 8):
        if free_gib >= EMBED_RESERVE_GIB[batch]:
            return batch, True
    return 8, False


def build_engine(gpu_frac=0.90, max_len=None, eager=None, window=30.0,
                 overlap=5.0, releasable=False):
    """Load vLLM. ~66s, so a batch loads it once and keeps it resident.

    On a card that cannot hold ENGINE_OVERHEAD_GIB the failure is opaque: vLLM
    names gpu_memory_utilization as the cause, after the weights have already
    loaded. plan_gpu_split() sizes gpu_frac so that does not happen, and explains
    the numbers first-hand.
    """
    # These used to shrink themselves on a small card, which measured WORSE, not
    # better -- see the ENGINE_OVERHEAD_GIB comment for why a shorter context
    # made the profiler allocate more. Sizing is plan_gpu_split's job now.
    if max_len is None:
        max_len = 8192
    if eager is None:
        eager = False
    max_seqs = _max_num_seqs()

    t0 = time.time()
    dt = engine_dtype()
    llm = LLM(model=MODEL, trust_remote_code=True, dtype=dt,
              gpu_memory_utilization=gpu_frac, max_model_len=max_len,
              enforce_eager=eager, max_num_seqs=max_seqs,
              # Lets the caller hand the card back with llm.sleep() once the
              # transcribing is done. Only requested on cards too small to hold
              # the engine and the embedder at the same time, since it swaps in
              # a custom allocator that a big card has no reason to pay for.
              enable_sleep_mode=releasable,
              # ONE audio per request, and say how LONG it is. The count is not a
              # tuning choice -- the model declares a maximum of 1 and vLLM rejects
              # more. The length is what reclaims 6 GiB.
              #
              # vLLM sizes its whole memory reservation from a dummy profiling
              # request, and without `length` it builds that request from the
              # model's declared MAX_AUDIO_DURATION_S of 90 minutes: 180 windows
              # pushed through the audio encoder in a single forward, reserving
              # 6.18 GiB (180 x the measured 35.2 MiB per-window peak). Nothing
              # here can produce that request -- plan_windows caps every one of
              # them at window + 2*overlap, which needs 0.071 GiB. Declaring the
              # real length took the engine's fixed cost from 8.30 to 2.26 GiB at
              # no cost in throughput, and is the difference between needing a
              # 12 GiB card and running on 6.
              #
              # This is now a CONTRACT, not a hint: a request carrying more audio
              # than `length` will attempt the real 6.18 GiB forward and OOM at
              # runtime. That is why it is derived from the same window/overlap
              # that plan_windows uses rather than hardcoded -- both are
              # user-settable, and a fixed reservation would break under --window.
              # The 1.25 is slack for the resampler and the model's own padding.
              limit_mm_per_prompt={"audio": {"count": 1,
                                             "length": int((window + 2 * overlap)
                                                           * SR * 1.25)}})
    print(f"engine up in {time.time()-t0:.1f}s ({dt}, ctx {max_len}"
          + (", eager)" if eager else ")"), flush=True)
    return llm


def plan_windows(wav, ptxt, window, overlap):
    """-> reqs, offsets, cores. Each window is decoded with `overlap` seconds of
    context on both sides but only owns the segments whose midpoint lands in its
    own core, so nothing is duplicated."""
    w = int(window * SR)
    ctx = int(overlap * SR)
    reqs, offsets, cores = [], [], []
    for i in range(0, len(wav), w):
        if len(wav[i:i + w]) < SR:
            break
        a = max(0, i - ctx)
        b = min(len(wav), i + w + ctx)
        c = wav[a:b]
        offsets.append(a / SR)
        lo = (i - a) / SR
        hi = lo + w / SR
        if i == 0:
            lo = 0.0
        if i + w >= len(wav):
            hi = len(c) / SR
        cores.append((lo, hi))
        need = w + 2 * ctx
        if len(c) < need:
            c = np.pad(c, (0, need - len(c)))
        reqs.append({"prompt": ptxt, "multi_modal_data": {"audio": (c, SR)}})
    return reqs, offsets, cores


def assemble(outs, offsets, cores, wav, dur, no_silence_gate=False):
    """Turn raw window generations into the final segment list.

    Owns every correctness guard measured for this model: context-padding
    dedup, seam trimming, the repetition guard, coverage, and the silence
    gate. Shared by the single-file path and batch.py so the two cannot
    drift apart.  -> (segments, coverage, capped_windows)
    """
    capped = [i for i, o in enumerate(outs) if o.outputs[0].finish_reason == "length"]
    if capped:
        print(f"WARNING: {len(capped)} windows hit the token cap: {capped}", flush=True)

    segments = []
    dropped_ctx = 0
    for wi, (o, off, core) in enumerate(zip(outs, offsets, cores)):
        lo, hi = core
        for s in parse_transcript(o.outputs[0].text):
            # A segment decoded inside the context padding belongs to the neighbouring
            # window, which owns it as core. Keeping both would duplicate the speech.
            mid = (s.start + s.end) / 2
            if not (lo <= mid < hi):
                dropped_ctx += 1
                continue
            segments.append({
                "start": round(off + s.start, 2),
                "end": round(off + s.end, 2),
                "window": wi,
                "local_speaker": s.speaker,          # window-local, NOT globally meaningful
                "speaker": f"w{wi:03d}_{s.speaker}",  # unique per window until linked
                "text": s.text,
            })
    if dropped_ctx:
        print(f"dropped {dropped_ctx} segments decoded in context padding "
              f"(owned by a neighbouring window)", flush=True)
    segments.sort(key=lambda x: x["start"])

    # --overlap gives neighbouring windows shared audio, and a segment straddling the
    # seam can restate words the previous window already emitted. Measured 0.69% of
    # words duplicated at 5s context. Trim a repeated opening ONLY across a window
    # boundary, so genuine stuttering inside a window survives untouched.
    trimmed = 0
    for a, b in zip(segments, segments[1:]):
        if a["window"] == b["window"]:
            continue
        wa, wb = a["text"].split(), b["text"].split()
        for n in range(min(10, len(wa), len(wb) - 1), 2, -1):
            if [w.lower().strip(".,?!") for w in wa[-n:]] == \
               [w.lower().strip(".,?!") for w in wb[:n]]:
                b["text"] = " ".join(wb[n:])
                trimmed += n
                break
    if trimmed:
        print(f"trimmed {trimmed} words restated across window seams", flush=True)

    # Whisper's encoder loops on degenerate input and MOSS has none of Whisper's
    # decoder guards (no compression_ratio_threshold etc, because its decoder is
    # Qwen). Identical lines repeated in a row is the clean signal: 22 hits on the
    # onefailure we have, zero on healthy meetings. Whisper's own 2.4
    # compression threshold was measured to flag legitimate speech here (someone
    # reading phone numbers aloud) so it is deliberately not used.
    runs, cur, prev, drop = [], 0, None, set()
    for idx, sg in enumerate(segments):
        t = sg["text"].strip().lower()
        if t == prev:
            cur += 1
            runs.append(idx)
        else:
            if cur >= 3:
                drop.update(runs)
            cur, prev, runs = 0, t, [idx - 1]
    if cur >= 3:
        drop.update(runs)
    if drop:
        print(f"repetition guard: dropped {len(drop)} looped segments", flush=True)
        segments = [sg for i, sg in enumerate(segments) if i not in drop]

    # Coverage is measured BEFORE the gate. It exists to catch MOSS terminating
    # early (it silently dropped 38% of a meeting in single-pass mode), which is a
    # generation failure. The gate legitimately removes trailing fabricated audio,
    # and conflating the two turns a correct cleanup into a false alarm.
    # max(end), not segments[-1]["end"]: segments are sorted by START, and once
    # overlapping speech is emitted they nest — a short late-starting backchannel
    # can be last by start while ending well before the real final segment.
    # Measure coverage against where speech actually ENDS, not file duration.
    # Bed014 is 59.1 min of audio whose last real turn is at 52.9 min; judging
    # against file length marks correct silence at the tail as "stopped early".
    # (At 180 s windows the model fabricated content there, which masked this.)
    hop = int(0.01 * SR)
    _nf = len(wav) // hop
    _db = 20 * np.log10(np.sqrt((wav[:_nf * hop].reshape(_nf, hop) ** 2).mean(1) + 1e-12))
    _voiced = np.flatnonzero(_db >= SILENCE_GATE_DB)
    speech_end = (_voiced[-1] * 0.01) if len(_voiced) else dur
    # Clamp: a segment can run past the last voiced frame when it extends into
    # the zero-padded tail of the final window, which reads as >100% coverage
    # and makes the guard unfireable on that meeting.
    for _s in segments:
        _s["end"] = min(_s["end"], round(dur, 2))
    cov = min(max(s["end"] for s in segments) / max(speech_end, 1.0), 1.0) if segments else 0.0

    # Drop segments sitting in digital silence. MOSS occasionally fabricates a
    # coherent block of speech over nothing -- 123 s of it on ICSI Bed014, worth
    # 3.7 DER points. Swept -80..-45 dB: -70 is the knee. Below it you start
    # trading false alarm for miss and the net goes positive on clean meetings.
    if not no_silence_gate:
        hop = int(0.01 * SR)
        nf = len(wav) // hop
        rms = np.sqrt((wav[: nf * hop].reshape(nf, hop) ** 2).mean(1) + 1e-12)
        db = 20 * np.log10(rms)
        kept = []
        for s in segments:
            a, b = int(s["start"] / 0.01), min(int(s["end"] / 0.01), nf)
            if b > a and db[a:b].max() >= SILENCE_GATE_DB:
                kept.append(s)
        dropped = len(segments) - len(kept)
        if dropped:
            print(f"silence gate ({SILENCE_GATE_DB} dB): dropped {dropped} segments", flush=True)
        segments = kept
    return segments, cov, capped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=5.0,
                    help="seconds of extra audio given to each window on BOTH sides as "
                         "context. Segments are still only kept from the window's own "
                         "core, so nothing is duplicated -- this only buys the decoder "
                         "context across a boundary. A name landing right on a boundary "
                         "is the known residual failure of --glossary.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-frac", type=float, default=0.90,
                    help="share of VRAM vLLM reserves. It claims the whole pool up "
                         "front, so on a small card lower this to leave room for the "
                         "speaker embedder.")
    ap.add_argument("--glossary", default="",
                    help="comma-separated proper nouns to bias decoding toward, "
                         "e.g. 'Sreeram Kannan,EigenLayer,EigenCloud'")
    ap.add_argument("--no-silence-gate", action="store_true",
                    help="keep segments that sit in digital silence")
    args = ap.parse_args()

    ptxt = build_prompt(args.glossary)

    wav = load_audio_item(args.audio, sampling_rate=SR)
    dur = len(wav) / SR
    print(f"{args.audio}: {dur/60:.2f} min", flush=True)

    ctx = int(args.overlap * SR)
    reqs, offsets, cores = plan_windows(wav, ptxt, args.window, args.overlap)
    print(f"{len(reqs)} windows of {args.window:.0f}s"
          + (f" (+{args.overlap:.0f}s context each side)" if ctx else ""), flush=True)

    llm = build_engine(args.gpu_frac, window=args.window, overlap=args.overlap)

    mt = int((args.window + 2 * args.overlap) * 20)
    t1 = time.time()
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=mt))
    gen = time.time() - t1

    segments, cov, capped = assemble(
        outs, offsets, cores, wav, dur, args.no_silence_gate)

    print(f"\ngen {gen:.1f}s | {len(segments)} segments | coverage {cov:.1%} of speech | "
          f"{dur/gen:.1f}x realtime", flush=True)
    assert cov >= 0.95, (f"coverage {cov:.1%} of speech (ends {speech_end:.0f}s of {dur:.0f}s audio) — model stopped early")

    json.dump({"audio": args.audio, "duration_s": round(dur, 2), "window_s": args.window,
               "n_windows": len(reqs), "gen_s": round(gen, 2), "coverage": round(cov, 4),
               "windows_hit_cap": capped, "segments": segments}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
