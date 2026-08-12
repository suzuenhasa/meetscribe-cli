"""Full-meeting windowed transcription via vLLM, with timestamps rebased to meeting time.

  python3 transcribe_meeting.py meeting.mp3 --out out.json
"""
import argparse, json, time

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


def build_engine(gpu_frac=0.90, max_len=None, eager=None):
    """Load vLLM. ~66s, so a batch loads it once and keeps it resident.

    On a small card the defaults do not fit and the failure is opaque -- "No
    available memory for the cache blocks", after the weights have already
    loaded. Three things are oversized for this workload, in order of how much
    they actually cost (measured on an 8 GiB card, where available KV cache was
    -7.7 GiB before any of this):

      max_num_seqs    the big one. The default assumes a server; the profiling
                      run sizes itself against it. Dropping 256 -> 2 recovered
                      about 6 GiB.
      max_model_len   8192 sizes the KV cache, but a 30 s window with 5 s of
                      context either side generates ~800 tokens.
      CUDA graphs     capture costs ~0.5 GiB.

    All three are chosen from free VRAM unless passed explicitly. Note this is
    mitigation, not a fix: 8 GiB still does not fit even at max_num_seqs=1,
    max_model_len=1024 and eager. The floor is audio-encoder activation, not the
    1.7 GiB of weights. 12 GiB is the real requirement.
    """
    free_gib = None
    try:
        import torch
        free_b, _ = torch.cuda.mem_get_info()
        free_gib = free_b / 2**30
    except Exception:
        pass
    if max_len is None:
        max_len = 4096 if (free_gib is not None and free_gib < 10) else 8192
    if eager is None:
        eager = bool(free_gib is not None and free_gib < 10)
    # Concurrency also sizes the profiling run; the default assumes a server.
    max_seqs = 16 if (free_gib is not None and free_gib < 10) else 256

    t0 = time.time()
    dt = engine_dtype()
    llm = LLM(model=MODEL, trust_remote_code=True, dtype=dt,
              gpu_memory_utilization=gpu_frac, max_model_len=max_len,
              enforce_eager=eager, max_num_seqs=max_seqs,
              # ONE audio per request. This is not a tuning choice: the model
              # declares a maximum of 1, and vLLM rejects more with
              # "At most 1 audio(s) may be provided in one prompt". The value was
              # 4 for a long time and was simply inert -- correcting it moved
              # available KV cache by 0.14 GiB, which is how we know it was never
              # the reservation it looked like.
              limit_mm_per_prompt={"audio": 1})
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

    llm = build_engine(args.gpu_frac)

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
