"""Batched speaker embedding — same vectors, far fewer GPU round trips.

The sequential version does one forward pass per clip, so the card idles between
Python iterations (measured 0% utilisation). Clips are variable length, so we
sort by length and batch adjacent ones: padding within a batch stays small,
which matters because the ResNet's statistics pooling would otherwise average
zeros into the speaker vector.

  python3 embed_batched.py --run runs/X_v2.json --wav data/icsi/X.wav \\
      --out link/X.npz [--batch 32] [--per-speaker 2] [--verify link/seq.npz]
"""
import argparse, json, os, sys, time, traceback
from collections import defaultdict

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

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
_WSP = os.path.join(WORK, "wespeaker_src")
sys.path.insert(0, _WSP)

# Import the ResNet definitions WITHOUT executing wespeaker/__init__.py. That
# init pulls in wespeaker.cli.speaker, which imports silero_vad, which drags in
# onnxruntime -- none of which this script uses. Registering stub packages that
# carry only __path__ lets the submodule resolve while the real inits never run.
# Without this the embedder dies with ModuleNotFoundError: silero_vad on any box
# where that happens not to be installed, long after transcription succeeded.
import types
for _name, _sub in (("wespeaker", ""), ("wespeaker.models", "models")):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(_WSP, "wespeaker", _sub) if _sub
                       else os.path.join(_WSP, "wespeaker")]
        sys.modules[_name] = _m
import wespeaker.models.resnet as m_resnet

INSET, MIN_DUR, MAX_DUR = 0.20, 0.50, 20.0


def fbank(pcm, backend="wespeaker"):
    """80-bin log-mel. The two backends were trained with different front-ends and
    the window type is NOT interchangeable -- ERes2Net's own code calls
    Kaldi.fbank(audio, num_mel_bins=80) and takes every other default, which means
    the povey window, so that is what it must be fed.

    Scale does not matter: WeSpeaker's path multiplies by 32768 and ERes2Net's does
    not, but per-utterance mean subtraction removes any constant offset.
    """
    if backend == "eres2net":
        return kaldi.fbank(pcm, num_mel_bins=80)
    return kaldi.fbank(pcm, num_mel_bins=80, frame_length=25, frame_shift=10,
                       dither=0.0, sample_frequency=16000, window_type="hamming",
                       use_energy=False)


def load_backend(name):
    """-> (module, expects_scaled_pcm). Both take [B, T, 80] and statistics-pool,
    so both need the length-sorted batching below; zero padding averaged into a
    statistics pooling layer corrupts the speaker vector."""
    if name == "eres2net":
        import warnings
        warnings.filterwarnings("ignore")
        from modelscope.pipelines import pipeline
        pl = pipeline(task="speaker-verification",
                      model="iic/speech_eres2net_sv_en_voxceleb_16k", device="cuda")
        m = pl.model.embedding_model
        m.eval()
        return m, False
    return None, True


def load_model(args):
    """Put the embedding backend on the GPU. -> (model, fp16, scale_pcm)

    Split out from embed_file so --serve can pay this once for a whole batch.
    Measured on a 32-minute recording: 2.0s of actual embedding against ~5s of
    torch import, checkpoint read and CUDA init. Per file that is invisible when
    the files are hour-long meetings and dominant when they are short ones.
    """
    scale_pcm = args.backend != "eres2net"
    fp16 = args.fp16
    if args.backend == "eres2net":
        model, _ = load_backend("eres2net")
        model = model.cuda().eval()
        fp16 = False               # not validated in half precision for this one
    else:
        cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
        model = getattr(m_resnet, cfg["model"])(**cfg["model_args"])
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.cuda().eval()
        # The ResNets are compute-bound (ResNet293 managed 90 clips/s fp32 against
        # ECAPA's 907 -- it convolves
        # the spectrogram in 2D). fp16 measured 1.8x with no accuracy cost here.
        if fp16:
            model = model.half()
    return model, fp16, scale_pcm


def serve(args):
    """Embed recordings named on stdin, one JSON job per line, until EOF.

    Exists because a process per recording does not survive a large batch. The
    caller used to spawn one embedder per transcription and only reap them after
    the whole loop, so in-flight embedders equalled the file count; with short
    recordings they arrive faster than they initialise and a 122-file run died
    after 16, taking the inference engine with it. One resident worker bounds
    that to a single CUDA context no matter how long the queue is.

      in   {"run": ..., "wav": ..., "out": ..., "per_speaker": N,
            "overlap_aware": bool}   -- the last two optional
      out  {"out": ..., "ok": true|false, "err": ...}

    per_speaker is per JOB, not per worker: the cap is per window, so the right
    value depends on the recording's window size, and one resident worker serves
    recordings that need different ones.

    stdout carries acks and nothing else -- diagnostics go to stderr, or they
    would corrupt the channel.
    """
    model, fp16, scale_pcm = load_model(args)
    print("SERVE ready", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        ack = {"out": job["out"], "ok": True}
        try:
            msg = embed_file(model, fp16, scale_pcm, args,
                             job["run"], job["wav"], job["out"],
                             per_speaker=job.get("per_speaker"),
                             overlap_aware=job.get("overlap_aware"))
            print(msg, file=sys.stderr, flush=True)
        except Exception as e:
            # One bad recording must not take the worker down: the queue behind
            # it is still good. Report and keep serving.
            ack = {"out": job["out"], "ok": False,
                   "err": f"{type(e).__name__}: {e}"}
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        print(json.dumps(ack), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--wav")
    ap.add_argument("--out")
    ap.add_argument("--serve", action="store_true",
                    help="read jobs from stdin until EOF, keeping the model loaded")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--pad-ratio", type=float, default=1.15,
                    help="max length spread within a batch")
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--overlap-aware", action="store_true",
                    help="embed only the part of a segment where no OTHER local "
                         "speaker is active, and rank the --per-speaker cap by "
                         "that clean duration rather than total. Off by default "
                         "so the two can be measured against each other.")
    ap.add_argument("--backend", default="wespeaker",
                    choices=["wespeaker", "eres2net"],
                    help="eres2net: ERes2Net en/voxceleb, 192-d. Measured far better "
                         "on platform-compressed audio (podcast pair-error 7.7%% -> "
                         "1.3%%) and equal on ICSI.")
    ap.add_argument("--verify", default=None, help="npz from the sequential run")
    ap.add_argument("--config", default=os.path.join(WORK, "wsp_ckpt/resnet34/config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(WORK, "wsp_ckpt/resnet34/avg_model.pt"))
    args = ap.parse_args()
    # Before anything is loaded. The check also sits in embed_file, which is
    # where --serve jobs arrive, but doing it here means a one-shot refusal
    # costs nothing rather than a model load and a fight for the card.
    if args.wav:
        _refuse_clips(args.wav)
    if args.serve:
        return serve(args)
    if not (args.run and args.wav and args.out):
        ap.error("--run, --wav and --out are required unless --serve is given")
    model, fp16, scale_pcm = load_model(args)
    print(embed_file(model, fp16, scale_pcm, args, args.run, args.wav, args.out))


def _refuse_clips(path):
    """Clips are for ears, never for voiceprints.

    They are lossy and they are fragments picked for being easy to listen to,
    not for being representative. An embedding built from one would be quietly
    wrong forever, and wrong in a way nothing downstream could detect -- so the
    rule is enforced here rather than left to everyone remembering it."""
    p = Path(path)
    if "clips" in p.parts:
        raise SystemExit(
            f"refusing to embed {p.name}: it is a clip, cut for listening to. "
            f"Embeddings come from the original recording only.")


def embed_file(model, fp16, scale_pcm, args, run, wav_path, out_path,
               per_speaker=None, overlap_aware=None):
    """Embed one recording and write its npz. -> the WROTE summary line.

    `per_speaker` overrides args.per_speaker for THIS recording. The worker is
    resident across a whole batch, so a process-wide value cannot vary per
    meeting -- and it has to be able to, because the cap is per WINDOW: at 30s
    windows it keeps ~60% of the speech and at 300s windows ~12%, silently. That
    alone was enough to shatter one speaker into ten.
    """
    if per_speaker is None:
        per_speaker = args.per_speaker
    if overlap_aware is None:
        overlap_aware = args.overlap_aware
    _refuse_clips(wav_path)
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(1)          # mix, do not discard a channel
    if sr != 16000:
        # These models are 16 kHz. Handing fbank 44.1 kHz audio while telling it
        # sample_frequency=16000 shifts every formant and destroys the embedding
        # silently -- ICSI is 16 kHz wav so this never fired there, while every
        # real-world mp3 is 44.1 kHz and was being embedded from nonsense.
        import torchaudio
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, 16000).numpy()
        sr = 16000
    audio = np.ascontiguousarray(audio)
    segs = json.load(open(run))["segments"]

    # Where does this segment have the speaker to ITSELF? MOSS emits spans for
    # every local speaker it heard, so a region another local label also covers
    # had two people in it, and an embedding taken there is a blend of both --
    # it resembles neither and lands between them, which is a way to split one
    # person in two that has nothing to do with the room.
    #
    # Measured on AliMeeting far-field: reference overlap correlates with our
    # cpCER at +0.91, and MOSS's own detected overlap tracks that reference
    # closely, so this is knowable at inference without any extra model. In the
    # worst session 65% of the audio being embedded had 2+ voices in it.
    spans = sorted((s["start"], s["end"], s["local_speaker"], i)
                   for i, s in enumerate(segs))
    clean_runs = [None] * len(segs)
    for pos, (a, b, who, i) in enumerate(spans):
        cuts = []
        for x, y, other, _ in spans[pos + 1:]:
            if x >= b:
                break
            if other != who:
                cuts.append((x, min(y, b)))
        for x, y, other, _ in reversed(spans[:pos]):
            if y <= a:
                continue
            if other != who:
                cuts.append((max(x, a), min(y, b)))
        cuts.sort()
        free, cur = [], a
        for x, y in cuts:
            if x > cur:
                free.append((cur, x))
            cur = max(cur, y)
        if cur < b:
            free.append((cur, b))
        clean_runs[i] = max(free, key=lambda r: r[1] - r[0]) if free else None

    def _win(i):
        """The span to embed for segment i, and how much of it is clean."""
        s = segs[i]
        if overlap_aware and clean_runs[i] is not None:
            a, b = clean_runs[i]
            return a, b, b - a
        return s["start"], s["end"], (0.0 if clean_runs[i] is None
                                      else clean_runs[i][1] - clean_runs[i][0])

    by = defaultdict(list)
    for i, s in enumerate(segs):
        # Rank by CLEAN duration, not total. "Keep the longest" actively selects
        # for contamination: a longer span has more chance of colliding with
        # someone else, so in a high-overlap meeting the two longest segments
        # are the two worst vectors to define a speaker with.
        _, _, cl = _win(i)
        key = cl if overlap_aware else (s["end"] - s["start"])
        by[(s["window"], s["local_speaker"])].append((key, i))
    wanted = set()
    for items in by.values():
        items.sort(key=lambda x: -x[0])
        wanted.update(i for _, i in (items if per_speaker <= 0
                                     else items[:per_speaker]))

    # meta covers EVERY segment (link.py walks it to build aggregates and to sum
    # per-aggregate seconds); only `wanted` ones get a vector.
    t0 = time.time()
    feats, ids, meta = [], [], []
    for i, s in enumerate(segs):
        w0, w1, clean_dur = _win(i)
        a, b = w0 + INSET, w1 - INSET
        if b - a > MAX_DUR:
            c = 0.5 * (a + b)
            a, b = c - MAX_DUR / 2, c + MAX_DUR / 2
        ia, ib = max(0, int(a * sr)), min(len(audio), int(b * sr))
        dur = (ib - ia) / sr
        meta.append(dict(idx=i, window=s["window"], local=s["local_speaker"],
                         start=s["start"], end=s["end"], dur_used=round(dur, 3),
                         clean_dur=round(clean_dur, 3)))
        if dur < MIN_DUR or i not in wanted:
            continue
        pcm = torch.from_numpy(audio[ia:ib]).unsqueeze(0)
        if scale_pcm:
            pcm = pcm * 32768.0
        f = fbank(pcm, args.backend)
        feats.append(f - f.mean(0, keepdim=True))   # per-utterance mean subtraction
        ids.append(i)
    prep = time.time() - t0

    # Sort by length, then close a batch as soon as the longest member exceeds
    # the shortest by --pad-ratio. Plain fixed-size batching measured a minimum
    # cosine of 0.747 against sequential: this ResNet uses statistics pooling, so
    # zero padding is averaged into the speaker vector and a short clip batched
    # with a long one comes out wrong.
    order = sorted(range(len(feats)), key=lambda k: feats[k].shape[0])
    groups, cur = [], []
    for k in order:
        if cur and (feats[k].shape[0] > feats[cur[0]].shape[0] * args.pad_ratio
                    or len(cur) >= args.batch):
            groups.append(cur); cur = []
        cur.append(k)
    if cur:
        groups.append(cur)

    with torch.no_grad():
        pz = torch.zeros(1, 200, 80).cuda()
        probe = model(pz.half() if fp16 else pz)
        probe = probe[-1] if isinstance(probe, (tuple, list)) else probe
        emb_dim = int(probe.shape[-1])
    embs = np.zeros((len(feats), emb_dim), dtype=np.float32)
    t1 = time.time()
    with torch.no_grad():
        for chunk in groups:
            mx = max(feats[k].shape[0] for k in chunk)
            batch = torch.zeros(len(chunk), mx, 80)
            for r, k in enumerate(chunk):
                batch[r, :feats[k].shape[0]] = feats[k]
            xb = batch.cuda()
            out = model(xb.half() if fp16 else xb)
            out = out[-1] if isinstance(out, (tuple, list)) else out
            out = torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy()
            for r, k in enumerate(chunk):
                embs[k] = out[r]
    gpu = time.time() - t1

    np.savez(out_path, emb=embs, seg_idx=np.array(ids, dtype=np.int64),
             meta=np.array(json.dumps(meta)))
    msg = (f"WROTE {out_path} backend={args.backend} dim={emb_dim} "
           f"n_segments={len(segs)} embedded={len(ids)} "
           f"prep={prep:.1f}s gpu={gpu:.1f}s total={prep+gpu:.1f}s batches={len(groups)}")

    if args.verify:
        z = np.load(args.verify, allow_pickle=True)
        ref = {int(i): z["emb"][r] for r, i in enumerate(z["seg_idx"])}
        sims = [float(embs[r] @ ref[i]) for r, i in enumerate(ids) if i in ref]
        if sims:
            msg += (f"\nVERIFY vs sequential: {len(sims)} shared, "
                    f"cosine min={min(sims):.5f} mean={np.mean(sims):.5f}")
    return msg


if __name__ == "__main__":
    main()
