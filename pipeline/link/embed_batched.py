"""Batched speaker embedding — same vectors, far fewer GPU round trips.

The sequential version does one forward pass per clip, so the card idles between
Python iterations (measured 0% utilisation). Clips are variable length, so we
sort by length and batch adjacent ones: padding within a batch stays small,
which matters because the ResNet's statistics pooling would otherwise average
zeros into the speaker vector.

  python3 embed_batched.py --run runs/X_v2.json --wav data/icsi/X.wav \\
      --out link/X.npz [--batch 32] [--per-speaker 2] [--verify link/seq.npz]
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

# MS_WORK lets the pipeline live anywhere; /workspace is only the default because
# that is where a rented GPU box puts its scratch volume.
def _default_work():
    """/workspace when it is actually writable (the rented-GPU-box convention),
    otherwise the home directory. Matches default_work() in setup.sh."""
    import os
    if os.environ.get("MS_WORK"):
        return os.environ["MS_WORK"]
    if os.path.isdir("/workspace") and os.access("/workspace", os.W_OK):
        return "/workspace"
    return os.path.join(os.path.expanduser("~"), "meetscribe")


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
PROS_DIM = 7

# Pitch extraction was the whole cost of this script once it was added: prep went
# from ~19s to 96.5s on a 74-minute file, which matters because transcription is
# now only ~8s, so embedding -- not MOSS -- is the pipeline bottleneck and cannot
# be hidden behind the overlap in worker.py.
#
# Two cheap fixes. F0 percentiles converge in a few seconds of voiced speech, so
# there is no reason to run over the full 20s clip; and human F0 tops out around
# 400 Hz, so 8 kHz is plenty of sample rate for the autocorrelation.
PITCH_MAX_SEC = 5.0
PITCH_SR = 8000


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


def prosody(x, sr=16000):
    """Pitch + spectral-shape descriptor for one clip.

    Codecs, AGC and dynamic-range compression strip the spectral fine detail the
    ResNet relies on, but they cannot move a speaker's fundamental frequency.
    Measured: on a compressed podcast these seven numbers separate speakers as
    well as the full 256-d embedding (d' 2.04 vs 1.97), while on clean far-field
    audio they are far weaker (1.16 vs 8.64) -- robust but low capacity, so they
    are fused as a minority vote rather than used alone.

    Returns [n_voiced, log median f0, log p10, log p90, log range,
             spectral centroid / 1000, spectral tilt]. n_voiced is the pooling
    weight, so aggregating several clips approximates pooling their raw pitch.
    """
    xt = torch.as_tensor(x, dtype=torch.float32)
    out = np.zeros(PROS_DIM, dtype=np.float32)
    if len(xt) < sr // 2:
        return out
    xp = xt[: int(PITCH_MAX_SEC * sr)]
    psr = sr
    if sr > PITCH_SR:
        xp = torch.from_numpy(np.ascontiguousarray(xp.numpy()[:: sr // PITCH_SR]))
        psr = sr // (sr // PITCH_SR)
    try:
        f = torchaudio.functional.detect_pitch_frequency(
            xp.unsqueeze(0), psr, freq_low=60, freq_high=400).squeeze(0).numpy()
    except Exception:
        return out
    f = f[(f > 60) & (f < 400)]
    spec = torch.stft(xt, 512, 256, window=torch.hann_window(512),
                      return_complex=True).abs().mean(1).numpy() + 1e-9
    freqs = np.linspace(0, sr / 2, len(spec))
    cent = float((spec * freqs).sum() / spec.sum()) / 1000.0
    lo, hi = spec[freqs < 1000].mean(), spec[freqs > 3000].mean()
    tilt = float(np.log(hi / lo))
    if len(f) < 5:
        return np.array([0, 0, 0, 0, 0, cent, tilt], dtype=np.float32)
    p10, p90 = np.percentile(f, 10), np.percentile(f, 90)
    return np.array([len(f), np.log(np.median(f)), np.log(p10), np.log(p90),
                     np.log(max(p90 - p10, 1.0)), cent, tilt], dtype=np.float32)


_POOL_AUDIO = None


def _pros_job(a_b):
    a, b = a_b
    return prosody(_POOL_AUDIO[a:b])


def _init_pool(audio):
    global _POOL_AUDIO
    _POOL_AUDIO = audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--pad-ratio", type=float, default=1.15,
                    help="max length spread within a batch")
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--backend", default="wespeaker",
                    choices=["wespeaker", "eres2net"],
                    help="eres2net: ERes2Net en/voxceleb, 192-d. Measured far better "
                         "on platform-compressed audio (podcast pair-error 7.7%% -> "
                         "1.3%%) and equal on ICSI.")
    ap.add_argument("--prosody", action="store_true",
                    help="extract pitch descriptors. OFF by default: it was added to "
                         "rescue embeddings that turned out to be broken by a missing "
                         "resample, costs ~90s on a 74-min file, and once the audio is "
                         "correct it changes nothing and causes speaker-count regressions.")
    ap.add_argument("--no-prosody", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--prosody-jobs", type=int, default=0,
                    help="worker processes for pitch; 0 = cpu_count-1")
    ap.add_argument("--verify", default=None, help="npz from the sequential run")
    ap.add_argument("--config", default=os.path.join(WORK, "wsp_ckpt/resnet293/config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(WORK, "wsp_ckpt/resnet293/avg_model.pt"))
    args = ap.parse_args()

    scale_pcm = args.backend != "eres2net"
    if args.backend == "eres2net":
        model, _ = load_backend("eres2net")
        model = model.cuda().eval()
        args.fp16 = False          # not validated in half precision for this one
    else:
        cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
        model = getattr(m_resnet, cfg["model"])(**cfg["model_args"])
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.cuda().eval()
        # ResNet293 is compute-bound (90 clips/s fp32 vs ECAPA's 907 -- it convolves
        # the spectrogram in 2D). fp16 measured 1.8x with no accuracy cost here.
        if args.fp16:
            model = model.half()

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
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
    segs = json.load(open(args.run))["segments"]

    by = defaultdict(list)
    for i, s in enumerate(segs):
        by[(s["window"], s["local_speaker"])].append((s["end"] - s["start"], i))
    wanted = set()
    for items in by.values():
        items.sort(key=lambda x: -x[0])
        wanted.update(i for _, i in (items if args.per_speaker <= 0
                                     else items[:args.per_speaker]))

    # meta covers EVERY segment (link.py walks it to build aggregates and to sum
    # per-aggregate seconds); only `wanted` ones get a vector.
    t0 = time.time()
    feats, ids, meta, spans = [], [], [], []
    for i, s in enumerate(segs):
        a, b = s["start"] + INSET, s["end"] - INSET
        if b - a > MAX_DUR:
            c = 0.5 * (a + b)
            a, b = c - MAX_DUR / 2, c + MAX_DUR / 2
        ia, ib = max(0, int(a * sr)), min(len(audio), int(b * sr))
        dur = (ib - ia) / sr
        meta.append(dict(idx=i, window=s["window"], local=s["local_speaker"],
                         start=s["start"], end=s["end"], dur_used=round(dur, 3)))
        if dur < MIN_DUR or i not in wanted:
            continue
        pcm = torch.from_numpy(audio[ia:ib]).unsqueeze(0)
        if scale_pcm:
            pcm = pcm * 32768.0
        f = fbank(pcm, args.backend)
        feats.append(f - f.mean(0, keepdim=True))   # per-utterance mean subtraction
        spans.append((ia, ib))
        ids.append(i)
    if not args.prosody:
        pros = np.zeros((len(ids), PROS_DIM), dtype=np.float32)
    else:
        t_p = time.time()
        n_jobs = args.prosody_jobs or max(1, (os.cpu_count() or 2) - 1)
        if n_jobs > 1 and len(spans) > 8:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(n_jobs, _init_pool, (audio,)) as pool:
                pros = np.array(pool.map(_pros_job, spans, chunksize=8),
                                dtype=np.float32)
        else:
            pros = np.array([prosody(audio[a:b]) for a, b in spans],
                            dtype=np.float32)
        print(f"prosody {time.time()-t_p:.1f}s over {n_jobs} workers", flush=True)
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
        probe = model(pz.half() if args.fp16 else pz)
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
            out = model(xb.half() if args.fp16 else xb)
            out = out[-1] if isinstance(out, (tuple, list)) else out
            out = torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy()
            for r, k in enumerate(chunk):
                embs[k] = out[r]
    gpu = time.time() - t1

    np.savez(args.out, emb=embs, seg_idx=np.array(ids, dtype=np.int64),
             pros=np.asarray(pros, dtype=np.float32),
             meta=np.array(json.dumps(meta)))
    print(f"WROTE {args.out} backend={args.backend} dim={emb_dim} "
          f"n_segments={len(segs)} embedded={len(ids)} "
          f"prep={prep:.1f}s gpu={gpu:.1f}s total={prep+gpu:.1f}s batches={len(groups)}")

    if args.verify:
        z = np.load(args.verify, allow_pickle=True)
        ref = {int(i): z["emb"][r] for r, i in enumerate(z["seg_idx"])}
        sims = [float(embs[r] @ ref[i]) for r, i in enumerate(ids) if i in ref]
        if sims:
            print(f"VERIFY vs sequential: {len(sims)} shared, "
                  f"cosine min={min(sims):.5f} mean={np.mean(sims):.5f}")


if __name__ == "__main__":
    main()
