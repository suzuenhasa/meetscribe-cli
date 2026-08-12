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
        # ResNet293 is compute-bound (90 clips/s fp32 vs ECAPA's 907 -- it convolves
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

      in   {"run": ..., "wav": ..., "out": ...}
      out  {"out": ..., "ok": true|false, "err": ...}

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
                             job["run"], job["wav"], job["out"])
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
    ap.add_argument("--backend", default="wespeaker",
                    choices=["wespeaker", "eres2net"],
                    help="eres2net: ERes2Net en/voxceleb, 192-d. Measured far better "
                         "on platform-compressed audio (podcast pair-error 7.7%% -> "
                         "1.3%%) and equal on ICSI.")
    ap.add_argument("--verify", default=None, help="npz from the sequential run")
    ap.add_argument("--config", default=os.path.join(WORK, "wsp_ckpt/resnet293/config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(WORK, "wsp_ckpt/resnet293/avg_model.pt"))
    args = ap.parse_args()
    if args.serve:
        return serve(args)
    if not (args.run and args.wav and args.out):
        ap.error("--run, --wav and --out are required unless --serve is given")
    model, fp16, scale_pcm = load_model(args)
    print(embed_file(model, fp16, scale_pcm, args, args.run, args.wav, args.out))


def embed_file(model, fp16, scale_pcm, args, run, wav_path, out_path):
    """Embed one recording and write its npz. -> the WROTE summary line."""
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
    feats, ids, meta = [], [], []
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
