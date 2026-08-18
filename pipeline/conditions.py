#!/usr/bin/env python3
"""What a recording actually sounds like, measured from the audio.

Sub-profiles are discovered by vector geometry, which is right -- matching is
done on vectors, so a profile has to be a centroid of things that are close in
that space or it will never match anything. What geometry cannot do is say what
a profile IS. It produces `auto-7`, a counter, and a human is left guessing.

This is the other half: open the wav and look. A telephone codec ends at
3.4 kHz and an open microphone runs to the Nyquist limit, so the difference is
visible in the spectrum and needs no reference to who is speaking or what corpus
this is. It is the least domain-specific signal available here -- the vector
geometry is fitted to whatever recordings a library happens to hold, while a
brick wall at 3.5 kHz means a phone call anywhere.

Measured per SPEAKER, not per file. In a remote meeting one participant dials in
while the rest share a room, and a per-file answer is then wrong for someone. On
the Court's own remote arguments this found one justice on a measurably
different line from the other twelve.

  cutoff    highest frequency still carrying energy
  cliff     the sharpest sustained fall above 2 kHz, and where it happens.
            A codec does not roll off, it stops: a first attempt measured how
            far the spectrum falls from its peak and reported the Court's most
            telephonic term as its WIDEST-band, because that statistic is
            dominated by the natural tilt of speech.
"""
import wave

import numpy as np

NFFT = 1024
BAND = 300.0            # Hz, the step used when hunting for an edge
CLIFF_DB = 12.0         # a fall this steep in one step is a codec, not a room
MIN_SEC = 20.0          # below this the spectrum is too noisy to call


def ltas(wav_path, spans=None, cap=90.0):
    """Long-term average spectrum of the speech. -> (power, sample_rate)

    `spans` limits it to one speaker's turns. Only the louder half of frames are
    kept, so silence does not flatten the average into meaninglessness.
    """
    with wave.open(str(wav_path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        if spans is None:
            mid = max(0, n // 2 - int(cap * sr / 2))
            spans_f = [(mid, min(n, mid + int(cap * sr)))]
        else:
            spans_f = [(int(a * sr), min(int(b * sr), n)) for a, b in spans]
        acc, got = np.zeros(NFFT // 2 + 1), 0.0
        win = np.hanning(NFFT).astype(np.float32)
        for i0, i1 in spans_f:
            if got >= cap or i1 - i0 < NFFT * 4:
                continue
            w.setpos(i0)
            x = np.frombuffer(w.readframes(i1 - i0), dtype=np.int16)
            x = x.astype(np.float32) / 32768.0
            nfr = (x.size - NFFT) // (NFFT // 2)
            if nfr < 4:
                continue
            fr = np.lib.stride_tricks.as_strided(
                x, shape=(nfr, NFFT),
                strides=(x.strides[0] * (NFFT // 2), x.strides[0]))
            P = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
            e = P.sum(axis=1)
            P = P[e >= np.median(e)]
            if len(P):
                acc += P.sum(axis=0)
                got += (i1 - i0) / sr
    return (acc / max(got, 1e-9), sr) if got >= MIN_SEC else (None, sr)


def cliff(power, sr, step=BAND):
    """-> (edge_hz, drop_db) of the sharpest sustained fall above 2 kHz."""
    db = 10 * np.log10(power + 1e-20)
    freqs = np.linspace(0, sr / 2, len(db))
    best, f = (0.0, 0.0), 2000.0
    while f + 2 * step <= sr / 2:
        lo = db[(freqs >= f) & (freqs < f + step)].mean()
        hi = db[(freqs >= f + step) & (freqs < f + 2 * step)].mean()
        if lo - hi > best[1]:
            best = (f + step, lo - hi)
        f += step / 2
    return best


def describe(wav_path, spans=None, cap=90.0):
    """-> {"label", "edge_hz", "drop_db"} or None if there is too little speech.

    `label` is deliberately a description of the channel and not a guess at the
    device: "narrowband" is what was measured, "telephone" is an inference about
    why, and only a human knows whether it was a phone, a headset in HFP mode or
    a bad codec.
    """
    try:
        power, sr = ltas(wav_path, spans, cap)
    except Exception:
        return None
    if power is None:
        return None
    edge, drop = cliff(power, sr)
    return {"label": "narrowband" if drop >= CLIFF_DB else "wideband",
            "edge_hz": float(edge), "drop_db": float(drop)}


def agree(descs):
    """One verdict for a set of recordings. -> (label, edge_hz, share) or None

    A sub-profile is only worth labelling from measurement if its members AGREE.
    Vector geometry grouped them; if that grouping does not line up with what the
    channel did, the honest answer is that measurement has nothing to say about
    this one.
    """
    ds = [d for d in descs if d]
    if not ds:
        return None
    lab = max(("narrowband", "wideband"),
              key=lambda L: sum(1 for d in ds if d["label"] == L))
    hits = [d for d in ds if d["label"] == lab]
    return lab, float(np.median([d["edge_hz"] for d in hits])), len(hits) / len(ds)
