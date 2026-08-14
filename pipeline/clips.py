#!/usr/bin/env python3
"""Cut a few seconds of each speaker out of the recording, so a voice can be
heard after the recording is gone.

Naming someone means hearing them: "G02" tells you nothing, and the only way to
put a name to a cluster is to listen to it. That currently needs the full
original beside the transcript, which makes the library as large as your audio
collection and useless the moment you archive or delete the source.

A handful of clips is enough for the job. Three per speaker, longest turns
first, at 64 kbps mono: about a megabyte for a five-speaker meeting against
54 MB for the mp3 it came from. Cut from the ORIGINAL rather than from the 16 kHz
decode cache, so they are faithful enough to judge a voice by.

CLIPS ARE NEVER EMBEDDED. They are lossy, and they are fragments chosen for
being easy to listen to rather than for being representative. A voiceprint built
from one would be quietly wrong forever, and wrong in a way nothing downstream
could detect -- so embed_batched.py refuses any path under a clips directory
rather than trusting everyone to remember.
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PER_SPEAKER = 3
MAX_SECONDS = 12.0
MIN_SECONDS = 1.5
BITRATE = "64k"


def pick(transcript_json, per_speaker=PER_SPEAKER):
    """-> {cluster: [(start, end), ...]}   the longest turns, in time order."""
    try:
        doc = json.loads(Path(transcript_json).read_text())
    except (OSError, ValueError):
        return {}
    by = {}
    for s in doc.get("segments", []):
        g = s.get("global") or s.get("speaker")
        if not g or str(g).startswith("G-"):
            continue
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b - a >= MIN_SECONDS:
            by.setdefault(g, []).append((a, min(b, a + MAX_SECONDS)))
    out = {}
    for g, spans in by.items():
        longest = sorted(spans, key=lambda s: -(s[1] - s[0]))[:per_speaker]
        out[g] = sorted(longest)
    return out


def cut(audio, transcript_json, into, per_speaker=PER_SPEAKER):
    """Write the clips. -> number written.

    Best effort throughout: a meeting without clips is a meeting you have to
    keep the original of, which is exactly where we were before. It is not worth
    failing a transcription over."""
    audio = Path(audio)
    if not audio.exists():
        return 0
    spans = pick(transcript_json, per_speaker)
    if not spans:
        return 0
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)

    jobs = []
    for g, ss in spans.items():
        for i, (a, b) in enumerate(ss, 1):
            jobs.append((g, i, a, max(0.5, b - a)))

    def one(job):
        g, i, start, dur = job
        dst = into / f"{g}-{i}.mp3"
        # -ss before -i seeks by keyframe and is fast; accuracy inside a second
        # does not matter for something a person listens to.
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                            "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                            "-i", str(audio), "-ac", "1", "-b:a", BITRATE,
                            str(dst)], capture_output=True)
        return dst.exists() and dst.stat().st_size > 0 and r.returncode == 0

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        return sum(1 for ok in pool.map(one, jobs) if ok)
