#!/usr/bin/env python3
"""Look at, and listen to, the voices in a transcript so you can name them.

  preview.py who  meeting.json                 every voice, with what they said
  preview.py play meeting.json G02 [n]         hear that voice (n clips, default 3)
  preview.py clips meeting.json G02            just print the timestamps

Runs where your audio is. Naming a cluster is impossible without knowing who it
is, and "G02" tells you nothing — this is how you find out.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm")


def load(arg):
    """Accept the .json, the .txt, or the bare stem."""
    p = Path(arg)
    if p.suffix.lower() == ".txt":
        p = p.with_suffix(".json")
    if not p.exists() and p.suffix != ".json":
        p = Path(str(p) + ".json")
    if not p.exists():
        raise SystemExit(f"no transcript at {p}\n"
                         f"pass the .json that transcribe wrote next to your audio")
    return p, json.loads(p.read_text())


def audio_for(js: Path, doc):
    """The recording this transcript came from, wherever it ended up.

    Beside the transcript is the common case, and it stops being true the moment
    transcripts are kept apart from the audio -- an out/ directory beside the
    recordings, which is a perfectly ordinary way to keep a folder tidy. Look up
    one level too, and in the audio path the run recorded, before giving up:
    "NOT FOUND beside the transcript" is a true statement about a file sitting
    one directory away, and it takes play and clips with it.
    """
    # The library names it <slug>-audio.<ext> beside <slug>-transcript.json, so
    # deriving the audio from the transcript's own stem finds nothing -- it looks
    # for <slug>-transcript.mp3. Try the library form first, then the old
    # side-by-side one, then a directory up for a transcript kept apart from its
    # recording.
    stem = js.stem
    for suffix in ("-transcript", "-linked", ""):
        if stem.endswith(suffix) and suffix:
            stem = stem[: -len(suffix)]
            break
    for d in (js.parent, js.parent.parent):
        for base in (f"{stem}-audio", stem, js.stem):
            for ext in AUDIO_EXT:
                c = d / (base + ext)
                if c.exists():
                    return c
    rec = doc.get("audio")
    if rec:
        rec = Path(rec)
        if rec.exists():
            return rec
        # The path a REMOTE run recorded is the box's, not yours. The name still
        # tells you what to look for here.
        for d in (js.parent, js.parent.parent):
            c = d / rec.name
            if c.exists():
                return c
    return None


def by_cluster(doc):
    out = {}
    for s in doc.get("segments", []):
        g = s.get("global")
        if not g or str(g).startswith("G-"):
            continue
        e = out.setdefault(g, {"secs": 0.0, "segs": []})
        e["secs"] += float(s["end"]) - float(s["start"])
        e["segs"].append(s)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["secs"]))


def hms(t):
    return f"{int(t//3600)}:{int(t%3600//60):02d}:{int(t%60):02d}"


def best_clips(segs, n):
    """The longest segments, in time order — the clearest samples of a voice."""
    picked = sorted(segs, key=lambda s: -(s["end"] - s["start"]))[:n]
    return sorted(picked, key=lambda s: s["start"])


def cmd_who(args):
    js, doc = load(args[0])
    groups = by_cluster(doc)
    aud = audio_for(js, doc)
    print(f"\n{js.stem}   {doc.get('duration_s', 0)/60:.1f} min   "
          f"{len(groups)} voice{'s' if len(groups) != 1 else ''}")
    n_clips = len(list((js.parent / "clips").glob("*.mp3")))
    if aud:
        print(f"audio: {aud}\n")
    elif n_clips:
        # The normal state of a library whose originals have been pruned, not a
        # problem: clips are what naming a voice needs.
        print(f"audio: pruned — {n_clips} clips here, enough to play and name\n")
    else:
        print("audio: NOT FOUND, and no clips — play unavailable\n")
    for g, e in groups.items():
        print(f"  {g}   {e['secs']/60:5.1f} min   {len(e['segs'])} turns")
        for s in best_clips(e["segs"], 2):
            txt = " ".join((s.get("text") or "").split())[:96]
            print(f"       [{hms(s['start'])}] {txt}")
        print()
    g0 = next(iter(groups), "G00")
    print(f'  hear one:  ./speakers play {shlex.quote(js.name)} {g0}')
    print(f'  name one:  ./speakers name {shlex.quote(js.stem)} {g0} "Their Name"')


def player():
    if shutil.which("ffplay"):
        return lambda f, ss, d: ["ffplay", "-v", "quiet", "-nodisp", "-autoexit",
                                 "-ss", str(ss), "-t", str(d), str(f)]
    if shutil.which("mpv"):
        return lambda f, ss, d: ["mpv", "--really-quiet", "--no-video",
                                 f"--start={ss}", f"--length={d}", str(f)]
    return None


def cmd_play(args):
    js, doc = load(args[0])
    cid = args[1]
    n = int(args[2]) if len(args) > 2 else 3
    groups = by_cluster(doc)
    if cid not in groups:
        raise SystemExit(f"no {cid} here. Voices: {', '.join(groups)}")
    play = player()
    if not play:
        raise SystemExit("need ffplay or mpv to play audio (apt-get install -y ffmpeg)")

    # Cut clips first. They are a few hundred KB against tens of MB, they are
    # what survives archiving or deleting the source, and playing them needs no
    # seeking into a 74-minute file.
    cut = sorted((js.parent / "clips").glob(f"{cid}-*.mp3"))[:n]
    if cut:
        print(f"\n{cid} — {groups[cid]['secs']/60:.1f} min of speech, "
              f"playing {len(cut)} clips")
        for c in cut:
            print(f"  {c.name}", flush=True)
            subprocess.run(play(c, 0.0, 12.0),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    aud = audio_for(js, doc)
    if not aud:
        raise SystemExit(f"no clips and no audio beside {js.name}. Transcribe it "
                         f"again to cut clips, or put the recording back.")
    clips = best_clips(groups[cid]["segs"], n)
    print(f"\n{cid} — {groups[cid]['secs']/60:.1f} min of speech, playing {len(clips)} clips")
    for s in clips:
        dur = max(1.5, min(float(s["end"]) - float(s["start"]), 12.0))
        txt = " ".join((s.get("text") or "").split())[:88]
        print(f"  [{hms(s['start'])}] {txt}", flush=True)
        subprocess.run(play(aud, float(s["start"]), dur),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'\n  name them:  ./speakers name {shlex.quote(js.stem)} {cid} "Their Name"')


def cmd_clips(args):
    js, doc = load(args[0])
    cid = args[1]
    groups = by_cluster(doc)
    if cid not in groups:
        raise SystemExit(f"no {cid} here. Voices: {', '.join(groups)}")
    for s in best_clips(groups[cid]["segs"], int(args[2]) if len(args) > 2 else 5):
        print(f"{hms(s['start'])}  {s['end']-s['start']:5.1f}s  "
              f"{' '.join((s.get('text') or '').split())[:90]}")


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    return {"who": cmd_who, "play": cmd_play, "clips": cmd_clips}.get(
        cmd, lambda a: print(__doc__.strip()))(args) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
