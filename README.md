# meetscribe

**Meeting audio in → speaker-attributed transcript out. On your own GPU.**

```
Platform Review — weekly
104.0 min · 1074 turns · 16 speakers
==================================================================

[0:00:00] Dana Whitfield
  Meeting open, welcome everybody.
  So we'll start with the apologies and I guess we have Priya, everyone
  else seems to be here, I think. So could I have a mover please?

[0:00:12] Marcus Elle
  Dana, just before you do, I have to leave at 1.30 to attend the
  roadmap session. So sorry about that.
```

> **Name a voice once** and it is recognised in every meeting after that —
> including ones you already transcribed.

---

## Contents

[Quickstart](#quickstart) · [Speed](#speed) · [Remote GPU](#remote-gpu--optional) ·
[How it works](#how-it-works) · [Requirements](#requirements) ·
[Layout](#layout) · [Licence](#licence)

---

## Quickstart

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli && ./setup.sh        # 2-10 min, mostly 2 GB of weights

cp ~/recordings/*.mp3 inbox/
./transcribe
```

Each recording becomes a directory in `library/` — transcript, audio, and a
few seconds of each voice. The inbox empties as they finish.

Everyone starts as `Speaker 1`. Name them once and they are recognised in every
recording afterwards, and in the ones you already have:

```bash
./speakers link --apply                   # group each voice across meetings
./speakers review                         # the groups waiting for a name
./speakers name 12 "Dana Whitfield"       # names them in every meeting at once
./speakers apply --apply                  # backfill the transcripts you have
```

`review` prints a line of what each voice said and a clip to play, which is
usually enough to tell who it is: `./speakers play <meeting> G02`.

**Supported formats:** `wav mp3 m4a flac ogg opus aac m4b aiff wma mp4 webm
mkv` — any sample rate, auto-converted to 16 kHz mono.

**Unfamiliar names?** Proper nouns the model has never heard come out as the
nearest familiar English. Give it a vocabulary — a file per subject, or a list:

```bash
./transcribe --glossary crypto.txt
./transcribe --glossary "Dana Whitfield,Northwind"
```

A `glossary.txt` beside where you run is picked up automatically.

### Keep the engine warm

The engine costs ~60-70 s to load and that is paid per run. On a 3-minute clip
it is nearly the whole wall clock: **99 s cold against 6.6 s with it already
up**, measured on a 3090.

```bash
./engine start      # once, then it stays
./engine stop       # hand the card back
```

If the engine is running, commands use it. Otherwise they load their own.

> Every command and flag is in **[RUNBOOK.md](RUNBOOK.md)**.

Day to day, the loop is in [WORKFLOW.md](WORKFLOW.md):
transcribe, group the voices, name whoever is new, backfill.

---

## Speed

One batch, 6 meetings, 6.27 hours of 16 kHz mono WAV, default flags. Three
rented boxes, each a fresh `git clone` and `./setup.sh`.

| GPU | VRAM | Engine load | 6.27 h of audio |
|:---|:---|:---|:---|
| RTX 5090 | 32 GB | 56 s | **29.5 s — 766×** |
| RTX 3090 | 24 GB | 66 s | 70.5 s — 320× |
| RTX 2060 | 6 GB | 127 s | 580 s — 39× |

**Decode is 85–97% of that time**, and every run prints the breakdown. Engine
load is once, not per meeting, and the first load on a new machine is ~3× slower
while `torch.compile` builds its cache — 219 s against 66 s on the same 3090.

Feeding it MP3 or any other format adds a conversion pass first, which runs
across files in parallel and splits a long recording across cores: 6.27 h of MP3
converts in about 22 s on a 48-core box, once.

---

## Remote GPU — optional

Everything above runs on your own card. This is for when you have not got one,
or want a faster one for a big batch. Skip the whole section if neither applies.

`--host` runs the GPU work on another machine. Your **library** is the thing
that stays local — transcripts, clips and `speakers.db` live here and are the
authoritative copy.

The audio itself does go over. It is uploaded, transcribed, and sits in the
box's own library until you destroy the instance. Nothing goes to a third-party
service, but it is on that machine while it works, so rent from someone you are
willing to put the recording on.

**One-time setup on the remote box:**

```bash
ssh mybox 'git clone https://github.com/suzuenhasa/meetscribe-cli.git /opt/meetscribe \
           && cd /opt/meetscribe && ./setup.sh'
```

**Then from your laptop:**

```bash
./transcribe --host mybox
```

`speakers.db` syncs both ways, so the box recognises your people and anything
it learns comes home. Destroy the instance — lose nothing.

For rented boxes, `vast/provision.sh` handles unattended install. See
[vast/README.md](vast/README.md).

---

## How it works

```
audio ──► MOSS on vLLM        →  text + per-window speaker labels
             │
      WeSpeaker ResNet34-LM  →  voice embedding per segment
             │
      constrained clustering  →  Speaker 1, Speaker 2, …
             │
      profile store (sqlite)  →  Dana Whitfield, Marcus Elle
```

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
labels speakers within each window, but window 4's "S01" is not necessarily
window 5's. [WeSpeaker](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM)
embeddings link segments across a recording, and the profile store puts names
on the result — across recordings and across months.

---

## Requirements

| | |
|:---|:---|
| **GPU** | NVIDIA, 6 GB+ VRAM, RTX 20-series or newer. Pascal will not run vLLM. The GTX 16-series meets the compute requirement but has no tensor cores and is untested. |
| **System** | FHS-style Linux. On NixOS/non-FHS distros, bring your own Python and set library paths for NVIDIA driver, GCC runtime, zlib, plus `TRITON_LIBCUDA_PATH`. |
| **Tools** | `ffmpeg` |

A 6 GB card transcribes and embeds in two passes rather than simultaneously —
slower, but complete.

`./setup.sh --check` verifies your environment without installing anything.

---

## Layout

```
inbox/          drop recordings here; empties as it finishes
library/        one directory per meeting
speakers.db     who people are — the one thing not rebuildable from audio
```

**Back up `speakers.db`.** Everything else can be rebuilt from the audio.

---

**[RUNBOOK.md](RUNBOOK.md)** — full command reference, flags, and
troubleshooting.

---

## Licence

Apache License 2.0 — see [LICENSE](LICENSE). Use it, change it, ship it, sell
it; keep the notices. Contributions are taken under the same terms.

MOSS, WeSpeaker and vLLM are Apache-2.0 as well and are downloaded at install
time rather than bundled — [NOTICE](NOTICE) lists them and `setup.sh` pins the
exact revisions.
