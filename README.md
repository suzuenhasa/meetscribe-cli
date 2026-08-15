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

[Quickstart](#quickstart) · [Speed](#speed) · [Remote GPU](#remote-gpu) ·
[How it works](#how-it-works) · [Requirements](#requirements) ·
[Layout](#layout)

---

## Quickstart

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli && ./setup.sh        # ~10 min, mostly 2 GB of weights

cp ~/recordings/*.mp3 inbox/
./transcribe
```

Each recording becomes a directory in `library/` — transcript, audio, and a
few seconds of each voice. The inbox empties as they finish.

```bash
./speakers who <meeting>                  # see speaker IDs
./speakers play <meeting> G02             # hear a voice sample
./speakers name <meeting> G02 "Dana Whitfield"
./speakers apply --apply                  # name them in older meetings too
```

**Supported formats:** `wav mp3 m4a flac ogg opus aac m4b aiff wma mp4 webm
mkv` — any sample rate, auto-converted to 16 kHz mono.

### Keep the engine warm

The engine costs ~70 s to load and that is paid per run. On a 3-minute clip it
is the whole wall clock: **145 s cold against 25 s with it already up.**

```bash
./engine start      # ~70 s, once
./engine stop       # hand the card back
```

If the engine is running, commands use it. Otherwise they load their own.

> Every command and flag is in **[RUNBOOK.md](RUNBOOK.md)**.

---

## Speed

One batch, 6 meetings, 6.27 hours of audio, default flags.

| GPU | VRAM | Engine load | 16 kHz mono WAV | 44.1 kHz stereo MP3 |
|:---|:---|:---|:---|:---|
| RTX 5090 | 32 GB | 73 s | **48 s — 468×** | 80 s — 282× |
| RTX 3090 | 24 GB | 63 s | 104 s — 217× | 130 s — 174× |
| RTX 2060 | 6 GB | 114 s | 857 s — 26× | 966 s — 23× |

*MP3 decoding is CPU-bound, so those figures vary by host — two RTX 3090s on
different hosts gave 174× and 142× on MP3 but 217× and 203× on WAV. First run
on a new machine is ~3× slower while `torch.compile` warms up.*

---

## Remote GPU

`--host` moves only the compute. Your audio, library, and voice profiles stay
local.

**One-time setup on the remote box:**

```bash
ssh mybox 'git clone https://github.com/suzuenhasa/meetscribe-cli.git /opt/meetscribe \
           && cd /opt/meetscribe && ./setup.sh'
```

**Then from your laptop:**

```bash
./transcribe --host mybox
```

Audio goes up, meetings come back. `speakers.db` syncs both ways so the box
recognises your people and anything it learns comes home. Destroy the instance
— lose nothing.

For rented boxes, `vast/provision.sh` handles unattended install. See
[vast/README.md](vast/README.md).

---

## How it works

```
audio ──► MOSS on vLLM        →  text + per-window speaker labels
             │
      WeSpeaker ResNet293-LM  →  voice embedding per segment
             │
      constrained clustering  →  Speaker 1, Speaker 2, …
             │
      profile store (sqlite)  →  Dana Whitfield, Marcus Elle
```

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
labels speakers within each window, but window 4's "S01" is not necessarily
window 5's. [WeSpeaker](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)
embeddings link segments across a recording, and the profile store puts names
on the result — across recordings and across months.

---

## Requirements

| | |
|:---|:---|
| **GPU** | NVIDIA, 6 GB+ VRAM, compute capability 7.0+ (Turing / GTX 16-series / RTX 20-series and newer). Pascal will not run vLLM. |
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
