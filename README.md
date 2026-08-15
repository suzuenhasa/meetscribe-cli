# meetscribe

Meeting audio in, speaker-attributed transcript out. Runs on your own GPU.

```
Platform Review — weekly
104.0 min · 1074 turns · 16 speakers
==================================================================

[0:00:00] Dana Whitfield
  Meeting open, welcome everybody.
  So we'll start with the apologies and I guess we have Priya, everyone else
  seems to be here, I think. So could I have a mover please?

[0:00:12] Marcus Elle
  Dana, just before you do, I have to leave at 1.30 to attend the roadmap
  session. So sorry about that.
```

Name a voice once and it is recognised in every meeting after that — including
the ones you already transcribed.

---

## Get started

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli
./setup.sh                        # ~10 min, mostly 2 GB of weights

cp ~/recordings/*.mp3 inbox/
./transcribe
```

Each recording becomes a directory in `library/` — transcript, audio, and a few
seconds of each voice. The inbox empties as they finish.

```bash
./speakers who <meeting>                   # which voice is which
./speakers play <meeting> G02              # hear one
./speakers name <meeting> G02 "Dana Whitfield"
./speakers apply --apply                   # name them in older meetings too
```

`wav mp3 m4a flac ogg opus aac m4b aiff wma mp4 webm mkv`, any sample rate.
Conversion to 16 kHz mono happens automatically.

**Keep the engine loaded** if you transcribe often. It costs ~70 s to start and
that is paid per run, so on a 3-minute clip it is the whole wall clock — 145 s
cold against 25 s with it already up.

```bash
./engine start      # ~70 s, once
./engine stop       # hand the card back
```

Nothing requires it. Everything uses it when it is there and loads its own
engine when it is not.

**Every command is in [RUNBOOK.md](RUNBOOK.md).**

---

## Speed

Six meetings, 6.27 hours of audio, one batch, default flags. The engine loads
once per batch however many files are in it, so it is listed separately:

| GPU | VRAM | engine load | 16 kHz mono WAV | 44.1 kHz stereo MP3 |
|---|---|---|---|---|
| RTX 5090 | 32 GB | 73 s | **48 s — 468×** | 80 s — 282× |
| RTX 3090 | 24 GB | 63 s | 104 s — 217× | 130 s — 174× |
| RTX 2060 | 6 GB | 114 s | 857 s — 26× | 966 s — 23× |

The MP3 column depends on the host as well as the card — decoding is CPU work.
Two RTX 3090s on different hosts measured 174× and 142× on MP3 but 217× and 203×
on WAV, so treat the MP3 figures as indicative rather than a property of the GPU.

Engine load is ~3× higher the first time on a new machine, while `torch.compile`
fills its cache — and it is paid per run unless you keep the engine up, which is
what `./engine start` above is for.

---

## Using a rented GPU

`--host` means one thing: where the compute happens. Your audio, library and
voice profiles stay on your machine.

**Install it on the box too** — `--host` needs the pipeline at the other end:

```bash
ssh mybox 'git clone https://github.com/suzuenhasa/meetscribe-cli.git /opt/meetscribe \
           && cd /opt/meetscribe && ./setup.sh'
```

Then from your laptop:

```bash
cp ~/recordings/*.mp3 inbox/
./transcribe --host mybox
```

Audio goes up, meetings come back, and `speakers.db` travels both ways so the
box recognises your people and anything it learns comes home. Destroy the
instance and you lose nothing.

For a rented box from nothing, `vast/provision.sh` does the whole install
unattended — see [vast/README.md](vast/README.md).

---

## How it works

```
audio ──► MOSS on vLLM        text + per-window speaker labels
             ▼
      WeSpeaker ResNet293-LM  a voice embedding per segment
             ▼
      constrained clustering  Speaker 1, Speaker 2, …
             ▼
      profile store (sqlite)  Dana Whitfield, Marcus Elle
```

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
labels speakers *within* each window, but window 4's "S01" is not necessarily
window 5's. [WeSpeaker](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)
embeddings link them across a recording, and the profile store puts names on the
result — across recordings, and across months.

---

## Requirements

NVIDIA GPU, **6 GB VRAM or more**, compute capability 7.0+ (Turing, GTX 16-series
and RTX 20-series onward). Pascal and older will not run vLLM. Plus `ffmpeg`.

A 6 GB card transcribes and embeds in two passes rather than at once — slower,
but complete.

Everything lands inside the checkout, so deleting the folder removes the install.
`./setup.sh --check` verifies without changing anything.

`setup.sh` assumes an FHS-style Linux. On NixOS and other non-FHS distributions
use your own Python and expect to set library paths for the NVIDIA driver, GCC
runtime and zlib, plus `TRITON_LIBCUDA_PATH`.

---

## Layout

```
inbox/                drop recordings here; empties as they finish
library/              one directory per meeting
speakers.db           who people are — the one thing not rebuildable from audio

transcribe            the inbox, a folder, or one file
speakers              meetings / who / play / name / apply / list / forget
engine                start / stop the resident engine
setup.sh              installs everything; --check verifies
```

`speakers.db` is the only thing here that cannot be rebuilt from the audio.
Back it up.

---

**[RUNBOOK.md](RUNBOOK.md)** — every command, every flag, and what to do when
something is wrong.
