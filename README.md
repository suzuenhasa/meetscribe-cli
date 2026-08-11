# meetscribe

Meeting audio in, speaker-attributed transcript out.

```bash
./transcribe "product sync.mp3"
```

```
product sync
42.7 min · 612 turns · 7 speakers
==================================================================

[0:00:31] Speaker 2
  Sharam, thank you so much for joining us today.

[0:00:33] Speaker 1
  Thank you so much, Ali. My pleasure.
```

Runs on your own hardware. A 74-minute recording takes about 25 seconds.

---

## What it does

Transcription is [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
on vLLM. That gives text plus speaker labels *within* each 30-second window — but
window 4's "S01" and window 5's "S01" are not necessarily the same person.

Linking them is the other half: [WeSpeaker ResNet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)
produces a voice embedding per speech segment, and those are clustered into
actual speakers across the whole recording.

```
audio ──► MOSS (vLLM)  ──►  text + per-window speaker labels
             │
             ▼
      WeSpeaker ResNet293-LM  ──►  voice embedding per segment
             │
             ▼
      constrained clustering  ──►  Speaker 1, Speaker 2, …
```

---

## Install

You need a machine with an NVIDIA GPU (12 GB VRAM is the practical floor) and
`ffmpeg`. Everything else installs itself.

**Renting a GPU box** (audio stays on your laptop, only the file being
transcribed is uploaded):

```bash
# add the box to ~/.ssh/config as e.g. `msbox`, then:
./setup.sh msbox
```

**Installing on the machine you're sitting at**, if it has the GPU:

```bash
./setup.sh
```

Either takes about ten minutes, mostly downloading ~2 GB of weights. It is
idempotent — re-run it any time, and after a container recycle if you're on a
rented box that wipes `/workspace`.

Verify an install without changing anything:

```bash
./setup.sh --check
```

---

## Use

```bash
cd ~/recordings
transcribe "weekly sync.mp3"
```

Writes `weekly sync.txt` and `weekly sync.json` into the current directory. The
JSON has every segment with start/end times and speaker, for anything you want
to build on top.

Make it a single word — in `~/.bashrc`:

```bash
alias transcribe='~/meetscribe-cli/transcribe'
```

Formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `mp4`, `webm`. Any sample rate —
it resamples.

### Names it has never heard

Proper nouns the model doesn't know are not misspelled, they are **replaced by
similar-sounding English**. `"I'm Sreeram"` came out as `"I'm sure I'm a, I'm"`;
`"EigenCloud"` became `"Igen Club"`. Search-and-replace can't fix that safely,
because the output is valid English.

Tell the decoder the words exist. Drop a `glossary.txt` in the directory you run
from — picked up automatically, no flag:

```
# one term per line
Sreeram Kannan
EigenCloud
EigenLayer
```

On a 32-minute podcast this took proper nouns from almost all wrong to 51 of 52
correct. It does not insert the terms into audio that lacks them.

Or inline, for a one-off:

```bash
transcribe podcast.mp3 --glossary "Sreeram Kannan,EigenCloud"
```

---

## Options

The defaults are measured. Change them only with a reason.

| flag | default | |
|---|---|---|
| `--glossary` | — | proper nouns, comma-separated |
| `--window` | `30` | seconds per transcription window |
| `--overlap` | `5` | context each side of a window |
| `--thr` | `auto` | speaker-clustering cut |
| `--host` | `msbox` | which box to use |

**`--window`**: longer is *slower* and no more accurate. 60 s and 90 s were both
measured — they lose throughput (437× realtime → 315× and 254×) and fix nothing.
Fewer, longer prompts batch worse.

**`--thr`**: this used to be a hardcoded constant, and a wrong value merged every
speaker into one *silently* — a confident, wrong, single-speaker transcript. It
now derives itself per recording from the model's own within-window labels.
Pinning it by hand re-introduces that failure.

---

## When something looks wrong

**Everyone is one speaker, or one person appears as several.** Check the
`CLUSTER` line: `k_est` is the speaker count found. A `FLOOR-VIOLATION` warning
means the clustering produced fewer speakers than the model heard talking at
once — a real bug, not a tuning problem.

**A name is mangled.** Add it to `glossary.txt`. If it's still wrong and falls
near a 30-second boundary, try `--overlap 10`.

**vLLM won't start**, with an opaque engine error. Usually the process budget;
`setup.sh` handles it, but if it recurs: `ssh msbox 'supervisorctl stop ray'`.

**Anything else** — `./setup.sh --check` rules out half the causes in seconds.

---

## Two things worth knowing

Both were bugs that took real time to find, and both are invisible from the
output:

**Audio is resampled to 16 kHz.** Every model here expects 16 kHz; every
real-world mp3 is 44.1 kHz. Feeding one to the other silently destroys the voice
embeddings — speaker separability collapsed from d′ 5.99 to 1.82 — while leaving
the transcript itself looking perfectly fine.

**The clustering threshold is derived per recording**, not fitted once. A
constant tuned on a meeting-room corpus gave the right answer there and merged
every speaker into one on real-world audio, without any error.

---

## Layout

```
transcribe              the command you run, on the machine with your audio
setup.sh                installs everything, locally or onto a box over ssh
glossary.txt.example    copy to glossary.txt beside your recordings
pipeline/               deployed to the GPU machine by setup.sh
  transcribe_meeting.py   MOSS on vLLM, windowed
  cluster_speakers.py     constrained clustering + self-calibrated cut
  mktxt.py                readable transcript
  link/embed_batched.py   WeSpeaker embeddings, batched
  link/link.py            segments -> global speakers
```
