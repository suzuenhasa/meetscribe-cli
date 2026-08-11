# meetscribe

Meeting audio in, speaker-attributed transcript out.

```bash
./transcribe "Product Marketing Meeting (weekly) 2021-06-28.mp3"
```

```
Product Marketing Meeting (weekly) 2021-06-28
42.7 min · 612 turns · 7 speakers
==================================================================

[0:00:00] Speaker 1
  I can record.
  And we don't have a ton of items to get to.
  So corporate events. I put this in Slack and I saw a little bit of kind of
  noise around it, which was good.

[0:01:12] Speaker 3
  I thought we had in Slack sort of farmed each one of them out.
```

Runs on your own hardware. A 74-minute recording takes about 25 seconds. Name a
voice once and it is recognised in every meeting after that.

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

## Naming voices

By default everyone is `Speaker 1`, `Speaker 2` — correct within one recording,
but Monday's "Speaker 1" has nothing to do with Tuesday's. Name someone once and
they are recognised in every meeting after that.

```bash
transcribe standup.mp3            # Speaker 1, Speaker 2, Speaker 3
speakers name standup G02 "Bob Smith"
transcribe "next week.mp3"        # Bob Smith
```

```
identify: 4 voices in this meeting, 3 enrolled candidates
  = G00      612s  Bob Smith              0.952
  = G02      388s  Jane Doe               0.871
  ? G01       74s  Ravi Patel             0.478  (2nd 0.443)
    G03       31s  -                      0.201
```

`=` recognised, `?` too close to call, blank is nobody on file. A voice is only
matched if it clears **0.55** *and* beats the runner-up by **0.10**; between 0.40
and 0.55 a person decides; below that it is treated as new. Those numbers are
measured for centroid-to-centroid comparison and are not valid at any other
level — a threshold fitted on clip-to-clip produced 8 false accepts out of 30
when applied here.

```bash
speakers list                       # who is on file
speakers meetings                   # what you can name voices from
speakers name <meeting> G02 "Name"
speakers rename <id> "New Name"
speakers forget <id>                # delete a person and their voiceprints
```

Acceptance is a max over the whole gallery, so false accepts grow with its size.
If you know who is in the room, say so — scoring 3 people is far safer than 500:

```bash
transcribe standup.mp3 --roster "Bob Smith,Jane Doe,Ravi Patel"
```

You can also name someone during the run:

```bash
transcribe standup.mp3 --name G02="Bob Smith"
```

The store is one SQLite file, `speakers.db`, created by `setup.sh` beside the
pipeline. It holds one voiceprint per person — a 256-dimension centroid averaged
over every meeting they have been named in — plus a log of every match decision.
It is the only state here that cannot be rebuilt from the audio, because it holds
the names a person typed. Back it up.

---

## Options

The defaults are measured. Change them only with a reason.

| flag | default | |
|---|---|---|
| `--glossary` | — | proper nouns, comma-separated |
| `--window` | `30` | seconds per transcription window |
| `--overlap` | `5` | context each side of a window |
| `--thr` | `auto` | speaker-clustering cut |
| `--roster` | — | restrict matching to these people |
| `--name` | — | `G02="Bob Smith"` — remember this voice |
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
  speakers.py             the profile store (sqlite)
  identify.py             match a meeting's voices against it
speakers                the command for naming, renaming, forgetting
```
