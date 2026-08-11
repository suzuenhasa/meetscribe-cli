# meetscribe

Meeting audio in, speaker-attributed transcript out. Runs on your own hardware.

```bash
./transcribe "product sync.mp3"
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

A 74-minute recording takes about 25 seconds. Name a voice once and it is
recognised in every meeting after that.

---

## Commands

```bash
./setup.sh msbox                          install onto a GPU box over ssh
./setup.sh                                install on this machine (needs the GPU)
./setup.sh --check                        verify an install, change nothing

./transcribe "meeting.mp3"                one file
./transcribe ~/recordings/                a folder — engine loads once
./transcribe m.mp3 --glossary "Acme,Bob Smith"
./transcribe m.mp3 --roster "Bob Smith,Jane Doe"
./transcribe m.mp3 --name G02="Bob Smith"

./speakers who "meeting.json"             the voices, and what each said
./speakers play "meeting.json" G02        HEAR that voice
./speakers list                           who is on file
./speakers meetings                       what you can name voices from
./speakers name <meeting> G02 "Bob Smith"
./speakers rename <id> "New Name"
./speakers forget <id>
```

---

## Setup

You need a machine with an NVIDIA GPU (12 GB VRAM is the practical floor) and
`ffmpeg`. Everything else installs itself.

### Onto a rented GPU box

Your audio stays on your laptop; only the file being transcribed is uploaded.
Add the box to `~/.ssh/config`:

```
Host msbox
    HostName 203.0.113.42
    Port 12345
    User root
    IdentityFile ~/.ssh/id_rsa
```

Check it works, then install:

```bash
ssh msbox 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
./setup.sh msbox
```

### On the machine you're sitting at

If that machine has the GPU:

```bash
./setup.sh
```

Either way it takes about ten minutes, mostly downloading ~2 GB of weights, and
installs vLLM, MOSS-Transcribe-Diarize, WeSpeaker ResNet293-LM, the pipeline
scripts, and an empty speaker database.

It is idempotent — re-run it any time, and after a container recycle if you're on
a rented box that wipes `/workspace`.

```bash
./setup.sh --check          # verifies everything, changes nothing
```

Set `MS_WORK` to install somewhere other than `/workspace`.

---

## Transcribing

```bash
cd ~/recordings
transcribe "weekly sync.mp3"
```

Writes `weekly sync.txt` and `weekly sync.json` into the **current directory**.
The JSON has every segment with start/end times and speaker, for anything you
want to build on top.

Formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `mp4`, `webm`. Any sample rate — it
resamples.

Make it a single word — in `~/.bashrc`:

```bash
alias transcribe='~/meetscribe-cli/transcribe'
alias speakers='~/meetscribe-cli/speakers'
```

### A whole folder

```bash
transcribe ~/recordings/
```

```
engine resident after 68.9s — 2 meetings queued

  wrapper test.mp3        3.0 min  transcribed  2.1s   38 segs  coverage 100%
  later meeting.mp3       2.0 min  transcribed  1.6s   33 segs  coverage 100%

2 meetings, 5 min of audio
  startup            68.9s  (once, not per meeting)
  transcribe+embed    7.7s  [overlapped]
```

The engine costs ~66 s to load and that is paid **once** for the batch rather
than once per file, and each meeting's embedding overlaps the next one's
transcription — measured to cost vLLM about 2%, because the two bottleneck on
different things. Ten short meetings go from about eleven minutes of pure startup
to one.

---

## Names it has never heard

Proper nouns the model doesn't know are not misspelled, they are **replaced by
similar-sounding English**. `"I'm Sreeram"` came out as `"I'm sure I'm a, I'm"`;
`"EigenCloud"` became `"Igen Club"`. Search-and-replace can't fix that safely,
because the output is valid English.

Tell the decoder the words exist. Drop a `glossary.txt` in the directory you run
from — it is picked up automatically, no flag:

```
# one term per line, # for comments
Sreeram Kannan
EigenCloud
EigenLayer
```

On a 32-minute podcast this took proper nouns from almost all wrong to 51 of 52
correct. It does not insert the terms into audio that lacks them — that was
checked against a recording containing none of them.

Or inline, for a one-off:

```bash
transcribe podcast.mp3 --glossary "Sreeram Kannan,EigenCloud"
```

---

## Naming voices

By default everyone is `Speaker 1`, `Speaker 2` — correct within one recording,
but Monday's "Speaker 1" has nothing to do with Tuesday's. Name someone once and
they are recognised from then on.

```bash
transcribe standup.mp3                      # Speaker 1, Speaker 2, Speaker 3
speakers name standup G02 "Bob Smith"       # remember that voice
transcribe "next week.mp3"                  # Bob Smith
```

### Working out who G02 is

You cannot name a voice you have not heard. `who` shows every voice with what
they actually said:

```bash
speakers who "wrapper test.json"
```

```
wrapper test   3.0 min   3 voices
audio: wrapper test.mp3

  G02     2.0 min   32 turns
       [0:01:55] And what Ethereum did is expand or modularize the system so that anybody...
       [0:02:43] year old kid in India from a random like place. He wrote this DeFi program...

  G00     0.3 min   3 turns
       [0:00:10] Welcome to CSX Week 3. This is Infrastructure Week and we have a lot of...

  G01     0.3 min   5 turns
       [0:00:34] It's great to talk about what you're up to and also about your journey...
```

Often the text alone gives it away. When it doesn't, listen — it plays the
clearest few clips of that voice straight from your audio:

```bash
speakers play "wrapper test.json" G02        # 3 clips
speakers play "wrapper test.json" G02 6      # 6 clips
speakers clips "wrapper test.json" G02       # just the timestamps, no playback
```

Then name them:

```bash
speakers name "wrapper test" G02 "Sreeram Kannan"
```

`who` and `play` read your local transcript and audio, so keep the recording
beside the `.json`. They need `ffplay` or `mpv` for playback; `ffmpeg` provides
`ffplay`.

Each run reports what it recognised:

```
identify: 4 voices in this meeting, 3 enrolled candidates
  = G00      612s  Bob Smith              0.952
  = G02      388s  Jane Doe               0.871
  ? G01       74s  Ravi Patel             0.478  (2nd 0.443)
    G03       31s  -                      0.201
```

`=` recognised, `?` too close to call, blank is nobody on file. A voice is
matched only if it clears **0.55** *and* beats the runner-up by **0.10**; between
0.40 and 0.55 a person decides; below that it is treated as new. Those numbers
are measured for centroid-to-centroid comparison and are not valid at any other
level — a threshold fitted on clip-to-clip produced 8 false accepts out of 30
when applied here.

Acceptance is a max over the whole gallery, so false accepts grow with its size.
If you know who is in the room, say so — scoring 3 people is far safer than 500:

```bash
transcribe standup.mp3 --roster "Bob Smith,Jane Doe,Ravi Patel"
```

You can also name someone during the run:

```bash
transcribe standup.mp3 --name G02="Bob Smith"
```

Managing the store:

```bash
speakers list                     # id, name, sessions, speech on file
speakers rename 3 "Robert Smith"
speakers forget 3                 # delete a person and their voiceprints
```

The store is one SQLite file, `speakers.db`, created by `setup.sh` beside the
pipeline. It holds one voiceprint per person — a 256-dimension centroid averaged
over every meeting they have been named in — plus a log of every match decision.
**It is the only state here that cannot be rebuilt from the audio**, because it
holds the names a person typed. Back it up.

---

## Options

The defaults are measured. Change them only with a reason.

| flag | default | |
|---|---|---|
| `--glossary` | — | proper nouns, comma-separated |
| `--roster` | — | restrict matching to these people |
| `--name` | — | `G02="Bob Smith"` — remember this voice |
| `--window` | `30` | seconds per transcription window |
| `--overlap` | `5` | context each side of a window |
| `--thr` | `auto` | speaker-clustering cut |
| `--host` | `msbox` | which box to use |

**`--window`**: longer is *slower* and no more accurate. 60 s and 90 s were both
measured — throughput falls from 437× realtime to 315× and 254×, and neither
fixes anything. Fewer, longer prompts batch worse.

**`--thr`**: this used to be a hardcoded constant, and a wrong value merged every
speaker into one *silently* — a confident, wrong, single-speaker transcript. It
now derives itself per recording from the model's own within-window labels.
Pinning it by hand re-introduces that failure.

Environment: `MS_WORK` (install location), `MS_HOST` (default box),
`MS_SPEAKER_DB` (profile store path).

---

## What it does

Transcription is [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
on vLLM. That gives text plus speaker labels *within* each 30-second window — but
window 4's "S01" and window 5's "S01" are not necessarily the same person.

Linking them is the other half: [WeSpeaker ResNet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)
produces a voice embedding per speech segment, those are clustered into actual
speakers across the recording, and each cluster is matched against the profile
store to put a name on it.

```
audio ──► MOSS (vLLM)  ──►  text + per-window speaker labels
             │
             ▼
      WeSpeaker ResNet293-LM  ──►  voice embedding per segment
             │
             ▼
      constrained clustering  ──►  Speaker 1, Speaker 2, …
             │
             ▼
      profile store (sqlite)  ──►  Bob Smith, Jane Doe
```

---

## When something looks wrong

**Everyone is one speaker, or one person appears as several.** Check the
`CLUSTER` line: `k_est` is the speaker count found. A `FLOOR-VIOLATION` warning
means the clustering produced fewer speakers than the model heard talking at
once — a real bug, not a tuning problem.

**A name is mangled.** Add it to `glossary.txt`. If it's still wrong and falls
near a 30-second boundary, try `--overlap 10`.

**Someone is recognised as the wrong person.** Use `--roster` to limit
candidates. If two people genuinely score close, the run marks it `?` and leaves
them numbered rather than guessing.

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
transcribe              one file or a folder, on the machine with your audio
speakers                who / play / name / rename / forget
preview.py              reads the transcript + audio for who / play / clips
setup.sh                installs everything, locally or onto a box over ssh
glossary.txt.example    copy to glossary.txt beside your recordings

pipeline/               deployed to the GPU machine by setup.sh
  transcribe_meeting.py   MOSS on vLLM, windowed, one file
  batch.py                a queue, engine resident, embedding overlapped
  cluster_speakers.py     constrained clustering + self-calibrated cut
  speakers.py             the profile store (sqlite)
  identify.py             match a meeting's voices against it
  mktxt.py                readable transcript
  link/embed_batched.py   WeSpeaker embeddings, batched
  link/link.py            segments -> global speakers
```
