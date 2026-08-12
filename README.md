# meetscribe

Meeting audio in, speaker-attributed transcript out. Runs on your own GPU.

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

## Install

On a machine with an NVIDIA GPU and `ffmpeg`:

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli
./setup.sh
```

It installs into `/workspace` when that is writable — the convention on rented
GPU boxes — and `~/meetscribe` otherwise. Set `MS_WORK` to choose.

Ten minutes or so, mostly downloading ~2 GB of weights. It installs vLLM,
MOSS-Transcribe-Diarize, WeSpeaker ResNet293-LM, the pipeline, and an empty
speaker database.

If the machine has no torch, it builds a virtualenv at `$MS_WORK/venv` and
installs vLLM into that — nothing to set up first. If torch is already there
(most GPU images), it installs alongside it rather than pulling a second copy,
because a duplicate torch shadows the working one and fails confusingly.

Re-run it any time — it is idempotent, and needed again after a container
recycle if your box wipes `/workspace`.

```bash
./setup.sh --check      # verify an install, change nothing
```

**Smaller cards.** The model is 0.9B: ~1.7 GiB of weights plus ~0.7 GiB overhead,
so 8 GB works — pass `--gpu-frac 0.80` for a single file, `0.65` for a folder, so
the concurrent embedder has room. Pre-Ampere cards (GTX 10xx, RTX 20xx, compute
capability under 8.0) cannot do bfloat16; the engine detects that and uses
float16, which costs nothing here. Expect roughly a third of a 3090's throughput
on a 2070.

---

## Use

```bash
cd ~/recordings          # wherever your audio is
~/meetscribe-cli/transcribe .
```

Transcripts land in the directory you run from — a `.txt` to read and a `.json`
with every segment, timestamp and speaker.

One file at a time works too:

```bash
transcribe "weekly sync.mp3"
```

Formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `mp4`, `webm`. Any sample rate.

**Point it at a folder whenever you have more than one file.** The engine takes
~66 s to load, and a folder pays that once for the whole batch instead of per
file, while each meeting's embedding overlaps the next one's transcription:

```
engine resident after 68.9s — 6 meetings queued
  startup            68.9s  (once, not per meeting)
  transcribe+embed   107.3s  [overlapped]
```

Shorter to type — add to `~/.bashrc`:

```bash
alias transcribe='/workspace/meetscribe-cli/transcribe'
alias speakers='/workspace/meetscribe-cli/speakers'
```

---

## Names it has never heard

Proper nouns the model doesn't know are not misspelled, they are **replaced by
similar-sounding English**. `"I'm Sreeram"` came out as `"I'm sure I'm a, I'm"`;
`"EigenCloud"` became `"Igen Club"`. Search-and-replace can't fix that safely,
because the output is valid English.

Tell the decoder the words exist. Drop a `glossary.txt` in the directory you run
from — picked up automatically, no flag:

```
# one term per line, # for comments
Sreeram Kannan
EigenCloud
EigenLayer
```

On a 32-minute podcast this took proper nouns from almost all wrong to 51 of 52
correct, and it does not insert the terms into audio that lacks them.

Inline, for a one-off:

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

```bash
speakers who "standup.json"
```

```
standup   3.0 min   3 voices

  G02     2.0 min   32 turns
       [0:01:55] And what Ethereum did is expand or modularize the system so that...
  G00     0.3 min   3 turns
       [0:00:10] Welcome to CSX Week 3. This is Infrastructure Week and we have...
```

Usually the content gives it away. If not, `speakers clips` prints more of what
one voice said, and `speakers play` plays the clearest samples — though that
needs an audio device, so it works on a machine you're sitting at, not over ssh.

```bash
speakers clips "standup.json" G02      # timestamps + text, works anywhere
speakers play  "standup.json" G02      # needs an audio device
```

### What it reports

```
identify: 4 voices in this meeting, 3 enrolled candidates
  = G00      612s  Bob Smith              0.952
  = G02      388s  Jane Doe               0.871
  ? G01       74s  Ravi Patel             0.478  (2nd 0.443)
    G03       31s  -                      0.201
```

`=` recognised, `?` too close to call, blank is nobody on file. A voice is
matched only if it clears **0.55** *and* beats the runner-up by **0.10**; between
0.40 and 0.55 a person decides; below that it is treated as new.

Acceptance is a max over the whole gallery, so false accepts grow with its size.
If you know who is in the room, say so:

```bash
transcribe standup.mp3 --roster "Bob Smith,Jane Doe,Ravi Patel"
```

Managing the store:

```bash
speakers list                     # id, name, sessions, speech on file
speakers meetings                 # what you can name voices from
speakers rename 3 "Robert Smith"
speakers forget 3                 # delete a person and their voiceprints
```

`speakers.db` sits beside the pipeline. One voiceprint per person — a
256-dimension centroid averaged over every meeting they've been named in — plus
a log of every match decision. **It is the only thing here that cannot be rebuilt
from the audio**, because it holds names a person typed. Back it up.

A voice needs **10 seconds** of speech to enroll; that is where the accuracy
curve flattens (99.55% top-1, and two more minutes buys 0.2 points).

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

**`--window`**: longer is *slower* and no more accurate. 60 s and 90 s were both
measured — throughput falls from 437× realtime to 315× and 254×, and neither
fixes anything. Fewer, longer prompts batch worse.

**`--overlap`**: gives each window context across its boundaries, which is what
fixes a name landing on a seam. It costs about 1.6× throughput, so `--overlap 0`
is worth considering on a large bulk run where names matter less.

**`--thr`**: this used to be a hardcoded constant, and a wrong value merged every
speaker into one *silently* — a confident, wrong, single-speaker transcript. It
now derives itself per recording. Pinning it by hand re-introduces that failure.

Environment: `MS_WORK` (install location), `MS_SPEAKER_DB` (profile store path).

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
candidates. If two people score close the run marks it `?` and leaves them
numbered rather than guessing.

**`ModuleNotFoundError: numpy`** when calling the pipeline scripts directly.
`setup.sh` installs into whichever python already owns torch, not necessarily
bare `python3`. Use the one it recorded:

```bash
source /workspace/env.sh
"$MS_PY" /workspace/batch.py /workspace/inbox/*.mp3 --out-dir /workspace/out
```

**vLLM won't start**, opaque engine error. Usually the process budget:
`supervisorctl stop ray`. If it says free memory is near zero, something else is
already holding the GPU — the box fits one vLLM at a time.

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
transcribe              one file or a folder
speakers                who / play / clips / name / rename / forget
setup.sh                installs everything; --check verifies
glossary.txt.example    copy to glossary.txt beside your recordings
preview.py              backs speakers who / play / clips

pipeline/               deployed to $MS_WORK by setup.sh
  transcribe_meeting.py   MOSS on vLLM, windowed, one file
  batch.py                a queue, engine resident, embedding overlapped
  cluster_speakers.py     constrained clustering + self-calibrated cut
  speakers.py             the profile store (sqlite)
  identify.py             match a meeting's voices against it
  mktxt.py                readable transcript
  link/embed_batched.py   WeSpeaker embeddings, batched
  link/link.py            segments -> global speakers
```

`transcribe` and `speakers` can also drive a separate GPU box over ssh
(`--host`), uploading only the audio; `./setup.sh <sshhost>` installs there.
