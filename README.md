# meetscribe

Meeting audio in, speaker-attributed transcript out. Runs on your own GPU.

```
Finance & Corporate Committee - Zoom Meeting
104.0 min · 1074 turns · 16 speakers
==================================================================

[0:00:00] Speaker 4
  Meeting open, welcome everybody.
  So we'll start with the apologies and I guess we have Hazel, everyone else
  seems to be here, I think. So could I have a mover please?

[0:00:12] Speaker 3
  Andrew, just before you do, I had to leave the meeting at 1.30 to attend a
  future proof meeting. So sorry about that.
```

That took 17.7 s of GPU time. Name a voice once and it is recognised in every
meeting after that.

---

## Install

Needs an NVIDIA GPU and `ffmpeg`. 12 GB VRAM or more — 8 GB does not fit.

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli
./setup.sh
```

Ten minutes, mostly ~2 GB of weights. Everything lands inside the checkout, so
deleting the folder removes the install. Re-run any time; `./setup.sh --check`
verifies without changing anything.

---

## Use

```bash
./transcribe ~/recordings/              a folder
./transcribe "weekly sync.mp3"          one file
```

Writes a `.txt` and a `.json` into the directory you run from. `wav mp3 m4a flac
ogg mp4 webm`, any sample rate.

Use a folder whenever you have more than one file — the engine loads once for the
batch instead of per file, and each meeting's embedding overlaps the next one's
transcription.

---

## Names it has never heard

Unfamiliar proper nouns are not misspelled, they are replaced by similar-sounding
English: `"I'm Sreeram"` became `"I'm sure I'm a, I'm"`. Put a `glossary.txt`
beside where you run — picked up automatically:

```
Sreeram Kannan
EigenCloud
```

Or `./transcribe m.mp3 --glossary "Sreeram Kannan,EigenCloud"`.

---

## Naming voices

Everyone starts as `Speaker 1`, `Speaker 2` — correct within a recording, but
Monday's "Speaker 1" is unrelated to Tuesday's. Name someone once and they are
recognised from then on.

```bash
./speakers who "standup.json"              # the voices, and what each said
./speakers play "standup.json" G02         # hear one (needs an audio device)
./speakers name standup G02 "Bob Smith"    # remember that voice
```

Then every later meeting reports what it found:

```
identify: 4 voices in this meeting, 3 enrolled candidates
  = G00      612s  Bob Smith              0.952
  ? G01       74s  Ravi Patel             0.478  (2nd 0.443)
    G03       31s  -                      0.201
```

`=` recognised, `?` too close to call, blank is nobody on file. Matching needs
0.55 *and* a 0.10 margin over the runner-up; 0.40–0.55 asks a person; below is
treated as new. A voice needs 10 s of speech to enroll.

If you know who is in the room, say so — false accepts grow with gallery size:

```bash
./transcribe standup.mp3 --roster "Bob Smith,Jane Doe"
```

Also: `./speakers list`, `./speakers rename <id> "New Name"`,
`./speakers forget <id>`.

`speakers.db` is the only thing here that cannot be rebuilt from the audio.
Back it up.

---

## Options

| flag | default | |
|---|---|---|
| `--glossary` | — | proper nouns, comma-separated |
| `--roster` | — | restrict matching to these people |
| `--name` | — | `G02="Bob Smith"` |
| `--window` | `30` | seconds per window |
| `--overlap` | `5` | context each side |
| `--thr` | `auto` | speaker-clustering cut |
| `--gpu-frac` | `0.90` / `0.72` | VRAM vLLM reserves |
| `--host` | — | run the GPU work on a remote box over ssh |

The defaults are measured, not guessed. `--window` longer is slower *and* no more
accurate. `--thr` used to be a constant and a wrong value silently merged every
speaker into one; it now derives itself per recording. `--gpu-frac` is a hard
reservation — the batch default is lower because the embedder runs concurrently.

---

## How it works

```
audio ──► MOSS on vLLM        text + per-window speaker labels
             ▼
      WeSpeaker ResNet293-LM  a voice embedding per segment
             ▼
      constrained clustering  Speaker 1, Speaker 2, …
             ▼
      profile store (sqlite)  Bob Smith, Jane Doe
```

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
labels speakers *within* each window, but window 4's "S01" is not necessarily
window 5's. [WeSpeaker](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)
embeddings link them across the recording, and the profile store puts names on
the result.

---

## Troubleshooting

**Everyone is one speaker, or one person appears as several.** `k_est` on the
`CLUSTER` line is the count found. `FLOOR-VIOLATION` means fewer speakers than
the model heard talking at once — a bug, not a tuning issue.

**A name is mangled.** Add it to `glossary.txt`. Still wrong near a 30 s
boundary? `--overlap 10`.

**Wrong person recognised.** Use `--roster`. Close calls are marked `?` and left
numbered rather than guessed.

**vLLM won't start.** Free memory near zero means something else holds the GPU —
one vLLM at a time. Otherwise `supervisorctl stop ray`.

**`ModuleNotFoundError`** running the pipeline scripts directly — use the
interpreter setup chose: `source env.sh && "$MS_PY" pipeline/batch.py ...`

**Anything else** — `./setup.sh --check`.

---

## Layout

```
transcribe            one file or a folder
speakers              who / play / clips / name / rename / forget
setup.sh              installs everything; --check verifies
preview.py            backs speakers who / play / clips
pipeline/             the pipeline itself, run in place
```
