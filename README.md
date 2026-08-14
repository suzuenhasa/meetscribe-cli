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

Name a voice once and it is recognised in every meeting after that.

## Speed

Six meetings, 6.27 hours of audio, one batch, default flags. The engine loads
once per batch however many files are in it, so it is listed separately:

| GPU | VRAM | engine load | 16 kHz mono WAV | 44.1 kHz stereo MP3 | format gain |
|---|---|---|---|---|---|
| RTX 5090 | 32 GB | 73 s | **48 s — 468×** | 80 s — 282× | 1.65× |
| RTX 3090 | 24 GB | 63 s | 104 s — 217× | 130 s — 174× | 1.25× |
| RTX 2060 | 6 GB | 114 s | 857 s — 26× | 966 s — 23× | 1.13× |

The MP3 column depends on the host as well as the card — decoding is CPU work
and it lands inside the same timer. Two RTX 3090s on different hosts measured
174× and 142× on MP3 but 217× and 203× on WAV, so treat the MP3 figures as
indicative rather than a property of the GPU. Engine load is also ~3× higher the
very first time on a new machine, while `torch.compile` fills its cache.

**Feeding it 16 kHz mono is free speed, but how much depends on your GPU.**
Decoding MP3 and resampling 44.1 kHz down to the 16 kHz the model wants is real
work and comes out of the same wall clock — so the faster the card, the larger a
share of the total that decode represents. It is worth 1.65× on a 5090 and only
1.13× on a 2060, where the GPU is the bottleneck anyway. If you have a fast card
and a library to get through, convert once:

```bash
ffmpeg -i meeting.mp3 -ac 1 -ar 16000 -c:a pcm_s16le meeting.wav
```

On a small card it is barely worth the disk space.

`--overlap 0` is a further ~1.5× on top, but it is the one setting here that
trades accuracy for speed — see Options.

### Keeping the engine loaded

That engine-load column is paid once per *run*, not once per machine. On a long
batch it disappears into the total. On a short recording it **is** the total —
a 3-minute clip on a 3090 spends 1.4 s transcribing and about 140 s loading the
thing that transcribes it.

So don't make it load:

```bash
./engine start          # ~70 s, once
./transcribe memo.mp3   # 25 s, where it was 145 s
./engine stop           # hand the card back
```

Everything uses it when it is there and loads its own engine when it is not, so
this is only ever an optimisation: `./transcribe` behaves the same either way,
and stopping it is safe at any time, including mid-queue — a running
job finishes on the engine it already has.

It holds VRAM while it runs. After 15 minutes idle it hands the card back and
keeps only the weights, waking in about a second when the next job arrives; set
`MS_ENGINE_IDLE_SLEEP=0` to keep it resident regardless. It serves one job at a
time, and only the `--window`/`--overlap` it was started with — a run asking for
different ones is told why and loads its own.

---

## Install

Needs an NVIDIA GPU and `ffmpeg`. **6 GB VRAM or more**, and compute capability
7.0+ (Turing, GTX 16-series and RTX 20-series onward) — Pascal and older will not
run vLLM. A 6 GB card transcribes and embeds in two passes rather than at once,
which is slower but complete.

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli
./setup.sh
```

Ten minutes, mostly ~2 GB of weights. Everything lands inside the checkout, so
deleting the folder removes the install. Re-run any time; `./setup.sh --check`
verifies without changing anything.

`setup.sh` assumes an FHS-style Linux — it fetches a prebuilt Python and expects
the usual loader paths. On NixOS and other non-FHS distributions it will not run
as-is: use your own Python, and expect to set library paths for the NVIDIA
driver, GCC runtime and zlib, plus a compiler and `TRITON_LIBCUDA_PATH` for
Triton. Everything works there once those are supplied — it is a packaging
assumption, not a limitation.

---

## Use

```bash
./transcribe ~/recordings/              a folder
./transcribe "weekly sync.mp3"          one file
```

Writes a `.txt` and a `.json` into the directory you run from. `wav mp3 m4a flac
ogg opus aac m4b aiff wma mp4 webm mkv`, any sample rate — anything not already
16 kHz mono is converted first, in parallel, which is most of the speed above.

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
| `--gpu-frac` | auto | VRAM vLLM reserves |
| `--host` | — | run the GPU work on a remote box over ssh |

The defaults are measured, not guessed. `--window` longer is slower *and* no more
accurate. `--thr` used to be a constant and a wrong value silently merged every
speaker into one; it now derives itself per recording. `--gpu-frac` likewise: the
engine's cost is a fixed ~2.6 GB regardless of card, so any single fraction is
wrong somewhere — it reserves what the speaker embedder needs alongside and gives
vLLM the rest. You should not need to set it.

`--overlap` is the one real trade here. Each 30 s window is decoded with 5 s of
context on both sides, so a word landing on a boundary is transcribed from a
window that can hear the whole of it. Setting `--overlap 0` is about 1.5× faster
and mangles words at the seams. Keep it unless throughput matters more than the
transcript.

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
engine                start / stop the resident engine, so runs skip the load
setup.sh              installs everything; --check verifies
preview.py            backs speakers who / play / clips
pipeline/             the pipeline itself, run in place
```
