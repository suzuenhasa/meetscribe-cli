# Runbook

Every command, what it does, and when you would reach for it.

- [Install](#install)
- [Transcribing](#transcribing)
- [Naming voices](#naming-voices)
- [The resident engine](#the-resident-engine)
- [Working on a rented GPU](#working-on-a-rented-gpu--optional)
- [The library on disk](#the-library-on-disk)
- [Maintenance](#maintenance)
- [Environment](#environment)
- [When something is wrong](#when-something-is-wrong)

---

## Install

```bash
git clone https://github.com/suzuenhasa/meetscribe-cli.git
cd meetscribe-cli
./setup.sh
```

| command | does |
|---|---|
| `./setup.sh` | installs everything: interpreter, vLLM, ~2 GB of weights, `speakers.db`, `inbox/`, `library/`. Idempotent — re-run any time |
| `./setup.sh --check` | verifies an install without changing anything |
| `./setup.sh --help` | what it will do before it does it |

Needs an NVIDIA GPU with **6 GB VRAM or more** and compute capability 7.0+, plus
`ffmpeg`. Everything lands inside the checkout: delete the folder, the install is
gone.

---

## Transcribing

```bash
cp ~/recordings/*.mp3 inbox/
./transcribe
```

| command | does |
|---|---|
| `./transcribe` | processes everything in `inbox/`, moving each file into `library/` as it completes |
| `./transcribe ~/audio/` | that folder instead. **Copies** rather than moves — only the inbox is a worklist |
| `./transcribe one.mp3` | a single file |
| `./transcribe --host msbox` | the GPU work happens on that box; everything else stays here |

Accepts `wav mp3 m4a flac ogg opus aac m4b aiff wma mp4 webm mkv` at any sample
rate. Anything not already 16 kHz mono is converted first, all files at once.

**Use a folder, not a loop.** The engine loads once for the whole queue, windows
are pooled across recordings, and each meeting's embedding overlaps the next
one's transcription. Six podcasts as one batch measured ~3 minutes against ~13
run individually.

Every run ends with **where the work went** — busy time per phase, largest
first. Phases overlap (decoding runs while the embedder works in another
process), so they sum to more than the elapsed time; the point is the ranking,
not the total. It is how you tell a slow run from a slow machine.

### Flags

| flag | default | |
|---|---|---|
| `--host <ssh>` | — | run the GPU work over there. Nothing else changes |
| `--out <dir>` | `library/` | keep meetings somewhere else |
| `--replace <id>` | — | redo a meeting in place, keeping its id and history |
| `--glossary "A,B"` | — | proper nouns the model has never heard |
| `--roster "A,B"` | — | only match against these people |
| `--name G02="Bob"` | — | name a cluster during the run |
| `--window <s>` | `30` | seconds of audio per decode window |
| `--overlap <s>` | `0` | context decoded either side of each window |
| `--thr <n\|auto>` | `auto` | speaker-clustering cut |
| `--gpu-frac <n>` | auto | share of VRAM vLLM reserves |
| `--per-speaker <n>` | `2` | segments embedded per speaker per window; `0` for all |
| `--min-core <s>` | `2.0` | speech needed to join the clustering core |
| `--durable <s>` | `6.0` | speech behind a binding "different people" claim |
| `--guard <n>` | `10` | windows a "different people" claim is trusted across |
| `--min-cluster-sec <s>` | `10` | below this a cluster is absorbed, not kept |
| `--refine <n>` | `3` | leave-one-out refinement passes |

`glossary.txt` in the directory you run from is picked up automatically, one
term per line, `#` for comments.

`--glossary` takes either terms or **a file of them**, which is how you keep one
per subject — a crypto interview and an archaeology seminar share no vocabulary:

```bash
./transcribe --glossary crypto.txt
./transcribe --glossary ~/glossaries/archaeology.txt
./transcribe --glossary "Dana Whitfield,Northwind"
```

A path-like name that is not a file is an error rather than a term, so a typo
says so instead of quietly telling the model to listen for "crypto.txt".

**`--overlap <s>`** decodes extra audio either side of each window as context.
It is off by default. Measured across 7 recordings, 7.94 hours, `5` against `0`:
the transcripts matched to within 0.05% on words, while `5` decoded 33% more
audio, left twice the holes, and emitted 21× as many segments overlapping each
other in time. Raise it only if a boundary is cutting something you need.

**`--per-speaker <n>`** caps how many segments per speaker get a voice embedding
**in each window**, so it is coupled to `--window`: raising the window without
raising this embeds less of the recording. Measured on one 27.6-minute meeting,
the share of speech that reached the clusterer was 60% at `--window 30`, 23% at
`120` and 12% at `300`, and the speaker count went 3, then 4, then 10 against a
true 2. If you raise `--window`, scale this with it, or pass `0` to embed
everything.

**`--durable <s>`** is how much speech MOSS must have heard before "these two are
different people" is treated as binding. There is no single right value: on
close-talking audio a low one scores better, and on a far-field mic array a high
one does — measured, they move in opposite directions. The default suits
far-field; lower it toward `2` for headset or single-speaker-per-track audio.

**`--replace`** is for redoing a meeting you already have — a better glossary, a
different `--thr`. It keeps the id, so every decision ever recorded about it
still applies. Without it, the same audio twice is two meetings.

---

## Naming voices

Everyone starts as `Speaker 1`, `Speaker 2`. Those are correct within a
recording and meaningless across recordings. Name someone once and they are
recognised everywhere, including in meetings you already have.

| command | does |
|---|---|
| `./speakers meetings` | everything in the library: folder, id, title |
| `./speakers who <meeting>` | the voices in it, how long each spoke, samples |
| `./speakers play <meeting> G02 [n]` | hear that voice — plays the clips |
| `./speakers name <meeting> G02 "Bob Smith"` | remember this voice |
| `./speakers apply` | which existing transcripts would change |
| `./speakers apply --apply` | re-identify and re-render them |
| `./speakers list` | who is on file |
| `./speakers rename <id> "New Name"` | fix a name |
| `./speakers forget <id>` | delete a person and their voiceprints |

`<meeting>` is anything that identifies one: its id (`9ajq9`), its directory,
a path, or enough of the title to be unambiguous.

### The normal loop

```bash
./speakers who platform-review                    # which cluster is who
./speakers play platform-review G02               # confirm by ear
./speakers name platform-review G02 "Dana Whitfield"
./speakers apply --apply                    # backfill everything else
```

`apply` is the one people miss. Identification runs when a recording is
processed, against whoever was enrolled at that moment — so naming someone
afterwards leaves every meeting they are already in still saying Speaker 3 until
you run it. It is cosine arithmetic over centroids already on disk: no GPU, no
re-transcription, seconds for a whole library.

### Reading the identify output

```
identify: 4 voices in this meeting, 3 enrolled candidates
  = G00      612s  Bob Smith              0.952
  ? G01       74s  Ravi Patel             0.478  (2nd 0.443)
    G03       31s  -                      0.201
```

`=` recognised · `?` too close to call, left numbered · blank, nobody on file.

Accepting needs **0.55 and a 0.10 margin** over the runner-up. 0.40–0.55 asks a
person. Below that is treated as a new voice. Enrolling needs 10 s of speech.

If everyone comes out `?` with identical scores, the same person is probably
enrolled twice — see [When something is wrong](#when-something-is-wrong).

---

## The resident engine

The engine costs ~70 s to load and that is paid **per run**. On an hour of audio
it disappears; on a 3-minute clip it is the whole wall clock.

| command | does |
|---|---|
| `./engine start` | load it and leave it running (~70 s, once) |
| `./engine status` | is it up, and is it running the code on disk |
| `./engine stop` | hand the card back |
| `./engine restart` | stop, then start |
| `./engine log [n]` | what it has been doing |

Measured on a 3090, 3-minute clip: **145 s cold, 25 s resident.**

Nothing requires it. Everything uses it when it is there and loads its own engine
when it is not, so starting it is an optimisation and stopping it is safe at any
time — including mid-queue, which finishes on the engine it already has.

It holds VRAM while running and hands the card back after 15 minutes idle,
keeping only the weights, waking in about a second. `MS_ENGINE_IDLE_SLEEP=0`
keeps it resident regardless.

**It caches pipeline code.** The daemon imports `batch.py` once at startup, so
editing pipeline files changes nothing until it restarts. `./engine status` says
so when the files on disk are newer.

---

## Working on a rented GPU — optional

Nothing needs this. It is for machines without a usable card, or for borrowing a
faster one when a queue is worth the setup.

`--host` means one thing: where the compute happens. The **library** is what
stays yours — transcripts, clips and `speakers.db` live here and are the
authoritative copy.

The audio does travel. It is uploaded, transcribed on the box, and remains in
the box's library until the instance is destroyed. Nothing is sent to a
third-party service — it is a machine you rented — but the recording is on it
for the duration, which is worth knowing before pointing this at anything
sensitive.

```bash
cp ~/recordings/*.mp3 inbox/
./transcribe --host msbox
```

The audio goes up, meetings come back as whole directories, and `speakers.db`
travels **both ways** — up before the run so the box identifies against your
people, back after with whatever it decided. Destroy the instance and you lose
nothing.

`~/.ssh/config`:

```
Host msbox
  HostName 1.2.3.4
  Port 44404
  User root
  IdentityFile ~/.ssh/id_ed25519
```

`export MS_HOST=msbox` drops the flag entirely.

| command | does |
|---|---|
| `./speakers list --host msbox` | the box's store rather than yours |
| `./transcribe ... --host msbox` | the GPU work over there |

`who`, `play` and `clips` always run locally — they read your transcripts and
clips.

Setting a box up from nothing: `vast/provision.sh`, see `vast/README.md`.

---

## The library on disk

```
inbox/                          drop recordings here; empties as they finish
library/
  platform-review-network-to-rule-9ajq9/
    meeting.json                id, title, source, duration, coverage
    …-audio.mp3                 your original — deletable
    …-transcript.txt            what you read
    …-transcript.json           per-segment, with speaker labels
    …-embeddings.npz            one voiceprint per segment
    …-clusters.npz              one centroid per speaker — naming needs this
    …-raw.json                  pre-clustering output, kept for re-linking
    …-names.json                cluster → name for this meeting
    clips/G00-1.mp3 …           a few seconds of each voice
speakers.db                     who people are
```

**The slug is for you, the id is for the software.** `speakers.db` records
against the id, so rename the directory, rename the files, reorganise the whole
library — nothing breaks.

**`clips/` is what lets the audio go.** About a megabyte against fifty, cut from
the original, enough to recognise a voice by ear. Delete `*-audio.mp3` when you
are short of space and naming still works. Clips are **never** used to compute a
voiceprint — that only ever comes from the original, and the embedder refuses a
path under `clips/`.

**`speakers.db` is the only thing here that cannot be rebuilt from the audio.**
Back it up.

---

## Maintenance

| command | does |
|---|---|
| `python3 pipeline/library.py` | list meetings: folder, id, title |
| `python3 pipeline/library.py <ref> <field>` | resolve one — `path` `id` `title` `stem` `clusters` `transcript` `text` `audio` |
| `python3 pipeline/relabel.py` | same as `speakers apply` |
| `python3 pipeline/migrate_ids.py` | what a filename-keyed store would become |
| `python3 pipeline/migrate_ids.py --apply` | re-key it, backing up first |

`migrate_ids.py` is a one-off for stores written before meetings had ids. It
copies to `speakers.db.pre-ids`, keeps every old value in a `legacy_name` column,
and **leaves alone** any row whose meeting is not in the library rather than
dropping it.

### Tests

```bash
python3 -m pytest tests/ -q
```

No GPU, no venv, ~15 seconds. Every test pins a defect this repo actually had
and names the commit that fixed it, so a failure tells you which behaviour
regressed rather than only that something did.

### Running the pipeline directly

Rarely needed. Use the interpreter setup chose:

```bash
source env.sh
"$MS_PY" pipeline/batch.py audio.mp3 --library library/ --out-dir /tmp/scratch
```

`batch.py` also takes `--no-convert` (decode inside the run, one file at a time),
`--no-clips`, `--move-audio`, and `--no-overlap-embed`.

---

## Using part of a card

`MS_VRAM_GIB` caps what the pipeline plans for, and **everything downstream
follows**: the fraction vLLM reserves, the embedder's batch size, whether
embedding overlaps transcription or waits for the engine to be released, and
`max_num_seqs`. Setting it to 10 on a 32 GiB card makes the pipeline behave in
every respect as though it were on a 10 GiB card.

```bash
./engine stop
MS_VRAM_GIB=10 ./engine start      # capped
./engine stop
./engine start                     # the whole card again
```

Nothing persists. The budget is read when the engine starts and lives only in
that process, so switching is stop-and-start — no reinstall, no reset, no state
to clean up.

**Put it on `engine start`, not on the run.** That is what sizes the engine.
`./transcribe` reads it too, but only when there is no daemon and it loads its
own engine; with one up, the daemon's budget applies whatever the run says.

Measured on a 24 GiB 3090, same audio both ways:

| | gpu-frac | card | throughput |
|---|---|---|---|
| no cap | 0.69 | 18,054 MiB | 213× |
| `MS_VRAM_GIB=10` | 0.17 | 5,678 MiB | 125× |

**It costs throughput.** Less VRAM is a smaller KV cache, so fewer windows
decode at once — roughly half the speed for a third of the card. That is the
trade being bought: a GPU you can share.

The fraction is rescaled against the real card, because that is what vLLM
measures `gpu_memory_utilization` against — asking for 7 of a 10 GiB budget on a
32 GiB card means telling vLLM 0.22, not 0.7.

Two reasons to want it. **Sharing a GPU**, where taking whatever is free right
now means taking it from whatever starts next. And **reproducing what a small
card does** without owning one — below about 8 GiB the pipeline releases the
engine before embedding rather than running both at once, and that path is
otherwise only testable by renting the hardware.

A budget too small to run at all is refused up front, naming the numbers, rather
than failing inside the allocator.

---

## Environment

| variable | |
|---|---|
| `MS_HOST` | default `--host` |
| `MS_LIBRARY` | where meetings are kept |
| `MS_SPEAKER_DB` | a different profile store |
| `MS_WORK` | the install directory |
| `MS_REMOTE` | the install directory **on the box**, if it cannot be found |
| `MS_FORCE_REMOTE` | use the box even when a local install exists |
| `MS_ENGINE_IDLE_SLEEP` | seconds before the engine hands the card back; `0` never |
| `MS_POST_POOL_MIN` | recordings below which post-processing runs in-process |
| `MS_VRAM_GIB` | cap the VRAM the pipeline plans for; behaves as if the card were that size |
| `MS_MAX_SEQS` | concurrent sequences vLLM will run |
| `MS_PY` | the interpreter that owns the dependencies (set by `setup.sh`) |
| `MS_SPLIT_MIN_S` | recordings longer than this are decoded in parallel ranges; default `1200` |
| `MS_SPLIT_TARGET_S` | seconds per range; default `600` |
| `MS_SPLIT_MAX_PARTS` | ranges at once; default `8`, measured optimum |
| `MS_SPLIT_RUN_UP_S` | discarded decoder run-up before each range; default `2` |
| `MS_UNPINNED` | `setup.sh` takes current HEAD and latest instead of the pinned set |

---

## When something is wrong

**A voice is enrolled but old transcripts still say Speaker 3.**
`./speakers apply --apply`.

**Everyone comes out `?` with scores of 1.000.** The same person is enrolled
twice: two identical voiceprints, so the 0.10 margin can never be met.
`./speakers list` shows it — two entries with near-identical speech time.
`./speakers forget <id>` on the duplicate.

**Everyone is one speaker, or one person appears as several.** `k_est` on the
`CLUSTER` line is the count found. `FLOOR-VIOLATION` means fewer speakers than
the model heard talking simultaneously — a bug, not a tuning problem.
`LOW-SEPARATION` means the count is weakly supported; check before naming.

**A name is mangled.** Add it to `glossary.txt`. Still wrong near a window
boundary? `--overlap 10`.

**`STOPPED EARLY`.** The transcript ends before the speech does — the model quit
partway. The recording is listed with what it covered, and the run exits
nonzero. The audio and embeddings are kept, so `./transcribe --replace <id>`
redoes it without losing the meeting's history.

**`recovered N segment(s) that fell through a window seam`.** Normal, and only
appears with `--overlap` above 0. Two windows each decided the other owned a
segment straddling their boundary; this puts it back. Zero on the default.

**`CANNOT-LINK-REPAIR`.** The model heard two people in one window and the
clustering still put them together; one has been moved to its own cluster. Rare,
and the transcript is written normally. Frequent occurrences on the same
recording are worth reporting.

**Wrong person recognised.** `--roster` narrows the gallery. False accepts grow
with gallery size.

**vLLM will not start.** Something else holds the card — one vLLM at a time.
`nvidia-smi` will show it. If the engine died badly, an orphaned
`VLLM::EngineCore` may still hold the memory; `./engine start` reaps those.

**Edits to pipeline code do nothing.** The resident engine is running the code it
loaded at startup. `./engine restart`.

**`ModuleNotFoundError`.** Use the right interpreter:
`source env.sh && "$MS_PY" …`.

**Transcripts are not coming back from `--host`.** Check the box has an install
where `transcribe` expects: `/etc/meetscribe-work`, or set `MS_REMOTE`.

**Anything else** — `./setup.sh --check`.
