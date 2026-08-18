# The loop

Setup and getting audio into `inbox/` are in the [runbook](RUNBOOK.md). This is
what to do after that, in order, and what to do the next time.

---

## 1. Transcribe

```bash
./transcribe inbox/*.m4a
```

Everyone comes out as `Speaker 1`, `Speaker 2`. Those numbers are correct within
one recording and mean nothing across recordings — `Speaker 2` in Monday's
standup is not `Speaker 2` in Tuesday's.

Transcripts land in `library/<slug>-<id>/`, one folder per meeting, with the
text, the audio, a few clips per voice, and the vectors naming will use.

## 2. Group the voices

```bash
./speakers link --apply
```

This is the step that makes a voice a *person*. It groups clusters across every
meeting in the library, so one group is one human wherever they appear, and
attaches any group that matches someone already named.

Run it after every batch. It is arithmetic over vectors already on disk — no
GPU, no audio, seconds for a whole library.

## 3. See who is worth naming

```bash
./speakers review
```

```
10 voices waiting to be named, 1.6 h of speech (445 people already named)

  group 451: 27 min across 4 meetings
      "Mr. Chief Justice, and may it please the Court. Over fifty years..."
      name it:  ./speakers name 451 "Their Name"
      or attach: ./speakers name 451 --speaker <id>   (see `list`)
```

Ordered by how many meetings each voice spans, because naming one that appears
in twelve fixes twelve transcripts and naming a one-off fixes one. The line of
what they said usually settles it; when it doesn't, there is a clip to play.

## 4. Name them

```bash
./speakers name 451 "Dana Whitfield"        # someone new
./speakers name 451 --speaker 6             # someone already on file
```

**The id is the identity and the name is a label on it.** Retyping a name that
is one character off creates a *second* person and splits their voice between
the two spellings, which nothing downstream can detect — so a near miss is
refused and points you at the id to use instead.

## 5. Backfill everything you already have

```bash
./speakers apply --apply
```

Naming someone does not, by itself, change transcripts you already made. This
re-labels them. Without `--apply` it prints what would change and touches
nothing.

---

## The next batch

```bash
./transcribe inbox/*.m4a      # 1. transcribe
./speakers link --apply       # 2. group, and match against everyone named
./speakers review             # 3. anyone new worth naming?
./speakers apply --apply      # 5. only if you named someone in 3
```

Steps 1 and 2 every time. Steps 3–5 only when there is somebody new — people
already named are recognised on the way in and need nothing.

---

## Per-meeting glossary and roster

A queue is usually a week of different meetings with different people and
different jargon. One `--glossary` for all of them helps the meeting it was
written for and adds noise to the rest.

```bash
./transcribe inbox/*.m4a --manifest week.json
```

```json
{
  "board-sync.m4a":  {"roster": ["Dana Whitfield", "Sam Okafor"],
                      "glossary": "Northwind, NorthwindDA, Kubernetes"},
  "eng-standup":     {"roster": ["Sam Okafor", "Priya Raman"],
                      "glossary": "glossary/eng.txt"},
  "client-call.wav": {"glossary": "Contoso, Fabrikam"}
}
```

- Keyed by filename. The extension is optional — `"eng-standup"` matches
  `eng-standup.wav`.
- `glossary` takes terms, or a path to a file of them one per line.
- `roster` is who could be in *that* recording.
- Anything you leave out falls back to `--glossary` / `--roster`, and a
  recording absent from the manifest entirely is fine.

**Why the roster matters.** Everyone on file who could not possibly be in a
recording is a comparison that cannot be right and can only cost accuracy.
Measured over 300 recordings with 391 people enrolled and about twelve actually
present, a roster cut wrong names from 2.21% to 1.78%. For a meeting the roster
is the calendar invite.

Nothing is forced: someone absent from the roster comes out unnamed rather than
pushed onto the nearest listed name. The flip side is that **a roster will also
suppress someone you just named** if you did not add them to it — if a name you
expect does not appear, check the roster before anything else.

---

## When a name is missing or wrong

```bash
./speakers suggest board-sync --roster "Dana Whitfield,Sam Okafor"
```

```
  UNKNOWN   3s  [6:27]  "For future payments."
       best is 0.16 clear of the next
       dana_whitfield       0.47  #########
       sam_okafor           0.31  ######
       (nobody)                   accept is 0.62
```

Shows what each unresolved voice scored against every candidate. A fragment at
0.59 with the next at 0.26 is a person sitting just under the bar; one at 0.47
with three others inside 0.06 is noise. Both come out unnamed and only one is
worth your time.

No verdict is printed on purpose — tried as one it named the wrong person on the
first fragment it was pointed at.

```bash
./speakers profiles "Dana Whitfield"     # is something filed under her that is not her?
./speakers link --apply                  # re-derive the grouping
```

---

## One person, more than one circumstance

A voice over a conference mic and the same voice on a phone are far apart as
vectors and are the same human being. A person is stored once per circumstance
rather than averaged into a profile matching neither.

```bash
./speakers profiles "Dana Whitfield" --measure
```

```
  auto-1      920 min   171 recordings   [wideband, 7700 Hz, 100% agree]
  auto-6       17 min    18 recordings   [narrowband, 3650 Hz, 100% agree]
      auto-6 is 100% narrowband at 3650 Hz -- a phone or a headset.
          ./speakers profile-rename 6 auto-6 telephone
```

These are found automatically and named `auto-1`, `auto-2` — a counter, which
says a second way of sounding was found and nothing about what it is.
`--measure` opens a few of the recordings behind each and reports the channel.

```bash
./speakers profile-rename 6 auto-6 telephone      # name one, and pin it
./speakers profile-merge 6 auto-3 auto-5          # same circumstance
./speakers profile-split 6 auto-1                 # two circumstances in one
./speakers name 451 "Dana" --condition potato     # declare one yourself
```

Renaming pins a profile: the automatic pass only ever redraws `auto-*`.

---

## Everything else

| command | does |
|---|---|
| `./speakers meetings` | what is in the library |
| `./speakers who <meeting>` | the voices in it, with samples |
| `./speakers play <meeting> G02` | hear one |
| `./speakers list` | who is on file, with ids |
| `./speakers groups` | every voice group, named or not |
| `./speakers rename <id> "New Name"` | fix a name |
| `./speakers forget <id>` | delete a person and their voiceprints |

Flags, tuning, the resident engine, running on a rented GPU, and what to do when
something breaks are in the [runbook](RUNBOOK.md).
