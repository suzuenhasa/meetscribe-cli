#!/usr/bin/env python3
"""The library: one directory per meeting, and the names inside it.

Everything about a meeting lives in one folder and everything about people lives
in one file beside them, so "where is my stuff" has a single answer.

    library/
      platform-review-weekly-q3-k49fj/
        meeting.json                                 id, title, source, dates
        platform-review-weekly-q3-audio.mp3
        platform-review-weekly-q3-transcript.txt
        platform-review-weekly-q3-transcript.json
        platform-review-weekly-q3-embeddings.npz
        platform-review-weekly-q3-clusters.npz
        clips/G00-1.mp3 ...
      speakers.db

THE FOLDER CARRIES IDENTITY, so filenames inside do not have to be unique across
the library. That is what removes the whole _raw/_linked/_emb prefix scheme, and
with it the class of bug where two recordings sanitised to one name and shared a
transcript with another meeting's embeddings.

NOTHING DEPENDS ON EITHER NAME. The id in meeting.json is what speakers.db keys
on, so a folder can be renamed, moved, or tidied up by hand and every decision
ever recorded about that meeting still points at it. The names are for you.
"""
import json
import os
import random
import re
import string
from pathlib import Path

# 5 words or 40 characters, whichever comes first. Long enough to recognise a
# meeting, short enough that the path is not the whole title -- "Platform
# Review: What Changes Next Quarter, Staffing and More | Weekly"
# is 88 characters and would be repeated on every file in the folder.
SLUG_WORDS = 5
SLUG_CHARS = 40
# 8, not 5. speakers.db keys on this and find() returns the first meeting whose
# id matches, so two meetings sharing one is not a cosmetic clash -- it is two
# recordings sharing a history that belongs to neither, which is the exact bug
# the ids were introduced to end.
#
# 31^5 is 28.6M, and the birthday bound over a library is not remote: 1.7% at
# 1,000 meetings, 6.7% at 2,000, 35% at 5,000. 31^8 is 852 billion, which puts
# 5,000 meetings at about 1 in 68 million. Three characters, and the name stays
# short enough to read aloud.
ID_CHARS = 8
# No 0/1/l/o: these get read aloud and typed by hand.
ID_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"


def slug(title):
    """A short, readable, filesystem-safe form of a meeting's title.

    Deliberately NOT stripping stopwords: "the-next-5-years-will-change" is
    recognisable at a glance and "next-5-years-change-bitcoin" is a puzzle. The
    slug is for finding things by eye, so keeping the words in the order a person
    said them matters more than packing information in.
    """
    t = re.sub(r"[^\w\s-]", " ", str(title).lower(), flags=re.UNICODE)
    words = [w for w in re.split(r"[\s_-]+", t) if w]
    out = []
    for w in words[:SLUG_WORDS]:
        if out and len("-".join(out + [w])) > SLUG_CHARS:
            break
        out.append(w)
    return "-".join(out) or "meeting"


def new_id(rng=random):
    """A short random id, minted once when a meeting enters the library.

    Random rather than a hash of the audio: processing the same recording twice
    is two meetings unless you said --replace, and a content hash would make
    that ambiguous rather than explicit."""
    return "".join(rng.choice(ID_ALPHABET) for _ in range(ID_CHARS))


def folder_name(title, mid):
    """<slug>-<id>. The id goes LAST.

    Id-first sorts the library randomly -- k49fj-, b2m7x-, q8w1n- -- so you
    cannot scan it, related meetings do not sit together, and tab completion
    needs five characters nobody remembers. Id-last keeps alphabetical browsing
    and `cd eigen<tab>`, and still makes the name unique by construction rather
    than by a collision check."""
    return f"{slug(title)}-{mid}"


class Meeting:
    """One meeting's directory, and the names of the things in it."""

    def __init__(self, path):
        self.path = Path(path)

    # -- the files ---------------------------------------------------------
    # Slug-prefixed rather than bare. Inside the folder the prefix is redundant,
    # but transcripts LEAVE the folder constantly -- emailed, dropped into a
    # document, saved to a desktop -- and a file called transcript.txt tells the
    # person who receives it nothing at all.
    @property
    def stem(self):
        return self.path.name.rsplit("-", 1)[0]

    def file(self, what, ext):
        return self.path / f"{self.stem}-{what}.{ext}"

    @property
    def meta_path(self):
        return self.path / "meeting.json"

    @property
    def clips_dir(self):
        return self.path / "clips"

    def audio(self):
        """The original, if it is still here. It is allowed not to be: clips are
        enough to name a voice by, so the source can be pruned."""
        for f in sorted(self.path.glob(f"{glob_escape(self.stem)}-audio.*")):
            return f
        return None

    # -- the metadata ------------------------------------------------------
    def read(self):
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return {}

    def write(self, **fields):
        d = self.read()
        d.update(fields)
        self.meta_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return d

    @property
    def id(self):
        # From the file, not the folder name, so renaming the folder cannot
        # detach a meeting from its history.
        return self.read().get("id") or self.path.name.rsplit("-", 1)[-1]

    @property
    def title(self):
        return self.read().get("title") or self.stem


def glob_escape(s):
    return re.sub(r"([\[\]?*])", r"[\1]", s)


def library_dir(work=None):
    """Where meetings live. -> Path

    Precedence, matching what ./transcribe already computes at its :259 --
    `--out` if the run was sent somewhere, else MS_LIBRARY, else the working
    directory's library/:

        --out <dir>   this run goes there instead
        MS_LIBRARY    a library that is not under MS_WORK
        MS_WORK       the usual case

    This used to read MS_WORK/library and nothing else, so a run sent elsewhere
    with --out was WRITTEN correctly and then invisible to everything that reads:
    `link --apply` indexed nothing, `review` and `meetings` showed an empty
    library, and the meeting existed on disk the whole time. The wrapper knew the
    rule and the library did not.
    """
    if work:
        return Path(work) / "library"
    lib = os.environ.get("MS_LIBRARY")
    if lib:
        return Path(lib)
    return Path(os.environ.get("MS_WORK") or
                Path(__file__).resolve().parent.parent) / "library"


def create(title, source_name, lib=None, mid=None):
    """Make a new meeting directory. -> Meeting

    The id is minted here and written before anything else, so a meeting has an
    identity from the moment it exists rather than acquiring one once it has
    been transcribed successfully."""
    lib = Path(lib) if lib else library_dir()
    lib.mkdir(parents=True, exist_ok=True)
    # Against every id in the library, NOT against the resulting path. The folder
    # name carries the slug too, so `cats-abcde` and `dogs-abcde` are two distinct
    # paths and the old check waved both through -- while speakers.db keyed on
    # `abcde` for both and find() returned whichever it met first. The one thing
    # the id has to be is unique, and that was the one thing not being tested.
    taken = {m.id for m in all_meetings(lib)}
    mid = mid or new_id()
    while mid in taken or (lib / folder_name(title, mid)).exists():
        mid = new_id()
    m = Meeting(lib / folder_name(title, mid))
    m.path.mkdir(parents=True)
    m.write(id=mid, title=str(title), source=str(source_name))
    return m


def all_meetings(lib=None):
    lib = Path(lib) if lib else library_dir()
    if not lib.is_dir():
        return []
    return sorted((Meeting(p) for p in lib.iterdir()
                   if p.is_dir() and (p / "meeting.json").exists()),
                  key=lambda m: m.path.name)


def find(ref, lib=None):
    """Resolve a meeting from whatever a person typed: an id, a folder name, a
    path, or enough of the title to be unambiguous. -> Meeting or None"""
    ref = str(ref).strip().rstrip("/")
    p = Path(ref)
    if (p / "meeting.json").exists():
        return Meeting(p)
    ms = all_meetings(lib)
    # Ids and folder names are unique by construction, so an exact hit on either
    # is the answer. STEMS are not: the stem is the slug, and two recordings of
    # the same title share it -- this library holds two called
    # one-trust-network-to-rule. Returning the first match silently picked one of
    # them, so `--replace <stem>` could redo the wrong meeting and `speakers name`
    # could name a voice in it. Ambiguity has to come back as None, which is what
    # the title search below already does.
    for m in ms:
        if ref in (m.id, m.path.name):
            return m
    stem_hits = [m for m in ms if m.stem == ref]
    if stem_hits:
        return stem_hits[0] if len(stem_hits) == 1 else None
    low = ref.lower()
    hits = [m for m in ms if low in m.path.name.lower() or low in m.title.lower()]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------- the CLI
# So ./speakers can resolve a meeting without reimplementing any of the above in
# bash. Prints one field per line for whatever it was asked about.
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        for m in all_meetings():
            print(f"{m.path.name}\t{m.id}\t{m.title}")
        raise SystemExit(0)
    m = find(sys.argv[1])
    if m is None:
        raise SystemExit(2)
    what = sys.argv[2] if len(sys.argv) > 2 else "path"
    print({"path": str(m.path), "id": m.id, "title": m.title,
           "stem": m.stem,
           "clusters": str(m.file("clusters", "npz")),
           "transcript": str(m.file("transcript", "json")),
           "text": str(m.file("transcript", "txt")),
           "audio": str(m.audio() or ""),
           }.get(what, str(m.path)))
