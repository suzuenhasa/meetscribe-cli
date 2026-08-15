"""pipeline/library.py -- names, identity, and resolving a meeting from a ref.

The centre of gravity is commit b9e2659 "Make the meeting id unique". The old
collision check tested whether the resulting PATH existed; the folder name
carries the slug as well as the id, so `cats-abcde` and `dogs-abcde` are two
distinct paths and both were accepted -- while speakers.db keys on `abcde` for
both and find() returns whichever it meets first. Two recordings then share a
history belonging to neither, which is the exact failure ids were introduced to
end. The tests under "the id must be unique" all fail against that old check.

Everything else here defends the surrounding contract that made ids worth
having: the id lives in meeting.json and not in the folder name, the slug is
only a name, and find() either resolves a ref to exactly one meeting or to
nothing.

library.py imports json/os/random/re/string/pathlib and nothing else, so it is
imported directly -- no stubbing, no exec-ing of source text.
"""
import json
import random
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import library as LIB  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """No default-argument call may reach the real library in this checkout.

    library_dir() falls back to MS_WORK and then to the checkout root, and
    find() treats its argument as a path before anything else, so both the
    environment and the cwd have to be neutral.
    """
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    monkeypatch.setenv("MS_WORK", str(work))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def lib(tmp_path):
    """A throwaway library. Deliberately NOT created here: create() has to make
    it, and all_meetings()/find() have to cope with it not existing yet."""
    return tmp_path / "library"


@pytest.fixture
def write_meeting(lib):
    """Write a meeting folder by hand. -> Meeting

    Bypasses create() deliberately: these tests need folders create() would not
    mint today -- legacy 5-character ids, a folder renamed after the fact, a
    meeting.json whose id disagrees with the folder name it sits in.
    """
    def _write(folder, **meta):
        p = Path(lib) / folder
        p.mkdir(parents=True, exist_ok=True)
        meta.setdefault("id", folder.rsplit("-", 1)[-1])
        meta.setdefault("title", folder.rsplit("-", 1)[0].replace("-", " "))
        (p / "meeting.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))
        return LIB.Meeting(p)
    return _write


@pytest.fixture
def three(write_meeting):
    """One distinct meeting, plus two whose folder names share a prefix."""
    return {
        "platform": write_meeting("platform-review-weekly-q3-k49fj",
                                  id="k49fj", title="Platform Review: Weekly Q3"),
        "b2027": write_meeting("budget-planning-2027-mn4p7q2r",
                               id="mn4p7q2r", title="Budget Planning 2027"),
        "b2028": write_meeting("budget-planning-2028-x9wq3v5k",
                               id="x9wq3v5k", title="Budget Planning 2028"),
    }


# ======================================================================
# The id must be unique -- b9e2659
# ======================================================================

def test_two_titles_cannot_share_an_explicit_id(lib):
    """The regression itself. Same id, different titles, different paths.

    Old check: `while (lib / folder_name(title, mid)).exists()`. `dogs-abcde`
    does not exist, so `abcde` was handed out twice.
    """
    cats = LIB.create("Meeting about cats", "cats.mp3", lib=lib, mid="abcde")
    dogs = LIB.create("Meeting about dogs", "dogs.mp3", lib=lib, mid="abcde")

    assert cats.id == "abcde"
    assert dogs.id != "abcde"
    assert dogs.path != cats.path

    ids = [m.id for m in LIB.all_meetings(lib)]
    assert len(ids) == 2
    assert len(set(ids)) == 2, "speakers.db keys on this: %r" % (ids,)


def test_a_regenerated_id_is_written_everywhere(lib):
    """When the requested id is refused, meeting.json and the folder agree."""
    LIB.create("Meeting about cats", "cats.mp3", lib=lib, mid="abcde")
    dogs = LIB.create("Meeting about dogs", "dogs.mp3", lib=lib, mid="abcde")

    assert dogs.read()["id"] == dogs.id
    assert dogs.path.name == LIB.folder_name("Meeting about dogs", dogs.id)
    assert dogs.path.name.endswith("-" + dogs.id)
    # A fresh id, so it is a full-width one even though "abcde" was 5.
    assert len(dogs.id) == LIB.ID_CHARS


def test_a_minted_id_that_collides_is_reminted(lib, monkeypatch):
    """The retry loop consults ids in use, not paths in use.

    With new_id() forced to repeat itself, the old check would have accepted the
    repeat for the second meeting because `dogs-aaaaaaaa` was a free path.
    """
    minted = iter(["aaaaaaaa", "aaaaaaaa", "bbbbbbbb"])
    monkeypatch.setattr(LIB, "new_id", lambda *a, **k: next(minted))

    cats = LIB.create("cats", "cats.mp3", lib=lib)
    dogs = LIB.create("dogs", "dogs.mp3", lib=lib)

    assert cats.id == "aaaaaaaa"
    assert dogs.id == "bbbbbbbb"


def test_ids_in_use_are_read_from_meeting_json(lib, write_meeting):
    """A folder renamed by hand still owns its id.

    The folder says `zzzzzzzz`; meeting.json says `abcde`. Nothing about the
    path collides, so only reading the metadata can catch this.
    """
    write_meeting("tidied-up-by-hand-zzzzzzzz", id="abcde", title="Renamed")

    m = LIB.create("Something else", "x.mp3", lib=lib, mid="abcde")

    assert m.id != "abcde"
    ids = [x.id for x in LIB.all_meetings(lib)]
    assert sorted(ids) == sorted({"abcde", m.id})


def test_the_same_title_twice_still_gets_two_ids(lib):
    """The path check was not wrong, only insufficient -- it still has to hold.

    The real library contains this case: two recordings of the same episode.
    """
    a = LIB.create("Weekly sync", "one.mp3", lib=lib, mid="abcde")
    b = LIB.create("Weekly sync", "two.mp3", lib=lib, mid="abcde")

    assert a.id != b.id
    assert a.path != b.path
    assert a.stem == b.stem  # the NAMES may collide; the identity may not


def test_a_library_of_meetings_has_no_duplicate_ids(lib):
    """End to end, through the real minting path."""
    made = [LIB.create("Meeting number %d" % i, "%d.mp3" % i, lib=lib)
            for i in range(60)]
    ids = [m.id for m in made]

    assert len(set(ids)) == 60
    assert {m.id for m in LIB.all_meetings(lib)} == set(ids)
    assert all(len(i) == LIB.ID_CHARS for i in ids)


def test_create_writes_identity_before_anything_else(lib):
    m = LIB.create("Platform Review: Weekly Q3", "raw upload.mp3", lib=lib)

    assert m.meta_path.exists()
    assert m.read() == {"id": m.id, "title": "Platform Review: Weekly Q3",
                        "source": "raw upload.mp3"}
    assert m.path.parent == Path(lib)
    assert Path(lib).is_dir()


def test_later_writes_never_drop_the_id(lib):
    """write() merges. Losing the id would orphan every decision in the store."""
    m = LIB.create("Platform Review", "src.mp3", lib=lib)
    mid = m.id

    m.write(duration_s=2287.79, n_segments=456)
    m.write(coverage=0.9987)

    d = m.read()
    assert d["id"] == mid
    assert d["title"] == "Platform Review"
    assert d["source"] == "src.mp3"
    assert (d["duration_s"], d["n_segments"], d["coverage"]) == (2287.79, 456, 0.9987)
    assert LIB.Meeting(m.path).id == mid


# ======================================================================
# The id's shape
# ======================================================================

def test_id_chars_is_eight():
    """5 -> 8. 31**5 is 28.6M: 5,000 meetings collide about a third of the time."""
    assert LIB.ID_CHARS == 8


def test_alphabet_omits_the_characters_people_misread():
    for ch in "01lo":
        assert ch not in LIB.ID_ALPHABET, "%r is confusable when read aloud" % ch
    # Also absent, though the comment above it only names 0/1/l/o.
    assert "i" not in LIB.ID_ALPHABET
    assert len(LIB.ID_ALPHABET) == 31, "the 31 in 31**8 comes from here"
    assert len(set(LIB.ID_ALPHABET)) == len(LIB.ID_ALPHABET)
    assert LIB.ID_ALPHABET == LIB.ID_ALPHABET.lower()
    assert all(c.isalnum() and c.isascii() for c in LIB.ID_ALPHABET)


def test_new_id_shape_and_alphabet():
    rng = random.Random(20260815)
    ids = [LIB.new_id(rng) for _ in range(500)]

    assert all(len(i) == LIB.ID_CHARS for i in ids)
    assert set("".join(ids)) <= set(LIB.ID_ALPHABET)
    assert len(set(ids)) == 500, "500 draws from 31**8 should not repeat"


def test_new_id_takes_its_randomness_from_the_caller():
    """Seedable, so a test can force a collision without patching the module."""
    assert LIB.new_id(random.Random(7)) == LIB.new_id(random.Random(7))
    assert LIB.new_id(random.Random(7)) != LIB.new_id(random.Random(8))


# ======================================================================
# Old 5-character ids keep working -- nothing assumes a length
# ======================================================================

def test_a_legacy_five_character_id_still_resolves(lib, write_meeting):
    """Straight from the real library: 5bvyc, minted before the widening."""
    write_meeting("one-trust-network-to-rule-5bvyc", id="5bvyc",
                  title="One Trust Network to Rule them All with Sreeram")

    m = LIB.find("5bvyc", lib)
    assert m is not None
    assert m.id == "5bvyc"
    assert len(m.id) == 5, "existing ids are not rewritten or padded"
    assert LIB.find("one-trust-network-to-rule-5bvyc", lib).id == "5bvyc"


def test_a_legacy_id_is_not_handed_out_again(lib, write_meeting):
    write_meeting("one-trust-network-to-rule-5bvyc", id="5bvyc")

    m = LIB.create("Some other meeting", "x.mp3", lib=lib, mid="5bvyc")

    assert m.id != "5bvyc"


def test_old_and_new_ids_live_in_one_library(lib, write_meeting):
    write_meeting("one-trust-network-to-rule-5bvyc", id="5bvyc")
    write_meeting("how-eigenlayer-supercharges-eth-6zch5", id="6zch5")
    fresh = LIB.create("Brand new meeting", "new.mp3", lib=lib)

    got = {m.id: len(m.id) for m in LIB.all_meetings(lib)}
    assert got == {"5bvyc": 5, "6zch5": 5, fresh.id: 8}
    for ref in got:
        assert LIB.find(ref, lib) is not None


# ======================================================================
# slug() -- a name, and only a name
# ======================================================================

@pytest.mark.parametrize("title,expected", [
    # 5 words, whichever comes first...
    ("one two three four five six seven", "one-two-three-four-five"),
    ("A B C D E F G", "a-b-c-d-e"),
    # ...or 40 characters.
    ("Platform Review: What Changes Next Quarter, Staffing and More | Weekly",
     "platform-review-what-changes-next"),
    ("supercalifragilistic expialidocious antidisestablishmentarianism",
     "supercalifragilistic-expialidocious"),
    # Punctuation becomes a separator, never a character in the name.
    ("Hello   World!!! How are you?", "hello-world-how-are-you"),
    ("don't stop believing", "don-t-stop-believing"),
    ("The_next_5_years", "the-next-5-years"),
    ("  --  spaced  out  --  ", "spaced-out"),
    ("Q3/Q4 (draft) [v2]", "q3-q4-draft-v2"),
    # Word order is kept -- no stopword stripping, no reordering.
    ("The next 5 years will change everything", "the-next-5-years-will"),
    # Unicode survives: \w is unicode-aware, so a non-ASCII title stays readable
    # instead of degrading to "meeting".
    ("Café Réunion — Q3", "café-réunion-q3"),
    ("会議 notes", "会議-notes"),
    # Nothing usable at all still yields a directory name.
    ("", "meeting"),
    ("   ", "meeting"),
    ("!!!???", "meeting"),
    ("---", "meeting"),
    ("___", "meeting"),
])
def test_slug(title, expected):
    assert LIB.slug(title) == expected


def test_slug_respects_both_caps():
    for title in ["one two three four five six seven",
                  "Platform Review: What Changes Next Quarter, Staffing and More",
                  "The next 5 years will change everything"]:
        s = LIB.slug(title)
        assert len(s.split("-")) <= LIB.SLUG_WORDS
        assert len(s) <= LIB.SLUG_CHARS


def test_slug_stops_at_the_cap_rather_than_repacking():
    """The character cap truncates; it does not skip a long word for a short one.

    Keeping the words a person said, in order, is the whole point -- a slug that
    hopped over `elaborations` to reach `ok` would read as a different title.
    """
    s = LIB.slug("z" * 35 + " elaborations ok")
    assert s == "z" * 35
    assert "ok" not in s


def test_slug_of_one_very_long_word_is_the_word():
    """The only way past SLUG_CHARS, and deliberate: `out and ...` lets the
    first word through unconditionally, because a bare "meeting" would be worse
    than a long name."""
    s = LIB.slug("Antidisestablishmentarianismically" * 2)
    assert s == "antidisestablishmentarianismicallyantidisestablishmentarianismically"
    assert len(s.split("-")) == 1


def test_slug_of_a_very_long_title():
    title = " ".join("word%d" % i for i in range(200))
    s = LIB.slug(title)
    assert s == "word0-word1-word2-word3-word4"
    assert len(s) <= LIB.SLUG_CHARS


@pytest.mark.parametrize("title", [
    "Platform Review: Weekly Q3", "", "!!!", "Café Réunion", "a" * 90,
])
def test_folder_name_puts_the_id_last(title):
    """Id-last keeps the library alphabetically browsable and tab-completable,
    and makes the name unique by construction."""
    name = LIB.folder_name(title, "k49fj7wq")

    assert name == LIB.slug(title) + "-k49fj7wq"
    assert name.endswith("-k49fj7wq")
    assert not name.startswith("k49fj7wq")
    # And the split back out is the one Meeting relies on.
    assert name.rsplit("-", 1) == [LIB.slug(title), "k49fj7wq"]


def test_the_library_sorts_by_name_not_by_id(lib, monkeypatch):
    """What id-last buys: related meetings sit together."""
    minted = iter(["zzzzzzzz", "aaaaaaaa", "mmmmmmmm"])
    monkeypatch.setattr(LIB, "new_id", lambda *a, **k: next(minted))
    LIB.create("Alpha sync", "1.mp3", lib=lib)
    LIB.create("Alpha retro", "2.mp3", lib=lib)
    LIB.create("Beta sync", "3.mp3", lib=lib)

    assert [m.path.name for m in LIB.all_meetings(lib)] == [
        "alpha-retro-aaaaaaaa", "alpha-sync-zzzzzzzz", "beta-sync-mmmmmmmm"]


# ======================================================================
# find()
# ======================================================================

def test_find_by_id(three):
    for key, m in three.items():
        assert LIB.find(m.id, m.path.parent).path == m.path


def test_find_by_folder_name(three, lib):
    m = three["platform"]
    assert LIB.find(m.path.name, lib).path == m.path


def test_find_by_stem(three, lib):
    m = three["platform"]
    assert m.stem == "platform-review-weekly-q3"
    assert LIB.find(m.stem, lib).path == m.path


def test_find_by_path(three, lib):
    m = three["platform"]
    for ref in (str(m.path), m.path, str(m.path) + "/", str(m.path) + "///",
                "  %s  " % m.path):
        assert LIB.find(ref, lib).path == m.path
    # Relative to the cwd, too (the _isolate fixture chdir'd to tmp_path). The
    # Meeting keeps the path as typed rather than resolving it, so compare
    # resolved -- everything downstream opens files through this path.
    rel = LIB.find(Path("library") / m.path.name, lib)
    assert rel.path.resolve() == m.path
    assert rel.id == m.id


def test_find_by_path_does_not_need_a_library(three):
    """A path that holds a meeting.json is a meeting, wherever it sits."""
    m = three["platform"]
    assert LIB.find(str(m.path), Path("/nonexistent")).path == m.path


def test_find_by_unique_title_substring(three, lib):
    assert LIB.find("Weekly Q3", lib).path == three["platform"].path
    assert LIB.find("weekly q3", lib).path == three["platform"].path
    assert LIB.find("PLATFORM", lib).path == three["platform"].path
    assert LIB.find("2028", lib).path == three["b2028"].path


def test_find_returns_none_for_an_ambiguous_prefix(three, lib):
    """`budget-planning` is a prefix of two folder names, so it names neither."""
    assert LIB.find("budget-planning", lib) is None
    assert LIB.find("Budget Planning", lib) is None
    assert LIB.find("budget", lib) is None


def test_find_returns_none_for_a_miss(three, lib):
    assert LIB.find("nosuchmeeting", lib) is None
    assert LIB.find("k49fj7wq", lib) is None       # a full-width id nobody has
    assert LIB.find("platform-review-weekly-q4", lib) is None
    assert LIB.find(Path("/nope/at/all"), lib) is None


def test_find_on_an_empty_library(lib):
    assert LIB.all_meetings(lib) == []
    assert LIB.find("anything", lib) is None
    assert LIB.all_meetings(Path(lib) / "not-created-yet") == []


def test_find_prefers_an_exact_id_to_a_substring(lib, write_meeting):
    """An id typed in full wins over a meeting that merely contains it."""
    target = write_meeting("short-one-abcde", id="abcde")
    write_meeting("notes-about-abcde-mn4p7q2r", id="mn4p7q2r",
                  title="Notes about abcde")

    assert LIB.find("abcde", lib).path == target.path


def test_find_of_a_blank_ref(lib, write_meeting):
    """A blank ref matches every meeting, so it is only ambiguity that stops it.

    Documented, not endorsed: with two meetings this is None, but a library with
    exactly one meeting hands that meeting back for `find("")`. Nothing in
    library.py rejects an empty ref up front.
    """
    write_meeting("budget-planning-2027-mn4p7q2r", id="mn4p7q2r")
    write_meeting("budget-planning-2028-x9wq3v5k", id="x9wq3v5k")
    assert LIB.find("", lib) is None
    assert LIB.find("   ", lib) is None

    solo = Path(lib).parent / "solo-library"
    only = LIB.create("Only meeting", "one.mp3", lib=solo)
    assert LIB.find("", solo).path == only.path


@pytest.mark.xfail(strict=True, reason=(
    "find() resolves an exact stem hit before it checks for ambiguity, so two "
    "meetings of the same title -- which the real library contains today, "
    "one-trust-network-to-rule-5bvyc and -9ajq9 -- silently resolve to "
    "whichever all_meetings() sorted first"))
def test_find_refuses_a_stem_two_meetings_share(lib, write_meeting):
    """The same 'whichever it met first' failure as b9e2659, one branch over.

    Folder names differ only by the id, so both meetings have the stem
    `one-trust-network-to-rule`, and the exact-match loop in find() returns the
    first match instead of reporting the ambiguity -- while the title-substring
    branch below it would correctly return None for the very same ref.

    To pass, find() would have to gather ALL exact hits on (id, folder name,
    stem) and return None unless exactly one meeting matched -- or at minimum
    treat a stem that more than one meeting shares as ambiguous, since a stem is
    a name and only the id is unique.
    """
    write_meeting("one-trust-network-to-rule-5bvyc", id="5bvyc",
                  title="One Trust Network to Rule them All")
    write_meeting("one-trust-network-to-rule-9ajq9", id="9ajq9",
                  title="One Trust Network to Rule them All")

    assert LIB.find("one-trust-network-to-rule", lib) is None


# ======================================================================
# Identity lives in meeting.json, names live in the folder
# ======================================================================

def test_renaming_the_folder_does_not_change_the_id(lib, write_meeting):
    """"a folder can be renamed, moved, or tidied up by hand and every decision
    ever recorded about that meeting still points at it"."""
    m = write_meeting("platform-review-weekly-q3-k49fj",
                      id="k49fj", title="Platform Review: Weekly Q3")
    renamed = m.path.parent / "2026-q3-archive"
    m.path.rename(renamed)

    after = LIB.Meeting(renamed)
    assert after.id == "k49fj"
    assert after.title == "Platform Review: Weekly Q3"
    assert LIB.find("k49fj", lib).path == renamed
    # The NAMES follow the folder, which is exactly what they are for.
    assert after.stem == "2026-q3"
    assert after.file("transcript", "txt").name == "2026-q3-transcript.txt"


def test_meeting_json_beats_the_folder_suffix(lib, write_meeting):
    m = write_meeting("cats-abcde", id="9ajq9", title="Cats")

    assert LIB.Meeting(m.path).id == "9ajq9"
    assert LIB.find("9ajq9", lib).path == m.path
    # `abcde` is now just letters in a folder name, not an id.
    assert LIB.find("abcde", lib).id == "9ajq9"


def test_the_id_falls_back_to_the_folder_suffix(lib):
    """No meeting.json, or one without an id: the name is all there is."""
    p = Path(lib) / "platform-review-weekly-q3-k49fj"
    p.mkdir(parents=True)
    assert LIB.Meeting(p).id == "k49fj"
    assert LIB.Meeting(p).title == "platform-review-weekly-q3"

    (p / "meeting.json").write_text(json.dumps({"title": "Platform Review"}))
    assert LIB.Meeting(p).id == "k49fj"
    assert LIB.Meeting(p).title == "Platform Review"


def test_unreadable_metadata_degrades_to_the_name(lib):
    """A truncated meeting.json must not raise -- read() swallows it."""
    p = Path(lib) / "platform-review-weekly-q3-k49fj"
    p.mkdir(parents=True)
    (p / "meeting.json").write_text('{"id": "k49f')

    m = LIB.Meeting(p)
    assert m.read() == {}
    assert m.id == "k49fj"
    assert m.title == "platform-review-weekly-q3"


def test_metadata_round_trips_unicode(lib):
    m = LIB.create("Café Réunion — Q3", "Café.mp3", lib=lib)

    assert m.title == "Café Réunion — Q3"
    assert "Café" in m.meta_path.read_text()          # not \\u00e9 escapes
    assert json.loads(m.meta_path.read_text())["source"] == "Café.mp3"


# ======================================================================
# The files inside a meeting
# ======================================================================

def test_file_names_are_slug_prefixed(lib):
    """Transcripts leave the folder constantly; `transcript.txt` names nothing."""
    m = LIB.create("Platform Review: Weekly Q3", "raw.mp3", lib=lib)

    assert m.stem == "platform-review-weekly-q3"
    assert m.file("transcript", "txt").name == "platform-review-weekly-q3-transcript.txt"
    assert m.file("transcript", "json").name == "platform-review-weekly-q3-transcript.json"
    assert m.file("embeddings", "npz").name == "platform-review-weekly-q3-embeddings.npz"
    assert m.file("clusters", "npz").parent == m.path
    assert m.meta_path == m.path / "meeting.json"
    assert m.clips_dir == m.path / "clips"


def test_audio_is_none_when_it_is_not_there(lib):
    """"It is allowed not to be: clips are enough to name a voice by."""
    m = LIB.create("Platform Review", "raw.mp3", lib=lib)
    assert m.audio() is None

    # Other files in the folder are not audio, and neither is a near miss.
    m.file("transcript", "txt").write_text("hello")
    (m.path / "audio.mp3").write_bytes(b"")
    (m.path / "platform-review-audio-extra.mp3").write_bytes(b"")
    assert m.audio() is None


def test_audio_is_found_whatever_the_extension(lib):
    m = LIB.create("Platform Review", "raw.wav", lib=lib)
    (m.path / "platform-review-audio.wav").write_bytes(b"")
    assert m.audio().name == "platform-review-audio.wav"

    # Deterministic when there is more than one: sorted, first wins.
    (m.path / "platform-review-audio.mp3").write_bytes(b"")
    assert m.audio().name == "platform-review-audio.mp3"


def test_audio_does_not_reach_into_another_meeting(lib):
    a = LIB.create("Platform Review", "a.mp3", lib=lib)
    b = LIB.create("Platform Review Extended", "b.mp3", lib=lib)
    (b.path / ("%s-audio.mp3" % b.stem)).write_bytes(b"")

    assert a.audio() is None
    assert b.audio().parent == b.path


@pytest.mark.parametrize("raw,escaped", [
    ("[", "[[]"),
    ("]", "[]]"),
    ("?", "[?]"),
    ("*", "[*]"),
    ("a[b]c?d*e", "a[[]b[]]c[?]d[*]e"),
    ("**", "[*][*]"),
    ("plain-stem-2027", "plain-stem-2027"),
    ("", ""),
])
def test_glob_escape(raw, escaped):
    assert LIB.glob_escape(raw) == escaped


def test_audio_survives_glob_metacharacters_in_the_stem(lib):
    """A folder named by hand can contain anything; audio() globs on the stem.

    Unescaped, `we[i]rd?-x*y-audio.*` is a character class and two wildcards --
    it would miss the real file and could match a different one.
    """
    p = Path(lib) / "we[i]rd?-x*y-k49fj7wq"
    p.mkdir(parents=True)
    (p / "meeting.json").write_text(json.dumps({"id": "k49fj7wq", "title": "Weird"}))
    m = LIB.Meeting(p)
    assert m.stem == "we[i]rd?-x*y"

    # The decoy is what an unescaped pattern would happily match instead.
    (p / "wird-xy-audio.mp3").write_bytes(b"")
    assert m.audio() is None

    real = p / "we[i]rd?-x*y-audio.mp3"
    real.write_bytes(b"")
    assert m.audio() == real
