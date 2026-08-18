"""Persistent speaker profiles: name a voice once, recognise it thereafter.

The store. Most people reach it through ./speakers at the repo root, which also
does who/play/clips (reading the transcript and audio). This module is the
sqlite layer plus a direct CLI:

  speakers.py link --apply                      group voices across every meeting
  speakers.py review                            who is worth naming next
  speakers.py name <group> "Bob Smith"          name them, in every meeting at once
  speakers.py list
  speakers.py rename  <speaker_id> "New Name"
  speakers.py forget  <speaker_id>              delete a person and their voiceprints

Design follows what was measured, not intuition:

* ONE CENTROID per person, not a prototype set. Measured on 17 speakers: plain
  centroid 0.760% EER vs 5 prototypes 0.760% but 93.4% open-set accuracy at
  0.1% FA against the centroid's 97.94%. Multi-prototype raises impostor scores
  as much as target scores.
* NO OUTLIER TRIMMING. 20% trim measured identical, 40% trim measured worse.
* 10 SECONDS of clean speech is the enrollment knee (99.55% top-1); two minutes
  buys 0.2 points.
* THRESHOLDS ARE LEVEL-SPECIFIC. A threshold fitted on clip-vs-clip comparisons
  produced 8 false accepts out of 30 when applied to centroid-vs-centroid,
  because averaging pushes target scores from ~0.61 to ~0.90 while impostor
  scores barely move. Every stored threshold records its level.
* ROSTER FIRST. Acceptance is a max over N candidates, so false-accept risk
  grows with gallery size. Scoring 3 known attendees is far safer than 500.
"""
import argparse, collections, json, os, re, sqlite3, time

import numpy as np

def _default_work():
    """The checkout this file lives in. Everything -- venv, weights, profile
    store, work directories -- stays inside it, so nothing is written outside
    and two checkouts never share state. MS_WORK overrides."""
    import os
    if os.environ.get("MS_WORK"):
        return os.environ["MS_WORK"]
    here = os.path.dirname(os.path.abspath(__file__))
    # pipeline/x.py -> repo root; pipeline/link/x.py -> repo root
    while os.path.basename(here) in ("link", "pipeline"):
        here = os.path.dirname(here)
    return here


WORK = _default_work()
DB = os.environ.get("MS_SPEAKER_DB", os.path.join(WORK, "speakers.db"))
EMBED_MODEL = "wespeaker-resnet34-LM"

# Every threshold the naming path reads, and where it lives. A
# centroid-vs-centroid ACCEPT of 0.55, a 0.10 margin and a 0.40 review floor sat
# here too, read only by identify.py; they went with it, so nothing in this
# module now holds a number a normal run does not use.
#
#   match_speakers.ACCEPT (0.62)   an atom joins a person
#   match_speakers.SUBPROFILE      two atoms could be the same person at all
#   LINK_ACCEPT (0.75)             two clusters merge into one group
#   COVER, KEEP_EXEMPLAR           how finely a known voice is described
#   NAME_INHERIT_SHARE             a regrouped group keeps a name
#   MIN_LINK_SEC, MIN_ENROLL_SEC   speech needed to match, and to enrol
MIN_ENROLL_SEC = 10.0

# Corpus-wide linking is a DIFFERENT level from matching one recording, and
# gets its own numbers, for the reason in this module's header: a threshold
# fitted at one level misfires at another. Two things make linking the harsher
# case. It scores every cluster against every other, so acceptance is a max
# over the whole corpus rather than over a roster -- the widest gallery there
# is. And a wrong link is not one bad row, it is one name spread across every
# meeting it joins.
#
# Measured on the podcast corpus, cross-meeting centroid pairs, truth by ear:
#
#   same person   (across 5 recordings)   0.754 - 0.889
#   IMPOSTOR      (a 24s cluster)         0.708 - 0.741
#
# The per-recording bar -- match_speakers.ACCEPT, 0.62 -- takes that impostor
# without pausing. Duration is what made it dangerous: a 53-second cluster of a
# different audience member scored 0.521-0.547 against the same person, and
# every true pair came from clusters of 1395s or more. Short clusters give
# unstable centroids that drift toward whoever dominates the recording.
#
# Requiring 60s to be eligible drops the worst impostor from 0.741 to 0.458 and
# widens the gap to the nearest true pair from 0.013 to 0.296. The gate does far
# more work here than the threshold does, which is why it is not optional.
LINK_ACCEPT = 0.75

# How well a recording must match what we already store before it may become a
# sub-profile itself. Above ACCEPT, because this feeds back: see refresh_exemplars.
KEEP_EXEMPLAR = float(os.environ.get("MS_KEEP_EXEMPLAR", "0.72"))

# How many recordings must agree with each other before they are treated as a
# circumstance rather than as noise. One outlier never becomes a sub-profile.
MIN_CORROBORATION = int(os.environ.get("MS_MIN_CORROBORATION", "4"))

# Share of a regrouped group's speech that must already belong to one person
# before the group inherits their name. Below it the group is left unnamed and
# `review` surfaces it, which is recoverable; a wrong inherited name is not.
NAME_INHERIT_SHARE = float(os.environ.get("MS_NAME_INHERIT_SHARE", "0.5"))

# How WELL every recording of a person must be covered by one of their stored
# sub-profiles. Separate from ACCEPT, which decides whether a stranger is them:
# this decides how finely we describe someone we already know, and at ACCEPT the
# first profile covers everyone immediately so almost nobody gets a second.
# Swept over 300 arguments with 391 people enrolled:
#
#     0.62   399 exemplars,  8 people with >1   2.21% wrong
#     0.72   417            21                  2.17%
#     0.80   458            42                  2.15%
#     0.86   530            63                  2.35%   <- describing noise
#
# The gain is small HERE because these are 300 recordings of one courtroom, and
# a person who only ever sounds one way needs only one profile. It is not what
# the mechanism is for: the Court's two telephone terms are where a single
# frozen reference put 22-28% of speech under the wrong name, and that is the
# case a second profile exists to survive.
COVER = float(os.environ.get("MS_COVER", "0.80"))
MIN_LINK_SEC = 60.0    # too short to enrol is too short to match


def db(path=None):
    """Open the profile store and make sure its schema exists.

    `path` lets a caller open a store somewhere other than the default -- the UI
    keeps one per library. It defaults to DB so every existing caller is
    unaffected, and it exists so the schema has exactly ONE definition: the UI
    used to carry its own copy of this DDL, which is two things to keep in step
    over the one file in the project that cannot be rebuilt from the audio.
    """
    c = sqlite3.connect(path or DB)
    # sqlite ignores every FOREIGN KEY in the DDL below unless this is set, and
    # it is per CONNECTION, not per database file. Declared and never enabled is
    # the worst of the three states: `forget 1` deleted the speaker and left
    # exemplars auto-1/auto-2 still pointing at id 1 and group 2 still claiming
    # it, so `groups` showed the person as "-- unnamed --", `review` skipped
    # them, and the next name minted id 1 again and inherited both voiceprints.
    # With this on, a delete that misses a referencing row fails loudly instead.
    c.execute("PRAGMA foreign_keys=ON")
    # prototypes and decisions were the legacy naming path's storage. Removing
    # them from the schema stopped NEW stores having them and left every
    # existing store carrying the tables, their rows, and -- the part that bites
    # -- their FOREIGN KEY onto speakers(id). With the pragma now on, that made
    # `forget` fail with "FOREIGN KEY constraint failed" on any store older than
    # the change, which is every store that has ever been used. Nothing reads
    # either table.
    for dead in ("prototypes", "decisions"):
        c.execute("DROP TABLE IF EXISTS %s" % dead)
    c.commit()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS speakers(
      id INTEGER PRIMARY KEY, name TEXT UNIQUE, created_at REAL);

    -- Every cluster the pipeline has ever produced, named or not. `exemplars`
    -- cannot hold these: it requires a speaker_id, and the whole point of
    -- linking is that a voice belongs to a group BEFORE anyone names it.
    CREATE TABLE IF NOT EXISTS clusters(
      id INTEGER PRIMARY KEY, meeting TEXT, cluster TEXT,
      emb BLOB, dim INTEGER, embed_model TEXT, seconds REAL,
      group_id INTEGER, created_at REAL,
      UNIQUE(meeting, cluster, embed_model));
    CREATE INDEX IF NOT EXISTS clusters_group ON clusters(group_id);

    -- A person's voice as it actually sounds, once per CIRCUMSTANCE. One
    -- averaged vector cannot stand for someone across a change of recording
    -- condition: measured on the Court's 2020-21 telephone arguments, a
    -- courtroom reference put 22-28% of speech under the WRONG name, against
    -- 0.0-0.4% once each condition had its own exemplar.
    --
    -- `condition` is a free string and nothing reads meaning into it:
    -- "telephone", "far-field", "2015", "headset", "the bad conference room".
    -- It exists so that "this is also her, through a potato" is a thing someone
    -- can say and have stored. Pooling within one keeps this bounded by people
    -- x conditions rather than by recordings.
    CREATE TABLE IF NOT EXISTS exemplars(
      id INTEGER PRIMARY KEY, speaker_id INTEGER, condition TEXT, emb BLOB,
      dim INTEGER, embed_model TEXT, seconds REAL, created_at REAL,
      UNIQUE(speaker_id, condition, embed_model),
      FOREIGN KEY(speaker_id) REFERENCES speakers(id));
    CREATE INDEX IF NOT EXISTS exemplars_speaker ON exemplars(speaker_id);

    -- An identity discovered from audio. speaker_id stays NULL until a human
    -- names it, and naming is then one UPDATE here rather than a rescan: every
    -- meeting in the group inherits the name through the join.
    CREATE TABLE IF NOT EXISTS groups(
      id INTEGER PRIMARY KEY, speaker_id INTEGER, embed_model TEXT,
      linked_at REAL,
      FOREIGN KEY(speaker_id) REFERENCES speakers(id));
    """)
    # The column was called `era` for part of one afternoon, which read as a
    # date and it is not one. Rename in place rather than asking anyone to
    # rebuild a store whose whole purpose is being the file you cannot rebuild.
    cols = [r[1] for r in c.execute("PRAGMA table_info(exemplars)")]
    if "era" in cols and "condition" not in cols:
        c.execute("ALTER TABLE exemplars RENAME COLUMN era TO condition")
        c.commit()
    return c


def centroids_from_npz(path):
    """-> {cluster_id: (centroid, seconds)} from the _clusters.npz link.py writes.

    link.py already computes exactly the vector this store enrols -- one
    seconds-weighted centroid per speaker in the meeting -- so read that rather
    than rebuilding it from the per-segment embeddings.
    """
    z = np.load(path, allow_pickle=True)
    out = {}
    for i, c in enumerate(z["cluster"]):
        c = str(c)
        if c.startswith("G-"):          # the linker's leftover bucket, not a person
            continue
        v = z["centroid"][i].astype(np.float64)
        n = np.linalg.norm(v)
        if n <= 0:
            continue
        out[c] = (v / n, float(z["secs"][i]))
    return out


def index_clusters(conn, run_dir=None):
    """Record every meeting's cluster centroids in `clusters`. -> n_indexed

    Idempotent, and it must PRESERVE group_id on a cluster that still exists.
    Wiping and reinserting looks equivalent and is not: link_groups reads
    group_id to find which clusters a human has already named, so clearing it
    here silently empties the must-link set and every naming decision is lost on
    the next relink. That is a real defect this function shipped with, caught by
    naming a group and relinking at a stricter threshold.

    A cluster that no longer exists after a re-decode is dropped, since its id
    may now mean a different voice.

    Reads the LIBRARY, not the run directory. `out/` is scratch that a run
    empties; the library is what survives, what `./speakers meetings` lists, and
    where --move-audio puts the audio. Centroids come from each meeting's
    -clusters.npz via centroids_from_npz, which is the vector link.py already
    computed for exactly this purpose rather than one rebuilt from segments.
    """
    import library as LIB
    try:
        meetings = LIB.all_meetings()
    except Exception:
        meetings = []
    n = 0
    for m in meetings:
        meeting = m.id
        try:
            cents = centroids_from_npz(str(m.file("clusters", "npz")))
        except (OSError, KeyError, ValueError):
            continue
        if cents:
            conn.execute(
                "DELETE FROM clusters WHERE meeting=? AND embed_model=? AND"
                " cluster NOT IN (%s)" % ",".join("?" * len(cents)),
                [meeting, EMBED_MODEL] + sorted(cents))
        for g, (v, secs) in cents.items():
            conn.execute(
                "INSERT INTO clusters(meeting, cluster, emb, dim, embed_model,"
                " seconds, group_id, created_at) VALUES(?,?,?,?,?,?,NULL,?)"
                " ON CONFLICT(meeting, cluster, embed_model) DO UPDATE SET"
                " emb=excluded.emb, dim=excluded.dim, seconds=excluded.seconds,"
                # group_id follows the VOICE, not the label. Matching renumbers
                # `global`, so (meeting, G05) can be a different person after a
                # relabel while keeping G05's old row -- and with it the name
                # attached to it. That is how one-second fragments ended up
                # filed under a justice. If the vector changed, whatever
                # identity was attached to the old one is no longer justified.
                " group_id=CASE WHEN clusters.emb IS excluded.emb"
                " THEN clusters.group_id ELSE NULL END",
                (meeting, g, v.astype(np.float32).tobytes(), len(v),
                 EMBED_MODEL, secs, time.time()))
            n += 1
    conn.commit()
    return n


def _similar_pairs(A, thr, block=4096):
    """Above-threshold pairs of a normalised matrix, without materialising NxN.

    The full matrix is never wanted -- only pairs over the threshold, which are
    a vanishing fraction. Blocking holds peak memory at block x N: measured on
    50k centroids that is 0.38 GB against 9.31 GB, and 13s against 180s, on CPU
    alone. Nothing here needs a GPU until roughly 4,000 meetings.
    """
    n = len(A)
    out = []
    for i in range(0, n, block):
        S = A[i:i + block] @ A.T
        for r, c in zip(*np.where(S >= thr)):
            if i + r < c:                       # upper triangle only
                out.append((float(S[r, c]), i + int(r), int(c)))
    return out



def refresh_exemplars(conn, speaker_id, keep_named=True):
    """Rebuild a person's sub-profiles to COVER everything known about them.

    Averaging every recording of someone into one vector produces a point that
    describes no actual circumstance -- measured, naming that way ran at 2.63%
    wrong against 0.18% for per-circumstance exemplars. So cover instead of
    average: keep taking the recording least well represented by what is already
    stored, until every one is within ACCEPT of something.

    This has to run after LINKING as well as after naming, and that is the whole
    trick. A group is formed at 0.75, so its members are already alike and
    naming it can only ever learn the circumstance it was formed in -- the
    recordings that would teach the others are precisely the ones that failed to
    join. Matching at ACCEPT reaches them, and rebuilding here turns each into a
    sub-profile, which lets the next pass reach further still.

    Sub-profiles a human named are never touched; auto- ones are ours to redraw.
    """
    import numpy as np

    import match_speakers as MS
    rows = conn.execute(
        "SELECT c.emb, c.dim, c.seconds FROM clusters c JOIN groups g"
        " ON g.id = c.group_id WHERE g.speaker_id = ? AND c.embed_model = ?"
        " AND c.seconds > 0", (speaker_id, EMBED_MODEL)).fetchall()
    if not rows:
        return 0
    E = np.stack([MS.unit(np.frombuffer(r[0], dtype=np.float32, count=r[1]))
                  for r in rows])
    w = np.array([r[2] for r in rows], dtype=float)

    # Only CONFIDENT members may become a sub-profile. This is a self-training
    # loop -- what it learns it then matches against -- so a marginal cluster
    # promoted to a sub-profile starts attracting things like itself, and the
    # error compounds instead of staying put. Measured: rebuilding from every
    # attached cluster took naming from 2.63% wrong to 3.31% while adding two
    # meetings of coverage, which is a bad trade in the direction that matters.
    #
    # Members at or above KEEP of an already-stored profile are the person as we
    # already know them; the rest are still NAMED, they just do not get a vote on
    # what she sounds like.
    have = conn.execute("SELECT emb, dim FROM exemplars WHERE speaker_id=?"
                        " AND embed_model=?", (speaker_id, EMBED_MODEL)).fetchall()
    if have:
        H = np.stack([MS.unit(np.frombuffer(h[0], dtype=np.float32, count=h[1]))
                      for h in have])
        near = (E @ H.T).max(axis=1) >= KEEP_EXEMPLAR
        # Recordings far from every stored profile are the interesting ones and
        # also the dangerous ones: either this person in a circumstance we have
        # never heard, or not this person at all. Similarity cannot tell those
        # apart, but AGREEMENT can. One odd recording is noise and gets no vote;
        # a dozen odd recordings that all sound like EACH OTHER are a
        # circumstance -- a microphone, a room, a decade of ageing.
        #
        # Dropping them outright is what made this inert: every named voice came
        # out with exactly one sub-profile, because the only members left were
        # the ones already alike. Requiring corroboration keeps the drift out
        # without throwing the feature away with it.
        far = np.where(~near)[0]
        keep = list(np.where(near)[0])
        if len(far) >= MIN_CORROBORATION:
            F = E[far]
            agree = ((F @ F.T) >= KEEP_EXEMPLAR).sum(axis=1) - 1
            keep += [int(far[i]) for i in np.where(agree >= MIN_CORROBORATION - 1)[0]]
        if keep:
            keep = sorted(set(keep))
            E, w = E[keep], w[keep]
    picked = [int(np.argmax(w))]
    while len(picked) < 24:
        cov = (E @ E[picked].T).max(axis=1)
        if cov.min() >= COVER:
            break
        picked.append(int(np.argmin(cov)))
    owner = (E @ E[picked].T).argmax(axis=1)
    conn.execute("DELETE FROM exemplars WHERE speaker_id=? AND embed_model=?"
                 " AND condition LIKE 'auto-%'", (speaker_id, EMBED_MODEL))
    for k in range(len(picked)):
        mine = np.where(owner == k)[0]
        if not len(mine):
            continue
        v = MS.unit((E[mine] * w[mine][:, None]).sum(axis=0))
        MS.save_exemplar(conn, speaker_id, "auto-%d" % (k + 1), v, EMBED_MODEL,
                         float(w[mine].sum()))
    return len(picked)


def link_groups(conn, thr=LINK_ACCEPT, min_sec=MIN_LINK_SEC):
    """Group eligible clusters into identities. -> (labels, rows, pairs)

    Constraints, both of which the linkage honours rather than repairs after:

      cannot-link  two clusters in the SAME meeting are already known to be
                   different people, so they must never land in one group.
      must-link    clusters a human has already named together stay together,
                   so relinking can never quietly undo a naming decision.
    """
    import cluster_speakers as cs

    rows = conn.execute(
        "SELECT id, meeting, cluster, emb, dim, seconds, group_id FROM clusters"
        " WHERE embed_model=? AND seconds >= ? ORDER BY id",
        (EMBED_MODEL, min_sec)).fetchall()
    if len(rows) < 2:
        return np.zeros(len(rows), dtype=int), rows, []

    A = np.array([np.frombuffer(r[3], dtype=np.float32, count=r[4]) for r in rows])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    n = len(rows)

    cannot = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i][1] == rows[j][1]:
                cannot[i, j] = cannot[j, i] = True

    pairs = _similar_pairs(A.astype(np.float32), thr)
    _, labels_at = cs.constrained_linkage(A @ A.T, cannot)
    lab, _ = labels_at(thr)

    # Match against the sub-profiles BEFORE trusting the linkage. Agglomerative
    # linkage compares one averaged centroid per cluster at a single threshold,
    # so a recording that drifts past it forms its own singleton group and stays
    # unnamed however many times that person has been named elsewhere. Measured
    # over 300 SCOTUS arguments, that left Sotomayor missing from 30 of them
    # while she was already named and present in 270.
    #
    # A person stored under several circumstances is as close as their CLOSEST
    # one, so a phone recording is reached by the phone exemplar without the
    # courtroom exemplar having to compromise toward it. Claims are settled
    # strongest first, and one person may take at most one cluster per meeting --
    # two clusters in a meeting are different people by MOSS's own judgment.
    import match_speakers as MS
    bank = MS.load_bank(conn, EMBED_MODEL)
    speaker_of, pre = {}, [None] * n
    if len(bank):
        sids = {nm: conn.execute("SELECT id FROM speakers WHERE name=?",
                                 (nm,)).fetchone() for nm in bank.names}
        # MS.ACCEPT, not `thr`. 0.75 is the bar for agglomerative merging, where
        # a wrong merge drags a whole subtree and every later merge inherits it.
        # This is neither: a cluster is compared only to references, at most one
        # cluster per meeting may take a person, and a mistake costs that one
        # cluster. Measured over 293 arguments that runs at 0.18% wrong names --
        # holding it to 0.75 instead just leaves people unnamed who were named.
        P = bank.score(A.astype(np.float32))                 # (clusters, people)
        cand = np.argwhere(P >= MS.ACCEPT)
        order = np.argsort(-P[cand[:, 0], cand[:, 1]]) if len(cand) else []
        taken = set()
        for k in order:
            i, pi = int(cand[k, 0]), int(cand[k, 1])
            if pre[i] is not None or (rows[i][1], pi) in taken:
                continue
            pre[i] = pi
            taken.add((rows[i][1], pi))
        # everything matched to one person becomes one group, and that group
        # carries the person, so --apply attaches it instead of inventing a name
        for pi in {x for x in pre if x is not None}:
            members = [i for i in range(n) if pre[i] == pi]
            keep = lab[members[0]]
            for i in members:
                lab[lab == lab[i]] = keep
            row = sids.get(bank.names[pi])
            if row:
                speaker_of[int(keep)] = row[0]

    # must-link: re-unite anything a human already named into one group
    named = {}
    for i, r in enumerate(rows):
        if r[6] is None:
            continue
        sid = conn.execute("SELECT speaker_id FROM groups WHERE id=?",
                           (r[6],)).fetchone()
        if sid and sid[0] is not None:
            named.setdefault(sid[0], []).append(i)
    for members in named.values():
        keep = lab[members[0]]
        for i in members:
            lab[lab == lab[i]] = keep
    return lab, rows, pairs, speaker_of


def cmd_link(a):
    conn = db()
    n = index_clusters(conn, a.run_dir)
    lab, rows, pairs, speaker_of = link_groups(conn, a.threshold, a.min_sec)
    total = conn.execute("SELECT COUNT(*) FROM clusters WHERE embed_model=?",
                         (EMBED_MODEL,)).fetchone()[0]
    print(f"indexed {n} clusters ({total} on file); {len(rows)} eligible at "
          f">={a.min_sec:.0f}s, {total - len(rows)} too short to match")
    if not len(rows):
        return

    by = {}
    for i, r in enumerate(rows):
        by.setdefault(int(lab[i]), []).append(r)
    spanning = {g: m for g, m in by.items() if len({x[1] for x in m}) > 1}
    print(f"{len(by)} groups, {len(spanning)} spanning more than one meeting\n")
    for g, m in sorted(spanning.items(), key=lambda kv: -len(kv[1])):
        print(f"  group of {len(m)} across {len({x[1] for x in m})} meetings:")
        for r in sorted(m, key=lambda r: -r[5]):
            print(f"      {r[5]:6.0f}s  {r[1][:44]:<44} {r[2]}")
    if pairs:
        print(f"\nweakest accepted link {min(p[0] for p in pairs):.3f} "
              f"(threshold {a.threshold})")

    if not a.apply:
        print("\ndry run -- nothing written. re-run with --apply")
        return

    stamp = time.time()
    conn.execute("DELETE FROM groups WHERE embed_model=? AND speaker_id IS NULL",
                 (EMBED_MODEL,))
    matched = 0
    for g, m in by.items():
        keep = None
        sid = speaker_of.get(int(g))
        if sid is not None:
            # this group matched a person's sub-profiles: reuse their group so
            # the name they already have covers these meetings too
            row = conn.execute("SELECT id FROM groups WHERE speaker_id=?"
                               " AND embed_model=?", (sid, EMBED_MODEL)).fetchone()
            if row:
                keep = row[0]
            else:
                cur = conn.execute("INSERT INTO groups(speaker_id, embed_model,"
                                   " linked_at) VALUES(?,?,?)",
                                   (sid, EMBED_MODEL, stamp))
                keep = cur.lastrowid
            matched += len(m)
        if keep is None:
            # Inherit a name only if the named members are the DOMINANT share of
            # this group's speech. Taking the first named member found is how a
            # regroup cross-contaminates: clear group_id on a lot of clusters at
            # once -- which the emb-changed rule above does by design -- and a
            # new group holding ONE leftover of someone's old group takes their
            # name, however many other members it has. Measured, that took a
            # library from 1.78% wrong names to 34.41%.
            by_sid = {}
            for r in m:
                if not r[6]:
                    continue
                row = conn.execute("SELECT speaker_id FROM groups WHERE id=?",
                                   (r[6],)).fetchone()
                if row and row[0] is not None:
                    by_sid.setdefault(row[0], [0.0, r[6]])[0] += r[5] or 0.0
            total = sum(r[5] or 0.0 for r in m) or 1.0
            if by_sid:
                best = max(by_sid.values())
                if best[0] / total >= NAME_INHERIT_SHARE:
                    keep = best[1]
        if keep is None:
            cur = conn.execute("INSERT INTO groups(speaker_id, embed_model,"
                               " linked_at) VALUES(NULL,?,?)", (EMBED_MODEL, stamp))
            keep = cur.lastrowid
        for r in m:
            conn.execute("UPDATE clusters SET group_id=? WHERE id=?", (keep, r[0]))
    conn.commit()
    # Matching just attached recordings nobody had heard this person in. Fold
    # them back into the sub-profiles so the next pass can reach further -- this
    # is what breaks the bootstrap, where a group formed at 0.75 can only ever
    # teach the circumstance it was already formed in.
    grew = 0
    for (sid,) in conn.execute(
            "SELECT DISTINCT speaker_id FROM groups WHERE speaker_id IS NOT NULL"
            " AND embed_model=?", (EMBED_MODEL,)).fetchall():
        grew += refresh_exemplars(conn, sid)
    conn.commit()
    print(f"\nwrote {len(by)} groups")
    if grew:
        print(f"{grew} sub-profiles now describe the named voices")
    if speaker_of:
        print(f"{matched} clusters attached to someone already named, by matching "
              f"their sub-profiles")



def _fold(name):
    """Case, punctuation and spacing removed -- what two spellings share."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _edits(a, b, cap=2):
    """Levenshtein distance, abandoned once it exceeds `cap`. -> int"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def near_names(conn, name, cap=2):
    """Existing speakers whose name is a plausible misspelling of `name`.

    The name string IS the identity here: `INSERT OR IGNORE INTO speakers(name)`
    makes any distinct spelling a distinct person, silently, with that person's
    voice evidence split across both. Nothing downstream can detect it, and it
    gets worse every time either spelling is used again.

    -> [(existing_name, why)] worst first. An exact match after folding away case
    and punctuation is near-certain; an edit or two is worth a question.
    """
    rows = [r[0] for r in conn.execute("SELECT name FROM speakers WHERE name IS"
                                       " NOT NULL")]
    if name in rows:
        return []
    f = _fold(name)
    out = []
    for other in rows:
        g = _fold(other)
        if not g or not f:
            continue
        if g == f:
            out.append((other, "same name, different spacing or punctuation"))
        else:
            d = _edits(f, g, cap)
            if d <= cap:
                out.append((other, f"{d} character{'s' if d != 1 else ''} apart"))
    out.sort(key=lambda x: ("same name" not in x[1], x[0]))
    return out


def cmd_name(a):
    """Name a group -- and with it every meeting the group already spans."""
    conn = db()
    if a.speaker_id is not None:
        # The id IS the identity. The store made this mistake for MEETINGS
        # first -- the sanitised filename used to be the key, so renaming a
        # recording orphaned everything ever decided about it, and re-keying
        # onto meeting ids is what fixed it. `name TEXT UNIQUE` plus a lookup by
        # `WHERE name=?` puts speakers in exactly that position: two spellings
        # are two people, and the split is undetectable downstream.
        row = conn.execute("SELECT id, name FROM speakers WHERE id=?",
                           (a.speaker_id,)).fetchone()
        if not row:
            raise SystemExit(f"no speaker with id {a.speaker_id} -- see `list`")
        sid, a.name = row[0], row[1]
    elif not a.name:
        raise SystemExit("give a name to create a new person, or --speaker <id> "
                         "to attach this voice to one already in the store")
    else:
        sid = None
    close = near_names(conn, a.name) if sid is None else []
    if close and not a.new:
        print(f"\"{a.name}\" is not in the store, but these are:")
        for other, why in close[:5]:
            n = conn.execute(
                "SELECT COUNT(DISTINCT c.meeting) FROM clusters c JOIN groups g"
                " ON g.id = c.group_id JOIN speakers s ON s.id = g.speaker_id"
                " WHERE s.name = ?", (other,)).fetchone()[0]
            oid = conn.execute("SELECT id FROM speakers WHERE name=?",
                               (other,)).fetchone()[0]
            print(f"   #{oid:<4} {other!r:34} {why}, in {n} meeting"
                  f"{'s' if n != 1 else ''}")
        raise SystemExit(
            "\nNaming creates a SECOND person and splits their voice between the\n"
            "two spellings, which nothing downstream can detect. Attach to the\n"
            f"existing one by id:   ./speakers name {a.group_id} --speaker "
            f"{conn.execute('SELECT id FROM speakers WHERE name=?', (close[0][0],)).fetchone()[0]}\n"
            "or pass --new if they really are someone else.")
    if sid is None:
        conn.execute("INSERT OR IGNORE INTO speakers(name, created_at) VALUES(?,?)",
                     (a.name, time.time()))
        sid = conn.execute("SELECT id FROM speakers WHERE name=?",
                           (a.name,)).fetchone()[0]
    conn.execute("UPDATE groups SET speaker_id=? WHERE id=?", (sid, a.group_id))
    members = conn.execute(
        "SELECT meeting, cluster, seconds FROM clusters WHERE group_id=?"
        " ORDER BY seconds DESC", (a.group_id,)).fetchall()
    if not members:
        raise SystemExit(f"group {a.group_id} has no members")
    # every voice in the group, as vectors, for the exemplars below
    import numpy as np

    import match_speakers as MS
    pool = []
    for meeting, cluster, secs in members:
        row = conn.execute("SELECT emb, dim FROM clusters WHERE meeting=? AND"
                           " cluster=? AND embed_model=?",
                           (meeting, cluster, EMBED_MODEL)).fetchone()
        pool.append((np.frombuffer(row[0], dtype=np.float32, count=row[1]), secs))
    # and an exemplar for the circumstance, which is what matching reads. Naming
    # the same person again under a different --condition adds a sub-profile
    # rather than overwriting: that is how "this is also her, on the phone"
    # becomes storable, and it is the difference between 22% wrong names on the
    # telephone arguments and 0.0-0.4%.
    if pool:
        if a.condition is not None:
            # a human said what these recordings have in common, so believe them
            v = MS.unit(sum(e * w for e, w in pool))
            MS.save_exemplar(conn, sid, a.condition, v, EMBED_MODEL,
                             float(sum(w for _, w in pool)))
        else:
            conn.commit()
            refresh_exemplars(conn, sid)
    conn.commit()
    cond = f" [{a.condition}]" if a.condition else ""
    print(f"group {a.group_id} is {a.name}{cond} -- {len(members)} clusters "
          f"across {len({m[0] for m in members})} meetings:")
    have = conn.execute(
        "SELECT condition, seconds FROM exemplars WHERE speaker_id=? AND"
        " embed_model=? ORDER BY seconds DESC", (sid, EMBED_MODEL)).fetchall()
    if len(have) > 1:
        print(f"   {a.name} is now stored under {len(have)} circumstances: "
              + ", ".join(f"{c or 'default'} ({s:.0f}s)" for c, s in have))
    for meeting, cluster, secs in members:
        print(f"   {secs:6.0f}s  {meeting[:48]:<48} {cluster}")



def cmd_review(a):
    """The naming queue: who is worth identifying next, and the evidence to do it.

    An unnamed voice is not a failure, it is a person nobody has introduced yet.
    What matters is the ORDER: a provisional identity spanning twelve meetings is
    one naming decision that fixes twelve transcripts, and a one-off is one that
    fixes one. Sorting by meetings covered puts the leverage first.

    Prints, per group, enough to name it without opening anything: how far it
    reaches, a sample of what they said -- usually enough on its own -- and the
    clips to listen to when it is not, since clips.py already cut those so a
    voice could be judged after the source audio is gone.
    """
    import glob
    import os

    conn = db()
    rows = conn.execute(
        "SELECT g.id, COUNT(c.id), COUNT(DISTINCT c.meeting), SUM(c.seconds)"
        " FROM groups g JOIN clusters c ON c.group_id = g.id"
        " WHERE g.embed_model=? AND g.speaker_id IS NULL"
        " GROUP BY g.id HAVING SUM(c.seconds) >= ?"
        " ORDER BY COUNT(DISTINCT c.meeting) DESC, SUM(c.seconds) DESC",
        (EMBED_MODEL, a.min_sec)).fetchall()
    if not rows:
        print("nothing waiting to be named.")
        return

    named = conn.execute("SELECT COUNT(*) FROM groups WHERE speaker_id IS NOT NULL"
                         " AND embed_model=?", (EMBED_MODEL,)).fetchone()[0]
    tot = sum(r[3] or 0 for r in rows)
    print(f"{len(rows)} voices waiting to be named, {tot/3600:.1f} h of speech "
          f"({named} people already named)\n")

    lib = os.environ.get("MS_LIBRARY", os.path.join(WORK, "library"))
    by_id = {os.path.basename(d).rsplit("-", 1)[-1]: d
             for d in glob.glob(os.path.join(lib, "*")) if os.path.isdir(d)}

    for gid, nclus, nmeet, secs in rows[: a.limit]:
        members = conn.execute(
            "SELECT meeting, cluster, seconds FROM clusters WHERE group_id=?"
            " ORDER BY seconds DESC", (gid,)).fetchall()
        print(f"  group {gid}: {secs/60:.0f} min across {nmeet} "
              f"meeting{'s' if nmeet != 1 else ''}")
        # what they said -- reading one line names most people outright
        said = ""
        for meeting, cluster, _ in members[:3]:
            d = by_id.get(meeting)
            if not d:
                continue
            tj = glob.glob(os.path.join(d, "*-transcript.json"))
            if not tj:
                continue
            try:
                segs = json.load(open(tj[0]))["segments"]
            except Exception:
                continue
            txt = " ".join(x["text"] for x in segs
                           if x.get("global") == cluster)[:200]
            if len(txt) > 60:
                said = txt
                break
        if said:
            print(f'      "{said.strip()}..."')
        top = members[0]
        d = by_id.get(top[0])
        clip = None
        if d:
            hit = sorted(glob.glob(os.path.join(d, "clips", f"{top[1]}-*.mp3")))
            clip = hit[0] if hit else None
        if clip:
            print(f"      listen: {clip}")
        print(f"      name it:  ./speakers name {gid} \"Their Name\"")
        print(f"      or attach: ./speakers name {gid} --speaker <id>   "
              f"(see `list`)")
        print()

    if len(rows) > a.limit:
        rest = sum(r[3] or 0 for r in rows[a.limit:]) / 3600
        print(f"  ... and {len(rows) - a.limit} more, {rest:.1f} h. "
              f"--limit to see them.")



def _profile_members(conn, sid):
    """Which recordings each sub-profile actually stands for. -> {cond: [rows]}

    The exemplars table stores the vectors, not the membership: a covering set
    is chosen and every cluster is pooled into its nearest pick, and only the
    result is written. So membership is recomputed here the same way it was
    decided, rather than adding a column that could disagree with the vectors it
    claims to describe.
    """
    import numpy as np

    import match_speakers as MS
    ex = conn.execute(
        "SELECT condition, emb, dim, seconds FROM exemplars WHERE speaker_id=?"
        " AND embed_model=? ORDER BY seconds DESC",
        (sid, EMBED_MODEL)).fetchall()
    if not ex:
        return {}, []
    E = np.stack([MS.unit(np.frombuffer(r[1], dtype=np.float32, count=r[2]))
                  for r in ex])
    rows = conn.execute(
        "SELECT c.meeting, c.cluster, c.seconds, c.emb, c.dim FROM clusters c"
        " JOIN groups g ON g.id = c.group_id WHERE g.speaker_id=? AND"
        " c.embed_model=? AND c.seconds > 0 ORDER BY c.seconds DESC",
        (sid, EMBED_MODEL)).fetchall()
    out = {r[0]: [] for r in ex}
    for meeting, cluster, secs, blob, dim in rows:
        v = MS.unit(np.frombuffer(blob, dtype=np.float32, count=dim))
        sims = E @ v
        j = int(np.argmax(sims))
        out[ex[j][0]].append((meeting, cluster, secs, float(sims[j])))
    return out, ex



def _measure_profiles(members, sample):
    """Ask the AUDIO what each sub-profile is, since geometry cannot say.

    Vector geometry decided the grouping and that is the right basis for it --
    matching is done on vectors. But `auto-7` names nothing. Opening a few of
    its recordings and measuring the channel is what turns it into something a
    person can read, and unlike the geometry it does not depend on what else
    happens to be in the library: a brick wall at 3.5 kHz is a phone call in any
    corpus.

    Only a few members per profile: this reads wavs, and a verdict that needs
    more than a handful to agree is not a verdict worth printing.
    """
    import glob
    import os

    import conditions as CO
    lib = os.environ.get("MS_LIBRARY", os.path.join(WORK, "library"))
    by_id = {os.path.basename(d).rsplit("-", 1)[-1]: d
             for d in glob.glob(os.path.join(lib, "*")) if os.path.isdir(d)}
    out = {}
    for cond, rows in members.items():
        descs = []
        for meeting, cluster, secs, _ in rows[:sample]:
            d = by_id.get(meeting)
            if not d:
                continue
            wav = glob.glob(os.path.join(d, "*-audio.wav"))
            tj = glob.glob(os.path.join(d, "*-transcript.json"))
            if not wav or not tj:
                continue
            try:
                segs = json.load(open(tj[0]))["segments"]
            except Exception:
                continue
            spans = [(float(x["start"]), float(x["end"])) for x in segs
                     if x.get("global") == cluster]
            if spans:
                descs.append(CO.describe(wav[0], spans))
        v = CO.agree(descs)
        if v:
            out[cond] = v
    return out


def cmd_profiles(a):
    """Show a person's sub-profiles, and what each one is actually made of.

    `auto-2` is a counter, not a description -- it says a second way of sounding
    was found and nothing about what it is. Listing the recordings behind it is
    what lets a human look and say "those are all the phone calls", and then
    rename it to say so.
    """
    conn = db()
    sid, name = _resolve_speaker(conn, a.who)
    members, ex = _profile_members(conn, sid)
    if not ex:
        print(f"{name} (#{sid}) has no sub-profiles yet.")
        return
    print(f"{name}  (#{sid}) -- {len(ex)} sub-profile"
          f"{'s' if len(ex) != 1 else ''}\n")
    measured = _measure_profiles(members, a.measure) if a.measure else {}
    suspects = []
    for cond, _, _, secs in ex:
        rows = members.get(cond, [])
        tag = "" if str(cond).startswith("auto-") else "   [named by you]"
        m = measured.get(cond)
        if m:
            lab, edge, share = m
            tag += ("   [%s, %.0f Hz, %.0f%% agree]" % (lab, edge, 100 * share))
        print(f"  {cond:<22} {secs/60:6.0f} min   {len(rows):>3} recording"
              f"{'s' if len(rows) != 1 else ''}{tag}")
        for meeting, cluster, sec, sim in rows[: a.show]:
            print(f"       {meeting:<12} {cluster:<5} {sec:6.0f}s   fit {sim:.2f}")
        if len(rows) > a.show:
            print(f"       ... and {len(rows) - a.show} more")
        # Duration matters more than fit here. A one-second fragment has an
        # unreliable vector and fitting badly says nothing; a two-minute cluster
        # at 0.05 is someone else. Flagging both buried the second in the first
        # -- 86 rows, almost all of them a second long.
        suspects += [(sim, meeting, cluster, sec, cond)
                     for meeting, cluster, sec, sim in rows
                     if sim < a.suspect and sec >= a.min_sec]
        print()

    # A member that fits none of this person's profiles is the interesting row,
    # and sorting by speech buries it. Either the linking put someone else in
    # here -- which nothing else surfaces -- or it is a circumstance so unlike
    # the others that it deserves its own profile. Both need a human.
    if suspects:
        suspects.sort()
        print(f"  {len(suspects)} recording"
              f"{'s' if len(suspects) != 1 else ''} fit no profile of theirs "
              f"(below {a.suspect:.2f}) -- likely someone else:")
        for sim, meeting, cluster, sec, cond in suspects[:10]:
            print(f"       {meeting:<12} {cluster:<5} {sec:6.0f}s   fit {sim:.2f}"
                  f"   (filed under {cond})")
        print()
    for cond, m in measured.items():
        lab, edge, share = m
        if str(cond).startswith("auto-") and share >= 0.8 and lab == "narrowband":
            print(f"  {cond} is {share:.0%} narrowband at {edge:.0f} Hz -- a "
                  f"phone or a headset.\n"
                  f"      ./speakers profile-rename {sid} {cond} narrowband\n")
    print("  rename one:  ./speakers profile-rename %s <condition> <new name>"
          % sid)


def cmd_profile_rename(a):
    """Give a discovered sub-profile a name that means something.

    A renamed profile stops being ours to redraw -- refresh_exemplars only
    deletes `auto-*` -- so this is also how you pin one against being recomputed.
    """
    conn = db()
    sid, name = _resolve_speaker(conn, a.who)
    row = conn.execute("SELECT id FROM exemplars WHERE speaker_id=? AND"
                       " condition=? AND embed_model=?",
                       (sid, a.old, EMBED_MODEL)).fetchone()
    if not row:
        have = [r[0] for r in conn.execute(
            "SELECT condition FROM exemplars WHERE speaker_id=? AND embed_model=?",
            (sid, EMBED_MODEL))]
        raise SystemExit(f"{name} has no sub-profile {a.old!r}. "
                         f"It has: {', '.join(map(str, have)) or '(none)'}")
    clash = conn.execute("SELECT 1 FROM exemplars WHERE speaker_id=? AND"
                         " condition=? AND embed_model=?",
                         (sid, a.new, EMBED_MODEL)).fetchone()
    if clash:
        raise SystemExit(f"{name} already has a sub-profile called {a.new!r}. "
                         f"Pick another name, or they should be merged.")
    conn.execute("UPDATE exemplars SET condition=? WHERE id=?", (a.new, row[0]))
    conn.commit()
    print(f"{name}: {a.old} is now {a.new!r}, and is no longer redrawn "
          f"automatically.")


def _resolve_speaker(conn, who):
    """-> (id, name). Accepts an id or a name, because both get typed."""
    if str(who).isdigit():
        row = conn.execute("SELECT id, name FROM speakers WHERE id=?",
                           (int(who),)).fetchone()
    else:
        row = conn.execute("SELECT id, name FROM speakers WHERE name=?",
                           (who,)).fetchone()
    if not row:
        raise SystemExit(f"no speaker {who!r} -- see `list`")
    return row[0], row[1]



def _exemplar(conn, sid, cond):
    row = conn.execute(
        "SELECT id, emb, dim, seconds FROM exemplars WHERE speaker_id=? AND"
        " condition=? AND embed_model=?", (sid, cond, EMBED_MODEL)).fetchone()
    if not row:
        have = [str(r[0]) for r in conn.execute(
            "SELECT condition FROM exemplars WHERE speaker_id=? AND embed_model=?"
            " ORDER BY seconds DESC", (sid, EMBED_MODEL))]
        raise SystemExit(f"no sub-profile {cond!r}. There is: "
                         f"{', '.join(have) or '(none)'}")
    return row


def cmd_profile_merge(a):
    """Fold two sub-profiles into one, because they are the same circumstance.

    The covering set splits whenever a recording is far from everything stored,
    which is the right default -- it cannot tell "the same room on a bad day"
    from "a different room". Only a person can, and `profiles --measure` is what
    shows it: two profiles measuring narrowband at the same edge are one
    circumstance that geometry happened to cut in half.

    Pooled by speech, so the longer profile dominates rather than the two being
    averaged as equals. A human-given name survives an auto- one, since it is
    the half somebody vouched for.
    """
    import numpy as np

    import match_speakers as MS
    conn = db()
    sid, name = _resolve_speaker(conn, a.who)
    ra, rb = _exemplar(conn, sid, a.first), _exemplar(conn, sid, a.second)
    if ra[0] == rb[0]:
        raise SystemExit("those are the same sub-profile")
    va = MS.unit(np.frombuffer(ra[1], dtype=np.float32, count=ra[2]))
    vb = MS.unit(np.frombuffer(rb[1], dtype=np.float32, count=rb[2]))
    wa, wb = float(ra[3] or 1.0), float(rb[3] or 1.0)
    keep = a.as_name
    if not keep:
        auto_a = str(a.first).startswith("auto-")
        auto_b = str(a.second).startswith("auto-")
        keep = (a.second if auto_a and not auto_b else
                a.first if auto_b and not auto_a else
                (a.first if wa >= wb else a.second))
    sim = float(va @ vb)
    conn.execute("DELETE FROM exemplars WHERE id IN (?,?)", (ra[0], rb[0]))
    MS.save_exemplar(conn, sid, keep, MS.unit(va * wa + vb * wb), EMBED_MODEL,
                     wa + wb)
    conn.commit()
    print(f"{name}: {a.first} + {a.second} -> {keep!r} "
          f"({(wa + wb)/60:.0f} min, the two were {sim:.2f} apart)")
    if sim < 0.5:
        print("  note: those were not very alike. If matching gets worse, "
              "`link --apply` rebuilds the auto- ones from scratch.")


def cmd_profile_split(a):
    """Break one sub-profile apart, because it is holding two circumstances.

    Splits its own members the same way profiles are discovered in the first
    place -- take the member least well covered by what is already picked, until
    every member is within `--cover` of something -- but applied to one profile
    and with a tighter bar, since the point is to find structure the original
    pass smoothed over.

    Results are auto- names: they were derived, not vouched for. Look with
    `profiles --measure` and rename the ones that mean something.
    """
    import numpy as np

    import match_speakers as MS
    conn = db()
    sid, name = _resolve_speaker(conn, a.who)
    _exemplar(conn, sid, a.cond)
    members, _ = _profile_members(conn, sid)
    rows = members.get(a.cond, [])
    if len(rows) < 2:
        raise SystemExit(f"{a.cond!r} has {len(rows)} recording(s) -- nothing "
                         f"to split")
    got = []
    for meeting, cluster, secs, _ in rows:
        r = conn.execute(
            "SELECT emb, dim FROM clusters WHERE meeting=? AND cluster=? AND"
            " embed_model=?", (meeting, cluster, EMBED_MODEL)).fetchone()
        if r:
            got.append((MS.unit(np.frombuffer(r[0], dtype=np.float32,
                                              count=r[1])), secs))
    if len(got) < 2:
        raise SystemExit("not enough of its recordings are still on file")
    E = np.stack([v for v, _ in got])
    w = np.array([s for _, s in got], dtype=float)
    picked = [int(np.argmax(w))]
    while len(picked) < min(a.into, len(got)):
        cov = (E @ E[picked].T).max(axis=1)
        if cov.min() >= a.cover:
            break
        picked.append(int(np.argmin(cov)))
    if len(picked) < 2:
        raise SystemExit(
            f"{a.cond!r} does not split: every recording is within "
            f"{a.cover:.2f} of the same one. Lower --cover to force it.")
    owner = (E @ E[picked].T).argmax(axis=1)
    # A sub-profile is used for MATCHING, so one built from a few seconds is a
    # bad reference that everything gets compared against. Splitting 171
    # recordings produced a 168 and two singletons of no duration; those are
    # outliers worth seeing in `profiles`, not references worth scoring against.
    # Fold anything under the enrolment floor into its next-nearest pick.
    for _ in range(len(picked)):
        tiny = [k for k in range(len(picked))
                if w[owner == k].sum() < MIN_ENROLL_SEC]
        if not tiny or len(picked) - len(tiny) < 1:
            break
        drop = tiny[0]
        keep = [k for k in range(len(picked)) if k != drop]
        sub = E[[picked[k] for k in keep]]
        for i in np.where(owner == drop)[0]:
            owner[i] = keep[int(np.argmax(sub @ E[i]))]
        picked = [picked[k] for k in keep]
        owner = np.array([keep.index(o) if o in keep else o for o in owner])
    if len(picked) < 2:
        raise SystemExit(
            f"{a.cond!r} does not usefully split: everything outside the main "
            f"group is under {MIN_ENROLL_SEC:.0f}s, too little to be a profile. "
            f"`profiles {sid}` shows those as suspects instead.")
    nxt = 1 + max([int(str(c[0]).split("-")[-1]) for c in conn.execute(
        "SELECT condition FROM exemplars WHERE speaker_id=? AND condition LIKE"
        " 'auto-%'", (sid,)) if str(c[0]).split("-")[-1].isdigit()] or [0])
    old = _exemplar(conn, sid, a.cond)
    conn.execute("DELETE FROM exemplars WHERE id=?", (old[0],))
    made = []
    for k in range(len(picked)):
        mine = np.where(owner == k)[0]
        if not len(mine):
            continue
        v = MS.unit((E[mine] * w[mine][:, None]).sum(axis=0))
        cond = "auto-%d" % (nxt + len(made))
        MS.save_exemplar(conn, sid, cond, v, EMBED_MODEL, float(w[mine].sum()))
        made.append((cond, len(mine), w[mine].sum()))
    conn.commit()
    print(f"{name}: {a.cond} -> {len(made)} sub-profiles")
    for cond, n, secs in made:
        print(f"   {cond:<12} {secs/60:5.0f} min  {n:>3} recordings")
    print(f"\n  look at them:  ./speakers profiles {sid} --measure")



def cmd_suggest(a):
    """For everything left unresolved in a meeting, show who it might be.

    Declining to name something is the right answer when the evidence is thin,
    but it throws away what the evidence WAS. A fragment that scored 0.58 against
    one person and 0.19 against everyone else is a very different situation from
    one that scored 0.31 against four people, and only the first is worth a
    human's second. So print the scores rather than the silence.

    Pooled per cluster, not per atom: the cluster is what a reader sees and what
    gets named, and pooling its speech is a better estimate of the voice than any
    one fragment of it was.
    """
    import glob

    import numpy as np

    import library as LIB
    import match_speakers as MS

    m = LIB.find(a.meeting, None)
    if not m:
        raise SystemExit(f"no meeting matching {a.meeting!r} -- see `meetings`")
    raw_p, npz_p = m.file("raw", "json"), m.file("embeddings", "npz")
    if not raw_p.exists() or not npz_p.exists():
        raise SystemExit(f"{m.title}: no embeddings on file -- transcribe it again")
    raw = json.loads(raw_p.read_text())
    z = np.load(str(npz_p), allow_pickle=True)
    atoms = MS.atoms_from(raw["segments"], z["emb"], z["seg_idx"])
    if not atoms:
        raise SystemExit("nothing embeddable in that meeting")

    conn = db()
    roster = [x.strip() for x in a.roster.split(",") if x.strip()]
    bank = MS.load_bank(conn, EMBED_MODEL, names=roster or None)
    if not len(bank):
        raise SystemExit("nobody enrolled yet -- name someone first")
    names, prov, sim = MS.assign(atoms, bank)

    # group the unresolved by the label a reader would see
    segs = raw["segments"]
    text = collections.defaultdict(list)
    for sgm in segs:
        text[(int(sgm["window"]), sgm["local_speaker"])].append(sgm)
    unres = collections.defaultdict(list)
    for i, at in enumerate(atoms):
        if names[i] is None:
            unres[prov[i] or "UNKNOWN"].append(i)
    if not unres:
        print(f"{m.title}: everything is named.")
        return

    A = MS.unit_rows(np.stack([at["v"] for at in atoms]))
    print(f"{m.title}: {len(unres)} unresolved\n")
    for tag, idxs in sorted(unres.items(),
                            key=lambda kv: -sum(atoms[i]["sec"] for i in kv[1])):
        secs = sum(atoms[i]["sec"] for i in idxs)
        w = np.array([atoms[i]["sec"] for i in idxs], dtype=np.float32)
        v = MS.unit((A[idxs] * w[:, None]).sum(axis=0))
        P = bank.score(v[None, :])[0]
        order = np.argsort(-P)[: a.top]
        # Margin, not a verdict. It is the useful number -- a fragment at 0.59
        # with the next candidate at 0.26 is a person, one at 0.47 with three
        # others inside 0.06 is noise -- but it does not settle anything on its
        # own. Tried as a verdict, "looks like X at margin 0.15" named the wrong
        # justice on the first fragment it was pointed at: 0.47 to one person
        # and 0.31 to the right one. Scores are evidence for a human; a headline
        # over them is a guess wearing a number.
        gap = float(P[order[0]] - P[order[1]]) if len(order) > 1 else float("nan")
        first = min(idxs, key=lambda i: atoms[i]["start"])
        said = " ".join(x["text"] for x in text[atoms[first]["key"]])[:88]
        print(f"  {tag:<8} {secs:5.0f}s  [{int(atoms[first]['start'])//60}:"
              f"{int(atoms[first]['start'])%60:02d}]  \"{said.strip()}\"")
        print(f"       best is {gap:.2f} clear of the next")
        for j in order:
            bar = "#" * int(round(20 * max(float(P[j]), 0)))
            flag = "  <- over the bar" if float(P[j]) >= MS.ACCEPT else ""
            print(f"       {bank.names[j][:26]:<26} {float(P[j]):5.2f}  "
                  f"{bar}{flag}")
        print(f"       {'(nobody)':<26} {'':5}  accept is {MS.ACCEPT:.2f}\n")



def cmd_groups(a):
    conn = db()
    rows = conn.execute(
        "SELECT g.id, s.name, COUNT(c.id), COUNT(DISTINCT c.meeting),"
        " SUM(c.seconds) FROM groups g"
        " LEFT JOIN clusters c ON c.group_id = g.id"
        " LEFT JOIN speakers s ON s.id = g.speaker_id"
        " WHERE g.embed_model=? GROUP BY g.id ORDER BY 4 DESC, 3 DESC",
        (EMBED_MODEL,)).fetchall()
    if not rows:
        print("no groups yet -- run: ./speakers link --apply")
        return
    print(f"{'group':>6}  {'name':24}{'clusters':>9}{'meetings':>9}{'speech':>9}")
    for gid, name, nc, nm, secs in rows:
        print(f"{gid:>6}  {(name or '-- unnamed --'):24}{nc:>9}{nm:>9}"
              f"{(secs or 0):>8.0f}s")


def cmd_list(a):
    """Who is on file, and how much speech each name is standing on.

    Counts EXEMPLARS. It counted prototypes, which no reader was left for -- and
    the speech total is the column the duplicate-name check in RUNBOOK.md reads
    ("two entries with near-identical speech time"), so it has to come from the
    table matching actually uses.
    """
    conn = db()
    rows = conn.execute(
        "SELECT s.id, s.name, COUNT(e.id), SUM(e.seconds) "
        "FROM speakers s LEFT JOIN exemplars e ON e.speaker_id=s.id "
        "GROUP BY s.id ORDER BY s.name").fetchall()
    if not rows:
        print("no speakers enrolled")
        return
    print(f"{'id':>4}  {'name':24}{'profiles':>9}{'speech':>9}")
    for sid, name, n, secs in rows:
        print(f"{sid:>4}  {name:24}{n:>9}{(secs or 0):>8.0f}s")


def cmd_rename(a):
    conn = db()
    conn.execute("UPDATE speakers SET name=? WHERE id=?", (a.name, a.speaker_id))
    conn.commit()
    print(f"speaker {a.speaker_id} is now {a.name}")


def cmd_forget(a):
    """Delete a person: their voiceprints, and their claim on every group.

    The cascade is written out here rather than left to ON DELETE CASCADE in the
    DDL, because the store on disk was created before any of this and CREATE
    TABLE IF NOT EXISTS will not retrofit a clause onto it. A store that cascades
    only if it happens to be new is the divergence, not the fix.

    exemplars, not prototypes: the line here deleted the table identify.py read,
    so forgetting someone left the voiceprints the MATCHER reads behind and the
    next recording still recognised them.

    Groups are emptied, not deleted. A group is a voice the linker found; the
    name was a human's answer about it, and withdrawing the answer does not
    unmake the voice. speaker_id NULL is what `review` looks for, so the group
    goes back into the queue to be named again.
    """
    conn = db()
    row = conn.execute("SELECT name FROM speakers WHERE id=?",
                       (a.speaker_id,)).fetchone()
    if not row:
        raise SystemExit(f"no speaker with id {a.speaker_id} -- see `list`")
    # Order is load-bearing now that PRAGMA foreign_keys is on: speakers last,
    # or the DELETE fails on whichever row still references it.
    #
    # And the list of referencing tables is ASKED FOR rather than written down.
    # Written down, it went stale the moment a table was added to the schema and
    # forget started failing on every store older than that change -- the same
    # shape as every other bug in this project, code holding a copy of a fact
    # that lives somewhere else.
    n_ex = conn.execute("DELETE FROM exemplars WHERE speaker_id=?",
                        (a.speaker_id,)).rowcount
    n_gr = conn.execute("UPDATE groups SET speaker_id=NULL WHERE speaker_id=?",
                        (a.speaker_id,)).rowcount
    for (tbl,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if tbl in ("speakers", "exemplars", "groups"):
            continue
        for fk in conn.execute("PRAGMA foreign_key_list(%s)" % tbl).fetchall():
            if fk[2] == "speakers":
                conn.execute("DELETE FROM %s WHERE %s=?" % (tbl, fk[3]),
                             (a.speaker_id,))
    conn.execute("DELETE FROM speakers WHERE id=?", (a.speaker_id,))
    conn.commit()
    print(f"deleted {row[0]} (speaker {a.speaker_id}) and {n_ex} voiceprint"
          f"{'' if n_ex == 1 else 's'}")
    if n_gr:
        print(f"{n_gr} group{'' if n_gr == 1 else 's'} back in the review queue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    lk = sub.add_parser("link", help="group voices across every meeting on file")
    lk.add_argument("--threshold", type=float, default=LINK_ACCEPT)
    lk.add_argument("--min-sec", type=float, default=MIN_LINK_SEC, dest="min_sec")
    lk.add_argument("--apply", action="store_true",
                    help="write the groups; without it this is a dry run")
    lk.set_defaults(fn=cmd_link)

    nm = sub.add_parser("name", help="name a group, and every meeting it spans")
    nm.add_argument("--new", action="store_true",
                    help="this really is a different person, even though the "
                         "name is close to one already in the store")
    nm.add_argument("--condition", default=None,
                    help="what made this recording sound the way it does -- "
                         "'telephone', 'far-field', '2015', 'headset', anything. "
                         "Naming the same person again under a different one "
                         "ADDS a sub-profile instead of replacing: one averaged "
                         "voiceprint cannot span a change of microphone, and "
                         "pretending it can names the wrong human.")
    nm.add_argument("group_id", type=int)
    nm.add_argument("name", nargs="?", default=None,
                    help="a name, to create a NEW person. To attach this voice "
                         "to someone already in the store, pass --speaker with "
                         "their id instead: the id is the identity, the name is "
                         "only a label on it.")
    nm.add_argument("--speaker", type=int, default=None, dest="speaker_id",
                    help="id of an existing speaker (see `list`). Preferred over "
                         "re-typing a name, which is how one person becomes two.")
    nm.set_defaults(fn=cmd_name)

    sub.add_parser("groups").set_defaults(fn=cmd_groups)
    rv = sub.add_parser("review", help="who is worth naming next, and why")
    rv.add_argument("--limit", type=int, default=15)
    rv.add_argument("--min-sec", type=float, default=MIN_LINK_SEC,
                    dest="min_sec")
    rv.set_defaults(fn=cmd_review)

    pr = sub.add_parser("profiles",
                        help="a person's sub-profiles, and what each is made of")
    pr.add_argument("who", help="speaker id or name")
    pr.add_argument("--show", type=int, default=5,
                    help="recordings to list per sub-profile")
    pr.add_argument("--suspect", type=float, default=0.5,
                    help="below this fit, a member probably is not this person")
    pr.add_argument("--measure", type=int, nargs="?", const=6, default=0,
                    metavar="N",
                    help="open N recordings per sub-profile and measure the "
                         "channel, so a discovered profile gets a label that "
                         "means something instead of a counter")
    pr.add_argument("--min-sec", type=float, default=MIN_LINK_SEC, dest="min_sec",
                    help="ignore members shorter than this when flagging: a "
                         "short clip has an unreliable vector by itself")
    pr.set_defaults(fn=cmd_profiles)

    pn = sub.add_parser("profile-rename",
                        help="name a discovered sub-profile, and pin it")
    pn.add_argument("who"); pn.add_argument("old"); pn.add_argument("new")
    pn.set_defaults(fn=cmd_profile_rename)

    pm = sub.add_parser("profile-merge",
                        help="two sub-profiles are the same circumstance")
    pm.add_argument("who"); pm.add_argument("first"); pm.add_argument("second")
    pm.add_argument("--as", dest="as_name", default=None,
                    help="what to call the result (default: keeps the "
                         "human-given name, or the larger one)")
    pm.set_defaults(fn=cmd_profile_merge)

    ps = sub.add_parser("profile-split",
                        help="one sub-profile is holding two circumstances")
    ps.add_argument("who"); ps.add_argument("cond")
    ps.add_argument("--into", type=int, default=2)
    ps.add_argument("--cover", type=float, default=0.86,
                    help="how tightly members must be covered before it stops "
                         "splitting; tighter than the discovery default, since "
                         "the point is structure that pass smoothed over")
    ps.set_defaults(fn=cmd_profile_split)

    sg = sub.add_parser("suggest",
                        help="who the unresolved voices in a meeting might be")
    sg.add_argument("meeting")
    sg.add_argument("--top", type=int, default=4,
                    help="candidates to show per unresolved cluster")
    sg.add_argument("--roster", default="",
                    help="restrict the candidates to these names")
    sg.set_defaults(fn=cmd_suggest)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    r = sub.add_parser("rename"); r.add_argument("speaker_id", type=int)
    r.add_argument("name"); r.set_defaults(fn=cmd_rename)

    f = sub.add_parser("forget"); f.add_argument("speaker_id", type=int)
    f.set_defaults(fn=cmd_forget)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
