"""Persistent speaker profiles: name a voice once, recognise it thereafter.

The store. Most people reach it through ./speakers at the repo root, which also
does who/play/clips (reading the transcript and audio) and routes naming through
identify.py. This module is the sqlite layer plus a direct CLI:

  speakers.py enroll  <meeting> G03 "Bob Smith"     name a cluster from a meeting
  speakers.py identify <meeting> [--roster "Bob Smith,Jane Doe"]
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
import argparse, json, os, sqlite3, time

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

# centroid-vs-centroid operating point. Measured error-free band on ICSI was
# [0.41, 0.86]; 0.55 sits mid-band. Not valid at any other aggregation level.
ACCEPT = 0.55
MARGIN = 0.10          # best must beat second best by this much
REVIEW = 0.40          # below ACCEPT but above this -> tentative, needs a human
MIN_ENROLL_SEC = 10.0

# Corpus-wide linking is a DIFFERENT level from identify and gets its own
# numbers, for the reason in this module's header: a threshold fitted at one
# level misfires at another. Two things make linking the harsher case. It scores
# every cluster against every other, so acceptance is a max over the whole
# corpus rather than over a roster -- the widest gallery there is. And a wrong
# link is not one bad row, it is one name spread across every meeting it joins.
#
# Measured on the podcast corpus, cross-meeting centroid pairs, truth by ear:
#
#   same person   (across 5 recordings)   0.754 - 0.889
#   IMPOSTOR      (a 24s cluster)         0.708 - 0.741   <- ACCEPT is 0.55
#
# ACCEPT would take that impostor without pausing. Duration is what made it
# dangerous: a 53-second cluster of a different audience member scored
# 0.521-0.547 against the same person, and every true pair came from clusters of
# 1395s or more. Short clusters give unstable centroids that drift toward
# whoever dominates the recording.
#
# Requiring 60s to be eligible drops the worst impostor from 0.741 to 0.458 and
# widens the gap to the nearest true pair from 0.013 to 0.296. The gate does far
# more work here than the threshold does, which is why it is not optional.
LINK_ACCEPT = 0.75

# How well a recording must match what we already store before it may become a
# sub-profile itself. Above ACCEPT, because this feeds back: see refresh_exemplars.
KEEP_EXEMPLAR = float(os.environ.get("MS_KEEP_EXEMPLAR", "0.72"))
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
    c.executescript("""
    CREATE TABLE IF NOT EXISTS speakers(
      id INTEGER PRIMARY KEY, name TEXT UNIQUE, created_at REAL);
    CREATE TABLE IF NOT EXISTS prototypes(
      id INTEGER PRIMARY KEY, speaker_id INTEGER, emb BLOB, dim INTEGER,
      embed_model TEXT, level TEXT, meeting TEXT, seconds REAL, created_at REAL,
      FOREIGN KEY(speaker_id) REFERENCES speakers(id));
    CREATE TABLE IF NOT EXISTS decisions(
      id INTEGER PRIMARY KEY, meeting TEXT, cluster TEXT, speaker_id INTEGER,
      score REAL, second REAL, threshold REAL, level TEXT, roster TEXT,
      outcome TEXT, created_at REAL);

    -- Every cluster the pipeline has ever produced, named or not. `prototypes`
    -- cannot hold these: it requires a speaker_id, and the whole point of
    -- linking is that a voice belongs to a group BEFORE anyone names it.
    CREATE TABLE IF NOT EXISTS clusters(
      id INTEGER PRIMARY KEY, meeting TEXT, cluster TEXT,
      emb BLOB, dim INTEGER, embed_model TEXT, seconds REAL,
      group_id INTEGER, created_at REAL,
      UNIQUE(meeting, cluster, embed_model));
    CREATE INDEX IF NOT EXISTS clusters_group ON clusters(group_id);

    -- An identity discovered from audio. speaker_id stays NULL until a human
    -- names it, and naming is then one UPDATE here rather than a rescan: every
    -- meeting in the group inherits the name through the join.
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


def cluster_centroids(meeting, run_dir=None):
    """-> {cluster_id: (centroid, seconds)} for one meeting's linked output."""
    run_dir = run_dir or os.path.join(WORK, "out")
    linked = json.load(open(f"{run_dir}/{meeting}_linked.json"))
    segs = linked["segments"] if isinstance(linked, dict) else linked
    z = np.load(f"{run_dir}/{meeting}_emb.npz", allow_pickle=True)
    emb = {int(i): z["emb"][r] for r, i in enumerate(z["seg_idx"])}

    acc = {}
    for i, s in enumerate(segs):
        g = s.get("global") or s.get("speaker")
        # G-1 is the linker's leftover bucket: segments with too little audio to
        # cluster. It is not a person and must never be enrolled or matched.
        if not g or g == "UNASSIGNED" or str(g).startswith("G-"):
            continue
        a = acc.setdefault(g, [[], 0.0])
        a[1] += s["end"] - s["start"]
        if i in emb:
            a[0].append(emb[i])
    out = {}
    for g, (vs, secs) in acc.items():
        if not vs:
            continue
        v = np.mean(vs, axis=0)
        out[g] = (v / (np.linalg.norm(v) + 1e-9), secs)
    return out


def gallery(conn, roster=None):
    """-> [(speaker_id, name, centroid)]; roster restricts the candidate pool."""
    q = ("SELECT s.id, s.name, p.emb, p.dim FROM speakers s "
         "JOIN prototypes p ON p.speaker_id = s.id WHERE p.embed_model = ?")
    args = [EMBED_MODEL]
    if roster:
        q += " AND s.name IN (%s)" % ",".join("?" * len(roster))
        args += roster
    rows = {}
    for sid, name, blob, dim in conn.execute(q, args):
        rows.setdefault((sid, name), []).append(
            np.frombuffer(blob, dtype=np.float32, count=dim))
    out = []
    for (sid, name), vs in rows.items():
        v = np.mean(vs, axis=0)
        out.append((sid, name, v / (np.linalg.norm(v) + 1e-9)))
    return out


def enroll_centroid(conn, name, v, secs, meeting, cluster):
    """Store a voiceprint under `name`. Shared by the CLI and the folder pipeline."""
    conn.execute("INSERT OR IGNORE INTO speakers(name, created_at) VALUES(?,?)",
                 (name, time.time()))
    sid = conn.execute("SELECT id FROM speakers WHERE name=?", (name,)).fetchone()[0]
    conn.execute("INSERT INTO prototypes(speaker_id, emb, dim, embed_model, level,"
                 " meeting, seconds, created_at) VALUES(?,?,?,?,?,?,?,?)",
                 (sid, v.astype(np.float32).tobytes(), len(v), EMBED_MODEL,
                  "centroid", f"{meeting}:{cluster}", secs, time.time()))
    conn.commit()
    return sid


def cmd_enroll(a):
    conn = db()
    cents = cluster_centroids(a.meeting, a.run_dir)
    if a.cluster not in cents:
        raise SystemExit(f"cluster {a.cluster} not found; have {sorted(cents)}")
    v, secs = cents[a.cluster]
    if secs < MIN_ENROLL_SEC and not a.force:
        raise SystemExit(f"only {secs:.1f}s of speech; need {MIN_ENROLL_SEC:.0f}s "
                         f"(--force to override)")
    sid = enroll_centroid(conn, a.name, v, secs, a.meeting, a.cluster)
    n = conn.execute("SELECT COUNT(*) FROM prototypes WHERE speaker_id=?",
                     (sid,)).fetchone()[0]
    print(f"enrolled {a.name} from {a.meeting}:{a.cluster} "
          f"({secs:.0f}s speech, {n} session{'s' if n > 1 else ''} on file)")


def cmd_identify(a):
    conn = db()
    roster = [x.strip() for x in a.roster.split(",")] if a.roster else None
    G = gallery(conn, roster)
    if not G:
        raise SystemExit("nobody enrolled yet" + (" matching that roster" if roster else ""))
    cents = cluster_centroids(a.meeting, a.run_dir)
    print(f"{a.meeting}: {len(cents)} speakers found, "
          f"{len(G)} candidate{'s' if len(G) != 1 else ''}"
          + (f" (roster: {', '.join(roster)})" if roster else " (whole database)"))
    print(f"\n{'cluster':9}{'speech':>8}  {'identified as':22}{'score':>7}{'2nd':>7}")

    taken = {}
    for g in sorted(cents, key=lambda k: -cents[k][1]):
        v, secs = cents[g]
        scored = sorted(((float(v @ c), sid, name) for sid, name, c in G), reverse=True)
        best, second = scored[0], (scored[1] if len(scored) > 1 else (0.0, None, None))
        score, sid, name = best
        if score < REVIEW:
            label, outcome = "UNKNOWN", "unknown"
        elif score < ACCEPT or (score - second[0]) < MARGIN:
            label, outcome = f"? {name}", "review"
        elif sid in taken:
            # one-to-one: a person cannot be two clusters in one meeting
            label, outcome = f"? {name} (dup)", "review"
        else:
            label, outcome = name, "accept"
            taken[sid] = g
        print(f"{g:9}{secs:7.0f}s  {label:22}{score:7.3f}{second[0]:7.3f}")
        conn.execute("INSERT INTO decisions(meeting, cluster, speaker_id, score,"
                     " second, threshold, level, roster, outcome, created_at)"
                     " VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (a.meeting, g, sid if outcome == "accept" else None, score,
                      second[0], ACCEPT, "centroid",
                      ",".join(roster) if roster else None, outcome, time.time()))
    conn.commit()
    print(f"\naccept >= {ACCEPT} with margin >= {MARGIN}; "
          f"{REVIEW}-{ACCEPT} needs review; below {REVIEW} is UNKNOWN")


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
                " emb=excluded.emb, dim=excluded.dim, seconds=excluded.seconds",
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
    # promoted to a prototype starts attracting things like itself, and the
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
        ok = np.where((E @ H.T).max(axis=1) >= KEEP_EXEMPLAR)[0]
        if len(ok):
            E, w = E[ok], w[ok]
    picked = [int(np.argmax(w))]
    while len(picked) < 24:
        cov = (E @ E[picked].T).max(axis=1)
        if cov.min() >= MS.ACCEPT:
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
        for r in m:                              # inherit an existing name
            if keep is not None:
                break
            if r[6]:
                row = conn.execute("SELECT speaker_id FROM groups WHERE id=?",
                                   (r[6],)).fetchone()
                if row and row[0] is not None:
                    keep = r[6]
                    break
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


def cmd_name(a):
    """Name a group -- and with it every meeting the group already spans."""
    conn = db()
    conn.execute("INSERT OR IGNORE INTO speakers(name, created_at) VALUES(?,?)",
                 (a.name, time.time()))
    sid = conn.execute("SELECT id FROM speakers WHERE name=?", (a.name,)).fetchone()[0]
    conn.execute("UPDATE groups SET speaker_id=? WHERE id=?", (sid, a.group_id))
    members = conn.execute(
        "SELECT meeting, cluster, seconds FROM clusters WHERE group_id=?"
        " ORDER BY seconds DESC", (a.group_id,)).fetchall()
    if not members:
        raise SystemExit(f"group {a.group_id} has no members")
    # a named group is also an enrolment: give identify the voiceprints too
    import numpy as np

    import match_speakers as MS
    pool = []
    for meeting, cluster, secs in members:
        row = conn.execute("SELECT emb, dim FROM clusters WHERE meeting=? AND"
                           " cluster=? AND embed_model=?",
                           (meeting, cluster, EMBED_MODEL)).fetchone()
        conn.execute("INSERT INTO prototypes(speaker_id, emb, dim, embed_model,"
                     " level, meeting, seconds, created_at)"
                     " VALUES(?,?,?,?,?,?,?,?)",
                     (sid, row[0], row[1], EMBED_MODEL, "centroid",
                      f"{meeting}:{cluster}", secs, time.time()))
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
        print("no groups yet -- run: speakers.py link --apply")
        return
    print(f"{'group':>6}  {'name':24}{'clusters':>9}{'meetings':>9}{'speech':>9}")
    for gid, name, nc, nm, secs in rows:
        print(f"{gid:>6}  {(name or '-- unnamed --'):24}{nc:>9}{nm:>9}"
              f"{(secs or 0):>8.0f}s")


def cmd_list(a):
    conn = db()
    rows = conn.execute(
        "SELECT s.id, s.name, COUNT(p.id), SUM(p.seconds), MAX(p.created_at) "
        "FROM speakers s LEFT JOIN prototypes p ON p.speaker_id=s.id "
        "GROUP BY s.id ORDER BY s.name").fetchall()
    if not rows:
        print("no speakers enrolled")
        return
    print(f"{'id':>4}  {'name':24}{'sessions':>9}{'speech':>9}")
    for sid, name, n, secs, _ in rows:
        print(f"{sid:>4}  {name:24}{n:>9}{(secs or 0):>8.0f}s")


def cmd_rename(a):
    conn = db()
    conn.execute("UPDATE speakers SET name=? WHERE id=?", (a.name, a.speaker_id))
    conn.commit()
    print(f"speaker {a.speaker_id} is now {a.name}")


def cmd_forget(a):
    conn = db()
    conn.execute("DELETE FROM prototypes WHERE speaker_id=?", (a.speaker_id,))
    conn.execute("DELETE FROM speakers WHERE id=?", (a.speaker_id,))
    conn.commit()
    print(f"deleted speaker {a.speaker_id} and their voiceprints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll"); e.add_argument("meeting"); e.add_argument("cluster")
    e.add_argument("name"); e.add_argument("--force", action="store_true")
    e.set_defaults(fn=cmd_enroll)

    i = sub.add_parser("identify"); i.add_argument("meeting")
    i.add_argument("--roster", default=None,
                   help="comma-separated names known to be in this meeting")
    i.set_defaults(fn=cmd_identify)

    lk = sub.add_parser("link", help="group voices across every meeting on file")
    lk.add_argument("--threshold", type=float, default=LINK_ACCEPT)
    lk.add_argument("--min-sec", type=float, default=MIN_LINK_SEC, dest="min_sec")
    lk.add_argument("--apply", action="store_true",
                    help="write the groups; without it this is a dry run")
    lk.set_defaults(fn=cmd_link)

    nm = sub.add_parser("name", help="name a group, and every meeting it spans")
    nm.add_argument("--condition", default=None,
                    help="what made this recording sound the way it does -- "
                         "'telephone', 'far-field', '2015', 'headset', anything. "
                         "Naming the same person again under a different one "
                         "ADDS a sub-profile instead of replacing: one averaged "
                         "voiceprint cannot span a change of microphone, and "
                         "pretending it can names the wrong human.")
    nm.add_argument("group_id", type=int); nm.add_argument("name")
    nm.set_defaults(fn=cmd_name)

    sub.add_parser("groups").set_defaults(fn=cmd_groups)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    r = sub.add_parser("rename"); r.add_argument("speaker_id", type=int)
    r.add_argument("name"); r.set_defaults(fn=cmd_rename)

    f = sub.add_parser("forget"); f.add_argument("speaker_id", type=int)
    f.set_defaults(fn=cmd_forget)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
