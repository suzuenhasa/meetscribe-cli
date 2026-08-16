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
    """)
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

    sub.add_parser("list").set_defaults(fn=cmd_list)

    r = sub.add_parser("rename"); r.add_argument("speaker_id", type=int)
    r.add_argument("name"); r.set_defaults(fn=cmd_rename)

    f = sub.add_parser("forget"); f.add_argument("speaker_id", type=int)
    f.set_defaults(fn=cmd_forget)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
