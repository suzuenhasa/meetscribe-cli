#!/usr/bin/env python3
"""meetscribe UI — a local web app over the transcription pipeline.

  ./ui                                 the library in ./library
  ./ui ~/meetings --port 8800          somewhere else, on another port
  ./ui --host msbox                    send the GPU work to a box over ssh

Started through ./ui rather than directly, so it picks up the same interpreter
setup.sh installed the pipeline into.

Standard library only, so it runs anywhere the pipeline does without a second
dependency set to install or keep in step.

The library is a directory of meetings. Each is a linked transcript JSON beside
its audio, paired by basename, so the folder stays readable and portable -- you
can copy it, back it up, or hand-edit it without going through this server.
"""
import argparse
import ast
import json
import math
import mimetypes
import os
import re
import shutil
import sqlite3
import sys
import struct
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import backend as backend_mod

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm")

HOME = HERE.parent                      # the meetscribe-cli checkout
PIPE = HOME / "pipeline"
BACKUPS = HOME / "backups"
# The pipeline's own modules are the source of truth for the profile store's
# schema and thresholds. Importing them beats re-describing them here: this file
# used to carry its own copy of the DDL and scrape the constants out of the
# source with a regex, so a change on either side silently desynchronised.
sys.path.insert(0, str(PIPE))

STATE = {"library": None, "backend": None, "cfg": {}, "queue": [], "lock": threading.Lock(),
         "settings": {}, "facts": {}}


# ------------------------------------------------------------------ library
def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "meeting"


def audio_for(js: Path):
    """The recording beside a transcript, matched by basename."""
    for ext in AUDIO_EXT:
        p = js.with_suffix(ext)
        if p.exists():
            return p
    # transcripts imported before their audio was copied in still record where
    # they came from; use it only if it is readable from here.
    try:
        rec = json.loads(js.read_text()).get("audio")
        if rec and Path(rec).exists():
            return Path(rec)
    except Exception:
        pass
    return None


def speaker_label(gid):
    if gid in (None, "", "G-1"):
        return "Unknown"
    m = re.match(r"G(-?\d+)$", str(gid))
    return f"Speaker {int(m.group(1)) + 1}" if m else str(gid)


def read_meeting(js: Path, full=False, names=None):
    d = json.loads(js.read_text())
    if not isinstance(d, dict) or "segments" not in d:
        raise ValueError("not a transcript")
    segs = d.get("segments", [])
    dur = float(d.get("duration_s") or 0)
    # {cluster: {"id", "name"}} for this meeting, from the profile store. A
    # cluster nobody has placed keeps its "Speaker N" label.
    if names is None:
        names = meeting_names(js)

    by = {}
    for s in segs:
        g = s.get("global", "G-1")
        by[g] = by.get(g, 0.0) + max(0.0, float(s["end"]) - float(s["start"]))
    clusters = {g: t for g, t in by.items() if g != "G-1"}

    st = js.stat()
    meta = {
        "id": js.stem,
        "title": js.stem,
        "duration_s": dur,
        "n_turns": len(segs),
        "n_speakers": len(clusters),
        "n_named": sum(1 for g in clusters if g in names),
        "coverage": d.get("coverage"),
        "window_s": d.get("window_s"),
        "has_audio": audio_for(js) is not None,
        "mtime": st.st_mtime,
        "date": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%d %b %Y"),
        "speakers": [
            {"id": g, "name": (names[g]["name"] if g in names else speaker_label(g)),
             "named": g in names, "speaker_id": names.get(g, {}).get("id"),
             "seconds": round(t, 1),
             "share": (t / sum(clusters.values())) if clusters else 0}
            for g, t in sorted(clusters.items(), key=lambda kv: -kv[1])
        ],
    }
    if not full:
        return meta

    # Collapse consecutive segments by the same speaker into readable turns,
    # keeping every segment's own timestamp so a click can seek to the line.
    turns, cur = [], None
    for s in segs:
        g = s.get("global", "G-1")
        line = {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                "text": (s.get("text") or "").strip()}
        if not line["text"]:
            continue
        if cur and cur["speaker"] == g and line["start"] - cur["end"] < 12:
            cur["lines"].append(line)
            cur["end"] = line["end"]
        else:
            cur = {"speaker": g, "name": (names[g]["name"] if g in names
                                          else speaker_label(g)),
                   "named": g in names, "start": line["start"],
                   "end": line["end"], "lines": [line]}
            turns.append(cur)
    meta["turns"] = turns
    return meta


def library_json():
    """Transcript files in the library, newest first. Sidecars the pipeline and
    this server write beside them are not meetings."""
    lib = STATE["library"]
    out = [js for js in lib.glob("*.json")
           if not js.name.startswith(".")
           and not js.name.endswith((".raw.json", ".meta.json", ".emb.json"))]
    return sorted(out, key=lambda p: -p.stat().st_mtime)


def library_meetings():
    names = all_cluster_names()
    keys = meeting_keys()
    out = []
    for js in library_json():
        try:
            out.append(read_meeting(js, names=names_for(js, names, keys)))
        except Exception:
            continue
    out.sort(key=lambda m: -m["mtime"])
    return out


def library_stats(meetings):
    total = sum(m["duration_s"] for m in meetings)
    voices, named = set(), set()
    for m in meetings:
        for s in m["speakers"]:
            voices.add((m["id"], s["id"]))
            if s["named"]:
                named.add((m["id"], s["id"]))
    return {
        "meetings": len(meetings),
        "hours": round(total / 3600, 1),
        "voices": len(voices),
        "named": len(named),
        "unnamed": len(voices) - len(named),
        "profiles": profile_count(),
        "backend": STATE["backend"].name,
        "backend_up": STATE["cfg"].get("backend_up"),
    }


# ----------------------------------------------------------- pipeline facts
# The pipeline modules import numpy (and vllm), so they cannot be imported from
# here: this server is standard library only, so that it runs on a machine that
# has neither. Reading the assignments out of the source keeps one source of
# truth for the numbers instead of a second copy that can drift.
FALLBACK = {"ACCEPT": 0.55, "MARGIN": 0.10, "REVIEW": 0.40, "MIN_ENROLL_SEC": 10.0,
            "EMBED_MODEL": "wespeaker-resnet293-LM",
            "MODEL": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "SILENCE_GATE_DB": -70.0}


def pipeline_facts():
    """The thresholds the pipeline actually uses, read from the pipeline.

    These were previously recovered by regex from the source text, which worked
    until someone wrote a constant in a form the pattern did not match -- and
    then reported a plausible wrong number instead of failing. FALLBACK still
    exists for the case where the pipeline is not installed beside the UI.
    """
    f = dict(FALLBACK)
    try:
        import speakers as S
        for k in ("ACCEPT", "MARGIN", "REVIEW", "MIN_ENROLL_SEC", "EMBED_MODEL"):
            if hasattr(S, k):
                f[k] = getattr(S, k)
        f["source"] = str(PIPE / "speakers.py")
    except Exception:
        f["source"] = "defaults (pipeline not importable from here)"
    return f


# --------------------------------------------------------------- preferences
def settings_file():
    # dot-prefixed: it lives in the library but it is not part of it
    return STATE["library"] / ".ui-settings.json"


def load_settings():
    try:
        d = json.loads(settings_file().read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings():
    try:
        settings_file().write_text(json.dumps(STATE["settings"], indent=1, sort_keys=True))
    except Exception:
        traceback.print_exc()


def tun():
    """Recognition thresholds actually in force here: the pipeline's compiled-in
    values, plus whatever Settings has overridden."""
    f, s = STATE["facts"], STATE["settings"]
    return {k: float(s.get(k, f[c])) for k, c in
            (("accept", "ACCEPT"), ("margin", "MARGIN"),
             ("review", "REVIEW"), ("min_enroll_sec", "MIN_ENROLL_SEC"))}


def glossary_file():
    return STATE["library"] / "glossary.txt"


def glossary_terms():
    try:
        lines = glossary_file().read_text().splitlines()
    except Exception:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def save_glossary(terms):
    glossary_file().write_text(
        "# proper nouns the transcriber should get right, one per line\n"
        + "\n".join(terms) + ("\n" if terms else ""))


# ------------------------------------------------------------ speaker store
# A copy of pipeline/speakers.py's DDL, kept ONLY as a fallback for when the
# pipeline is not importable from here. speaker_db() below prefers speakers.db()
# so there is normally one definition, not two -- the CLI and this server open
# the same file, and two hand-maintained schemas for it is a bug waiting to
# happen rather than a design.
SCHEMA = """
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
"""

# outcomes written by a person here, on top of the pipeline's own
# accept / review / unknown. The log is append-only; the newest row wins.
HUMAN = ("accept", "left-unknown")


def db_path():
    p = STATE["cfg"].get("speaker_db")
    return Path(p) if p else STATE["library"] / "speakers.db"


def ensure_db():
    """The store lives in the library so it travels with it. On a first run,
    seed it from the newest backup rather than starting empty. A store named
    explicitly on the command line is left alone -- if you point at a file, you
    mean that file."""
    p = db_path()
    if not p.exists() and not STATE["cfg"].get("speaker_db") and BACKUPS.is_dir():
        seeds = sorted(BACKUPS.glob("speakers.db*"))
        if seeds:
            shutil.copy2(seeds[-1], p)
            print(f"  profiles  seeded from {seeds[-1].name}")
    return p


def sdb():
    """Open the profile store, letting the pipeline define its own schema.

    speakers.db() applies the real DDL, so the CLI and this server cannot drift
    apart. SCHEMA above is the fallback for a UI running without the pipeline
    beside it -- read-only browsing of an existing library still works then.
    """
    path = str(ensure_db())
    try:
        import speakers as S
        return S.db(path)
    except Exception:
        c = sqlite3.connect(path, timeout=10)
        c.executescript(SCHEMA)
        return c


def profile_count():
    try:
        with sdb() as c:
            return c.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
    except Exception:
        return 0


# ------------------------------------------------------------------- vectors
_DTYPE = {"f2": "e", "f4": "f", "f8": "d", "i1": "b", "i2": "h", "i4": "i",
          "i8": "q", "u1": "B", "u2": "H", "u4": "I", "u8": "Q"}


def _npy(buf):
    """Minimal .npy reader. Identification is cosine arithmetic and has to run
    wherever the server runs, including a machine with no numpy on it."""
    if buf[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy member")
    if buf[6] == 1:
        hlen, off = struct.unpack("<H", buf[8:10])[0], 10
    else:
        hlen, off = struct.unpack("<I", buf[8:12])[0], 12
    head = ast.literal_eval(buf[off:off + hlen].decode("latin1"))
    off += hlen
    descr = str(head["descr"])
    if descr.startswith(">") or head.get("fortran_order"):
        raise ValueError("unsupported .npy layout %r" % descr)
    code = _DTYPE.get(descr.lstrip("<=|"))
    if not code:
        raise ValueError("unsupported .npy dtype %r" % descr)
    n = 1
    for d in head["shape"]:
        n *= int(d)
    vals = struct.unpack("<%d%s" % (n, code), buf[off:off + n * struct.calcsize(code)])
    return tuple(head["shape"]), list(vals)


def read_npz(path):
    out = {}
    with zipfile.ZipFile(str(path)) as z:
        for n in z.namelist():
            if not n.endswith(".npy"):
                continue
            try:
                out[n[:-4]] = _npy(z.read(n))
            except Exception:
                continue          # e.g. the JSON meta blob; not needed here
    return out


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) + 1e-9
    return [x / n for x in v]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _blob(v, dim):
    return struct.unpack("<%df" % dim, bytes(v)[:4 * dim])


def bars(vec, n=48):
    """A voiceprint the Voices screen can draw: the stored 256-d centroid binned
    into n contiguous bins, each the RMS of its dimensions, scaled to the peak.
    Deterministic and reversible-ish -- the same vector always draws the same
    shape, and two similar voices look similar."""
    d = len(vec)
    if not d:
        return []
    out = []
    for k in range(n):
        a, b = k * d // n, (k + 1) * d // n
        seg = vec[a:b] or [0.0]
        out.append(math.sqrt(sum(x * x for x in seg) / len(seg)))
    mx = max(out) or 1.0
    return [round(x / mx, 4) for x in out]


def gallery(conn):
    """[{id, name, centroid}] -- one averaged, L2-normalised centroid per person,
    exactly as bench/speakers.py gallery() builds it."""
    model = STATE["facts"]["EMBED_MODEL"]
    rows = {}
    for sid, name, blob, dim in conn.execute(
            "SELECT s.id, s.name, p.emb, p.dim FROM speakers s"
            " JOIN prototypes p ON p.speaker_id = s.id WHERE p.embed_model = ?",
            (model,)):
        rows.setdefault((sid, name), []).append(_blob(blob, dim))
    out = []
    for (sid, name), vs in rows.items():
        mean = [sum(col) / len(vs) for col in zip(*vs)]
        out.append({"id": sid, "name": name, "centroid": _unit(mean)})
    out.sort(key=lambda r: r["name"].lower())
    return out


# ------------------------------------------------------- meetings <-> store
def meeting_keys():
    """{identification key: library meeting id}.

    The pipeline keys a meeting by the basename of its audio; the UI keys it by
    the basename of its transcript. They are normally the same file, but a
    transcript imported on its own records where it came from, so both spellings
    are accepted. Keys that match nothing in the library stay unresolved rather
    than being guessed at."""
    out = {}
    for js in library_json():
        out.setdefault(js.stem, js.stem)
        try:
            rec = json.loads(js.read_text()).get("audio")
        except Exception:
            rec = None
        if rec:
            out.setdefault(Path(rec).stem, js.stem)
    return out


def all_cluster_names():
    """{meeting key: {cluster: {"id", "name"}}} -- who each cluster was decided
    to be. Enrolment names a cluster outright (prototypes record meeting:cluster);
    after that the newest decision wins."""
    out = {}
    try:
        conn = sdb()
    except Exception:
        return out
    with conn:
        for meeting, sid, name in conn.execute(
                "SELECT p.meeting, s.id, s.name FROM prototypes p"
                " JOIN speakers s ON s.id = p.speaker_id ORDER BY p.created_at"):
            mt, _, cl = (meeting or "").partition(":")
            if cl:
                out.setdefault(mt, {})[cl] = {"id": sid, "name": name}
        for meeting, cluster, sid, name, outcome in conn.execute(
                "SELECT d.meeting, d.cluster, d.speaker_id, s.name, d.outcome"
                " FROM decisions d LEFT JOIN speakers s ON s.id = d.speaker_id"
                " WHERE d.id IN (SELECT MAX(id) FROM decisions GROUP BY meeting, cluster)"):
            if outcome == "accept" and sid is not None and name:
                out.setdefault(meeting, {})[cluster] = {"id": sid, "name": name}
            else:
                out.get(meeting, {}).pop(cluster, None)
    conn.close()
    return out


def names_for(js, names, keys):
    merged = {}
    for key, stem in keys.items():
        if stem == js.stem:
            merged.update(names.get(key, {}))
    return merged


def meeting_names(js):
    return names_for(js, all_cluster_names(), meeting_keys())


def cluster_detail(js):
    """{cluster: {seconds, n_lines, samples}} straight from the transcript."""
    try:
        segs = json.loads(js.read_text()).get("segments", [])
    except Exception:
        return {}
    out = {}
    for s in segs:
        g = s.get("global") or s.get("speaker")
        if not g or str(g).startswith("G-"):
            continue
        d = out.setdefault(g, {"seconds": 0.0, "n_lines": 0, "samples": []})
        d["seconds"] += max(0.0, float(s["end"]) - float(s["start"]))
        d["n_lines"] += 1
        text = (s.get("text") or "").strip()
        # a sample is only useful if there is something to read and hear
        if text and len(text) > 24 and len(d["samples"]) < 3:
            d["samples"].append({"t": round(float(s["start"]), 2), "text": text})
    for d in out.values():
        d["seconds"] = round(d["seconds"], 1)
    return out


def meeting_centroids(stem):
    """({cluster: {centroid, seconds}}, reason). The embeddings are what make a
    voice matchable; without them a meeting can be read but not identified."""
    js = STATE["library"] / f"{stem}.json"
    npz = STATE["library"] / f"{stem}.emb.npz"
    if not js.exists():
        return {}, "no-transcript"
    if not npz.exists():
        return {}, "no-embeddings"
    try:
        z = read_npz(npz)
        (n, dim), flat = z["emb"]
        idx = z["seg_idx"][1]
        segs = json.loads(js.read_text()).get("segments", [])
    except Exception as e:
        return {}, "unreadable-embeddings: %s" % e
    emb = {int(i): flat[r * dim:(r + 1) * dim] for r, i in enumerate(idx)}

    acc = {}
    for i, s in enumerate(segs):
        g = s.get("global") or s.get("speaker")
        # G-1 is the linker's leftover bucket: too little audio to cluster. It is
        # not a person and must never be enrolled or matched.
        if not g or g == "UNASSIGNED" or str(g).startswith("G-"):
            continue
        a = acc.setdefault(g, [[], 0.0])
        a[1] += float(s["end"]) - float(s["start"])
        if i in emb:
            a[0].append(emb[i])
    out = {}
    for g, (vs, secs) in acc.items():
        if not vs:
            continue
        mean = [sum(col) / len(vs) for col in zip(*vs)]
        out[g] = {"centroid": _unit(mean), "seconds": round(secs, 1)}
    return out, (None if out else "no-embedded-clusters")


def band(score, second, t):
    """The pipeline's own decision rule, at the thresholds in force."""
    if score < t["review"]:
        return "unknown"
    if score < t["accept"]:
        return "review"
    if score - second < t["margin"]:
        return "margin"
    return "accept"


def score_cluster(vec, G):
    scored = sorted(((_dot(vec, c["centroid"]), c["id"], c["name"]) for c in G),
                    reverse=True)
    return [{"id": sid, "name": nm, "score": round(sc, 4)} for sc, sid, nm in scored]


# -------------------------------------------------------------------- voices
def fmt_date(ts, short=False):
    if not ts:
        return None
    d = datetime.fromtimestamp(ts)
    return f"{d.day} {d:%b}" if short else f"{d.day} {d:%b %Y}"


def voice_rows(conn):
    model = STATE["facts"]["EMBED_MODEL"]
    protos = {}
    for sid, blob, dim, mdl, meeting, secs, at in conn.execute(
            "SELECT speaker_id, emb, dim, embed_model, meeting, seconds, created_at"
            " FROM prototypes ORDER BY created_at"):
        mt, _, cl = (meeting or "").partition(":")
        protos.setdefault(sid, []).append(
            {"vec": list(_blob(blob, dim)) if mdl == model else None, "dim": dim,
             "model": mdl, "meeting": mt or None, "cluster": cl or None,
             "seconds": round(secs or 0.0, 1), "at": at})

    heard, seen = {}, {}
    for sid, at, n in conn.execute(
            "SELECT speaker_id, MAX(created_at), COUNT(DISTINCT meeting) FROM decisions"
            " WHERE speaker_id IS NOT NULL AND outcome='accept' GROUP BY speaker_id"):
        heard[sid], seen[sid] = at, n

    rows = []
    for sid, name, created in conn.execute(
            "SELECT id, name, created_at FROM speakers ORDER BY name COLLATE NOCASE"):
        ps = protos.get(sid, [])
        vs = [p["vec"] for p in ps if p["vec"]]
        cent = _unit([sum(c) / len(vs) for c in zip(*vs)]) if vs else None
        meetings = {p["meeting"] for p in ps if p["meeting"]}
        for m, in conn.execute("SELECT DISTINCT meeting FROM decisions WHERE"
                               " speaker_id=? AND outcome='accept'", (sid,)):
            meetings.add(m)
        last = max([p["at"] for p in ps] + [heard.get(sid) or 0, created or 0])
        src = ps[-1] if ps else None
        rows.append({
            "id": sid, "name": name,
            "sessions": len(ps),
            "enrolled_s": round(sum(p["seconds"] for p in ps), 1),
            "meetings": len(meetings),
            "meeting_ids": sorted(m for m in meetings if m),
            "last_heard": last,
            "last_heard_str": fmt_date(last),
            "last_heard_short": fmt_date(last, short=True),
            "enrolled_from": (" · ".join(x for x in (src["meeting"], src["cluster"]) if x)
                              if src and src["meeting"] else None),
            "sources": [{"meeting": p["meeting"], "cluster": p["cluster"],
                         "seconds": p["seconds"], "at": p["at"],
                         "at_str": fmt_date(p["at"]), "embed_model": p["model"]}
                        for p in ps],
            "dim": ps[0]["dim"] if ps else None,
            "embed_model": model,
            "bars": bars(cent) if cent else [],
            "below_floor": sum(p["seconds"] for p in ps) < tun()["min_enroll_sec"],
            "voiceprint": cent is not None,
            "_c": cent,
        })

    # nearest other voice on file: the number that says how safe a match is
    for r in rows:
        best = None
        for o in rows:
            if o is r or not r["_c"] or not o["_c"]:
                continue
            sc = _dot(r["_c"], o["_c"])
            if best is None or sc > best["score"]:
                best = {"id": o["id"], "name": o["name"], "score": round(sc, 4)}
        r["nearest"] = best
    for r in rows:
        r.pop("_c", None)
    return rows


def voices_payload():
    conn = sdb()
    with conn:
        rows = voice_rows(conn)
    conn.close()
    reason = None
    if not rows:
        reason = "no-profiles"
    return {"voices": rows, "count": len(rows),
            "embed_model": STATE["facts"]["EMBED_MODEL"],
            "min_enroll_sec": tun()["min_enroll_sec"],
            "bars": 48,
            "bars_from": "RMS of the stored 256-d centroid in 48 contiguous bins, "
                         "scaled to the peak bin",
            "db": str(db_path()),
            "reason": reason}


def one_voice(conn, sid):
    return next((v for v in voice_rows(conn) if v["id"] == sid), None)


def _sid(body, key="id"):
    try:
        return int(body.get(key))
    except (TypeError, ValueError):
        return None


def voice_rename(body):
    sid, name = _sid(body), (body.get("name") or "").strip()
    if sid is None or not name:
        return {"error": "id and name required"}, 400
    conn = sdb()
    try:
        if not conn.execute("SELECT 1 FROM speakers WHERE id=?", (sid,)).fetchone():
            return {"error": "no such voice", "reason": "not-found"}, 404
        clash = conn.execute("SELECT id FROM speakers WHERE name=? AND id<>?",
                             (name, sid)).fetchone()
        if clash:
            # names are unique in the store: two profiles for one person is a
            # merge, not a rename, and the caller has to say which it wants.
            return {"error": f"{name} is already on file", "reason": "name-taken",
                    "existing_id": clash[0]}, 409
        conn.execute("UPDATE speakers SET name=? WHERE id=?", (name, sid))
        conn.commit()
        return {"ok": True, "voice": one_voice(conn, sid)}, 200
    finally:
        conn.close()


def voice_forget(body):
    sid = _sid(body)
    if sid is None:
        return {"error": "id required"}, 400
    conn = sdb()
    try:
        row = conn.execute("SELECT name FROM speakers WHERE id=?", (sid,)).fetchone()
        if not row:
            return {"error": "no such voice", "reason": "not-found"}, 404
        n = conn.execute("SELECT COUNT(*) FROM prototypes WHERE speaker_id=?",
                         (sid,)).fetchone()[0]
        conn.execute("DELETE FROM prototypes WHERE speaker_id=?", (sid,))
        conn.execute("DELETE FROM speakers WHERE id=?", (sid,))
        # the decision log is history and is left standing; the transcripts that
        # named this person stop showing the name because the join no longer hits.
        mentions = conn.execute("SELECT COUNT(*) FROM decisions WHERE speaker_id=?",
                                (sid,)).fetchone()[0]
        conn.commit()
        return {"ok": True, "forgotten": {"id": sid, "name": row[0]},
                "voiceprints_deleted": n, "decisions_kept": mentions,
                "remaining": conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]}, 200
    finally:
        conn.close()


def voice_merge(body):
    src, dst = _sid(body, "from"), _sid(body, "into")
    if src is None or dst is None:
        return {"error": "from and into required"}, 400
    if src == dst:
        return {"error": "a voice cannot be merged into itself"}, 400
    conn = sdb()
    try:
        a = conn.execute("SELECT name FROM speakers WHERE id=?", (src,)).fetchone()
        b = conn.execute("SELECT name FROM speakers WHERE id=?", (dst,)).fetchone()
        if not a or not b:
            return {"error": "no such voice", "reason": "not-found"}, 404
        n = conn.execute("SELECT COUNT(*) FROM prototypes WHERE speaker_id=?",
                         (src,)).fetchone()[0]
        # every voiceprint moves; the centroid is the mean of them all, so the
        # merged profile is what enrolling both sessions would have produced.
        conn.execute("UPDATE prototypes SET speaker_id=? WHERE speaker_id=?", (dst, src))
        conn.execute("UPDATE decisions SET speaker_id=? WHERE speaker_id=?", (dst, src))
        conn.execute("DELETE FROM speakers WHERE id=?", (src,))
        conn.commit()
        return {"ok": True, "merged": {"id": src, "name": a[0]},
                "into": {"id": dst, "name": b[0]}, "voiceprints_moved": n,
                "voice": one_voice(conn, dst)}, 200
    finally:
        conn.close()


# -------------------------------------------------------------------- review
def review_payload():
    t = tun()
    conn = sdb()
    try:
        G = gallery(conn)
        keys = meeting_keys()
        names = all_cluster_names()
        latest = {}
        for r in conn.execute(
                "SELECT d.meeting, d.cluster, d.speaker_id, s.name, d.score, d.second,"
                " d.outcome, d.created_at FROM decisions d"
                " LEFT JOIN speakers s ON s.id = d.speaker_id WHERE d.id IN"
                " (SELECT MAX(id) FROM decisions GROUP BY meeting, cluster)"):
            latest[(r[0], r[1])] = r
    finally:
        conn.close()

    pending, elsewhere, unidentified, seen_live = [], [], [], set()
    n_clusters = 0

    for js in library_json():
        stem = js.stem
        detail = cluster_detail(js)
        mkeys = [k for k, v in keys.items() if v == stem]
        cents, why = meeting_centroids(stem)
        if cents and G:
            # embeddings are here, so score this meeting now: identification is
            # cosine arithmetic over stored centroids, not GPU work.
            seen_live.add(stem)
            taken = {}
            for g in sorted(cents, key=lambda k: -cents[k]["seconds"]):
                n_clusters += 1
                cand = score_cluster(cents[g]["centroid"], G)
                best = cand[0] if cand else {"id": None, "name": None, "score": 0.0}
                second = cand[1]["score"] if len(cand) > 1 else 0.0
                b = band(best["score"], second, t)
                if b == "accept" and best["id"] in taken:
                    b = "margin"          # one person cannot be two clusters
                elif b == "accept":
                    taken[best["id"]] = g
                decided = any((k, g) in latest and latest[(k, g)][6] in HUMAN
                              for k in mkeys) or any(g in names.get(k, {}) for k in mkeys)
                if b == "accept" or decided:
                    continue
                pending.append(_pending(stem, stem, g, b, best, cand[1:3],
                                        cents[g]["seconds"], detail.get(g, {}),
                                        "live", True, None))
        elif cents and not G:
            # Embeddings are here and nobody is enrolled yet, so every cluster is
            # NAMEABLE: there is nothing to match against, but the voiceprint
            # needed to enrol one is sitting right there in the npz.
            #
            # This used to fall through to the inert list below, which said
            # "nobody is enrolled yet, so there is nothing to match against" and
            # offered no action at all -- while the Voices screen said a profile
            # appears "the first time a cluster is named on the Review screen".
            # The two screens pointed at each other and neither could enrol
            # anyone, so a fresh install could never get its first voice into the
            # store from the browser. Only ./speakers name could, which is not
            # something the UI ever mentions.
            #
            # Nothing is being matched here, so there is no band and no
            # candidate: just a cluster, its speech time and its samples,
            # waiting for a name. Everything after the first one goes through
            # the scoring path above.
            seen_live.add(stem)
            for g in sorted(cents, key=lambda k: -cents[k]["seconds"]):
                n_clusters += 1
                if any((k, g) in latest and latest[(k, g)][6] in HUMAN for k in mkeys) \
                        or any(g in names.get(k, {}) for k in mkeys):
                    continue
                pending.append(_pending(
                    stem, stem, g, "unknown",
                    {"id": None, "name": None, "score": 0.0}, [],
                    cents[g]["seconds"], detail.get(g, {}), "live", True, None))
        else:
            unidentified.append({
                "meeting": stem, "title": stem, "clusters": len(detail),
                "reason": why,
                "detail": {
                    "no-embeddings": f"{stem}.emb.npz is not beside this transcript, so "
                                     "its voices cannot be scored against the profile "
                                     "store on this machine",
                    "no-embedded-clusters": "the embeddings cover none of this "
                                            "meeting's clusters",
                }.get(why, why),
            })

    # anything the pipeline decided elsewhere and left open. Only the two scores
    # survive in the log, so the runner-up's identity is not always recoverable.
    for (mt, cl), r in sorted(latest.items()):
        stem = keys.get(mt)
        if stem in seen_live or r[6] not in ("review", "unknown"):
            continue
        n_clusters += 1
        if any(cl in names.get(k, {}) for k in ([mt] if stem is None else
                                                [k for k, v in keys.items() if v == stem])):
            continue
        best = {"id": r[2], "name": r[3], "score": round(r[4] or 0.0, 4)}
        second = {"id": None, "name": None, "score": round(r[5] or 0.0, 4)}
        detail = cluster_detail(STATE["library"] / f"{stem}.json") if stem else {}
        d = detail.get(cl, {})
        item = _pending(mt, stem, cl, band(best["score"], second["score"], t),
                        best, [second], d.get("seconds"), d, "log",
                        bool(stem) and (STATE["library"] / f"{stem}.emb.npz").exists(),
                        r[7])
        # a cluster whose meeting is not in this library cannot be played, read
        # or enrolled from here. It is real and still open, but it is not a
        # decision anyone can take on this screen, so it is kept apart.
        (pending if stem else elsewhere).append(item)
    for it in elsewhere:
        it["reason"] = "meeting-not-in-library"

    resolved = sum(1 for r in latest.values() if r[6] not in ("review", "unknown"))
    reason = None
    if not pending:
        if not library_json():
            reason = "empty-library"
        elif unidentified and not seen_live:
            # BEFORE no-profiles, deliberately. With no embeddings a cluster can
            # be neither matched nor ENROLLED, so reporting an empty profile
            # store sends you to fix the one thing that cannot be fixed until
            # this is: the library looks full, the message says nobody is
            # enrolled, and enrolling is exactly what is impossible.
            reason = "not-identified"
        elif not G:
            reason = "no-profiles"
        elif elsewhere:
            reason = "only-outside-library"
        else:
            reason = "all-resolved"
    return {
        "pending": pending,
        "elsewhere": elsewhere,
        "thresholds": t,
        "thresholds_source": STATE["facts"]["source"],
        "axis": {"min": 0.20, "max": 1.00},
        "counts": {"pending": len(pending), "elsewhere": len(elsewhere),
                   "clusters": n_clusters, "resolved": resolved,
                   "meetings": len(library_json()), "profiles": len(G)},
        "unidentified": unidentified,
        "reason": reason,
    }


def _pending(key, stem, cluster, b, best, others, secs, detail, source, can_enroll, at):
    cands = [best] + [c for c in others if c]
    named = all(c.get("name") for c in cands)
    return {
        "meeting": key,                     # the key the profile store uses
        "meeting_id": stem,                 # the library meeting, when it resolves
        "meeting_title": stem or key,
        "cluster": cluster,                 # G01 -- the id, for payloads and logs
        # ...and what a person calls it. The transcript has always shown
        # "Speaker 2"; Review showed "G01" for the same voice, so the two screens
        # named the same thing differently and neither said they were the same.
        "cluster_label": speaker_label(cluster),
        "band": b,                          # unknown | review | margin
        "seconds": secs,
        "seconds_reason": None if secs is not None else "transcript-not-in-library",
        "best": best,
        "second": cands[1] if len(cands) > 1 else None,
        "candidates": cands,
        "candidates_named": named,
        "candidates_reason": None if named else
            "the decision log stores both scores but only records who a match was "
            "when it was accepted, so the runner-up cannot be named from it",
        "samples": detail.get("samples", []),
        "samples_reason": None if detail.get("samples") else (
            "transcript-not-in-library" if not stem else "no-lines-long-enough"),
        "enough_to_enroll": (secs is not None and secs >= tun()["min_enroll_sec"]),
        "can_enroll": bool(can_enroll),
        "enroll_reason": None if can_enroll else "no-embeddings",
        "source": source,                   # live rescore, or the decision log
        "reason": None,
        "decided_at": at,
        "decided_at_str": fmt_date(at) if at else None,
    }


def reidentify(conn):
    """Re-score every meeting against the profile store and record what is
    certain. -> number of clusters newly named.

    Identification happens in two places and only one of them used to write
    anything down. identify.py runs ONCE, when a recording is processed, against
    whoever was enrolled at that moment -- so a voice enrolled later is never
    applied to the meetings it is already in. The Review screen scores those
    meetings live and gets the right answer, then drops any cluster it is
    confident about because there is nothing to ask about. Nothing persisted it.

    The result was a transcript reading "Speaker 3" for a voice the Review screen
    was, on the same data, naming correctly -- and no way to reconcile the two.

    Enrolling is the moment the gallery changes, so it is the moment to redo
    this. It is cosine arithmetic over stored centroids, not GPU work: a library
    of a few hundred meetings is milliseconds.
    """
    G = gallery(conn)
    if not G:
        return 0
    keys, names, t, now = meeting_keys(), all_cluster_names(), tun(), time.time()
    written = 0
    for js in library_json():
        stem = js.stem
        cents, _ = meeting_centroids(stem)
        if not cents:
            continue
        mkeys = [k for k, v in keys.items() if v == stem] or [stem]
        # One person cannot be two clusters in one meeting. Whoever is already
        # named here holds their slot, and the biggest cluster gets first claim
        # on the rest -- the same rule the live path applies, which the accept
        # path did not, and which is how one voice ended up on two clusters.
        taken = {info["id"]: cl for k in mkeys
                 for cl, info in names.get(k, {}).items() if info.get("id")}
        for g in sorted(cents, key=lambda k: -cents[k]["seconds"]):
            if any(g in names.get(k, {}) for k in mkeys):
                continue
            cand = score_cluster(cents[g]["centroid"], G)
            if not cand:
                continue
            best = cand[0]
            second = cand[1]["score"] if len(cand) > 1 else 0.0
            if band(best["score"], second, t) != "accept" or best["id"] in taken:
                continue
            taken[best["id"]] = g
            conn.execute(
                "INSERT INTO decisions(meeting, cluster, speaker_id, score, second,"
                " threshold, level, roster, outcome, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (stem, g, best["id"], best["score"], second, t["accept"],
                 "centroid", None, "accept", now))
            written += 1
    if written:
        conn.commit()
    return written


def resolve_cluster(body):
    meeting = str(body.get("meeting") or "")
    cluster = str(body.get("cluster") or "")
    action = str(body.get("action") or "")
    name = (body.get("name") or "").strip()
    if not meeting or not cluster or action not in ("accept", "name", "leave"):
        return {"error": "meeting, cluster and action=accept|name|leave required"}, 400
    if action == "name" and not name:
        return {"error": "a name is required"}, 400

    keys = meeting_keys()
    stem = keys.get(meeting, meeting if (STATE["library"] / f"{meeting}.json").exists()
                    else None)
    conn = sdb()
    try:
        G = gallery(conn)
        cents, why = meeting_centroids(stem) if stem else ({}, "transcript-not-in-library")
        vec = cents.get(cluster, {}).get("centroid")
        secs = cents.get(cluster, {}).get("seconds")
        t = tun()
        now = time.time()
        prev = conn.execute(
            "SELECT speaker_id, score, second FROM decisions WHERE meeting=? AND cluster=?"
            " ORDER BY id DESC LIMIT 1", (meeting, cluster)).fetchone()
        cand = score_cluster(vec, G) if (vec and G) else []
        best = cand[0] if cand else None
        second = cand[1]["score"] if len(cand) > 1 else 0.0

        if action == "leave":
            conn.execute(
                "INSERT INTO decisions(meeting, cluster, speaker_id, score, second,"
                " threshold, level, roster, outcome, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (meeting, cluster, None, best["score"] if best else (prev[1] if prev else None),
                 second if best else (prev[2] if prev else None), t["accept"], "centroid",
                 None, "left-unknown", now))
            conn.commit()
            return {"ok": True, "outcome": "left-unknown", "meeting": meeting,
                    "cluster": cluster, "speaker": None, "enrolled": False,
                    "enroll_reason": None}, 200

        if action == "accept":
            sid = best["id"] if best else (prev[0] if prev else None)
            if sid is None:
                return {"error": "no candidate to accept", "reason": "candidate-unknown",
                        "detail": "the decision log did not record who the best match "
                                  "was, and this meeting has no embeddings here to "
                                  "score it again"}, 409
            row = conn.execute("SELECT name FROM speakers WHERE id=?", (sid,)).fetchone()
            if not row:
                return {"error": "that voice has been forgotten",
                        "reason": "speaker-gone"}, 409
            conn.execute(
                "INSERT INTO decisions(meeting, cluster, speaker_id, score, second,"
                " threshold, level, roster, outcome, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (meeting, cluster, sid, best["score"] if best else (prev[1] if prev else None),
                 second if best else (prev[2] if prev else None), t["accept"], "centroid",
                 None, "accept", now))
            conn.commit()
            return {"ok": True, "outcome": "accept", "meeting": meeting,
                    "cluster": cluster, "speaker": {"id": sid, "name": row[0]},
                    "score": best["score"] if best else (prev[1] if prev else None),
                    "band": band(best["score"], second, t) if best else None,
                    "enrolled": False, "enroll_reason": None}, 200

        # action == "name": remember this voice, if there is a voiceprint to keep
        conn.execute("INSERT OR IGNORE INTO speakers(name, created_at) VALUES(?,?)",
                     (name, now))
        sid = conn.execute("SELECT id FROM speakers WHERE name=?", (name,)).fetchone()[0]
        enrolled, reason = False, None
        if vec:
            if secs is not None and secs < t["min_enroll_sec"] and not body.get("force"):
                conn.rollback()
                return {"error": "not enough speech to enrol",
                        "reason": "below-enrolment-floor",
                        "seconds": secs, "min_enroll_sec": t["min_enroll_sec"],
                        "detail": "10 s of clean speech is the measured knee; pass "
                                  "force to store it anyway"}, 409
            conn.execute(
                "INSERT INTO prototypes(speaker_id, emb, dim, embed_model, level,"
                " meeting, seconds, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (sid, struct.pack("<%df" % len(vec), *vec), len(vec),
                 STATE["facts"]["EMBED_MODEL"], "centroid",
                 f"{meeting}:{cluster}", secs, now))
            enrolled = True
        else:
            # the name is recorded for this meeting, but nothing is stored that
            # could recognise this person again. Say so; do not fake a voiceprint.
            reason = why or "no-embeddings"
        conn.execute(
            "INSERT INTO decisions(meeting, cluster, speaker_id, score, second,"
            " threshold, level, roster, outcome, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (meeting, cluster, sid, best["score"] if best else (prev[1] if prev else None),
             second if best else (prev[2] if prev else None), t["accept"], "centroid",
             None, "accept", now))
        conn.commit()
        # The gallery just changed, so every meeting already in the library can
        # now be scored against a voice that did not exist when it was processed.
        also = reidentify(conn) if enrolled else 0
        return {"ok": True, "outcome": "accept", "meeting": meeting, "cluster": cluster,
                "also_named": also,
                "speaker": {"id": sid, "name": name},
                "score": best["score"] if best else None,
                "seconds": secs, "enrolled": enrolled,
                "enroll_reason": reason,
                "detail": None if enrolled else
                "named in this transcript only: without the meeting's embeddings there "
                "is no voiceprint to store, so this voice will not be recognised in a "
                "later meeting"}, 200
    finally:
        conn.close()


# -------------------------------------------------------------------- search
def search(q, want_speakers, limit=80):
    q = (q or "").strip()
    if len(q) < 2:
        return {"q": q, "hits": [], "n_hits": 0, "meetings": 0,
                "speakers": search_speakers(), "truncated": False,
                "reason": "query-too-short"}
    needle = q.lower()
    names = all_cluster_names()
    keys = meeting_keys()
    hits, matched, truncated = [], set(), False
    for js in library_json():
        who = names_for(js, names, keys)
        try:
            segs = json.loads(js.read_text()).get("segments", [])
        except Exception:
            continue
        for s in segs:
            text = (s.get("text") or "").strip()
            i = text.lower().find(needle)
            if i < 0:
                continue
            g = s.get("global") or s.get("speaker") or "G-1"
            label = who[g]["name"] if g in who else speaker_label(g)
            if want_speakers and label not in want_speakers and g not in want_speakers:
                continue
            if len(hits) >= limit:
                truncated = True
                break
            pre, hit, post = text[:i], text[i:i + len(q)], text[i + len(q):]
            hits.append({
                "meeting": js.stem, "title": js.stem,
                "t": round(float(s["start"]), 2),
                "at": hms(float(s["start"])),
                "speaker": g, "who": label, "named": g in who,
                "speaker_id": who.get(g, {}).get("id"),
                "pre": ("…" + pre[-70:]) if len(pre) > 70 else pre,
                "hit": hit,
                "post": (post[:110] + "…") if len(post) > 110 else post,
            })
            matched.add(js.stem)
        if truncated:
            break
    return {"q": q, "hits": hits, "n_hits": len(hits), "meetings": len(matched),
            "searched": len(library_json()), "speakers": search_speakers(),
            "truncated": truncated,
            "reason": None if hits else ("empty-library" if not library_json() else "no-match")}


def search_speakers():
    """The filter chips: every voice that actually appears in the library."""
    names, keys = all_cluster_names(), meeting_keys()
    seen = {}
    for js in library_json():
        who = names_for(js, names, keys)
        for g, v in who.items():
            e = seen.setdefault(v["name"], {"name": v["name"], "id": v["id"],
                                            "meetings": 0, "named": True})
            e["meetings"] += 1
    return sorted(seen.values(), key=lambda s: -s["meetings"])


def hms(s):
    h, m, x = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return (f"{h}:{m:02d}:{x:02d}" if h else f"{m}:{x:02d}")


# ------------------------------------------------------------------ settings
RANGES = {"accept": (0.10, 0.99), "review": (0.05, 0.95),
          "margin": (0.0, 0.50), "min_enroll_sec": (1.0, 600.0)}


def settings_payload():
    f, t = STATE["facts"], tun()
    lib = STATE["library"]
    try:
        conn = sdb()
        with conn:
            store = {
                "voices": conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0],
                "voiceprints": conn.execute("SELECT COUNT(*) FROM prototypes").fetchone()[0],
                "decisions": conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
            }
        conn.close()
    except Exception as e:
        store = {"error": str(e)}
    meetings = library_json()
    windows = []
    for js in meetings[:20]:
        try:
            w = json.loads(js.read_text()).get("window_s")
        except Exception:
            w = None
        if w:
            windows.append(float(w))
    return {
        "recognition": {
            "accept": t["accept"], "review": t["review"], "margin": t["margin"],
            "min_enroll_sec": t["min_enroll_sec"],
            "defaults": {"accept": f["ACCEPT"], "review": f["REVIEW"],
                         "margin": f["MARGIN"], "min_enroll_sec": f["MIN_ENROLL_SEC"]},
            "overridden": sorted(k for k in RANGES if k in STATE["settings"]),
            "source": f["source"],
            "applies_to": "identification and review banding done by this server. "
                          "bench/speakers.py keeps its own compiled-in values when it "
                          "is run from the command line.",
        },
        "models": {
            "transcribe": f["MODEL"],
            "voiceprints": f["EMBED_MODEL"],
            "window_s": (windows[0] if windows else None),
            "window_s_observed": sorted(set(windows)),
            "silence_gate_db": f["SILENCE_GATE_DB"],
            "min_enroll_sec": t["min_enroll_sec"],
        },
        "glossary": {"path": str(glossary_file()), "terms": glossary_terms()},
        "backend": {
            "host": STATE["cfg"].get("host"),
            "work": STATE["cfg"].get("work"), "name": STATE["backend"].name,
            "reachable": STATE["cfg"].get("backend_up"),
            "checked_at": STATE["cfg"].get("backend_checked"),
            "note": "only transcribe and embed run there. Identification, renaming, "
                    "merging and forgetting are cosine arithmetic over stored "
                    "centroids and always run here.",
        },
        "paths": {
            "library": str(lib), "speaker_db": str(db_path()),
            "speaker_db_exists": db_path().exists(),
            "settings": str(settings_file()), "glossary": str(glossary_file()),
            "library_persisted": False,
            "note": "the library path applies immediately but is not remembered across "
                    "a restart; pass --library (or MS_LIBRARY) to make it the default.",
        },
        "store": dict(store, meetings=len(meetings)),
    }


def settings_write(body):
    changed = []
    draft = dict(STATE["settings"])
    for k, (lo, hi) in RANGES.items():
        if body.get(k) is None:
            continue
        try:
            v = float(body[k])
        except (TypeError, ValueError):
            return {"error": f"{k} must be a number"}, 400
        if not lo <= v <= hi:
            return {"error": f"{k} must be between {lo} and {hi}"}, 400
        draft[k] = round(v, 4)
        changed.append(k)
    f = STATE["facts"]
    if draft.get("review", f["REVIEW"]) > draft.get("accept", f["ACCEPT"]):
        return {"error": "the review floor cannot sit above the accept line"}, 400
    STATE["settings"] = draft

    # the library moves first: the glossary and the preferences live inside it
    if body.get("library"):
        p = Path(str(body["library"])).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"error": f"cannot use that library: {e}"}, 400
        STATE["library"] = p.resolve()
        ensure_db()
        changed.append("library")

    if "glossary" in body:
        g = body["glossary"]
        terms = ([str(x).strip() for x in g] if isinstance(g, list)
                 else [x.strip() for x in re.split(r"[\n,]", str(g))])
        save_glossary([x for x in terms if x])
        changed.append("glossary")

    rebuilt = False
    # host may legitimately be set to "" -- that is how you go back to running
    # the pipeline locally -- so this cannot test truthiness the way the others do
    for k in ("host", "work"):
        if k in body and body[k] != STATE["cfg"].get(k):
            if k == "work" and not body[k]:
                continue
            STATE["cfg"][k] = str(body[k])
            rebuilt = True
            changed.append(k)
    if rebuilt:
        STATE["backend"] = backend_mod.from_config(STATE["cfg"])
        STATE["cfg"]["backend_up"] = STATE["backend"].available()
        STATE["cfg"]["backend_checked"] = time.time()

    save_settings()
    out = settings_payload()
    out["changed"] = changed
    return out, 200


# -------------------------------------------------------------------- queue
def enqueue(paths, glossary=""):
    items = []
    with STATE["lock"]:
        for p in paths:
            it = {"id": uuid.uuid4().hex[:8], "file": Path(p).name, "path": str(p),
                  "stage": "queued", "pct": 0, "started": None, "elapsed": 0,
                  "error": None, "glossary": glossary}
            STATE["queue"].append(it)
            items.append(it)
    # Adding a file does NOT start it. A batch runs when you press Process
    # Queue and not before -- dropping forty recordings on this screen should
    # not commit the GPU the moment the first one lands, and once a batch is in
    # flight it is one process, so the decision to start is the last cheap one.
    return items


_draining = threading.Lock()


def _drain():
    """Run everything queued as ONE batch.

    The engine costs ~66s to load. Draining one file at a time paid that per
    file and never overlapped embedding with the next transcription -- six files
    took 13 minutes that way against about 3 batched.
    """
    if not _draining.acquire(blocking=False):
        return
    try:
        while True:
            with STATE["lock"]:
                batch = [i for i in STATE["queue"] if i["stage"] == "queued"]
                for i in batch:
                    i["stage"] = "starting"
                    i["started"] = time.time()
            if not batch:
                return

            def on_stage(name, pct):
                for it in batch:
                    it["stage"] = name
                    it["pct"] = pct
                    it["elapsed"] = round(time.time() - it["started"], 1)

            gloss = next((i["glossary"] for i in batch if i["glossary"]), "")
            try:
                done = STATE["backend"].run_batch(
                    [Path(i["path"]) for i in batch], STATE["library"], gloss,
                    on_stage=on_stage)
                produced = {str(src): linked for src, linked in done}
                for it in batch:
                    linked = produced.get(it["path"])
                    if not linked or not Path(linked).exists():
                        it["stage"] = "failed"
                        it["error"] = "no transcript produced"
                        continue
                    src = Path(it["path"])
                    dest = Path(linked).with_suffix(src.suffix)
                    if not dest.exists():
                        try:
                            os.link(src, dest)
                        except Exception:
                            shutil.copy2(src, dest)
                    it["stage"] = "done"
                    it["pct"] = 100
                    it["meeting"] = Path(linked).stem
            except Exception as e:
                traceback.print_exc()
                for it in batch:
                    it["stage"] = "failed"
                    it["error"] = str(e)
            for it in batch:
                it["elapsed"] = round(time.time() - it["started"], 1)
    finally:
        _draining.release()


# ------------------------------------------------------------------ serving
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _index(self):
        """index.html with a version stamped onto every local asset URL.

        Cache-Control: no-cache asks a browser to revalidate, but with no
        validator to revalidate against, browsers hold onto app.js and the screen
        scripts anyway -- edit a screen, reload, see the old one, and conclude the
        change did not work. Stamping the file's mtime into the URL makes a
        changed file a different URL, which no cache can get wrong.
        """
        html = (STATIC / "index.html").read_text()

        def stamp(m):
            url = m.group(2)
            rel = url.split("?")[0].lstrip("/")
            if rel.startswith("static/"):
                rel = rel[len("static/"):]
            f = STATIC / rel
            try:
                v = int(f.stat().st_mtime)
            except OSError:
                return m.group(0)
            return f'{m.group(1)}"{url}?v={v}"'

        html = re.sub(r'(href=|src=)"(/static/[^"?]+)"', stamp, html)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype=None):
        if not path.exists():
            return self._json({"error": "not found"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(str(path))[0]
                         or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _audio(self, path: Path):
        """Byte-range aware, because seeking a 74-minute recording otherwise
        re-downloads the whole file on every scrub."""
        if not path or not path.exists():
            return self._json({"error": "no audio"}, 404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "audio/mpeg"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                start = min(start, end)
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(262144, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # -- routes
    def do_GET(self):
        u = urlparse(self.path)
        p, q = u.path, parse_qs(u.query)
        try:
            if p == "/" or p == "/index.html":
                return self._index()
            if p.startswith("/static/"):
                f = (STATIC / p[len("/static/"):]).resolve()
                if STATIC.resolve() in f.parents:
                    return self._file(f)
                return self._json({"error": "denied"}, 403)

            if p == "/api/library":
                ms = library_meetings()
                return self._json({"meetings": ms, "stats": library_stats(ms)})
            if p == "/api/meeting":
                js = STATE["library"] / f"{q.get('id', [''])[0]}.json"
                if not js.exists():
                    return self._json({"error": "not found"}, 404)
                try:
                    return self._json(read_meeting(js, full=True))
                except ValueError:
                    return self._json({"error": "not a transcript"}, 404)
            if p == "/api/audio":
                js = STATE["library"] / f"{q.get('id', [''])[0]}.json"
                return self._audio(audio_for(js) if js.exists() else None)
            if p == "/api/queue":
                # the whole queue, not the last 40: the screen caps what it
                # renders and offers filters, and a truncated list made "3
                # waiting" disagree with the rows you could actually see
                running = any(i["stage"] not in ("queued", "held", "done", "failed")
                              for i in STATE["queue"])
                waiting = sum(1 for i in STATE["queue"] if i["stage"] == "queued")
                return self._json({"queue": STATE["queue"],
                                   "waiting": waiting, "running": running})
            if p == "/api/voices":
                return self._json(voices_payload())
            if p == "/api/review":
                return self._json(review_payload())
            if p == "/api/search":
                try:
                    lim = max(1, min(500, int(q.get("limit", ["80"])[0])))
                except ValueError:
                    lim = 80
                return self._json(search(q.get("q", [""])[0],
                                         [s for s in q.get("speaker", []) if s], lim))
            if p == "/api/settings":
                if q.get("check"):
                    STATE["cfg"]["backend_up"] = STATE["backend"].available()
                    STATE["cfg"]["backend_checked"] = time.time()
                return self._json(settings_payload())
            if p == "/api/browse":
                # for the ingest picker: list audio in a directory on this machine
                d = Path(unquote(q.get("dir", [str(Path.home())])[0])).expanduser()
                if not d.is_dir():
                    return self._json({"error": "no such directory", "dir": str(d)}, 404)
                files = sorted([f.name for f in d.iterdir()
                                if f.is_file() and f.suffix.lower() in AUDIO_EXT])
                dirs = sorted([f.name for f in d.iterdir()
                               if f.is_dir() and not f.name.startswith(".")])[:200]
                return self._json({"dir": str(d), "parent": str(d.parent),
                                   "files": files, "dirs": dirs})
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": str(e)}, 500)

    def _upload(self, u):
        """Take one file's bytes and put them where ingest can find them.

        The browser will not tell a page where a chosen file lives -- it hands
        over the contents and a bare name, never a path -- so a system file
        picker can only work if the bytes come with it. That is the whole reason
        the old custom directory browser existed, and uploading is what lets it
        be deleted.

        Raw body with the name in a header rather than multipart/form-data: the
        stdlib parser for multipart is deprecated and gone in 3.13, and hand
        rolling one to move bytes we already have in the body is work with a
        failure mode and no benefit. One request per file, streamed to disk so a
        two-hour recording is not read into memory first.
        """
        name = unquote(self.headers.get("X-Filename") or "").strip()
        # a name is all the browser gives, and it is attacker-controlled: keep
        # the basename only, so nothing can be written outside the inbox
        name = os.path.basename(name.replace("\\", "/"))
        if not name or name in (".", ".."):
            return self._json({"error": "X-Filename missing or unusable"}, 400)
        if Path(name).suffix.lower() not in AUDIO_EXT:
            return self._json({"error": f"{name}: not an audio file this can read"}, 400)

        inbox = STATE["library"] / "_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        dst = inbox / name
        stem, ext, i = dst.stem, dst.suffix, 1
        while dst.exists():                      # never clobber a queued file
            dst = inbox / f"{stem}-{i}{ext}"
            i += 1

        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._json({"error": "empty upload"}, 400)
        left = n
        try:
            with open(dst, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
        except OSError as e:
            return self._json({"error": f"could not write {dst.name}: {e}"}, 500)
        if left:
            dst.unlink(missing_ok=True)
            return self._json({"error": "upload ended early"}, 400)
        return self._json({"path": str(dst), "name": dst.name, "bytes": n})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            # uploads carry bytes, not JSON, so they are handled before the
            # body is parsed
            if u.path == "/api/upload":
                return self._upload(u)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "body must be JSON"}, 400)
            if not isinstance(body, dict):
                return self._json({"error": "body must be a JSON object"}, 400)

            if u.path == "/api/ingest":
                # Paths only, and they come from /api/upload -- the browser
                # cannot name a file on disk, so everything reaching here was
                # uploaded a moment ago and lives in the library's _inbox.
                paths = []
                for f in body.get("files", []):
                    fp = Path(f).expanduser()
                    if fp.is_file():
                        paths.append(fp)
                if not paths:
                    return self._json({"error": "no readable files"}, 400)
                # the stored glossary is the default, so ingest and settings agree
                gl = body.get("glossary")
                if gl is None:
                    gl = ", ".join(glossary_terms())
                return self._json({"queued": enqueue(paths, gl)})

            if u.path == "/api/queue/act":
                """hold | release | remove | retry per file; run | cancel for
                the queue as a whole.

                An individual file cannot be stopped once its batch is running:
                _drain hands every queued file to ONE ./transcribe so the engine
                loads once, which is the difference between ~70s per file and
                ~70s per batch. Holding a file before it starts, and cancelling
                the batch that is running, together give the same control without
                giving that up.
                """
                act = str(body.get("action") or "")
                ids = body.get("ids") or ([body["id"]] if body.get("id") else [])
                ids = set(str(x) for x in ids)

                if act == "cancel":
                    stopped = STATE["backend"].cancel()
                    with STATE["lock"]:
                        for it in STATE["queue"]:
                            if it["stage"] not in ("done", "failed", "queued", "held"):
                                it["stage"] = "failed"
                                it["error"] = "cancelled"
                    return self._json({"cancelled": bool(stopped)})

                if act == "run":
                    with STATE["lock"]:
                        n = sum(1 for it in STATE["queue"] if it["stage"] == "queued")
                    if not n:
                        return self._json({"error": "nothing waiting to run"}, 400)
                    threading.Thread(target=_drain, daemon=True).start()
                    return self._json({"started": n})

                if act not in ("hold", "release", "remove", "retry"):
                    return self._json({"error": f"unknown action {act!r}"}, 400)
                if not ids:
                    return self._json({"error": "no id given"}, 400)

                changed = 0
                with STATE["lock"]:
                    keep = []
                    for it in STATE["queue"]:
                        if it["id"] not in ids:
                            keep.append(it)
                            continue
                        if act == "remove":
                            # only something not in flight: a running file has a
                            # process behind it and vanishing its row would lie
                            if it["stage"] in ("queued", "held", "done", "failed"):
                                changed += 1
                                continue
                            keep.append(it)
                        elif act == "hold" and it["stage"] == "queued":
                            it["stage"] = "held"; changed += 1; keep.append(it)
                        elif act == "release" and it["stage"] == "held":
                            it["stage"] = "queued"; changed += 1; keep.append(it)
                        elif act == "retry" and it["stage"] == "failed":
                            it.update(stage="queued", pct=0, error=None,
                                      started=None, elapsed=0)
                            changed += 1; keep.append(it)
                        else:
                            keep.append(it)
                    STATE["queue"][:] = keep
                return self._json({"changed": changed})

            if u.path == "/api/voices/rename":
                return self._json(*voice_rename(body))
            if u.path == "/api/voices/forget":
                return self._json(*voice_forget(body))
            if u.path == "/api/voices/merge":
                return self._json(*voice_merge(body))
            if u.path == "/api/review/resolve":
                return self._json(*resolve_cluster(body))
            if u.path == "/api/settings":
                return self._json(*settings_write(body))
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    # No --mode any more. There used to be two backends, "split" and
    # "all-in-one", each with its own copy of upload -> run -> retrieve; both are
    # now ./transcribe, which decides for itself whether there is anything to ssh
    # to. An empty --host means the pipeline runs here.
    ap.add_argument("--host", default=os.environ.get("MS_HOST", ""),
                    help="ssh alias of a GPU box. Omit to run the pipeline here.")
    ap.add_argument("--work", default=os.environ.get("MS_WORK", str(HOME)),
                    help="pipeline install directory, local or remote")
    ap.add_argument("--library", default=os.environ.get("MS_LIBRARY", str(HERE.parent / "library")))
    ap.add_argument("--speaker-db", default=os.environ.get("MS_SPEAKER_DB"),
                    help="profile store; default <library>/speakers.db")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MS_PORT", "8765")))
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()

    lib = Path(a.library).expanduser().resolve()
    lib.mkdir(parents=True, exist_ok=True)
    STATE["library"] = lib
    STATE["facts"] = pipeline_facts()
    STATE["settings"] = load_settings()
    STATE["cfg"] = {"host": a.host, "work": a.work, "speaker_db": a.speaker_db}
    STATE["backend"] = backend_mod.from_config(STATE["cfg"])
    up = STATE["backend"].available()
    STATE["cfg"]["backend_up"] = up
    STATE["cfg"]["backend_checked"] = time.time()
    db = ensure_db()
    t = tun()

    print(f"  library   {lib}")
    print(f"  profiles  {db}  ({profile_count()} enrolled)")
    print(f"  matching  accept {t['accept']:.2f}, margin {t['margin']:.2f}, "
          f"review {t['review']:.2f}  (always runs here, never on the box)")
    print(f"  pipeline  {STATE['backend'].name} — "
          + ("reachable" if up
             else "NOT reachable; browsing and search work, ingest will fail"))
    print(f"\n  http://{a.bind}:{a.port}\n")
    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
