#!/usr/bin/env python3
"""Match a meeting's speaker clusters against the profile store.

  identify.py --clusters out/x_linked_clusters.npz --meeting x [--roster "Bob,Jane"]
              [--names names.json] [--enroll G01="Bob Smith" ...]

Writes a {cluster: name} map that mktxt.py turns into real names in the
transcript. Clusters that match nobody keep their positional label, so an empty
profile store degrades to exactly the old behaviour rather than failing.

Enrolling is explicit and separate: a voice is only remembered once you name it.
Auto-enrolling every unnamed cluster would fill the store with "Speaker 3"s that
can never be matched to a person.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speakers as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="the _clusters.npz link.py wrote")
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--roster", default="",
                    help="restrict candidates to these names. Acceptance is a max over "
                         "the gallery, so false accepts grow with its size -- scoring 3 "
                         "known attendees is far safer than 500.")
    ap.add_argument("--names", default="", help="write the {cluster: name} map here")
    ap.add_argument("--enroll", nargs="*", default=[],
                    help='CLUSTER="Full Name", e.g. G01="Bob Smith"')
    a = ap.parse_args()

    conn = S.db()
    cents = S.centroids_from_npz(a.clusters)

    # -- explicit enrolments first, so they are matchable in this same run
    for spec in a.enroll:
        if "=" not in spec:
            raise SystemExit(f"--enroll wants CLUSTER=\"Name\", got {spec!r}")
        cid, name = spec.split("=", 1)
        cid, name = cid.strip(), name.strip().strip('"').strip("'")
        if cid not in cents:
            raise SystemExit(f"no cluster {cid} in this meeting "
                             f"(have: {', '.join(sorted(cents))})")
        v, secs = cents[cid]
        if secs < S.MIN_ENROLL_SEC:
            print(f"!! {cid} has only {secs:.0f}s of speech; "
                  f"{S.MIN_ENROLL_SEC:.0f}s is the measured enrolment knee. Enrolling anyway.")
        S.enroll_centroid(conn, name, v, secs, a.meeting, cid)
        conn.commit()
        print(f"enrolled {cid} as {name} ({secs:.0f}s)")

    roster = [x.strip() for x in a.roster.split(",") if x.strip()] or None
    G = S.gallery(conn, roster)
    names, report = {}, []

    if not G:
        print("no voices enrolled yet — everyone stays numbered.")
        print("name someone with:  ./speakers name <meeting> <cluster> \"Their Name\"")
    else:
        taken = {}
        for cid in sorted(cents, key=lambda k: -cents[k][1]):
            v, secs = cents[cid]
            scored = sorted(((float(v @ c), sid, nm) for sid, nm, c in G), reverse=True)
            best = scored[0]
            second = scored[1] if len(scored) > 1 else (0.0, None, None)
            score, sid, nm = best
            if score < S.REVIEW:
                outcome, label = "unknown", None
            elif score < S.ACCEPT or (score - second[0]) < S.MARGIN:
                outcome, label = "review", None
            elif sid in taken:
                # one person cannot be two clusters in the same meeting
                outcome, label = "review", None
            else:
                outcome, label = "accept", nm
                taken[sid] = cid
            if label:
                names[cid] = label
            report.append((cid, secs, nm, score, second[0], outcome))
            conn.execute(
                "INSERT INTO decisions(meeting, cluster, speaker_id, score, second,"
                " threshold, level, roster, outcome, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (a.meeting, cid, sid if outcome == "accept" else None, score,
                 second[0], S.ACCEPT, "centroid",
                 ",".join(roster) if roster else None, outcome, time.time()))
        conn.commit()

        print(f"identify: {len(cents)} voices in this meeting, "
              f"{len(G)} enrolled candidate{'s' if len(G) != 1 else ''}"
              + (f" (roster: {', '.join(roster)})" if roster else ""))
        for cid, secs, nm, score, sec, outcome in report:
            mark = {"accept": "=", "review": "?", "unknown": " "}[outcome]
            shown = nm if outcome != "unknown" else "-"
            print(f"  {mark} {cid:5} {secs:6.0f}s  {shown:22} {score:5.3f}"
                  + (f"  (2nd {sec:.3f})" if outcome == "review" else ""))
        if any(r[5] != "accept" for r in report):
            print(f"  accept >= {S.ACCEPT} with a margin of {S.MARGIN} over the runner-up; "
                  f"{S.REVIEW}-{S.ACCEPT} needs a person to decide")

    if a.names:
        with open(a.names, "w") as fh:
            json.dump(names, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
