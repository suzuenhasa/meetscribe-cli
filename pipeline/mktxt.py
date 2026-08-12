"""Turn a linked run into the transcript a person reads.

  mktxt.py <linked.json> <raw.json> <out.txt> "<title>" [names.json]

Speakers are numbered by how much they spoke, so Speaker 1 is the person who
talked most -- the cluster IDs (G00, G07) are arbitrary and mean nothing to a
reader. If names.json is given, anyone identify.py recognised gets their real
name and the rest keep their number.

Consecutive segments from one speaker are printed under a single heading; a new
heading only appears when the speaker changes.
"""
import json, sys
from collections import defaultdict

linked, raw, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# optional 5th arg: {cluster: "Real Name"} from identify.py. Clusters it does not
# name keep their positional label, so no store means the old behaviour exactly.
named = {}
if len(sys.argv) > 5 and sys.argv[5]:
    try:
        named = json.load(open(sys.argv[5]))
    except Exception:
        named = {}
d = json.load(open(linked))
segs = d["segments"] if isinstance(d, dict) else d
dur = json.load(open(raw))["duration_s"]

secs = defaultdict(float)
for s in segs:
    secs[s.get("global") or "UNASSIGNED"] += s["end"] - s["start"]
real = [g for g in secs if g and not str(g).startswith("G-") and g != "UNASSIGNED"]
order = {g: i for i, g in enumerate(sorted(real, key=lambda g: -secs[g]))}


def who(g):
    if g in named:
        return named[g]
    return f"Speaker {order[g] + 1}" if g in order else "UNKNOWN"


def hms(t):
    return f"{int(t//3600)}:{int(t%3600//60):02d}:{int(t%60):02d}"


n_named = sum(1 for g in real if g in named)
tally = (f"{len(real)} speakers" if not n_named
         else f"{len(real)} speakers, {n_named} named")
lines = [title,
         f"{dur/60:.1f} min · {len(segs)} turns · {tally}",
         "=" * 66]
last = None
for s in segs:
    n = who(s.get("global"))
    if n != last:
        lines += ["", f"[{hms(s['start'])}] {n}"]
        last = n
    lines.append("  " + s["text"])
open(out_path, "w").write("\n".join(lines) + "\n")

print("speech time by speaker:")
for g in sorted(secs, key=lambda g: -secs[g]):
    print(f"  {who(g):22} {secs[g]/60:5.1f} min")
print(f"\nwrote {out_path}")
