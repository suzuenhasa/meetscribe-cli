import json, sys
from collections import defaultdict

linked, raw, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(linked))
segs = d["segments"] if isinstance(d, dict) else d
dur = json.load(open(raw))["duration_s"]

secs = defaultdict(float)
for s in segs:
    secs[s.get("global") or "UNASSIGNED"] += s["end"] - s["start"]
real = [g for g in secs if g and not str(g).startswith("G-") and g != "UNASSIGNED"]
order = {g: i for i, g in enumerate(sorted(real, key=lambda g: -secs[g]))}


def who(g):
    return f"Speaker {order[g] + 1}" if g in order else "UNKNOWN"


def hms(t):
    return f"{int(t//3600)}:{int(t%3600//60):02d}:{int(t%60):02d}"


lines = [title,
         f"{dur/60:.1f} min · {len(segs)} turns · {len(real)} speakers",
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
    print(f"  {who(g):12} {secs[g]/60:5.1f} min")
print(f"\nwrote {out_path}")
