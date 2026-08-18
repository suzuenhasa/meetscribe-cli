#!/usr/bin/env python3
"""Cross-window speaker linking for a MOSS windowed run.

Pipeline
  1. aggregate the per-segment WeSpeaker embeddings into one vector per
     (window, local_speaker)  -- plain centroid of L2-normalised vectors,
     re-normalised: the strategy that won the enrollment sweep.
  2. cluster the CORE aggregates (>= --min-core seconds of embedded speech).
     cluster_speakers.py owns this: constrained agglomerative linkage honouring
     MOSS's within-window "these are different people" claims, cut where the
     merge heights gap the most. The cut is DERIVED PER MEETING -- --thr auto is
     the default and a fixed value is the failure mode it exists to prevent. k is
     whatever falls out; MOSS's count and the true count are never used.
  3. refine leave-one-out: reassign each aggregate to its nearest centroid with
     ITSELF excluded from that centroid. Including it means a turn in a small
     cluster is compared against a centroid that is mostly itself, so it can
     never move however well it matches its real speaker.
  4. weak aggregates (below --min-core, or with no embeddable segment at all)
     are attached to the nearest global centroid.
  5. write the run back out with a `global` field per segment, plus a
     <out>_clusters.npz holding one centroid per speaker -- that file is what
     identify.py matches against the profile store, and it is 17 KB, so it
     travels back from a GPU box while the audio does not.

Prints FLOOR-VIOLATION when the speaker count comes out below the number MOSS
heard talking inside a single window. That contradiction sat in our own data
while the clustering returned 1, and nothing checked it.

k is also estimated independently by eigengap and by silhouette, purely as a
cross-check on the threshold-based k.
"""
import argparse, json, os, sys
# scipy is imported inside silhouette_k() and cluster() rather than here. Both are
# reachable only via --sweep, a diagnostic nothing in the pipeline passes, and the
# real path is cluster_speakers.cluster(), which is numpy only. Importing
# scipy.cluster.hierarchy at module scope cost ~0.3s of the 0.37s this script took
# to run -- per recording, so ~35s across a queue of 120 short ones, to load code
# that never executed.
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    sys.path.insert(0, _p)
import os
import cluster_speakers
import match_speakers


def load(run_path, npz_path):
    R = json.load(open(run_path))
    z = np.load(npz_path, allow_pickle=True)
    E = z["emb"]
    seg_idx = z["seg_idx"]
    meta = json.loads(str(z["meta"]))
    return R, E, seg_idx, meta


def aggregate(R, E, seg_idx, meta):
    """-> keys, A (n_keys,D), secs, nsegs, key_of_segment(list per segment)."""
    pos = {int(s): i for i, s in enumerate(seg_idx)}
    buckets, secs, nseg = {}, {}, {}
    key_of = []
    for m in meta:
        k = (m["window"], m["local"])
        key_of.append(k)
        secs.setdefault(k, 0.0)
        nseg.setdefault(k, 0)
        if m["idx"] in pos:
            buckets.setdefault(k, []).append(pos[m["idx"]])
            secs[k] += m["dur_used"]
            nseg[k] += 1
    keys = sorted(secs)
    A = np.zeros((len(keys), E.shape[1]), dtype=np.float64)
    for i, k in enumerate(keys):
        if k in buckets:
            v = E[buckets[k]].astype(np.float64).mean(0)
            nv = np.linalg.norm(v)
            A[i] = v / nv if nv > 0 else 0.0
    return keys, A, np.array([secs[k] for k in keys]), np.array([nseg[k] for k in keys]), key_of


def eigengap_k(S, kmax=15):
    """Spectral eigengap on the row-normalised affinity."""
    Aff = np.clip(S, 0, None)
    np.fill_diagonal(Aff, 0.0)
    d = Aff.sum(1)
    d[d <= 0] = 1e-9
    L = np.eye(len(Aff)) - (Aff / np.sqrt(np.outer(d, d)))
    w = np.linalg.eigvalsh(L)[: kmax + 1]
    gaps = np.diff(w[: kmax + 1])
    return int(np.argmax(gaps[: kmax]) + 1), w[: kmax + 1]


def silhouette_k(D, kmax=15):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score
    Z = linkage(squareform(D, checks=False), method="average")
    best, scores = None, {}
    for k in range(2, min(kmax, len(D) - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        scores[k] = float(s)
        if best is None or s > scores[best]:
            best = k
    return best, scores


def cluster(A, secs, thr_cos, min_core, refine_iters):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    core = np.where(secs >= min_core)[0]
    Ac = A[core]
    S = Ac @ Ac.T
    D = np.clip(1.0 - S, 0, 2)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    lab_core = fcluster(Z, 1.0 - thr_cos, criterion="distance")
    labs = sorted(set(lab_core))
    remap = {l: i for i, l in enumerate(labs)}
    lab_core = np.array([remap[l] for l in lab_core])
    k = len(labs)

    def centroids(idx, lab, kk):
        C = np.zeros((kk, A.shape[1]))
        for c in range(kk):
            sel = idx[lab == c]
            if len(sel):
                v = A[sel].mean(0)
                nv = np.linalg.norm(v)
                C[c] = v / nv if nv > 0 else 0.0
        return C

    C = centroids(core, lab_core, k)
    for _ in range(refine_iters):
        lab_core = np.argmax(Ac @ C.T, axis=1)
        C = centroids(core, lab_core, k)

    lab = np.full(len(A), -1, dtype=int)
    lab[core] = lab_core
    weak = np.where(secs < min_core)[0]
    for i in weak:
        lab[i] = int(np.argmax(A[i] @ C.T)) if np.linalg.norm(A[i]) > 0 else -1
    # final centroids over every assigned aggregate, weighted by embedded seconds
    Cf = np.zeros_like(C)
    for c in range(k):
        sel = np.where(lab == c)[0]
        sel = sel[np.linalg.norm(A[sel], axis=1) > 0]
        if len(sel):
            v = (A[sel] * secs[sel, None]).sum(0) / max(secs[sel].sum(), 1e-9)
            nv = np.linalg.norm(v)
            Cf[c] = v / nv if nv > 0 else 0.0
    return lab, k, core, weak, S, D, Cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clusters-out", default=None,
                    help="where the per-speaker centroids go. Defaults beside "
                         "--out, which is what the flat layout relied on; the "
                         "library names it <slug>-clusters.npz instead.")
    ap.add_argument("--thr", default="auto",
                    help="cosine cut, or 'auto' to self-calibrate per meeting "
                         "(constrained AHC + max merge-gap; see cluster_speakers.py)")
    ap.add_argument("--min-core", type=float, default=2.0,
                    help="speech a (window, local label) needs before it joins "
                         "the clustering core")
    ap.add_argument("--refine", type=int, default=3)
    # Defaults come FROM cluster_speakers, so each has one definition and the
    # flag cannot drift away from the value the library uses when unset.
    ap.add_argument("--durable", type=float,
                    default=cluster_speakers.DURABLE_S,
                    help="speech MOSS must have heard before 'these are two "
                         "different people' is binding. A low value suits "
                         "close-talking audio and a high one a far-field array; "
                         "there is no single right value, hence a knob.")
    ap.add_argument("--guard", type=int, default=10,
                    help="window span across which a cannot-link claim is trusted")
    ap.add_argument("--min-cluster-sec", type=float,
                    default=cluster_speakers.MIN_CLUSTER_SEC,
                    help="below this a cluster is absorbed rather than kept")
    ap.add_argument("--speaker-db", default=None,
                    help="speakers.db to match against. Every named voice in it "
                         "is a reference, so identity carries across recordings "
                         "without a separate linking pass. Absent, this meeting "
                         "is a cold start and its identities are provisional.")
    ap.add_argument("--roster", default="",
                    help="comma-separated names who could be in THIS recording. "
                         "Everyone else in the store is an impostor trial that "
                         "cannot be right and can only cost accuracy: measured "
                         "over 300 arguments, cutting a 391-person gallery to "
                         "the ~12 actually present halved wrong names, 0.82%% "
                         "to 0.35%%. For a meeting this is the calendar invite.")
    ap.add_argument("--condition", default=None,
                    help="what makes this recording sound the way it does -- "
                         "'telephone', 'far-field', a year, anything. A person "
                         "is stored once per circumstance: one averaged vector "
                         "cannot span a change of microphone, and pretending "
                         "otherwise names the wrong human.")
    ap.add_argument("--legacy-cluster", action="store_true",
                    help="use the agglomerative path instead of matching. Kept "
                         "for comparison: measured over 238 arguments it put "
                         "13.8%% of speech on the wrong person against 0.18%%.")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    fixed_thr = None if str(args.thr).lower() == "auto" else float(args.thr)

    R, E, seg_idx, meta = load(args.run, args.npz)
    keys, A, secs, nseg, key_of = aggregate(R, E, seg_idx, meta)
    name = R["audio"].split("/")[-1].replace(".wav", "")
    empty = int((nseg == 0).sum())
    print(f"AGG meeting={name} n_local_labels={len(keys)} embeddable={int((nseg>0).sum())} "
          f"empty={empty} core(>={args.min_core}s)={int((secs>=args.min_core).sum())} "
          f"median_secs={np.median(secs):.1f} total_secs={secs.sum():.0f}")

    if args.sweep:
        core = np.where(secs >= args.min_core)[0]
        Ac = A[core]
        Sc = Ac @ Ac.T
        Dc = np.clip(1.0 - Sc, 0, 2); np.fill_diagonal(Dc, 0)
        ke, w = eigengap_k(Sc.copy())
        ks, sc = silhouette_k(Dc)
        print(f"KEST meeting={name} eigengap_k={ke} silhouette_k={ks} "
              f"sil_scores={ {k: round(v,3) for k,v in sc.items()} }")
        for t in [0.10, 0.15, 0.20, 0.2656, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            lab, k, *_ = cluster(A, secs, t, args.min_core, args.refine)
            print(f"SWEEP meeting={name} thr={t:.4f} k={k}")

    name_of = {}
    if args.legacy_cluster:
        lab, k, core, weak, S, D, Cf, info = cluster_speakers.cluster(
            A, secs, keys, min_core=args.min_core, refine_iters=args.refine,
            thr=fixed_thr, durable=args.durable, guard=args.guard,
            min_cluster_sec=args.min_cluster_sec)
    else:
        # Match aggregates against known voices instead of clustering them into
        # each other. Agglomerative merges are transitive -- A joins B, B joins
        # C, and A and C are one speaker having never been compared -- which
        # over 238 SCOTUS arguments collapsed 11.3 speakers into 9.0 and put
        # 13.8% of all speech under the wrong name. Nothing here merges.
        bank = match_speakers.Bank()
        if args.speaker_db and os.path.exists(args.speaker_db):
            import sqlite3
            import speakers as spk
            names = [x.strip() for x in args.roster.split(",") if x.strip()]
            with sqlite3.connect(args.speaker_db) as _c:
                bank = match_speakers.load_bank(_c, spk.EMBED_MODEL,
                                                names=names or None)
            if names:
                print(f"ROSTER meeting={name} candidates={len(bank)} "
                      f"of {len(names)} named")
        lab, name_of, info = match_speakers.label_meeting(keys, A, secs, bank)
        k = info["k"]
        # centroids for the sidecar, weighted by speech so a long turn counts
        # for more than a short one
        An = match_speakers.unit_rows(np.asarray(A, dtype=np.float32))
        Cf = np.stack([match_speakers.unit(
            (An[lab == c] * np.asarray(secs)[lab == c][:, None]).sum(axis=0))
            if int((lab == c).sum()) else np.zeros(An.shape[1], np.float32)
            for c in range(k)]) if k else np.zeros((0, An.shape[1]), np.float32)
        info.setdefault("floor_ok", True)
    sizes = {int(c): float(secs[lab == c].sum()) for c in sorted(set(lab))}
    tshow = "n/a" if info["threshold"] is None else f"{info['threshold']:.4f}"
    print(f"CLUSTER meeting={name} thr={tshow} mode={info['mode']} "
          f"min_core={args.min_core} refine={args.refine} k_est={k} "
          f"cannot_link={info.get('n_cannot_link', 'n/a')} "
          f"min_k_floor={info.get('floor', 'n/a')} "
          f"cluster_secs={ {c: round(v) for c, v in sorted(sizes.items(), key=lambda x:-x[1])} }")
    if name_of:
        print(f"NAMED meeting={name} bank={info['bank']} "
              f"named={info['named']}/{k} "
              f"share_of_speech={info['named_share']:.1%} "
              f"who={ {f'G{c:02d}': n for c, n in sorted(name_of.items())} }")
    if info.get("low_separation"):
        # The tree had no clear gap to cut at, so the speaker count is a guess.
        # Worth saying out loud rather than presenting it with the same
        # confidence as a clean split.
        print(f"LOW-SEPARATION meeting={name} span={info.get('span')} "
              f"dominance={info.get('dominance')}: the merge heights barely spread, "
              f"so this speaker count is weakly supported. Check it before naming "
              f"anyone from this meeting.")
    if not info["floor_ok"]:
        # MOSS heard more people at once than the clustering produced. Something
        # is wrong -- do not let it pass as a clean transcript.
        print(f"FLOOR-VIOLATION meeting={name} k_est={k} < min_k_floor={info['floor']}: "
              f"MOSS labelled {info['floor']} distinct speakers inside one window, so "
              f"the clustering has collapsed people together.")

    lab_of_key = {kk: int(lab[i]) for i, kk in enumerate(keys)}
    # key_of has one entry per EMBEDDING META ROW and is indexed here by SEGMENT
    # position, so the two files have to agree on how many segments there are.
    # They always should -- embed_batched appends meta for every segment before
    # it decides whether to embed it -- and when they do not, the old code ran
    # off the end of a list and raised a bare "IndexError: list index out of
    # range" from inside a formatting expression, which says nothing about the
    # two counts that actually disagree or which file is short.
    if len(key_of) != len(R["segments"]):
        raise SystemExit(
            f"segment/embedding mismatch: {args.run} has {len(R['segments'])} "
            f"segments but {args.npz} carries metadata for {len(key_of)}. One of "
            f"them is stale or was written incompletely; re-run this meeting with "
            f"./transcribe --replace <id>.")
    # -1 means "not assigned to any speaker": an aggregate that was never
    # embedded, so the clustering had nothing to place it by. f"G{-1:02d}" spells
    # that "G-1", which reads as a speaker id and is counted as one by anything
    # consuming this file -- a phantom speaker in every meeting. mktxt renders it
    # as UNKNOWN correctly, so no reader ever saw it, but the JSON said otherwise
    # and an RTTM export built from it gained a whole extra speaker.
    for i, s in enumerate(R["segments"]):
        g = lab_of_key[key_of[i]]
        s["global"] = f"G{g:02d}" if g >= 0 else None
        # The name belongs on the segment, not only in a lookup elsewhere: a
        # consumer reading this file should not have to re-derive who someone is.
        if g >= 0 and g in name_of:
            s["speaker_name"] = name_of[g]
    if args.out:
        with open(args.out, "w") as _fh:
            json.dump(R, _fh)
        print(f"WROTE {args.out}")
        cs = np.array([secs[lab == c].sum() for c in range(k)])
        np.savez(args.clusters_out or args.out.replace(".json", "_clusters.npz"),
                 centroid=Cf.astype(np.float32),
                 cluster=np.array([f"G{c:02d}" for c in range(k)]),
                 secs=cs.astype(np.float32), meeting=np.array(name))

    if args.ref:
        ref = json.load(open(args.ref))
        # dominant reference speaker per local aggregate, by time overlap
        rs = sorted({t["speaker"] for t in ref})
        ridx = {s: i for i, s in enumerate(rs)}
        ov = {kk: np.zeros(len(rs)) for kk in keys}
        segtime = {kk: 0.0 for kk in keys}
        for i, s in enumerate(R["segments"]):
            kk = key_of[i]
            segtime[kk] += s["end"] - s["start"]
            for t in ref:
                a, b = max(s["start"], t["start"]), min(s["end"], t["end"])
                if b > a:
                    ov[kk][ridx[t["speaker"]]] += b - a
        pure_local = []
        # row k is the UNASSIGNED bucket (aggregates with no embeddable segment)
        conf = np.zeros((k + 1, len(rs)))
        nospeech = np.zeros(k + 1)   # segment time that overlaps no reference turn
        for i, kk in enumerate(keys):
            o = ov[kk]
            row = lab_of_key[kk] if lab_of_key[kk] >= 0 else k
            if o.sum() > 0:
                pure_local.append(o.max() / o.sum())
                conf[row, int(np.argmax(o))] += o.sum()
            nospeech[row] += max(0.0, segtime[kk] - o.sum())
        print(f"LOCALPURITY meeting={name} mean={np.mean(pure_local):.3f} "
              f"median={np.median(pure_local):.3f} frac_below_0.8={np.mean(np.array(pure_local)<0.8):.3f}")
        print(f"CONF meeting={name} rows=clusters cols={rs} (secs = reference-overlap seconds)")
        for c in range(k + 1):
            row = conf[c]
            nm = f"G{c:02d}" if c < k else "UNASSIGNED"
            if row.sum() == 0 and nospeech[c] == 0:
                continue
            pur = row.max() / row.sum() if row.sum() > 0 else float("nan")
            top = rs[int(np.argmax(row))] if row.sum() > 0 else "-"
            print(f"  {nm} ref_secs={row.sum():6.0f} offref_secs={nospeech[c]:5.0f} "
                  f"purity={pur:.3f} top={top} "
                  + " ".join(f"{rs[j]}:{row[j]:.0f}" for j in range(len(rs)) if row[j] > 0))
        # reference-side recall: how much of each speaker landed in its best cluster
        for j, sp in enumerate(rs):
            col = conf[:, j]
            tot = sum(t["end"] - t["start"] for t in ref if t["speaker"] == sp)
            print(f"  REFSPK {sp} ref_secs={tot:6.0f} captured={col.sum():6.0f} "
                  f"best_cluster={'G%02d'%int(np.argmax(col)) if int(np.argmax(col))<k else 'UNASSIGNED'} "
                  f"best_frac={(col.max()/col.sum() if col.sum()>0 else 0):.3f}")


if __name__ == "__main__":
    main()
