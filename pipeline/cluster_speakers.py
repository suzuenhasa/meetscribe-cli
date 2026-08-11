#!/usr/bin/env python3
"""Constrained speaker clustering with a per-meeting self-calibrated cut.

Replaces the fixed cosine threshold, which does not survive a change of recording
setup: 0.2656 is right for ICSI far-field room mics and collapses every speaker in
a Zoom recording into one, silently. Measured over four meetings (k, and DER under
the oracle speaker map where a reference exists):

  method                          Bdb001(6)   Bed014(6)   Bmr011(8)   Zoom(~6)
  fixed 0.2656          [old]     6  30.8%    6  35.6%    8  31.5%    1   <- silent failure
  NME-SC                          5           5           5  36.4%    7
  AHC + silhouette k              5           6           8  31.5%    2
  AHC + max merge-gap             6  30.8%    6  35.6%    8  31.5%    4
  CL-AHC to completion            5           5           5  36.4%    -
  THIS: CL-AHC + max merge-gap    6  30.8%    6  35.6%    8  31.5%    7

Both halves are load-bearing — the constraints alone give 5 on the Zoom file, the
self-calibrated cut alone gives 4, together they give 7:

1. CANNOT-LINK. Two speaker-turns from the SAME window carrying different MOSS
   labels (S01 vs S02) cannot be the same person. MOSS asserting "two people here"
   is a categorical claim rather than a distance, so codec damage, AGC and noise
   suppression cannot corrupt it the way they corrupt cosine similarity. Blocked
   pairs never merge and a merged node inherits its children's blocks.

   Two guards, both measured to matter: a turn must hold >= `durable` seconds to
   constrain anything (sub-second fragments are where MOSS's labelling is least
   reliable), and windows claiming more than `guard` distinct speakers are ignored
   as degenerate.

2. SELF-CALIBRATED CUT. Sort the constrained tree's merge heights, take the
   largest gap between consecutive heights, cut at its midpoint — the point at
   which the tree stops wanting to merge. Nothing crosses a meeting boundary and
   no constant is fitted. Derived cuts: 0.339 / 0.235 / 0.267 on the three ICSI
   meetings, bracketing the old hand-fitted 0.2656 and reproducing its DER
   exactly, and 0.659 on the Zoom file, which is the entire point.

3. ABSORB SPLINTERS. The cut tends to shave off 3-5 second fragments — one
   audience question, a cough — and each would become a spurious speaker. Any
   cluster under MIN_CLUSTER_SEC is folded into its nearest neighbour. This can
   only merge clusters too small to enroll a profile, so it never touches a real
   speaker.

4. FALLBACK. With zero cannot-link pairs MOSS never saw two people in one window,
   so the recording is plausibly one speaker — and a largest-gap rule always finds
   a largest gap, so it splits a monologue rather than returning k=1. With no
   constraints to calibrate against we fall back to the constant.

TRIED AND REJECTED -- do not rebuild this. A per-cluster outlier guard (expel a
turn whose similarity sits N robust deviations below its cluster's own internal
cohesion) was implemented and measured, aimed at the podcast's MC: they speak once
for 12.6s, sit at 0.891 against a cluster that is internally 0.920, and get absorbed
by cluster mass. It cannot work. On homogeneous synthetic clusters the guard expels
someone 13-28% of the time at z>3, and needs z>=5 before false positives reach zero
-- but the MC's own score is z=3.11, ranking only 5th of 129 in that cluster, behind
four genuine turns at 6.23/4.62/3.99/3.23. Every setting that catches the MC also
fragments real speakers. The MC is not separable by any rule over these embeddings:
its similarity to the cluster (0.891) is BELOW the mean of all pairs in the recording
(0.893). That is an embedding-capacity limit, not a clustering bug.

Reported but never acted on: `low_separation`, set when the merge heights barely
span any range or the winning gap does not stand out. It means the speaker split
is a guess. Acting on it was tried and reverted — the one recording that trips it
is a live event whose true speaker count was higher, not lower, than the
fall-back would have given.

The min-k floor (`min_k_floor`) is a separate, embedding-free check: the most
distinct labels MOSS emitted inside any one window. It is a floor, not a counter —
measured 4 / 5 / 5 / 4 against true 6 / 6 / 8 / 6 — so it is used only to catch a
clustering that has collapsed below what MOSS itself witnessed.
"""
import numpy as np
from collections import Counter

FALLBACK_THR = 0.2656   # only reached when there are no cannot-link pairs at all

# A cluster holding less speech than this is absorbed into its nearest surviving
# neighbour. Not a fitted number: it is speakers.py's MIN_ENROLL_SEC, and the
# argument is the same one -- too little speech to support a speaker profile is
# too little to call a speaker. It removes the 3-5 second splinters the max-gap
# cut leaves behind without touching any cluster big enough to matter.
MIN_CLUSTER_SEC = 10.0


# Weight of the prosody/pitch similarity when fused with the neural embedding's.
# Measured across 7 recordings (5 with real ground truth, 4 of those never used to
# design anything), as fraction of same-speaker pairs that look less alike than a
# random different-speaker pair:
#
#   w      Bdb001  Bed002  Bed003  Bed004  Bed005   Zoom  Podcast   WORST
#   0.00     0.0%    0.5%    1.4%    0.0%    3.1%   1.2%     6.7%    7.0%
#   0.25     0.0%    0.9%    1.0%    0.0%    2.4%   0.8%     3.7%    3.8%
#   1.00    20.3%   29.7%   34.2%   30.7%   31.7%  17.3%     9.4%   34.2%
#
# 0.2-0.35 is a plateau rather than a fitted point. Pitch alone is far worse
# everywhere, so this is a minority vote that rescues the recordings where the
# neural embedding has collapsed, at a cost of <=0.4pp on the ones where it has not.
# DEFAULT 0. Pitch fusion was built to rescue embeddings that were collapsing on
# real-world audio. That collapse turned out to be a missing resample in
# embed_batched.py -- 44.1 kHz mp3 handed to a 16 kHz model. With the audio fixed,
# WeSpeaker goes from d'=1.82 to d'=5.99 on the podcast unaided, and fusion changes
# nothing while costing ~90s and two ICSI speaker-count regressions. Kept, not
# deleted, because the machinery is measured and correct -- but off.
PROSODY_WEIGHT = 0.0

# Reported alongside the result, NOT used to change behaviour. Where the max-gap
# cut is reliable the merge heights span 0.37-0.83 and the winning gap is 2-12x
# the runner-up; on the one recording where the tree carries no usable structure
# the span is 0.16 and the ratio 1.4. Low values mean the speaker split is a
# guess, so say so rather than silently switching strategy.
LOW_SPAN = 0.25
LOW_DOMINANCE = 2.0


def cannot_link_matrix(keys_core, secs_core, durable=1.0, guard=10):
    """Boolean n x n over core nodes, True where a merge is forbidden.

    keys_core: [(window, local_label)] for the core nodes, aligned with secs_core.
    """
    n = len(keys_core)
    wins = [k[0] for k in keys_core]
    per_window = Counter(wins)
    ok = np.asarray(secs_core) >= durable
    C = np.zeros((n, n), dtype=bool)
    for a in range(n):
        if not ok[a] or per_window[wins[a]] > guard:
            continue
        for b in range(a + 1, n):
            if wins[a] == wins[b] and ok[b] and keys_core[a][1] != keys_core[b][1]:
                C[a, b] = C[b, a] = True
    return C


def constrained_linkage(S, cannot):
    """Greedy average-linkage honouring cannot-link, run to completion.

    Returns (heights, labels_at). `labels_at(thr)` replays the recorded merge
    order to give the labels you would get cutting at cosine `thr`, so sweeping
    thresholds costs one linkage pass rather than one per threshold.
    """
    n = len(S)
    alive = list(range(n))
    Ssum = np.asarray(S, dtype=np.float64).copy()
    cnt = np.ones((n, n))
    CL = cannot.copy()
    heights, order = [], []

    while True:
        best, bi, bj = -np.inf, -1, -1
        for a in range(len(alive)):
            i = alive[a]
            for j in alive[a + 1:]:
                if CL[i, j]:
                    continue
                s = Ssum[i, j] / cnt[i, j]
                if s > best:
                    best, bi, bj = s, i, j
        if bi < 0:                        # every remaining pair is blocked
            break
        heights.append(best)
        order.append((bi, bj))
        for j in alive:
            if j == bi or j == bj:
                continue
            Ssum[bi, j] = Ssum[j, bi] = Ssum[bi, j] + Ssum[bj, j]
            cnt[bi, j] = cnt[j, bi] = cnt[bi, j] + cnt[bj, j]
            CL[bi, j] = CL[j, bi] = CL[bi, j] or CL[bj, j]
        alive.remove(bj)

    def labels_at(thr):
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for (bi, bj), h in zip(order, heights):
            if h < thr:
                break
            par[find(bj)] = find(bi)
        roots, lab = {}, np.zeros(n, dtype=int)
        for i in range(n):
            r = find(i)
            if r not in roots:
                roots[r] = len(roots)
            lab[i] = roots[r]
        return lab, len(roots)

    return np.asarray(heights, dtype=float), labels_at


def maxgap_threshold(heights):
    """Cut at the midpoint of the largest gap between consecutive merge heights.

    Returns (thr, span, dominance). The two statistics describe how much gap
    structure there was to read — a largest-gap rule always finds a largest gap,
    including in noise — and are reported, not acted on:

      span       range covered by the merge heights. 0.37-0.83 where the cut is
                 reliable; 0.16 on a recording whose embeddings are so compressed
                 that every merge happens between cos 0.80 and 0.97.
      dominance  winning gap over the runner-up. 2-12x where reliable, 1.4x on
                 that same recording.
    """
    h = np.sort(np.asarray(heights, dtype=float))
    if len(h) < 3:
        return FALLBACK_THR, 0.0, 0.0
    g = np.diff(h)
    j = int(np.argmax(g))
    gs = np.sort(g)[::-1]
    return (float((h[j] + h[j + 1]) / 2), float(h[-1] - h[0]),
            float(gs[0] / max(gs[1], 1e-9)))


def refine_leave_one_out(Ac, lab_core, k, iters=3):
    """Reassign each turn to its nearest cluster centroid, EXCLUDING itself.

    Plain refinement compares a turn against a centroid it is itself part of. In
    a small cluster that centroid is mostly the turn, so it always wins and the
    turn can never move however well it matches the speaker it belongs to. On the
    podcast this pinned one of Sreeram's turns into a 9-member junk cluster at
    0.9328 while it sat at 0.9079 from his real cluster; removing it from its own
    centroid drops that to 0.9144, and once the neighbouring turns move the
    centroid shifts far enough that it rejoins him.

    Updates are sequential, not batched: a turn that moves changes the centroids
    the next turn sees, which is what makes the cascade converge. Measured five
    moves over two passes on the podcast.
    """
    lab = np.asarray(lab_core, dtype=int).copy()
    n, d = Ac.shape
    sums = np.zeros((k, d))
    cnt = np.zeros(k)
    for c in range(k):
        sel = lab == c
        cnt[c] = int(sel.sum())
        if cnt[c]:
            sums[c] = Ac[sel].sum(0)

    def unit(v):
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-9 else None

    for _ in range(max(1, iters)):
        moved = 0
        for i in range(n):
            c = int(lab[i])
            if cnt[c] <= 1:
                # A turn that is the only member of its cluster is a speaker who
                # appears once. Leaving its own similarity at -inf made its cluster
                # unreachable and FORCED it into someone else's -- which is exactly
                # how the podcast's one-off MC kept being swallowed by the host
                # despite sitting at 0.125 similarity against a 0.511 all-pairs mean.
                continue
            sims = np.full(k, -np.inf)
            for j in range(k):
                v = sums[j] - Ac[i] if j == c else sums[j]
                u = unit(v)
                if u is not None:
                    sims[j] = float(Ac[i] @ u)
            b = int(np.argmax(sims))
            if not np.isfinite(sims[b]) or b == c:
                continue
            sums[c] -= Ac[i]; cnt[c] -= 1
            sums[b] += Ac[i]; cnt[b] += 1
            lab[i] = b
            moved += 1
        if not moved:
            break
    return lab



def absorb_small(lab_core, core, A, secs, min_sec=MIN_CLUSTER_SEC):
    """Fold clusters holding under `min_sec` of speech into the nearest survivor.

    The max-gap cut tends to shave off 3-5 second splinters — a questioner's one
    remark, a cough — and each becomes a spurious speaker. Absorbing them is
    safe in a way that raising the threshold is not: it can only merge clusters
    too small to enroll a profile, and never touches a real speaker's turns.
    Always leaves at least one cluster.
    """
    lab_core = np.asarray(lab_core, dtype=int).copy()
    while True:
        ids = sorted(set(lab_core.tolist()))
        if len(ids) < 2:
            break
        tot = {c: float(secs[core[lab_core == c]].sum()) for c in ids}
        small = [c for c in ids if tot[c] < min_sec]
        if not small:
            break
        victim = min(small, key=lambda c: tot[c])
        cents = {}
        for c in ids:
            sel = core[lab_core == c]
            v = A[sel].mean(0)
            nv = np.linalg.norm(v)
            cents[c] = v / nv if nv > 0 else v
        others = [c for c in ids if c != victim]
        best = max(others, key=lambda c: float(cents[victim] @ cents[c]))
        lab_core[lab_core == victim] = best
    ids = sorted(set(lab_core.tolist()))
    remap = {c: i for i, c in enumerate(ids)}
    return np.array([remap[c] for c in lab_core]), len(ids)


def min_k_floor(keys, secs, durable=1.0, guard=10):
    """Lower bound on the speaker count, from MOSS alone — no embeddings involved.

    The most distinct local labels MOSS emitted inside any single window, counting
    only turns of at least `durable` seconds and ignoring windows that claim more
    than `guard` speakers. MOSS witnessed that many people at once, so the meeting
    cannot hold fewer.

    A floor, not a counter: measured 4 / 5 / 5 / 4 against true 6 / 6 / 8 / 6. It
    falsifies k=1 for free and should not be promoted beyond that.
    """
    per_win = {}
    for (w, l), s in zip(keys, secs):
        if s >= durable:
            per_win.setdefault(w, set()).add(l)
    counts = [len(v) for v in per_win.values() if len(v) <= guard]
    return max(counts) if counts else 1


def choose_threshold(A, keys, secs, min_core=2.0, durable=1.0, guard=10,
                     pros=None, weight=PROSODY_WEIGHT):
    """Pick this meeting's clustering cut. -> (thr, info)

    A: (n_keys, D) L2-normalised aggregates, aligned with `keys` and `secs`.

    Three modes, in order of preference:
      "max-gap"     the constrained tree has a clear gap; cut there.
      "constraints" it does not; merge everything the constraints permit. That
                    is the smallest speaker count consistent with every
                    cannot-link MOSS asserted, and it is >= the min-k floor by
                    construction, since the floor's speakers are mutually blocked
                    and so can never merge into each other.
      "fixed"       no constraints at all, so nothing to calibrate against.
    """
    core = np.where(np.asarray(secs) >= min_core)[0]
    Ac = A[core]
    S = Ac @ Ac.T
    if pros is not None:
        S = fuse(S, np.asarray(pros)[core], weight)

    C = cannot_link_matrix([keys[i] for i in core], np.asarray(secs)[core],
                           durable, guard)
    n_cl = int(C.sum() // 2)
    floor = min_k_floor(keys, secs, durable, guard)
    info = {"n_cannot_link": n_cl, "floor": int(floor), "n_core": int(len(core))}

    if n_cl == 0:
        # MOSS never saw two people in one window, so this is plausibly one
        # speaker -- and every self-calibrating rule invents splits there.
        info.update(threshold=FALLBACK_THR, mode="fixed", self_calibrated=False,
                    low_separation=False)
        return FALLBACK_THR, info

    heights, _ = constrained_linkage(S, C)
    thr, span, dominance = maxgap_threshold(heights)
    info.update(threshold=round(float(thr), 4), mode="max-gap",
                self_calibrated=True, span=round(span, 4),
                dominance=round(dominance, 2),
                low_separation=bool(span < LOW_SPAN or dominance < LOW_DOMINANCE))
    return thr, info


def fuse(Se, P, weight=PROSODY_WEIGHT):
    """Blend the embedding similarity matrix with a prosody similarity matrix.

    Score-level, not feature-level: concatenating the two feature vectors was
    measured to make things WORSE (podcast d' 1.97 -> 1.49) because the spaces
    have incomparable scales. Each matrix is z-normalised over its own
    off-diagonal distribution so `weight` means the same thing in both, then the
    result is mapped back onto the embedding's original scale so that cosine
    thresholds elsewhere in this module keep their meaning.
    """
    if P is None or weight <= 0 or len(Se) < 3:
        return Se
    P = np.asarray(P, dtype=float)
    if P.shape[0] != Se.shape[0] or not np.isfinite(P).all():
        return Se
    # A turn with no voiced frames has no pitch. Such rows arrive as NaN and must
    # stay NEUTRAL: if they are z-scored like everything else they all collapse
    # onto the same vector, look perfectly alike, and form a junk cluster. That
    # cost exactly +1 spurious speaker on all three ICSI meetings when this
    # function first shipped -- DER was unchanged, which is how it was spotted.
    valid = np.isfinite(P).all(1)
    if valid.sum() < 3:
        return Se
    V = P[valid]
    keep = V.std(0) > 1e-9
    if keep.sum() < 2:
        return Se
    Z = np.zeros((len(P), int(keep.sum())))
    Z[valid] = (V[:, keep] - V[:, keep].mean(0)) / V[:, keep].std(0)
    Z[valid] /= np.maximum(np.linalg.norm(Z[valid], axis=1, keepdims=True), 1e-9)
    Sp = Z @ Z.T

    iu = np.triu_indices(len(Se), 1)
    e_mu, e_sd = Se[iu].mean(), Se[iu].std() + 1e-9
    both = np.outer(valid, valid)                   # only pairs with pitch on both sides
    pv = Sp[both & ~np.eye(len(Se), dtype=bool)]
    if pv.size < 2 or pv.std() < 1e-9:
        return Se
    F = ((Se - e_mu) / e_sd).copy()
    fused = (1 - weight) * ((Se - e_mu) / e_sd) + weight * ((Sp - pv.mean()) / pv.std())
    F[both] = fused[both]                           # elsewhere: embedding alone
    F = (F / (F[iu].std() + 1e-9)) * e_sd + e_mu    # back onto the cosine scale
    F = (F + F.T) / 2
    np.fill_diagonal(F, 1.0)
    return np.clip(F, -1.0, 1.0)


def aggregate_prosody(pros, seg_idx, meta, keys):
    """Pool per-segment prosody into one vector per (window, local) key.

    Weighted by voiced-frame count (column 0), so pooling several clips
    approximates pooling their raw pitch samples. Returns (n_keys, PROS_DIM-1)
    aligned with `keys`, or None if the npz predates prosody extraction.
    """
    if pros is None or not len(pros):
        return None
    pos = {int(s): i for i, s in enumerate(seg_idx)}
    dim = pros.shape[1] - 1
    acc = {k: (np.zeros(dim), 0.0) for k in keys}
    for m in meta:
        k = (m["window"], m["local"])
        if k not in acc or m["idx"] not in pos:
            continue
        row = pros[pos[m["idx"]]]
        n = float(row[0])
        if n <= 0:
            continue
        v, w = acc[k]
        acc[k] = (v + row[1:] * n, w + n)
    out = np.full((len(keys), dim), np.nan)         # NaN = no pitch, stays neutral
    for i, k in enumerate(keys):
        v, w = acc[k]
        if w > 0:
            out[i] = v / w
    return out


def cluster(A, secs, keys, min_core=2.0, refine_iters=3, thr=None,
            durable=1.0, guard=10, pros=None, weight=PROSODY_WEIGHT):
    """Drop-in for link.py's cluster(), with the threshold chosen per meeting.

    Pass `thr` to force a fixed cut (and skip the constraints entirely); leave it
    None to self-calibrate. Returns (lab, k, core, weak, S, D, Cf, info) — the
    same tuple link.py already unpacks, plus the info dict.
    """
    secs = np.asarray(secs, dtype=float)
    core = np.where(secs >= min_core)[0]
    Ac = A[core]
    S = Ac @ Ac.T
    if pros is not None:
        S = fuse(S, np.asarray(pros)[core], weight)
    D = np.clip(1.0 - S, 0, 2)
    np.fill_diagonal(D, 0.0)

    if thr is None:
        thr, info = choose_threshold(A, keys, secs, min_core, durable, guard,
                                     pros=pros, weight=weight)
        C = cannot_link_matrix([keys[i] for i in core], secs[core], durable, guard)
    else:
        info = {"threshold": round(float(thr), 4), "n_cannot_link": 0,
                "floor": int(min_k_floor(keys, secs, durable, guard)),
                "self_calibrated": False, "mode": "fixed", "n_core": int(len(core)),
                "low_separation": False}
        C = np.zeros((len(core), len(core)), dtype=bool)

    _, labels_at = constrained_linkage(S, C)
    lab_core, k_raw = labels_at(thr)
    lab_core, k = absorb_small(lab_core, core, A, secs)
    info["k_before_absorb"] = int(k_raw)

    def centroids(lab, kk):
        Cm = np.zeros((kk, A.shape[1]))
        for c in range(kk):
            sel = core[lab == c]
            if len(sel):
                v = A[sel].mean(0)
                nv = np.linalg.norm(v)
                Cm[c] = v / nv if nv > 0 else 0.0
        return Cm

    lab_core = refine_leave_one_out(Ac, lab_core, k, refine_iters)
    ids = sorted(set(lab_core.tolist()))
    lab_core = np.array([{c: i for i, c in enumerate(ids)}[c] for c in lab_core])
    k = len(ids)
    Cm = centroids(lab_core, k)

    lab = np.full(len(A), -1, dtype=int)
    lab[core] = lab_core
    weak = np.where(secs < min_core)[0]
    for i in weak:
        lab[i] = int(np.argmax(A[i] @ Cm.T)) if np.linalg.norm(A[i]) > 0 else -1

    # final centroids over every assigned aggregate, weighted by embedded seconds
    Cf = np.zeros_like(Cm)
    for c in range(k):
        sel = np.where(lab == c)[0]
        sel = sel[np.linalg.norm(A[sel], axis=1) > 0]
        if len(sel):
            v = (A[sel] * secs[sel, None]).sum(0) / max(secs[sel].sum(), 1e-9)
            nv = np.linalg.norm(v)
            Cf[c] = v / nv if nv > 0 else 0.0

    k_final = len(set(lab_core.tolist()))
    info["k"] = k_final
    info["floor_ok"] = k_final >= info["floor"]
    return lab, k_final, core, weak, S, D, Cf, info
