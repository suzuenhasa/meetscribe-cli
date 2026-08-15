#!/usr/bin/env python3
"""Constrained speaker clustering with a per-meeting self-calibrated cut.

The cut used to be the constant 0.2656. It is right for ICSI far-field room mics
and collapses every speaker in a Zoom recording into one — a wrong threshold does
not raise, it just reports one speaker. So it is now derived per recording, from
two signals that fail in different ways:

1. CANNOT-LINK. Two speaker-turns from the SAME window carrying different MOSS
   labels (S01 vs S02) cannot be the same person. MOSS asserting "two people here"
   is a categorical claim rather than a distance, so codec damage, AGC and noise
   suppression cannot corrupt it the way they corrupt cosine similarity. Blocked
   pairs never merge and a merged node inherits its children's blocks.

2. SELF-CALIBRATED CUT. Cut the constrained tree at the midpoint of the largest
   gap between consecutive merge heights — where the tree stops wanting to merge.
   No constant is fitted and nothing crosses a meeting boundary.

Both halves are load-bearing: on the Zoom recording the constraints alone find 5
speakers, the calibrated cut alone finds 4, together they find 7. choose_threshold()
combines them and handles the case where there is nothing to calibrate against.

TRIED AND REJECTED — do not rebuild. A per-cluster outlier guard (expel a turn
sitting N robust deviations below its cluster's own internal cohesion) was built
and measured, aimed at a podcast MC who speaks once for 12.6s and gets absorbed by
cluster mass. Every setting that catches them also fragments real speakers: their
score is z=3.11, ranking 5th of 129 in that cluster, behind four genuine turns.
Their similarity to the cluster that swallowed them (0.891) is below the mean of
all pairs in the recording (0.893) — an embedding-capacity limit, not a bug here.
"""
import numpy as np
from collections import Counter

FALLBACK_THR = 0.2656   # only reached when there are no cannot-link pairs at all

# Absorption floor, see absorb_small(). Not a fitted number: it is speakers.py's
# MIN_ENROLL_SEC, on the same argument -- too little speech to support a speaker
# profile is too little to call a speaker.
MIN_CLUSTER_SEC = 10.0

# Below either of these the speaker split is a guess. Reported, never acted on:
# acting on it was tried and reverted, because the one recording that trips it is
# a live event whose true speaker count was HIGHER than any fallback would give.
# See maxgap_threshold() for what the two statistics measure.
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
    podcast this pinned one of the host's turns into a 9-member junk cluster at
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


def core_set(secs, min_core):
    """Indices of the aggregates long enough to cluster on. Never empty.

    A recording where nothing clears `min_core` still has a speaker in it. When
    this returned nothing the centroid matrix came out (0, D), and assigning the
    remaining turns called argmax on an empty array -- one recording whose only
    aggregate was 1.9s crashed link.py, and because batch.py did not check
    return codes the run reported success with that meeting's transcript simply
    missing. Falling back to the single longest aggregate gives k=1, which is
    the right answer for a recording that short.
    """
    secs = np.asarray(secs, dtype=float)
    core = np.where(secs >= min_core)[0]
    if len(core) or not len(secs):
        return core
    # Everything that was embedded at all, not merely the longest one. Falling
    # back to a single aggregate discards every cannot-link MOSS asserted, so the
    # tree has one node and the answer can only ever be k=1: ten windows of
    # two-party dialogue, none of whose aggregates cleared min_core, came out as
    # one speaker with all twenty aggregates labelled G00 and the run exiting 0.
    # That is worse than the crash it replaced, which at least got reported.
    # `secs > 0` is exactly "has an embedded segment", so no zero-norm row enters
    # the core and the centroid matrix stays well formed.
    alive = np.where(secs > 0)[0]
    if len(alive):
        return alive
    return np.array([int(np.argmax(secs))], dtype=int)


def choose_threshold(A, keys, secs, min_core=2.0, durable=1.0, guard=10):
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
    core = core_set(secs, min_core)
    Ac = A[core]
    S = Ac @ Ac.T

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


def cluster(A, secs, keys, min_core=2.0, refine_iters=3, thr=None,
            durable=1.0, guard=10):
    """Drop-in for link.py's cluster(), with the threshold chosen per meeting.

    Pass `thr` to force a fixed cut (and skip the constraints entirely); leave it
    None to self-calibrate. Returns (lab, k, core, weak, S, D, Cf, info) — the
    same tuple link.py already unpacks, plus the info dict.
    """
    secs = np.asarray(secs, dtype=float)
    core = core_set(secs, min_core)
    Ac = A[core]
    S = Ac @ Ac.T
    D = np.clip(1.0 - S, 0, 2)
    np.fill_diagonal(D, 0.0)

    if thr is None:
        thr, info = choose_threshold(A, keys, secs, min_core, durable, guard)
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
    # Everything core_set did not take, rather than everything under min_core.
    # Those were the same set until core_set gained its fallback; now the fallback
    # aggregate is below the threshold AND is core, and letting the weak pass
    # reassign it overwrote its label -- harmless for a unit vector, but a
    # never-embedded aggregate has norm 0 and came back -1, so the meeting
    # reported k=1 with nothing assigned to it.
    weak = np.setdiff1d(np.where(secs < min_core)[0], core)
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
