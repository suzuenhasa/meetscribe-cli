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
import os

import numpy as np
from collections import Counter

FALLBACK_THR = 0.2656   # only reached when there are no cannot-link pairs at all

# Absorption floor, see absorb_small(). Not a fitted number: it is speakers.py's
# MIN_ENROLL_SEC, on the same argument -- too little speech to support a speaker
# profile is too little to call a speaker.
MIN_CLUSTER_SEC = 10.0

# Least similarity at which a sub-min_core aggregate may be given to a cluster.
# Below it the aggregate goes unassigned rather than to the nearest centroid.
# Measured on this corpus, different-speaker pairs run median 0.104 and p90
# 0.332, so anything under ~0.35 is indistinguishable from an impostor.
WEAK_MIN = 0.35

# Speech MOSS must have heard before "these are two different people" is taken as
# binding. Above min_core, or it can never fire -- see cannot_link_matrix.
DURABLE_S = float(os.environ.get("MS_CANNOT_LINK_MIN_S", "6.0"))

# Below either of these the speaker split is a guess. Reported, never acted on:
# acting on it was tried and reverted, because the one recording that trips it is
# a live event whose true speaker count was HIGHER than any fallback would give.
# See maxgap_threshold() for what the two statistics measure.
LOW_SPAN = 0.25
LOW_DOMINANCE = 2.0


def cannot_link_matrix(keys_core, secs_core, durable=DURABLE_S, guard=10):
    """Boolean n x n over core nodes, True where a merge is forbidden.

    keys_core: [(window, local_label)] for the core nodes, aligned with secs_core.

    `durable` is how much speech MOSS needs to have heard before its "these are
    two different people" is trusted. The old default of 1.0 could never fire:
    this is handed secs[core], already filtered to >= min_core (2.0), so any
    value below that was unreachable and the parameter did nothing.

    It matters because the claim is not always right. On AliMeeting far-field
    the constraints forced a floor ABOVE the true speaker count in 6 of 8
    sessions -- unfixable at any threshold, because a blocked pair never merges.
    Raising this to 6s cut the mean floor from 4.5 to 3.5 against a true 3.1 and
    took cpCER from 24.8% to 22.4%. Dropping the constraints entirely is far
    worse (57.6%), so the answer is to trust the LONG claims, not to stop
    trusting them.
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
    Ssum = np.asarray(S, dtype=np.float64).copy()
    cnt = np.ones((n, n))
    CL = np.asarray(cannot).astype(bool).copy()
    alive = np.ones(n, dtype=bool)
    heights, order = [], []

    # `Rm` holds the current average-linkage similarity of every MERGEABLE pair
    # and -inf everywhere else: lower triangle, dead nodes, cannot-link pairs.
    # Keeping it live is what makes this affordable. The obvious way to write
    # this loop -- rescanning every surviving pair in Python on every merge --
    # is O(n^3) INTERPRETED, which is fine at the 91 labels a 32-minute meeting
    # produces and ruinous at the 1,163 an 8-hour one does: 117s of a 452s run,
    # twice over, since choose_threshold() links as well. Same merges, same
    # heights to the bit, same labels at every threshold -- 269x measured.
    idx = np.arange(n)
    Rm = Ssum / cnt
    Rm[np.tril_indices(n)] = -np.inf
    Rm[CL & np.triu(np.ones((n, n), dtype=bool), 1)] = -np.inf

    while True:
        f = int(np.argmax(Rm))
        best = Rm.flat[f]
        if not np.isfinite(best):         # every remaining pair is blocked
            break
        bi, bj = divmod(f, n)
        heights.append(float(best))
        order.append((bi, bj))
        # Lance-Williams update for average linkage, as slice arithmetic. `o` is
        # every node that survives and is neither side of this merge, which is
        # exactly what the per-j loop used to skip.
        o = alive.copy()
        o[bi] = False
        o[bj] = False
        Ssum[bi, o] += Ssum[bj, o]
        Ssum[o, bi] = Ssum[bi, o]
        cnt[bi, o] += cnt[bj, o]
        cnt[o, bi] = cnt[bi, o]
        CL[bi, o] |= CL[bj, o]
        CL[o, bi] = CL[bi, o]
        alive[bj] = False
        # bj is gone; bi absorbed it, so only its row and column moved.
        Rm[bj, :] = -np.inf
        Rm[:, bj] = -np.inf
        r = np.where(o & ~CL[bi], Ssum[bi] / cnt[bi], -np.inf)
        Rm[bi, :] = np.where(idx > bi, r, -np.inf)
        Rm[:, bi] = np.where(idx < bi, r, -np.inf)

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


def refine_leave_one_out(Ac, lab_core, k, iters=3, cannot=None):
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
        # Recomputed per sweep, not once: membership changes as the sweep runs,
        # so a cached table would forbid moves that became legal and permit ones
        # that stopped being.
        blocked = None
        if cannot is not None and len(cannot):
            M = np.zeros((k, n), dtype=bool)
            for c in range(k):
                M[c] = (lab == c)
            blocked = (cannot.astype(np.int8) @ M.T) > 0      # (n, k)
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
                # MOSS said this turn and a member of cluster j are different
                # people. Refinement used to move on cosine alone and could
                # place them together, undoing after the cut what the linkage
                # had honoured throughout. Unreachable, not merely unlikely.
                if blocked is not None and j != c and blocked[i, j]:
                    continue
                v = sums[j] - Ac[i] if j == c else sums[j]
                u = unit(v)
                if u is not None:
                    sims[j] = float(Ac[i] @ u)
            b = int(np.argmax(sims))
            if not np.isfinite(sims[b]) or b == c:
                continue
            # Re-check against the CURRENT labels before committing. `blocked`
            # is built once per sweep, but moves land one at a time as the sweep
            # runs -- that sequencing is deliberate, it is what lets a cascade
            # converge. So a forbidden partner can leave cluster X for cluster Y
            # mid-sweep while the table still records it in X, and the next turn
            # is waved into Y right on top of it. Measured: 1 meeting in 200
            # reached the postcondition this way. O(n) against a table lookup,
            # and it reads the labels as they are rather than as they were.
            if cannot is not None and len(cannot) and (cannot[i] & (lab == b)).any():
                continue
            sums[c] -= Ac[i]; cnt[c] -= 1
            sums[b] += Ac[i]; cnt[b] += 1
            lab[i] = b
            moved += 1
        if not moved:
            break
    return lab


def cluster_cannot_link(lab_core, cannot, k):
    """k x k: True where two CLUSTERS hold an aggregate pair MOSS forbade.

    Lifts the per-aggregate relation onto whatever clusters currently exist, so
    a merge or a reassignment can be tested in one lookup instead of scanning
    members. Recomputed rather than maintained: it is a (k x n) x (n x k)
    boolean product, microseconds at the sizes here, and a stale copy is exactly
    the bug this is here to prevent."""
    if cannot is None or not len(cannot):
        return np.zeros((k, k), dtype=bool)
    # int32, NOT int8. This counts forbidden MEMBER PAIRS between two clusters
    # and then asks whether the count is positive -- so the accumulator has to
    # hold it. In int8 a pair of 16-member clusters MOSS fully separated gives
    # 256, which wraps to exactly 0 and reports them as free to merge; 128..255
    # wrap negative and are just as false. Entirely reachable: the 7.94h
    # recording clusters 1,163 aggregates into 23 speakers, so a cluster pair
    # with more than 127 forbidden member pairs is ordinary.
    M = np.zeros((k, len(lab_core)), dtype=np.int32)
    for c in range(k):
        M[c] = (lab_core == c)
    return ((M @ cannot) @ M.T) > 0


def absorb_small(lab_core, core, A, secs, min_sec=MIN_CLUSTER_SEC, cannot=None):
    """Fold clusters holding under `min_sec` of speech into the nearest survivor.


    The max-gap cut tends to shave off 3-5 second splinters — a questioner's one
    remark, a cough — and each becomes a spurious speaker. Absorbing them is
    safe in a way that raising the threshold is not: it can only merge clusters
    too small to enroll a profile, and never touches a real speaker's turns.
    Always leaves at least one cluster.

    `cannot` is not optional in spirit. constrained_linkage honours MOSS's "these
    two are different people" while building the tree, and this ran afterwards
    with no knowledge of it -- so a 5-second questioner could be folded straight
    into the person who answered them, on cosine similarity alone, discarding the
    categorical evidence that they are not the same speaker. Enrolment needs 10
    seconds; EXISTING as a speaker does not, and duration was being allowed to
    overrule MOSS. A splinter with nowhere legal to go now stays put.
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
        # Compact to 0..k-1 so the cluster-level relation lines up with `ids`.
        pos = {c: i for i, c in enumerate(ids)}
        compact = np.array([pos[c] for c in lab_core])
        CC = cluster_cannot_link(compact, cannot, len(ids))
        # "Is there anywhere legal for this one to go?" -- which is NOT
        # `not CC[row].all()`. CC's diagonal is False for any well-formed
        # cluster, so that row can never be all-True and the test never rejected
        # anything: a fully blocked splinter was always chosen as victim,
        # `others` came out empty, and the loop broke -- abandoning absorption
        # for every remaining splinter, including ones nothing forbade. That
        # invents speakers without violating the postcondition, so nothing
        # downstream would ever have noticed.
        victim = None
        for cand in sorted(small, key=lambda c: tot[c]):
            p = pos[cand]
            if any(pos[c] != p and not CC[p, pos[c]] for c in ids):
                victim = cand
                break
        if victim is None:
            break                              # every splinter is blocked; leave them
        cents = {}
        for c in ids:
            sel = core[lab_core == c]
            v = A[sel].mean(0)
            nv = np.linalg.norm(v)
            cents[c] = v / nv if nv > 0 else v
        others = [c for c in ids if c != victim and not CC[pos[victim], pos[c]]]
        if not others:
            break
        best = max(others, key=lambda c: float(cents[victim] @ cents[c]))
        lab_core[lab_core == victim] = best
    ids = sorted(set(lab_core.tolist()))
    remap = {c: i for i, c in enumerate(ids)}
    return np.array([remap[c] for c in lab_core]), len(ids)


def min_k_floor(keys, secs, durable=DURABLE_S, guard=10):
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


def choose_threshold(A, keys, secs, min_core=2.0, durable=DURABLE_S, guard=10):
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
            durable=DURABLE_S, guard=10, min_cluster_sec=MIN_CLUSTER_SEC):
    """Drop-in for link.py's cluster(), with the threshold chosen per meeting.

    Pass `thr` to force a fixed cut; leave it None to self-calibrate. Returns
    (lab, k, core, weak, S, D, Cf, info) — the same tuple link.py already
    unpacks, plus the info dict.

    Every knob is a PARAMETER here, never a module constant read from inside the
    body. Only `thr` varies per meeting today; the rest take a module default.
    They are in the signature because the right value is NOT the same for every
    recording -- measured, `durable` wants a low value on AMI close-talking and
    a high one on AliMeeting far-field, monotonically opposite, so no single
    number serves both. Keeping them here is what lets a per-meeting rule
    replace a default later without touching any call site.
    """
    secs = np.asarray(secs, dtype=float)
    core = core_set(secs, min_core)
    Ac = A[core]
    S = Ac @ Ac.T
    D = np.clip(1.0 - S, 0, 2)
    np.fill_diagonal(D, 0.0)

    # The constraints are built EITHER WAY. Pinning the cut by hand used to
    # switch them off as a side effect, so `--thr 0.4` quietly removed the one
    # mechanism that stops two people MOSS heard talking at once being merged.
    # The two are independent -- one chooses where to cut the tree, the other
    # says which merges were never allowed -- and the coupling cost real
    # accuracy: measured on AliMeeting far-field, a 0.30 cut scores 22.4% cpCER
    # with the constraints and 46.7% without.
    C = cannot_link_matrix([keys[i] for i in core], secs[core], durable, guard)
    if thr is None:
        thr, info = choose_threshold(A, keys, secs, min_core, durable, guard)
    else:
        info = {"threshold": round(float(thr), 4),
                "n_cannot_link": int(np.triu(C, 1).sum()),
                "floor": int(min_k_floor(keys, secs, durable, guard)),
                "self_calibrated": False, "mode": "fixed", "n_core": int(len(core)),
                "low_separation": False}

    _, labels_at = constrained_linkage(S, C)
    lab_core, k_raw = labels_at(thr)
    lab_core, k = absorb_small(lab_core, core, A, secs,
                               min_sec=min_cluster_sec, cannot=C)
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

    lab_core = refine_leave_one_out(Ac, lab_core, k, refine_iters, cannot=C)
    ids = sorted(set(lab_core.tolist()))
    lab_core = np.array([{c: i for i, c in enumerate(ids)}[c] for c in lab_core])
    k = len(ids)

    # THE POSTCONDITION, and it REPAIRS rather than raises. Everything above only
    # tries to honour the constraints; this is where the result is made to.
    #
    # It was previously left to floor_ok, which compares the speaker COUNT to
    # MOSS's floor -- a violation can keep the count intact by trading two people
    # for two others, and floor_ok calls that clean. Count is not the invariant.
    #
    # Raising was the first version and was wrong for this pipeline. A violation
    # is rare (measured: 1 pair in 216, on maybe a third of runs on one 7.94h
    # recording, moving with the embedder's own run-to-run variation), and it is
    # LOCAL -- two aggregates that should not share a cluster. Aborting threw
    # away 155 seconds of GPU work over it, took the whole meeting down, and left
    # the caller with nothing rather than with a transcript and a warning.
    #
    # Splitting the pair restores the invariant by construction: give one side a
    # cluster of its own. That is what the clustering should have done, it is
    # what a reviewer would do by hand, and it converges because each repair
    # strictly reduces the number of violating pairs.
    viol = np.argwhere(np.triu(C, 1) & (lab_core[:, None] == lab_core[None, :]))
    info["cannot_link_violations"] = int(len(viol))
    info["cannot_link_repaired"] = 0
    for _ in range(len(viol) + 1):
        viol = np.argwhere(np.triu(C, 1) & (lab_core[:, None] == lab_core[None, :]))
        if not len(viol):
            break
        a, b = viol[0]
        # Move the SHORTER of the two: the longer one has more claim on the
        # cluster's identity, and the centroid it leaves behind changes least.
        move = a if secs[core[a]] <= secs[core[b]] else b
        lab_core = lab_core.copy()
        lab_core[move] = lab_core.max() + 1
        info["cannot_link_repaired"] += 1
        print(f"CANNOT-LINK-REPAIR {keys[core[a]]} and {keys[core[b]]} shared a "
              f"cluster; moved {keys[core[move]]} to its own", flush=True)
    else:
        # Only reachable if a repair failed to reduce the count, which cannot
        # happen with the rule above -- but say so rather than return silently.
        raise AssertionError("could not separate every cannot-link pair")
    if info["cannot_link_repaired"]:
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

    # A weak aggregate used to be handed to argmax with no floor and no
    # constraint: whichever centroid was nearest took it, however far away that
    # was. Measured on a SCOTUS argument, a 0.81s interjection by one justice was
    # assigned to the advocate she was interrupting at cosine 0.255, beating the
    # runner-up by 0.018 -- and different-speaker pairs in this corpus run a
    # median of 0.104 and a p90 of 0.332, so 0.255 is inside the impostor
    # distribution. That is not an identification, it is the nearest point in a
    # cloud of noise. ACCEPT (0.55) and LINK_ACCEPT (0.75) both exist in this
    # project and neither reached this path.
    #
    # Two things now apply, and the first matters more.
    #
    # MOSS separating two locals INSIDE one window is a categorical judgment made
    # with the whole window audible. A weak aggregate is a fraction of a second of
    # embedding. Where they disagree the model wins: core keys already get this
    # through cannot_link_matrix, and weak keys skipped it entirely -- so the one
    # place the model's judgment is most needed, because the embedding is least
    # trustworthy, was the one place it was ignored. In that SCOTUS window MOSS
    # had it right, labelling the interjection S01 against the advocate's S02.
    #
    # And below WEAK_MIN nobody is claimed at all. -1 is the linker's leftover
    # bucket, which is honest about not knowing; a wrong name is not.
    taken = {}                      # cluster -> {(window, local)} already in it
    for i in core:
        taken.setdefault(int(lab[i]), set()).add(keys[i])
    for i in weak:
        if np.linalg.norm(A[i]) <= 0:
            lab[i] = -1
            continue
        w_i, loc_i = keys[i]
        sim = A[i] @ Cm.T
        lab[i] = -1
        for c in np.argsort(sim)[::-1]:
            c = int(c)
            if any(kw == w_i and kl != loc_i for kw, kl in taken.get(c, ())):
                continue            # MOSS heard a DIFFERENT speaker here
            if sim[c] >= WEAK_MIN:
                lab[i] = c
            break

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
