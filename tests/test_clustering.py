"""pipeline/cluster_speakers.py -- constrained clustering, and what "constrained"
has to mean by the time the labels leave the function.

Two commits set the agenda.

6506bd8 "Honour cannot-link after the cut, not only while building the tree".
constrained_linkage carried MOSS's "these two are different people" through
every merge -- and then absorb_small folded any cluster under 10s into its
nearest neighbour on cosine alone, and refine_leave_one_out moved individual
turns to whichever centroid scored highest, neither having heard of the
relation. The invariant being maintained was therefore "cannot-links are
honoured while constructing the tree", which is not the claim worth anything.
Nothing caught it because the only check was floor_ok, a comparison of the
speaker COUNT against MOSS's floor: 6 violated pairs in the 7.94h recording and
2 in the 38-minute one, with k identical before and after the fix. So the tests
below assert the POSTCONDITION -- no forbidden pair shares a cluster -- stage by
stage, never a count.

2851014 "Stop rescanning every pair in Python on every merge" replaced
constrained_linkage's O(n^3)-interpreted inner scan with a live similarity
matrix and one Lance-Williams row update per merge, claiming identical output.
The old implementation is gone, so the invariants are pinned instead: merge
count, non-increasing heights, labels_at monotone in the threshold, the
documented first-in-(i,j)-order tie-break, and agreement with an independently
written naive linkage in conftest.

Four of these were written as strict xfails against defects that were live when
the suite was written -- a stale cannot-link table inside a refine sweep, an
int8 accumulator that wrapped, a guard that could never reject, and the
postcondition they all reached. All four are fixed; the tests now pin them shut
and the comment above each records what the defect was.
"""
import numpy as np
import pytest

from conftest import (DIM, axis_vec, brute_cluster_cannot_link, is_coarsening,
                      naive_constrained_linkage, partition_of, random_cannot,
                      random_meeting, random_similarity, speaker_vec,
                      violating_pairs)

# Speaker directions used by the hand-built meetings. Distinct axes are
# orthogonal, so "A and C look nothing alike" is exact.
AX_A, AX_C = 0, 1


# ==========================================================================
# the meeting the headline test runs on
# ==========================================================================

def blocked_splinter_meeting():
    """Two speakers over two windows plus a 6-second questioner MOSS says is
    neither of them. -> (A, secs, keys)

      idx  key           speaker  secs  role
       0   (0, 'S01')    A         40   A's turn in window 0
       1   (0, 'S02')    C         40   C's turn in window 0
       2   (1, 'S01')    A         40   A's turn in window 1
       3   (1, 'S02')    C         40   C's turn in window 1
       4   (1, 'S03')    B          6   THE SPLINTER

    The splinter is the whole point. It is under MIN_CLUSTER_SEC (10s) so
    absorb_small wants to fold it away; it shares window 1 with both A and C so
    MOSS cannot-links it to both; and it sits at cosine 0.95 to every one of A's
    aggregates -- higher than any threshold this meeting can produce -- so
    absorption on cosine alone folds it straight into the person it was
    talking to. That is exactly the "5-second questioner merged into whoever
    answered them" case 6506bd8 describes.

    A's two aggregates sit at 0.99 to each other, C's at 0.98, and A and C are
    orthogonal, so the merge order is fixed and does not depend on a draw.
    """
    a0 = speaker_vec(AX_A, 2, 0.99)
    a1 = speaker_vec(AX_A, 3, 0.99)
    c0 = speaker_vec(AX_C, 4, 0.98)
    c1 = speaker_vec(AX_C, 5, 0.98)
    # cos(b, a_i) = p * sqrt(0.99) = 0.95, and b shares no axis with C at all.
    p = 0.95 / np.sqrt(0.99)
    b = axis_vec({AX_A: p, 6: np.sqrt(1.0 - p * p)}, DIM)

    A = np.vstack([a0, c0, a1, c1, b])
    secs = np.array([40.0, 40.0, 40.0, 40.0, 6.0])
    keys = [(0, "S01"), (0, "S02"), (1, "S01"), (1, "S02"), (1, "S03")]
    return A, secs, keys


SPLINTER = 4          # index of the 6s aggregate in blocked_splinter_meeting()
A_TURNS = (0, 2)
C_TURNS = (1, 3)


def cut_labels(cs, A, secs, keys):
    """Everything cluster() does up to the cut, so a test can start after it.

    -> (core, C, thr, lab_after_cut)
    """
    core = cs.core_set(secs, 2.0)
    Ac = A[core]
    thr, _info = cs.choose_threshold(A, keys, secs)
    C = cs.cannot_link_matrix([keys[i] for i in core], np.asarray(secs)[core],
                              1.0, 10)
    _heights, labels_at = cs.constrained_linkage(Ac @ Ac.T, C)
    lab, _k = labels_at(thr)
    return core, C, thr, lab


# ==========================================================================
# 1. THE POSTCONDITION
# ==========================================================================
class TestPostcondition:
    """cluster() must not hand back a forbidden pair in one cluster."""

    def test_cannot_linked_splinter_is_never_absorbed_into_its_partner(self, cs):
        """The headline. Cosine wants the merge; MOSS forbids it; MOSS wins.

        This is the shape 6506bd8 was about: a short turn that the tree kept
        apart, that absorb_small then folded into the very cluster MOSS said it
        was not, on similarity alone, without changing the speaker count.
        """
        A, secs, keys = blocked_splinter_meeting()
        lab, k, core, weak, S, D, Cf, info = cs.cluster(A, secs, keys)

        # The premise: MOSS asserted four forbidden pairs, and the splinter is
        # both too short to survive absorption and too similar to A to be
        # separated by cosine.
        assert info["n_cannot_link"] == 4
        assert secs[SPLINTER] < cs.MIN_CLUSTER_SEC
        for i in A_TURNS:
            assert float(A[SPLINTER] @ A[i]) == pytest.approx(0.95)
            assert float(A[SPLINTER] @ A[i]) > info["threshold"]

        # The postcondition, as cluster() reports it and as recomputed here.
        assert info["cannot_link_violations"] == 0
        C = cs.cannot_link_matrix([keys[i] for i in core], secs[core], 1.0, 10)
        assert len(violating_pairs(lab[core], C)) == 0

        # ...and what that means for this meeting: the splinter is not with
        # either of the aggregates it is forbidden to join, and since it had
        # nowhere legal to go it survived as a speaker of its own.
        for i in A_TURNS[1:] + C_TURNS[1:]:      # its window-1 co-speakers
            assert lab[SPLINTER] != lab[i]
        assert list(lab).count(lab[SPLINTER]) == 1
        assert k == 3
        assert info["floor"] == 3 and info["floor_ok"]

    def test_cosine_alone_would_have_merged_them(self, cs):
        """The negative control: without the relation, the bug reappears.

        absorb_small with cannot=None -- which is what it was before 6506bd8 --
        folds the splinter into A on this very meeting. Without this the
        headline test could pass for the boring reason that nothing wanted to
        merge anything.
        """
        A, secs, keys = blocked_splinter_meeting()
        core, C, _thr, lab_cut = cut_labels(cs, A, secs, keys)
        assert len(violating_pairs(lab_cut, C)) == 0      # the tree was clean

        unconstrained, _k = cs.absorb_small(lab_cut, core, A, secs)
        assert unconstrained[SPLINTER] == unconstrained[A_TURNS[0]]
        assert len(violating_pairs(unconstrained, C)) == 1

        constrained, _k = cs.absorb_small(lab_cut, core, A, secs, cannot=C)
        assert len(violating_pairs(constrained, C)) == 0

    def test_postcondition_raises_rather_than_returning_a_bad_labelling(self, cs,
                                                                        monkeypatch):
        """The check itself is load-bearing, so break a stage and watch it fire.

        6506bd8's point is that a violation need not change k, so no downstream
        consumer would notice. Replacing absorb_small with one that collapses
        everything is the cheapest way to prove the guard is not vacuous.
        """
        A, secs, keys = blocked_splinter_meeting()

        def collapse(lab_core, core, Amat, s, min_sec=None, cannot=None):
            return np.zeros(len(lab_core), dtype=int), 1

        monkeypatch.setattr(cs, "absorb_small", collapse)
        with pytest.raises(AssertionError) as e:
            cs.cluster(A, secs, keys)
        assert "cannot-link pair(s) ended up in one cluster" in str(e.value)

    def test_cut_and_absorb_never_violate_on_random_meetings(self, cs):
        """Everything up to and including absorb_small honours the relation.

        200 synthetic meetings in link.py's aggregate() shape, each with three
        speakers in every window and therefore a full set of cannot-links.
        Pairs with the xfail below, which runs the same meetings through
        refine_leave_one_out and does not stay clean -- so the blame for the
        remaining violations lands on one named stage rather than on
        "clustering".
        """
        for seed in range(200):
            A, secs, keys = random_meeting(seed)
            core, C, _thr, lab_cut = cut_labels(cs, A, secs, keys)
            assert len(violating_pairs(lab_cut, C)) == 0, "cut, seed %d" % seed
            lab_abs, k = cs.absorb_small(lab_cut, core, A, secs, cannot=C)
            assert len(violating_pairs(lab_abs, C)) == 0, "absorb, seed %d" % seed

    # FIXED (was xfail; the defect this pins is closed):

    # refine_leave_one_out computes `blocked` ONCE per sweep but applies moves

    # sequentially within it, so the table is stale the moment a turn moves. A

    # forbidden partner that leaves cluster X for cluster Y during the sweep is still

    # recorded as being in X, and a later turn is allowed into Y and lands on top of

    # it. FIX: update `blocked` as part of the same commit that updates sums/cnt/lab

    # (row c and row b for every turn cannot-linked with i), or re-test the chosen

    # target against the current lab before committing the move. cluster() then raises

    # AssertionError, so on real recordings this is a crash, not silent corruption.
    def test_cluster_holds_the_postcondition_on_random_meetings(self, cs):
        """The same 200 meetings, end to end.

        8 of them raise at HEAD (seeds 52, 103, 127, 149, 155, 157, 160, 188),
        every one of them at refine_leave_one_out -- so this is not a synthetic
        worry about a function's contract, it is link.py dying on 4% of
        meetings whose windows carry a full set of cannot-links.
        """
        bad = []
        for seed in range(200):
            A, secs, keys = random_meeting(seed)
            try:
                cs.cluster(A, secs, keys)
            except AssertionError:
                bad.append(seed)
        assert not bad, "cluster() violated its own postcondition on seeds %r" % bad


# ==========================================================================
# 2. absorb_small
# ==========================================================================
class TestAbsorbSmall:

    @staticmethod
    def three_cluster_fixture():
        """Two full speakers and one splinter, with the splinter's similarity
        ordering fixed: nearest to cluster 0, second nearest to cluster 1.

        -> (A, secs, core, lab)
        """
        A = np.vstack([
            axis_vec({0: 1.0}),                  # 0  cluster 0
            axis_vec({0: 1.0, 5: 0.05}),         # 1  cluster 0
            axis_vec({1: 1.0}),                  # 2  cluster 1
            axis_vec({1: 1.0, 6: 0.05}),         # 3  cluster 1
            axis_vec({0: 0.90, 1: 0.30, 7: 0.3}),  # 4  splinter, nearer 0 than 1
        ])
        secs = np.array([40.0, 40.0, 40.0, 40.0, 4.0])
        return A, secs, np.arange(5), np.array([0, 0, 1, 1, 2])

    def test_refuses_an_illegal_absorption_and_takes_the_legal_one(self, cs):
        """Blocked from its nearest neighbour, the splinter goes to the runner-up.

        Not merely "it did not go into cluster 0": the point of carrying the
        relation into this stage is that absorption still happens, just legally.
        """
        A, secs, core, lab = self.three_cluster_fixture()
        assert float(A[4] @ A[0]) > float(A[4] @ A[2])      # cosine prefers 0

        free, _k = cs.absorb_small(lab, core, A, secs)
        assert free[4] == free[0]

        cannot = np.zeros((5, 5), dtype=bool)
        for i in (0, 1):
            cannot[4, i] = cannot[i, 4] = True
        out, k = cs.absorb_small(lab, core, A, secs, cannot=cannot)
        assert out[4] == out[2] and out[4] != out[0]
        assert k == 2
        assert len(violating_pairs(out, cannot)) == 0

    def test_splinter_with_nowhere_legal_to_go_survives_as_its_own_cluster(self, cs):
        """"Duration no longer overrules MOSS." Six seconds is still a speaker."""
        A, secs, core, lab = self.three_cluster_fixture()
        cannot = np.zeros((5, 5), dtype=bool)
        for i in range(4):
            cannot[4, i] = cannot[i, 4] = True

        out, k = cs.absorb_small(lab, core, A, secs, cannot=cannot)
        assert k == 3
        assert list(out).count(out[4]) == 1
        assert len(violating_pairs(out, cannot)) == 0

    def test_absorbs_splinters_the_relation_is_silent_about(self, cs):
        """The guard must not become "never absorb anything"."""
        A, secs, core, lab = self.three_cluster_fixture()
        cannot = np.zeros((5, 5), dtype=bool)
        cannot[0, 2] = cannot[2, 0] = True      # about the two big clusters only
        out, k = cs.absorb_small(lab, core, A, secs, cannot=cannot)
        assert k == 2 and out[4] == out[0]

    def test_relabels_contiguously_from_zero(self, cs):
        A, secs, core, _lab = self.three_cluster_fixture()
        lab = np.array([3, 3, 7, 7, 9])         # non-contiguous input ids
        out, k = cs.absorb_small(lab, core, A, secs)
        assert sorted(set(out.tolist())) == list(range(k))

    def test_never_returns_fewer_than_one_cluster(self, cs):
        """A meeting that is all splinter is still a meeting with a speaker."""
        A = np.vstack([axis_vec({0: 1.0}), axis_vec({0: 1.0, 5: 0.1})])
        secs = np.array([1.0, 1.0])
        out, k = cs.absorb_small(np.array([0, 1]), np.arange(2), A, secs)
        assert k == 1 and set(out.tolist()) == {0}

    def test_terminates_when_every_splinter_is_mutually_blocked(self, cs):
        """All-small, all-forbidden: the loop must stop, not spin or force a merge."""
        A = np.vstack([axis_vec({0: 1.0, i: 0.2}) for i in range(2, 6)])
        secs = np.full(4, 3.0)
        cannot = ~np.eye(4, dtype=bool)
        out, k = cs.absorb_small(np.array([0, 1, 2, 3]), np.arange(4), A, secs,
                                 cannot=cannot)
        assert k == 4
        assert len(violating_pairs(out, cannot)) == 0

    # FIXED (was xfail; the defect this pins is closed):

    # The `for cand in sorted(small, ...)` loop is meant to skip a splinter with

    # nowhere legal to go and try the next one, but its guard `if not

    # CC[pos[cand]].all()` can never reject a candidate: CC's DIAGONAL is False for any

    # well-formed cluster (a cluster holding no forbidden pair of its own), so .all()

    # is False even when every OTHER entry is True. The blocked splinter is therefore

    # always chosen as victim, `others` comes out empty, and `break` abandons

    # absorption for every remaining splinter -- including ones with a perfectly legal

    # home. Result: a spurious extra speaker. FIX: test `any(c != cand and not

    # CC[pos[cand], pos[c]] for c in ids)` instead of `not CC[pos[cand]].all()`, and

    # carry a set of already-rejected victims so the loop makes progress rather than

    # reconsidering the same splinter forever.
    def test_a_blocked_splinter_does_not_strand_the_other_splinters(self, cs):
        """X is blocked everywhere; Y is 5s and has a legal home in cluster 0.

        X staying put is correct and is what the commit promises. Y staying put
        is not: nothing forbids Y from joining cluster 0, and leaving it out
        invents a speaker.
        """
        A = np.vstack([
            axis_vec({0: 1.0}),                    # 0  cluster 0
            axis_vec({0: 1.0, 5: 0.05}),           # 1  cluster 0
            axis_vec({1: 1.0}),                    # 2  cluster 1
            axis_vec({1: 1.0, 6: 0.05}),           # 3  cluster 1
            axis_vec({0: 0.9, 7: 0.4}),            # 4  X, 4s, blocked everywhere
            axis_vec({0: 0.8, 8: 0.6}),            # 5  Y, 5s, legal in cluster 0
        ])
        secs = np.array([40.0, 40.0, 40.0, 40.0, 4.0, 5.0])
        lab = np.array([0, 0, 1, 1, 2, 3])
        cannot = np.zeros((6, 6), dtype=bool)
        for i in (0, 1, 2, 3, 5):
            cannot[4, i] = cannot[i, 4] = True

        out, k = cs.absorb_small(lab, np.arange(6), A, secs, cannot=cannot)
        assert len(violating_pairs(out, cannot)) == 0    # holds at HEAD
        assert list(out).count(out[4]) == 1              # X stays, correctly
        assert out[5] == out[0], "Y had a legal home and should have been absorbed"
        assert k == 3


# ==========================================================================
# 3. refine_leave_one_out
# ==========================================================================
class TestRefine:

    @staticmethod
    def wants_to_move_fixture():
        """Turn 0 sits in cluster 0 but matches cluster 1's centroid far better.

        -> (Ac, lab, k). Turn 2 is in cluster 1 and is the forbidden partner.
        """
        Ac = np.vstack([
            axis_vec({0: 0.30, 1: 0.954}),      # 0  in cluster 0, leans cluster 1
            axis_vec({0: 1.0}),                 # 1  cluster 0 anchor
            axis_vec({1: 1.0}),                 # 2  cluster 1
            axis_vec({1: 1.0, 4: 0.05}),        # 3  cluster 1
            axis_vec({0: 1.0, 5: 0.05}),        # 4  cluster 0 anchor
        ])
        return Ac, np.array([0, 0, 1, 1, 0]), 2

    def test_moves_a_mismatched_turn_when_nothing_forbids_it(self, cs):
        """The premise: refinement really does want to move turn 0."""
        Ac, lab, k = self.wants_to_move_fixture()
        out = cs.refine_leave_one_out(Ac, lab, k, iters=3)
        assert out[0] == out[2]

    def test_never_moves_a_turn_into_a_cluster_holding_a_forbidden_partner(self, cs):
        """Same fixture, same pull, one cannot-link -- the move must not happen.

        "Unreachable, not merely unlikely": the target cluster is skipped
        outright, so the turn stays where it is rather than being ranked
        second-best into it.
        """
        Ac, lab, k = self.wants_to_move_fixture()
        cannot = np.zeros((5, 5), dtype=bool)
        cannot[0, 2] = cannot[2, 0] = True
        out = cs.refine_leave_one_out(Ac, lab, k, iters=3, cannot=cannot)
        assert out[0] != out[2]
        assert list(out) == [0, 0, 1, 1, 0]
        assert len(violating_pairs(out, cannot)) == 0

    def test_a_turn_may_still_move_into_a_cluster_that_is_merely_nearby(self, cs):
        """Blocking is per-partner, not per-cluster-index: an unrelated
        cannot-link elsewhere in the meeting must not freeze the sweep."""
        Ac, lab, k = self.wants_to_move_fixture()
        cannot = np.zeros((5, 5), dtype=bool)
        cannot[1, 4] = cannot[4, 1] = True      # two turns already sharing cluster 0
        out = cs.refine_leave_one_out(Ac, lab, k, iters=3, cannot=cannot)
        assert out[0] == out[2]                 # unrelated block, move still allowed

    def test_a_lone_turn_is_not_forced_out_of_its_own_cluster(self, cs):
        """cnt[c] <= 1 is a guard, not an optimisation.

        A speaker who talks once has a cluster that is entirely themselves;
        scoring them against a centroid they are not in would always lose, and
        the podcast MC kept being swallowed by the host that way.
        """
        Ac = np.vstack([
            axis_vec({0: 1.0}),
            axis_vec({0: 1.0, 4: 0.05}),
            axis_vec({0: 0.99, 5: 0.1}),        # 2 is alone but looks like the rest
        ])
        out = cs.refine_leave_one_out(Ac, np.array([0, 0, 1]), 2, iters=3)
        assert out[2] == 1 and list(out).count(1) == 1

    def test_converges_and_returns_labels_in_range(self, cs):
        """Sequential updates must still terminate; iters is a cap, not a plan."""
        rng = np.random.default_rng(7)
        for seed in range(30):
            n, k = 12, 4
            X = rng.normal(size=(n, 24))
            Ac = X / np.linalg.norm(X, axis=1, keepdims=True)
            lab = rng.integers(0, k, size=n)
            out = cs.refine_leave_one_out(Ac, lab, k, iters=5)
            assert out.shape == (n,)
            assert out.min() >= 0 and out.max() < k

    # FIXED (was xfail; the defect this pins is closed):

    # `blocked` is built once at the top of each sweep from the labels as they stand

    # then, but the sweep commits moves as it goes -- the docstring's own reason for

    # not caching it across sweeps applies just as much WITHIN one. Here turn 0 leaves

    # cluster 2 for cluster 1 early in the sweep; by the time turn 4 is considered,

    # blocked[4] still says turn 0's forbidden partner is in cluster 2, so turn 4 is

    # allowed into cluster 1 and lands on it. FIX: update `blocked` when a move is

    # committed (rows for every turn cannot-linked with the mover), or re-check the

    # chosen target against the current lab before committing. This is the stage that

    # still breaks cluster()'s postcondition -- see

    # TestPostcondition.test_cluster_holds_the_postcondition_on_random_ meetings.
    def test_blocked_table_does_not_go_stale_within_a_sweep(self, cs):
        Ac = np.vstack([
            axis_vec({1: 0.95, 3: 0.31}),       # 0  cluster 2, leans cluster 1
            axis_vec({0: 1.0}),                 # 1  cluster 0 anchor
            axis_vec({1: 1.0}),                 # 2  cluster 1
            axis_vec({1: 0.98, 4: 0.2}),        # 3  cluster 1
            axis_vec({0: 0.55, 1: 0.84}),       # 4  cluster 0, leans cluster 1
            axis_vec({2: 1.0}),                 # 5  cluster 2 anchor
        ])
        lab = np.array([2, 0, 1, 1, 0, 2])
        cannot = np.zeros((6, 6), dtype=bool)
        cannot[0, 4] = cannot[4, 0] = True

        out = cs.refine_leave_one_out(Ac, lab, 3, iters=3, cannot=cannot)
        assert len(violating_pairs(out, cannot)) == 0, (
            "turns 0 and 4 are cannot-linked and both ended in cluster %d"
            % out[0])


# ==========================================================================
# 4. constrained_linkage
# ==========================================================================
class TestConstrainedLinkage:

    def test_blocked_pairs_never_merge(self, cs):
        S = np.array([
            [1.00, 0.98, 0.90],
            [0.98, 1.00, 0.95],
            [0.90, 0.95, 1.00],
        ])
        cannot = np.zeros((3, 3), dtype=bool)
        cannot[0, 2] = cannot[2, 0] = True
        heights, labels_at = cs.constrained_linkage(S, cannot)
        for thr in (-1.0, 0.0, 0.5, 0.9, 0.96, 0.99, 1.5):
            lab, _k = labels_at(thr)
            assert len(violating_pairs(lab, cannot)) == 0, "thr=%r" % thr

    def test_a_merged_node_inherits_its_children_blocks(self, cs):
        """0 and 2 are free to merge; 1 and 2 are not. Merging 0 with 1 first
        must make the resulting node unmergeable with 2, even though nothing
        ever forbade 0 and 2 directly."""
        S = np.array([
            [1.00, 0.98, 0.90],
            [0.98, 1.00, 0.95],
            [0.90, 0.95, 1.00],
        ])
        cannot = np.zeros((3, 3), dtype=bool)
        cannot[1, 2] = cannot[2, 1] = True
        heights, labels_at = cs.constrained_linkage(S, cannot)

        assert list(heights) == [0.98]           # 0+1, then nothing legal is left
        lab, k = labels_at(-np.inf)
        assert k == 2
        assert lab[0] == lab[1] and lab[2] != lab[0]

    def test_single_node(self, cs):
        heights, labels_at = cs.constrained_linkage(np.ones((1, 1)),
                                                    np.zeros((1, 1), dtype=bool))
        assert len(heights) == 0
        lab, k = labels_at(-np.inf)
        assert k == 1 and list(lab) == [0]

    def test_two_nodes_blocked(self, cs):
        S = np.array([[1.0, 0.99], [0.99, 1.0]])
        cannot = np.array([[False, True], [True, False]])
        heights, labels_at = cs.constrained_linkage(S, cannot)
        assert len(heights) == 0
        lab, k = labels_at(-np.inf)
        assert k == 2 and lab[0] != lab[1]

    def test_two_nodes_free(self, cs):
        S = np.array([[1.0, 0.99], [0.99, 1.0]])
        heights, labels_at = cs.constrained_linkage(S, np.zeros((2, 2), dtype=bool))
        assert list(heights) == [0.99]
        assert labels_at(-np.inf)[1] == 1
        assert labels_at(0.999)[1] == 2

    def test_every_pair_blocked(self, cs):
        n = 5
        S = np.full((n, n), 0.99)
        np.fill_diagonal(S, 1.0)
        cannot = ~np.eye(n, dtype=bool)
        heights, labels_at = cs.constrained_linkage(S, cannot)
        assert len(heights) == 0
        lab, k = labels_at(-np.inf)
        assert k == n and sorted(lab.tolist()) == list(range(n))

    def test_identical_vectors(self, cs):
        """Every similarity is exactly 1.0: no gap, no strict ordering, and a
        naive argmax loop that never advanced would hang here."""
        n = 6
        S = np.ones((n, n))
        heights, labels_at = cs.constrained_linkage(S, np.zeros((n, n), dtype=bool))
        assert len(heights) == n - 1
        assert list(heights) == [1.0] * (n - 1)
        assert labels_at(0.5)[1] == 1
        assert labels_at(1.5)[1] == n

    def test_labels_are_contiguous_from_zero(self, cs):
        rng = np.random.default_rng(3)
        S = random_similarity(9, rng)
        cannot = random_cannot(9, rng, p=0.2)
        _heights, labels_at = cs.constrained_linkage(S, cannot)
        for thr in np.linspace(-1.0, 1.0, 21):
            lab, k = labels_at(float(thr))
            assert sorted(set(lab.tolist())) == list(range(k))

    def test_tie_break_is_the_first_maximum_in_ij_order(self, cs):
        """2851014 promises "the first maximum in (i,j) order either way", which
        is the only thing keeping a rewrite's output identical under ties."""
        S = np.array([
            [1.0, 0.9, 0.9, 0.1],
            [0.9, 1.0, 0.2, 0.1],
            [0.9, 0.2, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ])
        cannot = np.zeros((4, 4), dtype=bool)
        heights, labels_at = cs.constrained_linkage(S, cannot)
        ref_heights, ref_order = naive_constrained_linkage(S, cannot)
        assert ref_order[0] == (0, 1)
        assert list(heights) == ref_heights


# ==========================================================================
# 5. cluster_cannot_link
# ==========================================================================
class TestClusterCannotLink:

    def test_lifts_the_pairwise_relation_onto_clusters(self, cs):
        lab = np.array([0, 0, 1, 1, 2])
        cannot = np.zeros((5, 5), dtype=bool)
        cannot[1, 2] = cannot[2, 1] = True      # a member of 0 vs a member of 1
        CC = cs.cluster_cannot_link(lab, cannot, 3)
        assert CC[0, 1] and CC[1, 0]
        assert not CC[0, 2] and not CC[1, 2]
        assert not CC.diagonal().any()

    def test_diagonal_is_true_when_a_cluster_holds_a_forbidden_pair(self, cs):
        """The self-entry is not decoration: it is how "this cluster is already
        illegal" is expressed, and absorb_small's victim guard reads it."""
        lab = np.array([0, 0, 1])
        cannot = np.zeros((3, 3), dtype=bool)
        cannot[0, 1] = cannot[1, 0] = True
        CC = cs.cluster_cannot_link(lab, cannot, 2)
        assert CC[0, 0] and not CC[1, 1]

    def test_no_relation_gives_an_all_false_matrix(self, cs):
        lab = np.array([0, 1, 1])
        for empty in (None, np.zeros((0, 0), dtype=bool)):
            CC = cs.cluster_cannot_link(lab, empty, 2)
            assert CC.shape == (2, 2) and not CC.any()

    def test_unused_cluster_ids_are_all_false(self, cs):
        """k may exceed the labels in use -- absorb_small compacts first, but
        refine_leave_one_out passes the k it was given."""
        lab = np.array([0, 0, 1, 1])
        cannot = np.zeros((4, 4), dtype=bool)
        cannot[0, 3] = cannot[3, 0] = True
        CC = cs.cluster_cannot_link(lab, cannot, 4)
        assert CC.shape == (4, 4)
        assert CC[0, 1] and CC[1, 0]
        assert not CC[2].any() and not CC[3].any()

    def test_matches_a_brute_force_lift_over_random_inputs(self, cs):
        """Sizes are kept small on purpose: see the xfail below for what happens
        once a cluster pair accumulates more than 127 forbidden member pairs."""
        rng = np.random.default_rng(11)
        for _ in range(200):
            n = int(rng.integers(2, 11))
            k = int(rng.integers(1, 5))
            lab = rng.integers(0, k, size=n)
            cannot = random_cannot(n, rng, p=float(rng.uniform(0.05, 0.6)))
            got = cs.cluster_cannot_link(lab, cannot, k)
            want = brute_cluster_cannot_link(lab, cannot, k)
            assert got.shape == (k, k)
            assert np.array_equal(got, want)

    def test_is_symmetric_for_a_symmetric_relation(self, cs):
        rng = np.random.default_rng(12)
        for _ in range(50):
            n = int(rng.integers(2, 11))
            k = int(rng.integers(1, 4))
            lab = rng.integers(0, k, size=n)
            CC = cs.cluster_cannot_link(lab, random_cannot(n, rng), k)
            assert np.array_equal(CC, CC.T)

    # FIXED (was xfail; the defect this pins is closed):

    # `(M @ cannot.astype(np.int8) @ M.T) > 0` accumulates the member-pair COUNT in

    # int8: numpy promotes bool @ int8 to int8, so a cluster pair with 256 forbidden

    # member pairs wraps to 0 and the block silently disappears (128..255 wrap

    # negative, which is just as false). Two 16-member clusters that MOSS fully

    # separated are enough. This is reachable: the 7.94h recording clusters 1,163

    # aggregates into 23 speakers, so a cluster pair holding >127 forbidden member

    # pairs is ordinary, and absorb_small then folds a splinter into a cluster MOSS

    # forbade -- caught only by cluster()'s postcondition, as a crash. FIX: widen the

    # accumulator (`cannot.astype(np.int64)`), or keep it boolean with

    # `np.einsum('cn,nm,dm->cd', M, cannot, M, optimize=True) > 0` on a wide dtype.

    # refine_leave_one_out's `cannot.astype(np.int8) @ M.T` has the same flaw, though

    # cannot_link_matrix's guard=10 keeps a single turn under 127 forbidden partners in

    # practice.
    def test_does_not_overflow_when_many_member_pairs_are_forbidden(self, cs):
        m = 16
        n = 2 * m
        lab = np.array([0] * m + [1] * m)
        cannot = np.zeros((n, n), dtype=bool)
        cannot[:m, m:] = True
        cannot[m:, :m] = True
        assert int(cannot.sum() // 2) == m * m == 256

        CC = cs.cluster_cannot_link(lab, cannot, 2)
        assert np.array_equal(CC, brute_cluster_cannot_link(lab, cannot, 2))
        assert CC[0, 1] and CC[1, 0]


# ==========================================================================
# 6. the vectorised linkage of 2851014 -- invariants, since the old code is gone
# ==========================================================================
SIZES = [1, 2, 3, 5, 9, 17, 33, 64]


def linkage_cases():
    """(label, S, cannot) over a few sizes, with and without constraints."""
    cases = []
    for n in SIZES:
        for seed in (0, 1, 2):
            rng = np.random.default_rng(1000 * n + seed)
            S = random_similarity(n, rng)
            for p, tag in ((0.0, "free"), (0.15, "sparse"), (0.45, "dense")):
                cannot = (np.zeros((n, n), dtype=bool) if p == 0.0
                          else random_cannot(n, rng, p=p))
                cases.append(pytest.param(S, cannot,
                                          id="n%d-s%d-%s" % (n, seed, tag)))
    return cases


CASES = linkage_cases()


class TestLinkageInvariants:
    """2851014 claims "same merges, same heights to the bit, same labels at
    every threshold" for a rewrite whose predecessor no longer exists. What can
    still be pinned is what the claim reduces to."""

    @pytest.mark.parametrize("S,cannot", CASES)
    def test_heights_are_non_increasing(self, cs, S, cannot):
        """Greedy average linkage has no inversions -- a merged node's
        similarity to anything is a weighted average of two values that already
        lost to the merge it just made. labels_at() DEPENDS on this: it stops at
        the first height below the threshold, so one inversion would silently
        drop every later merge."""
        heights, _labels_at = cs.constrained_linkage(S, cannot)
        assert np.all(np.diff(heights) <= 1e-12), heights

    @pytest.mark.parametrize("S,cannot", CASES)
    def test_merge_count_accounts_for_every_node(self, cs, S, cannot):
        """One merge removes exactly one node, and the run stops only when
        every surviving pair is blocked."""
        n = len(S)
        heights, labels_at = cs.constrained_linkage(S, cannot)
        _lab, k_min = labels_at(-np.inf)
        assert len(heights) == n - k_min
        assert len(heights) <= n - 1

    @pytest.mark.parametrize("S,cannot", CASES)
    def test_labels_at_is_monotone_in_the_threshold(self, cs, S, cannot):
        """Lowering the cut may only ever merge clusters, never split them."""
        heights, labels_at = cs.constrained_linkage(S, cannot)
        lo = float(min(heights)) - 0.1 if len(heights) else -1.0
        hi = float(max(heights)) + 0.1 if len(heights) else 1.0
        thrs = list(np.linspace(hi, lo, 25))
        prev_lab, prev_k = labels_at(thrs[0])
        for thr in thrs[1:]:
            lab, k = labels_at(float(thr))
            assert k <= prev_k, "k rose from %d to %d at thr=%r" % (prev_k, k, thr)
            assert is_coarsening(partition_of(lab), partition_of(prev_lab)), thr
            prev_lab, prev_k = lab, k

    @pytest.mark.parametrize("S,cannot", CASES)
    def test_no_blocked_pair_shares_a_cluster_at_any_threshold(self, cs, S, cannot):
        heights, labels_at = cs.constrained_linkage(S, cannot)
        for thr in np.linspace(-1.1, 1.1, 25):
            lab, _k = labels_at(float(thr))
            assert len(violating_pairs(lab, cannot)) == 0, "thr=%r" % thr

    @pytest.mark.parametrize("S,cannot", CASES)
    def test_matches_the_naive_linkage_merge_for_merge(self, cs, S, cannot):
        """The strongest form of "same algorithm, same greedy choice, same
        tie-break": an independently written O(n^3) scan picks the same pair in
        the same order at the same heights. Exact equality, not allclose --
        both do the same IEEE operations in the same order, and 2851014's claim
        was "identical", not "close"."""
        heights, _labels_at = cs.constrained_linkage(S, cannot)
        ref_heights, _ref_order = naive_constrained_linkage(S, cannot)
        assert list(heights) == ref_heights

    @pytest.mark.parametrize("S,cannot", CASES)
    def test_inputs_are_not_mutated(self, cs, S, cannot):
        """S and cannot are reused by the caller: cluster() links once for the
        threshold and again for the labels, and passes the same C to
        absorb_small and refine_leave_one_out afterwards."""
        S_before, cannot_before = S.copy(), cannot.copy()
        cs.constrained_linkage(S, cannot)
        assert np.array_equal(S, S_before)
        assert np.array_equal(cannot, cannot_before)

    def test_linking_twice_gives_the_same_answer(self, cs):
        """cluster() runs constrained_linkage on the same S and C twice --
        once inside choose_threshold and once for the labels -- and would
        report a different speaker count than it calibrated for if the two
        passes ever disagreed."""
        A, secs, keys = random_meeting(3)
        core = cs.core_set(secs, 2.0)
        Ac = A[core]
        S = Ac @ Ac.T
        C = cs.cannot_link_matrix([keys[i] for i in core], secs[core], 1.0, 10)
        h1, la1 = cs.constrained_linkage(S, C)
        h2, la2 = cs.constrained_linkage(S, C)
        assert np.array_equal(h1, h2)
        for thr in np.linspace(-0.5, 1.0, 16):
            assert np.array_equal(la1(float(thr))[0], la2(float(thr))[0])


# ==========================================================================
# supporting: where the relation comes from
# ==========================================================================
class TestCannotLinkMatrix:
    """cannot_link_matrix() is the only source of the relation the stages
    above defend, so the headline test's premise is only as good as these."""

    def test_same_window_different_label_is_blocked(self, cs):
        keys = [(0, "S01"), (0, "S02"), (1, "S01")]
        C = cs.cannot_link_matrix(keys, [5.0, 5.0, 5.0])
        assert C[0, 1] and C[1, 0]
        assert not C[0, 2] and not C[1, 2]

    def test_same_window_same_label_is_not_blocked(self, cs):
        keys = [(0, "S01"), (0, "S01")]
        C = cs.cannot_link_matrix(keys, [5.0, 5.0])
        assert not C.any()

    def test_a_turn_under_durable_asserts_nothing(self, cs):
        """A half-second of crosstalk is not MOSS witnessing a second person."""
        keys = [(0, "S01"), (0, "S02")]
        assert not cs.cannot_link_matrix(keys, [5.0, 0.5]).any()
        assert cs.cannot_link_matrix(keys, [5.0, 1.0])[0, 1]

    def test_a_window_claiming_more_than_guard_speakers_is_ignored(self, cs):
        keys = [(0, "S%02d" % i) for i in range(12)]
        secs = [5.0] * 12
        assert not cs.cannot_link_matrix(keys, secs, guard=10).any()
        assert cs.cannot_link_matrix(keys, secs, guard=20).any()

    def test_is_symmetric_and_hollow(self, cs):
        A, secs, keys = random_meeting(5)
        C = cs.cannot_link_matrix(keys, secs)
        assert np.array_equal(C, C.T)
        assert not C.diagonal().any()
