#!/usr/bin/env python3
"""Give each MOSS label an identity by matching it to a known voice.

The path this replaces clustered atoms into each other agglomeratively, so
merges were transitive: A joined B, B joined C, and A and C were one speaker
without ever having been compared. Measured over 238 SCOTUS arguments that put
13.8% of all speech on the wrong person and produced 9.0 identities where 11.3
people spoke -- it collapsed same-gender speakers, worst case 42% purity.

Nothing here merges. Atoms are scored against references, so a wrong pairing
costs one atom instead of dragging a subtree with it.

A person is a SET of reference vectors, not an average of them. The same voice
over a courtroom microphone and over a telephone are far apart as vectors and
are the same human being; averaging them produces a point that is neither, and
which of the two a later recording resembles is not a thing to compromise
between. Measured on the Court's 2020-21 telephone arguments, a single averaged
reference put 22-28% of speech under the WRONG name -- not unnamed, wrong --
while the same reference was 92-96% correct on courtroom audio near enrolment.
So each person owns exemplars, an atom matches the nearest ONE, and a new
circumstance becomes another exemplar rather than a correction to the old one.

A `condition` is whatever circumstance made someone sound different, as a free
string: "telephone", "far-field", "2015", "headset", "the bad conference room".
Nothing here parses it or gives any value special meaning -- it is a key, and it
exists so a person can say "that is also her, through a potato" and have that be
storable. Automatic values are only a default; a human naming one is the point.

Scoring is a single matmul of every atom against every exemplar at once. The
per-person maximum is a reduceat over exemplars sorted by owner. Nothing walks
references in Python: the reference bank is the feature, so consulting it has to
stay flat in the number of people already known.

An atom is one MOSS window-local label with its speech pooled. Those labels are
93% pure at >=90% one speaker (measured), so pooling is safe and it buys a
vector built from a whole turn rather than a fragment.
"""
import collections
import os
from collections import defaultdict

import numpy as np

# Least similarity at which an atom joins an existing person. Atoms of one
# speaker run a median of 0.82 and a p10 of 0.47 on this corpus, different
# speakers a median of 0.11 and a p90 of 0.31 -- far apart, so this sits between
# them and nearer the impostor side: an unnamed voice costs a click, a wrong name
# is asserted and propagates to every meeting that person appears in.
#
# The right value depends on how COMPLETE the gallery is, which is the one thing
# an evaluation quietly gets wrong. Named against a full gallery -- every speaker
# enrolled -- 0.55 runs at 0.18% wrong, because the true person is present and
# wins the comparison outright. Name four people out of thirteen and the same
# 0.55 runs at 3.59%: an advocate who is nobody in the store lands 0.57 from a
# justice and there is no correct answer available to beat it. Swept against the
# reference with four of thirteen named:
#
#     0.55  3.59% wrong     0.68  2.05%
#     0.62  2.19% wrong     0.74  1.91%
#
# 0.62 is the knee -- it cuts wrong names by 39% for almost no coverage, where
# every step above it buys a tenth of a point and costs a dozen meetings. A
# deployment that has enrolled everyone can lower it.
ACCEPT = float(os.environ.get("MS_MATCH_ACCEPT", "0.62"))

# Second exemplar territory. An atom this close is the person; an atom between
# SUBPROFILE and ACCEPT is probably them in a medium we have not stored yet, and
# is kept as a candidate rather than named or discarded.
SUBPROFILE = float(os.environ.get("MS_SUBPROFILE", "0.42"))

# Speech an unmatched atom needs before it may FOUND its own identity rather
# than join one. Not a matching threshold: an atom shorter than this is still
# assigned, it just does not get to be the reference everything else is measured
# against. Below a couple of seconds a vector is too noisy to represent anyone.
FOUND_SEC = float(os.environ.get("MS_FOUND_SEC", "4.0"))


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def unit_rows(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(n > 0, n, 1.0)


def atoms_from(segments, emb, seg_idx):
    """Pool embeddings by MOSS window-local label.

    -> [{key, v, sec, start, n}] in time order. `key` is (window, local); two
    atoms sharing a window are different people by MOSS's own judgment, which
    assign() takes as binding rather than as evidence.
    """
    vecs, secs, first = defaultdict(list), defaultdict(float), {}
    for row, si in enumerate(seg_idx):
        s = segments[int(si)]
        k = (int(s["window"]), s["local_speaker"])
        vecs[k].append(emb[row])
        secs[k] += float(s["end"]) - float(s["start"])
        first.setdefault(k, float(s["start"]))
    out = [{"key": k, "v": unit(np.mean(vs, axis=0)), "sec": secs[k],
            "start": first[k], "n": len(vs)} for k, vs in vecs.items()]
    out.sort(key=lambda a: a["start"])
    return out


class Bank:
    """Named people, each holding one or more exemplar vectors.

    Exemplars are stored stacked and grouped by owner so scoring is one matmul
    and one reduceat, whatever the number of people or media they have been
    heard in.
    """

    def __init__(self):
        self.names = []                       # person index -> name
        self._ex = []                         # person index -> [vector]
        self._cond = []                       # person index -> [condition or None]

    def __len__(self):
        return len(self.names)

    def add(self, name, v, condition=None, sec=1.0):
        """Enrol a person, or give an existing one another exemplar.

        With `condition`, a person keeps ONE exemplar per condition and later
        speech under it pools in rather than appending. A voice heard a hundred
        times in one room is not a hundred facts about it; the circumstance is
        what actually varies -- measured, a 2015 reference scores 0.56 on the
        same justice in 2019, against a 0.55 floor. Unbounded appending reaches
        the same accuracy by brute force and 1635 exemplars, which stores every
        redundancy and cannot say which one is relevant to a new recording.

        The value is opaque. "telephone" and "2021" and "lapel mic" are all just
        keys; what matters is that speech recorded under different circumstances
        lands under different ones.
        """
        v = unit(np.asarray(v, dtype=np.float32))
        if name not in self.names:
            self.names.append(name); self._ex.append([]); self._cond.append([])
        i = self.names.index(name)
        if condition is not None and condition in self._cond[i]:
            j = self._cond[i].index(condition)
            self._ex[i][j] = unit(self._ex[i][j] + v * sec)
            return
        self._ex[i].append(v); self._cond[i].append(condition)

    def stacked(self):
        """-> (E, owner_offsets). E is exemplars sorted by owner."""
        if not self._ex:
            return np.zeros((0, 0), np.float32), np.zeros(1, np.int64)
        E = np.stack([v for ex in self._ex for v in ex]).astype(np.float32)
        off = np.cumsum([0] + [len(ex) for ex in self._ex])
        return E, off

    def score(self, A):
        """Best-exemplar similarity of every atom to every person.

        -> (n_atoms, n_people). One matmul, then a segment maximum over the
        exemplar axis: a person is as close as their closest exemplar, which is
        the whole point of keeping more than one.
        """
        E, off = self.stacked()
        if E.size == 0 or len(A) == 0:
            return np.zeros((len(A), len(self.names)), np.float32)
        S = A @ E.T                                    # (atoms, exemplars)
        return np.maximum.reduceat(S, off[:-1], axis=1)


def assign(atoms, bank, accept=ACCEPT):
    """Name what can be named; leave the rest as provisional identities.

    -> (names, prov, sim) where names[i] is a bank name or None, prov[i] is a
    provisional id for the unnamed, and sim[i] is the winning similarity.

    Two atoms in one window never take the same person. That constraint is not a
    score and is not tunable -- MOSS heard the whole window and separated them.
    Conflicts are settled in favour of the atom that matched more strongly; the
    loser falls through to its next admissible person, or to provisional.
    """
    if not atoms:
        return [], [], np.zeros(0, np.float32)
    A = unit_rows(np.stack([a["v"] for a in atoms]))
    P = bank.score(A)                                  # (atoms, people)

    names = [None] * len(atoms)
    sim = np.zeros(len(atoms), np.float32)
    taken = defaultdict(dict)                          # window -> person -> atom
    if P.size:
        order = np.argsort(-P, axis=1)                 # best person first
        rank = np.zeros(len(atoms), np.int64)
        # Settle strongest claims first so a confident atom is never displaced
        # by a marginal one; each unsettled atom then falls to its next choice.
        pend = list(np.argsort(-P.max(axis=1)))
        while pend:
            nxt = []
            for i in pend:
                while rank[i] < P.shape[1]:
                    p = int(order[i, rank[i]])
                    s = float(P[i, p])
                    if s < accept:
                        rank[i] = P.shape[1]
                        break
                    w = atoms[i]["key"][0]
                    held = taken[w].get(p)
                    if held is None:
                        taken[w][p] = i
                        names[i] = bank.names[p]; sim[i] = s
                        break
                    if s > sim[held]:                  # stronger claim wins
                        names[held] = None; sim[held] = 0.0
                        rank[held] += 1; nxt.append(held)
                        taken[w][p] = i
                        names[i] = bank.names[p]; sim[i] = s
                        break
                    rank[i] += 1
            pend = nxt

    # Provisional identities for everything unnamed: each unmatched atom takes
    # the strongest unmatched atom it resembles as its representative. Chosen by
    # descending speech, so a representative is a real turn and not a sliver --
    # and representatives never chain, which is the failure being replaced.
    left = [i for i in range(len(atoms)) if names[i] is None]
    prov = [None] * len(atoms)
    if left:
        idx = np.array(left)
        L = A[idx]
        S = L @ L.T
        np.fill_diagonal(S, -1.0)
        secs = np.array([atoms[i]["sec"] for i in left])
        reps, owner = [], {}
        for j in np.argsort(-secs):
            i = int(j)
            if i in owner:
                continue
            cand = [r for r in reps
                    if S[i, r] >= accept
                    and atoms[left[r]]["key"][0] != atoms[left[i]]["key"][0]]
            if cand:
                owner[i] = max(cand, key=lambda r: S[i, r])
            elif secs[i] >= FOUND_SEC or not reps:
                reps.append(i); owner[i] = i
            else:
                # Too little speech to FOUND an identity, which is a different
                # question from whether it can join one. A two-second fragment
                # has a vector too noisy to be anybody's reference, and letting
                # it start its own speaker is what turns one argument's eleven
                # people into twenty: measured, nine surplus identities held
                # sixteen seconds of a fifty-nine minute recording, every one of
                # them a splinter of somebody already present.
                #
                # So it joins the nearest identity it does not co-occur with,
                # however weakly -- being a short piece of a speaker already in
                # the room is far likelier than being the only trace of someone
                # who never speaks again.
                ok = [r for r in reps
                      if atoms[left[r]]["key"][0] != atoms[left[i]]["key"][0]]
                owner[i] = max(ok, key=lambda r: S[i, r]) if ok else i
                if owner[i] == i:
                    reps.append(i)
        for j, i in enumerate(left):
            prov[i] = "P%02d" % reps.index(owner[j])

        # A provisional identity pools every atom that joined it, so it is a far
        # better estimate of the voice than any one atom was -- reading a real
        # argument, one justice's single continuous turn came out as two
        # identities because two of her windows scored 0.56 and the rest 0.54
        # against the same reference. Matched as a whole she is unambiguous.
        # Same matmul, once more, against the pooled vectors.
        groups = collections.defaultdict(list)
        for i in left:
            groups[prov[i]].append(i)
        keys = sorted(groups)
        C = np.stack([unit(sum(A[i] * atoms[i]["sec"] for i in groups[k]))
                      for k in keys])
        PC = bank.score(C)
        if PC.size:
            for r, k in enumerate(keys):
                p_best = int(np.argmax(PC[r]))
                if float(PC[r, p_best]) < accept:
                    continue
                # the cannot-link still binds: this identity may not take a
                # person already holding another atom in any window it spans
                blocked = any(taken[atoms[i]["key"][0]].get(p_best) not in
                              (None, i) for i in groups[k])
                if blocked:
                    continue
                for i in groups[k]:
                    names[i] = bank.names[p_best]
                    sim[i] = float(PC[r, p_best])
                    prov[i] = None

        # And the same argument between provisional identities themselves. A
        # voice nobody has named still has to come out as ONE speaker: reading a
        # re-rendered argument, three advocates absent from the bank had split
        # into roughly forty one-turn identities, because single atoms are poor
        # estimates and every comparison between them was made atom to atom.
        # Pooled, they are obvious. Merging is to the LARGEST matching group
        # rather than pairwise-transitively, so nothing chains.
        alive = [k for k in keys if any(prov[i] for i in groups[k])]
        if len(alive) > 1:
            Cg = np.stack([unit(sum(A[i] * atoms[i]["sec"] for i in groups[k]))
                           for k in alive])
            secg = np.array([sum(atoms[i]["sec"] for i in groups[k])
                             for k in alive])
            wins = [{atoms[i]["key"][0] for i in groups[k]} for k in alive]
            G = Cg @ Cg.T
            np.fill_diagonal(G, -1.0)
            into, held = {}, {}
            for r in np.argsort(-secg):          # biggest identity is the target
                r = int(r)
                if r in into:
                    continue
                held.setdefault(r, [r])
                for c in np.argsort(-G[r]):
                    c = int(c)
                    if G[r, c] < accept:
                        break
                    if c in into or c == r:
                        continue
                    if wins[r] & wins[c]:        # co-occur: different people
                        continue
                    # Every ATOM pair, not the pooled centroids. Two identities
                    # that each pooled a borderline member produce centroids that
                    # have drifted toward each other, and merging on those is
                    # chaining wearing a disguise -- measured, two atoms 0.30
                    # apart came out at 0.58 once one of them had absorbed the
                    # ambiguous atom sitting between them. Requiring every pair to
                    # clear the bar makes a merge mean what it says.
                    # The bar here is SUBPROFILE, not accept, because the
                    # question is different: not "is this that person" but "could
                    # these be the same person at all". Measured on this corpus
                    # same-speaker atom pairs run a p10 of 0.47 and different
                    # speakers a p90 of 0.31, so 0.42 separates them -- while
                    # holding every pair to accept blocks nearly all real merges
                    # and left 172 identities in a 13-speaker argument.
                    Ar = A[[i for m in held[r] for i in groups[alive[m]]]]
                    Ac = A[groups[alive[c]]]
                    if float((Ar @ Ac.T).min()) < SUBPROFILE:
                        continue
                    into[c] = r
                    held[r].append(c)
                    wins[r] |= wins[c]
            for c, r in into.items():
                for i in groups[alive[c]]:
                    prov[i] = alive[r]
    return names, prov, sim


def atoms_from_aggregate(keys, A, secs):
    """Adapt link.py's aggregate() output to atoms. -> [atom]

    link.py already pools embeddings by MOSS window-local label and calls the
    result an aggregate, which is the same object under a different name. Taking
    its arrays directly keeps one aggregation in the codebase rather than two
    that can drift.
    """
    A = unit_rows(np.asarray(A, dtype=np.float32))
    return [{"key": tuple(k), "v": A[i], "sec": float(secs[i]),
             "start": float(i), "n": 1} for i, k in enumerate(keys)]


def label_meeting(keys, A, secs, bank=None, accept=ACCEPT):
    """Identity per aggregate, as integer labels. -> (lab, name_of, info)

    Drop-in for cluster_speakers.cluster()'s first return value, so link.py's
    downstream -- global ids, the JSON contract, the RTTM export -- is unchanged.
    `name_of` maps a label to a person's name where the bank recognised them;
    labels absent from it are real identities that nobody has named yet, which is
    the honest state for a voice heard for the first time.
    """
    bank = bank if bank is not None else Bank()
    atoms = atoms_from_aggregate(keys, A, secs)
    names, prov, sim = assign(atoms, bank, accept)

    order, lab, name_of = {}, np.full(len(atoms), -1, np.int64), {}
    for i in range(len(atoms)):
        tag = names[i] or prov[i]
        if tag is None:                       # never embedded, nothing to match
            continue
        if tag not in order:
            order[tag] = len(order)
            if names[i]:
                name_of[order[tag]] = names[i]
        lab[i] = order[tag]
    named_sec = sum(atoms[i]["sec"] for i in range(len(atoms)) if names[i])
    total = sum(a["sec"] for a in atoms) or 1.0
    info = {"k": len(order), "named": len(name_of), "bank": len(bank),
            "named_share": round(named_sec / total, 4),
            "mode": "match", "threshold": accept}
    return lab, name_of, info


def load_bank(conn, embed_model, names=None):
    """Read named voices out of speakers.db. -> Bank

    Only named people are loaded. An unnamed identity is a guess this pipeline
    made about a stranger; letting those accumulate into the reference set is how
    a guess becomes a fact nobody ever asserted.
    """
    q = ("SELECT s.name, e.condition, e.emb, e.dim FROM exemplars e"
         " JOIN speakers s ON s.id = e.speaker_id"
         " WHERE e.embed_model = ? AND s.name IS NOT NULL")
    b = Bank()
    for name, cond, blob, dim in conn.execute(q, (embed_model,)):
        if names and name not in names:
            continue
        b.add(name, np.frombuffer(blob, dtype=np.float32, count=dim),
              condition=cond)
    return b


def save_exemplar(conn, speaker_id, condition, v, embed_model, seconds):
    """Store or pool one circumstance's exemplar for a person.

    Pooling on conflict keeps the store bounded by people x conditions instead
    of by recordings: measured over 293 arguments, 16 people needed 128 exemplars
    this way against 1635 when every confident match was appended, at a LOWER
    wrong-name rate.
    """
    import time
    v = unit(np.asarray(v, dtype=np.float32))
    row = conn.execute(
        "SELECT id, emb, dim, seconds FROM exemplars WHERE speaker_id=? AND"
        " condition IS ? AND embed_model=?",
        (speaker_id, condition, embed_model)).fetchone()
    if row:
        old = np.frombuffer(row[1], dtype=np.float32, count=row[2])
        merged = unit(old * float(row[3] or 1.0) + v * seconds)
        conn.execute("UPDATE exemplars SET emb=?, seconds=? WHERE id=?",
                     (merged.astype(np.float32).tobytes(),
                      float(row[3] or 0.0) + seconds, row[0]))
    else:
        conn.execute(
            "INSERT INTO exemplars(speaker_id, condition, emb, dim, embed_model,"
            " seconds, created_at) VALUES(?,?,?,?,?,?,?)",
            (speaker_id, condition, v.astype(np.float32).tobytes(), len(v),
             embed_model, seconds, time.time()))
    conn.commit()
