"""pipeline/match_speakers.py -- identity by matching, and what must never merge.

Written against a measured failure. The agglomerative path this replaces merged
aggregates into each other, so merges were transitive: A joined B, B joined C,
and A and C were one speaker without ever having been compared. Over 238 SCOTUS
oral arguments that produced 9.0 identities where 11.3 people spoke and put
13.82% of all speech under the wrong name -- worst case one argument at 42%
purity, with the three women on the bench welded into a single "Sotomayor".

So the tests below assert the properties that failure violated, not an accuracy
number: that two atoms far apart never end up together because something sat
between them, that MOSS's within-window separation is binding rather than
advisory, and that a person may sound like more than one thing.
"""
import numpy as np
import pytest

from conftest import DIM, at_cosine, basis

import match_speakers as ms


def atom(v, sec=10.0, window=0, local="S01"):
    return {"key": (window, local), "v": ms.unit(np.asarray(v, np.float32)),
            "sec": sec, "start": 0.0, "n": 1}


class TestNothingChains:
    def test_two_dissimilar_atoms_do_not_join_through_a_middle_one(self):
        """The defect this module exists for.

        A and C are far apart; B sits between them and is close to both. Under
        agglomerative linkage A-B and B-C merge and carry C in with A. Matching
        compares against representatives only, so C is never handed A's identity
        on B's evidence.
        """
        a, c = basis(0), at_cosine(0.30, 1)
        b = ms.unit(a + c)
        atoms = [atom(a, window=0), atom(b, window=1), atom(c, window=2)]
        names, prov, sim = ms.assign(atoms, ms.Bank(), accept=0.55)
        assert prov[0] != prov[2], "far-apart atoms joined through the middle"

    def test_a_representative_is_the_atom_with_the_most_speech(self):
        """A sliver must not become the thing everyone else is measured against."""
        big, small = basis(0), at_cosine(0.9, 1)
        atoms = [atom(small, sec=0.5, window=0), atom(big, sec=90.0, window=1),
                 atom(big, sec=40.0, window=2)]
        names, prov, sim = ms.assign(atoms, ms.Bank(), accept=0.55)
        assert prov[1] == prov[2]


class TestMossIsBinding:
    def test_two_atoms_in_one_window_never_take_the_same_person(self):
        """MOSS heard the whole window and separated them. That is not a score."""
        b = ms.Bank(); b.add("ada", basis(0))
        v = basis(0)
        atoms = [atom(v, window=7, local="S01"), atom(v, window=7, local="S02")]
        names, prov, sim = ms.assign(atoms, b, accept=0.55)
        assert names.count("ada") == 1, "one window gave one person two labels"

    def test_the_stronger_claim_keeps_the_name(self):
        b = ms.Bank(); b.add("ada", basis(0))
        near, far = basis(0), at_cosine(0.62, 1)
        atoms = [atom(far, window=3, local="S01"), atom(near, window=3, local="S02")]
        names, prov, sim = ms.assign(atoms, b, accept=0.55)
        assert names[1] == "ada" and names[0] is None

    def test_the_loser_falls_through_to_its_next_choice(self):
        b = ms.Bank(); b.add("ada", basis(0)); b.add("bo", at_cosine(0.7, 1))
        atoms = [atom(basis(0), window=1, local="S01"),
                 atom(ms.unit(basis(0) * 0.9 + at_cosine(0.7, 1)), window=1,
                      local="S02")]
        names, prov, sim = ms.assign(atoms, b, accept=0.55)
        assert set(n for n in names if n) <= {"ada", "bo"}
        assert names[0] != names[1]


class TestSubProfiles:
    def test_a_person_may_sound_like_two_things(self):
        """Courtroom and telephone are the same human being.

        Averaging them gives a point that is neither: measured on the Court's
        2020-21 telephone arguments a single averaged reference put 22-28% of
        speech under the WRONG name.
        """
        b = ms.Bank()
        b.add("ada", basis(0))
        b.add("ada", at_cosine(0.2, 1))
        assert len(b) == 1
        for v in (basis(0), at_cosine(0.2, 1)):
            names, _, _ = ms.assign([atom(v)], b, accept=0.55)
            assert names[0] == "ada"

    def test_one_circumstance_pools_rather_than_appends(self):
        b = ms.Bank()
        b.add("ada", basis(0), condition="courtroom")
        b.add("ada", at_cosine(0.95, 1), condition="courtroom")
        assert sum(len(e) for e in b._ex) == 1
        b.add("ada", at_cosine(0.2, 1), condition="telephone")
        assert sum(len(e) for e in b._ex) == 2

    def test_a_condition_is_an_opaque_key_not_a_date(self):
        """Nothing reads meaning into the value, which is the point.

        It exists so someone can say "that is also her, through a potato" and
        have that be storable. "telephone", "2015" and "the bad conference room"
        are the same kind of thing to this code.
        """
        b = ms.Bank()
        for c in ("2015", "telephone", "through a potato", "far-field"):
            b.add("ada", at_cosine(0.15, hash(c) % 8 + 1), condition=c)
        assert len(b) == 1
        assert sum(len(e) for e in b._ex) == 4

    def test_naming_someone_again_adds_a_profile_rather_than_replacing(self):
        """The recovery path for a voice that stopped being recognised.

        Someone appears over a new microphone, is not matched, and a human says
        "that is also her". That must not overwrite how she sounded before, or
        the fix for today's recording breaks every recording before it.
        """
        b = ms.Bank()
        near, far = basis(0), at_cosine(0.1, 1)
        b.add("ada", near, condition="room")
        assert ms.assign([atom(far)], b, accept=0.55)[0] == [None]
        b.add("ada", far, condition="potato")
        assert ms.assign([atom(far)], b, accept=0.55)[0] == ["ada"]
        assert ms.assign([atom(near)], b, accept=0.55)[0] == ["ada"]

    def test_score_takes_the_nearest_exemplar_not_their_mean(self):
        b = ms.Bank(); b.add("ada", basis(0)); b.add("ada", at_cosine(0.0, 1))
        S = b.score(np.stack([ms.unit(basis(0))]))
        assert S[0, 0] == pytest.approx(1.0, abs=1e-5)


class TestConsolidation:
    def test_a_pooled_identity_is_named_when_its_atoms_alone_were_not(self):
        """Found by reading a transcript, not by a metric.

        One justice's single continuous turn came out as two identities because
        two of her windows scored just over the accept bar and the rest just
        under. Pooled, she is unambiguous -- an atom is a noisier estimate of a
        voice than the identity it belongs to.
        """
        ref = basis(0)
        b = ms.Bank(); b.add("ada", ref)
        # each atom alone sits below accept; together they average onto ref
        off1, off2 = at_cosine(0.52, 1), at_cosine(0.52, 2)
        atoms = [atom(ms.unit(ref * 0.52 + off1 * 0.5), sec=20.0, window=1),
                 atom(ms.unit(ref * 0.52 + off2 * 0.5), sec=20.0, window=2)]
        names, prov, sim = ms.assign(atoms, b, accept=0.58)
        assert names == ["ada", "ada"] or prov[0] == prov[1]


class TestLabelMeeting:
    def test_labels_are_dense_and_start_at_zero(self):
        keys = [(0, "S01"), (0, "S02"), (1, "S01")]
        A = np.stack([basis(0), basis(1), basis(0)])
        lab, name_of, info = ms.label_meeting(keys, A, [10.0, 10.0, 10.0])
        assert set(lab) == set(range(info["k"]))

    def test_an_unnamed_voice_is_an_identity_not_an_error(self):
        keys, A = [(0, "S01")], np.stack([basis(0)])
        lab, name_of, info = ms.label_meeting(keys, A, [10.0])
        assert lab[0] >= 0 and name_of == {}

    def test_a_bank_name_reaches_the_label(self):
        b = ms.Bank(); b.add("ada", basis(0))
        keys, A = [(0, "S01")], np.stack([basis(0)])
        lab, name_of, info = ms.label_meeting(keys, A, [10.0], b)
        assert name_of[int(lab[0])] == "ada"
        assert info["named_share"] == pytest.approx(1.0)


class TestNameGuard:
    """A misspelling creates a second person and splits their voice in half.

    The name string is the identity: INSERT OR IGNORE on `name` makes any
    distinct spelling a distinct speaker, silently, and every later use of
    either spelling makes it worse. Nothing downstream can detect it.
    """

    def _conn(self, tmp_path):
        import sqlite3
        import speakers as SPK
        c = SPK.db(str(tmp_path / "s.db"))
        c.execute("INSERT INTO speakers(name, created_at) VALUES('Sonia "
                  "Sotomayor', 0)")
        c.commit()
        return c

    def test_a_one_character_slip_is_caught(self, tmp_path):
        import speakers as SPK
        near = SPK.near_names(self._conn(tmp_path), "Sonia Sotomayer")
        assert near and near[0][0] == "Sonia Sotomayor"

    def test_spacing_and_punctuation_are_the_same_name(self, tmp_path):
        import speakers as SPK
        conn = self._conn(tmp_path)
        for variant in ("sonia sotomayor", "Sonia  Sotomayor", "Sonia-Sotomayor"):
            near = SPK.near_names(conn, variant)
            assert near, variant
            assert "spacing" in near[0][1]

    def test_a_genuinely_different_person_passes(self, tmp_path):
        import speakers as SPK
        assert SPK.near_names(self._conn(tmp_path), "Elena Kagan") == []

    def test_the_exact_existing_name_is_not_flagged(self, tmp_path):
        import speakers as SPK
        assert SPK.near_names(self._conn(tmp_path), "Sonia Sotomayor") == []
