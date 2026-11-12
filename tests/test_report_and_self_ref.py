"""qmf_report status codes and tier-3 scope, and E13 (P + sign, its sign probe and its bound)."""
import math

import numpy as np

import qmf.core as q
import qmf.experiments.uniform as uniform
import qmf.experiments.references as e9
import qmf.experiments.spray as e11
import qmf.experiments.self_ref as e13

A4 = 0b10110110  # a* on 4 cells


def seal(T, p0, rng):
    """Vault records from tamper codes T (units x cells): benign noise on the seal and vault halves."""
    return T ^ e9.seal_noise(T.shape, p0, rng) ^ e9.seal_noise(T.shape, p0, rng)


def test_report_flags_attacks_it_does_not_support():
    rng = np.random.default_rng(11)
    R = np.array([3, 9, 17, 28])
    ext = qmf_status(seal(e9.tamper(3000, R, A4, 1, 0.6, "extend", rng), 0.005, rng))
    assert ext["status"] == "global_anomaly" and ext["cells"] is None and "fingerprint" not in ext
    g = e11.spray_groups(R, rng)
    assert qmf_status(seal(e11.spray_tamper(3000, R, A4, 1, 0.6, g, rng), 0.005, rng))["status"] == "global_anomaly"
    to = np.array([5, 12, 20, 30])
    mir = qmf_status(seal(e9.tamper(3000, R, A4, 1, 0.6, "mirror", rng, np.broadcast_to(to, (3000, 4))), 0.005, rng))
    assert mir["status"] == "copied_errors" and "first" not in mir
    assert sorted(mir["copies"]) == sorted(sorted([int(a), int(b)]) for a, b in zip(R, to))
    obl = qmf_status(seal(e9.tamper(3000, R, A4, 1, 0.6, "obl", rng), 0.005, rng))
    assert obl["status"] == "localized" and obl["cells"] == R.tolist()
    assert (obl["fingerprint"]["a"], obl["fingerprint"]["s"]) == (A4, 1)
    clean = qmf_status(seal(np.zeros((3000, 32), dtype=np.int64), 0.005, rng))
    assert clean["status"] == "clean" and clean["cells"] == []
    for rep in (ext, mir, obl, clean):  # every report carries tier3
        assert rep["tier3"] in ("switch", "no_switch", "unsupported_same_a", "not_tested")
    assert ext["tier3"] == mir["tier3"] == clean["tier3"] == "not_tested"


def qmf_status(codes):
    return q.qmf_report(codes, 0.005)


def test_rarely_hit_cells_are_not_copies():
    """Two cells hit with different Paulis 10% of the time agree on about 0.81 of all units, but never when hit."""
    rng = np.random.default_rng(12)
    x = np.where(rng.random(4000) < 0.1, 1, 0)
    z = np.where(rng.random(4000) < 0.1, 2, 0)
    assert (x == z).mean() > 0.8 and q.copied_pairs(np.stack([x, z], axis=1), [0, 1]) == []
    assert q.copied_pairs(np.stack([x, x], axis=1), [0, 1]) == [[0, 1]]


def test_tier3_scope():
    rng = np.random.default_rng(13)
    R = np.array([2, 5, 7, 11, 13, 20])
    A, B = 0b101100100111, 0b010011010010
    sw = q.qmf_report(q.simulate_seal(24, R, [(300, None, 0, 0.0), (3000, A, 1, 0.6), (3000, B, -1, 0.6)], 0.005, rng),
                      0.005, window=1500)
    assert sw["tier3"] == "switch" and "switch" in sw
    same = q.qmf_report(q.simulate_seal(24, R, [(300, None, 0, 0.0), (3000, A, 1, 0.6), (3000, A, -1, 0.6)], 0.005, rng),
                        0.005, window=1500)
    assert same["tier3"] == "unsupported_same_a" and "switch" not in same
    one = q.qmf_report(q.simulate_seal(24, R, [(300, None, 0, 0.0), (6000, A, 1, 0.6)], 0.005, rng), 0.005, window=1500)
    assert one["tier3"] == "no_switch"


def test_sign_probe_law():
    """A product eigenstate of a* read in a*'s local bases: mean s eps (1 - 4p0/3)^|a*| (seal-half noise only)."""
    r, eps, p0, n = 4, 0.6, 0.05, 200000
    rng = np.random.default_rng(14)
    E = q.depolarize(q.sample("scramble", n, r, A4, -1, eps, 0.0, rng), r, p0, rng)
    lam = np.mean(1 - 2 * q.symp(E, np.int64(A4), r))
    assert abs(lam - (-eps * (1 - 4 * p0 / 3) ** int(q.weight(np.int64(A4), r)))) < 0.01


def test_e13_is_p_plus_sign():
    r, eps, p0, t = 4, 0.6, 0.005, 0
    row = e13.e13_trial((r, eps, p0, t))
    P = uniform.trial(("scramble", "paired", r, eps, p0, t, None, None, "dep"))
    m = math.ceil(2 * math.log(100) / (eps * (1 - 4 * p0 / 3) ** r) ** 2)
    assert row["m"] == m and row["n_max"] == 2 * P["n_max"] + m
    assert [c["n"] for c in row["curve"]] == [c["n"] + m for c in P["curve"]]
    assert [c["a_correct"] for c in row["curve"]] == [c["a_correct"] for c in P["curve"]]
    assert all(c["correct"] == (c["a_correct"] and row["sign_ok"]) for c in row["curve"])
    assert row["curve"][-1]["correct"]


def test_paired_sign_bound():
    b = e13.paired_sign_bound(4, 0.6, 0.005)
    assert b["M"] == 3051240 and b["units"] == 2 * b["pairs"] + b["sign_units"]
    u = [e13.paired_sign_bound(r, 0.6, 0.005)["units"] for r in (4, 6, 8, 10)]
    assert u == sorted(u) and u[-1] < 3 * u[0]  # polynomial in r, not 2^r
