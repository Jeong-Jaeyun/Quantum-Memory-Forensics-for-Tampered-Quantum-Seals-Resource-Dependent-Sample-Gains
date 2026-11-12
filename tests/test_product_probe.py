"""E12: a product probe with no reference against spray, and the extend bound for the vault without R."""
import math

import numpy as np

import qmf.core as q
import qmf.experiments.references as e9
import qmf.experiments.spray as e11
import qmf.experiments.product as e12


def test_two_copies_read_the_whole_component():
    """With no noise, a Z-basis copy and an X-basis copy of group i give E_i exactly."""
    rng = np.random.default_rng(3)
    R = np.array([2, 8, 15, 27])
    g = e11.spray_groups(R, rng)
    T = e11.spray_tamper(300, R, 0b10011101, 1, 0.6, g, rng)
    groups = [np.flatnonzero(g == i) for i in range(4)]
    assert (e12.product_estimates(T, groups, rng) == T[:, R]).all()
    reps, rep = e12.tier1_product(T[:24] & 1)
    assert e12.groups_exact(reps, rep, g, 4) and reps == [int(G[0]) for G in sorted(groups, key=lambda G: G[0])]


def test_product_law_attenuates_by_bits():
    """Each bit comes from its own cell, so the parity of b is attenuated by (1 - 4p0/3)^bits(b), not ^(2|b|)."""
    r, eps, p0, n = 2, 0.6, 0.06, 200000
    rng = np.random.default_rng(4)
    R = np.array([4, 19])
    g = e11.spray_groups(R, rng)
    a = int(q.pack_cells(np.array([[3, 1]]))[0])  # Y X: 3 set bits
    X = e11.spray_tamper(n, R, a, 1, eps, g, rng) ^ e9.seal_noise((n, 32), p0, rng)
    E = q.pack_cells(e12.product_estimates(X, [np.flatnonzero(g == i) for i in range(r)], rng))
    lam = np.mean(1 - 2 * q.symp(E, np.int64(a), r))
    att = 1 - 4 * p0 / 3
    assert abs(lam - eps * att ** 3) < 0.01 and abs(lam - eps * att ** 4) > 0.03
    dec = e12.ProductFull(r, eps, p0)
    assert math.isclose(math.expm1(dec.lp[a]), eps * att ** 3)


def test_e12_uses_e11_seal_and_learns():
    t = 0
    v = e11.e11_trial(("vault_seal", 4, 0.6, 0.005, t))
    p = e12.e12_trial(("product", 4, 0.6, 0.005, t))
    assert p["R"] == v["R"] and p["groups"] == v["groups"] and p["n_max"] == v["n_max"]
    assert [c["n"] for c in p["curve"]] == [c["n"] for c in v["curve"]]
    assert p["groups_exact"] and p["n_groups"] == 4 and p["curve"][-1]["correct"]
    assert not any(c["correct"] for c in p["curve"] if c["n"] <= e12.T1)


def test_unknown_R_bound():
    b = e12.unknown_R_bound(4, 0.15, 0.005)
    assert b["M"] == 3051240  # E9 vault_sparse's candidate count
    em = 0.15 * (1 - 4 * 0.005 / 3) ** 8
    assert b["units"] == math.ceil(8 * math.log(20 * 3051240) / em ** 2)
    u = [e12.unknown_R_bound(r, 0.15, 0.005)["units"] for r in (4, 6, 8, 10)]
    assert u == sorted(u) and u[-1] < 2.5 * u[0]  # grows like r log(n / r), not like 2^r
