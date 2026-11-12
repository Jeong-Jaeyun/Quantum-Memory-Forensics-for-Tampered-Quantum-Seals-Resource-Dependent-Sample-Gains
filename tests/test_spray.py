"""E11: the spray-mirror attacker."""
import numpy as np

import qmf.core as q
import qmf.experiments.references as e9
import qmf.experiments.spray as e11


def test_spray_copies_every_component_onto_balanced_groups():
    rng = np.random.default_rng(0)
    R = np.array([1, 5, 9, 20, 30])
    g = e11.spray_groups(R, rng)
    out = np.setdiff1d(np.arange(32), R)
    sizes = np.bincount(g[out], minlength=5)
    assert (g[R] == np.arange(5)).all() and sizes.max() - sizes.min() <= 1
    T = e11.spray_tamper(50, R, 0b1101100111, 1, 0.6, g, rng)
    assert (T[:, out] == T[:, R[g[out]]]).all()


def test_inseal_data_are_invariant_under_the_diagonal():
    """Every cell carries a component, so any in-seal pair reads E_i E_k, unchanged by E -> E (P x ... x P)."""
    rng = np.random.default_rng(1)
    R = np.array([0, 7, 13, 21])
    g = e11.spray_groups(R, rng)
    T = e11.spray_tamper(200, R, 0b10110110, -1, 0.6, g, rng)
    mate = e9.random_matching(200, rng)
    P = rng.integers(1, 4, (200, 1))  # a different global Pauli in every unit
    assert (e9.inseal_outcomes(T, mate, 0.0, rng) == e9.inseal_outcomes(T ^ P, mate, 0.0, rng)).all()


def test_chain_law_is_the_vault_law_in_C_and_uniform_outside():
    r, eps, p0, n = 3, 0.6, 0.02, 200000
    rng = np.random.default_rng(2)
    R = np.array([3, 10, 25])
    g = e11.spray_groups(R, rng)
    groups = [np.flatnonzero((g == i) & ~np.isin(np.arange(32), R)) for i in range(r)]

    def chain(a, s):
        X = e11.spray_tamper(n, R, a, s, eps, g, rng) ^ e9.seal_noise((n, 32), p0, rng)
        partner = np.stack([rng.choice(groups[k + 1], n) for k in range(r - 1)], axis=1)
        return q.pack_cells(X[:, R[:-1]] ^ X[np.arange(n)[:, None], partner])
    a_in = int(q.pack_cells(np.array([[1, 2, 3]]))[0])  # X Z Y: XOR of codes 0, so a* in C
    acodes = q.unpack(np.array([a_in]), r)[0]
    b = e11.chain_index(acodes)
    lam = np.mean(1 - 2 * q.symp(chain(a_in, -1), np.int64(b), r - 1))
    att = (1 - 4 * p0 / 3) ** (2 * int(q.weight(np.int64(b), r - 1)))
    assert abs(lam - (-eps * att)) < 0.01
    a_out = int(q.pack_cells(np.array([[1, 2, 2]]))[0])  # XOR of codes 1: not in C
    assert e11.chain_index(q.unpack(np.array([a_out]), r)[0]) is None
    f = np.bincount(chain(a_out, 1), minlength=16) / n
    assert np.abs(f - 1 / 16).sum() < 0.02


def test_e11_labs_end_to_end():
    ins = [t for t in range(40) if e11.in_C(q.unpack(np.array([e9.instance(4, 0.6, 0.005, t)[1]]), 4)[0])]
    outs = [t for t in range(40) if t not in ins]
    v = e11.e11_trial(("vault_seal", 4, 0.6, 0.005, outs[0]))
    assert v["components_exact"] and v["cells_exact"] and v["curve"][-1]["correct"]
    o = e11.e11_trial(("inseal_oracle", 4, 0.6, 0.005, ins[0]))
    assert o["in_C"] and o["curve"][-1]["correct"]
    o = e11.e11_trial(("inseal_oracle", 4, 0.6, 0.005, outs[0]))
    assert not o["in_C"] and not any(p["correct"] for p in o["curve"])
    i = e11.e11_trial(("inseal", 4, 0.6, 0.005, outs[0]))
    assert not any(p["correct"] for p in i["curve"])
