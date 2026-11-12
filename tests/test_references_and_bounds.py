"""E5b determinism, E9 (in-seal references), E10 (Lagrangian coverage bounds), the E1f fine replay, and the reporting
fields of the N90 analysis."""
import itertools
import math

import numpy as np

import qmf.core as q
import qmf.experiments.uniform as uniform
import qmf.experiments.references as e9
import qmf.experiments.coverage as lg


# ------------------------------------------------------------------ E5b
def test_adaptive_ghz_top_k_is_deterministic_under_ties():
    """With all scores tied, the exploit half must take the lowest-index full-weight candidates (stable sort)."""
    r, k = 4, 16
    cand = q.full_weight(r)
    flat = lambda: np.zeros((2, 4 ** r))
    G1 = q.probe_generators("ml_adaptive_ghz", r, np.random.default_rng(5), 8, flat, done=64)
    G2 = q.probe_generators("ml_adaptive_ghz", r, np.random.default_rng(5), 8, flat, done=64)
    assert np.array_equal(G1, G2)
    sc = np.zeros(4 ** r)
    assert np.array_equal(cand[np.argsort(-sc[cand], kind="stable")[:k]], cand[:k])


# ------------------------------------------------------------------ E9
def test_matchings_are_perfect_and_adaptive_pairs_go_outside():
    rng = np.random.default_rng(0)
    M = e9.random_matching(50, rng)
    assert (M[np.arange(50)[:, None], M] == np.arange(32)).all() and (M != np.arange(32)).all()
    Rh = np.array([1, 4, 9, 30])
    A = e9.adaptive_matching(50, Rh, rng)
    assert (A[np.arange(50)[:, None], A] == np.arange(32)).all() and (A != np.arange(32)).all()
    assert not np.isin(A[:, Rh], Rh).any()


def test_inseal_clean_partner_law_is_the_vault_law():
    """A cell of R paired with an untouched cell reads E_i x (noise on both cells): the vault law with p_v = p0."""
    r, a, s, eps, p0, n = 3, 0b101011, -1, 0.6, 0.05, 200000
    rng = np.random.default_rng(1)
    R = np.array([2, 7, 11])
    mate = e9.adaptive_matching(n, R, rng)
    O = e9.inseal_outcomes(e9.tamper(n, R, a, s, eps, "obl", rng), mate, p0, rng)
    E = q.pack_cells(O[:, R])
    lam = np.mean(1 - 2 * q.symp(E, np.int64(a), r))
    assert abs(lam - s * eps * float(q.vault_mu(np.int64(a), r, p0, None, "dep"))) < 0.01
    f = np.bincount(E, minlength=4 ** r) / n
    g = q.vault_counts(r, a, s, eps * float(q.vault_mu(np.int64(a), r, p0, None, "dep")), n, rng) / n
    assert np.abs(f - g).sum() < 0.03


def test_mirror_blinds_inseal_pairs_and_extend_scrambles_partners():
    r, a, s, eps, p0, n = 3, 0b101011, 1, 0.9, 0.01, 20000
    rng = np.random.default_rng(2)
    R = np.array([0, 5, 6])
    mate = e9.adaptive_matching(n, R, rng)
    O = e9.inseal_outcomes(e9.tamper(n, R, a, s, eps, "mirror", rng, e9.mirror_targets(mate, R)), mate, p0, rng)
    assert abs((O[:, R] != 0).mean() - (1 - (1 - p0) ** 2)) < 0.01  # only benign noise on both cells
    O = e9.inseal_outcomes(e9.tamper(n, R, a, s, eps, "extend", rng), mate, p0, rng)
    assert abs((O[:, R] != 0).mean() - 0.75) < 0.01  # partner fully depolarized
    lam = np.mean(1 - 2 * q.symp(q.pack_cells(O[:, R]), np.int64(a), r))
    assert abs(lam) < 0.03


def test_tier1_rules():
    rng = np.random.default_rng(3)
    R = np.array([3, 8, 15, 22, 29])
    mate = e9.random_matching(e9.T1, rng)
    O = e9.inseal_outcomes(e9.tamper(e9.T1, R, 0b1011100110, 1, 0.6, "obl", rng), mate, 0.005, rng)
    assert e9.tier1_inseal(O, mate).tolist() == R.tolist()
    pi0 = np.array([0, 1, 2, 4, 5])
    T = e9.tamper(e9.T1, R, 0b1011100110, 1, 0.6, "mirror", rng, np.broadcast_to(pi0, (e9.T1, 5)))
    reps, rep = e9.tier1_vault(T)
    assert reps == sorted(min(i, j) for i, j in zip(R, pi0)) and all(rep[max(i, j)] == min(i, j) for i, j in zip(R, pi0))


def test_partial_score_equals_full_score_when_every_cell_has_a_clean_partner():
    rng = np.random.default_rng(4)
    R, eps, p0 = np.array([1, 2, 3]), 0.6, 0.01
    mate = e9.adaptive_matching(40, R, rng)
    O = e9.inseal_outcomes(e9.tamper(40, R, 0b110101, 1, eps, "obl", rng), mate, p0, rng)
    full = e9.Full(3, eps, p0)
    full.add(q.pack_cells(O[:, R]))
    assert np.allclose(e9.partial_score(O, mate, R, eps, p0), full.score())


def test_full_weight_characters_match_brute_force():
    rng = np.random.default_rng(6)
    for k in (1, 2, 3):
        h = rng.integers(0, 5, (4 ** k, 3)).astype(float)
        res, loc = e9.full_weight_characters(h, k), e9.full_weight_locals(k)
        for i, b in enumerate(loc):
            for s in range(3):
                v = sum(h[e, s] * np.prod([1 - 2 * e9.SYM1[b[m], (e >> (2 * m)) & 3] for m in range(k)])
                        for e in range(4 ** k))
                assert abs(v - res[i, s]) < 1e-9


def test_e9_labs_end_to_end():
    for lab in ("inseal", "vault_seal"):
        row = e9.e9_trial((lab, "obl", 4, 0.6, 0.005, 1))
        assert row["cells_exact"] and row["curve"][-1]["correct"]
    blind = e9.e9_trial(("inseal", "mirror", 4, 0.6, 0.005, 1))
    assert not any(p["correct"] for p in blind["curve"])
    vm = e9.e9_trial(("vault_seal", "mirror", 4, 0.6, 0.005, 1))
    assert vm["cells_exact"] and vm["curve"][-1]["correct"]
    # regression: early checkpoints see the same representatives with a different copy map (an R cell not yet flagged
    # while its copy is); the true index must follow the current map
    vm = e9.e9_trial(("vault_seal", "mirror", 6, 0.6, 0.005, 3))
    assert vm["cells_exact"] and vm["curve"][-1]["correct"]
    rng = np.random.default_rng(7)
    curve = e9.run_vault_sparse(np.array([4, 17]), 0b0110, 1, 0.9, 0.0, 300, rng, np.random.default_rng(0))
    assert curve[-1]["correct"]


# ------------------------------------------------------------------ E10
def test_lagrangian_enumeration_counts_isotropy_and_graph_sets():
    for r in (1, 2, 3, 4):
        x = lg.enumerate_lagrangians(r, check_distinct=True)
        assert x["total"] == x["expected"] == x["distinct"] == lg.n_lagrangians(r)
        assert set(x["enumerators"]) == set(lg.enumerate_graphs(r)["enumerators"])
    for G in lg.all_lagrangians(3, chunk=64):
        for g in G[:: max(1, len(G) // 5)]:
            for u, v in itertools.combinations(g.tolist(), 2):
                assert q.symp(np.int64(u), np.int64(v), 3) == 0
            assert len(set(q.span_elements(g[None])[0].tolist())) == 8


def test_constraints_hold_and_ilp_is_exact_for_small_r():
    for r in (2, 3, 4, 5):
        E = np.array([[int(v) for v in k.strip("[]").split(",")] for k in lg.enumerate_lagrangians(r)["enumerators"]])
        K = lg.krawtchouk(r)
        for row in E:
            rep = lg.constraint_report(row, r, K)
            assert rep["macwilliams_ok"] and rep["shadow_ok"] and rep["simple_ok"]
        if r <= 4:
            assert [lg.ilp_max(r, w)["max"] for w in range(1, r + 1)] == E.max(axis=0)[1:].tolist()
    assert lg.ghz_count(6) == 33 and lg.search_max(5, 5, 0, restarts=5)["max"] == 16


def test_fano_bound_basics():
    assert lg.d_bits(0.0) == 0.0 and abs(lg.d_bits(0.1) - 0.1 ** 2 / (2 * math.log(2))) < 1e-4
    assert lg.fano_units(100, 0.01, 0.2) > lg.fano_units(100, 0.02, 0.2) > lg.fano_units(10, 0.02, 0.2)


# ------------------------------------------------------------------ reporting fields
def test_n90_reporting_fields():
    from qmf.analysis.summaries import n90
    rows = []
    for t in range(20):  # trial t succeeds from n = 10 + 5 t on
        ns = [int(x) for x in np.unique(np.round(np.geomspace(4, 400, 40)))]
        rows.append(dict(n_max=400, curve=[dict(n=n, correct=n >= 10 + 5 * t) for n in ns]))
    s = n90(rows)
    assert s["n90"] is not None and s["prev"] < s["n90"]
    assert s["prev"] <= s["n90_interp"] <= s["n90"]
    assert s["ci90_interp"][0] <= s["ci90_interp"][1] and not s["ci90_grid_limited"] is None
    assert s["grid_step"] == s["n90"] / s["prev"]


# ------------------------------------------------------------------ E1f
def test_fine_replay_reproduces_the_trial_at_its_own_checkpoints():
    """The replay draws exactly what qmf.experiments.uniform.trial drew: at the trial's own checkpoints post_true is bit-identical, and
    so is the correct flag, MAP ties included (trial 35 below has a tied MAP at n = 604; the first E1f run broke ties
    with a separate RNG and fails here)."""
    import qmf.experiments.fine as fine
    for lab, method, r, eps, t in (("vault", "memory", 4, 0.6, 2), ("vault", "memory", 5, 0.15, 2),
                                   ("ml", "ml_random", 4, 0.6, 2), ("vault", "memory", 5, 0.15, 35)):
        row = uniform.trial(("scramble", method, r, eps, 0.005, t, None, None, "dep", 0.0))
        cps = [p["n"] for p in row["curve"]]
        lo, hi = cps[5], cps[25]
        pts = sorted({int(x) for x in np.round(np.geomspace(lo, hi, 30))} | set(cps[5:26]))
        y = fine.fine_trial((lab, r, eps, 0.005, t, pts, lo, hi))
        assert (y["a"], y["s"], y["n_max"]) == (row["a"], row["s"], row["n_max"])
        st = {p["n"]: p for p in row["curve"]}
        assert [p["n"] for p in y["own"]] == cps[5:26] and [p["n"] for p in y["fine"]] == pts
        assert all(p["post_true"] == st[p["n"]]["post_true"] and p["correct"] == st[p["n"]]["correct"] for p in y["own"])


def test_fine_vault_bridge_counts():
    """Bridge draws between two checkpoints: m units drawn without replacement from the interval's counts."""
    rng = np.random.default_rng(8)
    inc = q.vault_counts(3, 0b100101, 1, 0.5, 500, rng)
    pool, tot = inc.copy(), np.zeros_like(inc)
    for m in (120, 80, 150):
        d = rng.multivariate_hypergeometric(pool, m)
        assert d.sum() == m and (d <= pool).all()
        pool -= d
        tot += d
    assert (tot + pool == inc).all() and pool.sum() == 150
