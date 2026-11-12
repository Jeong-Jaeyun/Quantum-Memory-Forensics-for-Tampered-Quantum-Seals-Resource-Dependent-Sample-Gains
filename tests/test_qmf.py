import itertools

import numpy as np

from qmf.core import (depolarize, fwht, lagrangian_containing, product_stabilizer, random_stabilizer, run_memory, sample,
                      span_elements, swap_xz, symp, syndrome_parities, weight)

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1]); Y = 1j * X @ Z


def mat(p, r):
    m = (1 << r) - 1
    out = np.array([[1]])
    for i in range(r):
        x, z = (p >> i) & 1, (p >> (r + i)) & 1
        out = np.kron(out, [I2, X, Z, Y][x + 2 * z])  # order of kron does not matter for commutation
    return out


def test_symp_matches_commutation():
    r = 2
    for a, b in itertools.product(range(16), repeat=2):
        A, B = mat(a, r), mat(b, r)
        comm = np.allclose(A @ B, B @ A)
        assert symp(np.int64(a), np.int64(b), r) == (0 if comm else 1)


def test_fwht_and_swap():
    r = 2
    rng = np.random.default_rng(0)
    h = rng.integers(0, 5, 16)
    W = fwht(h)
    for b in range(16):
        assert np.isclose(W[b], sum(h[e] * (-1) ** bin(b & e).count("1") for e in range(16)))
    allp = np.arange(16, dtype=np.int64)
    for a in range(16):
        lam = W[swap_xz(np.int64(a), r)]
        assert np.isclose(lam, sum(h[e] * (-1) ** int(symp(np.int64(a), np.int64(e), r)) for e in range(16)))
    assert list(weight(allp, r)[:5]) == [0, 1, 1, 2, 1]


def test_scramble_bias_and_memory_exact():
    r, a, s, eps = 4, 0b10110011, -1, 0.5
    rng = np.random.default_rng(1)
    E = sample("scramble", 200000, r, a, s, eps, 0.0, rng)
    assert abs(np.mean(1 - 2 * symp(E, a, r)) - s * eps) < 0.01
    other = 0b01000001
    assert abs(np.mean(1 - 2 * symp(E, other, r))) < 0.01
    curve = run_memory("scramble", r, a, s, 1.0 - 1e-9, 0.0, 0.0, 200, np.random.default_rng(2), np.random.default_rng(0))
    assert curve[-1]["correct"]


def test_depolarize_rate():
    r, p = 3, 0.2
    E = depolarize(np.zeros(100000, dtype=np.int64), r, p, np.random.default_rng(3))
    hit = weight(E, r)
    assert abs(hit.mean() / r - p) < 0.01


def _isotropic_rank(gens, r):
    for g, h in itertools.combinations(gens, 2):
        assert symp(np.int64(g), np.int64(h), r) == 0
    el = span_elements(np.array([gens]))[0]
    assert len(set(el.tolist())) == 2 ** r


def test_stabilizer_groups_and_syndromes():
    r = 5
    rng = np.random.default_rng(4)
    for gens in list(random_stabilizer(r, rng, 20)) + list(product_stabilizer(r, rng, 5)):
        _isotropic_rank(gens.tolist(), r)
    gens = random_stabilizer(r, rng, 7)
    E = rng.integers(0, 4 ** r, 7, dtype=np.int64)
    sig = symp(gens, E[:, None], r)
    el, par = span_elements(gens), syndrome_parities(sig)
    assert np.array_equal(par, symp(el, E[:, None], r))


def test_memoryless_checkpoint_uses_only_n_units():
    """With q=0 nothing is ever observed, so at every checkpoint the posterior is flat over 4^r - 1 hypotheses
    and the true-hypothesis mass must equal 1/(4^r - 1); with q=1 a checkpoint at n must not see units beyond n."""
    from qmf.core import run_memoryless
    ev = np.random.default_rng
    r = 3
    curve = run_memoryless("local", "ml_product", r, 5, 1, 0.0, 1e-12, 0.0, 200, np.random.default_rng(6), ev(0))
    assert [p["n"] for p in curve][:3] == [4, 5, 6]
    a = 0b100111  # weight-3 Pauli: product probes on each qubit reveal it slowly
    for model, method, eps, q in (("local", "ml_product", 0.0, 0.5), ("scramble", "ml_random", 0.6, 0.0),
                                  ("scramble", "ml_mix", 0.6, 0.0)):
        seen = []
        for n_max in (4, 2000):  # scramble probes are chunked up to the next checkpoint
            c = run_memoryless(model, method, r, a, 1, eps, q, 0.0, n_max, np.random.default_rng(7), ev(0))
            seen.append(c[0]["post_true"])
        assert np.isclose(seen[0], seen[1])  # the n=4 checkpoint is identical whatever n_max is


def test_uniform_lagrangian_is_uniform():
    from qmf.core import uniform_lagrangian
    rng = np.random.default_rng(10)
    for gens in uniform_lagrangian(6, rng, 30):
        _isotropic_rank(gens.tolist(), 6)
    # r = 2: there are (2+1)(4+1) = 15 Lagrangians; each must appear with frequency 1/15
    G = uniform_lagrangian(2, rng, 30000)
    spans = [frozenset(x) for x in span_elements(G).tolist()]
    freq = {k: spans.count(k) for k in set(spans)}
    assert len(freq) == 15
    assert max(abs(v - 2000) for v in freq.values()) < 6 * np.sqrt(2000)
    # r = 4: a fixed weight-1 and a fixed weight-4 Pauli are each covered with probability (2^r - 1)/(4^r - 1)
    el = span_elements(uniform_lagrangian(4, rng, 40000))
    for p in (0b1, 0b1011_0110):
        assert abs((el == p).any(axis=1).mean() - 15 / 255) < 0.006


def test_random_pauli_weight_and_mix_generators():
    from qmf.core import probe_generators, random_pauli_weight
    rng = np.random.default_rng(11)
    for w in range(0, 7):
        assert weight(np.int64(random_pauli_weight(6, w, rng)), 6) == w
    for gens in probe_generators("ml_mix", 5, rng, 40):
        _isotropic_rank(gens.tolist(), 5)


def test_ghz_probe_covers_full_weight_paulis():
    """GHZ + random local GL(2,2): isotropic and full rank; 2^(r-1) (+1 for even r) full-weight elements; a fixed
    weight-r Pauli is covered with probability N_r / 3^r; the vault floor raises the horizon; ml_ghz recovers w = r."""
    from qmf.core import ghz_stabilizer, n_max_for
    from qmf.experiments.uniform import trial
    rng = np.random.default_rng(13)
    for r in (3, 4, 5):
        G = ghz_stabilizer(r, rng, 20000)
        for gens in G[:10]:
            _isotropic_rank(gens.tolist(), r)
        el = span_elements(G)
        nr = 2 ** (r - 1) + (r % 2 == 0)
        assert ((weight(el, r) == r).sum(axis=1) == nr).all()
        a = int((1 << r) - 1)  # X^r
        assert abs((el == a).any(axis=1).mean() - nr / 3 ** r) < 0.012
    assert n_max_for("scramble", "memory", 6, 0.15, 3, 0.9, 0.005, floor=20) == int(20 * 6 * 64 / 0.15 ** 2)
    row = trial(("scramble", "ml_ghz", 4, 0.9, 0.0, 0, 4, None, "dep"))
    assert row["weight"] == 4 and row["curve"][-1]["correct"]


def test_ghz_covering_and_adaptive_probe():
    from qmf.core import ghz_covering, ghz_frames, ghz_stabilizer, probe_generators, random_pauli_weight
    rng = np.random.default_rng(14)
    # refactor keeps the RNG stream: same frames as before for the same seed
    assert np.array_equal(ghz_stabilizer(5, np.random.default_rng(3), 7),
                          ghz_frames(np.random.default_rng(3).integers(0, 6, (7, 5)), 5))
    for r in (3, 4, 6):
        for _ in range(40):
            a = random_pauli_weight(r, r, rng)
            gens = ghz_frames(ghz_covering(a, r, rng)[None], r)
            _isotropic_rank(gens[0].tolist(), r)
            assert a in span_elements(gens)[0].tolist()
    score = lambda: np.zeros((2, 4 ** 4))
    for gens in probe_generators("ml_adaptive_ghz", 4, rng, 16, score, done=64):
        _isotropic_rank(gens.tolist(), 4)


def test_switch2_modes_and_multi_parity():
    from qmf.experiments.multi import multi_trial, sample_multi
    from qmf.experiments.tier3 import switch_instance, switch_trial2
    I0, I1 = switch_instance(4, 0.9, 0.0, 0), switch_instance(4, 0.9, 0.0, 1)
    assert I0["switch"] and not I1["switch"]
    for mode in ("vault", "ml_blind", "ml_library"):
        sw, ns = switch_trial2((mode, 4, 0.9, 0.0, 0, 16)), switch_trial2((mode, 4, 0.9, 0.0, 1, 16))
        assert (sw["L1"], sw["L2"]) == (I0["L1"], I0["L2"])
        assert not ns["switch_detected"]
        if mode != "ml_blind":  # r=4, eps=0.9: vault and registry both find the switch; blind has too few units
            assert sw["switch_detected"] and abs(sw["switch_err"]) < 30
    r, A, S, ek = 3, [0b000111, 0b110001], [1, -1], 0.3
    E = sample_multi(300000, r, A, S, ek, np.random.default_rng(15))
    for a, s in zip(A, S):
        assert abs(np.mean(1 - 2 * symp(E, np.int64(a), r)) - s * ek) < 0.01
    assert abs(np.mean(1 - 2 * symp(E, np.int64(A[0] ^ A[1]), r))) < 0.01
    for lab in ("memory", "ml_random"):
        row = multi_trial((lab, 3, 2, 0.9, 0.0, 0))
        assert row["curve"][-1]["correct"] and row["curve"][0]["n_sig"] <= 2


def test_vault_exact_law_matches_explicit_sampling():
    from qmf.core import vault_counts, vault_mu
    r, a, s, eps, p0, n = 3, 0b100110, -1, 0.5, 0.05, 400000
    rng = np.random.default_rng(12)
    E = depolarize(depolarize(sample("scramble", n, r, a, s, eps, 0.0, rng), r, p0, rng), r, p0, rng)
    f_explicit = np.bincount(E, minlength=64) / n
    mu = float(vault_mu(np.int64(a), r, p0, None, "dep"))
    f_exact = vault_counts(r, a, s, eps * mu, n, rng) / n
    assert np.abs(f_explicit - f_exact).sum() < 0.03
    # dephasing vault half: Z flips with probability pv -> eigenvalue (1 - 2 pv)^(#X/Y positions)
    pv = 0.2
    Z = np.where(rng.random((n, r)) < pv, 1, 0) << (np.arange(r) + r)
    noise = Z.sum(axis=1)
    lam = np.mean(1 - 2 * symp(noise, np.int64(a), r))
    assert abs(lam - float(vault_mu(np.int64(a), r, 0.0, pv, "deph"))) < 0.01


def test_trial_runs_every_lab_and_shares_the_instance():
    from qmf.experiments.uniform import trial
    rows = [trial(("scramble", m, 3, 0.9, 0.0, 0, wt, None, "dep")) for m in ("memory", "ml_random", "ml_product", "ml_mix")
            for wt in (None, 2)]
    assert len({(x["a"], x["s"]) for x in rows if x["wt"] is None}) == 1
    assert all(x["weight"] == 2 for x in rows if x["wt"] == 2)
    assert all(x["curve"][-1]["correct"] for x in rows)


def test_paired_product_bias_is_eps_squared():
    r, a, s, eps = 4, 0b10110011, -1, 0.6
    rng = np.random.default_rng(8)
    E = sample("scramble", 200000, r, a, s, eps, 0.0, rng) ^ sample("scramble", 200000, r, a, s, eps, 0.0, rng)
    assert abs(np.mean(1 - 2 * symp(E, a, r)) - eps ** 2) < 0.01


def test_qmf_report_pipeline():
    from qmf.core import pack_cells, qmf_report, simulate_seal, unpack
    rng = np.random.default_rng(9)
    E = rng.integers(0, 4 ** 5, 50, dtype=np.int64)
    assert np.array_equal(pack_cells(unpack(E, 5)), E)
    n, R = 24, np.array([2, 5, 7, 11, 13, 20])
    A, B = 0b101100100111, 0b010011010010
    codes = simulate_seal(n, R, [(300, None, 0, 0.0), (3000, A, 1, 0.6), (3000, B, -1, 0.6)], 0.005, rng)
    rep = qmf_report(codes, 0.005, window=1500, library={"inc1": (tuple(R), A, 1), "other": (tuple(R), 0b11, 1)})
    assert rep["cells"] == R.tolist()
    assert abs(rep["onset"] - 300) <= 5
    assert (rep["first"]["a"], rep["first"]["s"]) == (A, 1) and (rep["last"]["a"], rep["last"]["s"]) == (B, -1)
    assert abs(rep["switch"] - 3300) <= 100
    assert rep["links"]["inc1"]["match"] and not rep["links"]["other"]["match"]
    quiet = qmf_report(simulate_seal(n, R, [(2000, None, 0, 0.0)], 0.005, rng), 0.005)
    assert quiet["cells"] == []


def test_lagrangian_contains_top_candidate():
    r = 6
    rng = np.random.default_rng(5)
    for _ in range(20):
        cands = rng.integers(1, 4 ** r, 24, dtype=np.int64)
        gens = lagrangian_containing(cands, r, rng)
        _isotropic_rank(gens.tolist(), r)
        el = set(span_elements(gens[None, :])[0].tolist())
        assert int(cands[0]) in el
