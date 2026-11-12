"""E11: the spray-mirror attacker, which needs no secret.

The model is E9's (src/qmf/experiments/references.py):
n = 32 cells, the E1 instances (seed space 941, uniform a*), benign depolarizing p0 on every cell and on the vault half,
cell codes 0=I 1=X 2=Z 3=Y, and a Bell measurement on two cells returns the XOR of their codes.

Attacker (spray). Once per seal it splits the 32 - r cells outside R into r groups (uniformly at random; sizes differ by at
most one). In every unit it copies the Pauli E_i of R cell i onto every cell of group i. It needs neither the pairing nor
the Pauli frame. Every cell then carries one component of E = (E_1, ..., E_r).

Why in-seal references go blind (exact). An in-seal Bell pair reads the product E_i E_k of two components (plus benign
noise), which does not change under E -> E (P x ... x P) for a single-qubit Pauli P. So all in-seal data, for any
pairing, any number of units and any side information, are a function of E modulo the diagonal group D = {P^(x r)}. The
law of E mod D is uniform unless a* commutes with D, i.e. unless the XOR of a*'s cell codes is 0 (a* in C;
(4^(r-1) - 1)/(4^r - 1), about 1/4, of the uniform family). For a* not in C the in-seal data are independent of (a*, s).
For a* in C they carry the full parity.

Labs (each trial: same instance, same R and groups for every lab):
  vault_seal     E9's full-seal vault, R not given. Tier 1 from the first min(n, T1) units flags cells with error rate
                 > 0.3 and merges exact copies, so each component gets one representative; exact Bayes on those.
  inseal         E9's in-seal lab (tier 1 from secret random pairings, then adaptive pairing; R not given).
  inseal_oracle  the strongest in-seal verifier: told R and the groups. Every unit uses the chain pairing (R cell k with
                 a random cell of group k+1, k = 1..r-1), which reads O_k = E_k E_{k+1}, i.e. all of E mod D. Exact Bayes
                 over (a, s) with the uniform prior: for a in C, chi_a(E) = chi_b(O) with b_k = a_1 ... a_k (a bijection
                 between C and the Paulis on r-1 cells; the two cells' noise gives the vault attenuation
                 (1 - 4p0/3)^(2|b|)); every a not in C gives the uniform law (log-likelihood ratio 0). Ties: uniform.
Decoders know eps and p0 (as in E1). Rows: curve of {n, correct, a_correct}; in_C; for the vault, cells_exact and
components_exact.

usage: python -m qmf.experiments.spray run WORKERS OUT.jsonl LABS R_LIST EPS_LIST P0 TRIALS
"""
import json
import multiprocessing
import sys

import numpy as np

import qmf.core as q
import qmf.experiments.references as e9

N, T1 = e9.N, e9.T1
LABS = ("vault_seal", "inseal", "inseal_oracle")


def spray_groups(R, rng, n=N):
    """g[c]: index i of the R cell whose Pauli cell c carries (R cell i: i). Outside cells: balanced random groups."""
    r = len(R)
    out = rng.permutation(np.setdiff1d(np.arange(n), R))
    g = np.empty(n, dtype=np.int64)
    g[R] = np.arange(r)
    g[out] = np.arange(len(out)) % r
    return g


def spray_tamper(B, R, a, s, eps, g, rng):
    """Tamper codes (B, n): the tool's Pauli on R; every other cell a copy of its group's R-cell Pauli."""
    T = e9.tamper(B, R, a, s, eps, "obl", rng)
    return T[:, R[g]]


def in_C(acodes):
    return int(np.bitwise_xor.reduce(acodes)) == 0


def chain_index(acodes):
    """Packed index of b (b_k = a_1 ... a_k, k = 1..r-1) on r-1 cells, or None if a* is not in C."""
    if not in_C(acodes):
        return None
    return int(q.pack_cells(np.bitwise_xor.accumulate(acodes)[:-1][None])[0])


def run_vault_seal(R, a, s, eps, p0, g, n_max, rng, rng_eval):
    r = len(R)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    T = spray_tamper(n_max, R, a, s, eps, g, rng)
    O = T ^ e9.seal_noise(T.shape, p0, rng) ^ e9.seal_noise(T.shape, p0, rng)  # seal half + vault half
    out, key, dec, done, rep = [], None, None, 0, {}
    for n in q.checkpoints(n_max):
        reps, rep = e9.tier1_vault(O[:min(n, T1)])
        if not 1 <= len(reps) <= e9.R_MAX:
            out.append(dict(n=n, **e9.FAIL))
            key = None
            continue
        if key != tuple(reps):
            key, dec, done = tuple(reps), e9.Full(len(reps), eps, p0), 0
        ti = e9.true_index(acodes, R, reps, rep)
        dec.add(q.pack_cells(O[done:n][:, reps]))
        done = n
        out.append(dict(n=n, **e9.evaluate(dec.score(), ti, s, len(reps), rng_eval)))
    rep_of = {i: rep.get(i, i) for i in R.tolist()}
    cells_exact = bool(key) and len(key) == r and sorted(set(rep_of.values())) == list(key)
    comp = bool(key) and all(rep.get(c, c) == rep_of[int(R[g[c]])] for c in range(N))
    return out, dict(Rhat=list(key) if key else [], cells_exact=cells_exact, components_exact=comp)


def run_inseal(R, a, s, eps, p0, g, n_max, rng, rng_eval):
    """E9's in-seal lab (run_inseal) with the spray attacker."""
    r = len(R)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    mate1 = e9.random_matching(T1, rng)
    O1 = e9.inseal_outcomes(spray_tamper(T1, R, a, s, eps, g, rng), mate1, p0, rng)
    Rh = e9.tier1_inseal(O1, mate1)
    info = dict(Rhat=Rh.tolist(), cells_exact=Rh.tolist() == R.tolist())
    cps = q.checkpoints(n_max)
    if not 1 <= len(Rh) <= e9.R_MAX or 2 * len(Rh) > N:
        return [dict(n=n, **e9.FAIL) for n in cps], info
    k = len(Rh)
    ti = e9.true_index(acodes, R, Rh.tolist())
    base = e9.partial_score(O1, mate1, Rh, eps, p0)
    dec, done, out = e9.Full(k, eps, p0), T1, []
    for n in cps:
        if n < T1:
            out.append(dict(n=n, **e9.FAIL))
            continue
        B = n - done
        if B:
            mate = e9.adaptive_matching(B, Rh, rng)
            dec.add(q.pack_cells(e9.inseal_outcomes(spray_tamper(B, R, a, s, eps, g, rng), mate, p0, rng)[:, Rh]))
            done = n
        out.append(dict(n=n, **e9.evaluate(base + dec.score(), ti, s, k, rng_eval)))
    return out, info


def evaluate_oracle(sc, ti, s, r, rng_eval):
    """MAP over (a, s) with the uniform prior: a in C through its chain image b (scores sc, shape (2, 4^(r-1))); the
    2 (4^r - 4^(r-1)) hypotheses with a not in C have log-likelihood ratio 0. Ties are broken uniformly."""
    k = r - 1
    sc = sc.copy()
    sc[:, 0] = -np.inf  # b = 0 <-> a = 0
    best = max(float(sc.max()), 0.0)
    ties = np.flatnonzero(sc == best)
    n_out = 2 * (4 ** r - 4 ** k) if best == 0.0 else 0
    j = int(rng_eval.integers(len(ties) + n_out))
    if j >= len(ties):
        return dict(correct=False, a_correct=False)
    kk = int(ties[j])
    sh, bh = (1 if kk < 4 ** k else -1), kk % (4 ** k)
    return dict(correct=bool(ti is not None and bh == ti and sh == s), a_correct=bool(ti is not None and bh == ti))


def run_inseal_oracle(R, a, s, eps, p0, g, n_max, rng, rng_eval):
    r = len(R)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    ti = chain_index(acodes)
    outside = ~np.isin(np.arange(N), R)
    groups = [np.flatnonzero((g == i) & outside) for i in range(r)]
    dec, done, out = e9.Full(r - 1, eps, p0), 0, []
    for n in q.checkpoints(n_max):
        B = n - done
        if B:
            X = spray_tamper(B, R, a, s, eps, g, rng) ^ e9.seal_noise((B, N), p0, rng)
            partner = np.stack([rng.choice(groups[k + 1], B) for k in range(r - 1)], axis=1)  # (B, r-1)
            Oc = X[:, R[:-1]] ^ X[np.arange(B)[:, None], partner]  # O_k = E_k E_{k+1} (+ two cells' noise)
            dec.add(q.pack_cells(Oc))
            done = n
        out.append(dict(n=n, **evaluate_oracle(dec.score(), ti, s, r, rng_eval)))
    return out, {}


def e11_trial(job):
    lab, r, eps, p0, t = job
    tag, a, s = e9.instance(r, eps, p0, t)
    geo = np.random.default_rng(tag + [11])  # R and groups: the same seal for every lab
    R = np.sort(geo.choice(N, r, replace=False))
    g = spray_groups(R, geo)
    li = LABS.index(lab)
    rng, rng_eval = np.random.default_rng(tag + [11, 1 + li]), np.random.default_rng(tag + [12, li])
    w = int(q.weight(np.int64(a), r))
    mu_a = float(q.vault_mu(np.int64(a), r, p0, None, "dep"))
    n_max = 2 * q.n_max_for("scramble", "memory", r, eps, w, mu_a, p0)  # as E9's inseal and vault_seal
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    row = dict(lab=lab, attacker="spray", r=r, eps=eps, p0=p0, trial=t, a=a, s=s, weight=w, mu_a=mu_a, R=R.tolist(),
               groups=g.tolist(), in_C=in_C(acodes), T1=T1, n_max=n_max)
    fn = {"vault_seal": run_vault_seal, "inseal": run_inseal, "inseal_oracle": run_inseal_oracle}[lab]
    curve, info = fn(R, a, s, eps, p0, g, n_max, rng, rng_eval)
    return dict(row, **info, curve=curve)


def run(workers, out, labs, rs, epss, p0, trials):
    jobs = [(lab, r, e, p0, t) for lab in labs for r in rs for e in epss for t in range(trials)]
    jobs.sort(key=lambda j: -(4 ** j[1]) / j[2] ** 2)
    with multiprocessing.Pool(workers, maxtasksperchild=8) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(e11_trial, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["lab"], row["r"], row["eps"], row["trial"], row["in_C"], row.get("cells_exact"), row["curve"][-1],
                  flush=True)


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "run":
        run(int(A[2]), A[3], A[4].split(","), [int(x) for x in A[5].split(",")], [float(x) for x in A[6].split(",")],
            float(A[7]), int(A[8]))
