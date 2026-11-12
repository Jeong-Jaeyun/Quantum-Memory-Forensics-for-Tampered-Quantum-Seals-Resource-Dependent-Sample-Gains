"""E12: a product probe with no reference against the spray attacker.

The model, the instances, R and the spray
groups are E11's (src/qmf/experiments/spray.py): n = 32 cells, E1 instances (seed space 941, uniform a*), benign
depolarizing p0 on every cell, cell codes 0=I 1=X 2=Z 3=Y (x bit + 2 z bit).

Why spray does not blind a verifier without references. Spray copies E_i onto every cell of group i. A single-qubit
probe measured in the Z basis reads the x bit of its cell's Pauli, one measured in the X basis reads the z bit. Two
copies of the same component, one read each way, give the whole E_i in every unit. No vault, no in-seal Bell pair, no
pairing and no frame are needed. Chen's lower bound does not apply: it is a bound for the tool's channel on R, and the
attacker's copies turn the 32-cell channel into an easy one.

Lab (each trial: E11's instance, R and groups; its own data):
  product  Stage 1 (the first T1 = 24 units): every cell is measured in the Z basis. All 32 cells are grouped: a cell
           joins the lowest-index representative whose x bits agree with its own on >= 0.8 of the units, else it
           becomes a representative (E9's tier-1 merge rule on x bits, with no rate flag: under spray every cell carries
           a component). Stage-1 units serve only for grouping. Stage 2 (every later unit): in each group two distinct
           cells are drawn uniformly at random; one is measured in the Z basis (x bit) and the other in the X basis
           (z bit). Decoder: exact Bayes over (a, s) on the groups, uniform prior. Each bit is flipped by its own cell's
           depolarizing noise with probability 2 p0 / 3, so the parity of b is attenuated by (1 - 4 p0 / 3)^bits(b),
           bits(b) = number of set bits of b (x part + z part, summed over cells). No decode at n <= T1. The trial fails
           at every checkpoint if there are more than R_MAX groups or a group has fewer than 2 cells.
Decoders know eps and p0 (as in E1). Scoring is E11's vault scoring: a* is mapped onto the group representatives
(e9.true_index with the group map), so neither the vault nor this lab must tell which copy is the original.
Rows: curve of {n, correct, a_correct}; in_C; Rhat (representatives), n_groups, groups_exact.

usage: python -m qmf.experiments.product run WORKERS OUT.jsonl R_LIST EPS_LIST P0 TRIALS
"""
import json
import math
import multiprocessing
import sys

import numpy as np

import qmf.core as q
import qmf.experiments.references as e9
import qmf.experiments.spray as e11

N, T1 = e9.N, e9.T1
LABS = ("product",)
AGREE = 0.8


def tier1_product(xb):
    """xb (T, n): stage-1 x bits. Every cell joins the lowest-index representative it agrees with on >= AGREE of the
    units, else it becomes one. Returns (reps, rep) as e9.tier1_vault does."""
    reps, rep = [], {}
    for c in range(xb.shape[1]):
        for d in reps:
            if (xb[:, c] == xb[:, d]).mean() >= AGREE:
                rep[c] = d
                break
        else:
            rep[c] = c
            reps.append(c)
    return reps, rep


def groups_of(reps, rep, n=N):
    return [np.array([c for c in range(n) if rep[c] == d], dtype=np.int64) for d in reps]


def groups_exact(reps, rep, g, r, n=N):
    """The found partition is the spray partition (cells grouped by g)."""
    lab = np.array([rep[c] for c in range(n)])
    return len(reps) == r and all(len(set(lab[g == i].tolist())) == 1 for i in range(r))


def product_estimates(X, groups, rng):
    """X (B, n): codes after benign noise. For every unit and group, two distinct uniformly random cells: x bit from the
    first (Z basis), z bit from the second (X basis). Returns codes (B, k)."""
    B = len(X)
    rows = np.arange(B)
    out = np.empty((B, len(groups)), dtype=np.int64)
    for k, G in enumerate(groups):
        m = len(G)
        i = rng.integers(m, size=B)
        j = (i + 1 + rng.integers(m - 1, size=B)) % m
        out[:, k] = (X[rows, G[i]] & 1) | (X[rows, G[j]] & 2)
    return out


class ProductFull(e9.Full):
    """Exact Bayes from product-probe estimates: attenuation (1 - 4 p0 / 3)^bits(b) instead of the vault's
    (1 - 4 p0 / 3)^(2 |b|)."""

    def __init__(self, k, eps, p0):
        super().__init__(k, eps, p0)
        e = np.minimum(eps * (1 - 4 * p0 / 3) ** np.bitwise_count(self.allp).astype(np.int64), 1 - 1e-12)
        self.lp, self.lm = np.log1p(e), np.log1p(-e)


def run_product(R, a, s, eps, p0, g, n_max, rng, rng_eval):
    r = len(R)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    X1 = e11.spray_tamper(T1, R, a, s, eps, g, rng) ^ e9.seal_noise((T1, N), p0, rng)
    reps, rep = tier1_product(X1 & 1)
    groups = groups_of(reps, rep)
    info = dict(Rhat=reps, n_groups=len(reps), groups_exact=groups_exact(reps, rep, g, r))
    cps = q.checkpoints(n_max)
    if not 1 <= len(reps) <= e9.R_MAX or min(len(G) for G in groups) < 2:
        return [dict(n=n, **e9.FAIL) for n in cps], info
    k = len(reps)
    ti = e9.true_index(acodes, R, reps, rep)
    dec, done, out = ProductFull(k, eps, p0), T1, []
    for n in cps:
        if n <= T1:
            out.append(dict(n=n, **e9.FAIL))
            continue
        B = n - done
        if B:
            X = e11.spray_tamper(B, R, a, s, eps, g, rng) ^ e9.seal_noise((B, N), p0, rng)
            dec.add(q.pack_cells(product_estimates(X, groups, rng)))
            done = n
        out.append(dict(n=n, **e9.evaluate(dec.score(), ti, s, k, rng_eval)))
    return out, info


def unknown_R_bound(r, eps, p0, n=N, delta=0.1):
    """Extend, R unknown: units that suffice information-theoretically for the vault to name (a*, s) with error <= delta
    on every instance. ML over the sign and the M = sum_{k=1..r} C(n, k) 3^k Paulis of weight 1..r on the n cells (the
    vault law on all n cells is 4^-n (1 + s eps_a chi_a)). The Bhattacharyya coefficient of two hypotheses is
    g(eps_a) g(eps_b) (a != b, g(x) = (sqrt(1 + x) + sqrt(1 - x)) / 2 <= 1 - x^2 / 8) or sqrt(1 - eps_a^2) (a = b), so
    at most 1 - eps_min^2 / 8 with eps_min = eps (1 - 4 p0 / 3)^(2 r). Union bound: N >= 8 ln(2 M / delta) / eps_min^2.
    An upper bound on samples, not an efficient decoder (exact MAP is run at r = 4 only: E9 vault_sparse)."""
    M = sum(math.comb(n, k) * 3 ** k for k in range(1, r + 1))
    em = eps * (1 - 4 * p0 / 3) ** (2 * r)
    return dict(M=M, eps_min=em, units=math.ceil(8 * math.log(2 * M / delta) / em ** 2))


def e12_trial(job):
    lab, r, eps, p0, t = job
    tag, a, s = e9.instance(r, eps, p0, t)
    geo = np.random.default_rng(tag + [11])  # E11's seal: the same R and groups
    R = np.sort(geo.choice(N, r, replace=False))
    g = e11.spray_groups(R, geo)
    li = LABS.index(lab)
    rng, rng_eval = np.random.default_rng(tag + [13, li]), np.random.default_rng(tag + [14, li])
    w = int(q.weight(np.int64(a), r))
    mu_a = float(q.vault_mu(np.int64(a), r, p0, None, "dep"))
    n_max = 2 * q.n_max_for("scramble", "memory", r, eps, w, mu_a, p0)  # E11's horizon: the same checkpoints
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    row = dict(lab=lab, attacker="spray", r=r, eps=eps, p0=p0, trial=t, a=a, s=s, weight=w, mu_a=mu_a, R=R.tolist(),
               groups=g.tolist(), in_C=e11.in_C(acodes), T1=T1, n_max=n_max)
    curve, info = run_product(R, a, s, eps, p0, g, n_max, rng, rng_eval)
    return dict(row, **info, curve=curve)


def run(workers, out, rs, epss, p0, trials):
    jobs = [(lab, r, e, p0, t) for lab in LABS for r in rs for e in epss for t in range(trials)]
    jobs.sort(key=lambda j: -(4 ** j[1]) / j[2] ** 2)
    with multiprocessing.Pool(workers, maxtasksperchild=8) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(e12_trial, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["lab"], row["r"], row["eps"], row["trial"], row["in_C"], row["groups_exact"], row["curve"][-1],
                  flush=True)


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "run":
        run(int(A[2]), A[3], [int(x) for x in A[4].split(",")], [float(x) for x in A[5].split(",")], float(A[6]),
            int(A[7]))
