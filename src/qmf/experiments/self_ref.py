"""E13: self-referenced pairs with a sign phase (P + sign).

Resource class: self-referencing. At issue,
cell c of unit u is Bell-paired with cell c of unit u'; at verification the two units are measured together. No vault.

Phase 1 (P, as pl_paired): each pair of units gives E_u E_u' on the r cells (bias eps^2, sign erased); exact Bayes for
a* (E1 instances, R given, eps and p0 known, as E1's vault). The rows are E1's P rows (qmf.core.trial with
method 'paired', the same seeds), so phase 1 is identical to pl_paired.jsonl.
Phase 2 (sign): with the MAP a^ of phase 1, m single units are each prepared in the +1 eigenstate of a^'s local Paulis
and read in those bases. The product of the outcomes is (-1)^<a^, E> (seal-half noise: attenuation (1 - 4p0/3)^|a^|),
so its mean is s eps (1 - 4p0/3)^|a*| if a^ = a*. s^ = sign of the sum (ties: a fair coin). m is fixed per group from
the worst-case weight r: m = ceil(2 ln(100) / (eps (1 - 4p0/3)^r)^2) (Hoeffding: sign error <= 0.01).
Cost: a pair counts as 2 units; the curve point of a phase-1 checkpoint at n units is n + m, correct iff a^ = a* and
s^ = s. If a^ != a*, the sign phase measures the wrong Pauli and the trial fails whatever s^ is, so one sign draw per
trial serves every checkpoint (the success fraction at each checkpoint is the same as with fresh draws).
Seeds: phase 1 as pl_paired; sign draws default_rng(tag + [15]) with E1's instance tag.

paired_sign_bound: extend with R unknown, P + sign, information-theoretic (union bound over the M Paulis of weight 1..r
on the n cells with Bhattacharyya coefficient <= 1 - eps_P^2 / 8, eps_P = eps^2 (1 - 4p0/3)^(2r); then Hoeffding for
the sign). Error <= 0.09 + 0.01 on every instance. An upper bound on samples, not an efficient decoder.

usage: python -m qmf.experiments.self_ref run WORKERS OUT.jsonl R_LIST EPS_LIST P0_LIST TRIALS
"""
import json
import math
import multiprocessing
import sys

import numpy as np

import qmf.experiments.uniform as uniform

N = 32


def sign_units(r, eps, p0, delta=0.01):
    """Sign-phase units: Hoeffding, worst-case weight r."""
    es = eps * (1 - 4 * p0 / 3) ** r
    return math.ceil(2 * math.log(1 / delta) / es ** 2)


def tag_of(r, eps, p0, t):
    return [941, 0, r, int(round(eps * 1000)), int(round(p0 * 1e5)), 0, t]


def e13_trial(job):
    r, eps, p0, t = job
    P = uniform.trial(("scramble", "paired", r, eps, p0, t, None, None, "dep"))
    m = sign_units(r, eps, p0)
    rng = np.random.default_rng(tag_of(r, eps, p0, t) + [15])
    es = eps * (1 - 4 * p0 / 3) ** P["weight"]
    k = int(rng.binomial(m, (1 + P["s"] * es) / 2))
    s_hat = 1 if 2 * k > m else -1 if 2 * k < m else int(rng.choice([-1, 1]))
    ok = s_hat == P["s"]
    curve = [dict(n=c["n"] + m, a_correct=c["a_correct"], correct=bool(c["a_correct"] and ok)) for c in P["curve"]]
    return dict(lab="paired_sign", r=r, eps=eps, p0=p0, trial=t, a=P["a"], s=P["s"], weight=P["weight"],
                pairs_max=P["n_max"], m=m, sign_plus=k, sign_ok=ok, n_max=2 * P["n_max"] + m, curve=curve)


def paired_sign_bound(r, eps, p0, n=N, delta_a=0.09, delta_s=0.01):
    """Extend, R unknown: units that suffice information-theoretically for P + sign (error <= delta_a + delta_s on
    every instance). P's law on the n cells is 4^-n (1 + eps_P,a chi_a), eps_P,a = eps^2 (1 - 4p0/3)^(2|a|) (outside R
    uniform), with no sign; ML over the M Paulis of weight 1..r errs with probability <= M (1 - eps_P^2 / 8)^pairs."""
    M = sum(math.comb(n, k) * 3 ** k for k in range(1, r + 1))
    ep = eps ** 2 * (1 - 4 * p0 / 3) ** (2 * r)
    pairs = math.ceil(8 * math.log(M / delta_a) / ep ** 2)
    m = sign_units(r, eps, p0, delta_s)
    return dict(M=M, eps_pair_min=ep, pairs=pairs, sign_units=m, units=2 * pairs + m)


def run(workers, out, rs, epss, p0s, trials):
    jobs = [(r, e, p, t) for r in rs for e in epss for p in p0s for t in range(trials)]
    jobs.sort(key=lambda j: -(4 ** j[0]) / j[1] ** 4)
    with multiprocessing.Pool(workers, maxtasksperchild=4) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(e13_trial, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["r"], row["eps"], row["p0"], row["trial"], row["m"], row["sign_ok"], row["curve"][-1], flush=True)


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "run":
        run(int(A[2]), A[3], [int(x) for x in A[4].split(",")], [float(x) for x in A[5].split(",")],
            [float(x) for x in A[6].split(",")], int(A[7]))
