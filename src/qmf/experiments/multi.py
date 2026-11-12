"""E7: several hidden parities (K <= 3) with eps and K unknown to the decoder."""
import json
import math
import multiprocessing
import sys

import numpy as np

from qmf.core.channels import depolarize, sample, vault_counts, vault_mu
from qmf.core.labs import BATCH, checkpoints
from qmf.core.pauli import fwht, swap_xz, symp, weight
from qmf.core.stabilizer import span_elements, syndrome_parities, uniform_lagrangian


# ------------------------------------------------------------------ E7: several hidden parities, eps unknown
def multi_instance(r, K, eps_tot, p0, t):
    rng = np.random.default_rng([951, r, K, int(round(eps_tot * 1000)), int(round(p0 * 1e5)), t])
    A = []
    while len(A) < K:
        a = int(rng.integers(1, 4 ** r))
        if a not in A:
            A.append(a)
    return A, [int(x) for x in rng.choice([-1, 1], K)], eps_tot / K


def sample_multi(n, r, A, S, ek, rng):
    """p(E) = 4^-r (1 + sum_k S_k ek chi_{A_k}(E)) as an equal mixture of single parities with bias K ek."""
    comp = rng.integers(0, len(A), n)
    E = np.zeros(n, dtype=np.int64)
    for k, (a, s) in enumerate(zip(A, S)):
        sel = comp == k
        E[sel] = sample("scramble", int(sel.sum()), r, a, s, len(A) * ek, 0.0, rng)
    return E


def threshold_correct(sig, d, A, S):
    return bool(set(sig.tolist()) == set(A) and all(np.sign(d[a]) == s for a, s in zip(A, S)))


def run_vault_multi(r, A, S, ek, p0, n_max, rng):
    """Vault, exact law (seal + vault depolarizing p0), eps/K/noise unknown to the decoder: significance threshold."""
    allp = np.arange(4 ** r, dtype=np.int64)
    e = np.array([ek * float(vault_mu(np.int64(a), r, p0, None, "dep")) for a in A])
    Lb = math.log(2 * 4 ** r / 0.01)
    h, done, out = np.zeros(4 ** r, dtype=np.int64), 0, []
    for n in checkpoints(n_max):
        nk = rng.multinomial(n - done, e / e.sum())
        for a, s, m in zip(A, S, nk):
            h += vault_counts(r, a, s, e.sum(), int(m), rng)
        done = n
        lam = fwht(h)[swap_xz(allp, r)] / n
        lam[0] = 0.0
        sig = np.flatnonzero(np.abs(lam) > math.sqrt(2 * Lb / n))
        out.append(dict(n=n, correct=threshold_correct(sig, lam, A, S), n_sig=int(len(sig)),
                        found=int(len(set(sig.tolist()) & set(A)))))
    return out


def run_memoryless_multi(r, A, S, ek, p0, n_max, rng):
    """Uniform Lagrangian probes, seal-half noise p0, eps/K unknown: Hoeffding threshold per candidate."""
    Lb = math.log(2 * 4 ** r / 0.01)
    cnt = np.zeros(2 * 4 ** r, dtype=np.int64)
    cps, out, done = checkpoints(n_max), [], 0
    chunk = max(BATCH, 2 ** 23 // 2 ** r)
    while done < n_max:
        B = min(chunk, n_max - done, (cps[0] - done) if cps else chunk)
        gens = uniform_lagrangian(r, rng, B)
        E = depolarize(sample_multi(B, r, A, S, ek, rng), r, p0, rng)
        el = span_elements(gens)
        cnt += np.bincount((2 * el[:, 1:] + syndrome_parities(symp(gens, E[:, None], r))[:, 1:]).ravel(), minlength=2 * 4 ** r)
        done += B
        while cps and cps[0] <= done:
            m, d = cnt[0::2] + cnt[1::2], cnt[0::2] - cnt[1::2]
            ok = np.abs(d) > np.sqrt(2 * m * Lb)
            ok[0] = False
            sig = np.flatnonzero(ok & (m > 0))
            out.append(dict(n=cps.pop(0), correct=threshold_correct(sig, d, A, S), n_sig=int(len(sig)),
                            found=int(len(set(sig.tolist()) & set(A)))))
    return out


def multi_trial(job):
    lab, r, K, eps_tot, p0, t = job
    A, S, ek = multi_instance(r, K, eps_tot, p0, t)
    rng = np.random.default_rng([951, r, K, int(round(eps_tot * 1000)), int(round(p0 * 1e5)), t, 1 + (lab == "ml_random")])
    Lb = math.log(2 * 4 ** r / 0.01)
    if lab == "memory":
        att = min(float(vault_mu(np.int64(a), r, p0, None, "dep")) for a in A)
        n_max = int(min(40 * Lb / (ek * att) ** 2, 1e12))
        curve = run_vault_multi(r, A, S, ek, p0, n_max, rng)
    else:
        att = min((1 - 4 * p0 / 3) ** int(weight(np.int64(a), r)) for a in A)
        n_max = int(min(8 * (2 ** r + 1) * Lb / (ek * att) ** 2, 1e7))
        curve = run_memoryless_multi(r, A, S, ek, p0, n_max, rng)
    return dict(lab=lab, r=r, K=K, eps_tot=eps_tot, p0=p0, trial=t, A=A, S=S, n_max=n_max, curve=curve)


def run_multi(workers, out, labs, rs, Ks, epss, p0, trials):
    """trials: {lab: {r: count}} or an int."""
    jobs = [(lab, r, K, e, p0, t) for lab in labs for r in rs for K in Ks for e in epss
            for t in range(trials[lab].get(r, 0) if isinstance(trials, dict) else trials)]
    jobs.sort(key=lambda j: -(2 ** j[1] if j[0] == "ml_random" else 1) * j[2] ** 2 / j[3] ** 2)
    with multiprocessing.Pool(workers, maxtasksperchild=4) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(multi_trial, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()


if __name__ == "__main__":
    # multi WORKERS OUT LAB R_LIST K_LIST EPS_TOT_LIST P0 TRIALS_BY_R (e.g. 4:50,6:50,8:30)
    A = sys.argv
    if A[1] == "multi":
        run_multi(int(A[2]), A[3], [A[4]], [int(x) for x in A[5].split(",")], [int(x) for x in A[6].split(",")],
                  [float(x) for x in A[7].split(",")], float(A[8]),
                  {A[4]: {int(k): int(v) for k, v in (p.split(":") for p in A[9].split(","))}})
