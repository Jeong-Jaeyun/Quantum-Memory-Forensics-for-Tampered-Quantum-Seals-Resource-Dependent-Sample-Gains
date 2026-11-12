"""N90 runs of the labs in qmf.core.labs: E1 (uniform fingerprint), E2 (weight), E3 (vault noise),
E5 (adaptive GHZ), P and local tampering."""
import json
import math
import multiprocessing
import sys

import numpy as np

from qmf.core.channels import vault_mu
from qmf.core.labs import METHODS, n_max_for, run_memory, run_memoryless, run_vault
from qmf.core.pauli import random_pauli_weight, weight


def trial(job):
    """job = (model, method, r, eps, p0, t, wt, pv, vkind[, floor]). wt: None = a* uniform over 4^r - 1, else |a*| = wt.
    pv: vault noise (None = p0); vkind 'dep' | 'deph'; floor: see n_max_for. (a*, s) depend only on
    (model, r, eps, p0, wt, t), so all labs and vault-noise settings see the same instance. Separate RNGs for data and
    for MAP tie-breaking."""
    model, method, r, eps, p0, t, wt, pv, vkind = job[:9]
    floor = job[9] if len(job) > 9 else 0.0
    tag = [941, ["scramble", "local"].index(model), r, int(round(eps * 1000)), int(round(p0 * 1e5)),
           0 if wt is None else wt + 1, t]  # seed entries must be non-negative
    rng = np.random.default_rng(tag)
    a = random_pauli_weight(r, wt, rng) if wt is not None else int(rng.integers(1, 4 ** r))
    s = int(rng.choice([-1, 1]))
    q = eps  # the local model reuses eps as its tamper probability q
    w = int(weight(np.int64(a), r))
    sub = [METHODS.index(method), 0 if pv is None else int(round(pv * 1e5)) + 1, ["dep", "deph"].index(vkind)]
    rng_data, rng_eval = np.random.default_rng(tag + [1] + sub), np.random.default_rng(tag + [2] + sub)
    mu_a = float(vault_mu(np.int64(a), r, p0, pv, vkind))
    n_max = n_max_for(model, method, r, eps, w, mu_a, p0, floor)
    if method == "memory" and model == "scramble":
        curve = run_vault(r, a, s, eps, p0, pv, vkind, n_max, rng_data, rng_eval)
    elif method in ("memory", "paired"):
        curve = run_memory(model, r, a, s, eps, q, p0, n_max, rng_data, rng_eval, paired=method == "paired")
    else:
        curve = run_memoryless(model, method, r, a, s, eps, q, p0, n_max, rng_data, rng_eval)
    return dict(model=model, method=method, r=r, eps=eps, p0=p0, trial=t, wt=wt, pv=pv, vkind=vkind, a=a, s=s,
                weight=w, mu_a=mu_a, n_max=n_max, floor=floor, curve=curve)


def run(workers, out, model, methods, rs, epss, p0s, trials, wts=("u",), pvs=(None,), vkind="dep", floor=0.0):
    def wt_of(x, r):
        return None if x == "u" else r if x == "r" else math.ceil(r / 2) if x == "h" else int(x)
    jobs = [(model, m, r, e, p, t, wt_of(x, r), pv, vkind, floor) for m in methods for r in rs for e in epss
            for p in p0s for x in wts for pv in pvs for t in range(trials)]
    jobs = list(dict.fromkeys(jobs))  # e.g. w = ceil(r/2) coinciding with w = 3
    jobs.sort(key=lambda j: -(2 ** j[2]) / j[3] ** 2 * (1 if j[1] == "memory" else 2 ** j[2]))
    with multiprocessing.Pool(workers, maxtasksperchild=4) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(trial, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            last = row["curve"][-1]
            print(row["model"], row["method"], row["r"], row["eps"], row["p0"], row["wt"], row["pv"], row["vkind"],
                  row["trial"], last["n"], last["correct"], flush=True)


if __name__ == "__main__":
    # run WORKERS OUT MODEL METHODS R_LIST EPS_LIST P0_LIST TRIALS [WT_LIST] [PV_LIST] [VKIND] [FLOOR]
    A = sys.argv
    if A[1] == "run":
        run(int(A[2]), A[3], A[4], A[5].split(","), [int(x) for x in A[6].split(",")],
            [float(x) for x in A[7].split(",")], [float(x) for x in A[8].split(",")], int(A[9]),
            A[10].split(",") if len(A) > 10 else ("u",),
            [None if x == "-" else float(x) for x in A[11].split(",")] if len(A) > 11 else (None,),
            A[12] if len(A) > 12 else "dep", float(A[13]) if len(A) > 13 else 0.0)
