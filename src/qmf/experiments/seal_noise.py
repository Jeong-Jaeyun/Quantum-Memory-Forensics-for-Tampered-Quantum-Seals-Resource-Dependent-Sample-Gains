"""E14: seal-noise sweep. The rules were fixed before the run.

Question: how does the cost of attribution grow with the benign noise p0 of the seal cells, for the vault and for the
per-unit strategy, with the vault's own noise held at p_v = 0.005? The answer is a seal requirement curve: a seal cell of
coherence time tau_s that is held for T reaches p0 = (3/4)(1 - exp(-T/tau_s)), and N90(p0) is the attribution cost.

- Instances: E1's instances at p0 = 0.005 (seed tag 941 with the p0 entry 500), so every p0 and both labs see the same
  (a*, s) for trial t, and the p0 = 0.005 groups are E1's instances again (control against E1f).
- Labs: vault (qmf.core.labs.run_vault, seal depolarizing p0, vault depolarizing p_v = 0.005) and the per-unit uniform
  Lagrangian probe (qmf.core.labs.run_memoryless, method ml_random, seal depolarizing p0). Decoders know eps and the noise.
- Checkpoints: one list per group (lab, r, eps, p0), shared by every trial of the group, so N90 is read without the
  stale-checkpoint bias of per-trial grids: geometric with ratio <= 1.03 from lo = N_b / 2 to hi = 2 N_b A_r(p0), where
  N_b is E1f's fine N90 of (lab, r, eps, p0 = 0.005) and A_w(p0) = ((1 - 4(0.005)/3) / (1 - 4 p0 / 3))^(2w) is the
  predicted cost factor of a weight-w fingerprint. A trial's horizon is hi.
- Seeds: data default_rng([961, lab, r, eps*1000, p0*1e5, t, 1]), MAP tie-breaks [..., 2].

usage: python -m qmf.experiments.seal_noise run WORKERS OUT.jsonl LAB R_LIST EPS_LIST P0_LIST TRIALS E1F.json
       (LAB: vault | ml)
"""
import json
import math
import multiprocessing
import sys

import numpy as np

import qmf.core as q

LABS = ("vault", "ml")
P0_REF, PV, STEP = 0.005, 0.005, 1.03


def instance(r, eps, t):
    """(a*, s) of E1's trial t at p0 = 0.005 (qmf.experiments.uniform.trial, scramble model, a* uniform)."""
    rng = np.random.default_rng([941, 0, r, int(round(eps * 1000)), int(round(P0_REF * 1e5)), 0, t])
    a = int(rng.integers(1, 4 ** r))
    return a, int(rng.choice([-1, 1]))


def factor(w, p0):
    """Predicted N90 factor at seal noise p0 relative to p0 = 0.005 for a weight-w fingerprint (both labs: the seal
    attenuation (1 - 4 p0 / 3)^w enters the bias, N90 ~ bias^-2; the vault's own noise is fixed)."""
    return ((1 - 4 * P0_REF / 3) / (1 - 4 * p0 / 3)) ** (2 * w)


def points(n_base, r, p0):
    lo, hi = n_base / 2, 2 * n_base * factor(r, p0)
    k = math.ceil(math.log(hi / lo) / math.log(STEP)) + 1
    return [int(x) for x in np.unique(np.round(np.geomspace(lo, hi, k)).astype(np.int64))]


def e14_trial(job):
    lab, r, eps, p0, t, pts, n_base = job
    a, s = instance(r, eps, t)
    tag = [961, LABS.index(lab), r, int(round(eps * 1000)), int(round(p0 * 1e5)), t]
    rng, rng_eval = np.random.default_rng(tag + [1]), np.random.default_rng(tag + [2])
    if lab == "vault":
        curve = q.run_vault(r, a, s, eps, p0, PV, "dep", pts[-1], rng, rng_eval, points=pts)
    else:
        curve = q.run_memoryless("scramble", "ml_random", r, a, s, eps, eps, p0, pts[-1], rng, rng_eval, points=pts)
    if [c["n"] for c in curve] != list(pts):
        raise RuntimeError(f"checkpoints missing: {lab} r={r} eps={eps} p0={p0} t={t}")
    return dict(lab=lab, r=r, eps=eps, p0=p0, pv=PV if lab == "vault" else None, trial=t, a=a, s=s,
                weight=int(q.weight(np.int64(a), r)), n_base=n_base, lo=pts[0], hi=pts[-1], n_max=pts[-1],
                curve=[dict(n=c["n"], correct=c["correct"], a_correct=c["a_correct"], post_true=c["post_true"])
                       for c in curve])


def bases(e1f_path):
    """E1f's fine N90 at p0 = 0.005, per (lab, r, eps)."""
    return {(x["lab"], x["r"], x["eps"]): x["fine"]["n90"] for x in json.load(open(e1f_path))["table"]
            if x["p0"] == P0_REF}


def run(workers, out, lab, rs, epss, p0s, trials, e1f_path):
    B = bases(e1f_path)
    jobs = []
    for r in rs:
        for e in epss:
            nb = B[(lab, r, e)]
            for p in p0s:
                pts = points(nb, r, p)
                jobs += [(lab, r, e, p, t, pts, nb) for t in range(trials)]
    jobs.sort(key=lambda j: -(4 ** j[1] if j[0] == "vault" else 2 ** j[1]) * len(j[5]) * (1 if j[0] == "vault" else j[5][-1]))
    with multiprocessing.Pool(workers, maxtasksperchild=4) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(e14_trial, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            last = row["curve"][-1]
            print(row["lab"], row["r"], row["eps"], row["p0"], row["trial"], len(row["curve"]), last["n"],
                  last["correct"], flush=True)


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "run":
        run(int(A[2]), A[3], A[4], [int(x) for x in A[5].split(",")], [float(x) for x in A[6].split(",")],
            [float(x) for x in A[7].split(",")], int(A[8]), A[9])
