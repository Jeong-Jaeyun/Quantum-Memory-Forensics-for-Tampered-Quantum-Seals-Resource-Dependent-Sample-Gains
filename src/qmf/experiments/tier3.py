"""Tier-3 experiments (E6 vault, E8 per-unit detectors): one tool switch on a 32-cell seal."""
import json
import math
import multiprocessing
import sys

import numpy as np

from qmf.core.channels import depolarize, sample
from qmf.core.pauli import symp, weight
from qmf.core.report import qmf_report, simulate_seal
from qmf.core.stabilizer import span_elements, syndrome_parities, uniform_lagrangian


def switch_trial(job):
    r, eps, p0, t, n = job
    rng = np.random.default_rng([931, r, int(round(eps * 1000)), int(round(p0 * 1e5)), t])
    W = int(10 * r / eps ** 2)
    R = np.sort(rng.choice(n, r, replace=False))
    a1 = int(rng.integers(1, 4 ** r))
    a2 = a1
    while a2 == a1:
        a2 = int(rng.integers(1, 4 ** r))
    s1, s2 = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
    switch = t % 2 == 0
    if not switch:
        a2, s2 = a1, s1
    L0, L1, L2 = int(rng.integers(W // 4, W + 1)), int(rng.integers(W, 2 * W + 1)), int(rng.integers(W, 2 * W + 1))
    codes = simulate_seal(n, R, [(L0, None, 0, 0.0), (L1, a1, s1, eps), (L2, a2, s2, eps)], p0, rng)
    rep = qmf_report(codes, p0, window=W)
    f, l = rep.get("first", {}), rep.get("last", {})
    return dict(r=r, eps=eps, p0=p0, trial=t, switch=switch, W=W, L0=L0, L1=L1, L2=L2,
                cells_exact=rep["cells"] == R.tolist(), onset_err=(rep["onset"] - L0) if "onset" in rep else None,
                first_ok=bool(f.get("a") == a1 and f.get("s") == s1), last_ok=bool(l.get("a") == a2 and l.get("s") == s2),
                switch_detected="switch" in rep, switch_err=(rep["switch"] - L0 - L1) if "switch" in rep else None)


def run_switch(workers, out, rs, epss, p0, trials, n=32):
    jobs = [(r, e, p0, t, n) for r in rs for e in epss for t in range(trials)]
    with multiprocessing.Pool(workers, maxtasksperchild=8) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(switch_trial, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()


# ------------------------------------------------------------------ E6 / E8: switch scenario on shared instances
def switch_instance(r, eps, p0, t, n=32):
    """Seed space 932. Even t: tool switch (a1, s1) -> (a2, s2); odd t: no switch (a2, s2 is then only the second
    registry entry). Segment lengths as in switch_trial."""
    rng = np.random.default_rng([932, r, int(round(eps * 1000)), int(round(p0 * 1e5)), t])
    W = int(10 * r / eps ** 2)
    R = np.sort(rng.choice(n, r, replace=False))
    a1 = int(rng.integers(1, 4 ** r))
    a2 = a1
    while a2 == a1:
        a2 = int(rng.integers(1, 4 ** r))
    s1, s2 = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
    L0, L1, L2 = int(rng.integers(W // 4, W + 1)), int(rng.integers(W, 2 * W + 1)), int(rng.integers(W, 2 * W + 1))
    return dict(W=W, R=R, a1=a1, s1=s1, a2=a2, s2=s2, switch=t % 2 == 0, L0=L0, L1=L1, L2=L2)


def ml_window_fingerprint(c0, c1, r, att):
    """Memoryless fingerprint with unknown eps: per-candidate covered units m_b, Hoeffding threshold, Bonferroni 1%."""
    m, d = c0 + c1, c0 - c1
    z = np.abs(d) / np.sqrt(np.maximum(m, 1))
    z[0] = 0.0
    a = int(np.argmax(z))
    sig = bool(m[a] > 0 and abs(d[a]) > math.sqrt(2 * m[a] * math.log(2 * 4 ** r / 0.01)))
    return dict(a=a, s=int(np.sign(d[a])) or 1, eps=float(abs(d[a]) / max(m[a], 1) / att[a]), significant=sig)


def tail_change_point(gain, N):
    tail = np.concatenate([np.cumsum(gain[::-1])[::-1], [0.0]])  # tail[k] = sum_{u >= k} gain_u
    k = int(np.argmax(tail))
    return (k, float(tail[k])) if tail[k] > 2 * math.log(N) else (None, float(tail[k]))


def switch_trial2(job):
    """job = (mode, r, eps, p0, t, n). mode 'vault': qmf_report on the full seal (as switch_trial). 'ml_blind': uniform
    Lagrangian probes on R (given, with the onset) and unknown eps; fingerprints of the first/last halves after onset,
    change point on the units covering them. 'ml_library': the two registry signatures (a1, s1, eps), (a2, s2, eps) are
    known; units alternate probes containing a1 / a2. Memoryless records carry the seal-half noise only."""
    mode, r, eps, p0, t, n = job
    I = switch_instance(r, eps, p0, t, n)
    b2, t2 = (I["a2"], I["s2"]) if I["switch"] else (I["a1"], I["s1"])
    rng = np.random.default_rng([932, r, int(round(eps * 1000)), int(round(p0 * 1e5)), t, 1 + ["vault", "ml_blind", "ml_library"].index(mode)])
    w1, w2 = int(weight(np.int64(I["a1"]), r)), int(weight(np.int64(b2), r))
    row = dict(mode=mode, r=r, eps=eps, p0=p0, trial=t, switch=I["switch"], W=I["W"], L0=I["L0"], L1=I["L1"], L2=I["L2"],
               att_min=(1 - 4 * p0 / 3) ** (2 * max(w1, w2)))
    if mode == "vault":
        codes = simulate_seal(n, I["R"], [(I["L0"], None, 0, 0.0), (I["L1"], I["a1"], I["s1"], eps), (I["L2"], b2, t2, eps)], p0, rng)
        rep = qmf_report(codes, p0, window=I["W"])
        f, l = rep.get("first", {}), rep.get("last", {})
        return dict(row, cells_exact=rep["cells"] == I["R"].tolist(), onset_err=(rep["onset"] - I["L0"]) if "onset" in rep else None,
                    first_ok=bool(f.get("a") == I["a1"] and f.get("s") == I["s1"]), last_ok=bool(l.get("a") == b2 and l.get("s") == t2),
                    switch_detected="switch" in rep, switch_err=(rep["switch"] - I["L0"] - I["L1"]) if "switch" in rep else None)
    L1, L2 = I["L1"], I["L2"]
    E = np.concatenate([sample("scramble", L1, r, I["a1"], I["s1"], eps, 0.0, rng), sample("scramble", L2, r, b2, t2, eps, 0.0, rng)])
    E = depolarize(E, r, p0, rng)
    N = len(E)
    att = (1 - 4 * p0 / 3) ** weight(np.arange(4 ** r, dtype=np.int64), r)
    if mode == "ml_library":
        cand = [(I["a1"], I["s1"]), (I["a2"], I["s2"])]
        k_of = np.arange(N) % 2  # unit u probes registry entry u % 2
        a_u = np.array([cand[0][0], cand[1][0]], dtype=np.int64)[k_of]
        o = symp(E, a_u, r)
        e = np.array([min(eps * att[c[0]], 1 - 1e-3) for c in cand])[k_of]
        sgn = np.array([c[1] for c in cand])[k_of]
        ll = np.log1p(sgn * e * (1 - 2 * o))  # log-likelihood ratio of "this entry's tool" vs uniform
        gain = np.where(k_of == 1, ll, -ll)  # tool 2 vs tool 1
        k, g = tail_change_point(gain, N)
        return dict(row, first_ok=None, last_ok=None, switch_detected=k is not None, switch_err=(k - L1) if k is not None else None,
                    switch_gain=g)
    gens = uniform_lagrangian(r, rng, N)
    half = N // 2
    c = np.zeros((2, 2 * 4 ** r), dtype=np.int64)
    for lo in range(0, N, 4096):
        hi = min(N, lo + 4096)
        el = span_elements(gens[lo:hi])
        par = syndrome_parities(symp(gens[lo:hi], E[lo:hi, None], r))
        idx = 2 * el[:, 1:] + par[:, 1:]
        u = np.arange(lo, hi)
        for wdw, sel in ((0, u < half), (1, u >= N - half)):
            if sel.any():
                c[wdw] += np.bincount(idx[sel].ravel(), minlength=2 * 4 ** r)
    f = ml_window_fingerprint(c[0, 0::2], c[0, 1::2], r, att)
    l = ml_window_fingerprint(c[1, 0::2], c[1, 1::2], r, att)
    out = dict(row, first_ok=bool(f["a"] == I["a1"] and f["s"] == I["s1"]), last_ok=bool(l["a"] == b2 and l["s"] == t2),
               first_sig=f["significant"], last_sig=l["significant"], switch_detected=False, switch_err=None)
    if f["significant"] and l["significant"] and f["a"] != l["a"]:
        gain = np.zeros(N)
        for lo in range(0, N, 4096):
            hi = min(N, lo + 4096)
            el = span_elements(gens[lo:hi])
            for fp, sign in ((l, 1.0), (f, -1.0)):
                cov = (el == fp["a"]).any(axis=1)
                o = symp(E[lo:hi], np.int64(fp["a"]), r)
                e = min(max(fp["eps"] * att[fp["a"]], 1e-3), 1 - 1e-3)
                gain[lo:hi] += sign * cov * np.log1p(fp["s"] * e * (1 - 2 * o))
        k, g = tail_change_point(gain, N)
        out.update(switch_detected=k is not None, switch_err=(k - L1) if k is not None else None, switch_gain=g)
    return out


def run_switch2(workers, out, modes, rs, epss, p0, trials, n=32):
    jobs = [(m, r, e, p0, t, n) for m in modes for r in rs for e in epss for t in range(trials)]
    with multiprocessing.Pool(workers, maxtasksperchild=8) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(switch_trial2, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "switch2":  # switch2 WORKERS OUT MODES R_LIST EPS_LIST P0 TRIALS
        run_switch2(int(A[2]), A[3], A[4].split(","), [int(x) for x in A[5].split(",")], [float(x) for x in A[6].split(",")],
                    float(A[7]), int(A[8]))
    elif A[1] == "switch":  # switch WORKERS OUT R_LIST EPS_LIST P0 TRIALS
        run_switch(int(A[2]), A[3], [int(x) for x in A[4].split(",")], [float(x) for x in A[5].split(",")],
                   float(A[6]), int(A[7]))
