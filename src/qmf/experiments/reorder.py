"""E15 (post hoc): tier 3 when transit reorders the items.

E6's vault records are regenerated from their seeds exactly as switch_trial2('vault') draws them. Before the report reads
a record, a fraction f of its units (positions drawn uniformly) is permuted uniformly among those positions; f = 1 is a
full random permutation. The reordering acts on the whole record (benign stretch, tool 1, tool 2), as transit delays
would. f = 0 reproduces E6's stored rows (the control).

- Seeds: data as E6 (default_rng([932, r, eps*1000, p0*1e5, t, 1])); reordering default_rng([935, r, eps*1000, p0*1e5,
  t, f*1000]).
- Row fields: E6's vault fields (switch_err against the attack-order switch L0 + L1, onset_err against L0) plus f, the
  number of units moved, and tool2_first / tool1_last: the fraction of tool-2 units in the first W units after the true
  onset and of tool-1 units in the last W units, in arrival order (how far the reordering mixed the two windows).
"""
import json
import multiprocessing

import numpy as np

from qmf.core.pauli import weight
from qmf.core.report import qmf_report, simulate_seal
from qmf.experiments.tier3 import switch_instance


def reorder_trial(job):
    f, r, eps, p0, t, n = job
    I = switch_instance(r, eps, p0, t, n)
    b2, t2 = (I["a2"], I["s2"]) if I["switch"] else (I["a1"], I["s1"])
    rng = np.random.default_rng([932, r, int(round(eps * 1000)), int(round(p0 * 1e5)), t, 1])
    L0, L1, L2, W = I["L0"], I["L1"], I["L2"], I["W"]
    codes = simulate_seal(n, I["R"], [(L0, None, 0, 0.0), (L1, I["a1"], I["s1"], eps), (L2, b2, t2, eps)], p0, rng)
    N = len(codes)
    order = np.arange(N)
    k = int(round(f * N))
    if k > 1:
        rs = np.random.default_rng([935, r, int(round(eps * 1000)), int(round(p0 * 1e5)), t, int(round(f * 1000))])
        pos = np.sort(rs.choice(N, k, replace=False))
        order[pos] = rs.permutation(pos)
    codes = codes[order]
    seg = np.repeat(np.array([0, 1, 2]), [L0, L1, L2])[order]  # 0 benign, 1 tool 1, 2 tool 2, in arrival order
    rep = qmf_report(codes, p0, window=W)
    fp, lp = rep.get("first", {}), rep.get("last", {})
    w1, w2 = int(weight(np.int64(I["a1"]), r)), int(weight(np.int64(b2), r))
    att_min = (1 - 4 * p0 / 3) ** (2 * max(w1, w2))
    return dict(f=f, moved=k if k > 1 else 0, r=r, eps=eps, p0=p0, trial=t, switch=I["switch"], W=W, L0=L0, L1=L1,
                L2=L2, att_min=att_min, cells_exact=rep["cells"] == I["R"].tolist(),
                onset_err=(rep["onset"] - L0) if "onset" in rep else None,
                first_ok=bool(fp.get("a") == I["a1"] and fp.get("s") == I["s1"]),
                last_ok=bool(lp.get("a") == b2 and lp.get("s") == t2),
                switch_detected="switch" in rep, switch_err=(rep["switch"] - L0 - L1) if "switch" in rep else None,
                tool2_first=float(np.mean(seg[L0:L0 + W] == 2)), tool1_last=float(np.mean(seg[N - W:] == 1)))


def run(workers, out, fs, rs, epss, p0, trials, n=32):
    jobs = [(f, r, e, p0, t, n) for f in fs for r in rs for e in epss for t in range(trials)]
    with multiprocessing.Pool(workers, maxtasksperchild=8) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(reorder_trial, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()
