"""E10: coverage limits of non-adaptive stabilizer probes. The rules were fixed before the run.

A memoryless unit that probes a Lagrangian L (stabilizer group up to signs) learns the parity of a* only if a* is in L;
otherwise its syndrome is uniform. For a fingerprint a* drawn uniformly from the weight-w Paulis W_w (|W_w| =
C(r,w) 3^w), the best coverage of any non-adaptive stabilizer probe is c*(r, w) = max_L N_w(L) / |W_w|, where N_w(L) is
the number of weight-w elements of L (local Cliffords and qubit permutations make the worst case equal the average).
Fano's inequality then bounds the units any such verifier needs to find (a*, s) with probability 0.9:
    N >= (0.9 log2(2 |W_w|) - 1) / (c* d(eps')),  d(e) = 1 - h2((1 + e)/2) bits,  eps' = eps (1 - 4 p0/3)^w.

This module computes max_L N_w(L):
  - exactly, by enumerating every Lagrangian of F_2^{2r} (r <= 6) through the parametrization L = {(x, z): x in V,
    z.v = B(x, v) for v in V} (V = X-part projection in RREF, B symmetric on V), and every labeled graph state (r <= 8;
    every Lagrangian is local-Clifford equivalent to a graph state, and local Cliffords preserve weight);
  - as an upper bound for any r, by an integer program over weight enumerators (MacWilliams self-duality, shadow
    nonnegativity and integrality, 0 <= N_k <= C(r,k) 3^k, N_0 = 1, N_1 <= r), checked against every enumerated one;
  - as a lower bound for r = 9, 10, by GHZ, product and a best-improvement search over graph states.

usage: python -m qmf.experiments.coverage enum R_LAGR R_GRAPH WORKERS OUT.json
       python -m qmf.experiments.coverage bound ENUM.json OUT.json
"""
import itertools
import json
import math
import multiprocessing
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import qmf.core as q


# ------------------------------------------------------------------ weight enumerators
def enumerators(gens, r):
    """gens (B, r) packed generators -> (B, r+1) weight enumerators N_0..N_r of the spanned groups."""
    el = q.span_elements(gens)
    W = q.weight(el, r)
    B = len(gens)
    return np.bincount((W + (r + 1) * np.arange(B)[:, None]).ravel(), minlength=B * (r + 1)).reshape(B, r + 1)


def merge(counter, enums):
    u, c = np.unique(enums, axis=0, return_counts=True)
    for row, k in zip(u.tolist(), c.tolist()):
        counter[tuple(row)] += k


# ------------------------------------------------------------------ all Lagrangians (r <= 6)
def rref_subspaces(r, k):
    """Yield (pivots, rows): every k-dim subspace of F_2^r once, as RREF row vectors (bit c = column c). Row i has a 1 at
    pivot p_i, 0 at the other pivots and at columns below p_i, and free bits at non-pivot columns above p_i."""
    for piv in itertools.combinations(range(r), k):
        free = [[c for c in range(p + 1, r) if c not in piv] for p in piv]
        slots = [(i, c) for i, F in enumerate(free) for c in F]
        for bits in range(1 << len(slots)):
            rows = [1 << p for p in piv]
            for m, (i, c) in enumerate(slots):
                if (bits >> m) & 1:
                    rows[i] |= 1 << c
            yield piv, rows


def lagrangian_generators(r, piv, rows, t):
    """Generators (len(t), r) of the Lagrangians with X-projection V = span(rows) and symmetric forms B encoded by t
    (bit m of t = B_ij for the m-th pair i <= j). z_i = sum_j B_ij e_{p_j} solves z_i . v_j = B_ij; V^perp is spanned
    by y_c = e_c + sum_i [row_i has c] e_{p_i} over non-pivot columns c."""
    k = len(piv)
    t = np.asarray(t, dtype=np.int64)
    G = np.zeros((len(t), r), dtype=np.int64)
    z = np.zeros((len(t), k), dtype=np.int64)
    for m, (i, j) in enumerate((i, j) for i in range(k) for j in range(i, k)):
        b = (t >> m) & 1
        z[:, i] |= b << piv[j]
        if i != j:
            z[:, j] |= b << piv[i]
    for i in range(k):
        G[:, i] = rows[i] | (z[:, i] << r)
    for n, c in enumerate(c for c in range(r) if c not in piv):
        y = (1 << c) | sum(1 << piv[i] for i in range(k) if (rows[i] >> c) & 1)
        G[:, k + n] = y << r
    return G


def all_lagrangians(r, chunk=1 << 16):
    """Yield generator chunks covering every Lagrangian of F_2^{2r} exactly once."""
    for k in range(r + 1):
        nb = 1 << (k * (k + 1) // 2)
        for piv, rows in rref_subspaces(r, k):
            for lo in range(0, nb, chunk):
                yield lagrangian_generators(r, piv, rows, np.arange(lo, min(nb, lo + chunk)))


def n_lagrangians(r):
    return math.prod(2 ** i + 1 for i in range(1, r + 1))


def enumerate_lagrangians(r, check_distinct=False):
    cnt, total, seen = Counter(), 0, set()
    for G in all_lagrangians(r):
        merge(cnt, enumerators(G, r))
        total += len(G)
        if check_distinct:
            for el in np.sort(q.span_elements(G), axis=1):
                seen.add(el.tobytes())
    out = dict(r=r, total=total, expected=n_lagrangians(r), enumerators={json.dumps(k): v for k, v in cnt.items()})
    if check_distinct:
        out["distinct"] = len(seen)
    return out


# ------------------------------------------------------------------ all labeled graph states (r <= 8)
def graph_generators(r, t):
    """Graph-state generators X_i Z_{N(i)} for the graphs encoded by t (bit m = m-th edge of combinations(r, 2))."""
    t = np.asarray(t, dtype=np.int64)
    adj = np.zeros((len(t), r), dtype=np.int64)
    for m, (i, j) in enumerate(itertools.combinations(range(r), 2)):
        b = (t >> m) & 1
        adj[:, i] |= b << j
        adj[:, j] |= b << i
    return (1 << np.arange(r, dtype=np.int64))[None, :] | (adj << r)


def _graph_chunk(args):
    r, lo, hi = args
    cnt = Counter()
    for a in range(lo, hi, 1 << 15):
        merge(cnt, enumerators(graph_generators(r, np.arange(a, min(hi, a + (1 << 15)))), r))
    return cnt


def enumerate_graphs(r, workers=1):
    m = r * (r - 1) // 2
    total = 1 << m
    step = max(1 << 15, total // 512)
    jobs = [(r, lo, min(total, lo + step)) for lo in range(0, total, step)]
    cnt = Counter()
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            for c in pool.imap_unordered(_graph_chunk, jobs):
                cnt.update(c)
    else:
        for j in jobs:
            cnt.update(_graph_chunk(j))
    return dict(r=r, total=total, enumerators={json.dumps(k): v for k, v in cnt.items()})


# ------------------------------------------------------------------ linear constraints on weight enumerators
def krawtchouk(r):
    """K[j, k] = sum_i (-1)^i 3^(j-i) C(k, i) C(r-k, j-i) (quaternary Krawtchouk)."""
    K = np.zeros((r + 1, r + 1), dtype=object)
    for j in range(r + 1):
        for k in range(r + 1):
            K[j, k] = sum((-1) ** i * 3 ** (j - i) * math.comb(k, i) * math.comb(r - k, j - i) for i in range(j + 1))
    return K


def constraint_report(N, r, K=None):
    """MacWilliams residual (W(x+3y, x-y)/2^r - W), shadow coefficients W(x+3y, y-x)/2^r, and the simple bounds."""
    K = krawtchouk(r) if K is None else K
    N = [int(x) for x in N]
    mac = [sum(K[j, k] * N[k] for k in range(r + 1)) - 2 ** r * N[j] for j in range(r + 1)]
    sh = [sum((-1) ** k * K[j, k] * N[k] for k in range(r + 1)) for j in range(r + 1)]
    return dict(macwilliams_ok=all(x == 0 for x in mac),
                shadow_ok=all(x >= 0 and x % 2 ** r == 0 for x in sh),
                simple_ok=N[0] == 1 and (r < 1 or N[1] <= r) and all(0 <= N[k] <= math.comb(r, k) * 3 ** k for k in range(r + 1)),
                shadow=[x // 2 ** r for x in sh])


def ilp_max(r, w, shadow=True):
    """max N_w over integer weight enumerators that satisfy the constraints above (scipy milp / HiGHS)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    K = krawtchouk(r)
    n = r + 1
    nv = 2 * n if shadow else n
    A, lo, hi = [], [], []
    for j in range(n):  # sum_k K_jk N_k - 2^r N_j = 0
        row = np.zeros(nv)
        row[:n] = [float(K[j, k]) for k in range(n)]
        row[j] -= 2 ** r
        A.append(row); lo.append(0); hi.append(0)
    if shadow:
        for j in range(n):  # sum_k (-1)^k K_jk N_k - 2^r S_j = 0, S_j >= 0 integer
            row = np.zeros(nv)
            row[:n] = [float((-1) ** k * K[j, k]) for k in range(n)]
            row[n + j] = -2 ** r
            A.append(row); lo.append(0); hi.append(0)
    ub = [math.comb(r, k) * 3 ** k for k in range(n)]
    lb_v = [1] + [0] * r + ([0] * n if shadow else [])
    ub_v = [1] + [min(ub[1], r) if r >= 1 else 0] + ub[2:] + (ub if shadow else [])
    c = np.zeros(nv)
    c[w] = -1
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(lb_v, ub_v), options=dict(presolve=True))
    if not res.success:
        return None
    return dict(max=int(round(-res.fun)), enumerator=[int(round(x)) for x in res.x[:n]])


# ------------------------------------------------------------------ constructions and graph search (lower bounds)
SEARCH_RESTARTS = 2000


def ghz_count(r):
    return 2 ** (r - 1) + (r % 2 == 0)


def search_max(r, w, seed, restarts=200):
    """Best-improvement local search over graph states for max N_w: every step evaluates all single-edge flips at once.
    Restarts from the empty graph (product), the star (GHZ) and random graphs; returns the best N_w and its graph."""
    rng = np.random.default_rng([97, r, w, seed])
    edges = list(itertools.combinations(range(r), 2))
    m = len(edges)

    def nw(T):
        return enumerators(graph_generators(r, T), r)[:, w]

    flip = np.array([1 << e for e in range(m)], dtype=np.int64)
    star = sum(1 << e for e, (i, j) in enumerate(edges) if i == 0)
    starts = [0, star] + [int(x) for x in rng.integers(0, 1 << m, restarts, dtype=np.int64)]
    best, best_t = -1, None
    for t in starts:
        cur = int(nw(np.array([t]))[0])
        while True:
            cand = t ^ flip
            v = nw(cand)
            j = int(np.argmax(v))
            if v[j] <= cur:
                break
            t, cur = int(cand[j]), int(v[j])
        if cur > best:
            best, best_t = cur, t
    return dict(max=best, graph=best_t)


# ------------------------------------------------------------------ Fano bound
def d_bits(e):
    p = (1 + e) / 2
    return 1 - (-(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0)


def fano_units(M, c, e):
    """Units for success 0.9 over X = (a*, s) uniform on 2M values when each unit carries <= c d(e) bits."""
    return (0.9 * math.log2(2 * M) - 1) / (c * d_bits(e))


# ------------------------------------------------------------------ commands
def cmd_enum(r_lagr, r_graph, workers, out):
    res = dict(lagrangians={}, graphs={})
    for r in range(1, r_lagr + 1):
        x = enumerate_lagrangians(r, check_distinct=r <= 5)
        res["lagrangians"][str(r)] = x
        print(f"lagrangians r={r}: {x['total']} (expected {x['expected']}, distinct {x.get('distinct')}), "
              f"{len(x['enumerators'])} enumerators", flush=True)
    for r in range(1, r_graph + 1):
        x = enumerate_graphs(r, workers if r >= 7 else 1)
        res["graphs"][str(r)] = x
        print(f"graphs r={r}: {x['total']}, {len(x['enumerators'])} enumerators", flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


def exact_max(enum_block):
    E = np.array([json.loads(k) for k in enum_block["enumerators"]])
    return E.max(axis=0).tolist(), E


def cmd_bound(enum_path, out, r_max=10):
    with open(enum_path) as fh:
        en = json.load(fh)
    res = dict(per_r={}, checks={})
    exact = {}
    for kind in ("lagrangians", "graphs"):
        for rs, blk in en[kind].items():
            r = int(rs)
            mx, E = exact_max(blk)
            K = krawtchouk(r)
            bad = [row.tolist() for row in E if not all(v for k, v in constraint_report(row, r, K).items() if k != "shadow")]
            res["checks"][f"{kind},r={r}"] = dict(n_enumerators=len(E), violations=bad[:5], n_violations=len(bad),
                                                 count_ok=(blk["total"] == blk.get("expected", blk["total"])),
                                                 distinct_ok=blk.get("distinct", blk["total"]) == blk["total"])
            prev = exact.get(r)
            if prev is not None and prev["set"] != set(map(tuple, E.tolist())):
                res["checks"][f"{kind},r={r}"]["set_differs_from_lagrangians"] = True
            exact[r] = dict(max=mx, set=set(map(tuple, E.tolist())), source=kind)
    for r in range(1, r_max + 1):
        row = dict(r=r, w={})
        for w in range(1, r + 1):
            M = math.comb(r, w) * 3 ** w
            ilp = ilp_max(r, w)
            d = dict(M=M, ilp_upper=ilp and ilp["max"], ilp_enumerator=ilp and ilp["enumerator"], product=math.comb(r, w),
                     ghz=ghz_count(r) if w == r else None, kernel_upper=3 * 2 ** (r - 2) if w == r and r >= 2 else None)
            if r in exact:
                d["exact"] = exact[r]["max"][w]
                d["exact_source"] = exact[r]["source"]
            else:
                s = search_max(r, w, seed=0, restarts=SEARCH_RESTARTS)
                d["search_lower"], d["search_graph"] = s["max"], s["graph"]
            lower = d.get("exact", d.get("search_lower"))
            upper = d.get("exact", d["ilp_upper"])
            d["c_lower"], d["c_upper"] = lower / M, upper / M
            d["ghz_optimal"] = (None if w != r else (d.get("exact") == d["ghz"] if "exact" in d else
                                                     (True if d["ilp_upper"] == d["ghz"] else None)))
            row["w"][str(w)] = d
            print(f"r={r} w={w}: exact {d.get('exact')} search {d.get('search_lower')} ILP {d['ilp_upper']} "
                  f"product {d['product']} ghz {d['ghz']}", flush=True)
        res["per_r"][str(r)] = row
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "enum":
        cmd_enum(int(A[2]), int(A[3]), int(A[4]), A[5])
    elif A[1] == "bound":
        cmd_bound(A[2], A[3])
