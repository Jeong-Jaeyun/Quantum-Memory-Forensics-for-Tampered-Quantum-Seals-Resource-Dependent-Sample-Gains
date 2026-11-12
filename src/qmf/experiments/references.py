"""E9: untouched in-seal cells as references vs the vault.

Full-seal simulation with n = 32 cells, of which
the attacker's tool scrambles R (|R| = r) with the hidden-parity fingerprint (a*, s, eps) of the E1 instances (seed space
941, uniform a*). Benign seal noise: depolarizing p0 on every cell; the vault half has depolarizing p0 too (E1 default).
Cell codes 0=I 1=X 2=Z 3=Y (x bit + 2 z bit); a Bell measurement on a pair returns the XOR of the two cells' codes (the
product Pauli; the secret frame is undone, and the transpose does not change the code).

Labs:
  inseal       no vault. Tier 1: the first T1 units use a fresh, secret, uniformly random perfect matching of the 32 cells
               into in-seal Bell pairs; R^ is read from the pair flags (two-pass rule in tier1_inseal). Later units pair
               every cell of R^ with a distinct, fresh, uniformly random cell outside R^. Decoder: exact Bayes over (a, s)
               on R^, with every unit's coverage (tier-1 units enter with their partial coverage). No decode before T1.
  vault_seal   vault half for every cell. Tier 1 from the first min(n, T1) units: cells with error rate > 0.3, then
               exact copies (code agreement >= 0.8) are merged into one representative. Decoder: exact Bayes on the
               representatives from all n units (law of E1's vault on R when the representatives are R).
  paired       P (cross-unit pairs, sign erased); only under the mirror attacker here (obl/extend leave P's law on R
               unchanged, so pl_paired applies exactly).
  vault_sparse vault with R unknown under the extend attacker (tier 1 cannot localize): MAP over every Pauli of weight
               1..r on the 32 cells with the exact prior of the instance family; r = 4 only (3.06e6 candidates).
Attackers:
  obl     acts on R only (oblivious).
  mirror  knows the in-seal pairing: copies E_i onto the partner of every R cell paired outside R, so that pair reads
          only benign noise ((E x E*)|Phi> = |Phi>). Against the vault (no pairing) it copies onto a fixed set of r cells
          outside R. Against P it mirrors across the two units of every pair.
  extend  fully depolarizes every cell outside R (needs no knowledge of the pairing).
Decoders know eps and p0 (as in E1). N90 rows: curve of {n, correct, a_correct}.

usage: python -m qmf.experiments.references run WORKERS OUT.jsonl LABS ATTACKERS R_LIST EPS_LIST P0 TRIALS
       python -m qmf.experiments.references tier1 R_LIST T1_LIST RUNS        (tier-1-only profile used to fix T1)
"""
import itertools
import json
import math
import multiprocessing
import sys

import numpy as np

import qmf.core as q

N = 32
T1 = 24
R_MAX = 12  # largest region the exact (4^k) decoders handle
LABS = ("inseal", "vault_seal", "paired", "vault_sparse")
ATTACKERS = ("obl", "mirror", "extend")
SYM1 = np.array([[((a & 1) * (o >> 1) + (a >> 1) * (o & 1)) & 1 for o in range(4)] for a in range(4)], dtype=np.int64)


def instance(r, eps, p0, t):
    """E1 instance (trial() with model scramble, wt None): same seed tag, same (a*, s)."""
    tag = [941, 0, r, int(round(eps * 1000)), int(round(p0 * 1e5)), 0, t]
    rng = np.random.default_rng(tag)
    return tag, int(rng.integers(1, 4 ** r)), int(rng.choice([-1, 1]))


def seal_noise(shape, p0, rng):
    return rng.integers(1, 4, shape) * (rng.random(shape) < p0)


def random_matching(B, rng, n=N):
    """mate[u, c]: partner of cell c in unit u under a uniformly random perfect matching."""
    perm = rng.permuted(np.tile(np.arange(n), (B, 1)), axis=1)
    mate = np.empty((B, n), dtype=np.int64)
    rows = np.arange(B)[:, None]
    mate[rows, perm[:, 0::2]] = perm[:, 1::2]
    mate[rows, perm[:, 1::2]] = perm[:, 0::2]
    return mate


def adaptive_matching(B, Rh, rng, n=N):
    """Every cell of Rh paired with a distinct uniformly random cell outside Rh; the rest paired at random."""
    Rh = np.asarray(Rh, dtype=np.int64)
    clean = np.setdiff1d(np.arange(n), Rh)
    k = min(len(Rh), len(clean))
    rows = np.arange(B)[:, None]
    mate = np.empty((B, n), dtype=np.int64)
    cp = rng.permuted(np.tile(clean, (B, 1)), axis=1)
    mate[rows, np.broadcast_to(Rh[:k], (B, k))] = cp[:, :k]
    mate[rows, cp[:, :k]] = Rh[:k]
    rest = rng.permuted(np.concatenate([np.broadcast_to(Rh[k:], (B, len(Rh) - k)), cp[:, k:]], axis=1), axis=1)
    mate[rows, rest[:, 0::2]] = rest[:, 1::2]
    mate[rows, rest[:, 1::2]] = rest[:, 0::2]
    return mate


def tamper(B, R, a, s, eps, attacker, rng, mirror_to=None, n=N):
    """Tamper codes (B, n). mirror_to (B, r): cell that receives a copy of each R cell's Pauli (-1 = none)."""
    r = len(R)
    T = np.zeros((B, n), dtype=np.int64)
    T[:, R] = q.unpack(q.sample("scramble", B, r, a, s, eps, 0.0, rng), r)
    if attacker == "extend":
        out = np.setdiff1d(np.arange(n), R)
        T[:, out] = rng.integers(0, 4, (B, len(out)))
    elif attacker == "mirror" and mirror_to is not None:
        uu, jj = np.nonzero(mirror_to >= 0)
        T[uu, mirror_to[uu, jj]] = T[uu, R[jj]]
    return T


def inseal_outcomes(T, mate, p0, rng):
    X = T ^ seal_noise(T.shape, p0, rng)
    return X ^ X[np.arange(len(T))[:, None], mate]  # pair outcome, stored on both cells of the pair


def mirror_targets(mate, R):
    """Partner of each R cell if it lies outside R, else -1 (pairs inside R are left to the tool)."""
    m = mate[:, R]
    return np.where(np.isin(m, R), -1, m)


def tier1_inseal(O, mate):
    """Two passes. f_c: fraction of tier-1 units whose pair (c, mate) read non-identity; R0 = {f > 1/2}. g_c: the same
    fraction over the units where c's partner is outside R0 (a clean partner flags only through an R cell); R^ = {g > 0.3}."""
    flag = O != 0
    f = flag.mean(axis=0)
    R0 = f > 0.5
    use = ~R0[mate]
    cnt = use.sum(axis=0)
    g = np.where(cnt > 0, (flag & use).sum(axis=0) / np.maximum(cnt, 1), f)
    return np.flatnonzero(g > 0.3)


def tier1_vault(O):
    """Cells with error rate > 0.3; exact copies (agreement >= 0.8) merged into the lowest index. Returns (reps, rep)."""
    flagged = np.flatnonzero((O != 0).mean(axis=0) > 0.3)
    rep = {}
    for i, c in enumerate(flagged):
        c = int(c)
        if c in rep:
            continue
        rep[c] = c
        for d in flagged[i + 1:]:
            d = int(d)
            if d not in rep and (O[:, c] == O[:, d]).mean() >= 0.8:
                rep[d] = c
    return sorted(set(rep.values())), rep


def true_index(acodes, R, cells, rep=None):
    """Packed index of a* on the decoder's cells (rep maps a cell to its representative), or None if not representable."""
    pos = {c: m for m, c in enumerate(cells)}
    codes = np.zeros(len(cells), dtype=np.int64)
    for i, c in zip(R.tolist(), acodes.tolist()):
        if c == 0:
            continue
        j = rep.get(i, i) if rep else i
        if j not in pos:
            return None
        codes[pos[j]] = c
    return int(q.pack_cells(codes[None])[0])


class Full:
    """Exact Bayes from full error samples on k cells (vault law: attenuation (1 - 4p0/3)^(2|a|))."""

    def __init__(self, k, eps, p0):
        self.k, self.allp = k, np.arange(4 ** k, dtype=np.int64)
        e = np.minimum(eps * (1 - 4 * p0 / 3) ** (2 * q.weight(self.allp, k)), 1 - 1e-12)
        self.lp, self.lm = np.log1p(e), np.log1p(-e)
        self.h, self.n = np.zeros(4 ** k, dtype=np.int64), 0

    def add(self, E):
        self.h += np.bincount(E, minlength=4 ** self.k)
        self.n += len(E)

    def score(self):
        if self.n == 0:
            return np.zeros((2, 4 ** self.k))
        lam = q.fwht(self.h)[q.swap_xz(self.allp, self.k)] / self.n
        c0, c1 = self.n * (1 + lam) / 2, self.n * (1 - lam) / 2
        return np.stack([c0 * self.lp + c1 * self.lm, c0 * self.lm + c1 * self.lp])


def partial_score(O, mate, cells, eps, p0):
    """Exact log-likelihoods from in-seal units with arbitrary pairings, on the decoder's cells (assumed the only
    tampered ones). A candidate a is covered by a unit iff a_i = a_j on every pair inside `cells`; its parity is the sum
    of <a_i, outcome> over the observed pairs, attenuated by (1 - 4p0/3)^2 per pair that carries a non-identity."""
    k = len(cells)
    cand = q.unpack(np.arange(4 ** k, dtype=np.int64), k)
    pos = {int(c): m for m, c in enumerate(cells)}
    att = 1 - 4 * p0 / 3
    sc = np.zeros((2, 4 ** k))
    for o, mt in zip(O, mate):
        cover = np.ones(4 ** k, dtype=bool)
        par = np.zeros(4 ** k, dtype=np.int64)
        expo = np.zeros(4 ** k, dtype=np.int64)
        seen = set()
        for m, c in enumerate(cells):
            if m in seen:
                continue
            j = int(mt[c])
            if j in pos:
                seen.add(pos[j])
                cover &= cand[:, m] == cand[:, pos[j]]
            par ^= SYM1[cand[:, m], o[c]]
            expo += 2 * (cand[:, m] != 0)
        e = np.minimum(eps * att ** expo, 1 - 1e-12) * (1 - 2 * par)
        sc[0] += np.where(cover, np.log1p(e), 0.0)
        sc[1] += np.where(cover, np.log1p(-e), 0.0)
    return sc


def evaluate(sc, ti, s, k, rng_eval):
    sc = sc.copy()
    sc[:, 0] = -np.inf
    best = np.flatnonzero(sc == sc.max())
    kk = int(rng_eval.choice(best))
    sh, ah = (1 if kk < 4 ** k else -1), kk % (4 ** k)
    return dict(correct=bool(ti is not None and ah == ti and sh == s), a_correct=bool(ti is not None and ah == ti))


FAIL = dict(correct=False, a_correct=False)


def run_inseal(R, a, s, eps, p0, att, n_max, rng, rng_eval):
    r = len(R)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    mate1 = random_matching(T1, rng)
    T = tamper(T1, R, a, s, eps, att, rng, mirror_targets(mate1, R))
    O1 = inseal_outcomes(T, mate1, p0, rng)
    Rh = tier1_inseal(O1, mate1)
    info = dict(Rhat=Rh.tolist(), cells_exact=Rh.tolist() == R.tolist())
    cps = q.checkpoints(n_max)
    if not 1 <= len(Rh) <= R_MAX or 2 * len(Rh) > N:
        return [dict(n=n, **FAIL) for n in cps], info
    k = len(Rh)
    ti = true_index(acodes, R, Rh.tolist())
    base = partial_score(O1, mate1, Rh, eps, p0)
    dec, done, out = Full(k, eps, p0), T1, []
    for n in cps:
        if n < T1:
            out.append(dict(n=n, **FAIL))
            continue
        B = n - done
        if B:
            mate = adaptive_matching(B, Rh, rng)
            T = tamper(B, R, a, s, eps, att, rng, mirror_targets(mate, R))
            dec.add(q.pack_cells(inseal_outcomes(T, mate, p0, rng)[:, Rh]))
            done = n
        out.append(dict(n=n, **evaluate(base + dec.score(), ti, s, k, rng_eval)))
    return out, info


def run_vault_seal(R, a, s, eps, p0, att, n_max, rng, rng_eval):
    r = len(R)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    pi0 = rng.choice(np.setdiff1d(np.arange(N), R), r, replace=False)  # fixed mirror targets (worst case for tier 1)
    T = tamper(n_max, R, a, s, eps, att, rng, np.broadcast_to(pi0, (n_max, r)))
    O = T ^ seal_noise(T.shape, p0, rng) ^ seal_noise(T.shape, p0, rng)  # seal half + vault half
    out, key, dec, done = [], None, None, 0
    for n in q.checkpoints(n_max):
        if att == "extend":  # tier 1 cannot localize (every cell is at rate 3/4): R is given to the decoder
            reps, rep = R.tolist(), {}
        else:
            reps, rep = tier1_vault(O[:min(n, T1)])
        if not 1 <= len(reps) <= R_MAX:
            out.append(dict(n=n, **FAIL))
            key = None
            continue
        if key != tuple(reps):
            key, dec, done = tuple(reps), Full(len(reps), eps, p0), 0
        ti = true_index(acodes, R, reps, rep)  # the copy map can change while the representatives do not
        dec.add(q.pack_cells(O[done:n][:, reps]))
        done = n
        out.append(dict(n=n, **evaluate(dec.score(), ti, s, len(reps), rng_eval)))
    rep_of = {i: rep.get(i, i) for i in R.tolist()} if att != "extend" else {}
    info = dict(Rhat=list(key) if key else [], cells_exact=bool(key) and sorted(set(rep_of.values()) if rep_of else R.tolist()) == list(key),
                mirror_cells=pi0.tolist() if att == "mirror" else None)
    return out, info


def run_paired_mirror(r, a, eps, p0, rng, rng_eval):
    """P under a cross-unit mirror: each pair's Bell outcome is E_u E_u* = I up to benign noise on both halves."""
    n_s = int(min(60 * r / eps ** 4, 4e6))
    E = q.depolarize(q.depolarize(np.zeros(n_s, dtype=np.int64), r, p0, rng), r, p0, rng)
    allp = np.arange(4 ** r, dtype=np.int64)
    e = np.minimum(eps ** 2 * (1 - 4 * p0 / 3) ** (2 * q.weight(allp, r)), 1 - 1e-12)
    out = []
    for n in q.checkpoints(n_s):
        lam = q.fwht(np.bincount(E[:n], minlength=4 ** r))[q.swap_xz(allp, r)] / n
        sc = np.stack([n * ((1 + lam) / 2 * np.log1p(e) + (1 - lam) / 2 * np.log1p(-e)),
                       n * ((1 + lam) / 2 * np.log1p(-e) + (1 - lam) / 2 * np.log1p(e))])
        p = q.eval_point(sc, a, 1, r, rng_eval)
        out.append(dict(n=2 * n, correct=False, a_correct=p["a_correct"]))
    return out, 2 * n_s


CHI1 = (1 - 2 * SYM1[1:]).astype(np.float64)  # (3, 4): (-1)^<beta, c> for beta = X, Z, Y and cell code c


def full_weight_locals(k):
    """Row order of full_weight_characters: code tuples (c_0, ..., c_{k-1}) with c_{k-1} most significant."""
    return np.array([t[::-1] for t in itertools.product((1, 2, 3), repeat=k)], dtype=np.int64)


def full_weight_characters(h, k):
    """h: (4^k, nS) code histograms of k-subsets (index sum_m c_m 4^m). Returns (3^k, nS) sums of (-1)^<b, e> over the
    histogram for every full-weight b on the subset, rows in full_weight_locals(k) order. The character factorizes over
    the cells, so this is k batched products with the 3x4 matrix CHI1 on contiguous blocks (no full WHT)."""
    nS = h.shape[1]
    X = h.reshape(1, 4, -1)  # axis 1 = digit k-1
    for step in range(k):
        X = np.matmul(CHI1, X)  # (B, 3, rest)
        B, rest = X.shape[0] * 3, X.shape[2]
        X = X.reshape(B, 4, rest // 4) if step < k - 1 else X.reshape(B, rest)
    return X


def run_vault_sparse(R, a, s, eps, p0, n_max, rng, rng_eval):
    """Vault under the extend attacker, R unknown: MAP over (S, b, s), S a k-subset of the 32 cells (k = 1..r), b a
    Pauli of full weight on S, log prior log C(N - k, r - k) (R uniform, a* uniform on R), likelihood from the parity
    counts of b (exact, from each subset's code histogram). Ties among maximizers are broken uniformly (rng_eval), in
    the order (s = +1, then -1; k ascending; candidate index ascending)."""
    r = len(R)
    T = tamper(n_max, R, a, s, eps, "extend", rng)
    O = (T ^ seal_noise(T.shape, p0, rng) ^ seal_noise(T.shape, p0, rng)).astype(np.int64)
    acodes = q.unpack(np.array([a], dtype=np.int64), r)[0]
    true_S = tuple(int(c) for c, x in zip(R, acodes) if x)
    true_b = tuple(int(x) for x in acodes if x)
    att = 1 - 4 * p0 / 3
    parts = []
    for k in range(1, r + 1):
        S = np.array(list(itertools.combinations(range(N), k)), dtype=np.int64)
        loc = full_weight_locals(k)  # full-weight local codes, in the row order of full_weight_characters
        e = min(eps * att ** (2 * k), 1 - 1e-12)
        nS = len(S)
        parts.append(dict(k=k, S=S, loc=loc, h=np.zeros((4 ** k, nS)), cols=np.arange(nS, dtype=np.int64)[None, :],
                          A=0.5 * (math.log1p(e) + math.log1p(-e)), Bc=0.5 * (math.log1p(e) - math.log1p(-e)),
                          prior=math.log(math.comb(N - k, r - k))))
    pairs = np.array(list(itertools.combinations(range(N), 2)), dtype=np.int64)
    pid = np.full((N, N), -1, dtype=np.int64)
    pid[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
    for P in parts:  # local index from pair codes: digits (0, 1) and (2, 3) of the subset
        S = P["S"]
        P["p01"] = pid[S[:, 0], S[:, 1]] if P["k"] >= 2 else None
        P["p23"] = pid[S[:, 2], S[:, 3]] if P["k"] >= 4 else None
    done, out = 0, []
    for n in q.checkpoints(n_max):
        for lo in range(done, n, 512):
            blk = O[lo:min(n, lo + 512)]
            P2 = blk[:, pairs[:, 0]] + 4 * blk[:, pairs[:, 1]]  # (u, 496) pair codes
            for P in parts:
                k, S = P["k"], P["S"]
                if k == 1:
                    idx = blk[:, S[:, 0]]
                elif k == 2:
                    idx = P2[:, P["p01"]]
                elif k == 3:
                    idx = P2[:, P["p01"]] + (blk[:, S[:, 2]] << 4)
                elif k == 4:
                    idx = P2[:, P["p01"]] + (P2[:, P["p23"]] << 4)
                else:
                    idx = blk[:, S[:, 0]].copy()  # (u, nS): local code index
                    for m in range(1, k):
                        idx += blk[:, S[:, m]] << (2 * m)
                flat = (idx * P["h"].shape[1] + P["cols"]).ravel()
                if len(flat) * 8 < P["h"].size:
                    np.add.at(P["h"].reshape(-1), flat, 1.0)
                else:
                    P["h"] += np.bincount(flat, minlength=P["h"].size).reshape(P["h"].shape)
        done = n
        # score(s=+1) = n A + prior + n B lam, score(s=-1) = n A + prior - n B lam (B > 0): the maximizers of each part
        # are its largest (s=+1) or smallest (s=-1) lam
        lams, tops = [], []
        for P in parts:
            lam = full_weight_characters(P["h"], P["k"])  # n * lambda (exact integers in float64)
            base, b = n * P["A"] + P["prior"], P["Bc"]
            lams.append(lam)
            mx, mn = lam.max(), lam.min()
            tops.append(((base + b * mx), mx, (base - b * mn), mn))
        best = max(max(t[0], t[2]) for t in tops)
        ties = []
        for sgn, (vi, xi) in ((1, (0, 1)), (-1, (2, 3))):
            for P, lam, t in zip(parts, lams, tops):
                if t[vi] == best:
                    ties += [(sgn, P, int(j)) for j in np.flatnonzero(lam.ravel() == t[xi])]
        sgn, P, j = ties[int(rng_eval.integers(len(ties)))]
        nS = len(P["S"])
        S_hat, b_hat = tuple(P["S"][j % nS].tolist()), tuple(P["loc"][j // nS].tolist())
        ok_a = S_hat == true_S and b_hat == true_b
        out.append(dict(n=n, correct=bool(ok_a and sgn == s), a_correct=bool(ok_a)))
    return out


def e9_trial(job):
    lab, att, r, eps, p0, t = job
    tag, a, s = instance(r, eps, p0, t)
    sub = [LABS.index(lab), ATTACKERS.index(att)]
    rng, rng_eval = np.random.default_rng(tag + [9] + sub), np.random.default_rng(tag + [10] + sub)
    w = int(q.weight(np.int64(a), r))
    mu_a = float(q.vault_mu(np.int64(a), r, p0, None, "dep"))
    R = np.sort(rng.choice(N, r, replace=False))
    row = dict(lab=lab, attacker=att, r=r, eps=eps, p0=p0, trial=t, a=a, s=s, weight=w, mu_a=mu_a, R=R.tolist(), T1=T1)
    if lab == "paired":
        curve, n_max = run_paired_mirror(r, a, eps, p0, rng, rng_eval)
        return dict(row, n_max=n_max, curve=curve)
    n_max = q.n_max_for("scramble", "memory", r, eps, w, mu_a, p0)  # E1 vault horizon
    if lab == "vault_sparse":
        return dict(row, n_max=n_max, curve=run_vault_sparse(R, a, s, eps, p0, n_max, rng, rng_eval))
    n_max *= 2  # in-seal and full-seal vault: twice the E1 vault horizon (room for the tier-1 cost)
    fn = run_inseal if lab == "inseal" else run_vault_seal
    curve, info = fn(R, a, s, eps, p0, att, n_max, rng, rng_eval)
    return dict(row, n_max=n_max, **info, curve=curve)


def run(workers, out, labs, atts, rs, epss, p0, trials):
    jobs = [(lab, at, r, e, p0, t) for lab in labs for at in atts for r in rs for e in epss for t in range(trials)]
    jobs.sort(key=lambda j: -(4 ** j[2]) / j[3] ** 2)
    with multiprocessing.Pool(workers, maxtasksperchild=8) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(e9_trial, jobs):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(row["lab"], row["attacker"], row["r"], row["eps"], row["trial"], row.get("cells_exact"),
                  row["curve"][-1], flush=True)


def tier1_profile(rs, t1s, runs, p0=0.005, eps=0.6):
    """Tier-1-only profile (no fingerprint decoding): fraction of runs with R^ = R, obl attacker, seed space 9410."""
    res = {}
    for r in rs:
        for t1 in t1s:
            ok = 0
            for i in range(runs):
                rng = np.random.default_rng([9410, r, t1, i])
                R = np.sort(rng.choice(N, r, replace=False))
                a = int(rng.integers(1, 4 ** r))
                mate = random_matching(t1, rng)
                O = inseal_outcomes(tamper(t1, R, a, 1, eps, "obl", rng), mate, p0, rng)
                ok += tier1_inseal(O, mate).tolist() == R.tolist()
            res[f"r={r},T1={t1}"] = ok / runs
            print(f"r={r} T1={t1}: R^ exact {ok}/{runs}", flush=True)
    return res


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "run":
        run(int(A[2]), A[3], A[4].split(","), A[5].split(","), [int(x) for x in A[6].split(",")],
            [float(x) for x in A[7].split(",")], float(A[8]), int(A[9]))
    elif A[1] == "tier1":
        tier1_profile([int(x) for x in A[2].split(",")], [int(x) for x in A[3].split(",")], int(A[4]))
