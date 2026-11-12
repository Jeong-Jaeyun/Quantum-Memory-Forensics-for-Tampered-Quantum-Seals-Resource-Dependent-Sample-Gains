"""The laboratories behind every N90 run: vault (exact law, MAP by fast WHT), memory/paired (explicit
sampling) and per-unit stabilizer probes, with their checkpoints and horizons."""
import math

import numpy as np

from qmf.core.channels import depolarize, sample, vault_counts, vault_mu
from qmf.core.pauli import fwht, jaccard, swap_xz, symp, weight
from qmf.core.stabilizer import full_weight, ghz_covering, ghz_frames, ghz_stabilizer, lagrangian_containing, product_stabilizer, span_elements, syndrome_parities, uniform_lagrangian


BATCH = 64


# ------------------------------------------------------------------ inference
def checkpoints(n_max):
    return sorted(set(int(round(x)) for x in np.geomspace(4, n_max, 40)))


def eval_point(score2, a, s, r, rng):
    """score2: (2, 4^r) log-likelihood for s = +1, -1 (index 0 = identity excluded). MAP correctness and quality."""
    sc = score2.copy()
    sc[:, 0] = -np.inf
    best = np.flatnonzero(sc == sc.max())
    k = int(rng.choice(best))
    sh, ah = (1 if k < 4 ** r else -1), k % (4 ** r)
    flat = sc.reshape(-1)
    z = flat - flat.max()
    post = np.exp(z) / np.exp(z).sum()
    true_idx = (0 if s == 1 else 4 ** r) + a
    return dict(correct=bool(ah == a and sh == s), a_correct=bool(ah == a), jaccard=jaccard(ah, a, r),
                post_true=float(post[true_idx]))


def run_vault(r, a, s, eps, p0, pv, vkind, n_max, rng, rng_eval, points=None):
    """Vault lab (scramble model): exact outcome law, counts drawn incrementally between checkpoints; MAP over (a, s)
    by fast WHT with the decoder knowing the noise (mu for every candidate). points: checkpoint list (E14: one list per
    group); None = checkpoints(n_max), as every earlier experiment."""
    allp = np.arange(4 ** r, dtype=np.int64)
    mu = vault_mu(allp, r, p0, pv, vkind)
    e = np.minimum(eps * mu, 1 - 1e-12)
    lp, lm = np.log1p(e), np.log1p(-e)
    h, done, out = np.zeros(4 ** r, dtype=np.int64), 0, []
    for n in (checkpoints(n_max) if points is None else points):
        h += vault_counts(r, a, s, eps * mu[a], n - done, rng)
        done = n
        lam = fwht(h)[swap_xz(allp, r)] / n  # exact: integer sums below 2^53
        c0, c1 = n * (1 + lam) / 2, n * (1 - lam) / 2  # counts of parity 0 / 1 for every candidate
        out.append(dict(n=n, **eval_point(np.stack([c0 * lp + c1 * lm, c0 * lm + c1 * lp]), a, s, r, rng_eval)))
    return out


def run_memory(model, r, a, s, eps, q, p0, n_max, rng, rng_eval, paired=False):
    """Explicit-sampling memory lab (used for the paired lab and the local model; the scramble vault uses run_vault)."""
    E = sample(model, n_max, r, a, s, eps, q, rng)
    if paired:  # self-referenced pairs (no vault): Bell measurement across two attacked units gives E_u E_u'
        E ^= sample(model, n_max, r, a, s, eps, q, rng)
        eps, s = eps ** 2, 1  # parity bias (s eps)^2, sign erased
    E = depolarize(depolarize(E, r, p0, rng), r, p0, rng)  # two halves (seal + vault, or seal + seal)
    allp = np.arange(4 ** r, dtype=np.int64)
    att = (1 - 4 * p0 / 3) ** (2 * weight(allp, r))
    out = []
    for n in checkpoints(n_max):
        h = np.bincount(E[:n], minlength=4 ** r)
        if model == "scramble":
            lam = fwht(h)[swap_xz(allp, r)] / n
            e = np.minimum(eps * att, 1 - 1e-12)
            sc = np.stack([n * ((1 + lam) / 2 * np.log1p(e) + (1 - lam) / 2 * np.log1p(-e)),
                           n * ((1 + lam) / 2 * np.log1p(-e) + (1 - lam) / 2 * np.log1p(e))])
        else:  # local, noise-free: E_u in {0, a}; any other nonzero value excludes a
            nz = h.copy(); nz[0] = 0
            sc1 = np.where(nz == nz.sum(), nz * np.log(q) + h[0] * np.log1p(-q), -np.inf) if nz.sum() else \
                np.full(4 ** r, h[0] * np.log1p(-q))
            sc = np.stack([sc1, np.full(4 ** r, -np.inf)])
        out.append(dict(n=n * (2 if paired else 1), **eval_point(sc, a, s if model == "scramble" else 1, r, rng_eval)))
    return out


def probe_generators(method, r, rng, B, score_fn=None, done=0):
    if method == "ml_product":
        return product_stabilizer(r, rng, B)
    if method == "ml_ghz":  # best Lagrangian probe for the weight-r (fan-out) family
        return ghz_stabilizer(r, rng, B)
    if method == "ml_adaptive_ghz":  # explore/exploit (E5): half random GHZ frames, half frames through a top-4r
        Mi = rng.integers(0, 6, (B, r))  # weight-r candidate, rescored every batch
        if done > 0:
            sm = score_fn().max(axis=0)
            cand = full_weight(r)
            k = min(4 * r, len(cand) - 1)
            # top-k by a stable sort: ties go to the lower index, so the set and its order do not depend on the numpy
            # version (np.argpartition's output order did; E5b)
            top = cand[np.argsort(-sm[cand], kind="stable")[:k]]
            for u in range(B // 2):
                Mi[u] = ghz_covering(int(top[rng.integers(k)]), r, rng)
        return ghz_frames(Mi, r)
    if method == "ml_mix":  # each unit: product or uniform Lagrangian with probability 1/2
        pick = rng.random(B) < 0.5
        gens = np.empty((B, r), dtype=np.int64)
        gens[pick] = product_stabilizer(r, rng, int(pick.sum()))
        gens[~pick] = uniform_lagrangian(r, rng, int((~pick).sum()))
        return gens
    if method == "ml_adaptive" and done > 0:  # heuristic: one Lagrangian through the top-4r candidates per batch
        sm = score_fn().max(axis=0)
        part = np.argpartition(-sm, 4 * r)[:4 * r]
        top = part[np.argsort(-sm[part])]
        return np.broadcast_to(lagrangian_containing(top, r, rng), (B, r)).copy()
    return uniform_lagrangian(r, rng, B)  # ml_random, and the first ml_adaptive batch


def run_memoryless(model, method, r, a, s, eps, q, p0, n_max, rng, rng_eval, points=None):
    """Memoryless lab. Scramble model: integer sufficient statistics cnt[2b + parity] (units whose probe group holds b,
    split by the measured parity of b); log-likelihoods are formed only at checkpoints, so results do not depend on
    floating-point accumulation order. Non-adaptive probes are processed in chunks up to the next checkpoint.
    points: checkpoint list (E14); None = checkpoints(n_max), as every earlier experiment."""
    allp = np.arange(4 ** r, dtype=np.int64)
    e_eff = np.minimum(eps * (1 - 4 * p0 / 3) ** weight(allp, r), 1 - 1e-12)
    lp, lm = np.log1p(e_eff), np.log1p(-e_eff)
    cnt = np.zeros(2 * 4 ** r, dtype=np.int64)
    score = np.zeros((2, 4 ** r))  # local model only

    def scr():
        if model != "scramble":
            return score
        c0, c1 = cnt[0::2], cnt[1::2]
        return np.stack([c0 * lp + c1 * lm, c0 * lm + c1 * lp])

    chunk = BATCH if (method.startswith("ml_adaptive") or model != "scramble") else max(BATCH, 2 ** 23 // 2 ** r)
    cps, out, done = (checkpoints(n_max) if points is None else list(points)), [], 0
    while done < n_max:
        B = min(chunk, n_max - done, (cps[0] - done) if cps else chunk)  # never batch past a checkpoint
        gens = probe_generators(method, r, rng, B, scr, done)
        E = depolarize(sample(model, B, r, a, s, eps, q, rng), r, p0, rng)
        sig = symp(gens, E[:, None], r)  # (B, r)
        el = span_elements(gens)
        if model == "scramble":
            cnt += np.bincount((2 * el[:, 1:] + syndrome_parities(sig)[:, 1:]).ravel(), minlength=2 * 4 ** r)
        else:  # local, noise-free: sigma = sigma(a) w.p. q else 0, for every hypothesis a
            for g, sg in zip(gens, sig):
                sa = np.stack([symp(allp, np.int64(gj), r) for gj in g], axis=1)  # (4^r, r)
                match = (sa == sg).all(axis=1)
                zero_a = (sa == 0).all(axis=1)
                if sg.any():
                    score[0] += np.where(match, math.log(q), -np.inf)
                else:
                    score[0] += np.where(zero_a, 0.0, math.log1p(-q))
            score[1] = -np.inf
        done += B
        while cps and cps[0] <= done:
            out.append(dict(n=cps.pop(0), **eval_point(scr(), a, s if model == "scramble" else 1, r, rng_eval)))
    return out


METHODS = ["memory", "ml_random", "ml_adaptive", "ml_product", "paired", "ml_mix", "ml_ghz", "ml_adaptive_ghz"]
CAP = 6e6


def n_max_for(model, method, r, eps, w, mu_a, p0, floor=0.0):
    """Trial horizon (units). w = weight of a*, mu_a = vault noise eigenvalue at a* (memory lab). floor (memory lab
    only): horizon at least floor * r * 2^r / eps^2, so that every trial of a group reaches the memoryless scale."""
    if model == "local":
        return 400
    if method == "memory":  # capped at 1e12 so counts stay exact in float64 sums
        return int(min(max(80 * r / (eps * mu_a) ** 2, floor * r * 2 ** r / eps ** 2), 1e12))
    if method == "paired":  # samples (pairs); units = 2 x samples
        return int(min(60 * r / eps ** 4, 4e6))
    m2 = (1 - 4 * p0 / 3) ** (2 * w)  # memoryless attenuation^2 at a*
    if method == "ml_product":
        return int(min(120 * 3 ** w / (eps ** 2 * m2) + 200, CAP))
    if method == "ml_mix":
        return int(min(240 * min(3 ** w, 2 ** r) / (eps ** 2 * m2) + 200, CAP))
    if method in ("ml_ghz", "ml_adaptive_ghz"):
        cov =(2 ** (r - 1) + (r % 2 == 0)) / 3 ** r if w == r else (w % 2 == 0) / 3 ** w
        return int(min(240 / (cov * eps ** 2 * m2) + 200, CAP)) if cov else 1000
    return int(min(120 * 2 ** r / (eps ** 2 * m2), CAP))  # ml_random, ml_adaptive
