"""Tamper channels (scramble: hidden-parity fingerprint; local) and benign noise; the vault's exact
outcome law (vault_mu, vault_counts)."""
import numpy as np

from qmf.core.pauli import anticommuting, symp, weight


def depolarize(E, r, p, rng):
    if p <= 0:
        return E
    hit = rng.random((len(E), r)) < p
    kind = rng.integers(1, 4, (len(E), r)) * hit
    sh = np.arange(r, dtype=np.int64)
    noise = (((kind & 1) << sh).sum(axis=1)) | (((kind >> 1) << (sh + r)).sum(axis=1))
    return E ^ noise


# ------------------------------------------------------------------ tamper sampling
def sample(model, n_units, r, a, s, eps, q, rng):
    if model == "scramble":
        E = rng.integers(0, 4 ** r, n_units, dtype=np.int64)
        want = (rng.random(n_units) >= (1 + s * eps) / 2).astype(np.int64)  # P(parity 0) = (1 + s eps)/2
        E[symp(E, a, r) != want] ^= anticommuting(a, r)
        return E
    return np.where(rng.random(n_units) < q, a, 0).astype(np.int64)


def vault_mu(b, r, p0, pv, vkind):
    """Pauli eigenvalue of the benign noise seen by the vault lab at Pauli(s) b: seal half depolarizing p0, vault half
    depolarizing ('dep', default pv = p0) or dephasing ('deph': Z flip with probability pv)."""
    m = (1 << r) - 1
    pv = p0 if pv is None else pv
    seal = (1 - 4 * p0 / 3) ** weight(b, r)
    vault = (1 - 4 * pv / 3) ** weight(b, r) if vkind == "dep" else (1 - 2 * pv) ** np.bitwise_count(b & m).astype(np.int64)
    return seal * vault


def vault_counts(r, a, s, e, n, rng):
    """Histogram over E of n vault Bell outcomes from the exact law p(E) = 4^-r (1 + s e (-1)^<a,E>) (e = eps mu(a))."""
    allp = np.arange(4 ** r, dtype=np.int64)
    par = symp(allp, np.int64(a), r)
    h = np.zeros(4 ** r, dtype=np.int64)
    k0 = rng.binomial(n, (1 + s * e) / 2)
    for cls, k in ((par == 0, k0), (par == 1, n - k0)):
        L = int(cls.sum())  # uniform within each parity class
        h[cls] = np.bincount(rng.integers(0, L, k), minlength=L) if k < 4 * L else rng.multinomial(k, np.full(L, 1.0 / L))
    return h
