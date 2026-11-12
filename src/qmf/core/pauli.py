"""Pauli algebra on r qubits: a Pauli is an int in [0, 4^r), bits 0..r-1 = x part, bits r..2r-1 = z part;
symplectic product, weight, fast Walsh-Hadamard transform, random Paulis of a given weight."""
import numpy as np


# ------------------------------------------------------------------ Pauli algebra
def symp(a, b, r):
    m = (1 << r) - 1
    return (np.bitwise_count(((a & m) & (b >> r)) ^ ((a >> r) & (b & m))) & 1).astype(np.int64)  # uint8 otherwise


def weight(a, r):
    m = (1 << r) - 1
    return np.bitwise_count((a & m) | (a >> r)).astype(np.int64)


def swap_xz(a, r):
    m = (1 << r) - 1
    return ((a & m) << r) | (a >> r)


def anticommuting(a, r):
    """A single-qubit Pauli q with <a, q> = 1."""
    a = int(a)  # np.int64 has no bit_length
    m = (1 << r) - 1
    ax, az = a & m, a >> r
    if ax:
        i = (ax & -ax).bit_length() - 1
        return 1 << (r + i)  # Z_i anticommutes with X/Y on i
    i = (az & -az).bit_length() - 1
    return 1 << i  # X_i anticommutes with Z on i


def fwht(h):
    """Unnormalized Walsh-Hadamard transform, W[b] = sum_E h[E] (-1)^(b.E)."""
    h = h.astype(np.float64).copy()
    n, step = len(h), 1
    while step < n:
        h = h.reshape(-1, 2, step)
        h = np.stack([h[:, 0] + h[:, 1], h[:, 0] - h[:, 1]], axis=1).reshape(-1)
        step *= 2
    return h


def pack(X, Z, r):
    sh = np.arange(r, dtype=np.int64)
    return ((X.astype(np.int64) << sh).sum(axis=-1)) | ((Z.astype(np.int64) << (sh + r)).sum(axis=-1))


def random_pauli_weight(r, w, rng):
    """Weight-w Pauli: w uniformly chosen cells, each X/Y/Z uniformly."""
    cells = rng.choice(r, w, replace=False)
    kind = rng.integers(1, 4, w)  # bit 0 = x, bit 1 = z
    x = sum(int(k & 1) << int(c) for c, k in zip(cells, kind))
    z = sum(int(k >> 1) << int(c) for c, k in zip(cells, kind))
    return x | (z << r)


def jaccard(a, b, r):
    m = (1 << r) - 1
    sa, sb = (a & m) | (a >> r), (b & m) | (b >> r)
    u = bin(sa | sb).count("1")
    return bin(sa & sb).count("1") / u if u else 1.0
