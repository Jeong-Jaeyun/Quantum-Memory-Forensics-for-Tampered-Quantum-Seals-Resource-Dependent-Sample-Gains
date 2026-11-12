"""The QMF report (qmf_report: tiers 1-3, status codes) and the full-seal simulator it reads."""
import math

import numpy as np

from qmf.core.channels import sample
from qmf.core.pauli import fwht, swap_xz, symp, weight


# ------------------------------------------------------------------ QMF algorithm on a full seal (tiers 1-3)
def unpack(E, r):
    """Packed r-qubit Paulis -> per-cell codes 0=I 1=X 2=Z 3=Y (x bit + 2 z bit)."""
    sh = np.arange(r, dtype=np.int64)
    return ((E[:, None] >> sh) & 1) + 2 * ((E[:, None] >> (sh + r)) & 1)


def pack_cells(codes):
    r = codes.shape[1]
    sh = np.arange(r, dtype=np.int64)
    c = codes.astype(np.int64)
    return ((c & 1) << sh).sum(axis=1) | (((c >> 1) << sh).sum(axis=1) << r)


def simulate_seal(n, R, segments, p0, rng):
    """Vault Bell-measurement records (units in time order x n cells, codes 0..3).
    segments: list of (units, a, s, eps); a=None is an untampered stretch. R: tampered cells (sorted)."""
    total = sum(m for m, *_ in segments)
    codes = np.zeros((total, n), dtype=np.int64)
    row = 0
    for m, a, s, eps in segments:
        if a is not None:
            codes[row:row + m][:, R] = unpack(sample("scramble", m, len(R), a, s, eps, 0.0, rng), len(R))
        row += m
    for _ in range(2):  # benign noise on the seal half and on the vault half (XOR of codes = Pauli product)
        codes ^= rng.integers(1, 4, codes.shape) * (rng.random(codes.shape) < p0)
    return codes


def bernoulli_change_point(y):
    """MLE single change point k (y[:k] | y[k:]) of a 0/1 sequence, rates fitted on each side."""
    n = len(y)
    c = np.concatenate([[0], np.cumsum(y)])
    k = np.arange(1, n)

    def ll(ones, m):
        p = np.clip(ones / m, 1e-12, 1 - 1e-12)
        return ones * np.log(p) + (m - ones) * np.log1p(-p)
    return int(k[np.argmax(ll(c[k], k) + ll(c[n] - c[k], n - k))])


def fingerprint(E, r, att):
    """Profile-ML hidden-parity fingerprint from packed samples: a = argmax_a |lambda_a|, sign, strength, significance
    (Bonferroni 1% over the 4^r - 1 candidates)."""
    n = len(E)
    allp = np.arange(4 ** r, dtype=np.int64)
    lam = fwht(np.bincount(E, minlength=4 ** r))[swap_xz(allp, r)] / n
    lam[0] = 0.0
    a = int(np.argmax(np.abs(lam)))
    thr = math.sqrt(2 * math.log(2 * 4 ** r / 0.01) / n)
    return dict(a=a, s=int(np.sign(lam[a])), eps=float(abs(lam[a]) / att[a]), z=float(abs(lam[a]) * math.sqrt(n)),
                significant=bool(abs(lam[a]) > thr))


def benign_rate(p0):
    """Probability that a vault Bell outcome on an untouched cell is not the identity (seal and vault halves each
    depolarizing p0; the outcome is the product of the two Paulis)."""
    return 1 - (1 - p0) ** 2 - p0 ** 2 / 3


COPY_AGREE, COPY_MIN = 0.8, 20  # copies: codes agree on >= 0.8 of the (>= 20) units where either cell shows an error


def copied_pairs(codes, cells):
    """Pairs of cells whose codes agree on >= COPY_AGREE of the units where at least one of them is not the identity
    (independent scrambled cells: 0.2; exact copies: about 1). Units where both are clean are not counted, so two
    rarely hit cells are not mistaken for copies."""
    out = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            x, y = codes[:, i], codes[:, j]
            hit = (x != 0) | (y != 0)
            if hit.sum() >= COPY_MIN and (x[hit] == y[hit]).mean() >= COPY_AGREE:
                out.append([int(cells[i]), int(cells[j])])
    return out


def qmf_report(codes, p0, window=None, library=None, r_max=12):
    """QMF forensic report from vault Bell-measurement records (units in time order x cells).
    tier 1: tampered cells (where), their error rates (how much), onset (when tampering began);
    tier 2: hidden-parity fingerprint (a, s, eps) on those cells (tool/path signature);
    tier 3: tool switch time (first vs last window fingerprints + likelihood-ratio change point) and links to a library
            {name: (cells_tuple, a, s)} of earlier incidents (known-signature test; memoryless labs can run this part).

    Supported scope: an oblivious tool on a region R of fewer than half the cells, with at
    most one switch between tools with different a. 'status' says what the report could do:
      clean           no cell above the benign rate;
      localized       R found and the fingerprint decoded;
      not_decoded     R found, r > r_max;
      global_anomaly  the median cell error rate is above the benign rate: more than half of the cells look tampered
                      (e.g. extend, spray). R cannot be localized from rates; the fingerprint needs a separate decoder
                      (R given, or a sparse search over the seal). cells is None;
      copied_errors   cells of R carry copies of each other's errors (e.g. mirror; see copied_pairs). Which cell is
                      the original cannot be told and a* is ambiguous; the fingerprint is not decoded here (E9 merges
                      copies and decodes separately).
    'tier3' (in every report): switch | no_switch (different a, gain below threshold, or the same tool) |
    unsupported_same_a (both windows significant, same a, different s: outside the supported scope) | not_tested (no
    fingerprint decoded, or a window not significant). A change of strength only, or several changes, is not detected."""
    N = len(codes)
    rate = (codes != 0).mean(axis=0)
    base = float(np.median(rate))
    b0 = benign_rate(p0)
    if base > b0 + 5 * math.sqrt(max(b0, 1 / N) / N) + 5 / N:
        return dict(status="global_anomaly", cells=None, rate_median=round(base, 3), tier3="not_tested",
                    note="more than half of the cells look tampered: not localized; separate decoding needed")
    R = np.flatnonzero(rate > base + 5 * math.sqrt(max(base, 1 / N) / N) + 5 / N)
    rep = dict(cells=R.tolist(), rate=rate[R].round(3).tolist(), tier3="not_tested")  # every report has tier3
    if len(R) == 0:
        rep["status"] = "clean"
        return rep
    r = len(R)
    onset = bernoulli_change_point((codes[:, R] != 0).any(axis=1).astype(np.int64))
    rep["onset"] = onset
    copies = copied_pairs(codes[onset:, R], R)
    if copies:
        rep.update(status="copied_errors", copies=copies,
                   fingerprint="not decoded (copied errors: a* is ambiguous; merge copies and decode separately)")
        return rep
    if r > r_max:
        rep["status"] = "not_decoded"
        rep["fingerprint"] = f"not decoded (r = {r} > {r_max}: noisy sparse-parity regime)"
        return rep
    rep["status"] = "localized"
    E = pack_cells(codes[onset:, R])
    att = (1 - 4 * p0 / 3) ** (2 * weight(np.arange(4 ** r, dtype=np.int64), r))
    rep["fingerprint"] = fingerprint(E, r, att)
    W = window or len(E) // 3
    first, last = fingerprint(E[:W], r, att), fingerprint(E[-W:], r, att)
    rep["first"], rep["last"] = first, last
    rep["tier3"] = "not_tested" if not (first["significant"] and last["significant"]) else \
        "unsupported_same_a" if first["a"] == last["a"] and first["s"] != last["s"] else "no_switch"
    if first["significant"] and last["significant"] and first["a"] != last["a"]:
        def ll(f):
            e = min(max(f["eps"] * att[f["a"]], 1e-3), 1 - 1e-3)
            return np.log1p(f["s"] * e * (1 - 2 * symp(E, np.int64(f["a"]), r)))
        gain = ll(last) - ll(first)
        tail = np.concatenate([np.cumsum(gain[::-1])[::-1], [0.0]])  # tail[k] = sum_{u >= k} gain_u
        k = int(np.argmax(tail))
        if tail[k] > 2 * math.log(len(E)):
            rep["switch"], rep["switch_gain"] = onset + k, float(tail[k])
            rep["tier3"] = "switch"
    if library:
        rep["links"] = {}
        for name, (cells, a_lib, s_lib) in library.items():
            if tuple(cells) == tuple(R.tolist()):
                z = float(np.mean(1 - 2 * symp(E, np.int64(a_lib), r))) * s_lib * math.sqrt(len(E))
                rep["links"][name] = dict(z=z, match=bool(z > 4))
    return rep
