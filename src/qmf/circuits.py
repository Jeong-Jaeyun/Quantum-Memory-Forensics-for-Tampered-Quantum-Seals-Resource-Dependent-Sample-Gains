"""Qiskit circuits for the QMF blocks, an exact check against the simulator's laws, and the circuit figures.

The simulator (qmf.core) does not build circuits. It samples the outcome laws in closed form. This module writes the
circuits those laws describe, simulates them exactly (statevector or density matrix), and compares the outcome
distributions with the laws the simulator samples from.

- **Vault lab.** Bell pairs (seal, vault) and a secret Pauli frame on the seal half, then the tamper channel, benign
  noise on both halves, the frame undone and a Bell measurement.
  - Read the measured bits as x on the seal qubits and z on the vault qubits. They are then the packed Pauli E of
    qmf.core.
  - Their law must be 4^-r (1 + s eps mu(a*) (-1)^<a*,E>), with mu = vault_mu.
- **Tamper channel Lambda_{a*,s,eps}.** A Stinespring realization with 2r tool qubits, traced out.
  - Tool qubits 0..2r-2 start in |+>. Each one switches on one generator of the centralizer
    C(a*) = {E : <a*,E> = 0}, so together they apply a uniform element of C(a*).
  - Tool qubit 2r-1 starts in RY(theta)|0>, with sin^2(theta/2) = (1 - s eps)/2. It switches on Q = anticommuting(a*).
  - Pauli eigenvalues: lambda_0 = 1, lambda_a* = s eps, all others 0. This is the scramble model.
- **GHZ probe (ml_ghz).** The memoryless lab for the weight-r family. The measured syndromes must follow
  symp(ghz_frames(Mi), E) under the same law, with seal depolarizing p0.
- **Pauli frame.** Take a coherent attack (local X rotations) on a GHZ probe.
  - Without the frame, the syndrome law shows interference between Paulis in one syndrome coset.
  - Averaged over the 4^r frames, it equals the law of the Pauli-twirled attack, p(E) = |Tr(P_E U)|^2 / 4^r. That
    twirl is what the simulator assumes.

usage: conda run -n quantum python -m qmf.circuits check OUT.json
       conda run -n quantum python -m qmf.circuits draw OUT_DIR
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Pauli, Statevector

import qmf.core as q

TOL = 1e-10
CODE_GATE = {1: "x", 2: "z", 3: "y"}  # per-cell code of qmf.core.report: x bit + 2 z bit


def code(a, r, j):
    return ((a >> j) & 1) | (((a >> (r + j)) & 1) << 1)


def packed(codes):
    r = len(codes)
    return sum((c & 1) << j for j, c in enumerate(codes)) | sum((c >> 1) << (r + j) for j, c in enumerate(codes))


# ------------------------------------------------------------------ single-qubit Cliffords for GL(2,2)
_P1 = {1: np.array([[0, 1], [1, 0]]), 2: np.array([[1, 0], [0, -1]]), 3: np.array([[0, -1j], [1j, 0]])}


def _pauli_code(M):
    for c, P in _P1.items():
        if abs(abs(np.trace(P.conj().T @ M)) - 2) < 1e-9:
            return c
    raise ValueError("not a Pauli")


def _seq_op(seq):
    qc = QuantumCircuit(1)
    for g in seq:
        getattr(qc, g)(0)
    return Operator(qc).data


def _gl2_cliffords():
    """GL2 index -> gate sequence whose conjugation maps X to the image of (1,0) and Z to the image of (0,1), the way
    ghz_frames applies GL2[i] ((Xn, Zn) = M (x, z)^T)."""
    out = {}
    for seq in ([], ["h"], ["s"], ["h", "s"], ["s", "h"], ["h", "s", "h"]):
        U = _seq_op(seq)
        act = (_pauli_code(U @ _P1[1] @ U.conj().T), _pauli_code(U @ _P1[2] @ U.conj().T))
        for i, M in enumerate(q.GL2):
            if act == (M[0][0] + 2 * M[1][0], M[0][1] + 2 * M[1][1]):
                out[i] = seq
    assert sorted(out) == list(range(6)), out
    return out


GL2_GATES = _gl2_cliffords()


# ------------------------------------------------------------------ blocks
def apply_pauli(qc, qubits, P, r):
    for j in range(r):
        c = code(P, r, j)
        if c:
            getattr(qc, CODE_GATE[c])(qubits[j])


def controlled_pauli(qc, ctrl, qubits, P, r):
    for j in range(r):
        c = code(P, r, j)
        if c:
            getattr(qc, "c" + CODE_GATE[c])(ctrl, qubits[j])


def centralizer_generators(a, r):
    """2r - 1 independent generators of C(a) = {E : <a, E> = 0}. On a cell outside the support of a: X and Z. On a
    support cell: the local Pauli of a. Between consecutive support cells j < k: A_j A_k, where A_j anticommutes with
    a_j."""
    gens, supp = [], [j for j in range(r) if code(a, r, j)]
    anti = {1: 2, 3: 2, 2: 1}  # X, Y -> Z; Z -> X
    for j in range(r):
        c = code(a, r, j)
        if c:
            gens.append(packed([c if k == j else 0 for k in range(r)]))
        else:
            gens += [packed([1 if k == j else 0 for k in range(r)]), packed([2 if k == j else 0 for k in range(r)])]
    for j, k in zip(supp, supp[1:]):
        gens.append(packed([anti[code(a, r, i)] if i in (j, k) else 0 for i in range(r)]))
    assert len(gens) == 2 * r - 1 and all(q.symp(np.int64(g), np.int64(a), r) == 0 for g in gens)
    b = q.GF2Basis()
    assert all(b.add(g) for g in gens), "dependent generators"
    return gens


def tamper_theta(s, eps):
    return 2 * math.asin(math.sqrt((1 - s * eps) / 2))


def tamper_block(r, a, theta):
    """Lambda_{a*,s,eps} on a 'seal' register with a 'tool' register (2r qubits, traced out). theta: a float, or a
    Parameter for drawing."""
    seal, tool = QuantumRegister(r, "seal"), QuantumRegister(2 * r, "tool")
    qc = QuantumCircuit(seal, tool)
    gens = centralizer_generators(a, r)
    for k in range(2 * r - 1):
        qc.h(tool[k])
    qc.ry(theta, tool[2 * r - 1])
    for k, g in enumerate(gens + [q.anticommuting(a, r)]):
        controlled_pauli(qc, tool[k], seal, g, r)
    return qc


def vault_parts(r, a, s, eps, frame):
    """(before noise, after noise) circuits on registers seal, vault, tool (seal first, so outcome index = packed E)."""
    seal, vault, tool = QuantumRegister(r, "seal"), QuantumRegister(r, "vault"), QuantumRegister(2 * r, "tool")
    pre, post = QuantumCircuit(seal, vault, tool), QuantumCircuit(seal, vault, tool)
    for j in range(r):  # issue: Bell pairs, then the secret frame on the seal half
        pre.h(vault[j])
        pre.cx(vault[j], seal[j])
    apply_pauli(pre, seal, frame, r)
    pre.compose(tamper_block(r, a, tamper_theta(s, eps)), qubits=list(seal) + list(tool), inplace=True)
    apply_pauli(post, seal, frame, r)  # verification: undo the frame, Bell measurement
    for j in range(r):
        post.cx(vault[j], seal[j])
        post.h(vault[j])
    return pre, post


def ghz_prep(r, Mi):
    qc = QuantumCircuit(QuantumRegister(r, "seal"))
    qc.h(0)
    for j in range(r - 1):
        qc.cx(j, j + 1)
    for j in range(r):
        for g in GL2_GATES[int(Mi[j])]:
            getattr(qc, g)(j)
    return qc


def ghz_probe(r, Mi, frame, attack, n_tool):
    """Memoryless GHZ probe: prepare, frame, attack (circuit on seal + n_tool tool qubits), undo frame, unprepare.
    Measuring the seal register gives syndrome bit k of generator k of ghz_frames(Mi)."""
    seal, tool = QuantumRegister(r, "seal"), QuantumRegister(max(n_tool, 0), "tool")
    qc = QuantumCircuit(seal, tool) if n_tool else QuantumCircuit(seal)
    prep = ghz_prep(r, Mi)
    qc.compose(prep, qubits=list(seal), inplace=True)
    apply_pauli(qc, seal, frame, r)
    qc.compose(attack, qubits=list(seal) + list(tool) if n_tool else list(seal), inplace=True)
    apply_pauli(qc, seal, frame, r)
    qc.compose(prep.inverse(), qubits=list(seal), inplace=True)
    return qc


# ------------------------------------------------------------------ laws on the simulator side
def scramble_law(r, a, s, e):
    allp = np.arange(4 ** r, dtype=np.int64)
    return 4.0 ** -r * (1 + s * e * (1 - 2 * q.symp(allp, np.int64(a), r)))


def depol_law(r, p):
    allp = np.arange(4 ** r, dtype=np.int64)
    w = q.weight(allp, r)
    return (1 - p) ** (r - w) * (p / 3) ** w


def xor_conv(p1, p2):
    return q.fwht(q.fwht(p1) * q.fwht(p2)) / len(p1)


def syndrome_law(gens, pE, r):
    allp = np.arange(4 ** r, dtype=np.int64)
    sig = sum(q.symp(allp, np.int64(g), r) << k for k, g in enumerate(gens))
    return np.bincount(sig, weights=pE, minlength=2 ** r)


# ------------------------------------------------------------------ checks
def kraus_dep(p):
    return Kraus([math.sqrt(1 - p) * np.eye(2)] + [math.sqrt(p / 3) * _P1[c] for c in (1, 2, 3)])


def kraus_deph(p):
    return Kraus([math.sqrt(1 - p) * np.eye(2), math.sqrt(p) * _P1[2]])


def check_vault(r, a, s, eps, frame, p0=0.0, pv=None, vkind="dep"):
    pre, post = vault_parts(r, a, s, eps, frame)
    if p0 == 0 and not pv:
        probs = Statevector(pre.compose(post)).probabilities(list(range(2 * r)))
    else:
        rho = DensityMatrix(pre)
        for j in range(r):
            if p0:
                rho = rho.evolve(kraus_dep(p0), [j])  # seal half
            pvj = p0 if pv is None else pv
            if pvj:
                rho = rho.evolve(kraus_dep(pvj) if vkind == "dep" else kraus_deph(pvj), [r + j])  # vault half
        probs = rho.evolve(post).probabilities(list(range(2 * r)))
    mu = float(q.vault_mu(np.int64(a), r, p0, pv, vkind))
    law = scramble_law(r, a, s, eps * mu)
    return float(np.abs(probs - law).max())


def check_ghz(r, a, s, eps, Mi, frame, p0=0.0):
    attack = tamper_block(r, a, tamper_theta(s, eps))
    qc = ghz_probe(r, Mi, frame, attack, 2 * r)
    if p0 == 0:
        probs = Statevector(qc).probabilities(list(range(r)))
    else:  # depolarizing on the seal after the attack (Paulis commute up to phase, so the place does not matter)
        n = qc.num_qubits
        prep = ghz_prep(r, Mi)
        first = QuantumCircuit(n)
        first.compose(prep, qubits=list(range(r)), inplace=True)
        apply_pauli(first, list(range(r)), frame, r)
        first.compose(attack, qubits=list(range(n)), inplace=True)
        rho = DensityMatrix(first)
        for j in range(r):
            rho = rho.evolve(kraus_dep(p0), [j])
        second = QuantumCircuit(n)
        apply_pauli(second, list(range(r)), frame, r)
        second.compose(prep.inverse(), qubits=list(range(r)), inplace=True)
        probs = rho.evolve(second).probabilities(list(range(r)))
    gens = q.ghz_frames(np.array([Mi]), r)[0]
    pE = xor_conv(scramble_law(r, a, s, eps), depol_law(r, p0)) if p0 else scramble_law(r, a, s, eps)
    return float(np.abs(probs - syndrome_law(gens, pE, r)).max())


def check_frame_twirl(r=2, angles=(0.7, 0.4), Mi=(0, 0)):
    """Coherent attack U = RX(angles[0]) x RX(angles[1]) on a GHZ probe. Returns (deviation without frame, deviation
    of the 4^r-frame average) from the Pauli-twirled law p(E) = |Tr(P_E^dag U)|^2 / 4^r."""
    att = QuantumCircuit(r)
    for j, t in enumerate(angles):
        att.rx(t, j)
    U = Operator(att).data
    allp = np.arange(4 ** r)
    pE = np.array([abs(np.trace(Pauli((np.array([(E >> (r + j)) & 1 for j in range(r)], bool),
                                        np.array([(E >> j) & 1 for j in range(r)], bool))).to_matrix().conj().T
                                  @ U)) ** 2 / 4 ** r for E in allp])
    assert abs(pE.sum() - 1) < 1e-12
    law = syndrome_law(q.ghz_frames(np.array([Mi]), r)[0], pE, r)
    bare = Statevector(ghz_probe(r, Mi, 0, att, 0)).probabilities()
    avg = np.mean([Statevector(ghz_probe(r, Mi, F, att, 0)).probabilities() for F in range(4 ** r)], axis=0)
    return float(np.abs(bare - law).max()), float(np.abs(avg - law).max())


def run_checks():
    rng = np.random.default_rng([953, 1])
    res = []

    def add(name, dev, expect_zero=True, **kw):
        ok = dev < TOL if expect_zero else dev > 1e-3
        res.append(dict(check=name, max_abs_dev=dev, expected="0" if expect_zero else "> 1e-3", ok=bool(ok), **kw))
        print(f"{name:28s} {kw} dev={dev:.2e} {'OK' if ok else 'FAIL'}", flush=True)

    for r in (1, 2, 3):
        for _ in range(4):
            a, s = int(rng.integers(1, 4 ** r)), int(rng.choice([-1, 1]))
            eps, frame = float(rng.choice([0.15, 0.6])), int(rng.integers(0, 4 ** r))
            add("vault, noiseless", check_vault(r, a, s, eps, frame), r=r, a=a, s=s, eps=eps, frame=frame)
    for vkind in ("dep", "deph"):
        for r in (1, 2):
            a, s, frame = int(rng.integers(1, 4 ** r)), int(rng.choice([-1, 1])), int(rng.integers(0, 4 ** r))
            add("vault, benign noise", check_vault(r, a, s, 0.6, frame, p0=0.05, pv=0.2, vkind=vkind),
                r=r, a=a, s=s, eps=0.6, frame=frame, p0=0.05, pv=0.2, vkind=vkind)
    for r in (2, 3, 4):
        for _ in range(3):
            a, s = int(rng.integers(1, 4 ** r)), int(rng.choice([-1, 1]))
            Mi, frame = rng.integers(0, 6, r), int(rng.integers(0, 4 ** r))
            add("GHZ probe, noiseless", check_ghz(r, a, s, 0.6, Mi, frame), r=r, a=a, s=s, eps=0.6, Mi=Mi.tolist(),
                frame=frame)
    for r in (2, 3):
        a, s = int(rng.integers(1, 4 ** r)), int(rng.choice([-1, 1]))
        Mi, frame = rng.integers(0, 6, r), int(rng.integers(0, 4 ** r))
        add("GHZ probe, seal noise", check_ghz(r, a, s, 0.6, Mi, frame, p0=0.05), r=r, a=a, s=s, eps=0.6,
            Mi=Mi.tolist(), frame=frame, p0=0.05)
    bare, avg = check_frame_twirl()
    add("coherent attack, no frame", bare, expect_zero=False, r=2, attack="RX(0.7) x RX(0.4)")
    add("coherent attack, frame avg", avg, r=2, attack="RX(0.7) x RX(0.4)")
    return res


# ------------------------------------------------------------------ figures
FIG_A, FIG_S, FIG_EPS = packed([1, 3, 2]), 1, 0.6  # a* = X (x) Y (x) Z for the r = 3 drawings
FIG_FRAME = packed([3, 1, 2])  # one frame instance: Y, X, Z


def fig_issue(r=3):
    seal, vault = QuantumRegister(r, "seal"), QuantumRegister(r, "vault")
    qc = QuantumCircuit(vault, seal)
    for j in range(r):
        qc.h(vault[j])
    for j in range(r):
        qc.cx(vault[j], seal[j])
    qc.barrier(label="frame F")
    apply_pauli(qc, seal, FIG_FRAME, r)
    return qc


def fig_tamper(r=3):
    t = tamper_block(r, FIG_A, Parameter("θ"))
    return t


def fig_verify(r=3):
    seal, vault = QuantumRegister(r, "seal"), QuantumRegister(r, "vault")
    x, z = ClassicalRegister(r, "x"), ClassicalRegister(r, "z")
    qc = QuantumCircuit(vault, seal, x, z)
    apply_pauli(qc, seal, FIG_FRAME, r)
    qc.barrier(label="Bell measurement")
    for j in range(r):
        qc.cx(vault[j], seal[j])
    for j in range(r):
        qc.h(vault[j])
    for j in range(r):
        qc.measure(seal[j], x[j])
    for j in range(r):
        qc.measure(vault[j], z[j])
    return qc


def fig_ghz(r=4, Mi=(1, 3, 0, 4)):
    seal, sig = QuantumRegister(r, "seal"), ClassicalRegister(r, "σ")
    qc = QuantumCircuit(seal, sig)
    prep = ghz_prep(r, Mi)
    qc.compose(prep, inplace=True)
    qc.barrier()
    qc.append(Gate("F", r, [], label="F"), seal)
    qc.append(Gate("Lambda", r, [], label="Λ"), seal)
    qc.append(Gate("Finv", r, [], label="F⁻¹"), seal)
    qc.barrier()
    qc.compose(prep.inverse(), inplace=True)
    qc.measure(seal, sig)
    return qc


def draw_all(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, qc, fold in (("qmf_circ_issue", fig_issue(), -1), ("qmf_circ_tamper", fig_tamper(), -1),
                           ("qmf_circ_verify", fig_verify(), -1), ("qmf_circ_probe_ghz", fig_ghz(), -1)):
        fig = qc.draw("mpl", style="bw", fold=fold, scale=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300,
                        metadata={"CreationDate": None} if ext == "pdf" else None)
        made.append(dict(file=f"{name}.pdf", qubits=qc.num_qubits, gates=sum(1 for d in qc.data if d.operation.name
                                                                             not in ("barrier", "measure"))))
        print(out / f"{name}.pdf", qc.num_qubits, "qubits")
    return made


if __name__ == "__main__":
    if sys.argv[1] == "check":
        res = run_checks()
        bad = [x for x in res if not x["ok"]]
        json.dump(dict(tolerance=TOL, checks=res, all_ok=not bad), open(sys.argv[2], "w"), indent=1)
        print("ALL CHECKS OK" if not bad else f"{len(bad)} FAILED")
        sys.exit(1 if bad else 0)
    elif sys.argv[1] == "draw":
        draw_all(sys.argv[2])
