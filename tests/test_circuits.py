"""Qiskit circuits of the QMF blocks reproduce the simulator's outcome laws exactly (skipped without qiskit).
Run with: conda run -n quantum python -m pytest -q -p no:cacheprovider tests/test_circuits.py"""
import numpy as np
import pytest

pytest.importorskip("qiskit")
import qmf.circuits as qc  # noqa: E402


def test_gl2_cliffords_cover_all_six():
    assert sorted(qc.GL2_GATES) == list(range(6))


@pytest.mark.parametrize("r,a,s,eps,frame", [(1, 3, 1, 0.6, 2), (2, 9, -1, 0.15, 7), (3, qc.FIG_A, 1, 0.6, qc.FIG_FRAME)])
def test_vault_circuit_matches_law(r, a, s, eps, frame):
    assert qc.check_vault(r, a, s, eps, frame) < qc.TOL


@pytest.mark.parametrize("vkind", ["dep", "deph"])
def test_vault_circuit_with_benign_noise(vkind):
    assert qc.check_vault(2, 11, -1, 0.6, 5, p0=0.05, pv=0.2, vkind=vkind) < qc.TOL


def test_ghz_probe_matches_syndrome_law():
    assert qc.check_ghz(3, 42, 1, 0.6, np.array([3, 2, 3]), 63) < qc.TOL
    assert qc.check_ghz(2, 10, 1, 0.6, np.array([3, 5]), 0, p0=0.05) < qc.TOL


def test_frame_twirls_coherent_attack():
    bare, avg = qc.check_frame_twirl()
    assert bare > 1e-3 and avg < qc.TOL


def test_tamper_block_shape():
    t = qc.tamper_block(3, qc.FIG_A, 0.3)
    assert t.num_qubits == 9 and len(qc.centralizer_generators(qc.FIG_A, 3)) == 5
