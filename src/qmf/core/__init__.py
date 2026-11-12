"""QMF core: the model and the algorithm, independent of any experiment.

Paulis on r qubits are ints in [0, 4^r): bits 0..r-1 = x part, bits r..2r-1 = z part.
Symplectic product <a, b> = parity(|a_x & b_z| + |a_z & b_x|).

Tamper models (Pauli-twirled channel on the tampered region R, |R| = r):
  scramble: p(E) = 4^-r (1 + s eps (-1)^<a*, E>)   (hidden-parity fingerprint a*, sign s)
  local:    E = a* with probability q, else identity (no scrambling)
Benign noise: independent per-cell depolarizing p0 (vault lab: seal + vault halves; per-unit: seal only).

Modules: pauli (algebra), stabilizer (per-unit probes), channels (tamper and noise), labs (vault, memory/
paired, per-unit labs and N90 checkpoints), report (qmf_report, tiers 1-3). Every name is re-exported here,
so `import qmf.core as q` gives the whole model in one namespace.
"""
from qmf.core.channels import depolarize, sample, vault_counts, vault_mu
from qmf.core.labs import BATCH, CAP, METHODS, checkpoints, eval_point, n_max_for, probe_generators, run_memory, run_memoryless, run_vault
from qmf.core.pauli import anticommuting, fwht, jaccard, pack, random_pauli_weight, swap_xz, symp, weight
from qmf.core.report import COPY_AGREE, COPY_MIN, benign_rate, bernoulli_change_point, copied_pairs, fingerprint, pack_cells, qmf_report, simulate_seal, unpack
from qmf.core.stabilizer import COVER, FULL, GF2Basis, GL2, full_weight, ghz_covering, ghz_frames, ghz_stabilizer, lagrangian_containing, product_stabilizer, random_stabilizer, span_elements, syndrome_parities, uniform_lagrangian

__all__ = ["depolarize", "sample", "vault_counts", "vault_mu", "BATCH", "CAP", "METHODS", "checkpoints", "eval_point", "n_max_for", "probe_generators", "run_memory", "run_memoryless", "run_vault", "anticommuting", "fwht", "jaccard", "pack", "random_pauli_weight", "swap_xz", "symp", "weight", "COPY_AGREE", "COPY_MIN", "benign_rate", "bernoulli_change_point", "copied_pairs", "fingerprint", "pack_cells", "qmf_report", "simulate_seal", "unpack", "COVER", "FULL", "GF2Basis", "GL2", "full_weight", "ghz_covering", "ghz_frames", "ghz_stabilizer", "lagrangian_containing", "product_stabilizer", "random_stabilizer", "span_elements", "syndrome_parities", "uniform_lagrangian"]
