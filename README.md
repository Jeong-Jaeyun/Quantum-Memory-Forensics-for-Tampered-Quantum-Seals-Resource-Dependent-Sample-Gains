# Quantum-Memory Forensics for Tampered Quantum Seals: Resource-Dependent Sample Gains

Code, simulation results and analysis for quantum-memory forensics (QMF) of tampered quantum seals.

## Setting

- **Seal and vault.** Each cell of an n-cell seal module is one half of a Bell pair. The verifier keeps the other half
  in a protected quantum memory, the *vault*.
- **Frame.** A secret random Pauli frame, applied at issue and undone at verification, twirls any attack into a Pauli
  channel. A Bell measurement of a cell against its vault partner gives one sample of the attack's Pauli error per
  seal unit.
- **Tamper model.** The attacker's tool acts on a region R of r cells with a hidden-parity fingerprint (a\*, s, ε):
  every cell of R looks fully depolarized, and only the joint parity of a\* carries the fingerprint.
- **QMF report** (`qmf.core.report.qmf_report`). From raw vault records it returns a status code and three tiers:
  1. the tampered cells, their error rates and the onset;
  2. the fingerprint (a\*, s, ε);
  3. a switch between tools, and links to a registry of earlier incidents.

  Status codes: `clean`, `localized`, `not_decoded` (r > 12), `global_anomaly` (more than half of the cells look
  tampered) and `copied_errors` (cells carry copies of each other's errors).
- **Verifier classes.** Every result names one of four classes:
  - per-unit verifiers (no quantum coupling between units, no untouched reference);
  - self-referencing verifiers (two units measured jointly: P and P + sign);
  - in-seal reference verifiers (untouched cells of the same seal);
  - the protected reference (the vault).
- **Attackers.**
  - oblivious;
  - extend (also depolarizes every cell outside R);
  - spray (copies the errors of R onto groups of other cells);
  - mirror (copies each tampered cell's error onto its reference partner; needs the pairing).
- **Cost.** N90 is the number of seal units at which 90% of the trials identify (a\*, s).

## Experiments

Every trial draws its instance from a fixed seed space, so different verifiers face the same fingerprints. Each row
of a raw file is one trial and can be regenerated from its seed (`python -m qmf.reproduce` does this for samples).

| ID | Question | Config | Raw rows (`outputs/results/`) |
|---|---|---|---|
| E1 | Uniform fingerprint: vault vs per-unit uniform-Lagrangian probes, r = 4–10, ε ∈ {0.15, 0.6}, p0 ∈ {0, 0.005} | `e1_e3.json` | `e1_ml`, `e1_vault` |
| E1f | Fine checkpoints around N90: E1 replayed with the same draws | `e1f.json` | `e1f_ml`, `e1f_vault` |
| E2 | Fingerprint weight w ∈ {1, 2, 3, ⌈r/2⌉, r}, r ∈ {6, 8, 10} | `e1_e3.json` | `e2` |
| E2-GHZ | GHZ-type per-unit probes at w = r on E2's instances (post hoc) | `e2_ghz.json` | `e2_ghz` |
| E3 | Vault noise p_v (depolarizing, dephasing), r ∈ {6, 8, 10} | `e1_e3.json`, `e3b.json` | `e3_dep`, `e3_deph`, `e3b_dep`, `e3b_deph` |
| E5b | Adaptive GHZ heuristic (explore/exploit, deterministic top-k) | `e5b.json` | `e5b` |
| E6 | Tier 3 with the full report (vault): one switch between tools, 300 switch and 300 no-switch trials | `e5_e8.json` | `e6` |
| E7 | Several hidden parities (K ≤ 3), ε and K unknown, r ≤ 8 | `e5_e8.json` | `e7` |
| E8 | Tier 3, per-unit detectors (blind and registry) | `e5_e8.json` | `e8` |
| P, local | Self-referenced pairs (P) and local (unscrambled) tampering | `pl.json` | `pl_paired`, `pl_local` |
| E9 | In-seal references, P and the vault under oblivious, mirror and extend attackers (32 cells) | `e9.json` | `e9` |
| E10 | Coverage bound for non-adaptive stabilizer probes (enumeration, integer program, Fano bound) | `e10.json` | `e10_enum.json`, `e10_bound.json` |
| E11 | Spray attacker: vault, in-seal references, in-seal oracle (32 cells) | `e11.json` | `e11` |
| E12 | Reference-free product probe against spray; the vault's bound with R unknown | `e12.json` | `e12` |
| E13 | Self-referencing P + sign; its bound under extend | `e13.json` | `e13` |
| E14 | Seal noise p0 ≤ 0.1 with the vault's noise fixed at 0.005 | `e14.json` | `e14_ml`, `e14_vault` |
| E15 | Tier 3 under transit reordering: E6's vault records with a fraction of the units permuted (post hoc) | `e15.json` | `e15` |

`e5_e8.json` also runs E5, the first adaptive GHZ heuristic, which E5b replaces; its rows are not included.

## Layout

```
src/qmf/                   the qmf package
  core/                    model and algorithm (import qmf.core as q)
    pauli.py               packed Paulis: symplectic product, weight, Walsh-Hadamard transform, random Paulis
    stabilizer.py          stabilizer probes: random, product, GHZ (frames, covering), Lagrangians, GF(2) span
    channels.py            tamper and noise channels, outcome sampling, the vault's outcome law
    labs.py                laboratories and decoders: run_vault, run_memoryless, run_memory, checkpoints, horizons
    report.py              qmf_report (tiers 1-3): seal simulation, change point, fingerprint, copy detection
  experiments/             one module per experiment: trial functions and their parallel run
    uniform.py             N90 runs (E1-E3, E5, P, local tampering)
    tier3.py               tool switches (E6, E8)
    multi.py               several hidden parities (E7)
    references.py          E9: full-seal simulator with in-seal references, P and the vault
    coverage.py            E10: Lagrangians (r <= 6), graph states (r <= 8), integer-program bounds, Fano bound
    fine.py                E1f: fine checkpoints; E3f and the headline values (fine first)
    spray.py               E11: the spray attacker
    product.py             E12: product probe against spray; the vault's bound with R unknown
    self_ref.py            E13: P + sign; its bound under extend
    seal_noise.py          E14: seal-noise sweep
    reorder.py             E15: tier 3 under transit reordering
  analysis/
    summaries.py           analyses -> summary JSONs (outputs/results/*.json)
    derive.py              every summary recomputed in memory from the raw rows (used by tables, figures, numbers)
  paper/
    tables.py              raw results -> outputs/paper_tables (CSV + tables.md)
    figures.py             raw results -> outputs/paper_figure (fig2-fig9) and their source data
    numbers.py             raw results -> outputs/numbers/qmf_numbers.tex (headline numbers as LaTeX macros)
  circuits.py              Qiskit circuits of the protocol blocks, checked exactly against the simulator's laws
  reproduce.py             repository check (rows from seeds, E10, summaries, tables, figure data, numbers, circuits)
  run.py                   runs an experiment config: python -m qmf.run configs/e14.json
configs/                   one JSON per experiment group: the grid of every step
scripts/
  run_all.ps1              every config in order, into a new results directory
  make_hashes.py           regenerates SHA256SUMS
tests/                     51 unit tests (+9 circuit tests with Qiskit)
outputs/
  results/                 raw rows (*.jsonl), summaries (*.json), run log (progress.txt)
  paper_tables/            tables (CSV + tables.md)
  paper_figure/            result figures fig2-fig9 (PDF, PNG), source_data/ (plotted numbers), circuits/
  numbers/                 qmf_numbers.tex (headline numbers)
```

Figure files: `fig2_uniform` (uniform family; E1, E1f, E13), `fig3_weight` (fingerprint weight; E2, E5b, E10),
`fig4_vault_noise` (E3), `fig5_tier3` (E6, E8), `fig6_multi` (E7), `fig7_attacks` (results by attack; E9, E11, E12),
`fig8_seal_noise` (E14), `fig9_case_study` (one E6 vault trial through the full report).

## Install

Python 3.12 or later; numpy 2.0 or later (for `np.bitwise_count`).

```
pip install -r requirements.txt
pip install -e .                 # or set PYTHONPATH=src
pip install -e ".[circuits]"     # optional: Qiskit for the circuit checks
```

Environments used: Python 3.13.9 with numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.6 and pytest 8.4.2; and, for the
circuits, Python 3.12.0 with numpy 2.5.0, Qiskit 2.4.2 and qiskit-aer 0.17.2. Both pass the full check below.

## Check

From the repository root:

```
python -m pytest -q tests        # unit tests (the circuit tests are skipped without Qiskit)
python -m qmf.reproduce          # full check, about 15 minutes
```

`qmf.reproduce`:
- regenerates sample rows of every raw file from their seeds and compares them with the stored rows;
- recomputes E10's enumerations (r ≤ 5 Lagrangians, r ≤ 6 graph states), its integer-program bounds and one search;
- recomputes every summary JSON;
- rebuilds the tables, the plotted numbers of the figures and the headline numbers, and compares them byte for byte;
- with Qiskit installed, reruns the 29 circuit-versus-law checks.

Numbers must agree within a relative 1e-9; on the machine that produced the files they agree exactly.

## Rebuild the outputs

```
python -m qmf.paper.tables       # outputs/paper_tables
python -m qmf.paper.figures      # outputs/paper_figure (fig2-fig9 and source_data)
python -m qmf.paper.numbers      # outputs/numbers/qmf_numbers.tex
python -m qmf.circuits check outputs/paper_figure/circuits/circuit_check.json   # needs Qiskit
```

The tables, figures and numbers never read a stored summary: `qmf.analysis.derive` recomputes every summary in
memory from the raw rows. One summary can be recomputed with
`python -m qmf.analysis.summaries NAME INPUTS... OUT.json`; the inputs of each are listed in `SUMMARIES` in
`src/qmf/analysis/derive.py`.

## Rerun the experiments

Each config in `configs/` is run by `python -m qmf.run configs/NAME.json`:
- `--dry-run` prints each step's call, `--only STEP,STEP` runs some steps, `--results DIR` writes elsewhere.
- Each step runs in its own process with single-threaded BLAS, its stdout in `DIR/STEP.log` and START/END lines in
  `DIR/progress.txt`.
- The run functions append to their output files, so the runner refuses to start when an output exists. Rerun into a
  new directory, e.g. `--results outputs/rerun` (ignored by git). `scripts/run_all.ps1` reruns every config in order.

The runs used 6 workers on 8 cores.

| Config | Experiments | Time |
|---|---|---|
| `e1_e3.json` | E1, E2, E3 | ~11 h |
| `e3b.json` | E3 refinement (longer vault horizon) | ~1.5 h |
| `e5_e8.json` | E6, E8, E7, E5 | ~3.4 h |
| `pl.json` | P and local tampering | ~12 min |
| `e2_ghz.json` | GHZ-type baseline for E2 | ~15 min |
| `e5b.json` | E5b | ~3 h |
| `e9.json` | E9 | ~2.2 h |
| `e10.json` | E10 | ~20 min |
| `e1f.json` | E1f (replays E1's rows) | ~3.6 h |
| `e11.json` | E11 | ~45 min |
| `e12.json` | E12 | ~25 min |
| `e13.json` | E13 | ~15 min |
| `e14.json` | E14 (reads `e1f.json`) | ~7.5 h |
| `e15.json` | E15 | ~37 min |

After a rerun, the summaries are recomputed with `qmf.analysis.summaries` (see above).

## Integrity

`SHA256SUMS` holds the SHA-256 of every file; check it with `sha256sum -c SHA256SUMS`. `.gitattributes` turns off
line-ending conversion so that the files check out byte-exact on every OS.

## License

Apache License 2.0; see `LICENSE`.
