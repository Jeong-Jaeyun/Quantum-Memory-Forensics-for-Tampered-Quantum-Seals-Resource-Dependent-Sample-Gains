"""Repository check.

1. Regenerate sample raw rows of every file in outputs/results from their seeds with the current
   code, and compare them with the stored rows (E1f rows: qmf.experiments.fine replays). For E10, recompute the
   Lagrangian (r <= 5) and graph-state (r <= 6) enumerations, every integer-program bound and one graph search, and
   compare them with e10_enum.json and e10_bound.json.
2. Regenerate every summary JSON with qmf.analysis.summaries (E1f windows and summary, E3f and the headline
   numbers: qmf.experiments.fine), and compare it with the stored one.
3. Regenerate the paper tables (qmf.paper.tables), and compare them byte for byte with
   outputs/paper_tables.
4. Recompute the plotted numbers of the paper figures (qmf.paper.figures --data-only), and compare them byte
   for byte with outputs/paper_figure/source_data.
5. Regenerate the headline numbers (qmf.paper.numbers) and compare them byte for byte with
   outputs/numbers/qmf_numbers.tex.
6. If qiskit is installed, rerun the circuit-versus-law checks of qmf.circuits.

Numbers must agree within a relative 1e-9. On the machine that produced the files, they agree exactly. Rows written
before the `floor` field existed (E1-E3) are compared without it; the current code reports floor = 0.0 for them.

numpy version: the adaptive GHZ rows are those of E5b (e5b.jsonl), which pick their top-k candidates with a stable sort,
so they no longer depend on the numpy version (the superseded E5 rows used np.argpartition, whose
order does).

usage (from the repository root): python -m qmf.reproduce [ROWS_PER_FILE] [--rows-only]
                              python -m qmf.reproduce --write-summaries   (rewrite the stored summaries)
"""
import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import qmf.experiments.multi as multi
import qmf.experiments.tier3 as tier3
import qmf.experiments.uniform as uniform
import qmf.experiments.fine as fine
import qmf.analysis.summaries as rev

O = Path("outputs/results")


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)
    return a == b


def run_job(x):
    return uniform.trial((x["model"], x["method"], x["r"], x["eps"], x["p0"], x["trial"], x["wt"], x["pv"], x["vkind"],
                    x.get("floor", 0.0)))


def switch_job(x):
    return tier3.switch_trial2((x["mode"], x["r"], x["eps"], x["p0"], x["trial"], 32))


def multi_job(x):
    return multi.multi_trial((x["lab"], x["r"], x["K"], x["eps_tot"], x["p0"], x["trial"]))


def e9_job(x):
    import qmf.experiments.references as e9
    return e9.e9_trial((x["lab"], x["attacker"], x["r"], x["eps"], x["p0"], x["trial"]))


def e11_job(x):
    import qmf.experiments.spray as e11
    return e11.e11_trial((x["lab"], x["r"], x["eps"], x["p0"], x["trial"]))


def e11_rows(lab, eps=0.6):
    return lambda x: x["r"] == 4 and x["eps"] == eps and x["lab"] == lab


def e12_job(x):
    import qmf.experiments.product as e12
    return e12.e12_trial((x["lab"], x["r"], x["eps"], x["p0"], x["trial"]))


def e13_job(x):
    import qmf.experiments.self_ref as e13
    return e13.e13_trial((x["r"], x["eps"], x["p0"], x["trial"]))


def e14_job(x):
    import qmf.experiments.seal_noise as e14
    pts = e14.points(x["n_base"], x["r"], x["p0"])
    return e14.e14_trial((x["lab"], x["r"], x["eps"], x["p0"], x["trial"], pts, x["n_base"]))


def e15_job(x):
    import qmf.experiments.reorder as e15
    return e15.reorder_trial((x["f"], x["r"], x["eps"], x["p0"], x["trial"], 32))


def fine_job(x):
    import qmf.experiments.fine as fine
    w = json.load(open(O / "e1f_windows.json"))[fine.key(x["lab"], x["r"], x["eps"], x["p0"])]
    return fine.fine_trial((x["lab"], x["r"], x["eps"], x["p0"], x["trial"], w["points"], w["lo"], w["hi"]))


def e9_rows(lab, att, eps=0.6):
    return lambda x: x["r"] == 4 and x["eps"] == eps and x["lab"] == lab and x["attacker"] == att


small = lambda rmax: (lambda x: x["r"] <= rmax)
# file, cheap-row filter, regenerator
ROW_CHECKS = [("e1_ml", small(5), run_job), ("e1_vault", small(6), run_job),
              ("e1f_ml", small(5), fine_job), ("e1f_vault", small(6), fine_job), ("e2", small(6), run_job),
              ("e2_ghz", small(6), run_job), ("e3_dep", small(6), run_job), ("e3_deph", small(6), run_job),
              ("e3b_dep", small(6), run_job), ("e3b_deph", small(6), run_job),
              ("e5b", lambda x: x["r"] == 6 and x["eps"] == 0.6, run_job),
              ("e9", e9_rows("inseal", "obl"), e9_job), ("e9", e9_rows("vault_seal", "obl"), e9_job),
              ("e9", e9_rows("inseal", "mirror"), e9_job), ("e9", e9_rows("vault_seal", "mirror"), e9_job),
              ("e9", e9_rows("inseal", "extend"), e9_job), ("e9", e9_rows("vault_seal", "extend"), e9_job),
              ("e9", e9_rows("paired", "mirror"), e9_job), ("e9", e9_rows("vault_sparse", "extend"), e9_job),
              ("e11", e11_rows("vault_seal"), e11_job), ("e11", e11_rows("inseal"), e11_job),
              ("e11", e11_rows("inseal_oracle"), e11_job), ("e12", e11_rows("product"), e12_job),
              ("e13", lambda x: x["r"] <= 5 and x["eps"] == 0.6, e13_job),
              ("e14_ml", lambda x: x["r"] == 6 and x["eps"] == 0.6, e14_job),
              ("e14_vault", lambda x: x["r"] == 6 and x["eps"] == 0.6, e14_job),
              ("e15", lambda x: x["r"] == 6 and x["f"] == 0.2, e15_job),
              ("e15", lambda x: x["r"] == 6 and x["f"] == 1.0, e15_job),
              ("e6", lambda x: x["r"] == 6, switch_job),
              ("e8", lambda x: x["r"] == 6 and x["mode"] == "ml_blind", switch_job),
              ("e8", lambda x: x["r"] == 6 and x["mode"] == "ml_library", switch_job),
              ("e7", lambda x: x["r"] == 4 and x["lab"] == "memory", multi_job),
              ("e7", lambda x: x["r"] == 4 and x["lab"] == "ml_random", multi_job),
              ("pl_paired", lambda x: x["r"] <= 5 and x["eps"] == 0.6, run_job),
              ("pl_local", lambda x: x["method"] == "memory" and x["r"] <= 5, run_job),
              ("pl_local", lambda x: x["method"] == "ml_product" and x["r"] <= 5, run_job)]


# stored summary -> (analysis, inputs...): the map shared with the figures and tables (qmf.analysis.derive)
from qmf.analysis.derive import FUNCS, SUMMARIES  # noqa: E402


def check_rows(k):
    bad = 0
    for name, pred, regen in ROW_CHECKS:
        picked = []
        with open(O / f"{name}.jsonl") as fh:
            for line in fh:
                x = json.loads(line)
                if pred(x):
                    picked.append(x)
                    if len(picked) == k:
                        break
        if not picked:
            bad += 1
            print(f"{name:10s} no row matches the filter  <-- UNEXPECTED")
        for x in picked:
            got = json.loads(json.dumps(regen(x)))
            if "floor" not in x and got.get("floor") == 0.0:
                del got["floor"]
            same = close(x, got)
            bad += not same
            tag = {k2: x[k2] for k2 in ("model", "method", "mode", "lab", "r", "eps", "eps_tot", "K", "pv", "vkind",
                                        "trial") if k2 in x}
            print(f"{name:10s} {tag}: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    return bad


def check_e10():
    """E10: enumerations for small r, every integer-program bound, and one graph search (r = 9, w = 9)."""
    import qmf.experiments.coverage as lg
    bad = 0
    en = json.load(open(O / "e10_enum.json"))
    for kind, rmax, fn in (("lagrangians", 5, lambda r: lg.enumerate_lagrangians(r)), ("graphs", 6, lg.enumerate_graphs)):
        for r in range(1, rmax + 1):
            same = fn(r)["enumerators"] == en[kind][str(r)]["enumerators"]
            bad += not same
            print(f"e10       {kind} r={r}: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    B = json.load(open(O / "e10_bound.json"))["per_r"]
    ilp_same = all(lg.ilp_max(int(r), int(w))["max"] == b["ilp_upper"] for r, row in B.items() for w, b in row["w"].items())
    bad += not ilp_same
    print(f"e10       integer-program bounds (all r, w): {'MATCH' if ilp_same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    s = lg.search_max(9, 9, seed=0, restarts=lg.SEARCH_RESTARTS)
    same = s["max"] == B["9"]["w"]["9"]["search_lower"] and s["graph"] == B["9"]["w"]["9"]["search_graph"]
    bad += not same
    print(f"e10       graph search r=9 w=9: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    return bad


def check_summaries():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for stored, (fn, *inputs) in SUMMARIES.items():
            out = str(Path(tmp) / stored)
            with contextlib.redirect_stdout(io.StringIO()):
                FUNCS[fn](*inputs, out)
            same = close(json.load(open(O / stored)), json.load(open(out)))
            bad += not same
            print(f"summary {stored:20s}: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    return bad


def check_tables():
    """outputs/paper_tables must be exactly what qmf.paper.tables writes from the stored summaries."""
    import qmf.paper.tables as tables
    stored, bad = Path("outputs/paper_tables"), 0
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            tables.main(tmp)
        names = sorted(p.name for p in Path(tmp).iterdir())
        if names != sorted(p.name for p in stored.iterdir()):
            bad += 1
            print("paper_tables: file set differs  <-- UNEXPECTED")
        for n in names:
            same = (stored / n).is_file() and (stored / n).read_bytes() == (Path(tmp) / n).read_bytes()
            bad += not same
            print(f"table   {n:20s}: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    return bad


def check_figure_data():
    """outputs/paper_figure/source_data must be exactly what qmf.paper.figures computes (the plotted numbers)."""
    import qmf.paper.figures as figs
    stored, bad = Path("outputs/paper_figure/source_data"), 0
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            figs.main(tmp, data_only=True)
        new = Path(tmp) / "source_data"
        names = sorted(p.name for p in new.iterdir())
        if names != sorted(p.name for p in stored.iterdir()):
            bad += 1
            print("figure source data: file set differs  <-- UNEXPECTED")
        for n in names:
            same = (stored / n).is_file() and (stored / n).read_bytes() == (new / n).read_bytes()
            bad += not same
            print(f"figdata {n:28s}: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    return bad


def check_numbers():
    """outputs/numbers/qmf_numbers.tex (the headline numbers) must be exactly what qmf.paper.numbers writes."""
    import qmf.paper.numbers as numbers
    stored = Path("outputs/numbers/qmf_numbers.tex")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "qmf_numbers.tex"
        with contextlib.redirect_stdout(io.StringIO()):
            numbers.main(str(out))
        same = stored.is_file() and stored.read_bytes() == out.read_bytes()
    print(f"numbers outputs/numbers/qmf_numbers.tex: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}", flush=True)
    return 0 if same else 1


def check_circuits():
    """Qiskit circuits vs the simulator's laws (only where qiskit is installed, e.g. conda env 'quantum')."""
    try:
        import qmf.circuits as circ
    except ImportError:
        print("circuits: skipped (qiskit not installed; run in an environment with qiskit to include this check)")
        return 0
    with contextlib.redirect_stdout(io.StringIO()):
        res = circ.run_checks()
    stored = json.load(open("outputs/paper_figure/circuits/circuit_check.json"))
    same = [x["ok"] for x in res] == [x["ok"] for x in stored["checks"]] and all(x["ok"] for x in res)
    print(f"circuits: {sum(x['ok'] for x in res)}/{len(res)} checks OK: {'MATCH' if same else 'DIFF  <-- UNEXPECTED'}")
    return 0 if same else 1


def main(k, rows_only):
    bad = check_rows(k) + check_e10() + (0 if rows_only else check_summaries() + check_tables() + check_figure_data() +
                                         check_numbers() + check_circuits())
    print("ALL AS EXPECTED" if bad == 0 else f"{bad} UNEXPECTED")
    return bad


def write_summaries():
    """Write every stored summary JSON from its inputs, in SUMMARIES order (e10.json reads e2_e5.json and e1.json)."""
    for stored, (fn, *inputs) in SUMMARIES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            FUNCS[fn](*inputs, str(O / stored))
        print(f"wrote {stored}", flush=True)


if __name__ == "__main__":
    if "--write-summaries" in sys.argv:  # regenerate the stored summaries (not part of the check)
        write_summaries()
        sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(1 if main(int(args[0]) if args else 2, "--rows-only" in sys.argv) else 0)
