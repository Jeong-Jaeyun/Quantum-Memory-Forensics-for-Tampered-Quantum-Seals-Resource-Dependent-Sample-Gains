"""Run an experiment from its config: python -m qmf.run configs/e14.json [--dry-run] [--only STEP[,STEP]] [--results DIR]

A config (configs/*.json) is the grid of one experiment: a list of steps, each naming a task (TASKS below) and its
arguments. The runner types each argument and calls the task's function with them.

- Paths may use {results}, the results directory (default outputs/results, or --results DIR).
- Each step runs in its own process with stdout in {results}/<step>.log, and START/END lines go to {results}/progress.txt.
  When every step has run, the config's "done" text (e.g. "E14 DONE") is appended.
- The run functions append to their output files. Before anything runs, the runner refuses if an output of a selected
  step already exists, so a rerun cannot double the stored rows. Rerun into another directory with --results.
- BLAS threads are pinned to 1 (OMP, OpenBLAS, MKL); the workers are the parallelism.
- --dry-run prints each step's call without running it.
"""
import argparse
import datetime
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
RESULTS = "outputs/results"


def _list(conv):
    return lambda v: [conv(x) for x in (v.split(",") if isinstance(v, str) else v)]


def _float_or_none(x):
    return None if x in ("-", None) else float(x)


TYPES = {
    "str": str, "int": int, "float": float, "in": str, "out": str,
    "strs": _list(str), "ints": _list(int), "floats": _list(float), "floats_or_none": _list(_float_or_none),
}

# task -> (module, function, [(parameter, type)]). "workers" is filled from the config (or the step) at its position.
# Parameters marked "in" must exist before the step runs; parameters marked "out" must not exist before the config runs.
TASKS = {
    "uniform": ("qmf.experiments.uniform", "run",
                [("workers", "int"), ("out", "out"), ("model", "str"), ("methods", "strs"), ("rs", "ints"),
                 ("epss", "floats"), ("p0s", "floats"), ("trials", "int"), ("wts", "strs"), ("pvs", "floats_or_none"),
                 ("vkind", "str"), ("floor", "float")]),
    "switch2": ("qmf.experiments.tier3", "run_switch2",
                [("workers", "int"), ("out", "out"), ("modes", "strs"), ("rs", "ints"), ("epss", "floats"),
                 ("p0", "float"), ("trials", "int")]),
    "multi": ("qmf.experiments.multi", "run_multi",
              [("workers", "int"), ("out", "out"), ("labs", "strs"), ("rs", "ints"), ("Ks", "ints"),
               ("epss", "floats"), ("p0", "float"), ("trials", "trials_by_r")]),
    "references": ("qmf.experiments.references", "run",
                   [("workers", "int"), ("out", "out"), ("labs", "strs"), ("atts", "strs"), ("rs", "ints"),
                    ("epss", "floats"), ("p0", "float"), ("trials", "int")]),
    "coverage_enum": ("qmf.experiments.coverage", "cmd_enum",
                      [("r_lagr", "int"), ("r_graph", "int"), ("workers", "int"), ("out", "out")]),
    "coverage_bound": ("qmf.experiments.coverage", "cmd_bound", [("enum_path", "in"), ("out", "out")]),
    "fine_windows": ("qmf.experiments.fine", "windows", [("ml_path", "in"), ("vault_path", "in"), ("out", "out")]),
    "fine_run": ("qmf.experiments.fine", "run",
                 [("workers", "int"), ("win_path", "in"), ("lab", "str"), ("e1_path", "in"), ("out", "out"),
                  ("rs", "ints")]),
    "fine_analyze": ("qmf.experiments.fine", "analyze",
                     [("ml_f", "in"), ("vault_f", "in"), ("ml_path", "in"), ("vault_path", "in"), ("win_path", "in"),
                      ("out", "out")]),
    "spray": ("qmf.experiments.spray", "run",
              [("workers", "int"), ("out", "out"), ("labs", "strs"), ("rs", "ints"), ("epss", "floats"),
               ("p0", "float"), ("trials", "int")]),
    "product": ("qmf.experiments.product", "run",
                [("workers", "int"), ("out", "out"), ("rs", "ints"), ("epss", "floats"), ("p0", "float"),
                 ("trials", "int")]),
    "self_ref": ("qmf.experiments.self_ref", "run",
                 [("workers", "int"), ("out", "out"), ("rs", "ints"), ("epss", "floats"), ("p0s", "floats"),
                  ("trials", "int")]),
    "seal_noise": ("qmf.experiments.seal_noise", "run",
                   [("workers", "int"), ("out", "out"), ("lab", "str"), ("rs", "ints"), ("epss", "floats"),
                    ("p0s", "floats"), ("trials", "int"), ("e1f_path", "in")]),
    "reorder": ("qmf.experiments.reorder", "run",
                [("workers", "int"), ("out", "out"), ("fs", "floats"), ("rs", "ints"), ("epss", "floats"),
                 ("p0", "float"), ("trials", "int")]),
}


def resolve(config, step, results):
    """The (module, function, kwargs, paths) of one step. Arguments left out take the function's default."""
    module, func, params = TASKS[step["task"]]
    given = dict(step["args"])
    if any(p == "workers" for p, _ in params):
        given.setdefault("workers", config["workers"])
    unknown = set(given) - {p for p, _ in params}
    if unknown:
        raise ValueError(f"{step['name']}: unknown arguments {sorted(unknown)} for task {step['task']}")
    kwargs, paths = {}, {"in": [], "out": []}
    for p, t in params:
        if p not in given:
            continue
        v = given[p]
        if t in ("in", "out"):
            v = v.format(results=results)
            paths[t].append(v)
        if t == "trials_by_r":  # {r: trials} for the one lab of the step, as {lab: {r: trials}}
            v = {lab: {int(r): int(n) for r, n in v.items()} for lab in kwargs["labs"]}
        else:
            v = TYPES[t](v)
        kwargs[p] = v
    return module, func, kwargs, paths


def _call(config_path, name, results):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    step = next(s for s in config["steps"] if s["name"] == name)
    module, func, kwargs, _ = resolve(config, step, results)
    getattr(importlib.import_module(module), func)(**kwargs)


def _stamp():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def main(argv):
    ap = argparse.ArgumentParser(prog="python -m qmf.run", description="Run the steps of an experiment config.")
    ap.add_argument("config")
    ap.add_argument("--only", help="comma-separated step names (default: all steps)")
    ap.add_argument("--results", default=RESULTS, help=f"results directory (default {RESULTS})")
    ap.add_argument("--dry-run", action="store_true", help="print each step's call without running it")
    ap.add_argument("--call", metavar="STEP", help=argparse.SUPPRESS)  # internal: one step in this process
    a = ap.parse_args(argv)
    if a.call:  # the parent redirects this process's stdout to the step's log
        _call(a.config, a.call, a.results)
        return 0
    config_path = Path(a.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results = a.results
    only = set(a.only.split(",")) if a.only else None
    steps = [s for s in config["steps"] if only is None or s["name"] in only]
    if only and len(steps) != len(only):
        raise SystemExit(f"unknown steps: {sorted(only - {s['name'] for s in config['steps']})}")
    resolved = [(s, *resolve(config, s, results)) for s in steps]
    if a.dry_run:
        for s, module, func, kwargs, _ in resolved:
            print(f"{s['name']}: {module}.{func}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
        return 0
    outs = sorted({p for *_, paths in resolved for p in paths["out"]})
    existing = [p for p in outs if Path(p).exists()]
    if existing:
        raise SystemExit(f"outputs already exist (the run functions append): {existing}\n"
                         f"move them away or pass --results DIR")
    produced = set(outs)
    for s, *_, paths in resolved:
        missing = [p for p in paths["in"] if not Path(p).exists() and p not in produced]
        if missing:
            raise SystemExit(f"{s['name']}: missing inputs {missing}")
    Path(results).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{v: "1" for v in THREAD_VARS})
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    progress = Path(results) / "progress.txt"
    for s, *_ in resolved:
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} START {s['name']}\n")
        with open(Path(results) / f"{s['name']}.log", "w", encoding="utf-8") as log:
            code = subprocess.run([sys.executable, "-m", "qmf.run", str(config_path), "--call", s["name"],
                                   "--results", results], stdout=log, env=env).returncode
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} END {s['name']} exit={code}\n")
        if code:
            raise SystemExit(f"{s['name']} failed (exit {code}); see {results}/{s['name']}.log")
    if only is None:
        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} {config['done']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
