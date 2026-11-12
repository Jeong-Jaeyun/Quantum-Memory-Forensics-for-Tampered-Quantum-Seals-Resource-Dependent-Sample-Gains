"""Summaries recomputed from the raw results, for the paper figures and tables.

The paper's figures and tables must come from the raw results, not from stored summary files. This module recomputes
every summary JSON of outputs/results in memory (a temporary directory per process) with the
analysis code, from:
- the raw rows (*.jsonl) of every experiment;
- the outputs of the E10 computation (e10_bound.json: integer-program bounds and searches, checked by
  qmf.reproduce against a recomputation).
A summary that reads another summary reads the freshly computed one, never the stored file. The stored summary JSONs
remain as the record and are compared with these by qmf.reproduce; nothing that is plotted or tabulated is read
from them.

SUMMARIES lists every summary with its analysis and inputs, in dependency order; qmf.reproduce uses the same map.
"""
import atexit
import contextlib
import io
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import qmf.experiments.fine as fine
import qmf.analysis.summaries as rev

O = Path("outputs/results")


def f(name):
    return str(O / f"{name}.jsonl")


# stored summary -> (analysis, inputs...); inputs that are summaries come earlier in this order
SUMMARIES = {"e1.json": ("e1", f("e1_ml"), f("e1_vault")),
             "e1f_windows.json": ("e1f_windows", f("e1_ml"), f("e1_vault")),
             "e1f.json": ("e1f", f("e1f_ml"), f("e1f_vault"), f("e1_ml"), f("e1_vault"), str(O / "e1f_windows.json")),
             "e2.json": ("e2", f("e2"), f("e1_ml")),
             "e2_ghz.json": ("e2", f"{f('e2')},{f('e2_ghz')}", f("e1_ml")),
             "e2_e5.json": ("e2", f"{f('e2')},{f('e2_ghz')},{f('e5b')}", f("e1_ml")),
             "e3_dep.json": ("e3", f("e3_dep"), f("e1_ml")),
             "e3_deph.json": ("e3", f("e3_deph"), f("e1_ml")),
             "e3_final_dep.json": ("e3", f"{f('e3_dep')},{f('e3b_dep')}", f("e1_ml")),
             "e3_final_deph.json": ("e3", f"{f('e3_deph')},{f('e3b_deph')}", f("e1_ml")),
             "e5.json": ("e5", f"{f('e2_ghz')},{f('e5b')}"),
             "e6e8.json": ("e6", f"{f('e6')},{f('e8')}"),
             "e7.json": ("e7", f("e7")),
             "pl.json": ("pl", f"{f('pl_paired')},{f('pl_local')}", f("e1_vault")),
             "e9.json": ("e9", f("e9"), f("e1_vault")),
             # e10.json reads e2_e5.json and e1.json (above) and E10's bound file
             "e10.json": ("e10", str(O / "e10_bound.json"), str(O / "e2_e5.json"), str(O / "e1.json")),
             # final scope (proof against proof for E10; reads e10.json)
             "scope.json": ("scope", str(O / "e10.json")),
             "e11.json": ("e11", f("e11"), f("e9")),
             # fine values in the headline numbers (read e1f.json and e3_final_*.json)
             "e3f_dep.json": ("e3f", f"{f('e3_dep')},{f('e3b_dep')}", str(O / "e1f.json")),
             "e3f_deph.json": ("e3f", f"{f('e3_deph')},{f('e3b_deph')}", str(O / "e1f.json")),
             "headline.json": ("headline", str(O / "e1f.json"), str(O / "e3f_dep.json"), str(O / "e3f_deph.json"),
                               str(O / "e3_final_dep.json"), str(O / "e3_final_deph.json")),
             # product probe against spray, and the extend bound (reads e1f.json)
             "e12.json": ("e12", f("e12"), f("e11"), f("e9"), str(O / "e1f.json")),
             # P + sign and the self-referencing cost under extend
             "e13.json": ("e13", f("e13"), str(O / "pl.json"), str(O / "e1.json"), str(O / "e1f.json"),
                          str(O / "e10.json")),
             # seal-noise sweep (reads e1f.json for the control P3)
             "e14.json": ("e14", f("e14_ml"), f("e14_vault"), str(O / "e1f.json")),
             # tier 3 under transit reordering (post hoc; control against e6.jsonl)
             "e15.json": ("e15", f("e15"), f("e6"))}
FUNCS = {"e1": rev.e1, "e2": rev.e2, "e3": rev.e3, "e5": rev.e5, "e6": rev.e6, "e7": rev.e7, "pl": rev.pl,
         "e9": rev.e9, "e10": rev.e10, "e11": rev.e11, "e12": rev.e12, "e13": rev.e13, "e14": rev.e14,
         "e15": rev.e15, "scope": rev.scope,
         "e1f_windows": fine.windows, "e1f": fine.analyze, "e3f": fine.e3f, "headline": fine.headline}


def fresh_inputs(arg, d):
    """Replace stored summaries among comma-separated inputs by their copies recomputed in directory d."""
    return ",".join(str(Path(d) / Path(p).name) if Path(p).name in SUMMARIES else p for p in arg.split(","))


@lru_cache(maxsize=None)
def derived_dir():
    """Recompute every summary from the raw results once per process; returns the directory holding them."""
    d = Path(tempfile.mkdtemp(prefix="qmf_derived_"))
    atexit.register(shutil.rmtree, d, True)
    for name, (fn, *inputs) in SUMMARIES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            FUNCS[fn](*[fresh_inputs(i, d) for i in inputs], str(d / name))
    return d


def J(name):
    """A summary recomputed from the raw results (E10's bound file is itself a raw output and is read as stored)."""
    path = derived_dir() / name if name in SUMMARIES else O / name
    with open(path) as fh:
        return json.load(fh)
