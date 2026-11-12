"""E14 (seal-noise sweep): instances, shared checkpoints, the trial, and the unchanged default checkpoints of the
labs."""
import json
import math
from pathlib import Path

import numpy as np

import qmf.core as q
import qmf.experiments.seal_noise as e14

O = Path("outputs/results")


def test_e14_instances_are_e1s_at_p0_0005():
    """E14's (a*, s) for trial t equal E1's at p0 = 0.005 (both labs' rows), so every p0 sees E1's instances."""
    n = 0
    with open(O / "e1_vault.jsonl") as fh:
        for line in fh:
            x = json.loads(line)
            if x["p0"] == 0.005 and x["r"] <= 6:
                assert e14.instance(x["r"], x["eps"], x["trial"]) == (x["a"], x["s"])
                n += 1
    assert n > 100


def test_e14_points_window_and_step():
    for n_base, r, p0 in ((1456, 6, 0.1), (98, 8, 0.02), (440015, 8, 0.05)):
        pts = e14.points(n_base, r, p0)
        assert pts == sorted(set(pts))
        assert abs(pts[0] - n_base / 2) <= 1 and abs(pts[-1] - 2 * n_base * e14.factor(r, p0)) <= 1
        steps = np.array(pts[1:]) / np.array(pts[:-1])
        assert steps.max() <= e14.STEP + 2 / pts[0]  # geometric with ratio <= 1.03 up to integer rounding


def test_e14_factor():
    assert e14.factor(10, 0.005) == 1.0
    assert math.isclose(e14.factor(4, 0.05), ((1 - 0.02 / 3) / (1 - 0.2 / 3)) ** 8)


def test_e14_trial_small_and_deterministic():
    for lab in e14.LABS:
        pts = e14.points(60 if lab == "vault" else 400, 4, 0.05)
        job = (lab, 4, 0.6, 0.05, 3, pts, 60 if lab == "vault" else 400)
        a, b = e14.e14_trial(job), e14.e14_trial(job)
        assert a == b
        assert [c["n"] for c in a["curve"]] == pts and a["n_max"] == pts[-1]
        assert (a["a"], a["s"]) == e14.instance(4, 0.6, 3)
        assert a["pv"] == (0.005 if lab == "vault" else None)


def test_lab_default_checkpoints_unchanged():
    """points=None keeps every earlier experiment's checkpoints: the same rows as passing checkpoints(n_max)."""
    a, s = 77, 1
    for fn, args in ((q.run_vault, (4, a, s, 0.6, 0.005, None, "dep", 3000)),
                     (q.run_memoryless, ("scramble", "ml_random", 4, a, s, 0.6, 0.6, 0.005, 3000))):
        r1 = fn(*args, np.random.default_rng(5), np.random.default_rng(6))
        r2 = fn(*args, np.random.default_rng(5), np.random.default_rng(6), points=q.checkpoints(3000))
        assert r1 == r2
