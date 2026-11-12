"""E15 (tier 3 under transit reordering): the control and the reordering itself."""
import json

from qmf.experiments.reorder import reorder_trial
from qmf.experiments.tier3 import switch_trial2

E6_FIELDS = ("r", "eps", "p0", "trial", "switch", "W", "L0", "L1", "L2", "att_min", "cells_exact", "onset_err",
             "first_ok", "last_ok", "switch_detected", "switch_err")


def test_f0_is_e6():
    """f = 0 makes exactly E6's vault call: same record, same report."""
    for t in (0, 1):
        new = json.loads(json.dumps(reorder_trial((0.0, 6, 0.6, 0.005, t, 32))))
        old = json.loads(json.dumps(switch_trial2(("vault", 6, 0.6, 0.005, t, 32))))
        assert all(new[k] == old[k] for k in E6_FIELDS)
        assert new["moved"] == 0 and new["tool2_first"] == 0.0 and new["tool1_last"] == 0.0


def test_reordering_moves_round_fN():
    for f in (0.2, 1.0):
        x = reorder_trial((f, 6, 0.6, 0.005, 0, 32))
        N = x["L0"] + x["L1"] + x["L2"]
        assert x["moved"] == round(f * N)
        assert 0.0 <= x["tool2_first"] <= 1.0 and 0.0 <= x["tool1_last"] <= 1.0
