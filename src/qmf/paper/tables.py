"""Paper tables for QMF, built from the raw results of outputs/results.

There are no new runs and no new rules. The output is deterministic: one CSV per table plus tables.md, which holds all
of them. Every number comes from the raw rows (*.jsonl) and E10's computation outputs (e10_bound.json): the summaries
the tables need are recomputed in memory with the analysis code (qmf.analysis.derive);
no stored summary JSON is read. qmf.reproduce checks that the stored summaries equal the recomputed ones.

N90 cells follow qmf.analysis.summaries.fmt:
- a number: the first grid point with success >= 0.9.
- (a,b]: reached only at the horizon.
- >h: censored at horizon h.

usage: python -m qmf.paper.tables [OUT_DIR]   (default outputs/paper_tables)
"""
import csv
import inspect
import sys
from pathlib import Path

import qmf.analysis.derive as derive
from qmf.analysis.summaries import fmt


def J(name):
    """A summary recomputed from the raw results; e10_bound.json (E10's own output) is read as stored."""
    return derive.J(name)


def ci(s):
    if not s:
        return ""
    lo, hi = (str(int(v)) if v != float("inf") else "inf" for v in s["ci90"])
    return f"[{lo}, {hi}]"


def num(x, nd=1):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def ratio_cell(row):
    if row.get("ratio"):
        return num(row["ratio"])
    return f">{row['ratio_lower']:.1f}" if row.get("ratio_lower") else ""


def interp(s):
    return "" if not s or s.get("n90_interp") is None else str(int(round(s["n90_interp"])))


def ci_interp(s):
    if not s or s.get("n90_interp") is None:
        return ""
    lo, hi = (str(int(round(v))) if v != float("inf") else "inf" for v in s["ci90_interp"])
    return f"[{lo}, {hi}]"


def fine_cells(d):
    """E1f fine-grid N90 and CI; a dagger marks a value that fell back to the grid (outside the window)."""
    lo, hi = (str(int(v)) if v != float("inf") else "inf" for v in d["ci90"])
    return f"{int(d['n90'])}{'†' if d['flag'] else ''}", f"[{lo}, {hi}]"


def t1_uniform():
    """E1: uniform fingerprint, vault vs memoryless (uniform Lagrangian). The grid CIs (*_ci90) are limited by the grid
    resolution wherever *_grid_limited is True (endpoints at most one grid step apart; steps 1.14-1.44x). Two finer
    estimates are reported alongside; no rule uses them. *_interp: success fraction interpolated in log n between grid
    points, with the same bootstrap resamples. *_fine (E1f): the same trials and draws, with the decoder also evaluated
    at points at most 1.03x apart in a window around N90, with the same resamples. At p0 = 0.005 the trials of a group
    have different horizons, so each trial's own checkpoints fall between grid points; the grid (and *_interp) reads each
    trial at its last checkpoint at or below n, which delays N90, and *_fine does not (fine / grid N90, median: 0.64
    memoryless, 0.84 vault). At p0 = 0 the checkpoints are the grid and *_fine lies within the *_interp CI in every row."""
    F = {(d["lab"], d["r"], d["eps"], d["p0"]): d["fine"] for d in J("e1f.json")["table"]}
    rows = []
    for x in J("e1.json")["table"]:
        v, m = x["vault"], x["ml"]
        ri = (m["n90_interp"] / v["n90_interp"]) if m.get("n90_interp") and v.get("n90_interp") else None
        fv, fm = F[("vault", x["r"], x["eps"], x["p0"])], F[("ml", x["r"], x["eps"], x["p0"])]
        (vf, vfc), (mf, mfc) = fine_cells(fv), fine_cells(fm)
        rows.append(dict(eps=x["eps"], p0=x["p0"], r=x["r"], vault_n90=fmt(v), vault_ci90=ci(v),
                         vault_grid_limited=v["ci90_grid_limited"], memoryless_n90=fmt(m), memoryless_ci90=ci(m),
                         memoryless_grid_limited=m["ci90_grid_limited"], ratio=ratio_cell(x),
                         vault_n90_interp=interp(v), vault_ci90_interp=ci_interp(v), memoryless_n90_interp=interp(m),
                         memoryless_ci90_interp=ci_interp(m), ratio_interp=num(ri),
                         vault_n90_fine=vf, vault_ci90_fine=vfc, memoryless_n90_fine=mf, memoryless_ci90_fine=mfc,
                         ratio_fine=num(fm["n90"] / fv["n90"])))
    return rows


def t2_weight():
    """E2 + ml_ghz + E5b: fingerprint weight sweep (p0 = 0.005). ml_random marked * comes from E1 (uniform a*)."""
    labs = ["ml_random", "ml_product", "ml_mix", "ml_ghz", "ml_adaptive_ghz"]
    rows = []
    for x in J("e2_e5.json")["table"]:
        st = x["stats"]
        d = dict(r=x["r"], eps=x["eps"], w=x["w"], vault=fmt(st.get("memory")))
        for m in labs:
            d[m] = fmt(st.get(m)) + ("*" if m == "ml_random" and x["e1_ref"] and st.get(m) else "")
        d["best_over_vault"] = ratio_cell(x)
        d["predicted_min_3w_2r"] = x["predicted"]
        rows.append(d)
    return rows


def t3_adaptive():
    """E5b: explore/exploit adaptive GHZ probe (deterministic top-k rerun of E5) vs ml_ghz (w = r)."""
    return [dict(r=x["r"], eps=x["eps"], ml_ghz=fmt(x["ghz"]), ml_ghz_ci90=ci(x["ghz"]),
                 ml_adaptive_ghz=fmt(x["adaptive"]), ml_adaptive_ghz_ci90=ci(x["adaptive"]),
                 speedup=num(x["speedup"], 2)) for x in J("e5.json")["table"]]


def t4_vault_noise():
    """E3 (grid + floor refinement): crossover vault noise p_v* where the vault N90 reaches the memoryless N90 (E1,
    p0 = 0.005). *_fine (E3f): the same crossover with the E1f fine memoryless N90 as the target; the headline
    values. The vault side is read on the grid in both; its trials have different horizons, so its grid N90 comes late
    and p_v* is on the low (conservative) side."""
    rows = []
    for name, fname in (("e3_final_dep.json", "e3f_dep.json"), ("e3_final_deph.json", "e3f_deph.json")):
        F = J(fname)["crossover"]
        for key, c in J(name)["crossover"].items():
            noise, r, eps = key.split(",")
            lo, hi = c["bracket"] if c["bracket"] else (None, None)
            f = F[key]
            flo, fhi = f["bracket"] if f["bracket"] else (None, None)
            rows.append(dict(noise=noise, r=int(r[2:]), eps=float(eps[4:]), pv_star_lo=num(lo, 2), pv_star_hi=num(hi, 2),
                             pv_star_interp=num(c["interpolated"], 3), non_monotone=c["non_monotone"],
                             memoryless_n90=("≥" if c["memoryless_is_lower_bound"] else "") + str(c["memoryless_n90"]),
                             pv_star_lo_fine=num(flo, 2), pv_star_hi_fine=num(fhi, 2),
                             pv_star_interp_fine=num(f["interpolated"], 3), memoryless_n90_fine=str(f["memoryless_n90"])))
    return rows


def t5_tier3():
    """E6 (vault) and E8 (memoryless blind / registry): tool-switch detection on shared instances (p0 = 0.005)."""
    rows = []
    for key, d in J("e6e8.json").items():
        mode, r, eps = key.split(",")
        rows.append(dict(mode=mode, r=int(r[2:]), eps=float(eps[4:]), detect=num(d["detect"], 2),
                         false_alarms=f"{d['false_alarms']}/{d['n_noswitch']}", fa_cp95_upper=num(d["fa_cp95_upper"], 3),
                         cells_exact=num(d.get("cells_exact"), 2), first_ok=num(d.get("first_ok"), 2),
                         last_ok=num(d.get("last_ok"), 2), q90_err_eps2=num(d["q90_eps2"], 2),
                         q90_err_eps_att2=num(d["q90_att"], 2)))
    return rows


def t6_multi():
    """E7: K hidden parities, eps and K unknown to both labs (threshold decoders). The trials of a group have different
    horizons, so N90 values and ratios are +-1 grid step."""
    e7 = J("e7.json")
    rows = []
    for x in e7["table"]:
        dec = e7["decision"][f"K={x['K']},eps_tot={x['eps_tot']}"]
        rows.append(dict(K=x["K"], eps_tot=x["eps_tot"], r=x["r"], vault=fmt(x["vault"]), memoryless=fmt(x["ml"]),
                         ratio=ratio_cell(x), log2_ratio_slope=num(dec["ratio_slope"], 2)))
    return rows


def t7_paired():
    """Self-referenced pairs P vs vault, a*-only N90 in seal units (bundle rerun). At p0 = 0.005 the vault N90 (E1) comes
    late on the grid, so the ratio is +-1 grid step (on the low, conservative side)."""
    return [dict(eps=x["eps"], p0=x["p0"], r=x["r"], vault=fmt(x["vault"]), paired=fmt(x["paired"]),
                 ratio=num(x["ratio"]), predicted_2_over_eps2=num(x["predicted"])) for x in J("pl.json")["paired"]]


def t8_local():
    """Local (unscrambled) tamper q = 0.3, p0 = 0: vault vs product-probe memoryless (bundle rerun)."""
    return [dict(r=x["r"], vault=fmt(x["memory"]), ml_product=fmt(x["ml_product"]), ratio=num(x["ratio"], 2))
            for x in J("pl.json")["local"]]


def t9_references():
    """E9: untouched in-seal references vs the vault (n = 32, p0 = 0.005, E1 instances). Success is (a*, s); for P it is
    a* only. success_at_horizon: fraction of trials correct at the group horizon, with a two-sided 95% Clopper-Pearson
    interval. vault_seal under extend is given R (rates cannot localize it); vault_sparse (r = 4) is not. The trials of
    a group have different horizons, so N90 values and ratios are +-1 grid step."""
    rows = []
    for d in J("e9.json")["table"]:
        lo, hi = d["success_cp95"]
        rows.append(dict(lab=d["lab"], attacker=d["attacker"], r=d["r"], eps=d["eps"], trials=d["trials"],
                         n90=fmt(d["n90"]), ci90=ci(d["n90"]), n90_interp=interp(d["n90"]),
                         success_at_horizon=f"{d['successes_at_horizon']}/{d['trials']}", success_cp95=f"[{lo:.3f}, {hi:.3f}]",
                         horizon=d["horizon"], cells_exact=num(d.get("cells_exact"), 3),
                         mean_rhat_size=num(d.get("rhat_size_mean"), 2)))
    return rows


def t10_coverage():
    """E10: largest number of weight-w elements in an r-qubit Lagrangian (stabilizer group up to signs). exact: all
    Lagrangians (r <= 6) or all labeled graph states (r <= 8); otherwise [search, ILP] brackets the maximum. c* = max /
    (C(r,w) 3^w) is the best coverage of a uniform weight-w fingerprint by any non-adaptive stabilizer probe."""
    rows = []
    B = J("e10_bound.json")["per_r"]
    for r in sorted(B, key=int):
        for w in sorted(B[r]["w"], key=int):
            b = B[r]["w"][w]
            lo, hi = b.get("exact", b.get("search_lower")), b.get("exact", b["ilp_upper"])
            rows.append(dict(r=int(r), w=int(w), paulis=b["M"], max_n_w=str(lo) if lo == hi else f"[{lo}, {hi}]",
                             source=b.get("exact_source", "search/ILP"), ilp_upper=b["ilp_upper"], product=b["product"],
                             ghz="" if b["ghz"] is None else b["ghz"], c_star_upper=f"{b['c_upper']:.4g}",
                             ghz_optimal="" if b["ghz_optimal"] is None else b["ghz_optimal"]))
    return rows


def t11_bound():
    """E10: Fano lower bound N_LB for any non-adaptive stabilizer probe (success 0.9 for (a*, s); includes the
    identification log factor) vs measured N90 (E2, p0 = 0.005; uniform: E1). best_nonadaptive is the best of the
    non-adaptive stabilizer labs run on that family (ml_random, ml_product, ml_mix, ml_ghz); adaptive is
    ml_adaptive_ghz (E5b), to which the bound does not apply. c* is conservative where the best coverage is not known
    exactly (r = 10, w = r: the integer-program upper bound 540 is used; the known maximum is GHZ's 513, see below).
    Uniform rows: *_fine uses the E1f fine vault N90 (at p0 = 0.005 the grid N90 comes late); weight rows have one
    horizon per group and need none. lb_over_vault and lb_over_vault_fine divide a lower bound by a measurement and are
    not proved ratios. Final scope (scope.json): vault_ub is the vault's information-theoretic upper bound with R given
    (8 ln(20 (4^r - 1)) / eps_min^2, eps_min = eps (1 - 4p0/3)^(2r)), so lb_over_vault_ub is a proved ratio. For w = r
    the best coverage is known for every r (maximal full-weight sector length 2^(r-1) + [r even], attained by GHZ: Tran
    et al., PRA 94, 042302 (2016), odd r; Eltschka and Siewert, Quantum 4, 229 (2020), even r); n_lb_theorem uses it."""
    e10 = J("e10.json")
    F = {(d["r"], d["eps"], d["p0"]): d["fine"]["n90"] for d in J("e1f.json")["table"] if d["lab"] == "vault"}
    SW = {(x["r"], x["w"], x["eps"]): x for x in J("scope.json")["weight"]}
    SU = {(x["r"], x["eps"], x["p0"]): x for x in J("scope.json")["uniform"]}
    rows = []
    for x in e10["weight"]:
        s = SW[(x["r"], x["w"], x["eps"])]
        rows.append(dict(family=f"w={x['w']}", r=x["r"], eps=x["eps"], p0=x["p0"], c_star_upper=f"{x['c_upper']:.4g}",
                         n_lb=int(round(x["n_lb"])), vault_n90=x["vault_n90"], lb_over_vault=num(x["lb_over_vault"]),
                         best_nonadaptive=x["best_nonadaptive"] or "", best_over_lb=num(x["best_over_lb"], 2),
                         adaptive=x["adaptive_n90"] or "", vault_n90_fine="", lb_over_vault_fine="",
                         vault_ub=s["vault_ub"], lb_over_vault_ub=num(s["lb_over_ub"], 2),
                         n_lb_theorem="" if s["n_lb_theorem"] is None else int(round(s["n_lb_theorem"]))))
    for x in e10["uniform"]:
        vf = F[(x["r"], x["eps"], x["p0"])]
        s = SU[(x["r"], x["eps"], x["p0"])]
        rows.append(dict(family="uniform", r=x["r"], eps=x["eps"], p0=x["p0"],
                         c_star_upper=f"{(2 ** x['r'] - 1) / (4 ** x['r'] - 1):.4g}", n_lb=int(round(x["n_lb"])),
                         vault_n90=x["vault_n90"], lb_over_vault=num(x["lb_over_vault"]),
                         best_nonadaptive=x["memoryless_n90"] or "", best_over_lb=num(x["ml_over_lb"], 2), adaptive="",
                         vault_n90_fine=vf, lb_over_vault_fine=num(x["n_lb"] / vf),
                         vault_ub=s["vault_ub"], lb_over_vault_ub=num(s["lb_over_ub"], 2), n_lb_theorem=""))
    return rows


def t12_spray():
    """E11 and E12: the spray attacker, which needs no secret and copies E_i onto every cell of group i (n = 32,
    p0 = 0.005, E1 instances, 200 trials). The vault is not given R; inseal is E9's in-seal lab; inseal_oracle is told R
    and the copy groups and reads all of E modulo the diagonal Paulis every unit. product (E12) has no reference: it
    finds the copy groups from Z-basis reads, then reads one copy of each group in the Z basis and another in the X
    basis, which gives the whole E_i every unit. in_C: a* commutes with the diagonal Paulis (XOR of its cell codes is
    0); for a* not in C the in-seal data are independent of (a*, s). success: fraction correct at the group horizon, with
    a two-sided 95% Clopper-Pearson interval. components_exact: the vault merged the copies exactly (vault) or the copy
    groups were found exactly (product). vault_over_obl: vault N90 / E9 oblivious vault N90 (same instances);
    oracle_over_vault_in_C: on the in-C trials; product_over_vault: on the same trials and checkpoints. These groups have
    different horizons per trial, so N90 values are +-1 grid step."""
    e11, e12 = J("e11.json"), J("e12.json")
    A = {(x["r"], x["eps"]): x for x in e11["decision"]["a_vault"]["rows"]}
    C = {(x["r"], x["eps"]): x for x in e11["decision"]["c_oracle"]["rows"]}

    def succ(s):
        return f"{s['successes_at_horizon']}/{s['trials']}" if s else ""

    def cp(s):
        return f"[{s['success_cp95'][0]:.3f}, {s['success_cp95'][1]:.3f}]" if s else ""
    rows = []
    for d in e11["table"]:
        k = (d["r"], d["eps"])
        rows.append(dict(lab=d["lab"], r=d["r"], eps=d["eps"], in_C_fraction=num(d["in_C_fraction"], 2),
                         n90=fmt(d["all"]["n90"]), ci90=ci(d["all"]["n90"]), success=succ(d["all"]), success_cp95=cp(d["all"]),
                         n90_in_C=fmt(d["in_C"]["n90"]), success_in_C=succ(d["in_C"]),
                         success_not_in_C=succ(d["not_in_C"]), success_not_in_C_cp95=cp(d["not_in_C"]),
                         components_exact=num(d.get("components_exact"), 3), mean_rhat_size=num(d.get("rhat_size_mean"), 1),
                         vault_over_obl=num(A[k]["ratio"], 2) if d["lab"] == "vault_seal" else "",
                         oracle_over_vault_in_C=num(C[k]["oracle_over_vault_in_C"], 2) if d["lab"] == "inseal_oracle" else "",
                         product_over_vault="", product_over_vault_interp=""))
    for d in e12["table"]:
        a, i, o = d["product"], d["product_in_C"], d["product_not_in_C"]
        rows.append(dict(lab="product", r=d["r"], eps=d["eps"], in_C_fraction=num(i["trials"] / a["trials"], 2),
                         n90=fmt(a["n90"]), ci90=ci(a["n90"]), success=succ(a), success_cp95=cp(a),
                         n90_in_C=fmt(i["n90"]), success_in_C=succ(i), success_not_in_C=succ(o), success_not_in_C_cp95=cp(o),
                         components_exact=num(d["groups_exact"], 3), mean_rhat_size="", vault_over_obl="",
                         oracle_over_vault_in_C="", product_over_vault=num(d["ratio"], 2),
                         product_over_vault_interp=num(d["ratio_interp"], 2)))
    return rows


def t13_extend():
    """Extend (every cell outside R fully depolarized; no secret needed), p0 = 0.005. vault_given_R: E9 vault_seal under
    extend (100 trials, +-1 grid step). vault_unknown_R: E9 vault_sparse, exact MAP over every Pauli of weight <= r on the
    32 cells (r = 4 only). bound: units that suffice information-theoretically for the vault without R (maximum
    likelihood over the M Paulis of weight 1..r and the sign, error <= 0.1 on every instance: 8 ln(20 M) / eps_min^2,
    eps_min = eps (1 - 4 p0 / 3)^(2 r)); an upper bound on samples, not an efficient decoder. memoryless_fine: E1f fine
    N90 of the per-unit memoryless verifier told R, the same r-cell problem a per-unit verifier faces under extend (Chen
    setting); memoryless_over_bound divides a measurement by a bound, so it is neither a lower nor an upper bound on the
    advantage. fano_lb is E10's bound for every non-adaptive stabilizer per-unit probe
    (valid under extend with R unknown); paired_sign_bound is the same kind of upper bound for a self-referencing
    verifier (P + sign, E13; R unknown). fano_over_bound and fano_over_paired_sign_bound divide a lower bound by an upper
    bound, so they are proved ratios where they exceed 1."""
    X13 = {(x["r"], x["eps"]): x for x in J("e13.json")["extend"]}
    rows = []
    for x in J("e12.json")["extend"]:
        y = X13[(x["r"], x["eps"])]
        rows.append(dict(r=x["r"], eps=x["eps"], vault_given_R=x["vault_given_R_n90"] or "",
                         vault_unknown_R=x["vault_unknown_R_n90"] or "", M=x["M"], bound=x["bound_units"],
                         memoryless_fine=x["memoryless_fine_n90"], memoryless_over_bound=num(x["memoryless_over_bound"]),
                         fano_lb=int(round(y["fano_lb"])), fano_over_bound=num(y["fano_over_vault_bound"]),
                         paired_sign_bound=y["paired_sign_bound"],
                         fano_over_paired_sign_bound=num(y["fano_over_paired_sign_bound"], 2)))
    return rows


def t14_self_ref():
    """E13: self-referencing verifier, P + sign (20 trials, E1 instances, R given, eps and p0 known). Phase 1 is P
    (cell-wise Bell pairs across two units, measured together; a pair counts as 2 units); phase 2 reads m single units
    prepared in an eigenstate of the learned a^ (m = ceil(2 ln 100 / (eps (1 - 4p0/3)^r)^2), included in the cost).
    Success is (a*, s) on both sides. paired_a: P's a*-only N90 (pl.json). vault: E1 vault N90 for (a*, s), fine (E1f)
    and grid. Each group has one horizon, so the P + sign N90 is read without the late-grid effect; the vault's grid
    value at p0 = 0.005 is +-1 grid step."""
    rows = []
    for x in J("e13.json")["table"]:
        s = x["paired_sign"]
        lo, hi = s["success_cp95"]
        rows.append(dict(r=x["r"], eps=x["eps"], p0=x["p0"], m=x["m"], paired_sign=fmt(s["n90"]), ci90=ci(s["n90"]),
                         success_at_horizon=f"{s['successes_at_horizon']}/{s['trials']}", success_cp95=f"[{lo:.3f}, {hi:.3f}]",
                         sign_ok=num(x["sign_ok"], 2), paired_a=x["paired_a_n90"],
                         paired_sign_over_paired_a=num(x["ratio_to_paired"], 3), vault_fine=x["vault_n90_fine"],
                         vault_grid=x["vault_n90_grid"], over_vault_fine=num(x["over_vault_fine"]),
                         over_vault_grid=num(x["over_vault_grid"])))
    return rows


def t15_seal():
    """E14: seal-noise sweep (seal depolarizing p0; vault noise p_v = 0.005; E1 instances; vault 200 trials, per-unit
    50). Every trial of a group shares the group's checkpoints (geometric, ratio <= 1.03, from window_lo to window_hi),
    so N90 is read without the late-grid effect; flag 'below' = reached at the first checkpoint, 'above' = not reached.
    rho = N90(p0) / N90(0.005) on the same instances, with a paired 90% bootstrap CI; band = predicted
    [A_{3r/4}(p0), A_r(p0)], A_w = ((1 - 4(0.005)/3) / (1 - 4 p0 / 3))^(2w); P1 = rho's CI overlaps the band.
    seal_T_over_tau = -ln(1 - 4 p0 / 3): the storage time of a seal cell of coherence time tau_s at that p0."""
    def n(v):
        return "inf" if v == float("inf") else str(int(v))
    rows = []
    for x in J("e14.json")["table"]:
        rows.append(dict(lab=x["lab"], r=x["r"], eps=x["eps"], p0=x["p0"], trials=x["trials"], window_lo=x["lo"],
                         window_hi=x["hi"], n90=n(x["n90"]), flag=x["flag"], ci90=f"[{n(x['ci90'][0])}, {n(x['ci90'][1])}]",
                         rho=num(x.get("rho"), 2), rho_ci90=(f"[{x['rho_ci90'][0]:.2f}, {x['rho_ci90'][1]:.2f}]"
                                                             if "rho" in x else ""),
                         band=(f"[{x['band'][0]:.2f}, {x['band'][1]:.2f}]" if "band" in x else ""),
                         P1=x.get("P1", ""), seal_T_over_tau=f"{x['seal_T_over_tau']:.4f}"))
    return rows


def t16_reorder():
    """E15 (post hoc): tier 3 under transit reordering. E6's vault records (300 switch and 300 no-switch trials per
    setting) regenerated from their seeds; a fraction f of the units, at positions drawn uniformly, is permuted among
    those positions before the report reads the record (f = 1: full permutation). detect / false alarms as in E6, with
    the one-sided 95% Clopper-Pearson upper bound; E6_rule = detection >= 0.9, false-alarm upper bound <= 0.05 and cells
    exact >= 0.9. tool2_first / tool1_last: mean fraction of tool-2 units in the first W units after the true onset and
    of tool-1 units in the last W units (switch trials). q90_switch_err: 90% quantile of |switch unit - L0 - L1| over
    detected switches, measured against the attack order (ill-defined once the record is mixed; reported only)."""
    rows = []
    for x in J("e15.json")["table"]:
        rows.append(dict(f=x["f"], r=x["r"], eps=x["eps"], detect=num(x["detect"], 3),
                         false_alarms=f"{x['false_alarms']}/{x['n_noswitch']}", fa_cp95_upper=num(x["fa_cp95_upper"], 3),
                         cells_exact=num(x["cells_exact"], 3), first_ok=num(x["first_ok"], 3),
                         last_ok=num(x["last_ok"], 3), tool2_first=num(x["tool2_first"], 3),
                         tool1_last=num(x["tool1_last"], 3),
                         q90_switch_err="" if x["q90_switch_err"] is None else str(round(x["q90_switch_err"])),
                         E6_rule="PASS" if x["passes_E6"] else "FAIL"))
    return rows


TABLES = [("t1_uniform", t1_uniform), ("t2_weight", t2_weight), ("t3_adaptive", t3_adaptive),
          ("t4_vault_noise", t4_vault_noise), ("t5_tier3", t5_tier3), ("t6_multi", t6_multi),
          ("t7_paired", t7_paired), ("t8_local", t8_local), ("t9_references", t9_references),
          ("t10_coverage", t10_coverage), ("t11_bound", t11_bound), ("t12_spray", t12_spray),
          ("t13_extend", t13_extend), ("t14_self_ref", t14_self_ref), ("t15_seal", t15_seal),
          ("t16_reorder", t16_reorder)]


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# QMF paper tables", "", "Generated by `python -m qmf.paper.tables` from the raw results in "
          "`outputs/results` (rows and E10's bound file; every summary is recomputed in memory "
          "by `qmf.analysis.derive`, no stored summary is read). N90 cells: number = first grid point with "
          "success >= 0.9; "
          "`(a,b]` = reached only at the horizon; `>h` = censored at horizon h.", "",
          "Grid resolution: N90 and its 90% bootstrap CI are read on a 40-point geometric grid (neighbouring points "
          "1.14-1.44x apart), so most CIs span at most one grid step and some are degenerate (e.g. [1221, 1221]). "
          "Interpolated N90 values (`*_interp`: success fraction interpolated linearly in log n between the two grid "
          "points around 0.9, with the same bootstrap resamples) resolve within a step. For E1 (`t1_uniform`), "
          "`*_fine` columns give N90 measured on points at most 1.03x apart around N90 (E1f: the same trials and "
          "draws, decoder evaluated there too). Both are reported alongside; every decision rule uses the grid N90.", "",
          "Late grid reads: where the trials of a group have different horizons (E1 and P at p0 = 0.005, E3, E7, E9, "
          "E11, E12), each trial's checkpoints fall between grid points and the grid reads it at its last checkpoint, so the "
          "grid N90 comes late (E1f: fine / grid N90 median 0.64 memoryless, 0.84 vault at p0 = 0.005). Headline numbers "
          "use fine values where they exist (`t1_uniform`, `t4_vault_noise`, `t11_bound`); other N90 values in such "
          "groups are +-1 grid step.", ""]
    for name, fn in TABLES:
        rows = fn()
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        # cleandoc: Python 3.13 dedents docstrings at compile time, 3.12 does not; this gives the same text on both
        md += [f"## {name}", "", inspect.cleandoc(fn.__doc__), "", "| " + " | ".join(rows[0]) + " |",
               "|" + "---|" * len(rows[0])]
        md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
        md.append("")
    with open(out / "tables.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))
    print(f"wrote {len(TABLES)} tables to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/paper_tables")
