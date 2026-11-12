"""Headline numbers of the paper as LaTeX macros, computed from the raw results
(outputs/numbers/qmf_numbers.tex).

The paper's headline numbers are these macros. Each macro is computed here from the
summaries that qmf.analysis.derive recomputes from the raw rows (no stored summary is read), with the rule
written next to it. Rounding:
- a proved lower bound ("at least") is rounded DOWN, so the printed value is never larger than the bound;
- a measured range is rounded to the stated precision;
- an en dash joins the ends of a range.
qmf.reproduce checks that the stored file equals a fresh one.

usage: python -m qmf.paper.numbers [OUT.tex]   (default outputs/numbers/qmf_numbers.tex)
"""
import json
import math
import sys
from pathlib import Path

import qmf.analysis.derive as derive

J = derive.J


def fmt_num(x, nd):
    return f"{x:,.{nd}f}".replace(",", "{,}") if nd else f"{int(round(x)):,}".replace(",", "{,}")


def rng(vals, nd=0):
    """Measured range, rounded to nd decimals."""
    lo, hi = min(vals), max(vals)
    a, b = fmt_num(lo, nd), fmt_num(hi, nd)
    return a if a == b else f"{a}--{b}"


def floor_to(x, nd):
    f = 10 ** nd
    return math.floor(x * f + 1e-12) / f


def lb_rng(vals, nd=0):
    """Range of proved lower bounds, each rounded down."""
    lo, hi = floor_to(min(vals), nd), floor_to(max(vals), nd)
    a, b = (f"{lo:.{nd}f}", f"{hi:.{nd}f}") if nd else (f"{int(lo):,}".replace(",", "{,}"), f"{int(hi):,}".replace(",", "{,}"))
    return a if a == b else f"{a}--{b}"


def frac(k, n):
    return f"{k:,}/{n:,}".replace(",", "{,}")


def macros():
    m = {}  # name -> (value, source and rule)
    H = J("headline.json")
    m["EoneRatioFine"] = (rng([x["fine"] for x in H["e1_ratio_r10"]]),
                          "headline.json e1_ratio_r10: memoryless/vault fine N90 at r=10, eps and p0 both")
    m["EoneRatioGrid"] = (rng([x["grid"] for x in H["e1_ratio_r10"]]), "same, grid N90")
    m["LbOverMeasUniform"] = (rng([x["lb_over_vault_fine"] for x in H["e10_uniform_r10"]]),
                              "headline.json e10_uniform_r10: Fano bound / measured fine vault N90 (not a proof)")
    v10 = {(x["eps"], x["p0"]): x["vault_fine"] for x in H["e10_uniform_r10"]}
    m["VaultNninetyRtenLow"] = (fmt_num(v10[(0.15, 0.005)], 0), "vault fine N90, r=10, eps=0.15, p0=0.005")
    m["VaultNninetyRtenHigh"] = (fmt_num(v10[(0.6, 0.005)], 0), "vault fine N90, r=10, eps=0.6, p0=0.005")
    F1 = [x for x in J("e1f.json")["table"] if x["lab"] == "vault" and x["p0"] == 0.005]
    m["VaultConst"] = (rng([x["fine"]["n90"] * x["eps"] ** 2 / x["r"] for x in F1], 1),
                       "e1f.json vault fine N90 * eps^2 / r, r=4..10, both eps, p0=0.005 (the 'c r/eps^2' constant)")
    # E3f crossovers: lowest interpolated crossing (fine target) to highest bracket end
    for kind, name in (("dep", "Dep"), ("deph", "Deph")):
        rows = [x for x in H["e3"] if x["key"].startswith(kind + ",")]
        lo = min(x["interp_fine"] for x in rows if x["interp_fine"] is not None)
        hi = max(x["bracket_fine"][1] for x in rows)
        m[f"Cross{name}"] = (f"{lo:.2f}--{hi:.2f}", f"headline.json e3 ({kind}): min interpolated fine crossing to max bracket end")
    S = J("scope.json")
    m["ProvedUniform"] = (lb_rng([x["lb_over_ub"] for x in S["uniform"] if x["r"] == 10 and x["p0"] == 0.005]),
                          "scope.json uniform r=10 p0=0.005: Fano bound / vault upper bound (proved, rounded down)")
    m["ProvedUniformMin"] = (lb_rng([min(x["lb_over_ub"] for x in S["uniform"] if x["r"] == 10 and x["p0"] == 0.005)]),
                             "the smaller of the two (both eps): a lower bound valid in both settings")
    wr = [x for x in S["weight"] if x["r"] == 10 and x["w"] == 10]
    wh = [x for x in S["weight"] if x["r"] == 10 and x["w"] == 5]
    m["ProvedWr"] = (lb_rng([x["lb_over_ub"] for x in wr], 1), "scope.json w=r=10: Fano bound (ILP c*) / vault upper bound")
    m["ProvedWrThm"] = (lb_rng([x["lb_theorem_over_ub"] for x in wr], 1), "same with the theorem c* = 513/3^10")
    m["ProvedWhalf"] = (lb_rng([x["lb_over_ub"] for x in wh], 1), "scope.json w=5, r=10")
    m["LbOverMeasWr"] = (rng([x["lb_over_vault_measured"] for x in wr]), "Fano bound / measured vault N90, w=r=10 (not a proof)")
    m["LbOverMeasWhalf"] = (rng([x["lb_over_vault_measured"] for x in wh]), "same, w=5")
    E10 = J("e10.json")
    ad = [x["adaptive_n90"] / x["n_lb"] for x in E10["weight"] if x["w"] == x["r"] and x["adaptive_n90"]]
    m["AdaptiveOverLb"] = (rng(ad, 2), "e10.json weight w=r: adaptive GHZ N90 / Fano bound")
    m["WhalfMeas"] = (rng(E10["decision"]["claim3_w_half"]["measured_over_vault_r10"].values()),
                      "e10.json claim3: best non-adaptive / vault N90 at r=10, w=5")
    E25 = J("e2_e5.json")["decision"]
    m["WrMeas"] = (rng([v["ratio_r10"] for v in E25.values()]), "e2_e5.json: best stabilizer probe / vault N90, w=r=10")
    m["WrSlope"] = (rng([v["slope"] for v in E25.values()], 2), "e2_e5.json: log2 slope of that ratio in r")
    m["EsevenRatio"] = (rng([v["ratio_r8"] for v in J("e7.json")["decision"].values()]),
                        "e7.json: memoryless/vault N90 at r=8, K<=3, eps unknown")
    PL = J("pl.json")["decision"]
    m["LocalRatio"] = (rng(PL["local"]["ratios"], 1), "pl.json local: ml_product/vault N90")
    E9 = J("e9.json")
    d = E9["decision"]
    m["InsealObl"] = (rng([x["inseal_over_vault"] for x in d["a_oblivious"]["rows"]], 1), "e9.json (a): in-seal/vault N90, oblivious")
    m["ExtendGivenR"] = (rng([x["ratio"] for x in d["c_extend"]["vault_rows"]], 2), "e9.json (c): vault given R, extend/obl N90")
    m["SparseRatio"] = (rng([x["ratio"] for x in d["c_extend"]["sparse_rows"]], 1), "e9.json (c): vault_sparse (R unknown, r=4) / E1 vault N90")
    m["MirrorVault"] = (rng([x["ratio"] for x in d["b_mirror"]["vault_rows"]], 2), "e9.json (b): vault mirror/obl N90")
    for lab, att, name in (("inseal", "mirror", "MirrorInseal"), ("paired", "mirror", "MirrorPaired"),
                           ("inseal", "extend", "ExtendInseal")):
        rows = [x for x in E9["table"] if x["lab"] == lab and x["attacker"] == att]
        k, n = max(x["successes_at_horizon"] for x in rows), max(x["trials"] for x in rows)
        assert all(x["trials"] == n for x in rows)
        m[name] = (frac(k, n), f"e9.json table: most successes at the horizon in any setting, {lab} under {att}")
    E11 = J("e11.json")["decision"]
    c = E11["c_oracle"]["rows"]
    tot = sum(x["not_in_C_trials"] for x in c)
    hits = sum(round(x["not_in_C_success"] * x["not_in_C_trials"]) for x in c)
    m["OracleNotInC"] = (frac(hits, tot), "e11.json (c1): in-seal oracle successes at the horizon, a* not in C")
    m["OracleInCRatio"] = (rng([x["oracle_over_vault_in_C"] for x in c], 2), "e11.json (c2): oracle/vault N90 on a* in C")
    E12 = J("e12.json")
    T12 = E12["table"]
    m["ProductRatio"] = (rng([x["ratio"] for x in T12], 2), "e12.json: product/vault N90 (grid)")
    m["ProductRatioInterp"] = (rng([x["ratio_interp"] for x in T12], 2), "e12.json: same, interpolated")
    m["ProductSuccess"] = (frac(sum(x["product"]["successes_at_horizon"] for x in T12), sum(x["product"]["trials"] for x in T12)),
                           "e12.json: product successes at the horizon")
    X12 = {(x["r"], x["eps"]): x for x in E12["extend"]}
    m["VaultBoundRtenLow"] = (fmt_num(X12[(10, 0.15)]["bound_units"], 0), "e12.json extend: vault bound, R unknown, r=10, eps=0.15")
    m["VaultBoundRtenHigh"] = (fmt_num(X12[(10, 0.6)]["bound_units"], 0), "same, eps=0.6")
    E13 = J("e13.json")
    T13 = E13["table"]
    for eps, name, nd in ((0.15, "Low", 0), (0.6, "High", 1)):
        sel = [x for x in T13 if x["eps"] == eps]
        m[f"PsignOverVaultFine{name}"] = (rng([x["over_vault_fine"] for x in sel], nd), f"e13.json: P+sign/vault fine N90, eps={eps}, p0 both")
        m[f"PsignOverVaultGrid{name}"] = (rng([x["over_vault_grid"] for x in sel], nd), f"same, grid vault N90")
    m["PsignSignCost"] = (rng([x["ratio_to_paired"] for x in T13], 2), "e13.json: P+sign / P (a only) N90")
    m["PsignSuccess"] = (frac(sum(x["paired_sign"]["successes_at_horizon"] for x in T13),
                              sum(x["paired_sign"]["trials"] for x in T13)), "e13.json: P+sign successes at the horizon")
    X13 = {(x["r"], x["eps"]): x for x in E13["extend"]}
    m["PsignBoundLow"] = (f"{X13[(4, 0.15)]['paired_sign_bound'] / 1e6:.2f}--{X13[(10, 0.15)]['paired_sign_bound'] / 1e6:.2f}",
                          "e13.json extend: P+sign bound, R unknown, eps=0.15, r=4 to 10, in millions")
    m["PsignBoundHigh"] = (f"{X13[(4, 0.6)]['paired_sign_bound'] / 1e3:.1f}--{X13[(10, 0.6)]['paired_sign_bound'] / 1e3:.1f}",
                           "same, eps=0.6, in thousands")
    m["ProvedPsignRten"] = (lb_rng([X13[(10, 0.6)]["fano_over_paired_sign_bound"]], 1), "e13.json: Fano / P+sign bound, r=10, eps=0.6")
    m["ProvedPsignReight"] = (lb_rng([X13[(8, 0.6)]["fano_over_paired_sign_bound"]], 1), "same, r=8")
    m["ProvedExtendPerUnit"] = (lb_rng([X13[(10, e)]["fano_over_vault_bound"] for e in (0.15, 0.6)]),
                                "e13.json: Fano / vault bound (R unknown), r=10")
    # crossover between the per-unit strategy and P + sign (grid N90, eps = 0.15, p0 = 0.005; fig2(b))
    ml = {x["r"]: x["ml"]["n90"] for x in J("e1.json")["table"] if x["eps"] == 0.15 and x["p0"] == 0.005}
    ps = {x["r"]: x["paired_sign_n90"] for x in T13 if x["eps"] == 0.15 and x["p0"] == 0.005}
    cheaper = [r for r in sorted(ps) if ml[r] < ps[r]]
    assert cheaper == list(range(min(ps), max(cheaper) + 1)), cheaper  # one crossover
    m["CrossR"] = (str(max(cheaper)), "largest r at which the per-unit N90 is below P+sign's (eps=0.15, p0=0.005)")
    E68 = J("e6e8.json")
    vault = {k: v for k, v in E68.items() if k.startswith("vault,")}
    m["TierThreeVaultDetect"] = (f"{min(v['detect'] for v in vault.values()):.2f}", "e6e8.json: lowest vault detection rate")
    fa = {(v["false_alarms"], v["n_noswitch"]) for v in vault.values()}
    assert len(fa) == 1
    m["TierThreeVaultFA"] = (frac(*fa.pop()), "e6e8.json: vault false alarms per setting")
    m["TierThreeBlind"] = (f"{max(v['detect'] for k, v in E68.items() if k.startswith('ml_blind')):.2f}",
                           "e6e8.json: highest blind-detector detection rate")
    m["TierThreeRegistry"] = (f"{min(v['detect'] for k, v in E68.items() if k.startswith('ml_library')):.2f}",
                              "e6e8.json: lowest registry-detector detection rate")
    ratios = [E68["ml_library," + k.split(",", 1)[1]]["q90_att"] / v["q90_att"] for k, v in vault.items()]
    m["TierThreeTimeErr"] = (rng(ratios, 1), "e6e8.json: registry / vault q90 switch-time error (attenuation-scaled), per setting")
    # circuit checks (outputs/paper_figure/circuits/circuit_check.json; rerun by qmf.reproduce where qiskit exists)
    with open("outputs/paper_figure/circuits/circuit_check.json") as fh:
        cc = json.load(fh)["checks"]
    law = [x["max_abs_dev"] for x in cc if x["expected"] == "0" and "coherent" not in x["check"]]
    m["CircuitChecks"] = (str(len(cc)), "circuit_check.json: number of exact circuit-versus-law checks")
    m["CircuitLawChecks"] = (str(len(law)), "same, the checks that compare a circuit with the sampler's law")
    m["CircuitMaxDev"] = (sci_up(max(law)), "largest deviation among those, rounded up to one significant digit")
    nf = next(x for x in cc if x["check"] == "coherent attack, no frame")
    fa = next(x for x in cc if x["check"] == "coherent attack, frame avg")
    m["CircuitNoFrame"] = (f"{nf['max_abs_dev']:.3f}", "coherent attack without the frame: deviation from the Pauli law")
    m["CircuitFrameAvg"] = (sci_up(fa["max_abs_dev"]), "same attack averaged over the 16 frames, rounded up")
    # ---- resources, storage times and the explicit constant of Proposition 1 ----
    # (a) resources for the fixed scenario: units the attacker must touch before 90% attribution (fine N90, p0 = 0.005)
    # and the vault inventory n * N90 for an n = 32-cell seal when all of them are outstanding at once.
    n_cells = 32
    fine = {(x["lab"], x["r"], x["eps"]): x["fine"]["n90"] for x in J("e1f.json")["table"] if x["p0"] == 0.005}
    rows = []
    for r in (4, 6, 8, 10):
        for eps in (0.15, 0.6):
            v, u = fine[("vault", r, eps)], fine[("ml", r, eps)]
            rows.append(f"{r} & {eps:g} & {fmt_num(v, 0)} & {fmt_num(n_cells * v, 0)} & {fmt_num(u, 0)} & {fmt_num(u / v, 0)} \\\\")
    m["ResourceRows"] = ("\n".join(rows), "e1f.json fine N90 (p0=0.005): r & eps & vault N90 & vault qubits 32*N90 & "
                         "per-unit N90 & ratio; one table row per (r, eps)")
    m["VaultQubitsRtenLow"] = (fmt_num(n_cells * fine[("vault", 10, 0.15)], 0), "32 x vault fine N90, r=10, eps=0.15, p0=0.005")
    m["VaultQubitsRtenHigh"] = (fmt_num(n_cells * fine[("vault", 10, 0.6)], 0), "32 x vault fine N90, r=10, eps=0.6, p0=0.005")
    m["PerUnitNninetyRtenLow"] = (fmt_num(fine[("ml", 10, 0.15)], 0), "per-unit fine N90, r=10, eps=0.15, p0=0.005")
    m["PerUnitNninetyRtenHigh"] = (fmt_num(fine[("ml", 10, 0.6)], 0), "per-unit fine N90, r=10, eps=0.6, p0=0.005")
    # (b) storage-time limits under exponential decay (allowed maxima: rounded DOWN)
    m["LifeVaultDep"] = (f"{floor_to(-math.log(1 - 4 / 3 * 0.2), 3):.3f}", "T/tau with p_v=(3/4)(1-exp(-T/tau)) <= 0.2 (vault)")
    m["LifeVaultDeph"] = (f"{floor_to(-math.log(1 - 2 * 0.15), 3):.3f}", "T/T2 with p_v=(1/2)(1-exp(-T/T2)) <= 0.15 (vault)")
    m["LifeSeal"] = (f"{floor_to(-math.log(1 - 4 / 3 * 0.005), 4):.4f}",
                     "T/tau_s with (1-4p0/3) = exp(-T/tau_s), p0=0.005: the seal-cell noise assumed in every run")
    # (d) Proposition 1 with the explicit constant of Chen et al. (arXiv 2309.13461, Supplemental Material Sec. S3):
    # (S69) E_a TVD <= N' eps^2 2^r/(4^r-1) (1 + 2 sqrt f(eps)), f from (S67), for N' channel uses and eigenvalue
    # eps = eps_0 <= 1/3. Our player (Appendix A) reaches E_a TVD >= 0.8 with N' = N + m, m = ceil(8 ln 20 / eps^2), so
    # N >= 0.8 (4^r-1) / (2^r eps^2 (1 + 2 sqrt f(eps))) - m for every per-unit verifier (any p0, by the noise argument).
    import qmf.analysis.summaries as rev

    def chen_f(e):
        return 0.5 * ((2 / (1 - e * e)) ** 2 + 8 / ((1 - e) ** 2 * (1 - 2 * e - e * e)))

    def chen_lb(r, e):
        return 0.8 * (4 ** r - 1) / (2 ** r * e * e * (1 + 2 * math.sqrt(chen_f(e)))) - math.ceil(8 * math.log(20) / e ** 2)
    lb10, ub10 = chen_lb(10, 0.15), rev.vault_ub(10, 0.15, 0.005)
    m["ChenLbRten"] = (fmt_num(math.floor(lb10), 0),
                       "explicit per-unit lower bound from Chen et al. (S67, S69) via App. A, r=10, eps=0.15 (rounded down)")
    m["ChenVaultUbRten"] = (fmt_num(ub10, 0), "vault upper bound with R given (Prop. 3.1), r=10, eps=0.15, p0=0.005")
    r_star = min(r for r in range(2, 40) if chen_lb(r, 0.15) > rev.vault_ub(r, 0.15, 0.005))
    m["ChenRstar"] = (str(r_star), "smallest r at which that lower bound exceeds the vault upper bound (eps=0.15, p0=0.005)")
    # (e) E14, seal-noise sweep: rho = N90(p0) / N90(0.005), vault noise 0.005
    E14 = J("e14.json")
    t14 = {(x["lab"], x["r"], x["eps"], x["p0"]): x for x in E14["table"]}
    for p0, nm in ((0.02, "Two"), (0.05, "Five"), (0.1, "Ten")):
        m[f"SealRho{nm}"] = (rng([t14[("vault", 10, e, p0)]["rho"] for e in (0.15, 0.6)], 1),
                             f"e14.json: vault rho at p0={p0}, r=10, both eps")
        m[f"SealRho{nm}PerUnit"] = (rng([t14[("ml", 8, e, p0)]["rho"] for e in (0.15, 0.6)], 1),
                                    f"e14.json: per-unit rho at p0={p0}, r=8, both eps")
        m[f"SealLife{nm}"] = (f"{floor_to(-math.log(1 - 4 * p0 / 3), 4):.4f}",
                              f"T/tau_s with (1-4p0/3) = exp(-T/tau_s) at p0={p0} (rounded down)")
    m["SealLifeGainFive"] = (f"{math.floor(math.log(1 - 4 * 0.05 / 3) / math.log(1 - 4 * 0.005 / 3))}",
                             "seal lifetime at p0=0.05 over that at p0=0.005 (rounded down)")
    m["SealRatioChange"] = (rng([x["change"] for x in E14["ratio"]], 2),
                            "e14.json: (per-unit/vault N90)(p0) / (same)(0.005), r=6,8, p0=0.02-0.1, both eps")
    D = E14["decision"]
    m["SealPOne"] = (f"{D['P1_pass']}/{D['P1_groups']}", "e14.json: groups passing P1 (attenuation band)")
    m["SealPTwo"] = (f"{D['P2_pass']}/{D['P2_groups']}", "e14.json: groups passing P2 (ratio within x1.5)")
    m["SealPThree"] = (f"{D['P3_pass']}/{D['P3_groups']}", "e14.json: groups passing P3 (control vs E1f)")
    fails = [x for x in E14["table"] if x.get("P1") is False]
    assert len(fails) == D["P1_groups"] - D["P1_pass"]
    if fails:
        f0 = fails[0]
        m["SealFailRho"] = (f"{f0['rho']:.2f} [{f0['rho_ci90'][0]:.2f}, {f0['rho_ci90'][1]:.2f}]",
                            f"e14.json: the P1 failure ({f0['lab']}, r={f0['r']}, eps={f0['eps']}, p0={f0['p0']}): rho [CI]")
        m["SealFailBand"] = (f"[{f0['band'][0]:.2f}, {f0['band'][1]:.2f}]", "same: predicted band")
    # (f) E15, tier 3 under transit reordering (post hoc): E6's vault
    # records with a fraction f of the units permuted; E6's rule per (f, r, eps)
    E15 = J("e15.json")
    D15, T15 = E15["decision"], E15["table"]
    assert D15["C0_ok"] and E15["control"]["mismatches"] == 0
    at = lambda f, k: [d[k] for d in T15 if d["f"] == f]
    m["ReorderControl"] = (fmt_num(E15["control"]["rows"], 0), "e15.json: rows at f = 0, all equal to E6's stored rows")
    m["ReorderFstar"] = (f"{round(D15['f_star'] * 100, 6):g}\\%",
                         "e15.json: largest fraction reordered at which all six settings keep E6's rule, in percent")
    m["ReorderDetectFive"] = (rng(at(0.05, "detect"), 2), "e15.json: detection at f = 0.05, six settings")
    m["ReorderFailTwenty"] = (str(sum(not p for p in at(0.2, "passes_E6"))), "e15.json: settings failing E6's rule at f = 0.2")
    m["ReorderDetectTwenty"] = (rng(at(0.2, "detect"), 2), "e15.json: detection at f = 0.2, six settings")
    assert set(at(0.2, "n_noswitch")) == set(at(1.0, "n_noswitch")) == {300}
    m["ReorderFATwenty"] = (frac(max(at(0.2, "false_alarms")), 300), "e15.json: most false alarms at f = 0.2 (of 300)")
    m["ReorderDetectFull"] = (rng(at(1.0, "detect"), 2), "e15.json: detection at f = 1 (full permutation)")
    m["ReorderFAFull"] = (rng(at(1.0, "false_alarms")), "e15.json: false alarms (of 300) at f = 1, six settings")
    Dg = E15["diagnosis"]
    assert Dg["false_alarms_positive_sign"] == 0
    m["ReorderFATotal"] = (fmt_num(Dg["false_alarms_total"], 0),
                           "e15.json: false alarms over all f > 0 groups, all from negative-sign fingerprints (post hoc)")
    # (c) case study (fig9): the pre-registered E6 vault trial, regenerated from its seed and checked against the stored row
    import qmf.paper.figures as figs
    cs = figs.case_study(**figs.CASE)
    I, rep = cs["I"], cs["rep"]
    sgn = lambda s: "+" if s > 0 else "-"
    m["CaseUnits"] = (str(cs["N"]), "fig9 case study (E6 vault, r=6, eps=0.6, p0=0.005, trial 0): units in the record")
    m["CaseWindow"] = (str(cs["W"]), "same: tier-3 window W = 10 r / eps^2")
    m["CaseOnset"] = (str(I["L0"]), "same: true onset (found exactly)")
    assert rep["onset"] == I["L0"] and rep["cells"] == I["R"].tolist()
    m["CaseSwitchTrue"] = (str(I["L0"] + I["L1"]), "same: true switch unit")
    m["CaseSwitchFound"] = (str(rep["switch"]), "same: switch unit found by tier 3")
    m["CaseToolOne"] = (f"${sgn(I['s1'])}$\\texttt{{{figs.pauli_label(I['a1'], 6)}}}", "same: tool 1 fingerprint (sign, Pauli)")
    m["CaseToolTwo"] = (f"${sgn(cs['t2'])}$\\texttt{{{figs.pauli_label(cs['b2'], 6)}}}", "same: tool 2 fingerprint")
    return m


def sci_up(x):
    """x rounded UP to one significant digit, as LaTeX math a\\times10^{e}."""
    e = math.floor(math.log10(x))
    a = math.ceil(x / 10 ** e - 1e-12)
    if a == 10:
        a, e = 1, e + 1
    return f"{a}\\times10^{{{e}}}"


def main(out):
    lines = ["% Generated by `python -m qmf.paper.numbers` from the raw results (rows recomputed by",
             "% qmf.analysis.derive). Do not edit by hand. Proved lower bounds are rounded down.", ""]
    for name, (val, src) in macros().items():
        lines.append(f"\\newcommand{{\\qn{name}}}{{{val}}}  % {src}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 3} macros to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/numbers/qmf_numbers.tex")
