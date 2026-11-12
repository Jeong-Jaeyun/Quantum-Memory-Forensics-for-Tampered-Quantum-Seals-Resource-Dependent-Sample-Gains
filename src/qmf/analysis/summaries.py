"""Analyses of the experiments (each experiment's decision rules were fixed before it ran).

N90 on a common grid: each trial's success at n is its MAP correctness at its own last checkpoint <= n (False before its
first checkpoint). The group horizon is the smallest trial n_max; the grid is 40 geometric points from 4 to the horizon.
N90 = first grid point with success fraction >= 0.9; 'censored' if none (N90 > horizon); 'at_cap' if it is the last
grid point (N90 in (previous grid point, horizon]). 90% CI: 2000 bootstrap resamples of trials (inf = censored).

usage: python -m qmf.analysis.summaries e1 E1_ML.jsonl E1_VAULT.jsonl OUT.json
       python -m qmf.analysis.summaries e2 E2.jsonl E1_ML.jsonl OUT.json
       python -m qmf.analysis.summaries e3 E3.jsonl E1_ML.jsonl OUT.json
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

LEVEL, BOOT = 0.9, 2000


def load(*paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh]
    return rows


def group(rows, keys):
    g = defaultdict(list)
    for x in rows:
        g[tuple(x[k] for k in keys)].append(x)
    return g


def success_matrix(rows, grid, key="correct"):
    S = np.zeros((len(rows), len(grid)), dtype=bool)
    for i, x in enumerate(rows):
        ns = np.array([p["n"] for p in x["curve"]])
        ok = np.array([p[key] for p in x["curve"]])
        j = np.searchsorted(ns, grid, side="right") - 1  # last checkpoint <= n
        S[i] = np.where(j >= 0, ok[np.maximum(j, 0)], False)
    return S


def n90(rows, key="correct", rng=None):
    """Grid N90 and its bootstrap CI (the quantities every decision rule uses). For
    reporting only: grid_step (ratio of the grid points around N90), ci90_grid_limited (the CI endpoints are at most one
    grid step apart, so the grid sets the CI width), and n90_interp with ci90_interp (success fraction interpolated
    linearly in log n between the two grid points around 0.9). The bootstrap resamples are the same draws as before."""
    horizon = min(x["n_max"] for x in rows)
    grid = np.unique(np.round(np.geomspace(4, horizon, 40)).astype(np.int64))
    S = success_matrix(rows, grid, key)

    def first(M):
        hit = np.flatnonzero(M.mean(axis=0) >= LEVEL)
        return hit[0] if len(hit) else None

    def interp(M):
        f = M.mean(axis=0)
        hit = np.flatnonzero(f >= LEVEL)
        if not len(hit):
            return math.inf
        j = int(hit[0])
        if j == 0:
            return float(grid[0])
        g0, g1 = math.log(grid[j - 1]), math.log(grid[j])
        return float(math.exp(g0 + (LEVEL - f[j - 1]) / (f[j] - f[j - 1]) * (g1 - g0)))
    k = first(S)
    k = None if k is None else int(k)
    out = dict(trials=len(rows), horizon=int(horizon), n90=None if k is None else int(grid[k]),
               censored=k is None, at_cap=bool(k == len(grid) - 1), prev=None if not k else int(grid[k - 1]))
    rng = rng or np.random.default_rng(0)
    boot, boot_i = [], []
    for _ in range(BOOT):
        Sb = S[rng.integers(0, len(rows), len(rows))]
        kb = first(Sb)
        boot.append(math.inf if kb is None else grid[kb])
        boot_i.append(interp(Sb))
    lo, hi = np.percentile(boot, [5, 95], method="nearest")  # boot may contain inf (censored)
    out["ci90"] = [float(lo), float(hi)]
    out["usable"] = not out["censored"] and not out["at_cap"]  # usable for slope fits
    out["grid_step"] = float(grid[k] / grid[k - 1]) if k else None
    out["ci90_grid_limited"] = bool(k is not None and hi < math.inf and
                                    np.searchsorted(grid, hi) - np.searchsorted(grid, lo) <= 1)
    ni = interp(S)
    li, hi_i = np.percentile(boot_i, [5, 95], method="nearest")
    out["n90_interp"] = None if ni == math.inf else ni
    out["ci90_interp"] = [float(li), float(hi_i)]
    return out


def slope(points):
    """log2 N90 vs r over (r, N90) points."""
    if len(points) < 3:
        return None
    r, n = zip(*sorted(points))
    return float(np.polyfit(r, np.log2(n), 1)[0])


def fmt(s):
    if s is None:
        return "-"
    if s["censored"]:
        return f">{s['horizon']}"
    if s["at_cap"]:
        return f"({s['prev']},{s['n90']}]"
    return str(s["n90"])


def e1(ml_path, vault_path, out):
    ML = group(load(ml_path), ("r", "eps", "p0"))
    V = group(load(vault_path), ("r", "eps", "p0"))
    res = dict(table=[], fits={}, decision={})
    for key in sorted(set(ML) | set(V)):
        r, eps, p0 = key
        sm = n90(ML[key]) if key in ML else None
        sv = n90(V[key]) if key in V else None
        row = dict(r=r, eps=eps, p0=p0, ml=sm, vault=sv)
        if sm and sv and sv["n90"]:
            row["ratio"] = (sm["n90"] / sv["n90"]) if sm["usable"] else None
            row["ratio_lower"] = ((sm["prev"] if sm["at_cap"] else sm["horizon"]) / sv["n90"]) if not sm["usable"] else None
        res["table"].append(row)
    print("| eps | p0 | r | vault N90 (CI) | vault N90 eps^2/r | memoryless N90 (CI) | ratio |")
    for x in res["table"]:
        v, m = x["vault"], x["ml"]
        c = (v["n90"] * x["eps"] ** 2 / x["r"]) if v and v["n90"] else None
        rat = f"{x['ratio']:.0f}" if x.get("ratio") else (f">{x['ratio_lower']:.0f}" if x.get("ratio_lower") else "-")
        print(f"| {x['eps']} | {x['p0']} | {x['r']} | {fmt(v)} {v and v['ci90']} | {c and round(c, 2)} | "
              f"{fmt(m)} {m and m['ci90']} | {rat} |")
    withdraw = False
    for eps in sorted({k[1] for k in ML}):
        for p0 in sorted({k[2] for k in ML}):
            rows = [x for x in res["table"] if x["eps"] == eps and x["p0"] == p0]
            ci = all(x["vault"]["n90"] and x["vault"]["n90"] * eps ** 2 / x["r"] <= 10 for x in rows if x["vault"])
            pts = [(x["r"], x["ml"]["n90"]) for x in rows if x["ml"] and x["ml"]["usable"]]
            sl = slope(pts)
            usable = [x for x in rows if x.get("ratio")]
            top = max(usable, key=lambda x: x["r"]) if usable else None
            ok = bool(ci and sl is not None and sl >= 0.7 and top and top["ratio"] >= 10)
            res["fits"][f"eps={eps},p0={p0}"] = dict(rule_i=ci, slope=sl, slope_points=pts,
                                                    ratio_at=top and top["r"], ratio=top and top["ratio"], passes=ok)
            print(f"eps={eps} p0={p0}: (i) {ci}  (ii) slope {sl and round(sl, 2)} on r={[p[0] for p in pts]}  "
                  f"(iii) ratio {top and round(top['ratio'], 1)} at r={top and top['r']}  -> {'PASS' if ok else 'FAIL'}")
            if eps == 0.15 and not ok:
                withdraw = True
    res["decision"] = dict(withdraw_uniform_exponential_claim=withdraw)
    print("uniform exponential claim:", "WITHDRAW (pre-registered trigger met)" if withdraw else "retained")
    json.dump(res, open(out, "w"), indent=1, default=str)


def e2(path, e1_ml_path, out):
    """path: one file or several, comma-separated (e.g. the pre-registered labs plus ml_ghz)."""
    G = group(load(*path.split(",")), ("r", "eps", "wt", "method"))
    E1 = group(load(e1_ml_path), ("r", "eps", "p0"))
    res = dict(table=[], decision={})
    print("| r | eps | w | vault | ml_random | ml_product | ml_mix | ml_ghz | best/vault | min(3^w,2^r) |")
    for r, eps, wt in sorted({k[:3] for k in G}):
        st = {m: n90(G[(r, eps, wt, m)]) for m in ("memory", "ml_random", "ml_product", "ml_mix", "ml_ghz", "ml_adaptive_ghz")
              if (r, eps, wt, m) in G}
        ref = False
        if "ml_random" not in st and (r, eps, 0.005) in E1:  # r=10, w<r: E1 uniform reference (pre-registered)
            st["ml_random"], ref = n90(E1[(r, eps, 0.005)]), True
        v = st.get("memory")
        mls = [s for m, s in st.items() if m != "memory"]
        found = [s["n90"] for s in mls if not s["censored"]]
        best = min(found) if found else None
        bound = max(s["horizon"] for s in mls) if mls else None
        row = dict(r=r, eps=eps, w=wt, stats=st, e1_ref=ref, best=best,
                   ratio=(best / v["n90"]) if best and v and v["n90"] else None,
                   ratio_lower=(bound / v["n90"]) if best is None and v and v["n90"] else None,
                   predicted=min(3 ** wt, 2 ** r))
        res["table"].append(row)
        rat = f"{row['ratio']:.1f}" if row["ratio"] else (f">{row['ratio_lower']:.0f}" if row["ratio_lower"] else "-")
        print(f"| {r} | {eps} | {wt} | {fmt(v)} | {fmt(st.get('ml_random'))}{'*' if ref else ''} | "
              f"{fmt(st.get('ml_product'))} | {fmt(st.get('ml_mix'))} | {fmt(st.get('ml_ghz'))} | {rat} | {row['predicted']} |")
    for eps in sorted({x["eps"] for x in res["table"]}):
        full = [x for x in res["table"] if x["eps"] == eps and x["w"] == x["r"]]
        pts = [(x["r"], x["ratio"] or x["ratio_lower"]) for x in full if (x["ratio"] or x["ratio_lower"])]
        sl = slope(pts)
        r10 = next((x for x in full if x["r"] == 10), None)
        r10v = r10 and (r10["ratio"] or r10["ratio_lower"])
        keep = bool(r10v and r10v >= 100 and sl is not None and sl >= 0.7)
        withdraw = bool(r10 and r10["ratio"] is not None and r10["ratio"] < 10)
        res["decision"][f"eps={eps}"] = dict(ratio_r10=r10v, ratio_r10_is_lower_bound=bool(r10 and r10["ratio"] is None),
                                             slope=sl, high_weight_claim_retained=keep, withdraw_high_weight_claim=withdraw)
        print(f"eps={eps}: w=r ratio at r=10 {r10v and round(r10v, 1)}{' (lower bound)' if r10 and r10['ratio'] is None else ''}, "
              f"slope {sl and round(sl, 2)} -> retained {keep}, WITHDRAW {withdraw}")
    json.dump(res, open(out, "w"), indent=1, default=str)


def side(s, target):
    """Vault vs memoryless N90 (target) at one p_v. A censored vault group whose horizon is below the target is
    undecided (11-06 fix: it was read as 'behind'); an at-cap N90 in (prev, n90] is undecided if it straddles it."""
    if s["censored"]:
        return "behind" if s["horizon"] >= target else "unknown"
    if s["at_cap"] and s["prev"] < target <= s["n90"]:
        return "unknown"
    return "ahead" if s["n90"] < target else "behind"


def e3(path, e1_ml_path, out, targets=None):
    """path: one file or several, comma-separated; a later file replaces earlier groups with the same
    (vkind, r, eps, p_v) (the floor refinement re-runs the same instances with a longer horizon). targets
    (E3f): {(r, eps): memoryless N90} used instead of the E1 grid N90 at p0 = 0.005 (then e1_ml_path is not read)."""
    G = {}
    for p in path.split(","):
        G.update(group(load(p), ("vkind", "r", "eps", "pv")))
    E1 = group(load(e1_ml_path), ("r", "eps", "p0")) if targets is None else {}
    res = dict(crossover={})
    for kind, r, eps in sorted({k[:3] for k in G}):
        pvs = sorted(k[3] for k in G if k[:3] == (kind, r, eps))
        st = [n90(G[(kind, r, eps, pv)]) for pv in pvs]
        if targets is None:
            ml = n90(E1[(r, eps, 0.005)]) if (r, eps, 0.005) in E1 else None
            mlv = ml and (ml["n90"] if not ml["censored"] else ml["horizon"])
        else:
            ml, mlv = None, targets.get((r, eps))
        sides = [side(s, mlv) for s in st] if mlv else []
        behind = [p for p, sd in zip(pvs, sides) if sd == "behind"]
        hi = min(behind) if behind else None
        ahead = [p for p, sd in zip(pvs, sides) if sd == "ahead" and (hi is None or p < hi)]
        lo = max(ahead) if ahead else None
        interp = None
        if lo is not None and hi is not None:
            i, j = pvs.index(lo), pvs.index(hi)
            if j == i + 1 and st[i]["usable"] and st[j]["usable"]:
                a, b, t = math.log(st[i]["n90"]), math.log(st[j]["n90"]), math.log(mlv)
                interp = lo + (t - a) / (b - a) * (hi - lo)
        nonmono = any(sd == "ahead" and hi is not None and p > hi for p, sd in zip(pvs, sides))
        lower = bool(ml and ml["censored"])
        res["crossover"][f"{kind},r={r},eps={eps}"] = dict(bracket=[lo, hi], interpolated=interp, non_monotone=nonmono,
                                                          memoryless_n90=mlv, memoryless_is_lower_bound=lower,
                                                          vault=[dict(pv=p, side=sd, **s) for p, s, sd in zip(pvs, st, sides)])
        print(f"{kind} r={r} eps={eps}: memoryless N90 {mlv}{' (lower bound)' if lower else ''}; vault by p_v "
              f"{[(p, fmt(s), sd[0]) for p, s, sd in zip(pvs, st, sides)]} -> p_v* in ({lo}, {hi})"
              f"{f', interpolated {interp:.3f}' if interp is not None else ''}{' NON-MONOTONE' if nonmono else ''}")
    json.dump(res, open(out, "w"), indent=1, default=str)


def e5(path, out):
    """path: ml_ghz and ml_adaptive_ghz rows (comma-separated files). Pre-registered trigger: at r=10, adaptive N90 <
    ml_ghz N90 / 3 for some eps -> re-apply the E2 rules with the adaptive lab (run e2 on e2, e2_ghz and e5 files)."""
    G = group(load(*path.split(",")), ("r", "eps", "method"))
    res = dict(table=[], trigger=False)
    print("| r | eps | ml_ghz | ml_adaptive_ghz | ghz / adaptive |")
    for r, eps in sorted({k[:2] for k in G}):
        g, a = (n90(G[(r, eps, m)]) if (r, eps, m) in G else None for m in ("ml_ghz", "ml_adaptive_ghz"))
        sp = (g["n90"] / a["n90"]) if g and a and g["n90"] and a["n90"] else None
        res["table"].append(dict(r=r, eps=eps, ghz=g, adaptive=a, speedup=sp))
        print(f"| {r} | {eps} | {fmt(g)} | {fmt(a)} | {sp and round(sp, 2)} |")
        if r == 10 and sp and sp > 3:
            res["trigger"] = True
    print("E5 trigger (adaptive < ghz/3 at r=10):", res["trigger"])
    json.dump(res, open(out, "w"), indent=1, default=str)


def clopper_pearson_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def e6(path, out):
    """E6 (mode vault) and E8 (ml_blind, ml_library) from switch2 rows."""
    G = group(load(*path.split(",")), ("mode", "r", "eps"))
    res = {}
    print("| mode | r | eps | n sw/ns | cells | first/last | detect | false alarm (CP95 up) | q90 |err| eps^2 | q90 |err| (eps att)^2 | rule |")
    for (mode, r, eps), xs in sorted(G.items()):
        sw, ns = [x for x in xs if x["switch"]], [x for x in xs if not x["switch"]]
        k = sum(x["switch_detected"] for x in ns)
        det = float(np.mean([x["switch_detected"] for x in sw]))
        errs = [x for x in sw if x["switch_detected"]]
        q_old = float(np.quantile([abs(x["switch_err"]) for x in errs], 0.9)) * eps ** 2 if errs else None
        q_new = float(np.quantile([abs(x["switch_err"]) * (eps * x["att_min"]) ** 2 for x in errs], 0.9)) if errs else None
        d = dict(n_switch=len(sw), n_noswitch=len(ns), detect=det, false_alarms=int(k), false_alarm_rate=k / len(ns),
                 fa_cp95_upper=clopper_pearson_upper(k, len(ns)), q90_eps2=q_old, q90_att=q_new)
        if mode == "vault":
            d["cells_exact"] = float(np.mean([x["cells_exact"] for x in xs]))
            d["first_ok"], d["last_ok"] = float(np.mean([x["first_ok"] for x in xs])), float(np.mean([x["last_ok"] for x in xs]))
            d["passes_E6"] = bool(d["fa_cp95_upper"] <= 0.05 and det >= 0.9 and d["cells_exact"] >= 0.9)
            d["passes_att_time"] = bool(q_new is not None and q_new <= 5)
            rule = f"E6 {'PASS' if d['passes_E6'] else 'FAIL'}, time(att) {'PASS' if d['passes_att_time'] else 'FAIL'}"
        elif mode == "ml_blind":
            d["first_ok"], d["last_ok"] = float(np.mean([x["first_ok"] for x in xs])), float(np.mean([x["last_ok"] for x in xs]))
            d["passes_E8_blind"] = bool(det <= 0.1)
            rule = f"E8 blind {'PASS (cannot detect)' if det <= 0.1 else 'FAIL'}"
        else:
            rule = "boundary (registry)"
        res[f"{mode},r={r},eps={eps}"] = d
        fl = f"{d.get('first_ok', float('nan')):.2f}/{d.get('last_ok', float('nan')):.2f}"
        print(f"| {mode} | {r} | {eps} | {len(sw)}/{len(ns)} | {d.get('cells_exact', '-')} | {fl} | {det:.2f} | "
              f"{k}/{len(ns)} ({d['fa_cp95_upper']:.3f}) | {q_old and round(q_old, 2)} | {q_new and round(q_new, 2)} | {rule} |")
    json.dump(res, open(out, "w"), indent=1, default=str)


def e7(path, out):
    G = group(load(path), ("lab", "r", "K", "eps_tot"))
    res = dict(table=[], decision={})
    print("| K | eps_tot | r | vault N90 | memoryless N90 | ratio |")
    for K, eps in sorted({(k[2], k[3]) for k in G}):
        pts, ml_pts = [], []
        for r in sorted({k[1] for k in G if k[2:] == (K, eps)}):
            v = n90(G[("memory", r, K, eps)]) if ("memory", r, K, eps) in G else None
            m = n90(G[("ml_random", r, K, eps)]) if ("ml_random", r, K, eps) in G else None
            ratio = lower = None
            if v and m and v["n90"]:
                if m["usable"]:
                    ratio = m["n90"] / v["n90"]
                else:
                    lower = (m["prev"] if m["at_cap"] else m["horizon"]) / v["n90"]
            if ratio or lower:
                pts.append((r, ratio or lower))
            if m and m["usable"]:
                ml_pts.append((r, m["n90"]))
            res["table"].append(dict(K=K, eps_tot=eps, r=r, vault=v, ml=m, ratio=ratio, ratio_lower=lower))
            rat = f"{ratio:.0f}" if ratio else (f">{lower:.0f}" if lower else "-")
            print(f"| {K} | {eps} | {r} | {fmt(v)} | {fmt(m)} | {rat} |")
        sl, ml_sl = slope(pts), slope(ml_pts)
        r8 = next((p[1] for p in pts if p[0] == 8), None)
        ok = bool(r8 is not None and r8 >= 0.25 * 2 ** 8 and sl is not None and sl >= 0.7)
        res["decision"][f"K={K},eps_tot={eps}"] = dict(ratio_r8=r8, ratio_slope=sl, memoryless_slope=ml_sl, persists=ok,
                                                       restrict=bool(r8 is not None and r8 < 10))
        print(f"K={K} eps_tot={eps}: ratio r=8 {r8 and round(r8, 1)}, log2-ratio slope {sl and round(sl, 2)}, "
              f"memoryless slope {ml_sl and round(ml_sl, 2)} -> persists {ok}, restrict {bool(r8 is not None and r8 < 10)}")
    json.dump(res, open(out, "w"), indent=1, default=str)


def pl(paths, vault_path, out):
    """Bundle rerun of the 11-05 P and local-tamper measurements (no new rule). paths = 'PAIRED.jsonl,LOCAL.jsonl'.
    P: a*-only N90 (units) of self-referenced pairs vs the E1 vault, prediction 2/eps^2. Paired rows store n_max in
    samples (units = 2 x samples), so their horizon is the last checkpoint. Local: memory vs ml_product N90, 11-05 rule
    'within 2x' -> no advantage."""
    rows = load(*paths.split(","))
    P = group([x for x in rows if x["method"] == "paired"], ("r", "eps", "p0"))
    L = group([x for x in rows if x["model"] == "local"], ("method", "r"))
    V = group(load(vault_path), ("r", "eps", "p0"))
    res = dict(paired=[], local=[], decision={})
    print("| eps | p0 | r | vault a* N90 | P a* N90 | P / vault | 2/eps^2 |")
    for key in sorted(P):
        r, eps, p0 = key
        for x in P[key]:
            x["n_max"] = x["curve"][-1]["n"]
        sp, sv = n90(P[key], "a_correct"), (n90(V[key], "a_correct") if key in V else None)
        ratio = sp["n90"] / sv["n90"] if sp["usable"] and sv and sv["n90"] else None
        res["paired"].append(dict(r=r, eps=eps, p0=p0, paired=sp, vault=sv, ratio=ratio, predicted=2 / eps ** 2))
        print(f"| {eps} | {p0} | {r} | {fmt(sv)} | {fmt(sp)} | {ratio and round(ratio, 1)} | {2 / eps ** 2:.1f} |")
    for eps in sorted({k[1] for k in P}):
        rs = [x["ratio"] for x in res["paired"] if x["eps"] == eps and x["ratio"]]
        res["decision"][f"paired eps={eps}"] = dict(ratio_min=min(rs, default=None), ratio_max=max(rs, default=None),
                                                    n_ratio=len(rs), n_groups=sum(x["eps"] == eps for x in res["paired"]))
        print(f"P/vault eps={eps}: {min(rs, default=None)} - {max(rs, default=None)} ({len(rs)} usable)")
    print("| r | memory N90 | ml_product N90 | ratio |")
    pts = []
    for r in sorted({k[1] for k in L}):
        m, p = n90(L[("memory", r)]), n90(L[("ml_product", r)])
        ratio = p["n90"] / m["n90"] if m["usable"] and p["usable"] else None
        res["local"].append(dict(r=r, memory=m, ml_product=p, ratio=ratio))
        pts.append((r, ratio))
        print(f"| {r} | {fmt(m)} | {fmt(p)} | {ratio and round(ratio, 2)} |")
    ok = [x for x in pts if x[1]]
    res["decision"]["local"] = dict(within_2x=bool(len(ok) == len(pts) and all(x[1] <= 2 for x in ok)),
                                    ratio_slope=slope(ok), ratios=[x[1] for x in ok])
    print("local:", res["decision"]["local"])
    json.dump(res, open(out, "w"), indent=1, default=str)


def cp_two_sided(k, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    return (0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k)))


def e9(path, e1_vault_path, out):
    """E9 (rules fixed before running). path: e9 rows (comma-separated files allowed). Success is (a*, s)
    except for P (a* only). 'Blind' = success at the group horizon <= 0.05."""
    G = group(load(*path.split(",")), ("lab", "attacker", "r", "eps"))
    V1 = group(load(e1_vault_path), ("r", "eps", "p0"))
    res = dict(table=[], decision={})
    T = {}
    print("| lab | attacker | r | eps | trials | N90 (CI) | success at horizon (CP95) | R^ exact | mean |R^| |")
    for (lab, att, r, eps), xs in sorted(G.items()):
        key = "a_correct" if lab == "paired" else "correct"
        st = n90(xs, key)
        h = min(x["n_max"] for x in xs)
        k = int(success_matrix(xs, np.array([h]), key)[:, 0].sum())
        d = dict(lab=lab, attacker=att, r=r, eps=eps, trials=len(xs), key=key, n90=st, horizon=h,
                 success_at_horizon=k / len(xs), successes_at_horizon=k, success_cp95=cp_two_sided(k, len(xs)))
        if "cells_exact" in xs[0]:
            d["cells_exact"] = float(np.mean([bool(x["cells_exact"]) for x in xs]))
            d["rhat_size_mean"] = float(np.mean([len(x["Rhat"]) for x in xs]))
        T[(lab, att, r, eps)] = d
        res["table"].append(d)
        print(f"| {lab} | {att} | {r} | {eps} | {len(xs)} | {fmt(st)} {st['ci90']} | {k}/{len(xs)} "
              f"[{d['success_cp95'][0]:.3f}, {d['success_cp95'][1]:.3f}] | {d.get('cells_exact', '-')} | "
              f"{d.get('rhat_size_mean', '-')} |")
    rs, epss = sorted({k[2] for k in T}), sorted({k[3] for k in T})

    def nv(key):
        d = T.get(key)
        return d["n90"]["n90"] if d and not d["n90"]["censored"] else None
    # (a) oblivious attacker
    a_rows, a_ok, a_check = [], True, True
    for r in rs:
        for eps in epss:
            ins, vs = T.get(("inseal", "obl", r, eps)), T.get(("vault_seal", "obl", r, eps))
            if not ins or not vs:
                continue
            ni, nvs = nv(("inseal", "obl", r, eps)), nv(("vault_seal", "obl", r, eps))
            e1 = n90(V1[(r, eps, 0.005)]) if (r, eps, 0.005) in V1 else None
            ratio = ni / nvs if ni and nvs else None
            ok = bool(ins["cells_exact"] >= 0.95 and ratio is not None and ratio <= 2)
            chk = (nvs / e1["n90"]) if nvs and e1 and e1["n90"] else None
            a_ok &= ok
            a_check &= bool(chk is not None and 0.67 <= chk <= 1.5)
            a_rows.append(dict(r=r, eps=eps, inseal_n90=ni, vault_seal_n90=nvs, e1_vault_n90=e1 and e1["n90"],
                               inseal_over_vault=ratio, overhead_units=(ni - nvs) if ni and nvs else None,
                               inseal_cells_exact=ins["cells_exact"], vault_seal_over_e1=chk, passes=ok,
                               inseal_n90_interp=ins["n90"]["n90_interp"], vault_seal_n90_interp=vs["n90"]["n90_interp"]))
    res["decision"]["a_oblivious"] = dict(rows=a_rows, inseal_equals_vault_up_to_tier1=a_ok, vault_seal_matches_e1=a_check)
    # (b) mirror attacker
    blind_in = [T[k]["success_at_horizon"] <= 0.05 for k in T if k[0] == "inseal" and k[1] == "mirror"]
    blind_p = [T[k]["success_at_horizon"] <= 0.05 for k in T if k[0] == "paired" and k[1] == "mirror"]
    v_rows, v_ok = [], True
    for r in rs:
        for eps in epss:
            m, o = T.get(("vault_seal", "mirror", r, eps)), nv(("vault_seal", "obl", r, eps))
            if not m:
                continue
            nm = nv(("vault_seal", "mirror", r, eps))
            ratio = nm / o if nm and o else None
            ok = bool(m["cells_exact"] >= 0.95 and ratio is not None and ratio <= 1.5)
            v_ok &= ok
            v_rows.append(dict(r=r, eps=eps, mirror_n90=nm, obl_n90=o, ratio=ratio, cells_exact=m["cells_exact"], passes=ok))
    res["decision"]["b_mirror"] = dict(inseal_blind=bool(blind_in and all(blind_in)), paired_blind=bool(blind_p and all(blind_p)),
                                       vault_unaffected=bool(v_rows and v_ok), vault_rows=v_rows)
    # (c) extend attacker
    blind_x = [T[k]["success_at_horizon"] <= 0.05 for k in T if k[0] == "inseal" and k[1] == "extend"]
    x_rows, x_ok = [], True
    for r in rs:
        for eps in epss:
            ne, o = nv(("vault_seal", "extend", r, eps)), nv(("vault_seal", "obl", r, eps))
            if ("vault_seal", "extend", r, eps) not in T:
                continue
            ratio = ne / o if ne and o else None
            x_ok &= bool(ratio is not None and ratio <= 1.5)
            x_rows.append(dict(r=r, eps=eps, extend_n90=ne, obl_n90=o, ratio=ratio, passes=bool(ratio is not None and ratio <= 1.5)))
    sp_rows, sp_ok = [], True
    for eps in epss:
        if ("vault_sparse", "extend", 4, eps) not in T:
            continue
        ns = nv(("vault_sparse", "extend", 4, eps))
        e1 = n90(V1[(4, eps, 0.005)])
        ratio = ns / e1["n90"] if ns and e1["n90"] else None
        sp_ok &= bool(ratio is not None and ratio <= 4)
        sp_rows.append(dict(r=4, eps=eps, sparse_n90=ns, sparse_n90_interp=T[("vault_sparse", "extend", 4, eps)]["n90"]["n90_interp"],
                            e1_vault_n90=e1["n90"], e1_vault_n90_interp=e1["n90_interp"], ratio=ratio,
                            predicted=math.log2(2 * 3059124) / math.log2(2 * 255), passes=bool(ratio is not None and ratio <= 4)))
    res["decision"]["c_extend"] = dict(inseal_blind=bool(blind_x and all(blind_x)), vault_given_R_unaffected=bool(x_rows and x_ok),
                                       vault_rows=x_rows, vault_without_R=bool(sp_rows and sp_ok), sparse_rows=sp_rows)
    for k, v in res["decision"].items():
        print(k, {a: b for a, b in v.items() if not a.endswith("rows")})
        for row in v.get("rows", []) + v.get("vault_rows", []) + v.get("sparse_rows", []):
            print("   ", row)
    json.dump(res, open(out, "w"), indent=1, default=str)


def e10(bound_path, e2_path, e1_path, out):
    """E10 (rules fixed before running). Fano lower bound for non-adaptive stabilizer probes against the
    measured N90 of E2 (weight-w families, p0 = 0.005) and E1 (uniform family). bound_path: qmf.experiments.coverage bound."""
    from qmf.experiments.coverage import d_bits, fano_units
    B = json.load(open(bound_path))["per_r"]
    E2 = json.load(open(e2_path))["table"]
    E1 = json.load(open(e1_path))["table"]
    NONADAPT = ("ml_random", "ml_product", "ml_mix", "ml_ghz")
    res = dict(weight=[], uniform=[], decision={}, ghz_optimal={})
    print("| r | w | eps | c* in | N_LB | vault | best non-adaptive | best/N_LB | N_LB/vault | adaptive | violations |")
    for x in E2:
        r, w, eps, p0 = x["r"], x["w"], x["eps"], 0.005
        b = B[str(r)]["w"][str(w)]
        M = b["M"]
        ep = eps * (1 - 4 * p0 / 3) ** w
        lb = fano_units(M, b["c_upper"], ep)
        st = x["stats"]
        v = st["memory"]["n90"]
        labs = {m: st[m] for m in NONADAPT if m in st and not (m == "ml_random" and x["e1_ref"])}
        found = [s["n90"] for s in labs.values() if not s["censored"]]
        best = min(found) if found else None
        viol = [m for m, s in labs.items() if not s["censored"] and s["n90"] < lb / (s["grid_step"] or 1.0)]
        near = [m for m, s in labs.items() if not s["censored"] and lb / (s["grid_step"] or 1.0) <= s["n90"] < lb]
        ad = st.get("ml_adaptive_ghz")
        row = dict(r=r, w=w, eps=eps, p0=p0, M=M, c_lower=b["c_lower"], c_upper=b["c_upper"], eps_eff=ep, d_bits=d_bits(ep),
                   n_lb=lb, vault_n90=v, best_nonadaptive=best, best_over_lb=(best / lb) if best else None,
                   lb_over_vault=lb / v if v else None, adaptive_n90=ad and ad["n90"],
                   violations=viol, within_one_grid_step_below=near)
        res["weight"].append(row)
        print(f"| {r} | {w} | {eps} | [{b['c_lower']:.3g}, {b['c_upper']:.3g}] | {lb:.0f} | {v} | {best} | "
              f"{row['best_over_lb'] and round(row['best_over_lb'], 2)} | {row['lb_over_vault'] and round(row['lb_over_vault'], 1)} | "
              f"{row['adaptive_n90']} | {viol or ''}{' near ' + str(near) if near else ''} |")
    for x in E1:
        r, eps, p0 = x["r"], x["eps"], x["p0"]
        M = 4 ** r - 1
        lb = fano_units(M, (2 ** r - 1) / M, eps)
        m, v = x["ml"], x["vault"]
        res["uniform"].append(dict(r=r, eps=eps, p0=p0, n_lb=lb, memoryless_n90=m["n90"], vault_n90=v["n90"],
                                   ml_over_lb=(m["n90"] / lb) if m["n90"] else None, lb_over_vault=lb / v["n90"],
                                   violation=bool(m["n90"] and m["n90"] < lb / (m["grid_step"] or 1.0))))
    for r, row in B.items():
        b = row["w"][r]
        res["ghz_optimal"][r] = dict(ghz=b["ghz"], exact=b.get("exact"), ilp_upper=b["ilp_upper"],
                                     search_lower=b.get("search_lower"), optimal=b["ghz_optimal"])
    for name, wsel in (("claim2_w_eq_r", lambda r: r), ("claim3_w_half", lambda r: math.ceil(r / 2))):
        rows = [x for x in res["weight"] if x["r"] == 10 and x["w"] == wsel(10)]
        ok = bool(rows) and all(x["lb_over_vault"] >= 10 for x in rows)
        res["decision"][name] = dict(lb_over_vault_r10={str(x["eps"]): x["lb_over_vault"] for x in rows},
                                     measured_over_vault_r10={str(x["eps"]): x["best_nonadaptive"] / x["vault_n90"] for x in rows
                                                              if x["best_nonadaptive"]},
                                     proved_separation=ok)
    res["decision"]["violations"] = [dict(r=x["r"], w=x["w"], eps=x["eps"], labs=x["violations"]) for x in res["weight"]
                                     if x["violations"]] + [dict(r=x["r"], eps=x["eps"], p0=x["p0"], labs=["ml_random (E1)"])
                                                            for x in res["uniform"] if x["violation"]]
    for k, v in res["decision"].items():
        print(k, v)
    print("GHZ optimal (w = r):", {r: v["optimal"] for r, v in res["ghz_optimal"].items()})
    json.dump(res, open(out, "w"), indent=1, default=str)


def horizon_summary(xs):
    """N90 and the success at the group horizon (two-sided 95% Clopper-Pearson) of a set of trials (E11, E12)."""
    if not xs:
        return None
    h = min(x["n_max"] for x in xs)
    k = int(success_matrix(xs, np.array([h]))[:, 0].sum())
    return dict(trials=len(xs), n90=n90(xs), horizon=h, successes_at_horizon=k, success_at_horizon=k / len(xs),
                success_cp95=cp_two_sided(k, len(xs)))


def n90_value(s):
    return s["n90"]["n90"] if s and not s["n90"]["censored"] else None


def e11(path, e9_path, out):
    """E11 (rules fixed before running). Spray-mirror attacker. Success is (a*, s); 'blind' = success at
    the group horizon <= 0.05 (two-sided 95% Clopper-Pearson interval reported). Subsets by in_C (a* commutes with the
    diagonal Paulis P^(x r)); in_C depends on the instance only, so a subset is the same trials in every lab."""
    G = group(load(*path.split(",")), ("lab", "r", "eps"))
    E9o = group([x for x in load(e9_path) if x["lab"] == "vault_seal" and x["attacker"] == "obl"], ("r", "eps"))
    summary, nv = horizon_summary, n90_value
    res = dict(table=[], decision={})
    T = {}
    print("| lab | r | eps | in C | N90 all (CI) | success all | N90 in C | success in C | success not in C | exact |")
    for (lab, r, eps), xs in sorted(G.items()):
        d = dict(lab=lab, r=r, eps=eps, in_C_fraction=float(np.mean([x["in_C"] for x in xs])), all=summary(xs),
                 in_C=summary([x for x in xs if x["in_C"]]), not_in_C=summary([x for x in xs if not x["in_C"]]))
        for f in ("cells_exact", "components_exact"):
            if f in xs[0]:
                d[f] = float(np.mean([bool(x[f]) for x in xs]))
        if "Rhat" in xs[0]:
            d["rhat_size_mean"] = float(np.mean([len(x["Rhat"]) for x in xs]))
        T[(lab, r, eps)] = d
        res["table"].append(d)
        a, i, o = d["all"], d["in_C"], d["not_in_C"]
        print(f"| {lab} | {r} | {eps} | {d['in_C_fraction']:.2f} | {fmt(a['n90'])} {a['n90']['ci90']} | "
              f"{a['successes_at_horizon']}/{a['trials']} | {fmt(i['n90'])} | {i['successes_at_horizon']}/{i['trials']} | "
              f"{o['successes_at_horizon']}/{o['trials']} | {d.get('components_exact', d.get('cells_exact', '-'))} |")
    rs, epss = sorted({k[1] for k in T}), sorted({k[2] for k in T})
    # (a) vault, R not given
    a_rows, a_ok = [], True
    for r in rs:
        for eps in epss:
            v = T.get(("vault_seal", r, eps))
            if not v:
                continue
            obl = n90(E9o[(r, eps)]) if (r, eps) in E9o else None
            ns, no = nv(v["all"]), (obl["n90"] if obl and not obl["censored"] else None)
            ratio = ns / no if ns and no else None
            ok = bool(v["components_exact"] >= 0.95 and v["cells_exact"] >= 0.95 and ratio is not None and ratio <= 1.5)
            a_ok &= ok
            a_rows.append(dict(r=r, eps=eps, spray_n90=ns, obl_n90=no, ratio=ratio, spray_n90_interp=v["all"]["n90"]["n90_interp"],
                               obl_n90_interp=obl and obl["n90_interp"], grid_step=v["all"]["n90"]["grid_step"],
                               components_exact=v["components_exact"], cells_exact=v["cells_exact"], passes=ok))
    res["decision"]["a_vault"] = dict(vault_unaffected=bool(a_rows and a_ok), rows=a_rows)
    # (b) E9's in-seal lab
    b = [T[k]["all"]["success_at_horizon"] <= 0.05 for k in T if k[0] == "inseal"]
    res["decision"]["b_inseal"] = dict(inseal_blind=bool(b and all(b)),
                                       rhat_size_mean=[T[k]["rhat_size_mean"] for k in sorted(T) if k[0] == "inseal"])
    # (c) in-seal oracle (told R and the groups)
    c_rows, c1 = [], True
    for r in rs:
        for eps in epss:
            o, v = T.get(("inseal_oracle", r, eps)), T.get(("vault_seal", r, eps))
            if not o:
                continue
            blind = o["not_in_C"]["success_at_horizon"] <= 0.05
            c1 &= blind
            ni, nvc = nv(o["in_C"]), nv(v["in_C"]) if v else None
            c_rows.append(dict(r=r, eps=eps, in_C_trials=o["in_C"]["trials"], not_in_C_trials=o["not_in_C"]["trials"],
                               not_in_C_success=o["not_in_C"]["success_at_horizon"], not_in_C_cp95=o["not_in_C"]["success_cp95"],
                               blind_not_in_C=blind, oracle_in_C_n90=ni, vault_in_C_n90=nvc,
                               oracle_over_vault_in_C=(ni / nvc) if ni and nvc else None,
                               within_predicted=bool(ni and nvc and 0.67 <= ni / nvc <= 1.5),
                               all_success=o["all"]["success_at_horizon"], in_C_fraction=o["in_C_fraction"],
                               all_censored=o["all"]["n90"]["censored"]))
    res["decision"]["c_oracle"] = dict(blind_not_in_C=bool(c_rows and c1), rows=c_rows)
    for k, v in res["decision"].items():
        print(k, {a: b for a, b in v.items() if a != "rows"})
        for row in v.get("rows", []):
            print("   ", row)
    json.dump(res, open(out, "w"), indent=1, default=str)


def e12(path, e11_path, e9_path, e1f_path, out):
    """E12 and the extend bound (rules fixed before running). The product probe (path: e12 rows) is
    compared with E11's vault_seal on the same trials: same instance, R, groups, horizon and checkpoints (checked).
    Success is (a*, s). e9_path: E9 rows (extend: vault given R, vault_sparse at r = 4). e1f_path: E1f summary (the
    memoryless fine N90 at p0 = 0.005, a verifier told R solving the same r-cell problem)."""
    from qmf.experiments.product import AGREE, T1, unknown_R_bound
    P = group(load(*path.split(",")), ("r", "eps"))
    V = group([x for x in load(e11_path) if x["lab"] == "vault_seal"], ("r", "eps"))
    X9 = group([x for x in load(e9_path) if x["attacker"] == "extend" and x["lab"].startswith("vault")], ("lab", "r", "eps"))
    F = {(x["r"], x["eps"]): x["fine"]["n90"] for x in json.load(open(e1f_path))["table"]
         if x["lab"] == "ml" and x["p0"] == 0.005}
    k_min = math.ceil(AGREE * T1)
    p_pair = sum(math.comb(T1, k) for k in range(k_min, T1 + 1)) / 2 ** T1  # two components' x bits agree by chance
    res = dict(table=[], decision={}, extend=[], p_false_merge_pair=p_pair)
    print("| r | eps | product N90 (CI) | vault N90 | ratio (interp) | (N90-T1)/vault | expected | success | groups exact "
          "(pred) | in C / not in C | memoryless fine / product |")
    for (r, eps), xs in sorted(P.items()):
        xs, vs = sorted(xs, key=lambda x: x["trial"]), sorted(V[(r, eps)], key=lambda x: x["trial"])
        assert [x["trial"] for x in xs] == [x["trial"] for x in vs]
        assert all(x["n_max"] == v["n_max"] and x["R"] == v["R"] and x["groups"] == v["groups"] for x, v in zip(xs, vs))
        sp, sv = horizon_summary(xs), horizon_summary(vs)
        si, so = horizon_summary([x for x in xs if x["in_C"]]), horizon_summary([x for x in xs if not x["in_C"]])
        npd, nvd = n90_value(sp), n90_value(sv)
        ip, iv = sp["n90"]["n90_interp"], sv["n90"]["n90_interp"]
        ge = float(np.mean([bool(x["groups_exact"]) for x in xs]))
        d = dict(r=r, eps=eps, product=sp, vault=sv, product_in_C=si, product_not_in_C=so, product_n90=npd, vault_n90=nvd,
                 ratio=(npd / nvd) if npd and nvd else None, ratio_interp=(ip / iv) if ip and iv else None,
                 ratio_minus_T1=((npd - T1) / nvd) if npd and nvd else None,
                 expected_ratio=(1 + T1 / nvd) if nvd else None,
                 grid_step_product=sp["n90"]["grid_step"], grid_step_vault=sv["n90"]["grid_step"],
                 groups_exact=ge, groups_exact_predicted=1 - r * (r - 1) / 2 * p_pair,
                 memoryless_fine_n90=F.get((r, eps)),
                 memoryless_over_product=(F[(r, eps)] / npd) if npd and F.get((r, eps)) else None)
        d["within_predicted"] = bool(d["ratio"] is not None and 0.67 <= d["ratio"] <= 1.5)
        res["table"].append(d)
        print(f"| {r} | {eps} | {fmt(sp['n90'])} {sp['n90']['ci90']} | {nvd} | {d['ratio'] and round(d['ratio'], 2)} "
              f"({d['ratio_interp'] and round(d['ratio_interp'], 2)}) | {d['ratio_minus_T1'] and round(d['ratio_minus_T1'], 2)} | "
              f"{d['expected_ratio'] and round(d['expected_ratio'], 2)} | {sp['successes_at_horizon']}/{sp['trials']} | "
              f"{ge:.3f} ({d['groups_exact_predicted']:.3f}) | {si['successes_at_horizon']}/{si['trials']} / "
              f"{so['successes_at_horizon']}/{so['trials']} | {d['memoryless_over_product'] and round(d['memoryless_over_product'])} |")
    T = res["table"]
    res["decision"]["a_not_blind"] = dict(not_blind=bool(T) and all(x["product_n90"] is not None for x in T))
    res["decision"]["b_vault_level"] = dict(within_predicted=bool(T) and all(x["within_predicted"] for x in T),
                                            misses=[dict(r=x["r"], eps=x["eps"], ratio=x["ratio"], ratio_minus_T1=x["ratio_minus_T1"])
                                                    for x in T if not x["within_predicted"]],
                                            ratio_range=[min(x["ratio"] for x in T if x["ratio"]), max(x["ratio"] for x in T if x["ratio"])]
                                            if any(x["ratio"] for x in T) else None)
    res["decision"]["c_groups"] = dict(groups_exact={f"{x['r']}/{x['eps']}": x["groups_exact"] for x in T})
    res["decision"]["d_in_C"] = dict(success_in_C={f"{x['r']}/{x['eps']}": x["product_in_C"]["success_at_horizon"] for x in T},
                                     success_not_in_C={f"{x['r']}/{x['eps']}": x["product_not_in_C"]["success_at_horizon"] for x in T})
    # extend: the vault's cost with and without R, the information-theoretic bound, the memoryless comparator
    ok = True
    for r in (4, 6, 8, 10):
        for eps in (0.15, 0.6):
            b = unknown_R_bound(r, eps, 0.005)
            given = n90(X9[("vault_seal", r, eps)]) if ("vault_seal", r, eps) in X9 else None
            sparse = n90(X9[("vault_sparse", r, eps)]) if ("vault_sparse", r, eps) in X9 else None
            ns = sparse and sparse["n90"]
            if sparse is not None:
                ok &= bool(ns is not None and ns <= b["units"])
            m = F.get((r, eps))
            res["extend"].append(dict(r=r, eps=eps, M=b["M"], eps_min=b["eps_min"], bound_units=b["units"],
                                      vault_given_R_n90=given and given["n90"],
                                      vault_given_R_grid_step=given and given["grid_step"],
                                      vault_unknown_R_n90=ns, memoryless_fine_n90=m,
                                      bound_over_given=(b["units"] / given["n90"]) if given and given["n90"] else None,
                                      memoryless_over_bound=(m / b["units"]) if m else None))
    res["decision"]["extend_bound"] = dict(sparse_within_bound=ok)
    for x in res["extend"]:
        print("   extend", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
    for k, v in res["decision"].items():
        print(k, v)
    json.dump(res, open(out, "w"), indent=1, default=str)


def e13(path, pl_path, e1_path, e1f_path, e10_path, out):
    """E13, P + sign, and the self-referencing cost under extend (rules fixed before running).
    path: e13 rows. Success is (a*, s) for P + sign and the vault. pl_path: pl.json (P, a* only); e1_path: e1.json (vault
    grid N90, 200 trials); e1f_path: e1f.json (vault fine N90); e10_path: e10.json (uniform Fano bound n_lb for
    non-adaptive stabilizer per-unit probes). Phase 1 is checked against pl.json: P's a*-only N90 recomputed from the
    E13 rows (curve shifted back by m) must equal the stored one."""
    from qmf.experiments.product import unknown_R_bound
    from qmf.experiments.self_ref import paired_sign_bound
    G = group(load(path), ("r", "eps", "p0"))
    PL = {(x["r"], x["eps"], x["p0"]): x for x in json.load(open(pl_path))["paired"]}
    V1 = {(x["r"], x["eps"], x["p0"]): x["vault"] for x in json.load(open(e1_path))["table"]}
    VF = {(x["r"], x["eps"], x["p0"]): x["fine"]["n90"] for x in json.load(open(e1f_path))["table"] if x["lab"] == "vault"}
    LB = {(x["r"], x["eps"], x["p0"]): x["n_lb"] for x in json.load(open(e10_path))["uniform"]}
    res = dict(table=[], decision={}, extend=[])
    print("| r | eps | p0 | m | P+sign N90 (CI) | P a* N90 | ratio | success | sign ok | vault fine (grid) | P+sign/vault fine (grid) |")
    for (r, eps, p0), xs in sorted(G.items()):
        k = (r, eps, p0)
        s = horizon_summary(xs)
        shifted = [dict(n_max=x["n_max"] - x["m"], curve=[dict(n=c["n"] - x["m"], a_correct=c["a_correct"]) for c in x["curve"]])
                   for x in xs]
        pa_check = n90(shifted, "a_correct")
        assert pa_check["n90"] == PL[k]["paired"]["n90"], (k, pa_check["n90"], PL[k]["paired"]["n90"])
        nps, npa = n90_value(s), PL[k]["paired"]["n90"]
        vg, vf = V1[k]["n90"], VF.get(k)
        d = dict(r=r, eps=eps, p0=p0, m=xs[0]["m"], paired_sign=s, paired_sign_n90=nps, paired_a_n90=npa,
                 sign_ok=float(np.mean([x["sign_ok"] for x in xs])),
                 ratio_to_paired=(nps / npa) if nps and npa else None,
                 vault_n90_grid=vg, vault_n90_fine=vf,
                 over_vault_grid=(nps / vg) if nps and vg else None, over_vault_fine=(nps / vf) if nps and vf else None)
        d["within_predicted"] = bool(d["ratio_to_paired"] is not None and d["ratio_to_paired"] <= 1.5)
        res["table"].append(d)
        print(f"| {r} | {eps} | {p0} | {d['m']} | {fmt(s['n90'])} {s['n90']['ci90']} | {npa} | "
              f"{d['ratio_to_paired'] and round(d['ratio_to_paired'], 3)} | {s['successes_at_horizon']}/{s['trials']} | "
              f"{d['sign_ok']:.2f} | {vf} ({vg}) | {d['over_vault_fine'] and round(d['over_vault_fine'], 1)} "
              f"({d['over_vault_grid'] and round(d['over_vault_grid'], 1)}) |")
    T = res["table"]
    res["decision"]["a_learns_a_s"] = dict(learns=bool(T) and all(x["paired_sign_n90"] is not None for x in T))
    res["decision"]["b_sign_cost"] = dict(within_predicted=bool(T) and all(x["within_predicted"] for x in T),
                                          misses=[dict(r=x["r"], eps=x["eps"], p0=x["p0"], ratio=x["ratio_to_paired"])
                                                  for x in T if not x["within_predicted"]],
                                          ratio_range=[min(x["ratio_to_paired"] for x in T if x["ratio_to_paired"]),
                                                       max(x["ratio_to_paired"] for x in T if x["ratio_to_paired"])])
    for eps in sorted({x["eps"] for x in T}):
        for p0 in sorted({x["p0"] for x in T}):
            sel = [x for x in T if x["eps"] == eps and x["p0"] == p0]
            f = [x["over_vault_fine"] for x in sel if x["over_vault_fine"]]
            g = [x["over_vault_grid"] for x in sel if x["over_vault_grid"]]
            res["decision"][f"c_over_vault eps={eps} p0={p0}"] = dict(fine=[min(f), max(f)] if f else None,
                                                                    grid=[min(g), max(g)] if g else None)
    # extend: the self-referencing cost with R given (E13 measured, P's law on R unchanged) and R unknown (bound)
    ok = True
    for r in (4, 6, 8, 10):
        for eps in (0.15, 0.6):
            p0 = 0.005
            b, v = paired_sign_bound(r, eps, p0), unknown_R_bound(r, eps, p0)
            meas = next((x["paired_sign_n90"] for x in T if (x["r"], x["eps"], x["p0"]) == (r, eps, p0)), None)
            ok &= bool(meas is None or meas <= b["units"])
            lb = LB[(r, eps, p0)]
            res["extend"].append(dict(r=r, eps=eps, p0=p0, M=b["M"], paired_sign_bound=b["units"], pairs=b["pairs"],
                                      sign_units=b["sign_units"], paired_sign_measured_R_given=meas,
                                      vault_bound=v["units"], fano_lb=lb, fano_over_paired_sign_bound=lb / b["units"],
                                      fano_over_vault_bound=lb / v["units"]))
    res["decision"]["extend_bound"] = dict(measured_within_bound=ok)
    for x in res["extend"]:
        print("   extend", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
    for k, v in res["decision"].items():
        print(k, v)
    json.dump(res, open(out, "w"), indent=1, default=str)


def vault_ub(r, eps, p0, delta=0.1):
    """Units that suffice information-theoretically for the vault with R given: ML over the sign and all 4^r - 1
    Paulis on r cells, Bhattacharyya coefficient <= 1 - eps_min^2 / 8 (eps_min = eps (1 - 4p0/3)^(2r), both halves),
    union bound: 8 ln(2 (4^r - 1) / delta) / eps_min^2. Error <= delta on every instance."""
    em = eps * (1 - 4 * p0 / 3) ** (2 * r)
    return math.ceil(8 * math.log(2 * (4 ** r - 1) / delta) / em ** 2)


def scope(e10_path, out):
    """Scope analysis (analysis only, no new run). Proof against
    proof for E10: the Fano lower bound for every non-adaptive stabilizer per-unit probe (stored n_lb) over the vault's
    information-theoretic upper bound with R given (vault_ub). For w = r the best coverage is known for every r (the
    maximal full-weight sector length of an r-qubit state is 2^(r-1) + [r even], attained by GHZ: Tran et al. 2016 for
    odd r, Eltschka-Siewert 2020 for even r; for a stabilizer state it is the number of full-weight stabilizer
    elements), so n_lb_theorem uses c* = (2^(r-1) + [r even]) / 3^r. The stored n_lb (ILP upper bound for r = 10) is
    kept as the conservative value."""
    from qmf.experiments.coverage import fano_units
    e10 = json.load(open(e10_path))
    res = dict(weight=[], uniform=[], decision={})
    print("| family | r | eps | p0 | n_lb | n_lb (theorem c*) | vault UB | n_lb / UB | n_lb / measured vault |")
    for x in e10["weight"]:
        r, w, eps, p0 = x["r"], x["w"], x["eps"], x["p0"]
        ub = vault_ub(r, eps, p0)
        thm = None
        if w == r:
            c = (2 ** (r - 1) + (r % 2 == 0)) / 3 ** r
            thm = fano_units(x["M"], c, eps * (1 - 4 * p0 / 3) ** w)
        d = dict(family=f"w={w}", r=r, w=w, eps=eps, p0=p0, n_lb=x["n_lb"], n_lb_theorem=thm, c_upper=x["c_upper"],
                 vault_ub=ub, lb_over_ub=x["n_lb"] / ub, lb_theorem_over_ub=(thm / ub) if thm else None,
                 lb_over_vault_measured=x["lb_over_vault"])
        res["weight"].append(d)
        print(f"| w={w} | {r} | {eps} | {p0} | {x['n_lb']:.0f} | {thm and round(thm)} | {ub} | {d['lb_over_ub']:.2f} | "
              f"{x['lb_over_vault'] and round(x['lb_over_vault'], 1)} |")
    for x in e10["uniform"]:
        r, eps, p0 = x["r"], x["eps"], x["p0"]
        ub = vault_ub(r, eps, p0)
        res["uniform"].append(dict(family="uniform", r=r, eps=eps, p0=p0, n_lb=x["n_lb"], vault_ub=ub,
                                   lb_over_ub=x["n_lb"] / ub, lb_over_vault_measured=x["lb_over_vault"]))
    for name, sel in (("claim2_w_eq_r", lambda x: x["w"] == x["r"]), ("claim3_w_half", lambda x: x["w"] == math.ceil(x["r"] / 2))):
        rows = [x for x in res["weight"] if x["r"] == 10 and sel(x)]
        res["decision"][name] = dict(lb_over_ub_r10={str(x["eps"]): x["lb_over_ub"] for x in rows},
                                     lb_theorem_over_ub_r10={str(x["eps"]): x["lb_theorem_over_ub"] for x in rows},
                                     lb_over_vault_measured_r10={str(x["eps"]): x["lb_over_vault_measured"] for x in rows})
    rows = [x for x in res["uniform"] if x["r"] == 10 and x["p0"] == 0.005]
    res["decision"]["claim1_uniform"] = dict(lb_over_ub_r10={str(x["eps"]): x["lb_over_ub"] for x in rows},
                                             lb_over_vault_measured_r10={str(x["eps"]): x["lb_over_vault_measured"] for x in rows})
    for k, v in res["decision"].items():
        print(k, v)
    json.dump(res, open(out, "w"), indent=1, default=str)


def e14(ml_path, vault_path, e1f_path, out):
    """E14, seal-noise sweep (rules fixed before the run).
    Every trial of a group (lab, r, eps, p0) shares the group's checkpoints, so N90 is the first shared checkpoint with
    success fraction >= 0.9 ('below' if already at the first, 'above' if never). Trials are the same instances at every
    p0, so rho(p0) = N90(p0) / N90(0.005) gets a paired 90% CI: 2000 bootstrap resamples of trial indices per
    (lab, r, eps), the same resample for every p0 (default_rng(0)).
    P1: rho's CI overlaps [A_{3r/4}(p0), A_r(p0)], A_w(p0) = ((1 - 4(0.005)/3) / (1 - 4 p0 / 3))^(2w); below/above fail.
    P2: (per-unit / vault N90)(p0) / (same)(0.005) in [1/1.5, 1.5] for r in {6, 8}, p0 in {0.02, 0.05, 0.1}.
    P3: at p0 = 0.005 the E14 CI overlaps E1f's fine CI.
    D1: law stated if P1 holds for every vault group at r = 10. D2: 'ratio unchanged' if P2 holds everywhere.
    D3: abstract states the trade-off if the vault's rho(0.05) <= 4 at r = 10 for both eps."""
    from qmf.experiments.seal_noise import factor, P0_REF
    rows = load(ml_path) + load(vault_path)
    G = group(rows, ("lab", "r", "eps", "p0"))
    F = {(x["lab"], x["r"], x["eps"]): x["fine"] for x in json.load(open(e1f_path))["table"] if x["p0"] == P0_REF}
    res = dict(table=[], ratio=[], control=[], decision={})
    n90s = {}
    for (lab, r, eps) in sorted({k[:3] for k in G}):
        p0s = sorted(p for (l, rr, e, p) in G if (l, rr, e) == (lab, r, eps))
        mats, pts = {}, {}
        for p0 in p0s:
            rs = sorted(G[(lab, r, eps, p0)], key=lambda x: x["trial"])
            ns = [c["n"] for c in rs[0]["curve"]]
            assert all([c["n"] for c in x["curve"]] == ns for x in rs), "checkpoints differ within a group"
            assert [x["trial"] for x in rs] == list(range(len(rs))), "trials are not 0..T-1"
            mats[p0] = np.array([[c["correct"] for c in x["curve"]] for x in rs], dtype=bool)
            pts[p0] = np.array(ns)
        T = len(next(iter(mats.values())))

        def value(M, x):
            hit = np.flatnonzero(M.mean(axis=0) >= LEVEL)
            if not len(hit):
                return math.inf, "above"
            return float(x[hit[0]]), "below" if hit[0] == 0 else ""
        point = {p0: value(mats[p0], pts[p0]) for p0 in p0s}
        rng = np.random.default_rng(0)
        boot = {p0: [] for p0 in p0s}
        for _ in range(BOOT):
            i = rng.integers(0, T, T)
            for p0 in p0s:
                boot[p0].append(value(mats[p0][i], pts[p0])[0])
        for p0 in p0s:
            n, flag = point[p0]
            ci = [float(v) for v in np.percentile(boot[p0], [5, 95], method="nearest")]
            d = dict(lab=lab, r=r, eps=eps, p0=p0, trials=T, lo=int(pts[p0][0]), hi=int(pts[p0][-1]), n90=n, flag=flag,
                     ci90=ci, seal_T_over_tau=-math.log(1 - 4 * p0 / 3))
            if P0_REF in point and p0 != P0_REF:
                rho = n / point[P0_REF][0]
                rb = np.array(boot[p0]) / np.array(boot[P0_REF])
                rlo, rhi = (float(v) for v in np.percentile(rb, [5, 95], method="nearest"))
                band = [factor(0.75 * r, p0), factor(r, p0)]
                d.update(rho=rho, rho_ci90=[rlo, rhi], band=band,
                         P1=bool(not flag and not point[P0_REF][1] and rlo <= band[1] and rhi >= band[0]))
            res["table"].append(d)
            n90s[(lab, r, eps, p0)] = d
        if P0_REF in point:
            f1 = F[(lab, r, eps)]
            e = n90s[(lab, r, eps, P0_REF)]
            res["control"].append(dict(lab=lab, r=r, eps=eps, e14_n90=e["n90"], e14_ci90=e["ci90"], e1f_n90=f1["n90"],
                                       e1f_ci90=f1["ci90"],
                                       P3=bool(e["ci90"][0] <= f1["ci90"][1] and e["ci90"][1] >= f1["ci90"][0])))
    for (lab, r, eps, p0), d in sorted(n90s.items()):
        if lab != "ml" or p0 == P0_REF or ("vault", r, eps, p0) not in n90s:
            continue
        q_now = d["n90"] / n90s[("vault", r, eps, p0)]["n90"]
        q_ref = n90s[("ml", r, eps, P0_REF)]["n90"] / n90s[("vault", r, eps, P0_REF)]["n90"]
        res["ratio"].append(dict(r=r, eps=eps, p0=p0, ml_over_vault=q_now, ml_over_vault_ref=q_ref, change=q_now / q_ref,
                                 P2=bool(1 / 1.5 <= q_now / q_ref <= 1.5)))
    print("| lab | r | eps | p0 | N90 (90% CI) | rho (90% CI) | band [A_3r/4, A_r] | P1 |")
    for d in res["table"]:
        rho = f"{d['rho']:.2f} [{d['rho_ci90'][0]:.2f}, {d['rho_ci90'][1]:.2f}]" if "rho" in d else "-"
        band = f"[{d['band'][0]:.2f}, {d['band'][1]:.2f}]" if "band" in d else "-"
        print(f"| {d['lab']} | {d['r']} | {d['eps']} | {d['p0']} | {d['n90']}{d['flag'] and ' ' + d['flag']} {d['ci90']} | "
              f"{rho} | {band} | {d.get('P1', '-')} |")
    for d in res["ratio"]:
        print(f"P2 r={d['r']} eps={d['eps']} p0={d['p0']}: ml/vault {d['ml_over_vault']:.1f} vs {d['ml_over_vault_ref']:.1f} "
              f"at 0.005 -> x{d['change']:.2f} {'PASS' if d['P2'] else 'FAIL'}")
    for d in res["control"]:
        print(f"P3 {d['lab']} r={d['r']} eps={d['eps']}: E14 {d['e14_n90']} {d['e14_ci90']} vs E1f {d['e1f_n90']} "
              f"{d['e1f_ci90']} -> {'PASS' if d['P3'] else 'FAIL'}")
    v10 = [d for d in res["table"] if d["lab"] == "vault" and d["r"] == 10 and "rho" in d]
    p1_all = [d for d in res["table"] if "P1" in d]
    rho05 = {str(d["eps"]): d["rho"] for d in v10 if d["p0"] == 0.05}
    res["decision"] = dict(
        P1_pass=sum(d["P1"] for d in p1_all), P1_groups=len(p1_all),
        P1_failures=[f"{d['lab']},r={d['r']},eps={d['eps']},p0={d['p0']}" for d in p1_all if not d["P1"]],
        P2_pass=sum(d["P2"] for d in res["ratio"]), P2_groups=len(res["ratio"]),
        P3_pass=sum(d["P3"] for d in res["control"]), P3_groups=len(res["control"]),
        D1_state_law=bool(v10 and all(d["P1"] for d in v10)),
        D2_ratio_unchanged=bool(res["ratio"] and all(d["P2"] for d in res["ratio"])),
        D3_vault_rho_005_r10=rho05, D3_state_tradeoff=bool(len(rho05) == 2 and all(v <= 4 for v in rho05.values())))
    for k, v in res["decision"].items():
        print(k, v)
    json.dump(res, open(out, "w"), indent=1, default=str)


E6_FIELDS = ("r", "eps", "p0", "trial", "switch", "W", "L0", "L1", "L2", "att_min", "cells_exact", "onset_err",
             "first_ok", "last_ok", "switch_detected", "switch_err")


def e15(path, e6_path, out):
    """E15, tier 3 under transit reordering (post hoc; rules fixed before the
    run). Per (f, r, eps): E6's vault measures on the reordered records; E6's rule (detection
    >= 0.9, false-alarm CP95 upper <= 0.05, cells exact >= 0.9) per group.
    C0: at f = 0 every row equals E6's stored vault row on E6's fields.
    D1: f* = the largest f at which all six settings pass. D2: some setting fails at f = 0.05. D3: some group has a
    false-alarm CP95 upper bound above 0.05. The switch-time error is reported, not judged."""
    rows = load(path)
    e6 = {(x["r"], x["eps"], x["trial"]): x for x in load(e6_path) if x["mode"] == "vault"}
    f0 = [x for x in rows if x["f"] == 0.0]
    mism = [x for x in f0 if any(x[k] != e6[(x["r"], x["eps"], x["trial"])][k] for k in E6_FIELDS)]
    res = dict(table=[], control=dict(rows=len(f0), mismatches=len(mism)))
    print(f"C0: {len(f0)} rows at f = 0, {len(mism)} differ from E6's stored rows")
    print("| f | r | eps | n sw/ns | cells | first/last | detect | false alarm (CP95 up) | q90 |switch err| | tool2 in first / tool1 in last | E6 rule |")
    for (f, r, eps), xs in sorted(group(rows, ("f", "r", "eps")).items()):
        sw, ns = [x for x in xs if x["switch"]], [x for x in xs if not x["switch"]]
        k = sum(x["switch_detected"] for x in ns)
        det = float(np.mean([x["switch_detected"] for x in sw]))
        errs = [abs(x["switch_err"]) for x in sw if x["switch_detected"]]
        d = dict(f=f, r=r, eps=eps, n_switch=len(sw), n_noswitch=len(ns), detect=det, false_alarms=int(k),
                 false_alarm_rate=k / len(ns), fa_cp95_upper=clopper_pearson_upper(k, len(ns)),
                 cells_exact=float(np.mean([x["cells_exact"] for x in xs])),
                 first_ok=float(np.mean([x["first_ok"] for x in xs])), last_ok=float(np.mean([x["last_ok"] for x in xs])),
                 q90_switch_err=float(np.quantile(errs, 0.9)) if errs else None,
                 tool2_first=float(np.mean([x["tool2_first"] for x in sw])),
                 tool1_last=float(np.mean([x["tool1_last"] for x in sw])))
        d["passes_E6"] = bool(d["fa_cp95_upper"] <= 0.05 and det >= 0.9 and d["cells_exact"] >= 0.9)
        res["table"].append(d)
        q = d["q90_switch_err"]
        print(f"| {f} | {r} | {eps} | {len(sw)}/{len(ns)} | {d['cells_exact']:.2f} | {d['first_ok']:.2f}/{d['last_ok']:.2f} | "
              f"{det:.2f} | {k}/{len(ns)} ({d['fa_cp95_upper']:.3f}) | {q if q is None else round(q)} | "
              f"{d['tool2_first']:.2f} / {d['tool1_last']:.2f} | {'PASS' if d['passes_E6'] else 'FAIL'} |")
    fs = sorted({d["f"] for d in res["table"]})
    all_pass = {f: all(d["passes_E6"] for d in res["table"] if d["f"] == f) for f in fs}
    passing = [f for f in fs if all_pass[f]]
    res["decision"] = dict(
        C0_ok=not mism, all_pass_by_f={str(f): v for f, v in all_pass.items()},
        f_star=max(passing) if passing else None,
        monotone=all(all_pass[f] for f in fs if passing and f <= max(passing)),
        D2_fails_at_005=not all_pass.get(0.05, True),
        D3_false_switches=[f"f={d['f']},r={d['r']},eps={d['eps']}" for d in res["table"] if d["fa_cp95_upper"] > 0.05],
        detect_range_by_f={str(f): [min(d["detect"] for d in res["table"] if d["f"] == f),
                                    max(d["detect"] for d in res["table"] if d["f"] == f)] for f in fs},
        fa_max_by_f={str(f): max(d["false_alarms"] for d in res["table"] if d["f"] == f) for f in fs})
    for k_, v in res["decision"].items():
        print(k_, v)
    # Diagnosis written after the results (not a decision rule): false alarms and first-window decoding split by the
    # sign s of the true fingerprint (from the instance seed). Untampered units mixed in after the onset show the
    # identity, whose parities are all +1, so they lift every parity by their share and pull a negative fingerprint
    # toward zero.
    from qmf.experiments.tier3 import switch_instance
    sgn = {k: switch_instance(k[0], k[1], k[2], k[3], 32)["s1"]
           for k in {(x["r"], x["eps"], x["p0"], x["trial"]) for x in rows}}
    by = defaultdict(lambda: dict(noswitch=0, false_alarms=0, trials=0, first_ok=0))
    for x in rows:
        d = by[(x["f"], x["eps"], sgn[(x["r"], x["eps"], x["p0"], x["trial"])])]
        d["trials"] += 1
        d["first_ok"] += int(x["first_ok"])
        if not x["switch"]:
            d["noswitch"] += 1
            d["false_alarms"] += int(x["switch_detected"])
    res["diagnosis_by_sign"] = [dict(f=f, eps=e, s=s, **d) for (f, e, s), d in sorted(by.items())]
    fa_pos = sum(d["false_alarms"] for (f, e, s), d in by.items() if s > 0)
    fa_all = sum(d["false_alarms"] for d in by.values())
    res["diagnosis"] = dict(false_alarms_total=fa_all, false_alarms_positive_sign=fa_pos)
    print(f"diagnosis (post hoc): {fa_all} false alarms in all, {fa_pos} of them with a positive-sign fingerprint")
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    {"e1": e1, "e2": e2, "e3": e3, "e5": e5, "e6": e6, "e7": e7, "pl": pl, "e9": e9, "e10": e10,
     "e11": e11, "e12": e12, "e13": e13, "e14": e14, "e15": e15, "scope": scope}[sys.argv[1]](*sys.argv[2:8])
