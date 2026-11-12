"""E1f: fine checkpoints around N90 for E1 (rules fixed before running).

E1 reads N90 on a 40-point geometric grid, so most of its bootstrap CIs span at most one grid step. E1f replays every E1
trial from its seeds, with the same draws in the same order, up to the top of a window around its group's N90. It also
evaluates the MAP decoder at fine points inside the window: geometric points with ratio <= 1.03, plus the grid points in
the window.
- vault (run_vault): the counts between two of the trial's checkpoints are one draw. The counts at a fine point inside
  that interval are drawn from their exact conditional law given the interval's counts. The units are exchangeable, so
  the first m of them are a multivariate hypergeometric sample of the interval's counts. These draws use their own RNG,
  so the trial's own draws are unchanged.
- memoryless (run_memoryless, ml_random): chunks end at the trial's checkpoints as before. A fine point inside a chunk
  splits the chunk's integer sufficient statistics (the first m units of the chunk). The draws are unchanged.
- MAP ties: at the trial's own checkpoints the replay evaluates the decoder at every checkpoint from the first one on,
  with the trial's own tie-break RNG, as qmf.experiments.uniform.trial does, so the tie-breaks there are the trial's. At fine points
  that are not checkpoints, ties are broken by a separate RNG. (The first E1f run broke every tie with the separate RNG
  and failed the replay check at tied MAPs.)
- Replay check: the decoder's results at the trial's own checkpoints inside the window ('own'): post_true and the
  correct flag must equal the stored E1 values exactly.

Window of a group (lab, r, eps, p0), from the stored E1 rows: from the grid point before the first grid point with success
fraction >= 0.5, to the grid point after the upper end of the grid CI.

Fine N90 = first fine point with success fraction >= 0.9 (success = exact MAP correctness at that n). Its 90% CI uses the
same 2000 bootstrap resamples of trials as the grid CI (paired). A resample whose grid N90 lies below the window, or that
does not reach 0.9 inside the window, keeps its grid value and is counted (boot_below, boot_above).

usage: python -m qmf.experiments.fine windows E1_ML.jsonl E1_VAULT.jsonl OUT.json
       python -m qmf.experiments.fine run WORKERS WINDOWS.json LAB E1_ROWS.jsonl OUT.jsonl [R_LIST]   (LAB: vault | ml)
       python -m qmf.experiments.fine analyze E1F_ML.jsonl E1F_VAULT.jsonl E1_ML.jsonl E1_VAULT.jsonl WINDOWS.json OUT.json
"""
import json
import math
import multiprocessing
import sys

import numpy as np

import qmf.core as q
import qmf.analysis.summaries as rev

STEP = 1.03
METHOD = {"vault": "memory", "ml": "ml_random"}


def key(lab, r, eps, p0):
    return f"{lab},{r},{eps},{p0}"


def grid_of(rows):
    horizon = min(x["n_max"] for x in rows)
    return np.unique(np.round(np.geomspace(4, horizon, 40)).astype(np.int64))


def group_window(rows):
    grid = grid_of(rows)
    f = rev.success_matrix(rows, grid).mean(axis=0)
    s = rev.n90(rows)
    j0 = int(np.flatnonzero(f >= 0.5)[0])
    k_hi = int(np.searchsorted(grid, s["ci90"][1]))  # the grid CI's upper end is a grid point (or inf)
    lo, hi = int(grid[max(j0 - 1, 0)]), int(grid[min(k_hi + 1, len(grid) - 1)])
    k = math.ceil(math.log(hi / lo) / math.log(STEP)) + 1
    pts = np.unique(np.concatenate([np.round(np.geomspace(lo, hi, k)).astype(np.int64), grid[(grid >= lo) & (grid <= hi)]]))
    return dict(lo=lo, hi=hi, points=[int(p) for p in pts], grid_n90=s["n90"], grid_ci90=s["ci90"])


def windows(ml_path, vault_path, out):
    res = {}
    for lab, path in (("ml", ml_path), ("vault", vault_path)):
        for (r, eps, p0), rows in sorted(rev.group(rev.load(path), ("r", "eps", "p0")).items()):
            w = res[key(lab, r, eps, p0)] = group_window(rows)
            print(f"{lab:5s} r={r:2d} eps={eps} p0={p0}: grid N90 {w['grid_n90']} {w['grid_ci90']} -> window "
                  f"[{w['lo']}, {w['hi']}] ({w['hi'] / w['lo']:.2f}x), {len(w['points'])} points")
    json.dump(res, open(out, "w"), indent=1)


# ------------------------------------------------------------------ replay
def setup(method, r, eps, p0, t):
    """Instance, horizon and RNGs exactly as qmf.experiments.uniform.trial (scramble model, wt = None, pv = None, vkind = 'dep')."""
    tag = [941, 0, r, int(round(eps * 1000)), int(round(p0 * 1e5)), 0, t]
    rng = np.random.default_rng(tag)
    a = int(rng.integers(1, 4 ** r))
    s = int(rng.choice([-1, 1]))
    w = int(q.weight(np.int64(a), r))
    sub = [q.METHODS.index(method), 0, 0]
    mu_a = float(q.vault_mu(np.int64(a), r, p0, None, "dep"))
    n_max = q.n_max_for("scramble", method, r, eps, w, mu_a, p0, 0.0)
    # data and checkpoint tie-breaks as E1 (tags 1, 2); tie-breaks at the other fine points (3); vault bridge draws (4)
    rngs = [np.random.default_rng(tag + [k] + sub) for k in (1, 2, 3, 4)]
    return a, s, n_max, rngs


class Recorder:
    """Evaluates the decoder once per n, at every trial checkpoint up to hi (with the trial's tie-break RNG, in the
    trial's order) and at the window's fine points (the others with a separate tie-break RNG). Files the results under
    'fine' (window points) and 'own' (trial checkpoints in the window)."""

    def __init__(self, pts, lo, hi, cps, evaluate, rng_eval, rng_tie):
        self.fine_set, self.cps, self.lo = set(pts), {n for n in cps if n <= hi}, lo
        self.fine, self.own, self.evaluate = [], [], evaluate
        self.rng_eval, self.rng_tie = rng_eval, rng_tie

    def at(self, n, state):
        own = n in self.cps
        if not own and n not in self.fine_set:
            return
        d = self.evaluate(state, n, self.rng_eval if own else self.rng_tie)
        d = dict(n=int(n), correct=d["correct"], post_true=d["post_true"])
        if n in self.fine_set:
            self.fine.append(d)
        if own and n >= self.lo:
            self.own.append(d)


def replay_vault(r, eps, p0, t, pts, lo, hi):
    a, s, n_max, (rng, rng_eval, rng_tie, rng_bridge) = setup("memory", r, eps, p0, t)
    allp = np.arange(4 ** r, dtype=np.int64)
    mu = q.vault_mu(allp, r, p0, None, "dep")
    e = np.minimum(eps * mu, 1 - 1e-12)
    lp, lm = np.log1p(e), np.log1p(-e)
    sw = q.swap_xz(allp, r)

    def evaluate(h, n, rng_t):  # as run_vault
        lam = q.fwht(h)[sw] / n
        c0, c1 = n * (1 + lam) / 2, n * (1 - lam) / 2
        return q.eval_point(np.stack([c0 * lp + c1 * lm, c0 * lm + c1 * lp]), a, s, r, rng_t)
    cps = q.checkpoints(n_max)
    rec = Recorder(pts, lo, hi, cps, evaluate, rng_eval, rng_tie)
    inner = sorted(set(pts) - set(cps))
    h, done = np.zeros(4 ** r, dtype=np.int64), 0
    for n in cps:
        if done >= hi:
            break
        inc = q.vault_counts(r, a, s, eps * mu[a], n - done, rng)  # the trial's own draw, as in run_vault
        mid = [p for p in inner if done < p < n]
        if mid:  # counts at fine points inside (done, n): sequential multivariate hypergeometric draws from inc
            pool, cur, hc = inc.copy(), done, h.copy()
            for p in mid:
                d = rng_bridge.multivariate_hypergeometric(pool, p - cur)
                hc += d
                pool -= d
                cur = p
                rec.at(p, hc)
        h += inc
        done = n
        rec.at(n, h)
    return a, s, n_max, rec


def replay_ml(r, eps, p0, t, pts, lo, hi):
    a, s, n_max, (rng, rng_eval, rng_tie, _) = setup("ml_random", r, eps, p0, t)
    allp = np.arange(4 ** r, dtype=np.int64)
    e_eff = np.minimum(eps * (1 - 4 * p0 / 3) ** q.weight(allp, r), 1 - 1e-12)
    lp, lm = np.log1p(e_eff), np.log1p(-e_eff)

    def evaluate(cnt, n, rng_t):  # as run_memoryless (scramble)
        c0, c1 = cnt[0::2], cnt[1::2]
        return q.eval_point(np.stack([c0 * lp + c1 * lm, c0 * lm + c1 * lp]), a, s, r, rng_t)
    cnt = np.zeros(2 * 4 ** r, dtype=np.int64)
    chunk = max(q.BATCH, 2 ** 23 // 2 ** r)
    cps = q.checkpoints(n_max)
    rec = Recorder(pts, lo, hi, cps, evaluate, rng_eval, rng_tie)
    todo = sorted(set(pts) | rec.cps)
    done = 0
    while done < n_max and done < hi:
        B = min(chunk, n_max - done, (cps[0] - done) if cps else chunk)  # as run_memoryless: never past a checkpoint
        gens = q.probe_generators("ml_random", r, rng, B, None, done)
        E = q.depolarize(q.sample("scramble", B, r, a, s, eps, eps, rng), r, p0, rng)
        sig = q.symp(gens, E[:, None], r)
        idx = 2 * q.span_elements(gens)[:, 1:] + q.syndrome_parities(sig)[:, 1:]
        cur = 0
        while todo and todo[0] <= done + B:
            p = todo.pop(0)
            cnt += np.bincount(idx[cur:p - done].ravel(), minlength=2 * 4 ** r)
            cur = p - done
            rec.at(p, cnt)
        cnt += np.bincount(idx[cur:].ravel(), minlength=2 * 4 ** r)
        done += B
        while cps and cps[0] <= done:
            cps.pop(0)
    return a, s, n_max, rec


def fine_trial(job):
    lab, r, eps, p0, t, pts, lo, hi = job
    a, s, n_max, rec = (replay_vault if lab == "vault" else replay_ml)(r, eps, p0, t, pts, lo, hi)
    if [p["n"] for p in rec.fine] != list(pts):
        raise RuntimeError(f"fine points missing: {lab} r={r} eps={eps} p0={p0} t={t}")
    return dict(lab=lab, method=METHOD[lab], r=r, eps=eps, p0=p0, trial=t, a=a, s=s, n_max=n_max, lo=lo, hi=hi,
                fine=rec.fine, own=rec.own)


def run(workers, win_path, lab, e1_path, out, rs=None):
    W = json.load(open(win_path))
    jobs = []
    for x in rev.load(e1_path):
        if rs and x["r"] not in rs:
            continue
        w = W[key(lab, x["r"], x["eps"], x["p0"])]
        jobs.append((lab, x["r"], x["eps"], x["p0"], x["trial"], w["points"], w["lo"], w["hi"]))
    jobs.sort(key=lambda j: -(2 ** j[1]) * j[7])  # largest first
    with multiprocessing.Pool(workers, maxtasksperchild=4) as pool, open(out, "a") as fh:
        for row in pool.imap_unordered(fine_trial, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["lab"], row["r"], row["eps"], row["p0"], row["trial"], row["hi"], flush=True)


# ------------------------------------------------------------------ analysis
def fine_n90(F, C, pts, grid, lo, hi, rng):
    """F: exact success at the fine points (trials x points); C: E1 grid success matrix (same trial order)."""
    def first(M, xs):
        hit = np.flatnonzero(M.mean(axis=0) >= rev.LEVEL)
        return int(xs[hit[0]]) if len(hit) else None

    def value(Fb, Cb):
        g, f = first(Cb, grid), first(Fb, pts)
        if g is not None and g < lo:
            return g, "below"
        if f is None:
            return (math.inf if g is None else g), "above"
        return f, ""
    n, flag = value(F, C)
    boot, below, above = [], 0, 0
    for _ in range(rev.BOOT):
        i = rng.integers(0, len(F), len(F))
        v, fl = value(F[i], C[i])
        boot.append(v)
        below += fl == "below"
        above += fl == "above"
    lo_ci, hi_ci = np.percentile(boot, [5, 95], method="nearest")
    k = int(np.searchsorted(pts, n)) if not flag else None
    # ci_points: fine points inside the CI, both ends included (prediction 3: "covers >= 3 fine points")
    return dict(n90=n, flag=flag, ci90=[float(lo_ci), float(hi_ci)], boot_below=below, boot_above=above,
                step=float(pts[k] / pts[k - 1]) if k else None,
                ci_points=int(np.searchsorted(pts, hi_ci, "right") - np.searchsorted(pts, lo_ci, "left"))
                if hi_ci < math.inf else None)


def analyze(ml_f, vault_f, ml_path, vault_path, win_path, out):
    W = json.load(open(win_path))
    res = dict(table=[], replay={}, decision={})
    stats = dict(own=0, post_equal=0, correct_equal=0)
    for lab, fpath, path in (("ml", ml_f, ml_path), ("vault", vault_f, vault_path)):
        Fg = rev.group(rev.load(fpath), ("r", "eps", "p0"))
        for (r, eps, p0), rows in sorted(rev.group(rev.load(path), ("r", "eps", "p0")).items()):
            w = W[key(lab, r, eps, p0)]
            by_t = {x["trial"]: x for x in Fg.get((r, eps, p0), [])}
            if len(by_t) != len(rows):
                raise RuntimeError(f"{lab} r={r} eps={eps} p0={p0}: {len(by_t)} fine rows for {len(rows)} trials")
            fine = [by_t[x["trial"]] for x in rows]  # E1 row order, so the bootstrap draws the same trials
            for x, y in zip(rows, fine):
                stored = {p["n"]: p for p in x["curve"]}
                for p in y["own"]:
                    stats["own"] += 1
                    stats["post_equal"] += p["post_true"] == stored[p["n"]]["post_true"]
                    stats["correct_equal"] += p["correct"] == stored[p["n"]]["correct"]
            pts = np.array(w["points"])
            F = np.array([[p["correct"] for p in y["fine"]] for y in fine], dtype=bool)
            grid = grid_of(rows)
            C = rev.success_matrix(rows, grid)
            g = rev.n90(rows)  # default_rng(0): the grid CI's resamples
            fn = fine_n90(F, C, pts, grid, w["lo"], w["hi"], np.random.default_rng(0))
            dev = math.log(fn["n90"] / g["n90_interp"]) if g["n90_interp"] and fn["n90"] < math.inf else None
            res["table"].append(dict(lab=lab, r=r, eps=eps, p0=p0, trials=len(rows), lo=w["lo"], hi=w["hi"],
                                     points=len(pts), grid=g, fine=fn, fine_le_grid=bool(fn["n90"] <= g["n90"]),
                                     log_fine_over_interp=dev,
                                     interp_within_half_step=bool(dev is not None and abs(dev) <= math.log(g["grid_step"]) / 2)))
    res["replay"] = dict(stats, exact=bool(stats["post_equal"] == stats["own"] and stats["correct_equal"] == stats["own"]))
    T = res["table"]
    p0zero = [x for x in T if x["p0"] == 0.0]
    res["decision"] = dict(
        replay_exact=res["replay"]["exact"],
        fine_le_grid=sum(x["fine_le_grid"] for x in T) / len(T),
        resolved=sum(bool(x["fine"]["ci_points"] and x["fine"]["ci_points"] >= 3) for x in T) / len(T),
        degenerate=sum(x["fine"]["ci90"][0] == x["fine"]["ci90"][1] for x in T),
        flagged=sum(bool(x["fine"]["flag"]) for x in T),
        boot_outside_max=max((x["fine"]["boot_below"] + x["fine"]["boot_above"]) / rev.BOOT for x in T),
        interp_within_half_step_p0_0=sum(x["interp_within_half_step"] for x in p0zero) / len(p0zero),
        interp_within_half_step_all=sum(x["interp_within_half_step"] for x in T) / len(T),
        median_log_fine_over_interp={str(p0): float(np.median([x["log_fine_over_interp"] for x in T if x["p0"] == p0]))
                                     for p0 in sorted({x["p0"] for x in T})})
    # E1 rules on the fine values (robustness check; the pre-registered decision stays on the grid)
    rules = {}
    for eps in sorted({x["eps"] for x in T}):
        for p0 in sorted({x["p0"] for x in T}):
            V = {x["r"]: x["fine"]["n90"] for x in T if x["lab"] == "vault" and x["eps"] == eps and x["p0"] == p0}
            M = {x["r"]: x["fine"]["n90"] for x in T if x["lab"] == "ml" and x["eps"] == eps and x["p0"] == p0}
            ci = all(V[r] * eps ** 2 / r <= 10 for r in V)
            sl = rev.slope([(r, M[r]) for r in M])
            ratio = M[max(M)] / V[max(M)]
            rules[f"eps={eps},p0={p0}"] = dict(rule_i=ci, slope=sl, ratio_r10=ratio,
                                              passes=bool(ci and sl >= 0.7 and ratio >= 10))
    res["decision"]["e1_rules_on_fine"] = rules
    print("| lab | r | eps | p0 | window | pts | grid N90 [CI] | interp | fine N90 [CI] | fine CI pts | boot out |")
    for x in T:
        g, fn = x["grid"], x["fine"]
        print(f"| {x['lab']} | {x['r']} | {x['eps']} | {x['p0']} | [{x['lo']}, {x['hi']}] | {x['points']} | "
              f"{g['n90']} {g['ci90']} | {g['n90_interp'] and round(g['n90_interp'])} | {fn['n90']}{fn['flag']} {fn['ci90']} | "
              f"{fn['ci_points']} | {fn['boot_below']}/{fn['boot_above']} |")
    for k, v in res["decision"].items():
        print(k, v)
    print("replay:", res["replay"])
    json.dump(res, open(out, "w"), indent=1, default=str)


# ------------------------------------------------------------------ fine values in the headline numbers
def e3f(path, e1f_path, out):
    """E3f (analysis only, rules fixed before it ran). The E3 crossover with the E1f fine memoryless
    N90 (p0 = 0.005) as the target. The vault side stays on the grid; its grid N90 also comes late, so p_v* is on the
    low (conservative) side."""
    T = {(x["r"], x["eps"]): x["fine"]["n90"] for x in json.load(open(e1f_path))["table"]
         if x["lab"] == "ml" and x["p0"] == 0.005}
    rev.e3(path, None, out, targets=T)


def headline(e1f_path, e3f_dep, e3f_deph, e3_dep, e3_deph, out):
    """Headline numbers with fine values first and the grid alongside: the E1 ratio at
    r = 10, the E10 uniform-family Fano bound over the vault N90 at r = 10 (as qmf.analysis.summaries.e10, with the E1f fine
    vault N90 as well), and the E3 crossover p_v* (fine target vs grid target), with the claim-7 check: is the vault ahead
    at p_v = 0.2 (depolarizing) and 0.15 (dephasing) for every (r, eps) with the fine target?"""
    from qmf.experiments.coverage import fano_units
    T = json.load(open(e1f_path))["table"]
    G = {(x["lab"], x["r"], x["eps"], x["p0"]): x for x in T}
    res = dict(e1_ratio_r10=[], e10_uniform_r10=[], e3=[], decision={})
    for eps in (0.15, 0.6):
        for p0 in (0.0, 0.005):
            m, v = G[("ml", 10, eps, p0)], G[("vault", 10, eps, p0)]
            res["e1_ratio_r10"].append(dict(eps=eps, p0=p0, fine=m["fine"]["n90"] / v["fine"]["n90"],
                                            grid=m["grid"]["n90"] / v["grid"]["n90"]))
            lb = fano_units(4 ** 10 - 1, (2 ** 10 - 1) / (4 ** 10 - 1), eps)
            res["e10_uniform_r10"].append(dict(eps=eps, p0=p0, n_lb=lb, vault_fine=v["fine"]["n90"], vault_grid=v["grid"]["n90"],
                                               lb_over_vault_fine=lb / v["fine"]["n90"], lb_over_vault_grid=lb / v["grid"]["n90"]))
    thresholds = {"dep": 0.2, "deph": 0.15}
    ok = {}
    for kind, fpath, gpath in (("dep", e3f_dep, e3_dep), ("deph", e3f_deph, e3_deph)):
        F, Gr = json.load(open(fpath))["crossover"], json.load(open(gpath))["crossover"]
        ok[kind] = True
        for key, c in F.items():
            g = Gr[key]
            at = next(x["side"] for x in c["vault"] if x["pv"] == thresholds[kind])
            ok[kind] &= at == "ahead"
            res["e3"].append(dict(key=key, target_fine=c["memoryless_n90"], target_grid=g["memoryless_n90"],
                                  bracket_fine=c["bracket"], bracket_grid=g["bracket"], interp_fine=c["interpolated"],
                                  interp_grid=g["interpolated"], side_at_threshold_fine=at))

    def rng(xs):
        xs = [x for x in xs if x is not None]
        return [min(xs), max(xs)] if xs else None
    res["decision"] = dict(
        e1_ratio_r10_fine=rng([x["fine"] for x in res["e1_ratio_r10"]]),
        e1_ratio_r10_grid=rng([x["grid"] for x in res["e1_ratio_r10"]]),
        e10_uniform_lb_over_vault_r10_fine=rng([x["lb_over_vault_fine"] for x in res["e10_uniform_r10"]]),
        e10_uniform_lb_over_vault_r10_grid=rng([x["lb_over_vault_grid"] for x in res["e10_uniform_r10"]]),
        **{f"e3_{kind}_{part}_{t}": rng([x[f"{part}_{t}"][i] if part == "bracket" else x[f"{part}_{t}"]
                                         for x in res["e3"] if x["key"].startswith(kind + ",")
                                         for i in ([0, 1] if part == "bracket" else [0])])
           for kind in ("dep", "deph") for part in ("interp", "bracket") for t in ("fine", "grid")},
        claim7_thresholds_hold=ok)
    for k, v in res["decision"].items():
        print(k, v)
    for x in res["e3"]:
        print("  ", x)
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    A = sys.argv
    if A[1] == "windows":
        windows(A[2], A[3], A[4])
    elif A[1] == "e3f":
        e3f(A[2], A[3], A[4])
    elif A[1] == "headline":
        headline(*A[2:8])
    elif A[1] == "run":
        run(int(A[2]), A[3], A[4], A[5], A[6], [int(x) for x in A[7].split(",")] if len(A) > 7 else None)
    elif A[1] == "analyze":
        analyze(*A[2:8])
