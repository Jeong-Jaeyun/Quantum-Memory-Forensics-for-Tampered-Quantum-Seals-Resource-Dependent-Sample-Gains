"""Main result figures for QMF (fig2-fig9), built from the raw results of outputs/results.

Source of every plotted number: the raw rows (*.jsonl) and E10's computation output
(e10_bound.json). The N90s, CIs, rates, fits and crossovers they need are recomputed in memory from those files with
the analysis code (qmf.analysis.derive); no stored summary JSON is read. The stored summaries are only the
record, and qmf.reproduce checks that they equal the recomputed ones.

Every plotted number is also written to source_data/*.csv. Those files are deterministic, so the bundle check compares
them byte for byte. Definitions follow qmf.analysis.summaries:
- N90 is read on the common grid.
- Error bars are 90% bootstrap CIs over trials.
- Rates carry two-sided 95% Clopper-Pearson intervals.
- A censored N90 (no grid point reaches 0.9) is drawn as a lower bound: open marker with an up arrow.

Encoding (fixed across figures):
- vault = blue circles, per-unit = orange squares, per-unit with registry = aqua triangles.
- fig7 (E9, E11, E12; results by attack) reuses the validated triplet with colour = verifier class: vault blue
  (R unknown: open diamond), per-unit product probe orange squares, in-seal reference aqua (E9 lab: triangles up; the
  oracle told R and the copies: triangles down, open for a* not in C), self-referencing P dark gray plus.
  Panel (b) compares the vault with the product probe on all trials (the in-C oracle comparison stays in
  t12_spray).
- Headline numbers fine first: fig2(c) adds the E1f fine ratios (gray crosses); fig4 divides by the E1f fine
  memoryless N90 (E3f), with the grid value kept in its source data.
- Verifier classes: fig2(b, c) add the self-referencing verifier P + sign (E13) as aqua triangles, the one
  triplet hue fig2 did not use. A fourth categorical hue was rejected: with the triplet, no candidate (yellow, magenta,
  violet) clears the all-pairs normal-vision floor (Delta E 15).
- Ordered series (r, K) use a gray ramp with distinct markers.
- eps = 0.15 is filled/solid; eps = 0.6 is open/dashed.

usage: python -m qmf.paper.figures [OUT_DIR] [--data-only]   (default outputs/paper_figure)
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

import qmf.analysis.derive as derive
import qmf.analysis.summaries as rev

S = Path("outputs/results")  # raw rows are read from here
VAULT, ML, REG = "#2a78d6", "#eb6834", "#1baf7a"  # validated: all pairs, white surface (worst CVD dE 9.2)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RAMP = ["#9a9891", "#5f5e5a", "#0b0b0b"]  # ordinal gray, light -> dark
RAMP_MK = ["o", "s", "D"]
EPS_STYLE = {0.15: dict(ls="-", fill=True), 0.6: dict(ls="--", fill=False)}
FULL, HALF, HEIGHT = 7.0, 3.4, 2.25


def J(name):
    """A summary recomputed from the raw results (never the stored summary file)."""
    return derive.J(name)


# ------------------------------------------------------------------ helpers
def cp(k, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ratio_ci(ml, v):
    """Interval from CI endpoints: [ml_lo / v_hi, ml_hi / v_lo]."""
    return ml["ci90"][0] / v["ci90"][1], ml["ci90"][1] / v["ci90"][0]


class Sink:
    """Collects source-data rows per figure and writes them as CSV."""
    def __init__(self):
        self.tables = {}

    def add(self, name, **row):
        self.tables.setdefault(name, []).append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

    def write(self, out):
        d = Path(out) / "source_data"
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in self.tables.items():
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 4.5, "errorbar.capsize": 0,
        "pdf.fonttype": 42, "savefig.dpi": 300})
    return plt


def tag(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=8.5, color=INK, ha="right", va="bottom")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(Path(out) / f"{name}.{ext}", bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else None)


def mk(color, fill, marker):
    return dict(color=color, marker=marker, markerfacecolor=color if fill else "white", markeredgecolor=color,
                markeredgewidth=0.9)


def lower_bound(ax, x, y, color):
    ax.plot([x], [y], marker=r"$\uparrow$", color=color, markersize=7, linestyle="none")


# ------------------------------------------------------------------ fig2: uniform fingerprint (E1)
def success_curve(rows):
    horizon = min(x["n_max"] for x in rows)
    grid = np.unique(np.round(np.geomspace(4, horizon, 40)).astype(np.int64))
    frac = rev.success_matrix(rows, grid).mean(axis=0)
    return grid, frac, len(rows)


def fig2(plt, out, sink, p0=0.005, r_curve=10, name="fig2_uniform"):
    """Uniform fingerprint (E1, E1f, E13). Layout as fig9: (a) success curves at r = r_curve across the
    top; (b), (c) N90 vs r for eps = 0.15 and 0.6 (vault, per-unit, P + sign; shared y axis); (d) N90 relative to the
    vault. Series are labelled in the panels; the source data are unchanged by the layout."""
    e1 = J("e1.json")["table"]
    fig = plt.figure(figsize=(FULL, 4.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.15], hspace=0.62, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)
    ax_d = fig.add_subplot(gs[1, 2])
    by_eps = {0.15: ax_b, 0.6: ax_c}
    # (a) success fraction vs units at r = r_curve
    V = [x for x in rev.load(S / "e1_vault.jsonl") if x["r"] == r_curve and x["p0"] == p0]
    M = [x for x in rev.load(S / "e1_ml.jsonl") if x["r"] == r_curve and x["p0"] == p0]
    ax = ax_a
    for lab, rows_all, color in (("vault", V, VAULT), ("memoryless", M, ML)):
        for eps in (0.15, 0.6):
            rows = [x for x in rows_all if x["eps"] == eps]
            grid, frac, n = success_curve(rows)
            lo, hi = wilson(frac, n)
            mid = np.flatnonzero((frac > 0) & (frac < 1))  # band drawn over the transition (+1 grid point each side)
            if len(mid):
                sl = slice(max(mid[0] - 1, 0), mid[-1] + 2)
                ax.fill_between(grid[sl], lo[sl], hi[sl], color=color, alpha=0.15, linewidth=0)
            ax.plot(grid, frac, color=color, ls=EPS_STYLE[eps]["ls"], lw=1.1)
            for g, f, l, h in zip(grid, frac, lo, hi):
                sink.add(f"{name}_a", lab=lab, eps=eps, r=r_curve, p0=p0, trials=n, n=int(g), success=float(f),
                         wilson95_lo=float(l), wilson95_hi=float(h))
            row = next(x for x in e1 if x["r"] == r_curve and x["eps"] == eps and x["p0"] == p0)
            s = row["vault" if lab == "vault" else "ml"]
            if s["n90"]:
                ax.plot([s["n90"]], [LEVEL], linestyle="none", **mk(color, EPS_STYLE[eps]["fill"],
                                                                     "o" if lab == "vault" else "s"))
            # direct label left of the curve's rise
            x50 = grid[int(np.flatnonzero(frac >= 0.5)[0])] if (frac >= 0.5).any() else grid[-1]
            ax.text(x50 / 1.35, 0.5, f"{'vault' if lab == 'vault' else 'per-unit'}\n$\\varepsilon={eps:g}$",
                    color=color, fontsize=6.5, ha="right", va="center", linespacing=1.0)
    ax.axhline(LEVEL, color=MUTED, lw=0.6)
    ax.text(5, LEVEL + 0.02, "0.9", color=INK2, fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(4, None)
    ax.set_xlabel("seal units $N$")
    ax.set_ylabel("fraction correct")
    ax.set_title(f"identifying $(a^*,s)$ at $r={r_curve}$, $p_0={p0:g}$ (solid: $\\varepsilon=0.15$, dashed: "
                 f"$\\varepsilon=0.6$)", color=INK2)
    tag(ax, "(a)")
    # (b), (c) N90 vs r per eps; (d) ratio vs r
    axc = ax_d
    for eps in (0.15, 0.6):
        st = EPS_STYLE[eps]
        axb = by_eps[eps]
        rows = sorted((x for x in e1 if x["eps"] == eps and x["p0"] == p0), key=lambda x: x["r"])
        for lab, key, color, m in (("vault", "vault", VAULT, "o"), ("memoryless", "ml", ML, "s")):
            rs = [x["r"] for x in rows]
            n90 = [x[key]["n90"] for x in rows]
            lo = [x[key]["ci90"][0] for x in rows]
            hi = [x[key]["ci90"][1] for x in rows]
            axb.errorbar(rs, n90, yerr=[np.subtract(n90, lo), np.subtract(hi, n90)], ls=st["ls"], lw=1.1, elinewidth=0.7,
                         **mk(color, st["fill"], m))
            if lab == "vault":
                axb.text(rs[-1] + 0.25, n90[-1], "vault", color=color, fontsize=6.5, va="center")
            else:
                sl = J("e1.json")["fits"][f"eps={eps},p0={p0}"]["slope"]
                axb.text(rs[-1] + 0.25, n90[-1], f"per-unit\n(slope {sl:.2f})", color=color, fontsize=6.5,
                         va="center", linespacing=1.0)
            for x, n, l, h in zip(rows, n90, lo, hi):
                sink.add(f"{name}_b", lab=lab, eps=eps, p0=p0, r=x["r"], n90=n, ci90_lo=l, ci90_hi=h,
                         censored=x[key]["censored"], at_cap=x[key]["at_cap"])
        rat = [x["ratio"] for x in rows]
        cis = [ratio_ci(x["ml"], x["vault"]) for x in rows]
        axc.errorbar([x["r"] for x in rows], rat, yerr=[np.subtract(rat, [c[0] for c in cis]),
                                                       np.subtract([c[1] for c in cis], rat)],
                     ls=st["ls"], lw=1.1, elinewidth=0.7, **mk(INK, st["fill"], "o"))
        for x, c in zip(rows, cis):
            sink.add(f"{name}_c", eps=eps, p0=p0, r=x["r"], ratio=x["ratio"], ci_endpoint_lo=c[0], ci_endpoint_hi=c[1])
        # fine checkpoints (E1f: the headline ratios), drawn beside the grid ratios
        fine = sorted((x for x in J("e1f.json")["table"] if x["eps"] == eps and x["p0"] == p0), key=lambda x: x["r"])
        fr = {x["r"]: {} for x in fine}
        for x in fine:
            fr[x["r"]][x["lab"]] = x["fine"]["n90"]
        frs = sorted(fr)
        axc.plot(np.add(frs, 0.18), [fr[r]["ml"] / fr[r]["vault"] for r in frs], linestyle="none", marker="x",
                 color=INK2, markersize=4, markeredgewidth=0.8)
        for r in frs:
            sink.add(f"{name}_c_fine", eps=eps, p0=p0, r=r, vault_n90_fine=fr[r]["vault"], memoryless_n90_fine=fr[r]["ml"],
                     ratio_fine=fr[r]["ml"] / fr[r]["vault"])
        # self-referencing verifier, P + sign (E13): N90 for (a*, s) and its ratio to the vault (grid)
        ps = sorted((x for x in J("e13.json")["table"] if x["eps"] == eps and x["p0"] == p0), key=lambda x: x["r"])
        prs = [x["r"] for x in ps]
        pn = [x["paired_sign"]["n90"]["n90"] for x in ps]
        plo = [x["paired_sign"]["n90"]["ci90"][0] for x in ps]
        phi = [x["paired_sign"]["n90"]["ci90"][1] for x in ps]
        axb.errorbar(prs, pn, yerr=[np.subtract(pn, plo), np.subtract(phi, pn)], ls=st["ls"], lw=1.1, elinewidth=0.7,
                     **mk(REG, st["fill"], "^"))
        axb.text(prs[-1] + 0.25, pn[-1], "P + sign", color=REG, fontsize=6.5, va="center")
        vrow = {x["r"]: x["vault"] for x in rows}
        prat = [x["over_vault_grid"] for x in ps]
        pci = [ratio_ci(x["paired_sign"]["n90"], vrow[x["r"]]) for x in ps]
        axc.errorbar(prs, prat, yerr=[np.subtract(prat, [c[0] for c in pci]), np.subtract([c[1] for c in pci], prat)],
                     ls=st["ls"], lw=1.1, elinewidth=0.7, **mk(REG, st["fill"], "^"))
        for x, n, l, h, c in zip(ps, pn, plo, phi, pci):
            sink.add(f"{name}_b", lab="paired_sign", eps=eps, p0=p0, r=x["r"], n90=n, ci90_lo=l, ci90_hi=h,
                     censored=x["paired_sign"]["n90"]["censored"], at_cap=x["paired_sign"]["n90"]["at_cap"])
            sink.add(f"{name}_c_self_ref", eps=eps, p0=p0, r=x["r"], trials=x["paired_sign"]["trials"],
                     ratio_grid=x["over_vault_grid"], ci_endpoint_lo=c[0], ci_endpoint_hi=c[1],
                     ratio_fine=x["over_vault_fine"])
    rs = np.arange(4, 11)
    ref = [x["ratio"] for x in e1 if x["eps"] == 0.15 and x["p0"] == p0 and x["r"] == 4][0]
    axc.plot(rs, ref * 2.0 ** (rs - 4), color=MUTED, lw=0.7)
    axc.set_ylim(2, 4e3)
    axc.text(3.8, 2.6e3, r"per-unit / vault $\propto 2^r$", color=INK, fontsize=6.5, ha="left", va="top")
    axc.text(3.8, 1.25e3, r"($\times$: fine checkpoints)", color=INK2, fontsize=6.5, ha="left", va="top")
    axc.text(3.8, 3.4, r"P + sign / vault $\approx 2/\varepsilon^2$", color=REG, fontsize=6.5, ha="left", va="top")
    fits = J("e1.json")["fits"]
    for eps in (0.15, 0.6):
        sink.add(f"{name}_b_slopes", eps=eps, p0=p0, memoryless_log2_slope=fits[f"eps={eps},p0={p0}"]["slope"])
    for ax in (ax_b, ax_c, ax_d):
        ax.set_yscale("log")
        ax.set_xticks(rs)
        ax.set_xlim(3.6, 13.6 if ax is not ax_d else 10.4)
        ax.set_xlabel("tampered cells $r$")
    ax_b.set_ylim(30, 3e7)
    ax_b.set_ylabel("$N_{90}$ (seal units)")
    plt.setp(ax_c.get_yticklabels(), visible=False)
    ax_d.set_ylabel(r"$N_{90}\,/\,N_{90}^{\rm vault}$")
    ax_b.set_title(r"$N_{90}$, $\varepsilon=0.15$", color=INK2)
    ax_c.set_title(r"$N_{90}$, $\varepsilon=0.6$", color=INK2)
    ax_d.set_title("relative to the vault", color=INK2)
    tag(ax_b, "(b)")
    tag(ax_c, "(c)")
    tag(ax_d, "(d)")
    return fig


LEVEL = rev.LEVEL


# ------------------------------------------------------------------ fig3: fingerprint weight (E2 + ml_ghz + E5)
REGISTERED = ("ml_random", "ml_product", "ml_mix")
ALL_ML = REGISTERED + ("ml_ghz", "ml_adaptive_ghz")


def best(stats, labs):
    v = [stats[m]["n90"] for m in labs if m in stats and stats[m] and stats[m]["usable"]]
    return min(v) if v else None


def fig3(plt, out, sink, name="fig3_weight"):
    tab = J("e2_e5.json")["table"]
    bound = J("e10.json")["weight"]  # Fano bound for any non-adaptive stabilizer probe (E10), / vault N90
    # Mark the proved ratio (Fano lower bound / vault information-theoretic upper
    # bound, scope.json) distinctly from the measured points; drawn where it exceeds 1.
    proved = {(x["r"], x["w"], x["eps"]): x["lb_over_ub"] for x in J("scope.json")["weight"] if x["p0"] == 0.005}
    fig, axs = plt.subplots(1, 2, figsize=(FULL * 0.78, HEIGHT), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, eps in zip(axs, (0.15, 0.6)):
        for i, r in enumerate((6, 8, 10)):
            rows = sorted((x for x in tab if x["r"] == r and x["eps"] == eps), key=lambda x: x["w"])
            vault = {x["w"]: x["stats"]["memory"]["n90"] for x in rows}
            reg = {x["w"]: best(x["stats"], REGISTERED) / vault[x["w"]] for x in rows}
            allb = {x["w"]: best(x["stats"], ALL_ML) / vault[x["w"]] for x in rows}
            ws = [x["w"] for x in rows]
            ax.plot(ws[:-1] + [ws[-1]], [reg[w] for w in ws[:-1]] + [allb[ws[-1]]], color=RAMP[i], lw=1.0,
                    **{k: v for k, v in mk(RAMP[i], True, RAMP_MK[i]).items() if k != "color"})
            ax.plot([ws[-1]], [reg[ws[-1]]], linestyle="none", **mk(RAMP[i], False, RAMP_MK[i]))
            ax.annotate("", xy=(ws[-1], allb[ws[-1]] * 1.12), xytext=(ws[-1], reg[ws[-1]] / 1.12),
                        arrowprops=dict(arrowstyle="->", color=RAMP[i], lw=0.7))
            ax.text(ws[-1] + 0.25, allb[ws[-1]], f"$r={r}$", color=INK2, fontsize=7, va="center")
            lb = {(b["w"]): b["lb_over_vault"] for b in bound if b["r"] == r and b["eps"] == eps}
            ax.plot(ws, [lb[w] for w in ws], color=RAMP[i], lw=0.7, ls=":")
            pw = [w for w in ws if proved.get((r, w, eps), 0) > 1]
            ax.plot(pw, [proved[(r, w, eps)] for w in pw], color=RAMP[i], lw=0.8, ls="-.", marker="_", markersize=7,
                    markeredgewidth=1.2)
            for x in rows:
                w = x["w"]
                sink.add(name, eps=eps, r=r, w=w, vault_n90=vault[w], best_registered=best(x["stats"], REGISTERED),
                         best_with_ghz=best(x["stats"], ALL_ML), ratio_registered=reg[w], ratio_with_ghz=allb[w],
                         predicted_min_3w_2r=x["predicted"], nonadaptive_bound_over_vault=lb[w],
                         proved_lb_over_vault_ub=proved.get((r, w, eps)))
        w = np.arange(1, 5.6, 0.1)
        ax.plot(w, 3.0 ** w, color=MUTED, lw=0.7)
        ax.text(2.0, 3 ** 2.0 * 2.6, "$3^w$", color=INK2, fontsize=7, ha="center")
        ax.set_yscale("log")
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.6, 11.3)
        ax.set_xlabel("fingerprint weight $w=|a^*|$")
        ax.set_title(rf"$\varepsilon={eps}$", color=INK2)
    axs[0].set_ylabel(r"best per-unit $N_{90}$ / vault $N_{90}$")
    tag(axs[0], "(a)")
    tag(axs[1], "(b)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=RAMP[i], lw=1.0, **{k: v for k, v in mk(RAMP[i], True, RAMP_MK[i]).items() if k != "color"})
         for i in range(3)] + [Line2D([], [], linestyle="none", **mk(INK2, False, "o")),
                               Line2D([], [], marker=r"$\downarrow$", color=INK2, linestyle="none", markersize=7),
                               Line2D([], [], color=INK2, lw=0.7, ls=":"),
                               Line2D([], [], color=INK2, lw=0.8, ls="-.", marker="_", markersize=7, markeredgewidth=1.2)]
    fig.legend(h, ["$r=6$", "$r=8$", "$r=10$", "$w=r$: pre-registered probes", "adding GHZ-type probes (post hoc)",
                   "Fano bound / measured vault (not a proof)", "proved: Fano bound / vault upper bound (where > 1)"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.4)
    return fig


# ------------------------------------------------------------------ fig4: vault noise (E3)
def fig4(plt, out, sink, pv_max=0.4, name="fig4_vault_noise"):
    fig, axs = plt.subplots(1, 2, figsize=(FULL * 0.78, HEIGHT), sharey=True, gridspec_kw=dict(wspace=0.08))
    limit = {"dep": 0.2, "deph": 0.15}
    for ax, kind in zip(axs, ("dep", "deph")):
        # the memoryless reference is the E1f fine N90 (E3f, the headline crossover); grid kept in the data
        cross, grid = J(f"e3f_{kind}.json")["crossover"], J(f"e3_final_{kind}.json")["crossover"]
        for i, r in enumerate((6, 8, 10)):
            for eps in (0.15, 0.6):
                c = cross[f"{kind},r={r},eps={eps}"]
                ml, ml_grid = c["memoryless_n90"], grid[f"{kind},r={r},eps={eps}"]["memoryless_n90"]
                pts = [v for v in c["vault"] if v["pv"] <= pv_max + 1e-9]
                xs = [v["pv"] for v in pts if not v["censored"]]
                ys = [v["n90"] / ml for v in pts if not v["censored"]]
                st = EPS_STYLE[eps]
                ax.plot(xs, ys, ls=st["ls"], lw=1.0, **mk(RAMP[i], st["fill"], RAMP_MK[i]))
                for v in pts:
                    if v["censored"]:
                        lower_bound(ax, v["pv"], v["horizon"] / ml, RAMP[i])
                    sink.add(name, noise=kind, r=r, eps=eps, pv=v["pv"], vault_n90=v["n90"], vault_horizon=v["horizon"],
                             censored=v["censored"], memoryless_n90_fine=ml,
                             ratio=(v["n90"] / ml) if v["n90"] else None,
                             ratio_lower_bound=None if v["n90"] else v["horizon"] / ml, side=v["side"],
                             memoryless_n90_grid=ml_grid)
        ax.axhline(1, color=INK2, lw=0.7)
        ax.text(0.005, 1.25, "crossover", color=INK2, fontsize=7)
        ax.axvline(limit[kind], color=MUTED, lw=0.6)
        ax.text(limit[kind] - 0.006, 20, f"stated limit\n$p_v\\leq{limit[kind]}$", color=INK2, fontsize=6.5, ha="right",
                va="center")
        ax.set_yscale("log")
        ax.set_xlim(-0.01, pv_max + 0.02)
        ax.set_xlabel(f"vault noise $p_v$ ({'depolarizing' if kind == 'dep' else 'dephasing'})")
        # Storage time under exponential decay, p_v = (3/4)(1 - e^{-T/tau}) (depolarizing) or
        # (1/2)(1 - e^{-T/T2}) (dephasing). The axis is a change of variable; no number is plotted on it.
        k = 4 / 3 if kind == "dep" else 2.0
        fwd = (lambda p, k=k: -np.log(np.clip(1 - k * np.asarray(p, dtype=float), 1e-9, None)))
        inv = (lambda t, k=k: (1 - np.exp(-np.asarray(t, dtype=float))) / k)
        top = ax.secondary_xaxis("top", functions=(fwd, inv))
        top.set_xlabel(r"storage time $T/\tau$" if kind == "dep" else r"storage time $T/T_2$", color=INK2)
        top.set_xticks([0.0, 0.2, 0.4, 0.6] if kind == "dep" else [0.0, 0.2, 0.5, 1.0, 1.5])
    axs[0].set_ylabel(r"vault $N_{90}$ / per-unit $N_{90}^{\rm fine}$")
    tag(axs[0], "(a)")
    tag(axs[1], "(b)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], lw=1.0, **mk(RAMP[i], True, RAMP_MK[i])) for i in range(3)] + \
        [Line2D([], [], color=INK, ls="-", lw=1.0), Line2D([], [], color=INK, ls="--", lw=1.0),
         Line2D([], [], marker=r"$\uparrow$", color=INK2, linestyle="none", markersize=7)]
    fig.legend(h, ["$r=6$", "$r=8$", "$r=10$", r"$\varepsilon=0.15$", r"$\varepsilon=0.6$", "censored (lower bound)"],
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, 1.08))
    return fig


# ------------------------------------------------------------------ fig5: tier 3 (E6, E8)
def fig5(plt, out, sink, name="fig5_tier3"):
    d = J("e6e8.json")
    groups = [(r, eps) for r in (6, 8, 10) for eps in (0.15, 0.6)]
    labs = [("vault", "vault", VAULT, "o"), ("ml_blind", "per-unit, no registry", ML, "s"),
            ("ml_library", "per-unit + registry", REG, "^")]
    fig, axs = plt.subplots(1, 3, figsize=(FULL, HEIGHT + 0.15), gridspec_kw=dict(wspace=0.38, width_ratios=[1.25, 1, 1]))
    x0 = np.arange(len(groups))
    for j, (mode, label, color, m) in enumerate(labs):
        off = (j - 1) * 0.22
        for g, (r, eps) in enumerate(groups):
            e = d[f"{mode},r={r},eps={eps}"]
            k = int(round(e["detect"] * e["n_switch"]))
            lo, hi = cp(k, e["n_switch"])
            axs[0].errorbar([g + off], [e["detect"]], yerr=[[e["detect"] - lo], [hi - e["detect"]]], elinewidth=0.7,
                            linestyle="none", **mk(color, True, m))
            fhi = e["fa_cp95_upper"]  # one-sided 95% Clopper-Pearson upper bound (the pre-registered E6 metric)
            axs[1].errorbar([g + off], [e["false_alarm_rate"]], yerr=[[0], [fhi - e["false_alarm_rate"]]],
                            elinewidth=0.8, capsize=2.2, capthick=0.8, linestyle="none", **mk(color, True, m))
            if e["q90_att"] is not None:
                axs[2].plot([g + off], [e["q90_att"]], linestyle="none", **mk(color, True, m))
            sink.add(name, lab=mode, r=r, eps=eps, n_switch=e["n_switch"], detected=k, detect=e["detect"],
                     detect_cp95_lo=lo, detect_cp95_hi=hi, n_noswitch=e["n_noswitch"], false_alarms=e["false_alarms"],
                     false_alarm_cp95_onesided_hi=fhi, q90_err_eps_att2=e["q90_att"])
    axs[1].axhline(0.05, color=INK2, lw=0.7)
    axs[1].text(len(groups) - 0.5, 0.053, "0.05 (pre-registered)", color=INK2, fontsize=6.5, ha="right", va="bottom")
    axs[2].axhline(5, color=INK2, lw=0.7)
    axs[2].text(len(groups) - 0.5, 5.15, "5 (pre-registered)", color=INK2, fontsize=6.5, ha="right", va="bottom")
    labels = [f"{r}\n{eps:g}" for r, eps in groups]
    for ax in axs:
        ax.set_xticks(x0)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.text(-0.6, -0.135, "$r$\n$\\varepsilon$", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                ha="right", va="top")
    axs[0].set_ylim(-0.03, 1.05)
    axs[0].set_ylabel("switch detection rate")
    axs[1].set_ylim(-0.003, 0.075)
    axs[1].set_ylabel("false-alarm rate")
    axs[2].set_ylim(0, 9.5)
    axs[2].set_ylabel(r"$q_{90}$ of $|\hat k-k|\,(\varepsilon\,\mu_{\min})^2$")
    # as fig9: panel titles and series names inside the panels instead of a figure legend
    for ax, t, title in zip(axs, ("(a)", "(b)", "(c)"),
                            ("switch detection", "false alarms (none observed)", "switch-time error")):
        tag(ax, t)
        ax.set_title(title, color=INK2)
    yt = axs[0].get_yaxis_transform()
    for y, text, color in ((0.78, "vault (full report)", VAULT), (0.68, "per-unit + registry", REG),
                           (0.12, "per-unit, no registry", ML)):
        axs[0].text(0.04, y, text, transform=yt, color=color, fontsize=6.5, va="center")
    axs[1].text(0.04, 0.0735, "bars: one-sided 95% upper bounds\n(vault 300, per-unit 100 trials)",
                transform=axs[1].get_yaxis_transform(), color=INK2, fontsize=6.5, va="top", linespacing=1.1)
    yt2 = axs[2].get_yaxis_transform()
    axs[2].text(0.04, 9.0, "per-unit + registry", transform=yt2, color=REG, fontsize=6.5, va="top")
    axs[2].text(0.04, 1.2, "vault", transform=yt2, color=VAULT, fontsize=6.5, va="center")
    axs[2].text(0.96, 0.4, "no registry: no switch found", transform=yt2, color=ML, fontsize=6.5, ha="right",
                va="center")
    return fig


# ------------------------------------------------------------------ fig6: several parities, eps unknown (E7)
def fig6(plt, out, sink, name="fig6_multi"):
    e7 = J("e7.json")
    fig, axs = plt.subplots(1, 2, figsize=(FULL * 0.62, HEIGHT), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, eps in zip(axs, (0.3, 0.6)):
        for i, K in enumerate((1, 2, 3)):
            rows = sorted((x for x in e7["table"] if x["K"] == K and x["eps_tot"] == eps), key=lambda x: x["r"])
            rs, rat = [x["r"] for x in rows], [x["ratio"] for x in rows]
            cis = [ratio_ci(x["ml"], x["vault"]) for x in rows]
            off = (i - 1) * 0.12
            ax.errorbar(np.add(rs, off), rat, yerr=[np.subtract(rat, [c[0] for c in cis]),
                                                    np.subtract([c[1] for c in cis], rat)],
                        lw=1.0, elinewidth=0.7, **mk(RAMP[i], True, RAMP_MK[i]))
            dec = e7["decision"][f"K={K},eps_tot={eps}"]
            for x, c in zip(rows, cis):
                sink.add(name, K=K, eps_tot=eps, r=x["r"], vault_n90=x["vault"]["n90"], memoryless_n90=x["ml"]["n90"],
                         ratio=x["ratio"], ci_endpoint_lo=c[0], ci_endpoint_hi=c[1], log2_ratio_slope=dec["ratio_slope"])
        r = np.array([4, 6, 8])
        ax.plot(r, 0.25 * 2.0 ** r, color=MUTED, lw=0.7)
        ax.text(4.1, 0.25 * 2 ** 4 * 0.62, "$0.25\\cdot2^r$\n(pre-registered)", color=INK2, fontsize=7, va="top")
        ax.set_yscale("log")
        ax.set_xticks([4, 6, 8])
        ax.set_xlabel("tampered cells $r$")
        ax.set_title(rf"$\varepsilon_{{\rm tot}}={eps}$", color=INK2)
    axs[0].set_ylabel(r"per-unit $N_{90}$ / vault $N_{90}$")
    tag(axs[0], "(a)")
    tag(axs[1], "(b)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], lw=1.0, **mk(RAMP[i], True, RAMP_MK[i])) for i in range(3)]
    fig.legend(h, ["$K=1$", "$K=2$", "$K=3$"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    return fig


# ------------------------------------------------------------------ fig7: results by attack (E9, E11, E12)
# One attack figure, panels by attack (E9, E11, E12). Colour is the verifier class, marker the lab:
#   vault blue (circle; R unknown: open diamond), per-unit product probe orange square, in-seal reference aqua
#   (E9 lab: triangle up; oracle told R and the copies: triangle down, filled for a* in C, open otherwise),
#   self-referencing P dark gray plus. Orange squares now mean the per-unit product probe only, so P left orange.
# The source-data rows keep the order and fields of the former fig7/fig8 files (only the file names changed).
INSEAL, PAIRED, PRODUCT = REG, INK2, ML
GROUP_LABEL = "$r$\n$\\varepsilon$"


def n90_panel(ax, series, rs, title, panel):
    """N90 vs r with 90% bootstrap bars; series: (lab, colour, marker, {(r, eps): n90 stats}, sink row function)."""
    for lab, color, m, stats, row in series:
        for eps in (0.15, 0.6):
            st = EPS_STYLE[eps]
            pts = [(r, stats[(r, eps)]) for r in rs if (r, eps) in stats]
            x, n = [p[0] for p in pts], [p[1]["n90"] for p in pts]
            lo, hi = [p[1]["ci90"][0] for p in pts], [p[1]["ci90"][1] for p in pts]
            ax.errorbar(x, n, yerr=[np.subtract(n, lo), np.subtract(hi, n)], ls=st["ls"], lw=1.0, elinewidth=0.7,
                        **mk(color, st["fill"], m))
            for r, s in pts:
                row(lab, r, eps, s)
    ax.set_yscale("log")
    ax.set_xticks(rs)
    ax.set_xlabel("tampered cells $r$")
    ax.set_ylabel("$N_{90}$ (seal units)")
    ax.set_title(title, color=INK2)
    tag(ax, panel)


def horizon_panel(ax, groups, series, title, panel):
    """Success at the horizon with 95% Clopper-Pearson bars, one column of markers per (r, eps);
    series: (colour, marker, fill, {(r, eps): entry}, sink row function)."""
    step = 0.22 if len(series) <= 3 else 0.16
    for j, (color, m, fill, data, row) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * step
        for g, (r, eps) in enumerate(groups):
            e = data.get((r, eps))
            if not e:
                continue
            y = e["success_at_horizon"]
            lo, hi = e["success_cp95"]
            ax.errorbar([g + off], [y], yerr=[[y - lo], [hi - y]], elinewidth=0.7, linestyle="none", **mk(color, fill, m))
            row(r, eps, e, y, lo, hi)
    ax.axhline(0.05, color=MUTED, lw=0.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{r}\n{eps:g}" for r, eps in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.text(-0.6, -0.135, GROUP_LABEL, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, ha="right", va="top")
    ax.set_ylim(-0.04, 1.06)
    ax.set_ylabel("success at the horizon")
    ax.set_title(title, color=INK2)
    tag(ax, panel)


def fig7(plt, out, sink, name="fig7_attacks"):
    e9 = {(x["lab"], x["attacker"], x["r"], x["eps"]): x for x in J("e9.json")["table"]}
    sp = {(x["lab"], x["r"], x["eps"]): x for x in J("e11.json")["table"]}
    for x in J("e12.json")["table"]:
        sp[("product", x["r"], x["eps"])] = dict(all=x["product"], in_C=x["product_in_C"], not_in_C=x["product_not_in_C"])
    rs9, rs11 = sorted({k[2] for k in e9}), sorted({k[1] for k in sp})
    g9 = [(r, eps) for r in rs9 for eps in (0.15, 0.6)]
    g11 = [(r, eps) for r in rs11 for eps in (0.15, 0.6)]

    fig = plt.figure(figsize=(FULL, 2 * HEIGHT + 0.1))
    gs = fig.add_gridspec(2, 6, hspace=0.58, wspace=1.15, top=0.95)
    ax_a, ax_b = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])
    ax_c, ax_d, ax_e = (fig.add_subplot(gs[1, 2 * i:2 * i + 2]) for i in range(3))

    # (a) oblivious attacker: N90 of the in-seal reference and of the full-seal vault (E9), both with tier 1
    def row_a(lab, r, eps, s):
        sink.add(f"{name}_a", lab=lab, attacker="obl", eps=eps, r=r, n90=s["n90"], ci90_lo=s["ci90"][0],
                 ci90_hi=s["ci90"][1], n90_interp=s["n90_interp"], censored=s["censored"])
    n90_panel(ax_a, [(lab, c, m, {(r, eps): e9[(lab, "obl", r, eps)]["n90"] for r, eps in g9 if (lab, "obl", r, eps) in e9},
                      row_a) for lab, c, m in (("vault_seal", VAULT, "o"), ("inseal", INSEAL, "^"))],
              rs9, "oblivious: untouched cells are a reference", "(a)")

    # (b) spray: N90 on all trials, vault (R not given) vs product probe (no reference), same trials and checkpoints
    def row_b(lab, r, eps, s):
        sink.add(f"{name}_b", lab=lab, subset="all", eps=eps, r=r, trials=sp[(lab, r, eps)]["all"]["trials"],
                 n90=s["n90"], ci90_lo=s["ci90"][0], ci90_hi=s["ci90"][1], n90_interp=s["n90_interp"],
                 censored=s["censored"])
    n90_panel(ax_b, [(lab, c, m, {(r, eps): sp[(lab, r, eps)]["all"]["n90"] for r, eps in g11}, row_b)
                     for lab, c, m in (("vault_seal", VAULT, "o"), ("product", PRODUCT, "s"))],
              rs11, "spray: copies read without a reference", "(b)")

    # (c) extend, (e) mirror (E9): success at the horizon; (d) spray (E11, E12)
    def e9_series(att, labs, sub):
        out_ = []
        for lab, color, m, fill in labs:
            data = {(r, eps): e9[(lab, att, r, eps)] for r, eps in g9 if (lab, att, r, eps) in e9}
            if not data:
                continue

            def row(r, eps, e, y, lo, hi, lab=lab):
                sink.add(f"{name}_{sub}", lab=lab, attacker=att, r=r, eps=eps, trials=e["trials"],
                         successes=e["successes_at_horizon"], success=y, cp95_lo=lo, cp95_hi=hi, horizon=e["horizon"],
                         n90=e["n90"]["n90"])
            out_.append((color, m, fill, data, row))
        return out_
    horizon_panel(ax_c, g9, e9_series("extend", (("vault_seal", VAULT, "o", True), ("inseal", INSEAL, "^", True),
                                                 ("vault_sparse", VAULT, "D", False)), "c"),
                  "extend: references destroyed", "(c)")
    spray = []
    for lab, part, color, m, fill in (("vault_seal", "all", VAULT, "o", True), ("product", "all", PRODUCT, "s", True),
                                      ("inseal_oracle", "in_C", INSEAL, "v", True),
                                      ("inseal_oracle", "not_in_C", INSEAL, "v", False),
                                      ("inseal", "all", INSEAL, "^", True)):
        def row(r, eps, e, y, lo, hi, lab=lab, part=part):
            sink.add(f"{name}_d", lab=lab, subset=part, r=r, eps=eps, trials=e["trials"],
                     successes=e["successes_at_horizon"], success=y, cp95_lo=lo, cp95_hi=hi, horizon=e["horizon"])
        spray.append((color, m, fill, {(r, eps): sp[(lab, r, eps)][part] for r, eps in g11}, row))
    horizon_panel(ax_d, g11, spray, "spray: Bell pairs blind", "(d)")
    horizon_panel(ax_e, g9, e9_series("mirror", (("vault_seal", VAULT, "o", True), ("inseal", INSEAL, "^", True),
                                                 ("paired", PAIRED, "P", True)), "e"),
                  "mirror: pairing leaked", "(e)")

    # As fig9: series named inside the panels, in the series colour, instead of a figure legend.
    # Success panels: labels stacked in the empty middle (data sit at 0 and 1); x in axes fraction, y in data.
    def stack(ax, items):
        for y, text, color in items:
            ax.text(0.04, y, text, transform=ax.get_yaxis_transform(), color=color, fontsize=6.5, ha="left",
                    va="center")
    for ax in (ax_a, ax_b):
        ax.set_ylim(None, 2.5e4)
        ax.text(0.97, 0.44, r"solid, filled: $\varepsilon=0.15$" + "\n" + r"dashed, open: $\varepsilon=0.6$",
                transform=ax.transAxes, color=INK2, fontsize=6.5, ha="right", va="center", linespacing=1.1)
    ax_a.text(0.04, 0.93, "in-seal reference (no vault)", transform=ax_a.transAxes, color=INSEAL, fontsize=6.5, va="top")
    ax_a.text(0.04, 0.84, "vault (full seal)", transform=ax_a.transAxes, color=VAULT, fontsize=6.5, va="top")
    ax_b.text(0.04, 0.93, "per-unit product probe (no reference)", transform=ax_b.transAxes, color=PRODUCT, fontsize=6.5,
              va="top")
    ax_b.text(0.04, 0.84, r"vault ($R$ not given)", transform=ax_b.transAxes, color=VAULT, fontsize=6.5, va="top")
    stack(ax_c, [(0.84, r"vault, $R$ given", VAULT), (0.74, r"vault, $R$ unknown ($r=4$, open)", VAULT),
                 (0.16, "in-seal reference", INSEAL)])
    stack(ax_d, [(0.86, r"vault ($R$ not given)", VAULT), (0.77, "per-unit product probe", PRODUCT),
                 (0.68, r"in-seal oracle, $a^*\in C$", INSEAL),
                 (0.26, r"in-seal oracle, $a^*\notin C$ (open)", INSEAL), (0.16, "in-seal reference", INSEAL)])
    stack(ax_e, [(0.84, "vault", VAULT), (0.26, "in-seal reference", INSEAL), (0.16, "P (self-referencing)", PAIRED)])
    return fig


# ------------------------------------------------------------------ fig8: seal-noise sweep (E14)
def fig8(plt, out, sink, name="fig8_seal_noise"):
    """E14: cost factor rho(p0) = N90(p0) / N90(0.005) for the vault (a) and the per-unit strategy (b), with paired 90%
    bootstrap bars and the predicted band [A_{3r/4}, A_r] (shaded, one tone per r); (c) the per-unit / vault N90 ratio
    relative to p0 = 0.005, with the pre-registered P2 band [1/1.5, 1.5]. Top axes: storage time of a seal cell,
    T / tau_s = -ln(1 - 4 p0 / 3). Encoding as fig4: r by the gray ramp and marker, eps by fill and line style."""
    from qmf.experiments.seal_noise import factor, P0_REF
    E = J("e14.json")
    T = {(x["lab"], x["r"], x["eps"], x["p0"]): x for x in E["table"]}
    from matplotlib.ticker import NullFormatter, NullLocator
    fig = plt.figure(figsize=(FULL, HEIGHT + 0.3))
    gs = fig.add_gridspec(1, 3, wspace=0.36, width_ratios=[1.12, 1.12, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    axs = [ax0, fig.add_subplot(gs[0, 1], sharey=ax0), fig.add_subplot(gs[0, 2])]
    fwd = (lambda p: -np.log(np.clip(1 - 4 / 3 * np.asarray(p, dtype=float), 1e-9, None)))
    inv = (lambda t: 0.75 * (1 - np.exp(-np.asarray(t, dtype=float))))
    pp = np.geomspace(P0_REF, 0.1, 80)
    p0_ticks = [0.005, 0.01, 0.02, 0.05, 0.1]

    def p0_axis(ax, xlabel):
        ax.set_xlim(0.0042, 0.17)
        ax.set_xscale("log")
        ax.set_xticks(p0_ticks)
        ax.set_xticklabels([f"{p:g}" for p in p0_ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(xlabel)
        top = ax.secondary_xaxis("top", functions=(fwd, inv))
        top.set_xscale("log")
        top.set_xticks([0.01, 0.03, 0.1])
        top.set_xticklabels(["0.01", "0.03", "0.1"])
        top.xaxis.set_minor_locator(NullLocator())
        top.xaxis.set_minor_formatter(NullFormatter())
        top.set_xlabel(r"seal storage time $T/\tau_s$", color=INK2, fontsize=7)
    for ax, lab, rs, title, panel in ((axs[0], "vault", (6, 8, 10), "vault ($p_v=0.005$)", "(a)"),
                                      (axs[1], "ml", (6, 8), "per-unit (uniform Lagrangian)", "(b)")):
        for i, r in enumerate((6, 8, 10)):
            if r not in rs:
                continue
            ax.fill_between(pp, factor(0.75 * r, pp), factor(r, pp), color=RAMP[i], alpha=0.14, lw=0)
            for eps in (0.15, 0.6):
                st = EPS_STYLE[eps]
                p0s = sorted(p for (l, rr, e, p) in T if (l, rr, e) == (lab, r, eps))
                xs, ys, lo, hi = [], [], [], []
                for p0 in p0s:
                    d = T[(lab, r, eps, p0)]
                    rho, ci = (1.0, [1.0, 1.0]) if p0 == P0_REF else (d["rho"], d["rho_ci90"])
                    xs.append(p0 * (1.0 if eps == 0.15 else 1.06))
                    ys.append(rho)
                    lo.append(rho - ci[0])
                    hi.append(ci[1] - rho)
                    sink.add(f"{name}_{'a' if lab == 'vault' else 'b'}", lab=lab, r=r, eps=eps, p0=p0, n90=d["n90"],
                             flag=d["flag"], ci90_lo=d["ci90"][0], ci90_hi=d["ci90"][1], rho=rho, rho_ci90_lo=ci[0],
                             rho_ci90_hi=ci[1], band_lo=factor(0.75 * r, p0), band_hi=factor(r, p0), P1=d.get("P1", ""),
                             seal_T_over_tau=d["seal_T_over_tau"])
                ax.errorbar(xs, ys, yerr=[lo, hi], ls=st["ls"], lw=1.0, elinewidth=0.7, **mk(RAMP[i], st["fill"], RAMP_MK[i]))
                if eps == 0.15:
                    ax.text(xs[-1] * 1.18, ys[-1], f"$r={r}$", color=RAMP[i], fontsize=6.5, va="center")
        ax.text(0.04, 0.95, title, transform=ax.transAxes, color=INK, fontsize=7, va="top")
        ax.text(0.04, 0.86, r"solid, filled: $\varepsilon=0.15$" + "\n" + r"dashed, open: $\varepsilon=0.6$" + "\n" +
                r"shaded: predicted $[A_{3r/4},A_r]$", transform=ax.transAxes, color=INK2, fontsize=6.5, va="top",
                linespacing=1.1)
        ax.set_yscale("log")
        p0_axis(ax, "seal noise $p_0$")
        tag(ax, panel)
    axs[0].set_ylim(0.8, 22)
    axs[0].set_yticks([1, 2, 4, 8, 16])
    axs[0].set_yticklabels(["1", "2", "4", "8", "16"])
    axs[0].yaxis.set_minor_formatter(NullFormatter())
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r"cost factor $N_{90}(p_0)\,/\,N_{90}(0.005)$")
    ax = axs[2]
    ax.axhspan(1 / 1.5, 1.5, color=GRID, alpha=0.6, lw=0)
    ax.axhline(1, color=INK2, lw=0.7)
    for i, r in enumerate((6, 8)):
        for eps in (0.15, 0.6):
            st = EPS_STYLE[eps]
            pts = [(P0_REF, 1.0)] + [(x["p0"], x["change"]) for x in E["ratio"] if x["r"] == r and x["eps"] == eps]
            pts.sort()
            ax.plot([p * (1.0 if eps == 0.15 else 1.06) for p, _ in pts], [c for _, c in pts], ls=st["ls"], lw=1.0,
                    **mk(RAMP[i], st["fill"], RAMP_MK[i]))
            if eps == 0.15:
                ax.text(pts[-1][0] * 1.18, pts[-1][1], f"$r={r}$", color=RAMP[i], fontsize=6.5, va="center")
            for x in E["ratio"]:
                if x["r"] == r and x["eps"] == eps:
                    sink.add(f"{name}_c", r=r, eps=eps, p0=x["p0"], ml_over_vault=x["ml_over_vault"],
                             ml_over_vault_ref=x["ml_over_vault_ref"], change=x["change"], P2=x["P2"])
    ax.text(0.04, 0.95, "vault's advantage over per-unit", transform=ax.transAxes, color=INK, fontsize=7, va="top")
    ax.text(0.04, 0.86, r"shaded: pre-registered $\times1.5$ band", transform=ax.transAxes, color=INK2, fontsize=6.5,
            va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.45, 2.6)
    ax.set_yticks([0.5, 0.67, 1, 1.5, 2])
    ax.set_yticklabels(["0.5", "0.67", "1", "1.5", "2"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    p0_axis(ax, "seal noise $p_0$")
    ax.set_ylabel(r"per-unit / vault $N_{90}$, relative")
    tag(ax, "(c)")
    return fig


# ------------------------------------------------------------------ fig9: one incident, raw records to report (E6)
CASE = dict(r=6, eps=0.6, p0=0.005, t=0)  # chosen before it was drawn


def pauli_label(a, r):
    return "".join("IXZY"[((a >> i) & 1) + 2 * ((a >> (i + r)) & 1)] for i in range(r))


def case_study(r, eps, p0, t, n=32):
    """Regenerate the E6 vault trial (seed space 932) exactly as switch_trial2('vault') does, check it against the
    stored row, and return the intermediate quantities of qmf_report (tiers 1-3)."""
    import json
    import qmf.core as q
    import qmf.experiments.tier3 as tier3
    I = tier3.switch_instance(r, eps, p0, t, n)
    b2, t2 = (I["a2"], I["s2"]) if I["switch"] else (I["a1"], I["s1"])
    rng = np.random.default_rng([932, r, int(round(eps * 1000)), int(round(p0 * 1e5)), t, 1])
    codes = q.simulate_seal(n, I["R"], [(I["L0"], None, 0, 0.0), (I["L1"], I["a1"], I["s1"], eps), (I["L2"], b2, t2, eps)],
                            p0, rng)
    rep = q.qmf_report(codes, p0, window=I["W"])
    regenerated = json.loads(json.dumps(tier3.switch_trial2(("vault", r, eps, p0, t, n))))
    stored = next(x for x in map(json.loads, open(S / "e6.jsonl")) if x["mode"] == "vault" and x["r"] == r
                  and x["eps"] == eps and x["p0"] == p0 and x["trial"] == t)
    if regenerated != stored:
        raise AssertionError(f"case study: regenerated E6 row differs from the stored row\n{regenerated}\n{stored}")
    N = len(codes)
    rate = (codes != 0).mean(axis=0)
    base = float(np.median(rate))
    thr = base + 5 * math.sqrt(max(base, 1 / N) / N) + 5 / N
    R = np.array(rep["cells"])
    y = (codes[:, R] != 0).any(axis=1).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(y)])
    k = np.arange(1, N)

    def ll(ones, m):
        p = np.clip(ones / m, 1e-12, 1 - 1e-12)
        return ones * np.log(p) + (m - ones) * np.log1p(-p)
    onset_ll = ll(c[k], k) + ll(c[N] - c[k], N - k)
    onset = rep["onset"]
    E = q.pack_cells(codes[onset:, R])
    allp = np.arange(4 ** r, dtype=np.int64)
    att = (1 - 4 * p0 / 3) ** (2 * q.weight(allp, r))
    W = I["W"]

    def spectrum(Ew):
        lam = q.fwht(np.bincount(Ew, minlength=4 ** r))[q.swap_xz(allp, r)] / len(Ew)
        lam[0] = 0.0
        return lam
    lam_first, lam_last = spectrum(E[:W]), spectrum(E[-W:])
    sig = math.sqrt(2 * math.log(2 * 4 ** r / 0.01) / W)
    f, l = rep["first"], rep["last"]

    def llf(fp):
        e = min(max(fp["eps"] * att[fp["a"]], 1e-3), 1 - 1e-3)
        return np.log1p(fp["s"] * e * (1 - 2 * q.symp(E, np.int64(fp["a"]), r)))
    gain = llf(l) - llf(f)
    tail = np.concatenate([np.cumsum(gain[::-1])[::-1], [0.0]])
    return dict(I=I, b2=b2, t2=t2, codes=codes, rep=rep, row=stored, N=N, rate=rate, base=base, thr=thr,
                b0=q.benign_rate(p0), R=R, onset=onset, onset_ll=onset_ll, lam_first=lam_first, lam_last=lam_last,
                sig=sig, W=W, tail=tail, tail_thr=2 * math.log(len(E)))


def fig9(plt, out, sink, name="fig9_case_study"):
    cs = case_study(**CASE)
    I, rep, r = cs["I"], cs["rep"], CASE["r"]
    L0, L1 = I["L0"], I["L1"]
    fig = plt.figure(figsize=(FULL, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.34)
    # (a) raw records: non-identity Bell outcomes, units x cells
    ax = fig.add_subplot(gs[0, :])
    u, cell = np.nonzero(cs["codes"] != 0)
    tampered = np.isin(cell, cs["R"])
    ax.scatter(u[~tampered], cell[~tampered], s=1.2, color=MUTED, linewidths=0, rasterized=True)
    ax.scatter(u[tampered], cell[tampered], s=1.2, color=INK, linewidths=0, rasterized=True)
    for x0, lab, ls in ((L0, "true onset", "--"), (L0 + L1, "true tool switch", ":")):
        ax.axvline(x0, color=VAULT, lw=0.9, ls=ls)
        ax.text(x0 + 5, 29.5, lab, color=VAULT, fontsize=6.5, va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_xlim(0, cs["N"])
    ax.set_ylim(-1, 32)
    ax.set_xlabel("seal unit (arrival order)")
    ax.set_ylabel("cell")
    tag(ax, "(a)")
    for uu, cc in zip(u.tolist(), cell.tolist()):
        sink.add(f"{name}_events", unit=uu, cell=cc, tampered=bool(cc in cs["R"].tolist()))
    # (b) tier 1: cell error rates and the localization threshold
    ax = fig.add_subplot(gs[1, 0])
    cells = np.arange(32)
    inR = np.isin(cells, cs["R"])
    ax.bar(cells[~inR], cs["rate"][~inR], color=MUTED, width=0.8)
    ax.bar(cells[inR], cs["rate"][inR], color=INK, width=0.8)
    ax.axhline(cs["thr"], color=VAULT, lw=0.9)
    ax.text(31.5, cs["thr"] * 1.15, "threshold", color=VAULT, fontsize=6.5, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("cell")
    ax.set_ylabel(r"error rate $\rho_c$")
    ax.set_title("tier 1: where", color=INK2)
    tag(ax, "(b)")
    for cc in cells:
        sink.add(f"{name}_cells", cell=int(cc), rate=float(cs["rate"][cc]), tampered=bool(inR[cc]),
                 localized=bool(cc in rep["cells"]), threshold=cs["thr"], median=cs["base"])
    # (c) tier 2: parity spectrum of the first and the last window
    ax = fig.add_subplot(gs[1, 1])
    idx = np.arange(4 ** r)
    ax.scatter(idx, np.abs(cs["lam_first"]), s=1.0, color=VAULT, linewidths=0, alpha=0.6, rasterized=True)
    ax.scatter(idx, np.abs(cs["lam_last"]), s=1.0, color=MUTED, linewidths=0, alpha=0.6, rasterized=True)
    ax.axhline(cs["sig"], color=INK2, lw=0.7, ls="--")
    ax.text(60, cs["sig"] - 0.015, "significance", color=INK2, fontsize=6.5, va="top")
    for a, s, lab, col, lam, dy in ((I["a1"], I["s1"], "first window", VAULT, cs["lam_first"], 1),
                                    (cs["b2"], cs["t2"], "last window", MUTED, cs["lam_last"], -1)):
        ax.plot([a], [abs(lam[a])], marker="o", ms=4.5, markerfacecolor="white", markeredgecolor=col,
                markeredgewidth=1.0, linestyle="none")
        ax.annotate(f"{lab}: {'+' if s > 0 else '-'}{pauli_label(a, r)}", xy=(a, abs(lam[a])), xytext=(-8, dy),
                    textcoords="offset points", fontsize=6.5, color=INK if col == MUTED else col, ha="right",
                    va="center")
    ax.set_ylim(0, 0.7)
    ax.set_xlim(-50, 4 ** r + 50)
    ax.set_xlabel(f"Pauli $b$ on $\\hat R$ (all $4^{r}$)")
    ax.set_ylabel(r"$|\hat\lambda_b|$")
    ax.set_title(f"tier 2: first / last {cs['W']} units", color=INK2)
    tag(ax, "(c)")
    # the five largest |lambda| of each window, by a stable sort: ties go to the lower index, so the listed set does not
    # depend on the numpy version (the default quicksort's order at ties did: numpy 2.3.5 and 2.5.0 differed)
    top = sorted(set(np.argsort(-np.abs(cs["lam_first"]), kind="stable")[:5].tolist()
                     + np.argsort(-np.abs(cs["lam_last"]), kind="stable")[:5].tolist() + [I["a1"], cs["b2"]]))
    for b in top:
        sink.add(f"{name}_spectrum", b=int(b), pauli=pauli_label(int(b), r), lam_first=float(cs["lam_first"][b]),
                 lam_last=float(cs["lam_last"][b]), tool1=bool(b == I["a1"]), tool2=bool(b == cs["b2"]),
                 significance_threshold=cs["sig"], window=cs["W"])
    # (d) tier 3: switch statistic G_k
    ax = fig.add_subplot(gs[1, 2])
    ks = np.arange(len(cs["tail"])) + cs["onset"]
    ax.plot(ks, cs["tail"], color=INK, lw=0.9)
    ax.axhline(cs["tail_thr"], color=INK2, lw=0.7, ls="--")
    ax.axvline(L0 + L1, color=VAULT, lw=0.9, ls=":")
    ax.text((L0 + L1 + ks[-1]) / 2, cs["tail_thr"] - 2, r"$2\,\ln\,N_{\rm post}$", color=INK2, fontsize=6.5,
            ha="center", va="top")  # below the line, right of the switch: clear of the curve
    ax.set_xlabel("candidate switch unit $k$")
    ax.set_ylabel(r"switch statistic $G_k$")
    ax.set_title("tier 3: when the tool changed", color=INK2)
    tag(ax, "(d)")
    for kk, g in zip(ks[::4].tolist(), cs["tail"][::4].tolist()):
        sink.add(f"{name}_tier3", k=int(kk), G=float(g), threshold=cs["tail_thr"])
    f, l = rep["first"], rep["last"]
    sink.add(f"{name}_summary", r=r, eps=CASE["eps"], p0=CASE["p0"], trial=CASE["t"], units=cs["N"], L0=L0, L1=L1,
             L2=I["L2"], window=cs["W"], status=rep["status"], tier3=rep["tier3"], cells_true=" ".join(map(str, I["R"])),
             cells_found=" ".join(map(str, rep["cells"])), onset_true=L0, onset_found=rep["onset"],
             tool1=f"{'+' if I['s1'] > 0 else '-'}{pauli_label(I['a1'], r)}",
             first_window=f"{'+' if f['s'] > 0 else '-'}{pauli_label(f['a'], r)}",
             tool2=f"{'+' if cs['t2'] > 0 else '-'}{pauli_label(cs['b2'], r)}",
             last_window=f"{'+' if l['s'] > 0 else '-'}{pauli_label(l['a'], r)}",
             switch_true=L0 + L1, switch_found=rep.get("switch"), stored_row_matches=True)
    return fig


FIGS = [("fig2_uniform", fig2), ("fig3_weight", fig3), ("fig4_vault_noise", fig4), ("fig5_tier3", fig5),
        ("fig6_multi", fig6), ("fig7_attacks", fig7), ("fig8_seal_noise", fig8), ("fig9_case_study", fig9)]


def main(out, data_only=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt = setup()
    sink = Sink()
    for name, fn in FIGS:
        fig = fn(plt, out, sink)
        if not data_only:
            save(fig, out, name)
        plt.close(fig)
        print(name)
    sink.write(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "outputs/paper_figure", "--data-only" in sys.argv)
