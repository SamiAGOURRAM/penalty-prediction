"""Generate the report figures (consistent style) into outputs/figs/."""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

import common as C

warnings.filterwarnings("ignore")
FIGS = C.OUT / "figs"
FIGS.mkdir(exist_ok=True)

# ---- consistent visual identity ----
NAVY, TEAL, AMBER, RED, GREY = "#2b3a67", "#1f9e89", "#e8a33d", "#c0392b", "#9aa0a6"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
REL = ["natural", "center", "nonnatural"]


def fig_calibration():
    """3-way reliability: hierarchical vs footedness vs per-player table."""
    import fit_project1 as F
    import os
    os.environ["CORPUS_FILE"] = "men_penalties_full_v2.csv"
    d = C.load_men(1.5); d["y"] = d["dir_rel"].map({c: i for i, c in enumerate(REL)})
    d = d.sample(frac=1, random_state=42).reset_index(drop=True)
    test = d.groupby("dir_rel", group_keys=False).sample(frac=0.25, random_state=42)
    train = d.drop(test.index)
    P_h = np.load(C.OUT / "p1_test_probs_hier.npy"); y = np.load(C.OUT / "p1_test_y.npy")
    P_m = F.base_marginal(train, test); P_f = F.base_freq(train, test)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for P, lab, mk, col in [(P_h, "hierarchical", "o", NAVY),
                            (P_f, "per-player table", "^", RED),
                            (P_m, "footedness", "s", AMBER)]:
        conf = P.max(1); correct = (P.argmax(1) == y).astype(float)
        bins = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
        for i in range(10):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.sum() >= 3:
                xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
        ax.plot(xs, ys, mk + "-", color=col, lw=1.8, ms=7,
                label=f"{lab}  (ECE {C.ece(P, y):.3f})")
        for x, yv, nn in zip(xs, ys, ns):
            ax.annotate(str(nn), (x, yv), fontsize=6.5, color=col,
                        textcoords="offset points", xytext=(3, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set(xlabel="model confidence (top class)", ylabel="empirical accuracy",
           xlim=(0.28, 0.86), ylim=(0.15, 1.0))
    ax.set_title("Reliability: predicted confidence vs empirical accuracy")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.savefig(FIGS / "fig_calibration.png"); plt.close(fig)


def fig_shrinkage():
    """Raw vs shrunken natural-side share by sample size."""
    t = pd.read_csv(C.OUT / "player_tendencies.csv")
    raw_nat = np.where(t["foot"] == "Left", t["raw_L"], t["raw_R"])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.axhspan(0.47, 0.53, color=AMBER, alpha=0.18, zorder=0, label="population (~0.50)")
    sc = ax.scatter(raw_nat, t["P_natural"], s=40 + 5 * t["n"], c=t["n"],
                    cmap="viridis", edgecolor="k", linewidth=0.4, zorder=3)
    for x, y, nm, nn in zip(raw_nat, t["P_natural"], t["player"], t["n"]):
        if nn >= 20 or abs(x - y) > 0.18:   # only the anchors + dramatic shrinkers
            ax.annotate(f"{nm.split()[-1]} ({nn})", (x, y), fontsize=7.5,
                        textcoords="offset points", xytext=(7, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no shrinkage")
    ax.set(xlabel="raw natural-side share", ylabel="shrunken estimate",
           xlim=(0.05, 1.08), ylim=(0.30, 0.62))
    ax.set_title("Raw vs partially pooled natural-side estimate, by sample size")
    fig.colorbar(sc, label="# penalties (n)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "fig_shrinkage.png"); plt.close(fig)


def fig_keeper():
    """Panel A: dive distribution (action + side bias). Panel B: gambler's-fallacy
    streak rates with bootstrap CIs vs the 0.5 null + power annotation."""
    p2 = json.load(open(C.OUT / "project2_results.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.4))

    # A: dive distribution (relative to kicker's natural side)
    br = json.load(open(C.OUT / "bridge_results.json"))["OFFENCE_H8"]["keeper_dive_dist"]
    vals = [br["natural"], br["center"], br["nonnatural"]]
    bars = a.bar(["natural\nside", "centre", "non-natural\nside"], vals,
                 color=[TEAL, GREY, TEAL], edgecolor="k", linewidth=0.5)
    a.axhline(1 / 3, ls="--", color="k", lw=1)
    a.text(2.3, 0.345, "uniform 1/3", fontsize=8, ha="right")
    for r, v in zip(bars, vals):
        a.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=10)
    a.set(ylim=(0, 0.58), ylabel="share of dives")
    a.set_title("Goalkeeper dive distribution\n(relative to the kicker's natural side)")

    # B: gambler's fallacy streaks
    gf = p2["H6_gamblers_fallacy"]
    xs, rates, los, his, ns, pw = [], [], [], [], [], []
    for k in [1, 2, 3]:
        s = gf[f"streak_{k}"]
        xs.append(k); rates.append(s["opposite_rate"])
        los.append(s["opposite_rate"] - s["bootstrap_rate_ci95"][0])
        his.append(s["bootstrap_rate_ci95"][1] - s["opposite_rate"])
        ns.append(s["n_cases"]); pw.append(s.get("power_at_true_0.65", float("nan")))
    b.errorbar(xs, rates, yerr=[los, his], fmt="o", color=NAVY, ms=9, capsize=5, lw=1.8)
    b.axhline(0.5, ls="--", color=RED, lw=1.4)
    b.text(3.05, 0.505, "no fallacy (0.5)", color=RED, fontsize=8, va="bottom", ha="right")
    for x, r, n, p in zip(xs, rates, ns, pw):
        b.annotate(f"n={n}\npower={p:.2f}", (x, r), fontsize=7.5,
                   textcoords="offset points", xytext=(10, -4))
    b.set(xlabel="same-direction kick streak length", ylabel="P(keeper dives opposite)",
          xticks=[1, 2, 3], xlim=(0.6, 3.6), ylim=(0.1, 0.95))
    b.set_title("Opposite-dive rate after a same-direction\nkick streak (95% CI)")
    fig.savefig(FIGS / "fig_keeper.png"); plt.close(fig)


def fig_bridge():
    """Panel A: payoff heatmap P(goal|kick,dive). Panel B: expected score by kick
    choice vs the always-natural baseline (the exploitation)."""
    br = json.load(open(C.OUT / "bridge_results.json"))["OFFENCE_H8"]
    pay = br["payoff_P_goal"]
    M = np.array([[pay[f"kick_{kr}"][f"dive_{dc}"] for dc in REL] for kr in REL])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.5),
                               gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
    a.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    a.set_xticks(range(3), labels=["natural", "centre", "non-nat"])
    a.set_yticks(range(3), labels=["natural", "centre", "non-nat"])
    a.set(xlabel="keeper dive", ylabel="kick placement")
    a.grid(False)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                   color="black", fontsize=12,
                   fontweight="bold" if i == j else "normal")
    a.set_title("P(goal) by kick placement and keeper dive\n(green = goal, red = save)")

    ev = br["EV_score_by_kick"]; rr = br["realized_score_rate_by_kick"]
    order = ["natural", "center", "nonnatural"]
    lab = ["natural", "centre", "non-natural"]
    evv = [ev[k] for k in order]; rv = [rr[k]["rate"] for k in order]
    x = np.arange(3)
    bars = b.bar(x, evv, color=[GREY, GREY, TEAL], edgecolor="k", linewidth=0.5,
                 width=0.6, label="model expected score")
    b.scatter(x, rv, color=NAVY, zorder=5, s=55, label="realized score rate")
    b.axhline(ev["natural"], ls="--", color=RED, lw=1.3)
    b.text(2.45, ev["natural"] - 0.022, "always-natural baseline", color=RED,
           fontsize=8, ha="right")
    for xi, v in zip(x, evv):
        b.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=10)
    gain = br["model_gain_nonnatural_vs_natural"]
    b.annotate(f"+{(ev['nonnatural']-ev['natural'])*100:.0f} pp vs natural\n95% CI {gain['ci95']}",
               (1.0, 0.80), fontsize=8.5, ha="center", color=TEAL, fontweight="bold")
    b.set(xticks=x, ylim=(0.55, 0.84), ylabel="P(score)")
    b.set_xticklabels(lab)
    b.set_title("Expected and realized score by kick placement")
    b.legend(fontsize=8, loc="lower left")
    fig.savefig(FIGS / "fig_bridge.png"); plt.close(fig)


def _kaggle_rel():
    d = C.load_kaggle()
    d["natural_side"] = np.where(d.Foot == "R", "L", "R")
    rel = lambda s, n: "center" if s == "C" else ("natural" if s == n else "nonnatural")
    d["kick_rel"] = [rel(s, n) for s, n in zip(d.kick_LCR, d.natural_side)]
    d["dive_rel"] = [rel(s, n) for s, n in zip(d.Keeper, d.natural_side)]
    d["matched"] = d.kick_rel == d.dive_rel
    return d


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


ORDER = ["natural", "center", "nonnatural"]
LABS = ["natural", "centre", "non-natural"]


def fig_execution():
    """Panel A: P(goal | on target) by placement. Panel B: how often the keeper
    ends up on your side (follow/match rate). Both with Wilson 95% CIs."""
    d = _kaggle_rel()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    x = np.arange(3)
    # A: execution quality
    pts = [_wilson(d[(d.kick_rel == z) & (d.OnTarget == 1)].Goal.sum(),
                   len(d[(d.kick_rel == z) & (d.OnTarget == 1)])) for z in ORDER]
    p = [v[0] for v in pts]; lo = [v[0] - v[1] for v in pts]; hi = [v[2] - v[0] for v in pts]
    cols = [GREY, GREY, TEAL]
    a.bar(x, p, color=cols, edgecolor="k", linewidth=0.5, width=0.6)
    a.errorbar(x, p, yerr=[lo, hi], fmt="none", ecolor="k", capsize=4, lw=1.2)
    for xi, v in zip(x, p):
        a.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    a.set(xticks=x, ylim=(0.4, 0.95), ylabel="P(goal | shot on target)")
    a.set_xticklabels(LABS)
    a.set_title("P(goal | shot on target) by placement")
    # B: keeper follow / match rate
    pts2 = [_wilson(d[d.kick_rel == z].matched.sum(), len(d[d.kick_rel == z])) for z in ORDER]
    p2 = [v[0] for v in pts2]; lo2 = [v[0] - v[1] for v in pts2]; hi2 = [v[2] - v[0] for v in pts2]
    cols2 = [RED, TEAL, AMBER]
    b.bar(x, p2, color=cols2, edgecolor="k", linewidth=0.5, width=0.6)
    b.errorbar(x, p2, yerr=[lo2, hi2], fmt="none", ecolor="k", capsize=4, lw=1.2)
    for xi, v in zip(x, p2):
        b.text(xi, v + 0.03, f"{v:.0%}", ha="center", fontsize=10)
    b.set(xticks=x, ylim=(0, 0.8), ylabel="P(keeper is on your side)")
    b.set_xticklabels(LABS)
    b.set_title("Keeper match rate by placement\nP(keeper on the kicked side)")
    fig.tight_layout(); fig.savefig(FIGS / "fig_execution.png"); plt.close(fig)


def fig_middle():
    """The clincher: P(goal) when the keeper guesses WRONG vs RIGHT, by placement,
    with the follow rate annotated. Answers 'why not just shoot the middle?'."""
    d = _kaggle_rel()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(3); w = 0.38
    wrong, right, foll = [], [], []
    for z in ORDER:
        g = d[d.kick_rel == z]
        wrong.append(_wilson(g[~g.matched].Goal.sum(), len(g[~g.matched])))
        right.append(_wilson(g[g.matched].Goal.sum(), len(g[g.matched])))
        foll.append(g.matched.mean())
    def unpack(s): return [v[0] for v in s], [v[0]-v[1] for v in s], [v[2]-v[0] for v in s]
    wp, wlo, whi = unpack(wrong); rp, rlo, rhi = unpack(right)
    ax.bar(x - w/2, wp, w, yerr=[wlo, whi], capsize=4, color=TEAL, edgecolor="k",
           linewidth=0.5, label="keeper guesses wrong (dives away)")
    ax.bar(x + w/2, rp, w, yerr=[rlo, rhi], capsize=4, color=GREY, edgecolor="k",
           linewidth=0.5, label="keeper guesses right (on your side)")
    for xi, v in zip(x, wp):
        ax.text(xi - w/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x, rp):
        ax.text(xi + w/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set(xticks=x, ylim=(0, 1.12), ylabel="P(goal)")
    ax.set_xticklabels([f"{l}\n(keeper here {f:.0%})" for l, f in zip(LABS, foll)])
    ax.set_title("P(goal) by placement, split by whether\nthe keeper guessed the side")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIGS / "fig_middle.png"); plt.close(fig)


if __name__ == "__main__":
    fig_calibration(); print("fig_calibration")
    fig_shrinkage(); print("fig_shrinkage")
    fig_keeper(); print("fig_keeper")
    fig_bridge(); print("fig_bridge")
    fig_execution(); print("fig_execution")
    fig_middle(); print("fig_middle")
    print("saved ->", FIGS)
