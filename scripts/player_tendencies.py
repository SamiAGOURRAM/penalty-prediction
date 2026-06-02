"""Per-player SHRUNKEN direction tendencies from the hierarchical model.

Fits the Project-1 model on the FULL corpus (no holdout - we want the best
evidence-weighted estimate of every player's tendency), then for each player
extracts:
  * raw empirical P(L/C/R)                          (what a per-player table sees)
  * shrunken model P(L/C/R)                          (evidence-weighted estimate)
  * the personal lean u_nonnatural (log-odds of NON-natural vs natural, beyond
    footedness) and the posterior probability that the player favours their
    natural side MORE than the population average  (P(u_nonnatural < 0)).

Ranks players by the strength of their *established* natural-side bias and writes
outputs/player_tendencies.csv + a raw-vs-shrunk scatter (shrinkage in action).
"""
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

warnings.filterwarnings("ignore")
CATS = C.CATS  # ['natural','center','nonnatural']
K = len(CATS)


def fit_full(d):
    y = d["y"].values
    sh = d["is_shootout"].values.astype(float)
    kc = d["kcode"].values
    nK = int(d["kcode"].max()) + 1
    with pm.Model():
        a = pm.Normal("a", 0, 1.5, shape=K - 1)
        b_sh = pm.Normal("b_sh", 0, 1.0, shape=K - 1)
        sigma = pm.HalfNormal("sigma", 1.0, shape=K - 1)
        z = pm.Normal("z", 0, 1, shape=(nK, K - 1))
        u = pm.Deterministic("u", z * sigma)
        eta1 = a + b_sh * sh[:, None] + u[kc]
        eta = pm.math.concatenate([pm.math.zeros_like(eta1[:, :1]), eta1], axis=1)
        p = pm.math.softmax(eta, axis=1)
        pm.Categorical("obs", p=p, observed=y)
        idata = pm.sample(2000, tune=2000, chains=4, cores=1, target_accept=0.9,
                          random_seed=42, progressbar=False, nuts_sampler="nutpie")
    return idata


def to_LCR(prob_rel, foot):
    """Map P(natural,center,nonnatural) -> absolute P(L,C,R) given dominant foot.
    Left-footer natural=L; right-footer natural=R."""
    pn, pc, pnn = prob_rel
    if foot == "Left Foot":     # natural = L, nonnatural = R
        return {"L": pn, "C": pc, "R": pnn}
    return {"L": pnn, "C": pc, "R": pn}   # right: natural = R


def main(min_n=8):
    d = C.load_men(1.5)
    d["y"] = d["dir_rel"].map({c: i for i, c in enumerate(CATS)})
    idata = fit_full(d)
    post = idata.posterior
    a_m = post["a"].mean(("chain", "draw")).values        # [center, nonnatural] intercepts
    u_vals = post["u"].values                              # (chain, draw, player, 2)
    u_m = u_vals.mean(axis=(0, 1))                          # (player, 2)
    # posterior prob player favours natural MORE than average: u_nonnatural < 0
    p_natural_lean = (u_vals[..., 1] < 0).mean(axis=(0, 1))  # (player,)

    pop = d["direction"].value_counts(normalize=True).reindex(["L", "C", "R"]).values

    rows = []
    for k, g in d.groupby("kcode"):
        n = len(g)
        if n < min_n:
            continue
        name = g["kicker"].iloc[0]
        foot = g["foot_dom"].iloc[0]
        raw = g["direction"].value_counts(normalize=True).reindex(["L", "C", "R"]).fillna(0).values
        # shrunken estimate at neutral (in-play) context
        eta1 = a_m + u_m[int(k)]
        eta = np.concatenate([[0.0], eta1])
        e = np.exp(eta - eta.max()); prob_rel = e / e.sum()      # [nat,cen,nonnat]
        lcr = to_LCR(prob_rel, foot)
        rows.append({
            "player": name.split()[0] + " " + name.split()[-1],
            "foot": foot.split()[0], "n": n,
            "raw_L": round(raw[0], 2), "raw_C": round(raw[1], 2), "raw_R": round(raw[2], 2),
            "shrunk_L": round(lcr["L"], 2), "shrunk_C": round(lcr["C"], 2), "shrunk_R": round(lcr["R"], 2),
            "P_natural": round(prob_rel[0], 3),
            "u_nonnat": round(float(u_m[int(k), 1]), 3),
            "P(favours_natural>avg)": round(float(p_natural_lean[int(k)]), 3),
        })
    out = pd.DataFrame(rows)
    # established natural-side bias: high P_natural AND high posterior confidence
    out["established_bias"] = out["P_natural"] * out["P(favours_natural>avg)"]
    out = out.sort_values("P_natural", ascending=False).reset_index(drop=True)
    out.to_csv(C.OUT / "player_tendencies.csv", index=False)

    print(f"population L/C/R = {pop[0]:.2f}/{pop[1]:.2f}/{pop[2]:.2f}\n")
    print(out.to_string(index=False))

    _shrinkage_plot(out, pop)
    return out


def _shrinkage_plot(out, pop):
    """Raw vs shrunken natural-side share; arrows show how far each player is
    pulled toward the population by partial pooling (longer for low-n)."""
    # natural-side raw share depends on foot: raw_natural = raw_L (left) or raw_R (right)
    raw_nat = np.where(out["foot"] == "Left", out["raw_L"], out["raw_R"])
    shrunk_nat = out["P_natural"].values
    n = out["n"].values
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sizes = 30 + 4 * n
    sc = ax.scatter(raw_nat, shrunk_nat, s=sizes, c=n, cmap="viridis",
                    edgecolor="k", linewidth=0.4, zorder=3)
    for x, y, nm, nn in zip(raw_nat, shrunk_nat, out["player"], n):
        if nn >= 12 or abs(x - y) > 0.12:
            ax.annotate(nm.split()[-1], (x, y), fontsize=6.5,
                        textcoords="offset points", xytext=(4, 3))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no shrinkage (raw = shrunk)")
    pop_nat = None  # population natural share differs by foot; draw a band instead
    ax.axhspan(0.45, 0.55, color="orange", alpha=0.12, zorder=0,
               label="≈ population natural share")
    ax.set_xlabel("raw empirical natural-side share")
    ax.set_ylabel("shrunken model estimate of natural-side prob")
    ax.set_title("Partial pooling pulls noisy (low-n) leans toward the population;\n"
                 "well-sampled players keep theirs")
    cb = fig.colorbar(sc); cb.set_label("# penalties (n)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0.2, 0.8)
    fig.tight_layout(); fig.savefig(C.OUT / "player_shrinkage.png", dpi=140)
    plt.close(fig)
    print("\nsaved outputs/player_tendencies.csv + outputs/player_shrinkage.png")


if __name__ == "__main__":
    main()
