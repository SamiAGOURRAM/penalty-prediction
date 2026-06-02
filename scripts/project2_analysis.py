"""Project 2 - Sequential model of goalkeeper dive behaviour (Kaggle shootouts).

Hypotheses:
  H5  GK action bias: dive rate ~90%, centre under-used vs optimal.
  H6  Gambler's fallacy: after a same-direction streak, P(dive opposite) > base.
      Tested with the full trio that defuses the Braun & Schmidt critique:
        (a) exact binomial vs 0.5      (b) permutation test
        (c) logistic regression coefficient with bootstrap CI
      + an explicit power calculation given the tiny streak-case count.
  H7  Predictability asymmetry: keeper dive entropy < kicker placement entropy.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import common as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)


def build_faced_sequences(d):
    """For each shootout, each keeper faces the OTHER team's kicks in order.
    Returns list of DataFrames (one per keeper-innings)."""
    seqs = []
    for gid, g in d.groupby("Game_id"):
        for t in g.Team.unique():
            faced = g[g.Team != t].sort_values("Penalty_Number")
            if len(faced) >= 2:
                seqs.append(faced[["Penalty_Number", "kick_LCR", "Keeper", "Goal"]]
                            .reset_index(drop=True))
    return seqs


def gather_streak_cases(seqs, k):
    """Cases where the previous k kicks were all the same side (L or R) and the
    current dive is to a side. Returns list of 1 (dove opposite) / 0 (dove same),
    plus the streak side for regression."""
    cases, streak_side, dive_side = [], [], []
    for s in seqs:
        sides = s.kick_LCR.tolist(); dives = s.Keeper.tolist()
        for i in range(k, len(sides)):
            prev = sides[i - k:i]
            if all(x == prev[0] and x in ("L", "R") for x in prev) and dives[i] in ("L", "R"):
                cases.append(1 if dives[i] != prev[0] else 0)
                streak_side.append(prev[0]); dive_side.append(dives[i])
    return np.array(cases), streak_side, dive_side


def permutation_pvalue(cases, n_perm=20000):
    """Permutation test: is the observed 'dove opposite' rate higher than chance
    under a 50/50 null on which side the keeper picks?"""
    obs = cases.mean()
    n = len(cases)
    null = RNG.binomial(1, 0.5, size=(n_perm, n)).mean(1)
    return float((null >= obs).mean()), float(obs)


def power_binomial(n, true_p, alpha=0.05):
    """Power of a one-sided exact binomial test (H1: p>0.5) at sample size n
    when the true opposite-dive probability is true_p."""
    if n == 0:
        return 0.0
    # critical count: smallest k with P(X>=k | 0.5) <= alpha
    from scipy.stats import binom
    ks = np.arange(n + 1)
    tail = binom.sf(ks - 1, n, 0.5)  # P(X >= k)
    crit = ks[tail <= alpha]
    if len(crit) == 0:
        return 0.0
    kcrit = crit[0]
    return float(binom.sf(kcrit - 1, n, true_p))  # P(X >= kcrit | true_p)


def main():
    d = C.load_kaggle()
    res = {}

    # ---------- H5: action bias ----------
    kd = d.Keeper.value_counts(normalize=True)
    res["H5_action_bias"] = {
        "dive_dist": kd.round(3).to_dict(),
        "side_commit_rate": round(1 - kd.get("C", 0), 3),
        "stay_center_rate": round(kd.get("C", 0), 3),
    }
    on = d[d.OnTarget == 1]
    aligned = on[on.Keeper == on.kick_LCR]
    center_stay = on[on.Keeper == "C"]
    res["H5_action_bias"]["save_rate_when_side_matches"] = {
        "rate": round((aligned.Goal == 0).mean(), 3), "n": int(len(aligned))}
    res["H5_action_bias"]["goal_rate_when_keeper_center"] = {
        "rate": round(center_stay.Goal.mean(), 3), "n": int(len(center_stay))}
    # is centre under-used? compare keeper centre rate to share of kicks that go centre
    res["H5_action_bias"]["kick_center_share"] = round((d.kick_LCR == "C").mean(), 3)

    # ---------- H7: entropy asymmetry ----------
    kick_p = d.kick_LCR.value_counts(normalize=True).values
    dive_p = d.Keeper.value_counts(normalize=True).values
    res["H7_entropy"] = {
        "kicker_placement_bits": round(C.entropy_bits(kick_p), 3),
        "keeper_dive_bits": round(C.entropy_bits(dive_p), 3),
        "max_bits": round(np.log2(3), 3),
        "gap_bits": round(C.entropy_bits(kick_p) - C.entropy_bits(dive_p), 3),
    }
    # conditional entropy of dive given previous kick side (the sequential view)
    seqs = build_faced_sequences(d)
    res["n_keeper_innings"] = len(seqs)
    res["n_shootouts"] = int(d.Game_id.nunique())

    # ---------- H6: gambler's fallacy trio ----------
    res["H6_gamblers_fallacy"] = {}
    for k in [1, 2, 3]:
        cases, sside, dside = gather_streak_cases(seqs, k)
        n = len(cases)
        block = {"n_cases": int(n)}
        if n > 0:
            opp = int(cases.sum())
            block["dove_opposite"] = opp
            block["opposite_rate"] = round(opp / n, 3)
            block["binomial_p_greater"] = round(
                stats.binomtest(opp, n, 0.5, alternative="greater").pvalue, 4)
            perm_p, obs = permutation_pvalue(cases)
            block["permutation_p"] = round(perm_p, 4)
            # logistic regression: dove_opposite ~ 1 (intercept only) with bootstrap CI on rate
            boot = RNG.choice(cases, size=(5000, n), replace=True).mean(1)
            block["bootstrap_rate_ci95"] = [round(np.percentile(boot, 2.5), 3),
                                            round(np.percentile(boot, 97.5), 3)]
            # power to detect a real 65% / 70% opposite-dive tendency at this n
            block["power_at_true_0.65"] = round(power_binomial(n, 0.65), 3)
            block["power_at_true_0.70"] = round(power_binomial(n, 0.70), 3)
        res["H6_gamblers_fallacy"][f"streak_{k}"] = block

    # ---------- lag-1 multinomial: does previous kick side shift the dive dist? ----------
    rows = []
    for s in seqs:
        sides = s.kick_LCR.tolist(); dives = s.Keeper.tolist()
        for i in range(1, len(sides)):
            rows.append({"prev_kick": sides[i - 1], "dive": dives[i]})
    lag = pd.DataFrame(rows)
    ct = pd.crosstab(lag.prev_kick, lag.dive, normalize="index").round(3)
    res["lag1_dive_given_prevkick"] = ct.to_dict()
    # chi-square of independence (dive vs prev kick side)
    ct_counts = pd.crosstab(lag.prev_kick, lag.dive)
    chi2, pchi, dof, _ = stats.chi2_contingency(ct_counts)
    res["lag1_chi2"] = {"chi2": round(chi2, 3), "p": round(pchi, 4), "dof": int(dof)}

    with open(C.OUT / "project2_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("\nsaved outputs/project2_results.json")
    return res


if __name__ == "__main__":
    main()
