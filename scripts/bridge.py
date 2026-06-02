"""The bridge - within-shootout exploitation engine + decision-theoretic layer.

Everything is expressed in {natural, center, nonnatural} relative to the kicker's
foot, which makes it FRAME-INVARIANT and dissolves the StatsBomb/Kaggle mirrored
L/R convention (StatsBomb right-footers favour GK-'R'; Kaggle right-footers favour
GK-'L' - the two corpora are mirror images, so a raw L/R join would be wrong).

Kaggle's natural side (verified empirically): right-footers -> 'L', left-footers -> 'R'.

OFFENCE (our kicker vs their keeper): given the keeper's dive distribution, pick the
  kick direction maximising P(goal). H8: does a keeper-aware policy beat "always
  shoot natural side"?  All within-Kaggle, self-consistent.

DEFENCE (our keeper vs their kicker): given the opposing kicker's calibrated P(L/C/R)
  from Project 1, pick the dive maximising P(save). Illustrative cross-dataset demo.

Inference is cluster-bootstrap over shootouts (the unit of resampling) so CIs respect
the small sample. The exploitation gain is a model-based COUNTERFACTUAL, framed as such.
"""
import json
import warnings
import numpy as np
import pandas as pd

import common as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(7)
REL = ["natural", "center", "nonnatural"]


def add_rel(d):
    d = d.copy()
    # Kaggle frame mirrored vs StatsBomb: right-footers' natural side = GK 'L'
    d["natural_side"] = np.where(d.Foot == "R", "L", "R")

    def rel(side, nat):
        if side == "C":
            return "center"
        return "natural" if side == nat else "nonnatural"
    d["kick_rel"] = [rel(s, n) for s, n in zip(d.kick_LCR, d.natural_side)]
    d["dive_rel"] = [rel(s, n) for s, n in zip(d.Keeper, d.natural_side)]
    return d


def payoff_table(d):
    """P(goal | kick_rel, dive_rel) as a 3x3 dict; falls back to row/global mean
    for thin/empty cells so policies are always defined."""
    glob = d.Goal.mean()
    tab = {}
    for kr in REL:
        sub_k = d[d.kick_rel == kr]
        row_mean = sub_k.Goal.mean() if len(sub_k) else glob
        for dr in REL:
            cell = sub_k[sub_k.dive_rel == dr]
            tab[(kr, dr)] = cell.Goal.mean() if len(cell) >= 1 else row_mean
    return tab, glob


def dive_dist(d):
    return d.dive_rel.value_counts(normalize=True).reindex(REL).fillna(0).values


def kick_dist(d):
    return d.kick_rel.value_counts(normalize=True).reindex(REL).fillna(0).values


def ev_kick(kr, tab, ddist):
    """Expected score of kicking direction kr given keeper dive distribution."""
    return sum(ddist[j] * tab[(kr, dr)] for j, dr in enumerate(REL))


def ev_dive(dr, tab, kdist):
    """Expected SAVE prob of diving dr given kicker placement distribution."""
    return sum(kdist[i] * (1 - tab[(kr, dr)]) for i, kr in enumerate(REL))


def offence(d):
    """H8: keeper-aware kick policy vs always-natural, with cluster bootstrap."""
    tab, _ = payoff_table(d)
    ddist = dive_dist(d)
    evs = {kr: ev_kick(kr, tab, ddist) for kr in REL}
    optimal = max(evs, key=evs.get)

    # realized score rate by kick direction (what kickers actually got)
    realized = d.groupby("kick_rel").Goal.agg(["mean", "count"]).reindex(REL)

    # cluster bootstrap over shootouts for the model-based EV gains
    games = d.Game_id.unique()
    gain_opt, gain_nonnat, ev_nat_b, ev_opt_b = [], [], [], []
    real_diff = []  # realized nonnatural - natural
    for _ in range(4000):
        samp = RNG.choice(games, size=len(games), replace=True)
        bd = pd.concat([d[d.Game_id == g] for g in samp], ignore_index=True)
        tb, _ = payoff_table(bd); dd = dive_dist(bd)
        e = {kr: ev_kick(kr, tb, dd) for kr in REL}
        opt = max(e, key=e.get)
        gain_opt.append(e[opt] - e["natural"])
        gain_nonnat.append(e["nonnatural"] - e["natural"])
        ev_nat_b.append(e["natural"]); ev_opt_b.append(e[opt])
        rn = bd[bd.kick_rel == "nonnatural"].Goal.mean()
        rnat = bd[bd.kick_rel == "natural"].Goal.mean()
        real_diff.append(rn - rnat)

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]

    return {
        "keeper_dive_dist": dict(zip(REL, np.round(ddist, 3).tolist())),
        "payoff_P_goal": {f"kick_{kr}": {f"dive_{dr}": round(tab[(kr, dr)], 3) for dr in REL} for kr in REL},
        "EV_score_by_kick": {k: round(v, 3) for k, v in evs.items()},
        "optimal_kick": optimal,
        "always_natural_EV": round(evs["natural"], 3),
        "model_gain_optimal_vs_natural": {"mean": round(float(np.mean(gain_opt)), 3), "ci95": ci(gain_opt)},
        "model_gain_nonnatural_vs_natural": {"mean": round(float(np.mean(gain_nonnat)), 3), "ci95": ci(gain_nonnat)},
        "realized_score_rate_by_kick": {kr: {"rate": round(realized.loc[kr, "mean"], 3),
                                             "n": int(realized.loc[kr, "count"])} for kr in REL},
        "realized_nonnatural_minus_natural": {"mean": round(float(np.mean(real_diff)), 3), "ci95": ci(real_diff)},
    }


def defence(d, project1_kicker_dists):
    """Our keeper vs an opposing kicker whose P(natural/center/nonnatural) comes
    from Project 1. Recommend dive by expected save value. Cross-dataset demo."""
    tab, _ = payoff_table(d)
    out = {}
    for name, kdist in project1_kicker_dists.items():
        kdist = np.asarray(kdist, dtype=float); kdist = kdist / kdist.sum()
        evs = {dr: ev_dive(dr, tab, kdist) for dr in REL}
        best = max(evs, key=evs.get)
        # baselines: always dive natural (scout default), stay center, random
        out[name] = {
            "kicker_dist_natural_center_nonnatural": np.round(kdist, 3).tolist(),
            "EV_save_by_dive": {k: round(v, 3) for k, v in evs.items()},
            "recommended_dive": best,
            "gain_vs_always_natural_pp": round(100 * (evs[best] - evs["natural"]), 1),
            "gain_vs_random_pp": round(100 * (evs[best] - np.mean(list(evs.values()))), 1),
        }
    return out


def main():
    d = add_rel(C.load_kaggle())
    res = {"n_kicks": int(len(d)), "n_shootouts": int(d.Game_id.nunique()),
           "note": "natural/center/nonnatural relative to kicker foot; frame-invariant."}

    res["OFFENCE_H8"] = offence(d)

    # kicker distributions for the defence demo:
    #  - population (Project 1 corpus marginal at hw 1.5)
    #  - a strong natural-side specialist (illustrative)
    men = C.load_men(1.5)
    pop = men.dir_rel.value_counts(normalize=True).reindex(REL).fillna(0).values
    dists = {"population_corpus": pop.tolist(),
             "natural_specialist_70pct": [0.70, 0.10, 0.20]}
    res["DEFENCE_demo"] = defence(d, dists)

    with open(C.OUT / "bridge_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("\nsaved outputs/bridge_results.json")
    return res


if __name__ == "__main__":
    main()
