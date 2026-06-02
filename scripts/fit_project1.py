"""Project 1 - Bayesian hierarchical multinomial-logit for penalty direction.

Outcome dir_rel in {natural, center, nonnatural} (reference = natural).
Fixed effects: intercept + is_shootout. Random effect: shooter intercept
(partial pooling). Rigor layer:
  * full sampling (4 chains x 2000 draws) for the primary centre width
  * baselines: footedness marginal (= population Nash mixing), per-player
    frequency table, and a multinomial logit WITHOUT random effects
  * metrics: log-loss, Brier, accuracy, top-2, ECE (+ reliability diagram)
  * low-n vs high-n kicker split (the shrinkage story, H1)
  * exact-binomial McNemar for paired classifier comparison
  * forward-chaining temporal CV (scout-at-t, predict t+1)
  * centre-width sensitivity over half-widths {1.0, 1.5, 2.0}
Tests H1-H4 from the study plan.
"""
import json
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
RNG = np.random.default_rng(42)
CATS = C.CATS
K = len(CATS)


# ---------------------------------------------------------------- model
def fit_hier(train, n_players, draws=2000, tune=2000, chains=4, seed=42):
    """Fit the hierarchical multinomial logit. RE dimension spans ALL players
    in the corpus (n_players) so unseen-in-train players fall back to the
    population (u ~ prior, posterior mean ~ 0). Returns idata."""
    yt = train["y"].values
    sh_t = train["is_shootout"].values.astype(float)
    kc_t = train["kcode"].values
    with pm.Model():
        a = pm.Normal("a", 0, 1.5, shape=K - 1)
        b_sh = pm.Normal("b_sh", 0, 1.0, shape=K - 1)
        sigma = pm.HalfNormal("sigma", 1.0, shape=K - 1)
        z = pm.Normal("z", 0, 1, shape=(n_players, K - 1))
        u = pm.Deterministic("u", z * sigma)
        eta1 = a + b_sh * sh_t[:, None] + u[kc_t]
        eta = pm.math.concatenate([pm.math.zeros_like(eta1[:, :1]), eta1], axis=1)
        p = pm.math.softmax(eta, axis=1)
        pm.Categorical("obs", p=p, observed=yt)
        # nutpie (numba backend) JIT-compiles the logp -> no g++/C-compiler needed
        # on Windows, and runs all chains without the multiprocessing-spawn hang.
        idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                          target_accept=0.9, random_seed=seed, progressbar=False,
                          nuts_sampler="nutpie")
    return idata


def hier_probs(idata, rows):
    """Posterior-mean predicted probabilities for given rows."""
    post = idata.posterior
    a_m = post["a"].mean(("chain", "draw")).values
    bsh_m = post["b_sh"].mean(("chain", "draw")).values
    u_m = post["u"].mean(("chain", "draw")).values  # (n_players, K-1)
    out = np.zeros((len(rows), K))
    for j, (_, row) in enumerate(rows.iterrows()):
        eta1 = a_m + bsh_m * row["is_shootout"] + u_m[int(row["kcode"])]
        eta = np.concatenate([[0.0], eta1])
        e = np.exp(eta - eta.max())
        out[j] = e / e.sum()
    return out


# ---------------------------------------------------------------- baselines
def base_marginal(train, test):
    p = train["dir_rel"].value_counts(normalize=True).reindex(CATS).fillna(1e-6).values
    p = p / p.sum()
    return np.tile(p, (len(test), 1))


def base_freq(train, test, prior_strength=2.0):
    """Per-player frequency table, Laplace-shrunk toward the train marginal."""
    base = train["dir_rel"].value_counts(normalize=True).reindex(CATS).fillna(1e-6).values
    base = base / base.sum()
    counts = {k: g["dir_rel"].value_counts().reindex(CATS).fillna(0).values
              for k, g in train.groupby("kcode")}
    out = np.zeros((len(test), K))
    for j, (_, r) in enumerate(test.iterrows()):
        c = counts.get(int(r["kcode"]))
        if c is None:
            out[j] = base
        else:
            sm = c + base * prior_strength
            out[j] = sm / sm.sum()
    return out


def base_logit_noRE(train, test):
    """Multinomial logit with intercept + is_shootout only (no player effects)."""
    from sklearn.linear_model import LogisticRegression
    Xtr = train[["is_shootout"]].values
    ytr = train["y"].values
    # guard: need >=2 classes present
    clf = LogisticRegression(max_iter=1000, C=1e6)
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(test[["is_shootout"]].values)
    # align columns to CATS order (clf.classes_ are the y integer codes)
    out = np.zeros((len(test), K))
    for ci, cls in enumerate(clf.classes_):
        out[:, int(cls)] = P[:, ci]
    return np.clip(out, 1e-9, 1)


# ---------------------------------------------------------------- evaluation
def metrics_block(P, y):
    return dict(logloss=round(C.logloss(P, y), 4), brier=round(C.brier(P, y), 4),
                acc=round(C.accuracy(P, y), 4), top2=round(C.top2_accuracy(P, y), 4),
                ece=round(C.ece(P, y), 4))


def run_width(hw, draws, tune, chains, primary=False):
    d = C.load_men(hw)
    d["y"] = d["dir_rel"].map({c: i for i, c in enumerate(CATS)})
    n_players = int(d["kcode"].max()) + 1

    # stratified 25% holdout
    d = d.sample(frac=1, random_state=42).reset_index(drop=True)
    test = d.groupby("dir_rel", group_keys=False).sample(frac=0.25, random_state=42)
    train = d.drop(test.index)
    yte = test["y"].values

    idata = fit_hier(train, n_players, draws=draws, tune=tune, chains=chains)
    P_hier = hier_probs(idata, test)
    P_marg = base_marginal(train, test)
    P_freq = base_freq(train, test)
    P_noRE = base_logit_noRE(train, test)

    res = {"half_width": hw, "n": len(d), "kickers": int(d["kcode"].nunique()),
           "dir_dist": d["direction"].value_counts(normalize=True).round(3).to_dict(),
           "rel_dist": d["dir_rel"].value_counts(normalize=True).round(3).to_dict()}
    res["full_test"] = {
        "footedness_marginal": metrics_block(P_marg, yte),
        "per_player_freq": metrics_block(P_freq, yte),
        "logit_noRE": metrics_block(P_noRE, yte),
        "hierarchical": metrics_block(P_hier, yte),
    }

    # low-n vs high-n
    tn = train["kcode"].value_counts()
    test = test.copy()
    test["ntrain"] = test["kcode"].map(tn).fillna(0)
    low = (test["ntrain"] < 5).values
    hi = ~low
    res["low_n"] = {"n_kicks": int(low.sum()),
                    "per_player": round(C.logloss(P_freq[low], yte[low]), 4),
                    "hierarchical": round(C.logloss(P_hier[low], yte[low]), 4)}
    res["high_n"] = {"n_kicks": int(hi.sum()),
                     "per_player": round(C.logloss(P_freq[hi], yte[hi]), 4),
                     "hierarchical": round(C.logloss(P_hier[hi], yte[hi]), 4)}

    # McNemar exact (paired correctness): hier vs each baseline
    corr_h = (P_hier.argmax(1) == yte)
    res["mcnemar"] = {}
    for name, P in [("vs_footedness", P_marg), ("vs_per_player", P_freq), ("vs_logit_noRE", P_noRE)]:
        b, c, p = C.mcnemar_exact(corr_h, (P.argmax(1) == yte))
        res["mcnemar"][name] = {"hier_only_right": b, "base_only_right": c, "p_exact": round(p, 4)}

    # posteriors
    post = idata.posterior
    res["sigma_shooter_RE"] = np.round(post["sigma"].mean(("chain", "draw")).values, 3).tolist()
    sig_hdi = az.hdi(idata, var_names=["sigma"])["sigma"].values
    res["sigma_hdi94"] = np.round(sig_hdi, 3).tolist()
    res["b_shootout"] = np.round(post["b_sh"].mean(("chain", "draw")).values, 3).tolist()
    bsh_hdi = az.hdi(idata, var_names=["b_sh"])["b_sh"].values
    res["b_shootout_hdi94"] = np.round(bsh_hdi, 3).tolist()
    rh = az.rhat(idata)
    res["max_rhat"] = round(max(float(np.nanmax(rh[v].values)) for v in rh.data_vars), 4)
    ess = az.ess(idata)
    res["min_ess_bulk"] = round(min(float(np.nanmin(ess[v].values)) for v in ess.data_vars), 1)

    if primary:
        # slim posterior summary only (full netcdf is ~240 MB due to 484 player REs)
        az.summary(idata, var_names=["a", "b_sh", "sigma"], hdi_prob=0.94) \
          .to_csv(C.OUT / "project1_posterior_summary.csv")
        _reliability_plot(P_hier, P_marg, yte, hw)
        # save preds for downstream
        np.save(C.OUT / "p1_test_probs_hier.npy", P_hier)
        np.save(C.OUT / "p1_test_y.npy", yte)
    return res


def _reliability_plot(P_hier, P_marg, y, hw):
    fig, ax = plt.subplots(figsize=(5, 5))
    for P, lab, mk in [(P_hier, f"hierarchical (ECE={C.ece(P_hier,y):.3f})", "o"),
                       (P_marg, f"footedness (ECE={C.ece(P_marg,y):.3f})", "s")]:
        conf = P.max(1); pred = P.argmax(1); correct = (pred == y).astype(float)
        bins = np.linspace(0, 1, 11); xs = []; ys = []
        for i in range(10):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.sum() >= 3:
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
        ax.plot(xs, ys, mk + "-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Project 1 reliability (centre half-width {hw})")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(C.OUT / "p1_reliability.png", dpi=130)
    plt.close(fig)


def temporal_cv(hw=1.5, n_folds=4, draws=1000, tune=1000, chains=4):
    """Forward-chaining: expanding train window on match time order, predict the
    next block. Compares hierarchical vs footedness vs per-player on log-loss."""
    d = C.load_men(hw)
    d["y"] = d["dir_rel"].map({c: i for i, c in enumerate(CATS)})
    n_players = int(d["kcode"].max()) + 1
    d = d.sort_values("t_order").reset_index(drop=True)
    cuts = np.linspace(0.5, 0.9, n_folds + 1)
    folds = []
    for i in range(n_folds):
        tr_end = int(len(d) * cuts[i])
        te_end = int(len(d) * cuts[i + 1])
        train = d.iloc[:tr_end]; test = d.iloc[tr_end:te_end]
        if len(test) < 10:
            continue
        idata = fit_hier(train, n_players, draws=draws, tune=tune, chains=chains)
        yte = test["y"].values
        Ph = hier_probs(idata, test)
        Pm = base_marginal(train, test)
        Pf = base_freq(train, test)
        folds.append({"fold": i + 1, "n_train": len(train), "n_test": len(test),
                      "footedness": round(C.logloss(Pm, yte), 4),
                      "per_player": round(C.logloss(Pf, yte), 4),
                      "hierarchical": round(C.logloss(Ph, yte), 4),
                      "hier_acc": round(C.accuracy(Ph, yte), 4)})
    agg = {k: round(float(np.mean([f[k] for f in folds])), 4)
           for k in ["footedness", "per_player", "hierarchical", "hier_acc"]}
    return {"folds": folds, "mean": agg}


if __name__ == "__main__":
    results = {}
    print("=== Project 1: primary fit (half-width 1.5, 4 chains x 2000 draws) ===")
    results["primary_hw1.5"] = run_width(1.5, draws=2000, tune=2000, chains=4, primary=True)
    print(json.dumps(results["primary_hw1.5"], indent=2))

    print("\n=== centre-width sensitivity (4 chains x 1000) ===")
    results["sensitivity"] = {}
    for hw in [1.0, 2.0]:
        results["sensitivity"][f"hw{hw}"] = run_width(hw, draws=1000, tune=1000, chains=4)
        print(f"hw={hw}:", json.dumps(results["sensitivity"][f"hw{hw}"]["full_test"], indent=2))
        print(f"   sigma={results['sensitivity'][f'hw{hw}']['sigma_shooter_RE']} "
              f"b_sh={results['sensitivity'][f'hw{hw}']['b_shootout']} "
              f"low_n hier/freq={results['sensitivity'][f'hw{hw}']['low_n']}")

    print("\n=== forward-chaining temporal CV (hw 1.5) ===")
    results["temporal_cv"] = temporal_cv(1.5)
    print(json.dumps(results["temporal_cv"], indent=2))

    with open(C.OUT / "project1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved outputs/project1_results.json + idata + reliability plot")
