"""Regenerate a 3-way reliability diagram (hierarchical vs footedness vs per-player)
from the saved primary-split predictions — no model refit needed. Reconstructs the
deterministic split (seed 42) to recompute the baseline probabilities."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
import fit_project1 as F

CATS = C.CATS


def main():
    d = C.load_men(1.5)
    d["y"] = d["dir_rel"].map({c: i for i, c in enumerate(CATS)})
    d = d.sample(frac=1, random_state=42).reset_index(drop=True)
    test = d.groupby("dir_rel", group_keys=False).sample(frac=0.25, random_state=42)
    train = d.drop(test.index)

    P_hier = np.load(C.OUT / "p1_test_probs_hier.npy")
    y = np.load(C.OUT / "p1_test_y.npy")
    P_marg = F.base_marginal(train, test)
    P_freq = F.base_freq(train, test)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    series = [(P_hier, "hierarchical", "o", "#1f77b4"),
              (P_freq, "per-player table", "^", "#d62728"),
              (P_marg, "footedness marginal", "s", "#ff7f0e")]
    for P, lab, mk, col in series:
        conf = P.max(1); pred = P.argmax(1); correct = (pred == y).astype(float)
        bins = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
        for i in range(10):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.sum() >= 3:
                xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
        ax.plot(xs, ys, mk + "-", color=col, label=f"{lab} (ECE={C.ece(P, y):.3f})")
        for x, yv, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, yv), fontsize=6, color=col,
                        textcoords="offset points", xytext=(3, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("predicted confidence (top class)")
    ax.set_ylabel("empirical accuracy in bin")
    ax.set_title("Project 1 reliability — calibration is the win\n(bin counts annotated)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0.25, 0.85); ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(C.OUT / "p1_reliability.png", dpi=140)
    print("saved outputs/p1_reliability.png")


if __name__ == "__main__":
    main()
