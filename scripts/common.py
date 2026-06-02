"""Shared paths, direction coding, and metrics for the penalty study.

All scripts resolve data/ and outputs/ relative to the repo root so they run
from any working directory (Windows or POSIX).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# StatsBomb goal frame: posts at y=36 and y=44, centre line y=40, goal width 8 yd.
GOAL_CENTER = 40.0
CATS = ["natural", "center", "nonnatural"]   # Project-1 outcome categories (ref = natural)
LCR = ["L", "C", "R"]


def code_direction(end_y, half_width):
    """Recode a StatsBomb end_y into GK-perspective L/C/R for a given centre
    HALF-width (yards). Convention baked into the corpus: high end_y = GK Left,
    low end_y = GK Right. half_width in {1.0, 1.5, 2.0} -> centre band width 2*hw.
    """
    end_y = np.asarray(end_y, dtype=float)
    out = np.full(end_y.shape, "C", dtype=object)
    out[end_y > GOAL_CENTER + half_width] = "L"
    out[end_y < GOAL_CENTER - half_width] = "R"
    return out


def load_men(half_width=1.5):
    """Load the StatsBomb men's corpus, recoding direction at the given centre
    half-width and deriving footedness + natural-side relative coding.
    Set env CORPUS_FILE to switch corpus (e.g. men_penalties_full_v2.csv)."""
    import os
    fname = os.environ.get("CORPUS_FILE", "men_penalties_full.csv")
    df = pd.read_csv(DATA / fname)
    df = df.dropna(subset=["end_y"]).copy()
    df["direction"] = code_direction(df["end_y"], half_width)

    # modal kicking foot per player = footedness proxy
    foot_mode = (df.dropna(subset=["body_part"])
                   .groupby("kicker")["body_part"]
                   .agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan))
    df["foot_dom"] = df["kicker"].map(foot_mode)
    df = df[df["foot_dom"].isin(["Left Foot", "Right Foot"])].copy()

    # right-footers' natural/open side ~ GK's RIGHT; left-footers ~ GK's LEFT
    df["natural_side"] = np.where(df["foot_dom"] == "Right Foot", "R", "L")

    def rel(row):
        if row["direction"] == "C":
            return "center"
        return "natural" if row["direction"] == row["natural_side"] else "nonnatural"
    df["dir_rel"] = df.apply(rel, axis=1)

    df["is_shootout"] = df["is_shootout"].astype(bool).astype(int)
    df["kcode"] = df["kicker"].astype("category").cat.codes
    # season-ish ordering for temporal CV: use match_id as a monotone time proxy
    df["t_order"] = df["match_id"].rank(method="dense").astype(int)
    return df.reset_index(drop=True)


ZONE_LCR = {1: "L", 4: "L", 7: "L", 2: "C", 5: "C", 8: "C", 3: "R", 6: "R", 9: "R"}
ZONE_HEIGHT = {1: "T", 2: "T", 3: "T", 4: "M", 5: "M", 6: "M", 7: "B", 8: "B", 9: "B"}


def load_kaggle():
    """Load the Kaggle World Cup shootouts (held-out high-pressure set) with
    L/C/R kick placement and keeper dive direction."""
    df = pd.read_csv(DATA / "WorldCupShootouts.csv")
    df["Keeper"] = df["Keeper"].replace({"l": "L"})
    df["kick_LCR"] = df["Zone"].map(lambda z: ZONE_LCR.get(int(z)) if pd.notna(z) else None)
    df["kick_height"] = df["Zone"].map(lambda z: ZONE_HEIGHT.get(int(z)) if pd.notna(z) else None)
    return df.dropna(subset=["kick_LCR", "Keeper", "Team"]).copy()


# ---------- metrics ----------
def logloss(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1)
    return -np.log(p).mean()


def per_sample_logloss(probs, y):
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))


def brier(probs, y):
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1
    return ((probs - oh) ** 2).sum(1).mean()


def accuracy(probs, y):
    return (probs.argmax(1) == y).mean()


def top2_accuracy(probs, y):
    top2 = np.argsort(probs, axis=1)[:, -2:]
    return np.mean([y[i] in top2[i] for i in range(len(y))])


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ece(probs, y, n_bins=10):
    """Expected Calibration Error (multiclass, top-label)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def mcnemar_exact(correct_a, correct_b):
    """Exact-binomial McNemar for paired classifier comparison.
    correct_a/b: boolean arrays of per-sample correctness. Returns (b, c, p)
    where b = A right & B wrong, c = A wrong & B right, two-sided exact p."""
    from scipy import stats
    a = np.asarray(correct_a, bool); b_ = np.asarray(correct_b, bool)
    n01 = int((a & ~b_).sum())   # A right, B wrong
    n10 = int((~a & b_).sum())   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = stats.binomtest(n01, n, 0.5, alternative="two-sided").pvalue
    return n01, n10, p
