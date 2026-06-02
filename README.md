# Where Will the Penalty Go?

A Bayesian study of penalty kicker placement and goalkeeper dives on public data.

**Read the report: [report.pdf](report.pdf)** (GitHub renders it in the browser).

## What is here

- `report.typ`, `report.pdf` — the report (source and rendered PDF).
- `scripts/` — the code that produces every number in the report.
- `data/` — the input corpora (StatsBomb penalties and the Kaggle shootouts).
- `outputs/` — the results and figures the code generates.

## Setup

    python -m venv venv && venv\Scripts\activate   # POSIX: source venv/bin/activate
    pip install -r requirements.txt

No C or C++ compiler is needed: sampling uses **nutpie** (numba/LLVM), which compiles the model
without g++ or MSVC and avoids the Windows multiprocessing hang.

## Reproduce the numbers and figures

    python scripts/prep_features.py        # EDA sanity checks
    python scripts/fit_project1.py         # hierarchical kicker model + rigor layer
    python scripts/project2_analysis.py    # goalkeeper action bias, gambler's fallacy, entropy
    python scripts/bridge.py               # exploitation engine + decision layer
    python scripts/player_tendencies.py    # per-player shrunken tendencies
    python scripts/make_report_figs.py     # rebuild the report figures
    typst compile report.typ               # rebuild report.pdf

`scripts/common.py` holds shared data loading, direction coding, and metrics. Set the env var
`CORPUS_FILE` to choose which kicker corpus the scripts load (default `men_penalties_full.csv`;
the report uses `men_penalties_full_v2.csv`). Rebuild or expand the StatsBomb corpus with
`python scripts/pull_statsbomb.py loop` then `python scripts/pull_statsbomb.py merge`.

## Headline results

- **Kickers (StatsBomb, 1,032 penalties):** footedness dominates; top-1 accuracy is capped near
  50%. The model's value is calibration and sample-size-aware shrinkage, not raw accuracy.
- **Keepers (Kaggle, 320 shootout kicks):** they commit to a side 88% of the time and lean to the
  kicker's natural side, but show no usable sequential pattern (four null tests).
- **Bridge:** shooting to the non-natural corner beats always going to the obvious side by about
  8 goals per 100. Suggestive on 35 shootouts, not yet significant.

Note: StatsBomb does not log penalty dive direction (the keeper is always logged on his line), so
the goalkeeper analyses use the hand-coded Kaggle set. The two corpora label left and right from
opposite views, so everything is recoded as natural, centre, and non-natural relative to the
kicker's foot.
