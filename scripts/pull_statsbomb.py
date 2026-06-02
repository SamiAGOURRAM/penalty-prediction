"""Resumable full StatsBomb pull of MEN'S penalties (Project 1 / kicker direction).

Caches one CSV per competition-season under data/cache/, skipping finished ones, so
it can be re-run until complete. Schema matches data/men_penalties_full.csv.

NOTE: StatsBomb does NOT log goalkeeper DIVE direction for penalties
(goalkeeper_end_location is NaN; the keeper location is always the set point [1,40]).
So this pull can only expand the KICKER model (Project 1). Project 2 / the keeper side
of the bridge remain Kaggle-only by necessity.

Usage:
    python scripts/pull_statsbomb.py            # pull/refresh all male comp-seasons
    python scripts/pull_statsbomb.py merge       # merge cache -> data/men_penalties_full_v2.csv
"""
import sys
import time
import warnings
import pandas as pd
import numpy as np
from statsbombpy import sb

import common as C

warnings.filterwarnings("ignore")
CACHE = C.DATA / "cache"
CACHE.mkdir(exist_ok=True)


def _retry(fn, tries=4, base=1.5):
    """Call fn() with retries+backoff. The sandbox network has intermittent DNS
    failures (getaddrinfo), so transient errors must not be treated as 'no data'.
    Raises the last exception if all tries fail."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base * (i + 1))
    raise last


def lr_c(y, hw=1.5):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return None
    if y < 40 - hw:
        return "R"
    if y > 40 + hw:
        return "L"
    return "C"


def pull_all():
    comps = sb.competitions()
    male = comps[comps.competition_gender == "male"].reset_index(drop=True)
    n_done = 0
    for _, c in male.iterrows():
        tag = CACHE / f"{c.competition_id}_{c.season_id}.csv"
        if tag.exists():
            continue
        try:
            ms = _retry(lambda: sb.matches(competition_id=c.competition_id, season_id=c.season_id))
        except Exception:
            print(f"SKIP (matches failed) {c.competition_name} {c.season_name}", flush=True)
            continue  # do NOT cache -> retried next run
        rows = []
        incomplete = False
        for _, m in ms.iterrows():
            try:
                ev = _retry(lambda mid=m.match_id: sb.events(match_id=mid))
            except Exception:
                incomplete = True  # a real fetch failure; don't trust this season's count
                break
            if "shot_type" not in ev.columns:
                continue
            pk = ev[(ev["type"] == "Shot") & (ev["shot_type"] == "Penalty")].copy()
            pk = pk.sort_values(["period", "minute", "second"], na_position="last")
            order = {}
            for _, s in pk.iterrows():
                el = s.get("shot_end_location")
                ey = el[1] if isinstance(el, (list, tuple)) and len(el) >= 2 else None
                is_so = (s.get("period") == 5)
                key = (m.match_id, s.get("team"))
                order[key] = order.get(key, 0) + 1
                rows.append(dict(
                    match_id=m.match_id, competition=c.competition_name, season=str(c.season_name),
                    period=s.get("period"), minute=s.get("minute"), is_shootout=is_so,
                    shootout_kick_order=order[key] if is_so else None, team=s.get("team"),
                    kicker=s.get("player"), kicker_id=s.get("player_id"),
                    body_part=s.get("shot_body_part"), outcome=s.get("shot_outcome"),
                    end_y=ey, direction=lr_c(ey)))
        if incomplete:
            print(f"SKIP (events failed) {c.competition_name} {c.season_name} -> will retry", flush=True)
            continue  # do NOT cache an incomplete season
        pd.DataFrame(rows).to_csv(tag, index=False, encoding="utf-8")
        n_done += 1
        print(f"cached {c.competition_name} {c.season_name}: {len(rows)} pens", flush=True)
    remaining = sum(1 for _, c in male.iterrows()
                    if not (CACHE / f"{c.competition_id}_{c.season_id}.csv").exists())
    print(f"[run complete] processed {n_done} this run; {remaining} comp-seasons remaining", flush=True)
    return remaining


DEDUP = ["match_id", "kicker_id", "period", "minute", "team", "end_y"]


def merge():
    # start from the canonical 910-penalty corpus so a partial pull can never SHRINK it
    frames = []
    base = C.DATA / "men_penalties_full.csv"
    if base.exists():
        frames.append(pd.read_csv(base))
    for f in sorted(CACHE.glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df):
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["end_y"]).drop_duplicates(subset=DEDUP)
    out = C.DATA / "men_penalties_full_v2.csv"
    full.to_csv(out, index=False, encoding="utf-8")
    vc = full.kicker.value_counts()
    print(f"saved {out}")
    print(f"rows: {len(full)} | kickers: {vc.size} | >=5 pens: {(vc>=5).sum()} | >=10: {(vc>=10).sum()}")
    print(f"in-play / shootout: {(~full.is_shootout.astype(bool)).sum()} / {full.is_shootout.astype(bool).sum()}")
    print("direction:", full.direction.value_counts(dropna=False).to_dict())
    print("competitions:", full.competition.nunique())
    return full


def pull_until_done(max_passes=8):
    """Re-run the resumable pull until no comp-seasons remain or no progress is
    made (the flaky-network case). Each pass only retries the uncached ones."""
    prev = None
    for p in range(max_passes):
        print(f"--- pass {p + 1} ---", flush=True)
        remaining = pull_all()
        if remaining == 0:
            print("ALL comp-seasons cached.", flush=True)
            return 0
        if remaining == prev:
            print(f"no progress ({remaining} remaining) — network likely down; stop.", flush=True)
            return remaining
        prev = remaining
    return remaining


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "merge":
        merge()
    elif arg == "loop":
        pull_until_done()
    else:
        pull_all()
