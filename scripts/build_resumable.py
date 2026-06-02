"""Resumable men's penalty extractor: caches one parquet per competition-season,
skips finished ones, so it can be run repeatedly until complete."""
import sys, os, warnings
import pandas as pd, numpy as np
from statsbombpy import sb
warnings.filterwarnings("ignore")
os.makedirs("cache", exist_ok=True)
HW = 1.5
def lr_c(y, hw=HW):
    if y is None or (isinstance(y,float) and np.isnan(y)): return None
    if y < 40-hw: return 'R'
    if y > 40+hw: return 'L'
    return 'C'

comps = sb.competitions()
male = comps[comps.competition_gender=='male'].reset_index(drop=True)
budget = int(sys.argv[1]) if len(sys.argv)>1 else 4   # comps to process this run
done = 0
for _, c in male.iterrows():
    tag = f"cache/{c.competition_id}_{c.season_id}.csv"
    if os.path.exists(tag): continue
    if done >= budget: break
    try:
        ms = sb.matches(competition_id=c.competition_id, season_id=c.season_id)
    except Exception:
        pd.DataFrame().to_csv(tag, index=False); continue
    rows=[]
    for _, m in ms.iterrows():
        try: ev = sb.events(match_id=m.match_id)
        except Exception: continue
        if 'shot_type' not in ev.columns: continue
        pk = ev[(ev['type']=='Shot') & (ev['shot_type']=='Penalty')].copy()
        pk = pk.sort_values(['period','minute','second'], na_position='last')
        order={}
        for _, s in pk.iterrows():
            el=s.get('shot_end_location')
            ey=el[1] if isinstance(el,(list,tuple)) and len(el)>=2 else None
            is_so=(s.get('period')==5); key=(m.match_id,s.get('team'))
            order[key]=order.get(key,0)+1
            rows.append(dict(match_id=m.match_id, competition=c.competition_name, season=str(c.season_name),
                period=s.get('period'), minute=s.get('minute'), is_shootout=is_so,
                shootout_kick_order=order[key] if is_so else None, team=s.get('team'),
                kicker=s.get('player'), kicker_id=s.get('player_id'),
                body_part=s.get('shot_body_part'), outcome=s.get('shot_outcome'),
                end_y=ey, direction=lr_c(ey)))
    pd.DataFrame(rows).to_csv(tag, index=False)
    done += 1
    print(f"cached {c.competition_name} {c.season_name}: {len(rows)} pens", flush=True)
remaining = sum(1 for _,c in male.iterrows() if not os.path.exists(f"cache/{c.competition_id}_{c.season_id}.csv"))
print(f"[run complete] processed {done} this run; {remaining} comp-seasons remaining")
