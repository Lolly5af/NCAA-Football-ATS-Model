import argparse
from pathlib import Path
import pandas as pd
from .cfbd_client import CFBD
from .features import normalize_games, normalize_lines, make_training_frame

def build_year(client,year,force=False):
    games=normalize_games(client.games(year,force)); lines=[]
    max_week=int(games.week.max()) if not games.empty else 15
    for w in range(1,max_week+1):
        try: lines.extend(client.lines(year,w,force))
        except Exception: pass
    lines=normalize_lines(lines)
    if lines.empty: raise RuntimeError(f"No betting lines returned for {year}")
    return make_training_frame(games,lines)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--build-history",action="store_true"); ap.add_argument("--current-season",type=int); ap.add_argument("--years",default="2021,2022,2023,2024,2025"); a=ap.parse_args()
    c=CFBD(); years=[int(x) for x in a.years.split(",")] if a.build_history else []
    if a.current_season: years.append(a.current_season)
    frames=[]
    for y in sorted(set(years)): print("Building",y); frames.append(build_year(c,y))
    df=pd.concat(frames,ignore_index=True).sort_values(["season","week","date","game_id"])
    Path("data/processed").mkdir(parents=True,exist_ok=True); df.to_parquet("data/processed/training_frame.parquet",index=False)
    print(f"Wrote {len(df):,} lined FBS games.")
if __name__=="__main__": main()
