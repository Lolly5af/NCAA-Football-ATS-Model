import argparse
import numpy as np
import pandas as pd
from .model import ATSModel, FEATURES

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--season",type=int,default=2026); ap.add_argument("--week",type=int,required=True); a=ap.parse_args()
    df=pd.read_parquet("data/processed/training_frame.parquet")
    current=df[(df.season==a.season)&(df.week==a.week)].copy()
    if current.empty: raise RuntimeError("No current-season games found.")
    train=df[((df.season<a.season)|((df.season==a.season)&(df.week<a.week)))].dropna(subset=["y"]+FEATURES)
    m=ATSModel().fit(train); p,margin=m.predict(current)
    current["home_cover_probability"]=p; current["away_cover_probability"]=1-p; current["model_margin"]=margin
    current["pick"]=np.where(p>=0.5,current.home,current.away); current["pick_probability"]=np.maximum(p,1-p); current["edge_vs_market"]=margin+current.spread
    cols=["date","away","home","spread","pick","pick_probability","model_margin","edge_vs_market"]
    out=current[cols].sort_values("pick_probability",ascending=False)
    out.to_csv(f"reports/{a.season}_week_{a.week}_picks.csv",index=False); print(out.to_string(index=False))
if __name__=="__main__": main()
