import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from .model import ATSModel, FEATURES

def run_backtest(df,start_season,end_season):
    df=df.sort_values(["season","week","date","game_id"]).copy(); all_preds=[]
    for season in range(start_season,end_season+1):
        test=df[df.season==season]
        if test.empty: continue
        for week in sorted(test.week.unique()):
            prior=df[(df.season<season)|((df.season==season)&(df.week<week))].dropna(subset=["y"]+FEATURES)
            if len(prior)<500: continue
            m=ATSModel().fit(prior); tw=test[test.week==week].dropna(subset=FEATURES).copy()
            if tw.empty: continue
            p,margin=m.predict(tw); tw["p_home_cover"]=p; tw["pred_home_cover"]=(p>=0.5).astype(int); tw["model_margin"]=margin
            tw["correct"]=np.where(tw.cover_margin>0,tw.pred_home_cover==1,np.where(tw.cover_margin<0,tw.pred_home_cover==0,np.nan)); all_preds.append(tw)
    pred=pd.concat(all_preds,ignore_index=True) if all_preds else pd.DataFrame()
    if pred.empty: raise RuntimeError("No predictions generated. Build historical data first.")
    settled=pred.dropna(subset=["correct","y"]).copy(); rows=[]
    for season,g in settled.groupby("season"):
        rows.append(summary_row(g,season))
    rows.append(summary_row(settled,"ALL")); summary=pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True); summary.to_csv("reports/backtest_summary.csv",index=False); pred.to_csv("reports/backtest_predictions.csv",index=False)
    print(summary.to_string(index=False)); return summary,pred

def summary_row(g,label):
    # Market baseline = favorite/side implied by the spread: home covers when model market side is home.
    market_pick=(g.spread>=0).astype(int)
    market_correct=np.where(g.cover_margin>0,market_pick==1,np.where(g.cover_margin<0,market_pick==0,np.nan))
    return {"season":label,"games":len(g),"ats_accuracy":g.correct.mean(),"market_baseline_accuracy":np.nanmean(market_correct),"brier":brier_score_loss(g.y,g.p_home_cover),"log_loss":log_loss(g.y,g.p_home_cover,labels=[0,1]),"avg_abs_spread_edge":np.mean(np.abs(g.model_margin+g.spread))}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--start-season",type=int,default=2023); ap.add_argument("--end-season",type=int,default=2025); ap.add_argument("--data",default="data/processed/training_frame.parquet"); a=ap.parse_args(); run_backtest(pd.read_parquet(a.data),a.start_season,a.end_season)
