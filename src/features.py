from collections import defaultdict
import numpy as np
import pandas as pd

INITIAL_ELO = 1500.0
K = 22.0
HOME_ADV = 55.0
REGRESSION = 0.65

def _safe(x, default=0.0):
    try:
        if x is None or pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def normalize_games(raw):
    rows=[]
    for g in raw:
        if not g.get("completed"): continue
        if g.get("homeClassification") != "fbs" or g.get("awayClassification") != "fbs": continue
        rows.append({"game_id":g["id"],"season":g["season"],"week":g["week"],"date":g["startDate"],"home":g["homeTeam"],"away":g["awayTeam"],"home_pts":g["homePoints"],"away_pts":g["awayPoints"],"neutral":bool(g.get("neutralSite",False))})
    return pd.DataFrame(rows)

def normalize_lines(raw):
    out=[]
    for g in raw:
        gid=g.get("id") or g.get("gameId")
        for line in g.get("lines", []):
            spread=line.get("spread")
            if spread is None: continue
            out.append({"game_id":gid,"spread":float(spread)})
            break
    return pd.DataFrame(out).drop_duplicates("game_id")

def team_stats_features(games):
    games=games.sort_values(["season","week","date","game_id"]).copy()
    elo=defaultdict(lambda:INITIAL_ELO)
    prior_season=defaultdict(lambda:INITIAL_ELO)
    state=defaultdict(lambda:{"m":[],"o":[],"d":[],"last":None,"n":0})
    records=[]
    for _,g in games.iterrows():
        h,a=g.home,g.away
        if g.season not in []:
            if state[h]["n"]==0 and state[a]["n"]==0 and (h in prior_season or a in prior_season):
                pass
        he,ae=elo[h],elo[a]
        expected=1/(1+10**((ae-(he+(0 if g.neutral else HOME_ADV)))/400))
        records.append({"game_id":g.game_id,"season":g.season,"week":g.week,
                        "elo_diff":he-ae,"home_adv":0.0 if g.neutral else HOME_ADV,
                        "recent_margin_diff":np.mean(state[h]["m"][-5:]) if state[h]["m"] else 0.0 - (np.mean(state[a]["m"][-5:]) if state[a]["m"] else 0.0),
                        "recent_off_diff":np.mean(state[h]["o"][-5:]) if state[h]["o"] else 0.0 - (np.mean(state[a]["o"][-5:]) if state[a]["o"] else 0.0),
                        "recent_def_diff":np.mean(state[a]["d"][-5:]) if state[a]["d"] else 0.0 - (np.mean(state[h]["d"][-5:]) if state[h]["d"] else 0.0),
                        "rest_diff":0.0,"home_games":state[h]["n"],"away_games":state[a]["n"]})
        hm=float(g.home_pts); am=float(g.away_pts); margin=hm-am
        elo[h]=he+K*((1 if margin>0 else 0 if margin<0 else .5)-expected)
        elo[a]=ae+K*((1 if margin<0 else 0 if margin>0 else .5)-(1-expected))
        state[h]["m"].append(margin); state[a]["m"].append(-margin)
        state[h]["o"].append(hm); state[a]["o"].append(am)
        state[h]["d"].append(am); state[a]["d"].append(hm)
        state[h]["n"]+=1; state[a]["n"]+=1
        state[h]["last"]=g.date; state[a]["last"]=g.date
    return pd.DataFrame(records)

def make_training_frame(games, lines):
    feat=team_stats_features(games)
    df=games.merge(feat,on=["game_id","season","week"],how="left").merge(lines,on="game_id",how="inner")
    df["home_margin"]=df.home_pts-df.away_pts
    df["cover_margin"]=df.home_margin+df.spread
    df["y"]=(df.cover_margin>0).astype(int)
    df.loc[df.cover_margin==0,"y"]=np.nan
    return df
