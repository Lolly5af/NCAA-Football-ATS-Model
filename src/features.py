from collections import defaultdict
import numpy as np
import pandas as pd

INITIAL_ELO=1500.0; K=22.0; HOME_ADV=55.0

def normalize_games(raw):
    rows=[]
    for g in raw:
        if not g.get("completed"): continue
        if g.get("homeClassification")!="fbs" or g.get("awayClassification")!="fbs": continue
        rows.append({"game_id":g["id"],"season":g["season"],"week":g["week"],"date":g["startDate"],"home":g["homeTeam"],"away":g["awayTeam"],"home_pts":g["homePoints"],"away_pts":g["awayPoints"],"neutral":bool(g.get("neutralSite",False))})
    return pd.DataFrame(rows)

def normalize_lines(raw):
    out=[]
    for g in raw:
        gid=g.get("id") or g.get("gameId")
        for line in g.get("lines",[]):
            if line.get("spread") is not None: out.append({"game_id":gid,"spread":float(line["spread"])}); break
    return pd.DataFrame(out).drop_duplicates("game_id")

def _mean(vals): return float(np.mean(vals[-5:])) if vals else 0.0

def team_stats_features(games):
    games=games.sort_values(["season","week","date","game_id"]).copy(); elo=defaultdict(lambda:INITIAL_ELO); state=defaultdict(lambda:{"m":[],"o":[],"d":[],"n":0}); records=[]; last_season=None
    for _,g in games.iterrows():
        if last_season is not None and g.season!=last_season:
            for team in state: elo[team]=INITIAL_ELO+0.65*(elo[team]-INITIAL_ELO); state[team]["n"]=0
        last_season=g.season; h,a=g.home,g.away; he,ae=elo[h],elo[a]
        expected=1/(1+10**((ae-(he+(0 if g.neutral else HOME_ADV)))/400))
        records.append({"game_id":g.game_id,"season":g.season,"week":g.week,"elo_diff":he-ae,"home_adv":0.0 if g.neutral else HOME_ADV,"recent_margin_diff":_mean(state[h]["m"])-_mean(state[a]["m"]),"recent_off_diff":_mean(state[h]["o"])-_mean(state[a]["o"]),"recent_def_diff":_mean(state[a]["d"])-_mean(state[h]["d"]),"rest_diff":0.0,"home_games":state[h]["n"],"away_games":state[a]["n"]})
        hm,am=float(g.home_pts),float(g.away_pts); margin=hm-am
        elo[h]=he+K*((1 if margin>0 else 0 if margin<0 else .5)-expected); elo[a]=ae+K*((1 if margin<0 else 0 if margin>0 else .5)-(1-expected))
        state[h]["m"].append(margin); state[a]["m"].append(-margin); state[h]["o"].append(hm); state[a]["o"].append(am); state[h]["d"].append(am); state[a]["d"].append(hm); state[h]["n"]+=1; state[a]["n"]+=1
    return pd.DataFrame(records)

def make_training_frame(games,lines):
    df=games.merge(team_stats_features(games),on=["game_id","season","week"],how="left").merge(lines,on="game_id",how="inner"); df["home_margin"]=df.home_pts-df.away_pts; df["cover_margin"]=df.home_margin+df.spread; df["y"]=(df.cover_margin>0).astype(int); df.loc[df.cover_margin==0,"y"]=np.nan; return df
