from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=["spread","elo_diff","recent_margin_diff","recent_off_diff","recent_def_diff","home_adv","rest_diff","home_games","away_games"]

class ATSModel:
    def __init__(self):
        self.logit=make_pipeline(StandardScaler(),LogisticRegression(C=0.35,max_iter=2000))
        self.gb=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=0.045,l2_regularization=2.0,random_state=7)
        self.margin=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=0.045,l2_regularization=2.0,random_state=7)
        self.fitted=False
    def fit(self, df):
        d=df.dropna(subset=FEATURES+["y"]).copy(); X=d[FEATURES].astype(float); y=d.y.astype(int)
        self.logit.fit(X,y); self.gb.fit(X,y); self.margin.fit(X,d.home_margin); self.fitted=True; return self
    def predict(self, df):
        X=df[FEATURES].astype(float)
        p1=self.logit.predict_proba(X)[:,1]; p2=self.gb.predict_proba(X)[:,1]
        margin=self.margin.predict(X); sigma=14.0
        p3=1/(1+np.exp(-(margin+df.spread.values)/sigma))
        p=np.clip(0.35*p1+0.40*p2+0.25*p3,0.025,0.975)
        n=(df.home_games.fillna(0)+df.away_games.fillna(0)).values
        shrink=np.minimum(0.35,0.25/(1+0.12*n)); p=0.5+(p-0.5)*(1-shrink)
        return p,margin
    def save(self,path): Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)
    @staticmethod
    def load(path): return joblib.load(path)
