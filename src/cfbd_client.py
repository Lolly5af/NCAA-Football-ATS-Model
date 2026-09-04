import os, time, json
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = "https://api.collegefootballdata.com"
CACHE = Path("data/raw")
CACHE.mkdir(parents=True, exist_ok=True)

class CFBD:
    def __init__(self):
        self.key = os.getenv("CFBD_API_KEY")
        if not self.key:
            raise RuntimeError("Set CFBD_API_KEY in .env or the environment.")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.key}"})

    def get(self, endpoint, params, cache_name=None, force=False):
        if cache_name:
            p = CACHE / cache_name
            if p.exists() and not force:
                return json.loads(p.read_text())
        r = self.s.get(BASE + endpoint, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if cache_name:
            p.write_text(json.dumps(data))
        time.sleep(0.15)
        return data

    def games(self, year, force=False):
        return self.get("/games", {"year":year, "seasonType":"regular", "classification":"fbs"}, f"games_{year}.json", force)

    def lines(self, year, week, force=False):
        return self.get("/lines", {"year":year, "week":week, "seasonType":"regular"}, f"lines_{year}_{week}.json", force)

    def team_games(self, year, week, force=False):
        return self.get("/games/teams", {"year":year, "week":week, "seasonType":"regular", "classification":"fbs"}, f"teamgames_{year}_{week}.json", force)

    def sp(self, year, force=False):
        return self.get("/ratings/sp", {"year":year}, f"sp_{year}.json", force)

    def talent(self, year, force=False):
        return self.get("/talent", {"year":year}, f"talent_{year}.json", force)
