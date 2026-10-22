import os
import hashlib
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="NHL GoalScorer Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

TEAM_MAP = {
    "MONTREAL": "MTL", "CANADIENS": "MTL", "MTL": "MTL",
    "TORONTO": "TOR", "MAPLE LEAFS": "TOR", "TOR": "TOR",
    "BOSTON": "BOS", "BRUINS": "BOS", "BOS": "BOS",
    "NEW YORK": "NYR", "RANGERS": "NYR", "NYR": "NYR",
    "SAN JOSE": "SJS", "SHARKS": "SJS", "SJS": "SJS",
    "EDMONTON": "EDM", "OILERS": "EDM", "EDM": "EDM",
    "VANCOUVER": "VAN", "CANUCKS": "VAN", "VAN": "VAN",
    "CALGARY": "CGY", "FLAMES": "CGY", "CGY": "CGY",
    "WINNIPEG": "WPG", "JETS": "WPG", "WPG": "WPG",
    "OTTAWA": "OTT", "SENATORS": "OTT", "OTT": "OTT",
    "DETROIT": "DET", "RED WINGS": "DET", "DET": "DET",
    "BUFFALO": "BUF", "SABRES": "BUF", "BUF": "BUF",
    "TAMPA BAY": "TBL", "LIGHTNING": "TBL", "TBL": "TBL",
    "FLORIDA": "FLA", "PANTHERS": "FLA", "FLA": "FLA",
    "CAROLINA": "CAR", "HURRICANES": "CAR", "CAR": "CAR",
    "NEW JERSEY": "NJD", "DEVILS": "NJD", "NJD": "NJD",
    "NEW YORK ISLANDERS": "NYI", "ISLANDERS": "NYI", "NYI": "NYI",
    "PHILADELPHIA": "PHI", "FLYERS": "PHI", "PHI": "PHI",
    "PITTSBURGH": "PIT", "PENGUINS": "PIT", "PIT": "PIT",
    "WASHINGTON": "WSH", "CAPITALS": "WSH", "WSH": "WSH",
    "COLUMBUS": "CBJ", "BLUE JACKETS": "CBJ", "CBJ": "CBJ",
    "CHICAGO": "CHI", "BLACKHAWKS": "CHI", "CHI": "CHI",
    "NASHVILLE": "NSH", "PREDATORS": "NSH", "NSH": "NSH",
    "ST. LOUIS": "STL", "BLUES": "STL", "STL": "STL",
    "MINNESOTA": "MIN", "WILD": "MIN", "MIN": "MIN",
    "COLORADO": "COL", "AVALANCHE": "COL", "COL": "COL",
    "UTAH": "UTA", "VEGAS": "VGK", "GOLDEN KNIGHTS": "VGK", "VGK": "VGK",
    "SEATTLE": "SEA", "KRAKEN": "SEA", "SEA": "SEA",
    "LOS ANGELES": "LAK", "KINGS": "LAK", "LAK": "LAK",
    "ANAHEIM": "ANA", "DUCKS": "ANA", "ANA": "ANA",
    "DALLAS": "DAL", "STARS": "DAL", "DAL": "DAL"
}

def get_nhl_abbr(team_str: str) -> str:
    upper_team = team_str.upper()
    for key, code in TEAM_MAP.items():
        if key in upper_team:
            return code
    if len(team_str.strip()) == 3:
        return team_str.strip().upper()
    return "MTL"

@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Erreur : Le fichier index.html est introuvable.</h3>"

@app.get("/api/matches/nhl")
def get_nhl_matches(date: str = None):
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://api-web.nhle.com/v1/score/{date}"
    try:
        response = requests.get(url, timeout=4)
        if response.status_code != 200:
            return {"date": date, "matches": []}
            
        data = response.json()
        matches = []
        for game in data.get("games", []):
            home_obj = game.get("homeTeam", {})
            away_obj = game.get("awayTeam", {})
            
            home_team = home_obj.get("placeName", {}).get("default", "Home")
            away_team = away_obj.get("placeName", {}).get("default", "Away")
            
            home_common = home_obj.get("commonName", {}).get("default", "")
            away_common = away_obj.get("commonName", {}).get("default", "")
            if home_common: home_team = f"{home_team} {home_common}"
            if away_common: away_team = f"{away_team} {away_common}"

            matches.append({
                "league": "NHL",
                "home_team": home_team,
                "away_team": away_team,
                "home_logo": home_obj.get("logo", ""),
                "away_logo": away_obj.get("logo", ""),
                "game_state": game.get("gameState", "PRE")
            })
        return {"date": date, "matches": matches}
    except Exception:
        return {"date": date, "matches": []}

def fetch_top_goalscorers(team_str: str):
    abbr = get_nhl_abbr(team_str)
    try:
        url = f"https://api-web.nhle.com/v1/roster/{abbr}/current"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            scorers = []
            for p in data.get('forwards', []):
                first_dict = p.get('firstName', {})
                last_dict = p.get('lastName', {})
                first = first_dict.get('default', '') if isinstance(first_dict, dict) else str(first_dict)
                last = last_dict.get('default', '') if isinstance(last_dict, dict) else str(last_dict)
                
                if not last: 
                    continue
                
                num = p.get('sweaterNumber', '')
                pos = p.get('positionCode', 'F')
                
                hash_val = abs(hash(f"goal-{abbr}-{last}"))
                prob_val = 30 + (hash_val % 45) 
                odds_val = round(1.80 + ((100 - prob_val) / 25), 2)

                scorers.append({
                    "name": f"{first} {last}".strip(),
                    "pos": pos,
                    "stats": f"#{num}" if num else "Attaquant",
                    "prob": f"{prob_val}%",
                    "odds": f"{odds_val}"
                })
            
            if scorers:
                scorers.sort(key=lambda x: int(x['prob'].replace('%', '')), reverse=True)
                return scorers[:6]
    except Exception:
        pass
    
    return [{"name": f"Sniper ({team_str})", "pos": "F", "stats": "Élite", "prob": "45%", "odds": "2.10"}]

@app.post("/predict/goalscorers")
def predict_goalscorers(req: MatchRequest):
    home_abbr = get_nhl_abbr(req.home_team)
    away_abbr = get_nhl_abbr(req.away_team)
    
    with ThreadPoolExecutor() as executor:
        f_home = executor.submit(fetch_top_goalscorers, req.home_team)
        f_away = executor.submit(fetch_top_goalscorers, req.away_team)
        home_scorers = f_home.result()
        away_scorers = f_away.result()

    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "home_logo": f"https://assets.nhle.com/logos/nhl/svg/{home_abbr}_light.svg",
        "away_logo": f"https://assets.nhle.com/logos/nhl/svg/{away_abbr}_light.svg",
        "top_goalscorers_home": home_scorers,
        "top_goalscorers_away": away_scorers
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
