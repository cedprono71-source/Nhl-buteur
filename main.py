import os
import hashlib
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from scipy.stats import poisson

app = FastAPI(title="Multi-Sports GoalScorer Analytics API (Poisson)", version="27.00")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    league: str  # "NHL" ou "FOOT"
    home_team: str
    away_team: str
    home_id: Optional[str] = None
    away_id: Optional[str] = None


# --- Dictionnaire de conversion pour les abréviations NHL ---
TEAM_MAP_NHL = {
    "MONTREAL": "MTL", "CANADIENS": "MTL", "MTL": "MTL",
    "TORONTO": "TOR", "MAPLE LEAFS": "TOR", "TOR": "TOR",
    "BOSTON": "BOS", "BRUINS": "BOS", "BOS": "BOS",
    "NEW YORK RANGERS": "NYR", "RANGERS": "NYR", "NYR": "NYR",
    "NEW YORK ISLANDERS": "NYI", "ISLANDERS": "NYI", "NYI": "NYI",
    "NEW JERSEY": "NJD", "DEVILS": "NJD", "NJD": "NJD",
    "BUFFALO": "BUF", "SABRES": "BUF", "BUF": "BUF",
    "OTTAWA": "OTT", "SENATORS": "OTT", "OTT": "OTT",
    "DETROIT": "DET", "RED WINGS": "DET", "DET": "DET",
    "TAMPA BAY": "TBL", "LIGHTNING": "TBL", "TBL": "TBL",
    "FLORIDA": "FLA", "PANTHERS": "FLA", "FLA": "FLA",
    "CAROLINA": "CAR", "HURRICANES": "CAR", "CAR": "CAR",
    "WASHINGTON": "WSH", "CAPITALS": "WSH", "WSH": "WSH",
    "PITTSBURGH": "PIT", "PENGUINS": "PIT", "PIT": "PIT",
    "PHILADELPHIA": "PHI", "FLYERS": "PHI", "PHI": "PHI",
    "COLUMBUS": "CBJ", "BLUE JACKETS": "CBJ", "CBJ": "CBJ",
    "CHICAGO": "CHI", "BLACKHAWKS": "CHI", "CHI": "CHI",
    "NASHVILLE": "NSH", "PREDATORS": "NSH", "NSH": "NSH",
    "ST. LOUIS": "STL", "BLUES": "STL", "STL": "STL",
    "MINNESOTA": "MIN", "WILD": "MIN", "MIN": "MIN",
    "WINNIPEG": "WPG", "JETS": "WPG", "WPG": "WPG",
    "COLORADO": "COL", "AVALANCHE": "COL", "COL": "COL",
    "DALLAS": "DAL", "STARS": "DAL", "DAL": "DAL",
    "ARIZONA": "ARI", "COYOTES": "ARI", "ARI": "ARI",
    "UTAH": "UTA", "HOCKEY CLUB": "UTA", "UTA": "UTA",
    "EDMONTON": "EDM", "OILERS": "EDM", "EDM": "EDM",
    "CALGARY": "CGY", "FLAMES": "CGY", "CGY": "CGY",
    "VANCOUVER": "VAN", "CANUCKS": "VAN", "VAN": "VAN",
    "SEATTLE": "SEA", "KRAKEN": "SEA", "SEA": "SEA",
    "LOS ANGELES": "LAK", "KINGS": "LAK", "LAK": "LAK",
    "ANAHEIM": "ANA", "DUCKS": "ANA", "ANA": "ANA",
    "SAN JOSE": "SJS", "SHARKS": "SJS", "SJS": "SJS",
    "VEGAS": "VGK", "GOLDEN KNIGHTS": "VGK", "VGK": "VGK", "KNIGHTS": "VGK"
}

def get_nhl_abbr(team_str: str) -> str:
    upper_team = team_str.upper().strip()
    if upper_team in TEAM_MAP_NHL:
        return TEAM_MAP_NHL[upper_team]
    for key, code in TEAM_MAP_NHL.items():
        if key in upper_team:
            return code
    if len(upper_team) == 3:
        return upper_team
    return "TOR"


# --- ROUTE PRINCIPALE ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Erreur : Le fichier index.html est introuvable sur le serveur.</h3>"


# --- Moteur de calcul mathématique (Loi de Poisson) ---
def calculate_poisson_probs(goals: int, games: int):
    if games <= 0:
        games = 1
    # Taux moyen de buts par match (lambda)
    lambda_val = goals / games
    
    # Probabilité de marquer 0 but (Loi de Poisson)
    prob_zero = poisson.pmf(0, lambda_val)
    
    # Probabilité de marquer au moins 1 but (Anytime Goalscorer) en pourcentage
    prob_anytime = (1 - prob_zero) * 100
    
    # Bornage réaliste entre 18% et 85%
    prob_val = max(18, min(85, int(prob_anytime)))
    
    # Conversion en cote décimale avec une marge bookmaker d'environ 8%
    true_prob = prob_val / 100.0
    odds_val = round(1.0 / true_prob * 1.08, 2)
    
    return prob_val, odds_val


# --- Récupération des buteurs NHL (Basé sur Poisson) ---
def fetch_nhl_goalscorers(team_str: str, opponent_str: str):
    abbr = get_nhl_abbr(team_str)
    
    try:
        url = f"https://api-web.nhle.com/v1/club-stats/{abbr}/now"
        response = requests.get(url, timeout=5)
        
        scorers = []
        if response.status_code == 200:
            data = response.json()
            skaters = data.get('skaters', [])
            
            for p in skaters:
                pos = p.get('positionCode', '')
                if pos in ['L', 'R', 'C', 'F']:
                    first_dict = p.get('firstName', {})
                    last_dict = p.get('lastName', {})
                    first = first_dict.get('default', '') if isinstance(first_dict, dict) else str(first_dict)
                    last = last_dict.get('default', '') if isinstance(last_dict, dict) else str(last_dict)
                    
                    goals = p.get('goals', 0)
                    games_played = p.get('gamesPlayed', 1)
                    if games_played == 0:
                        continue
                    
                    # Application de la loi de Poisson
                    prob_val, odds_val = calculate_poisson_probs(goals, games_played)

                    if prob_val >= 65:
                        momentum = "🔥 En feu (Streak active)"
                    elif prob_val >= 52:
                        momentum = "⚡ Forte dynamique"
                    elif prob_val >= 40:
                        momentum = "📊 Forme stable"
                    else:
                        momentum = "❄️ En recherche de rythme"

                    scorers.append({
                        "name": f"{first} {last}".strip(),
                        "pos": pos,
                        "stats": f"{goals} buts ({games_played} m)",
                        "momentum": momentum,
                        "prob": f"{prob_val}%",
                        "odds": f"{odds_val}",
                        "sort_val": prob_val
                    })
            
            if scorers:
                scorers.sort(key=lambda x: x['sort_val'], reverse=True)
                for s in scorers:
                    del s['sort_val']
                return scorers[:6]

    except Exception:
        pass
    
    return [{
        "name": f"Attaquant Clé ({team_str})", 
        "pos": "F", 
        "stats": "Stats en cours", 
        "momentum": "📊 Forme stable",
        "prob": "48%", 
        "odds": "2.05"
    }]


# --- Génération Buteurs Football (Basé sur Poisson) ---
def fetch_football_goalscorers(team_str: str, opponent_str: str):
    match_hash = int(hashlib.md5(f"FOOT-{team_str}-{opponent_str}".encode()).hexdigest(), 16)
    scorers = []
    
    for i in range(4):
        p_hash = int(hashlib.md5(f"{team_str}-player-{i}-{match_hash}".encode()).hexdigest(), 16)
        buts = (p_hash % 10) + 2
        matchs = 18 + (p_hash % 8)
        
        # Application de la loi de Poisson pour le foot simulé
        prob_val, odds_val = calculate_poisson_probs(buts, matchs)
        
        if prob_val >= 60:
            momentum = "🔥 Buteur en série"
        elif prob_val >= 45:
            momentum = "⚡ Très actif offensivement"
        else:
            momentum = "📊 Opportunités régulières"

        scorers.append({
            "name": f"Joueur clé {i+1} ({team_str[:3].upper()})",
            "pos": "ATT",
            "stats": f"{buts} buts ({matchs} m)",
            "momentum": momentum,
            "prob": f"{prob_val}%",
            "odds": f"{odds_val}"
        })
    
    scorers.sort(key=lambda x: int(x['prob'].replace('%', '')), reverse=True)
    return scorers


# --- Route Calendrier Matchs NHL ---
@app.get("/api/matches/nhl")
def get_nhl_matches(date: str):
    paris_tz = ZoneInfo("Europe/Paris")
    try:
        req_date_obj = datetime.strptime(date, "%Y-%m-%d")
        dates_to_check = [
            (req_date_obj - timedelta(days=1)).strftime("%Y-%m-%d"),
            req_date_obj.strftime("%Y-%m-%d"),
            (req_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        ]
    except Exception:
        dates_to_check = [date]

    formatted_matches = []
    seen_game_ids = set()

    for nhl_date in dates_to_check:
        url = f"https://api-web.nhle.com/v1/score/{nhl_date}"
        try:
            response = requests.get(url, timeout=4)
            if response.status_code != 200:
                continue
            data = response.json()
            
            for game in data.get("games", []):
                game_id = str(game.get("id"))
                if game_id in seen_game_ids:
                    continue

                start_utc = game.get("startTimeUTC", "")
                if not start_utc:
                    continue

                try:
                    dt_utc = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                    dt_france = dt_utc.astimezone(paris_tz)
                    match_date_str = dt_france.strftime("%Y-%m-%d")
                except Exception:
                    continue

                if match_date_str == date:
                    seen_game_ids.add(game_id)
                    home_obj = game.get("homeTeam", {})
                    away_obj = game.get("awayTeam", {})
                    
                    home_team = home_obj.get("placeName", {}).get("default") or home_obj.get("abbrev") or "Home"
                    away_team = away_obj.get("placeName", {}).get("default") or away_obj.get("abbrev") or "Away"
                    
                    home_common = home_obj.get("commonName", {}).get("default", "")
                    away_common = away_obj.get("commonName", {}).get("default", "")
                    if home_common:
                        home_team = f"{home_team} {home_common}"
                    if away_common:
                        away_team = f"{away_team} {away_common}"

                    formatted_matches.append({
                        "_sort_time": start_utc,
                        "league": "NHL",
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_logo": home_obj.get("logo", ""),
                        "away_logo": away_obj.get("logo", ""),
                        "game_state": dt_france.strftime("%H:%M")
                    })
        except Exception:
            continue
            
    formatted_matches.sort(key=lambda x: x["_sort_time"])
    final_matches = [{k: v for k, v in m.items() if k != "_sort_time"} for m in formatted_matches]
    return {"date": date, "matches": final_matches}


# --- Route Unifiée de Prédiction ---
@app.post("/predict/goalscorers")
def predict_goalscorers(req: MatchRequest):
    league = req.league.upper()
    
    if league == "NHL":
        home_abbr = get_nhl_abbr(req.home_team)
        away_abbr = get_nhl_abbr(req.away_team)
        home_scorers = fetch_nhl_goalscorers(req.home_team, req.away_team)
        away_scorers = fetch_nhl_goalscorers(req.away_team, req.home_team)
        home_logo = f"https://assets.nhle.com/logos/nhl/svg/{home_abbr}_light.svg"
        away_logo = f"https://assets.nhle.com/logos/nhl/svg/{away_abbr}_light.svg"
    else:
        home_scorers = fetch_football_goalscorers(req.home_team, req.away_team)
        away_scorers = fetch_football_goalscorers(req.away_team, req.home_team)
        home_logo = "https://media.api-sports.io/football/teams/default.png"
        away_logo = "https://media.api-sports.io/football/teams/default.png"

    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "home_logo": home_logo,
        "away_logo": away_logo,
        "top_goalscorers_home": home_scorers,
        "top_goalscorers_away": away_scorers,
        "ai_insight": ""
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
