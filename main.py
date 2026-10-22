import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai  # SDK officiel Google GenAI

app = FastAPI(title="NHL GoalScorer Analytics API", version="4.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation sécurisée du client Gemini
try:
    ai_client = genai.Client()
except Exception as e:
    print(f"Attention: Client Gemini non initialisé ({e})")
    ai_client = None

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
        url = f"https://api-web.nhle.com/v1/club-stats/{abbr}/now"
        response = requests.get(url, timeout=5)
        
        scorers = []
        if response.status_code == 200:
            data = response.json()
            skaters = data.get('skaters', [])
            
            for p in skaters:
                pos = p.get('positionCode', '')
                if pos in ['L', 'R', 'C', 'F']:
                    first = p.get('firstName', {}).get('default', '')
                    last = p.get('lastName', {}).get('default', '')
                    goals = p.get('goals', 0)
                    games_played = p.get('gamesPlayed', 1)
                    shots = p.get('shots', 0)
                    
                    if games_played == 0:
                        continue
                    
                    goals_per_game = goals / games_played
                    shooting_pct = (goals / shots) * 100 if shots > 0 else 10.0
                    
                    base_prob = 30 + (goals_per_game * 40) + (shooting_pct * 0.4)
                    prob_val = max(25, min(82, int(base_prob)))
                    odds_val = round(1.65 + ((100 - prob_val) / 20), 2)

                    if prob_val >= 68:
                        momentum = "🔥 En feu (Streak active)"
                        conf = "⭐⭐⭐⭐⭐ (Top Pick)"
                    elif prob_val >= 58:
                        momentum = "⚡ Forte dynamique"
                        conf = "⭐⭐⭐⭐ (Recommandé)"
                    elif prob_val >= 48:
                        momentum = "📊 Forme stable"
                        conf = "⭐⭐⭐ (Intéressant)"
                    else:
                        momentum = "❄️ En recherche de rythme"
                        conf = "⭐⭐ (Risqué)"

                    scorers.append({
                        "name": f"{first} {last}".strip(),
                        "pos": pos,
                        "stats": f"{goals} buts ({games_played} m)",
                        "momentum": momentum,
                        "prob": f"{prob_val}%",
                        "odds": f"{odds_val}",
                        "confidence": conf,
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
        "prob": "45%", 
        "odds": "2.10", 
        "confidence": "⭐⭐⭐ (Intéressant)"
    }]

def generate_gemini_ai_insight(home_team: str, away_team: str, home_scorers: list, away_scorers: list):
    """Génère une analyse experte unique et dynamique pour chaque match."""
    if not ai_client:
        return f"🎯 **Pronostic IA** : Client Gemini non initialisé."
        
    try:
        # On force l'IA à analyser précisément les données uniques des deux équipes du match
        prompt = f"""
        Tu es un analyste expert en paris sportifs et un spécialiste de la Ligue Nationale de Hockey (NHL).
        Analyse de manière unique et spécifique le match du jour opposant l'équipe à domicile '{home_team}' à l'équipe visiteuse '{away_team}'.
        
        Voici les meilleurs buteurs et leurs statistiques actuelles pour ce match précis :
        - Équipe Domicile ({home_team}) : {home_scorers[:3]}
        - Équipe Extérieur ({away_team}) : {away_scorers[:3]}
        
        Rédige un pronostic court, percutant et 100% personnalisé pour ce match (maximum 3-4 phrases en français) pour un pari "Anytime Goalscorer". 
        Cite explicitement les noms des joueurs de ce match précis, leurs probabilités, et donne ton avis d'expert. Utilise des balises markdown en gras (**texte**) pour les noms et les cotes. Ne fais pas de réponse générique.
        """
        
        response = ai_client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        print(f"ERREUR API GEMINI : {error_msg}")
        
        if "503" in error_msg or "high demand" in error_msg:
            top_player = home_scorers[0]['name'] if home_scorers else "l'attaquant vedette"
            top_odds = home_scorers[0]['odds'] if home_scorers else "2.10"
            return f"🎯 **Pronostic IA (Mode Statistiques)** : Pour le choc **{home_team}** vs **{away_team}**, **{top_player}** dégage le meilleur profil statistique (cote estimée à **{top_odds}**)."
            
        return f"🎯 **Pronostic IA** : Analyse tactique en cours pour {home_team} face à {away_team}."
    
    return f"🎯 **Pronostic IA** : Analyse non disponible pour l'instant."

@app.post("/predict/goalscorers")
def predict_goalscorers(req: MatchRequest):
    home_abbr = get_nhl_abbr(req.home_team)
    away_abbr = get_nhl_abbr(req.away_team)
    
    with ThreadPoolExecutor() as executor:
        f_home = executor.submit(fetch_top_goalscorers, req.home_team)
        f_away = executor.submit(fetch_top_goalscorers, req.away_team)
        home_scorers = f_home.result()
        away_scorers = f_away.result()

    ai_insight = generate_gemini_ai_insight(req.home_team, req.away_team, home_scorers, away_scorers)

    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "home_logo": f"https://assets.nhle.com/logos/nhl/svg/{home_abbr}_light.svg",
        "away_logo": f"https://assets.nhle.com/logos/nhl/svg/{away_abbr}_light.svg",
        "top_goalscorers_home": home_scorers,
        "top_goalscorers_away": away_scorers,
        "ai_insight": ai_insight
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
