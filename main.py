
import os
import json
import time
import math
import sqlite3
import asyncio
import logging
import requests

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scipy.stats import poisson


# ============================================================
# NHL GOALSCORER ANALYTICS API - VERSION 38.0
# ============================================================

VERSION = "38.00"
DB_FILE = os.getenv("DB_FILE", "nhl_history.db")
PORT = int(os.getenv("PORT", "8000"))
CACHE_TTL = 300
TIMEOUT = 12
PARIS_TZ = ZoneInfo("Europe/Paris")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NHL-IA")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "NHL-Analytics-38.0",
    "Accept": "application/json"
})

API = "https://api-web.nhle.com/v1"


# ============================================================
# ÉQUIPES NHL
# ============================================================

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
    "ARIZONA": "UTA", "COYOTES": "UTA",
    "UTAH": "UTA", "HOCKEY CLUB": "UTA", "UTA": "UTA",
    "EDMONTON": "EDM", "OILERS": "EDM", "EDM": "EDM",
    "CALGARY": "CGY", "FLAMES": "CGY", "CGY": "CGY",
    "VANCOUVER": "VAN", "CANUCKS": "VAN", "VAN": "VAN",
    "SEATTLE": "SEA", "KRAKEN": "SEA", "SEA": "SEA",
    "LOS ANGELES": "LAK", "KINGS": "LAK", "LAK": "LAK",
    "ANAHEIM": "ANA", "DUCKS": "ANA", "ANA": "ANA",
    "SAN JOSE": "SJS", "SHARKS": "SJS", "SJS": "SJS",
    "VEGAS": "VGK", "GOLDEN KNIGHTS": "VGK",
    "KNIGHTS": "VGK", "VGK": "VGK"
}

ALL_TEAMS = sorted(set(TEAM_MAP_NHL.values()))


def get_nhl_abbr(team_str: str) -> str:
    name = team_str.upper().strip()

    if name in TEAM_MAP_NHL:
        return TEAM_MAP_NHL[name]

    for key in sorted(TEAM_MAP_NHL, key=len, reverse=True):
        if key in name:
            return TEAM_MAP_NHL[key]

    raise ValueError(f"Équipe NHL inconnue : {team_str}")


# ============================================================
# CACHE DES DONNÉES
# ============================================================

CACHE = {}
CACHE_LOCK = asyncio.Lock() if False else None


def cached_get(key, fetcher, ttl=CACHE_TTL):
    now = time.time()

    item = CACHE.get(key)

    if item and now - item["time"] < ttl:
        return item["data"]

    try:
        data = fetcher()

        CACHE[key] = {
            "time": now,
            "data": data
        }

        return data

    except Exception as exc:
        logger.warning("Cache %s : %s", key, exc)

        if item:
            return item["data"]

        return None


def nhl_get(endpoint):
    response = SESSION.get(
        f"{API}/{endpoint.lstrip('/')}",
        timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# BASE DE DONNÉES
# ============================================================

def db_connection():
    db = sqlite3.connect(DB_FILE, timeout=20)
    db.row_factory = sqlite3.Row
    return db


def init_database():
    with db_connection() as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                match_name TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_xg REAL,
                away_xg REAL,
                home_win REAL,
                away_win REAL,
                over_55 REAL,
                prediction_json TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT
            )
        """)

        db.commit()


def save_prediction(result):
    try:
        prediction = result["match_prediction"]

        with db_connection() as db:
            db.execute("""
                INSERT INTO predictions (
                    created_at, match_name,
                    home_team, away_team,
                    home_xg, away_xg,
                    home_win, away_win,
                    over_55, prediction_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                result["match"],
                result["home_team"],
                result["away_team"],
                prediction["expected_goals_home"],
                prediction["expected_goals_away"],
                prediction["home_win_pct"],
                prediction["away_win_pct"],
                prediction["over_5_5_goals_prob"],
                json.dumps(result, ensure_ascii=False)
            ))

            db.commit()

    except Exception as exc:
        logger.error("Erreur sauvegarde : %s", exc)


# ============================================================
# STATISTIQUES OFFICIELLES DES ÉQUIPES
# ============================================================

def fetch_team_stats_summary(team_str):
    abbr = get_nhl_abbr(team_str)

    data = cached_get(
        f"stats_{abbr}",
        lambda: nhl_get(f"club-stats/{abbr}/now"),
        900
    )

    if not data:
        return None

    skaters = data.get("skaters", [])
    goalies = data.get("goalies", [])

    total_goals = sum(
        int(p.get("goals", 0) or 0)
        for p in skaters
    )

    games = max(
        [
            int(p.get("gamesPlayed", 0) or 0)
            for p in skaters
        ] or [0]
    )

    goalie_gaa = []
    goalie_save = []

    for goalie in goalies:
        gp = int(goalie.get("gamesPlayed", 0) or 0)

        if gp <= 0:
            continue

        gaa = goalie.get("goalsAgainstAvg")
        sv = goalie.get("savePctg")

        if gaa is not None:
            goalie_gaa.append(float(gaa))

        if sv is not None:
            goalie_save.append(float(sv))

    return {
        "abbr": abbr,
        "goals": total_goals,
        "games": games,
        "goals_per_game": round(
            total_goals / max(1, games), 3
        ),
        "skaters": skaters,
        "goalies": goalies,
        "average_goalie_gaa": (
            round(sum(goalie_gaa) / len(goalie_gaa), 3)
            if goalie_gaa else None
        ),
        "average_goalie_save_pct": (
            round(sum(goalie_save) / len(goalie_save), 4)
            if goalie_save else None
        )
    }


# ============================================================
# EFFECTIFS
# ============================================================

def fetch_team_roster(team_str):
    abbr = get_nhl_abbr(team_str)

    data = cached_get(
        f"roster_{abbr}",
        lambda: nhl_get(f"roster/{abbr}/current"),
        1800
    )

    if not data:
        return []

    players = []

    for group in ["forwards", "defensemen", "goalies"]:
        for player in data.get(group, []):

            first = player.get("firstName", {})
            last = player.get("lastName", {})

            if isinstance(first, dict):
                first = first.get("default", "")

            if isinstance(last, dict):
                last = last.get("default", "")

            players.append({
                "player_id": player.get("id"),
                "name": f"{first} {last}".strip(),
                "position": player.get("positionCode", ""),
                "number": player.get("sweaterNumber"),
                "status": "Présent dans l'effectif NHL"
            })

    return players


# ============================================================
# FORME DES JOUEURS : 10 DERNIERS MATCHS
# ============================================================

def fetch_player_recent_form(player_id):
    if not player_id:
        return None

    data = cached_get(
        f"form_{player_id}",
        lambda: nhl_get(
            f"player/{player_id}/game-log/now"
        ),
        1800
    )

    if not data:
        return None

    games = data.get("gameLog", [])

    if not games:
        return None

    recent = games[:10]
    count = len(recent)

    goals = sum(
        int(g.get("goals", 0) or 0)
        for g in recent
    )

    assists = sum(
        int(g.get("assists", 0) or 0)
        for g in recent
    )

    shots = sum(
        int(g.get("shots", 0) or 0)
        for g in recent
    )

    return {
        "games": count,
        "goals": goals,
        "assists": assists,
        "shots": shots,
        "goals_per_game": round(goals / count, 3),
        "assists_per_game": round(assists / count, 3),
        "shots_per_game": round(shots / count, 2)
    }


# ============================================================
# GARDIENS
# ============================================================

def goalie_analysis(stats):
    if not stats:
        return []

    result = []

    for goalie in stats.get("goalies", []):

        games = int(
            goalie.get("gamesPlayed", 0) or 0
        )

        if games <= 0:
            continue

        first = goalie.get("firstName", {})
        last = goalie.get("lastName", {})

        if isinstance(first, dict):
            first = first.get("default", "")

        if isinstance(last, dict):
            last = last.get("default", "")

        result.append({
            "name": f"{first} {last}".strip(),
            "games": games,
            "save_pct": goalie.get("savePctg"),
            "gaa": goalie.get("goalsAgainstAvg"),
            "wins": goalie.get("wins"),
            "losses": goalie.get("losses"),
            "shutouts": goalie.get("shutouts")
        })

    result.sort(
        key=lambda x: x.get("save_pct") or 0,
        reverse=True
    )

    return result[:3]


# ============================================================
# MOTEUR DE PROBABILITÉ JOUEUR
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def probability_to_odds(probability):
    if probability <= 0:
        return None

    return round(1 / probability, 2)


def calculate_player_probability(
    goals,
    assists,
    games,
    recent_form=None,
    opponent_factor=1.0,
    team_factor=1.0,
    metric="goal"
):
    games = max(1, int(games or 1))

    if metric == "goal":
        season_rate = goals / games

        recent_rate = (
            recent_form.get("goals_per_game")
            if recent_form else None
        )
    else:
        season_rate = assists / games

        recent_rate = (
            recent_form.get("assists_per_game")
            if recent_form else None
        )

    if recent_rate is not None:
        player_rate = (
            0.70 * season_rate
            + 0.30 * recent_rate
        )
    else:
        player_rate = season_rate

    expected_rate = (
        player_rate
        * opponent_factor
        * team_factor
    )

    probability = 1 - math.exp(
        -max(0, expected_rate)
    )

    probability = clamp(
        probability,
        0.001,
        0.95
    )

    return {
        "probability": round(probability * 100, 1),
        "fair_odds": probability_to_odds(probability),
        "expected_rate": round(expected_rate, 3),
        "season_rate": round(season_rate, 3),
        "recent_rate": (
            round(recent_rate, 3)
            if recent_rate is not None else None
        )
    }


# ============================================================
# BUTEURS
# ============================================================

def fetch_nhl_goalscorers(
    team_str,
    opponent_factor=1.0,
    team_factor=1.0
):
    stats = fetch_team_stats_summary(team_str)

    if not stats:
        return []

    scorers = []

    for player in stats["skaters"]:

        position = player.get("positionCode", "")

        if position not in ["L", "R", "C", "F"]:
            continue

        goals = int(player.get("goals", 0) or 0)
        assists = int(player.get("assists", 0) or 0)
        games = int(player.get("gamesPlayed", 0) or 0)

        if games <= 0:
            continue

        first = player.get("firstName", {})
        last = player.get("lastName", {})

        if isinstance(first, dict):
            first = first.get("default", "")

        if isinstance(last, dict):
            last = last.get("default", "")

        player_id = player.get("playerId") or player.get("id")

        recent = fetch_player_recent_form(player_id)

        prediction = calculate_player_probability(
            goals,
            assists,
            games,
            recent,
            opponent_factor,
            team_factor,
            "goal"
        )

        scorers.append({
            "player_id": player_id,
            "name": f"{first} {last}".strip(),
            "pos": position,
            "stats": f"{goals} buts ({games} m)",
            "recent_form": recent,
            "momentum": (
                "🔥 Forme récente positive"
                if recent and recent["goals"] > 0
                else "📊 Statistiques de saison"
            ),
            "prob": f'{prediction["probability"]}%',
            "odds": prediction["fair_odds"],
            "fair_odds": prediction["fair_odds"],
            "expected_rate": prediction["expected_rate"],
            "sort_val": prediction["probability"]
        })

    scorers.sort(
        key=lambda x: x["sort_val"],
        reverse=True
    )

    for player in scorers:
        del player["sort_val"]

    return scorers[:10]


# ============================================================
# PASSEURS
# ============================================================

def fetch_nhl_playmakers(
    team_str,
    opponent_factor=1.0,
    team_factor=1.0
):
    stats = fetch_team_stats_summary(team_str)

    if not stats:
        return []

    playmakers = []

    for player in stats["skaters"]:

        position = player.get("positionCode", "")

        if position not in ["L", "R", "C", "F", "D"]:
            continue

        goals = int(player.get("goals", 0) or 0)
        assists = int(player.get("assists", 0) or 0)
        games = int(player.get("gamesPlayed", 0) or 0)

        if games <= 0:
            continue

        first = player.get("firstName", {})
        last = player.get("lastName", {})

        if isinstance(first, dict):
            first = first.get("default", "")

        if isinstance(last, dict):
            last = last.get("default", "")

        player_id = player.get("playerId") or player.get("id")

        recent = fetch_player_recent_form(player_id)

        prediction = calculate_player_probability(
            goals,
            assists,
            games,
            recent,
            opponent_factor,
            team_factor,
            "assist"
        )

        playmakers.append({
            "player_id": player_id,
            "name": f"{first} {last}".strip(),
            "pos": position,
            "stats": f"{assists} passes ({games} m)",
            "recent_form": recent,
            "momentum": (
                "🔥 Forme récente positive"
                if recent and recent["assists"] > 0
                else "📊 Statistiques de saison"
            ),
            "prob": f'{prediction["probability"]}%',
            "odds": prediction["fair_odds"],
            "fair_odds": prediction["fair_odds"],
            "expected_rate": prediction["expected_rate"],
            "sort_val": prediction["probability"]
        })

    playmakers.sort(
        key=lambda x: x["sort_val"],
        reverse=True
    )

    for player in playmakers:
        del player["sort_val"]

    return playmakers[:10]


# ============================================================
# MODÈLE DE MATCH : POISSON
# ============================================================

def calculate_match_model(home_xg, away_xg):

    home_xg = clamp(float(home_xg), 0.2, 8)
    away_xg = clamp(float(away_xg), 0.2, 8)

    home_win = 0.0
    away_win = 0.0
    draw = 0.0
    over_55 = 0.0
    under_55 = 0.0

    best_probability = -1
    best_home = 0
    best_away = 0

    for home_goals in range(13):
        for away_goals in range(13):

            probability = (
                poisson.pmf(home_goals, home_xg)
                * poisson.pmf(away_goals, away_xg)
            )

            if home_goals > away_goals:
                home_win += probability

            elif away_goals > home_goals:
                away_win += probability

            else:
                draw += probability

            if home_goals + away_goals >= 6:
                over_55 += probability
            else:
                under_55 += probability

            if probability > best_probability:
                best_probability = probability
                best_home = home_goals
                best_away = away_goals

    # Probabilités de victoire après prolongation
    home_final = home_win + draw * 0.5
    away_final = away_win + draw * 0.5

    total = home_final + away_final

    home_final /= total
    away_final /= total

    return {
        "home_win_pct": round(home_final * 100, 1),
        "away_win_pct": round(away_final * 100, 1),
        "draw_or_ot_pct": round(draw * 100, 1),
        "over_5_5_goals_prob": round(over_55 * 100, 1),
        "under_5_5_goals_prob": round(under_55 * 100, 1),
        "expected_goals_home": round(home_xg, 2),
        "expected_goals_away": round(away_xg, 2),
        "expected_total_goals": round(home_xg + away_xg, 2),
        "most_likely_score": {
            "home": best_home,
            "away": best_away
        }
    }


# ============================================================
# ANALYSE DE VALUE
# ============================================================

def calculate_value(probability_percent, market_odds):

    probability = probability_percent / 100

    fair_odds = probability_to_odds(probability)

    if market_odds is None or market_odds <= 1:
        return {
            "market_odds": None,
            "fair_odds": fair_odds,
            "value_percent": None,
            "value_detected": False,
            "message": "Cote bookmaker non renseignée"
        }

    implied_probability = 1 / market_odds

    edge = (
        probability - implied_probability
    ) * 100

    return {
        "market_odds": market_odds,
        "fair_odds": fair_odds,
        "value_percent": round(edge, 2),
        "value_detected": edge >= 3,
        "message": (
            "Value potentielle"
            if edge >= 3
            else "Pas de value suffisante"
        )
    }


# ============================================================
# ESTIMATION DES BUTS PAR ÉQUIPE
# ============================================================

def calculate_team_expectation(
    team_stats,
    opponent_stats,
    home=False
):
    if not team_stats:
        return 3.0

    attack = team_stats["goals_per_game"]

    if opponent_stats:
        goalie_gaa = opponent_stats.get(
            "average_goalie_gaa"
        )

        if goalie_gaa is not None:
            defensive_factor = clamp(
                goalie_gaa / 3.0,
                0.75,
                1.25
            )
        else:
            defensive_factor = 1.0
    else:
        defensive_factor = 1.0

    home_factor = 1.04 if home else 0.96

    return round(
        clamp(
            attack * defensive_factor * home_factor,
            1.5,
            5.5
        ),
        2
    )


# ============================================================
# MODÈLE PRINCIPAL
# ============================================================

def build_match_prediction(req):

    home_abbr = get_nhl_abbr(req.home_team)
    away_abbr = get_nhl_abbr(req.away_team)

    home_stats = fetch_team_stats_summary(req.home_team)
    away_stats = fetch_team_stats_summary(req.away_team)

    home_xg = calculate_team_expectation(
        home_stats,
        away_stats,
        True
    )

    away_xg = calculate_team_expectation(
        away_stats,
        home_stats,
        False
    )

    model = calculate_match_model(home_xg, away_xg)

    home_factor = (
        clamp(
            3.0 / away_stats["goals_per_game"],
            0.8,
            1.2
        )
        if away_stats and away_stats["goals_per_game"] > 0
        else 1.0
    )

    away_factor = (
        clamp(
            3.0 / home_stats["goals_per_game"],
            0.8,
            1.2
        )
        if home_stats and home_stats["goals_per_game"] > 0
        else 1.0
    )

    home_scorers = fetch_nhl_goalscorers(
        req.home_team,
        home_factor,
        1.04
    )

    away_scorers = fetch_nhl_goalscorers(
        req.away_team,
        away_factor,
        0.96
    )

    home_playmakers = fetch_nhl_playmakers(
        req.home_team,
        home_factor,
        1.04
    )

    away_playmakers = fetch_nhl_playmakers(
        req.away_team,
        away_factor,
        0.96
    )

    home_roster = fetch_team_roster(req.home_team)
    away_roster = fetch_team_roster(req.away_team)

    home_goalies = goalie_analysis(home_stats)
    away_goalies = goalie_analysis(away_stats)

    home_value = calculate_value(
        model["home_win_pct"],
        req.home_market_odds
    )

    away_value = calculate_value(
        model["away_win_pct"],
        req.away_market_odds
    )

    over_value = calculate_value(
        model["over_5_5_goals_prob"],
        req.over_55_market_odds
    )

    result = {
        "version": VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "match": f"{req.home_team} vs {req.away_team}",
        "home_team": req.home_team,
        "away_team": req.away_team,

        "home_logo": (
            f"https://assets.nhle.com/logos/nhl/svg/{home_abbr}_light.svg"
        ),

        "away_logo": (
            f"https://assets.nhle.com/logos/nhl/svg/{away_abbr}_light.svg"
        ),

        "match_prediction": {
            **model,
            "home_win_pct": f'{model["home_win_pct"]}%',
            "away_win_pct": f'{model["away_win_pct"]}%',
            "draw_or_ot_pct": f'{model["draw_or_ot_pct"]}%',
            "over_5_5_goals_prob": (
                f'{model["over_5_5_goals_prob"]}%'
            ),
            "under_5_5_goals_prob": (
                f'{model["under_5_5_goals_prob"]}%'
            )
        },

        "top_goalscorers_home": home_scorers,
        "top_goalscorers_away": away_scorers,

        "top_playmakers_home": home_playmakers,
        "top_playmakers_away": away_playmakers,

        "goalies_home": home_goalies,
        "goalies_away": away_goalies,

        "roster_home": home_roster,
        "roster_away": away_roster,

        "team_stats": {
            "home": (
                {
                    "goals_per_game": home_stats["goals_per_game"],
                    "games": home_stats["games"]
                }
                if home_stats else None
            ),
            "away": (
                {
                    "goals_per_game": away_stats["goals_per_game"],
                    "games": away_stats["games"]
                }
                if away_stats else None
            )
        },

        "value_analysis": {
            "home_win": home_value,
            "away_win": away_value,
            "over_5_5": over_value
        },

        "data_quality": {
            "home_stats_available": home_stats is not None,
            "away_stats_available": away_stats is not None,
            "recent_player_form": (
                "Données disponibles selon l'API NHL"
            ),
            "injuries": (
                "Blessures non confirmées par cette API. "
                "Disponibilité à vérifier avant le match."
            )
        }
    }

    save_prediction(result)

    return result


# ============================================================
# MODÈLES DE REQUÊTES
# ============================================================

class MatchRequest(BaseModel):
    league: str = "NHL"

    home_team: str
    away_team: str

    home_id: Optional[str] = None
    away_id: Optional[str] = None

    home_market_odds: Optional[float] = Field(
        default=None,
        gt=1,
        le=1000
    )

    away_market_odds: Optional[float] = Field(
        default=None,
        gt=1,
        le=1000
    )

    over_55_market_odds: Optional[float] = Field(
        default=None,
        gt=1,
        le=1000
    )


class LegacyMatchRequest(BaseModel):
    home_team: str
    away_team: str

    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7

    home_b2b: bool = False
    away_b2b: bool = False


# ============================================================
# CALENDRIER NHL
# ============================================================

def fetch_nhl_score(date_string):

    return cached_get(
        f"score_{date_string}",
        lambda: nhl_get(f"score/{date_string}"),
        60
    )


# ============================================================
# SYNCHRONISATION AUTOMATIQUE
# ============================================================

def perform_sync():

    success = 0
    errors = 0

    for team in ALL_TEAMS:

        try:
            stats = fetch_team_stats_summary(team)
            roster = fetch_team_roster(team)

            if stats is not None and roster:
                success += 1
            else:
                errors += 1

        except Exception as exc:
            errors += 1
            logger.warning("Sync %s : %s", team, exc)

    status = "success" if errors == 0 else "partial"

    with db_connection() as db:
        db.execute("""
            INSERT INTO sync_history (
                created_at, status, message
            )
            VALUES (?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            status,
            f"{success} équipes actualisées, {errors} erreurs"
        ))

        db.commit()

    return {
        "status": status,
        "teams_updated": success,
        "errors": errors,
        "time": datetime.now(timezone.utc).isoformat()
    }


async def background_nhl_sync():

    while True:

        try:
            logger.info("AUTOPILOT : synchronisation NHL")

            await asyncio.to_thread(perform_sync)

            logger.info("AUTOPILOT : synchronisation terminée")

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception("Erreur synchronisation automatique")

        await asyncio.sleep(43200)


# ============================================================
# INITIALISATION FASTAPI
# IMPORTANT : APP AVANT TOUTES LES ROUTES
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    init_database()

    logger.info("NHL IA VERSION %s : démarrage", VERSION)

    task = asyncio.create_task(background_nhl_sync())

    yield

    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="NHL GoalScorer Analytics API",
    version=VERSION,
    description="NHL Analytics - Buteurs, passeurs et prédictions",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# PAGE D'ACCUEIL
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_home():

    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        return """
        <html>
        <head><title>NHL IA 38.0</title></head>
        <body>
        <h2>NHL GoalScorer Analytics API 38.0</h2>
        <p>API opérationnelle.</p>
        <a href="/docs">Documentation API</a>
        </body>
        </html>
        """


# ============================================================
# SANTÉ DE L'APPLICATION
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "version": VERSION,
        "database": DB_FILE,
        "time": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# MATCHS NHL
# ============================================================

@app.get("/api/matches/nhl")
def get_nhl_matches(date: str):

    try:
        requested_date = datetime.strptime(
            date,
            "%Y-%m-%d"
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Date invalide. Format YYYY-MM-DD."
        )

    dates_to_check = [
        (requested_date + timedelta(days=offset)).strftime(
            "%Y-%m-%d"
        )
        for offset in [-1, 0, 1]
    ]

    matches = []
    seen = set()

    for nhl_date in dates_to_check:

        try:
            data = fetch_nhl_score(nhl_date)

            if not data:
                continue

            for game in data.get("games", []):

                game_id = str(game.get("id"))

                if game_id in seen:
                    continue

                start_utc = game.get("startTimeUTC")

                if not start_utc:
                    continue

                dt = datetime.fromisoformat(
                    start_utc.replace("Z", "+00:00")
                )

                local_dt = dt.astimezone(PARIS_TZ)

                if local_dt.strftime("%Y-%m-%d") != date:
                    continue

                seen.add(game_id)

                home = game.get("homeTeam", {})
                away = game.get("awayTeam", {})

                home_name = (
                    home.get("placeName", {}).get("default", "")
                    + " "
                    + home.get("commonName", {}).get("default", "")
                ).strip()

                away_name = (
                    away.get("placeName", {}).get("default", "")
                    + " "
                    + away.get("commonName", {}).get("default", "")
                ).strip()

                state = game.get("gameState", "")

                if state in ["FINAL", "OFF"]:
                    display_state = (
                        f'Final ({home.get("score", 0)} - '
                        f'{away.get("score", 0)})'
                    )

                elif state in ["LIVE", "CRIT"]:
                    display_state = (
                        f'EN DIRECT ({home.get("score", 0)} - '
                        f'{away.get("score", 0)})'
                    )

                else:
                    display_state = local_dt.strftime("%H:%M")

                matches.append({
                    "game_id": game_id,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_abbr": home.get("abbrev"),
                    "away_abbr": away.get("abbrev"),
                    "home_logo": home.get("logo", ""),
                    "away_logo": away.get("logo", ""),
                    "game_state": display_state,
                    "start_time": local_dt.isoformat()
                })

        except Exception as exc:
            logger.warning(
                "Calendrier %s : %s",
                nhl_date,
                exc
            )

    matches.sort(key=lambda x: x["start_time"])

    return {
        "date": date,
        "timezone": "Europe/Paris",
        "matches": matches
    }


# ============================================================
# PRÉDICTIONS BUTEURS ET PASSEURS
# ============================================================

@app.post("/predict/goalscorers")
def predict_goalscorers(req: MatchRequest):

    if req.league.upper() != "NHL":
        raise HTTPException(
            status_code=400,
            detail="Cette route est réservée à la NHL."
        )

    try:
        return build_match_prediction(req)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"API NHL indisponible : {exc}"
        )

    except Exception as exc:
        logger.exception("Erreur prédiction")

        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la prédiction."
        )


# ============================================================
# COMPATIBILITÉ ANCIENNE ROUTE /predict/match
# ============================================================

@app.post("/predict/match")
def predict_match_legacy(req: LegacyMatchRequest):

    home_xg = req.home_expected_goals
    away_xg = req.away_expected_goals

    if req.home_b2b:
        home_xg *= 0.92

    if req.away_b2b:
        away_xg *= 0.92

    model = calculate_match_model(
        home_xg,
        away_xg
    )

    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "home_team": req.home_team,
        "away_team": req.away_team,
        "prediction": model,
        "match_prediction": model,
        "version": VERSION
    }


# ============================================================
# HISTORIQUE DES PRÉDICTIONS
# ============================================================

@app.get("/api/history")
def get_history(limit: int = 50):

    limit = int(clamp(limit, 1, 500))

    with db_connection() as db:

        rows = db.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {
        "count": len(rows),
        "predictions": [dict(row) for row in rows]
    }


@app.get("/api/history/{prediction_id}")
def get_history_detail(prediction_id: int):

    with db_connection() as db:

        row = db.execute("""
            SELECT prediction_json
            FROM predictions
            WHERE id = ?
        """, (prediction_id,)).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Prédiction introuvable"
        )

    return json.loads(row["prediction_json"])


# ============================================================
# ACTUALISATION MANUELLE
# ============================================================

@app.get("/api/predict/auto-sync")
def auto_sync():

    try:
        return perform_sync()

    except Exception:
        logger.exception("Erreur synchronisation")

        raise HTTPException(
            status_code=500,
            detail="Erreur pendant la synchronisation NHL."
        )


# ============================================================
# HISTORIQUE SYNCHRONISATIONS
# ============================================================

@app.get("/api/sync/history")
def get_sync_history():

    with db_connection() as db:

        rows = db.execute("""
            SELECT *
            FROM sync_history
            ORDER BY id DESC
            LIMIT 30
        """).fetchall()

    return {
        "synchronizations": [dict(row) for row in rows]
    }


# ============================================================
# COMPATIBILITÉ FOOTBALL
# ============================================================

@app.get("/api/football/matches")
def football_matches(date: Optional[str] = None):

    return {
        "date": date,
        "matches": [],
        "status": "football_not_configured",
        "message": (
            "Le module football doit être raccordé "
            "à une source de données football."
        )
    }


# ============================================================
# LANCEMENT LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )
