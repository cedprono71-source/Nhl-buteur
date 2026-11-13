import os
import json
import time
import math
import sqlite3
import asyncio
import logging
import re
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scipy.stats import nbinom, poisson

# ============================================================
# NHL GOALSCORER ANALYTICS - VERSION 39.72
# Adapté pour Vercel : plus de boucle de fond au démarrage.
# Les mises à jour automatiques passent par GitHub Actions.
# ============================================================

VERSION = "39.72"
DB_FILE = os.getenv("DB_FILE", "nhl_history.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()   # si défini : PostgreSQL externe (ex. Neon), sinon SQLite local
USE_PG = bool(DATABASE_URL)
PORT = int(os.getenv("PORT", "8000"))
CACHE_TTL = 300
TIMEOUT = 15
MAX_WORKERS = 8
CRON_SECRET = os.getenv("CRON_SECRET", "")              # si défini, /api/cron exige ?key=...
AUTO_PREDICT_HOURS_AHEAD = int(os.getenv("AUTO_PREDICT_HOURS_AHEAD", "10"))
SYNC_MIN_INTERVAL_HOURS = 6
LEAGUE_SH_PCT = 0.105   # % de réussite moyen des tirs (régression vers la moyenne)
SH_PRIOR_SHOTS = 60      # poids de la moyenne ligue, en tirs
TOI_FACTOR_RANGE = (0.90, 1.10)
TEAM_FORM_WEIGHT = 0.35     # poids de la forme des 10 derniers matchs dans les buts attendus
TEAM_FORM_MIN_GAMES = 5     # nb minimum de matchs récents pour utiliser la forme
OWN_B2B_FACTOR = 0.95       # équipe en back-to-back : moins de buts marqués
OPP_B2B_FACTOR = 1.03       # adversaire en back-to-back : un peu plus de buts pour nous
GOALS_DISPERSION = 30       # loi binomiale négative : variance = mu + mu^2/30 (Poisson = infini)
ELO_START = 1500
ELO_K = 6                   # vitesse d'apprentissage par match
ELO_HOME_ADVANTAGE = 30     # points Elo d'avantage à domicile (~54 % de victoires)
ELO_SEASON_REGRESSION = 2 / 3   # part conservée de l'écart à la moyenne d'une saison à l'autre
ELO_WEIGHT = 0.40           # poids d'Elo dans la probabilité de victoire finale
OT_EDGE_SHARE = 0.25        # la prolongation reste proche du 50/50 : seulement 25 % de l'écart de force
PARIS_TZ = ZoneInfo("Europe/Paris")
API = "https://api-web.nhle.com/v1"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("NHL-IA-39")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 NHL-GoalScorer-Analytics/39.1",
    "Accept": "application/json"
})

TEAM_MAP_NHL = {
    "MONTREAL": "MTL", "MONTREAL CANADIENS": "MTL", "CANADIENS": "MTL", "MTL": "MTL",
    "TORONTO": "TOR", "TORONTO MAPLE LEAFS": "TOR", "MAPLE LEAFS": "TOR", "TOR": "TOR",
    "BOSTON": "BOS", "BOSTON BRUINS": "BOS", "BRUINS": "BOS", "BOS": "BOS",
    "NEW YORK RANGERS": "NYR", "NY RANGERS": "NYR", "RANGERS": "NYR", "NYR": "NYR",
    "NEW YORK ISLANDERS": "NYI", "NY ISLANDERS": "NYI", "ISLANDERS": "NYI", "NYI": "NYI",
    "NEW JERSEY": "NJD", "NEW JERSEY DEVILS": "NJD", "DEVILS": "NJD", "NJD": "NJD",
    "BUFFALO": "BUF", "BUFFALO SABRES": "BUF", "SABRES": "BUF", "BUF": "BUF",
    "OTTAWA": "OTT", "OTTAWA SENATORS": "OTT", "SENATORS": "OTT", "OTT": "OTT",
    "DETROIT": "DET", "DETROIT RED WINGS": "DET", "RED WINGS": "DET", "DET": "DET",
    "TAMPA BAY": "TBL", "TAMPA BAY LIGHTNING": "TBL", "LIGHTNING": "TBL", "TBL": "TBL",
    "FLORIDA": "FLA", "FLORIDA PANTHERS": "FLA", "PANTHERS": "FLA", "FLA": "FLA",
    "CAROLINA": "CAR", "CAROLINA HURRICANES": "CAR", "HURRICANES": "CAR", "CAR": "CAR",
    "WASHINGTON": "WSH", "WASHINGTON CAPITALS": "WSH", "CAPITALS": "WSH", "WSH": "WSH",
    "PITTSBURGH": "PIT", "PITTSBURGH PENGUINS": "PIT", "PENGUINS": "PIT", "PIT": "PIT",
    "PHILADELPHIA": "PHI", "PHILADELPHIA FLYERS": "PHI", "FLYERS": "PHI", "PHI": "PHI",
    "COLUMBUS": "CBJ", "COLUMBUS BLUE JACKETS": "CBJ", "BLUE JACKETS": "CBJ", "CBJ": "CBJ",
    "CHICAGO": "CHI", "CHICAGO BLACKHAWKS": "CHI", "BLACKHAWKS": "CHI", "CHI": "CHI",
    "NASHVILLE": "NSH", "NASHVILLE PREDATORS": "NSH", "PREDATORS": "NSH", "NSH": "NSH",
    "ST LOUIS": "STL", "ST. LOUIS": "STL", "ST LOUIS BLUES": "STL", "BLUES": "STL", "STL": "STL",
    "MINNESOTA": "MIN", "MINNESOTA WILD": "MIN", "WILD": "MIN", "MIN": "MIN",
    "WINNIPEG": "WPG", "WINNIPEG JETS": "WPG", "JETS": "WPG", "WPG": "WPG",
    "COLORADO": "COL", "COLORADO AVALANCHE": "COL", "AVALANCHE": "COL", "COL": "COL",
    "DALLAS": "DAL", "DALLAS STARS": "DAL", "STARS": "DAL", "DAL": "DAL",
    "ARIZONA": "UTA", "ARIZONA COYOTES": "UTA", "COYOTES": "UTA",
    "UTAH": "UTA", "UTAH HOCKEY CLUB": "UTA", "UTAH MAMMOTH": "UTA", "HOCKEY CLUB": "UTA", "UTA": "UTA",
    "EDMONTON": "EDM", "EDMONTON OILERS": "EDM", "OILERS": "EDM", "EDM": "EDM",
    "CALGARY": "CGY", "CALGARY FLAMES": "CGY", "FLAMES": "CGY", "CGY": "CGY",
    "VANCOUVER": "VAN", "VANCOUVER CANUCKS": "VAN", "CANUCKS": "VAN", "VAN": "VAN",
    "SEATTLE": "SEA", "SEATTLE KRAKEN": "SEA", "KRAKEN": "SEA", "SEA": "SEA",
    "LOS ANGELES": "LAK", "LOS ANGELES KINGS": "LAK", "KINGS": "LAK", "LAK": "LAK",
    "ANAHEIM": "ANA", "ANAHEIM DUCKS": "ANA", "DUCKS": "ANA", "ANA": "ANA",
    "SAN JOSE": "SJS", "SAN JOSE SHARKS": "SJS", "SHARKS": "SJS", "SJS": "SJS",
    "VEGAS": "VGK", "VEGAS GOLDEN KNIGHTS": "VGK", "GOLDEN KNIGHTS": "VGK", "KNIGHTS": "VGK", "VGK": "VGK"
}
ALL_TEAMS = sorted(set(TEAM_MAP_NHL.values()))
GENERIC_TEAM_LABELS = {"DOMICILE", "EXTERIEUR", "HOME", "AWAY", "LOCAL", "VISITEUR", "VS", ""}

CACHE = {}

def normalize_team_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Nom d'équipe manquant ou invalide.")
    value = value.strip().upper()
    value = re.sub(r"[^A-Z0-9 .'-]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def get_nhl_abbr(team_str: str) -> str:
    name = normalize_team_name(team_str)
    if name in GENERIC_TEAM_LABELS:
        raise ValueError(
            f"Nom d'équipe invalide : « {team_str} ». "
            "L'interface doit envoyer le vrai nom ou le code NHL (ex. BOS, TOR, NYR), pas Domicile/Extérieur."
        )
    if name in TEAM_MAP_NHL:
        return TEAM_MAP_NHL[name]
    # Correspondance partielle uniquement si elle désigne UNE seule équipe.
    # - la saisie est un fragment d'un alias connu (ex. "CANADI")
    # - ou la saisie contient un alias connu (ex. "THE BOSTON BRUINS")
    candidates = set()
    if len(name) >= 4:
        for key, abbr in TEAM_MAP_NHL.items():
            if name in key or (len(key) >= 4 and key in name):
                candidates.add(abbr)
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        raise ValueError(
            f"Nom d'équipe ambigu : « {team_str} » (possibles : {', '.join(sorted(candidates))}). "
            "Utilise le nom complet ou le code NHL."
        )
    raise ValueError(f"Équipe NHL inconnue : {team_str}")

def cached_get(key, fetcher, ttl=CACHE_TTL):
    now = time.time()
    item = CACHE.get(key)
    if item and now - item["time"] < ttl:
        return item["data"]
    try:
        data = fetcher()
        CACHE[key] = {"time": now, "data": data}
        return data
    except Exception as exc:
        logger.warning("Cache %s : %s", key, exc)
        return item["data"] if item else None

def nhl_get(endpoint):
    response = SESSION.get(f"{API}/{endpoint.lstrip('/')}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()

# ---------------------- DATABASE ----------------------------

class _PgConnection:
    """Enveloppe PostgreSQL avec la même interface que sqlite3 (execute / commit / with)."""

    def __init__(self):
        import psycopg2
        import psycopg2.extras
        self._conn = psycopg2.connect(DATABASE_URL, connect_timeout=10,
                                      cursor_factory=psycopg2.extras.RealDictCursor)

    def execute(self, sql, params=()):
        cur = self._conn.cursor()
        cur.execute(sql.replace("?", "%s"), params)
        return cur

    def commit(self):
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()
        return False

def db_connection():
    if USE_PG:
        return _PgConnection()
    db = sqlite3.connect(DB_FILE, timeout=30)
    db.row_factory = sqlite3.Row
    return db

def init_database():
    pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real = "DOUBLE PRECISION" if USE_PG else "REAL"
    with db_connection() as db:
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id {pk},
                created_at TEXT NOT NULL,
                match_name TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_xg {real}, away_xg {real}, home_win {real}, away_win {real},
                over_55 {real}, prediction_json TEXT NOT NULL
            )
        """)
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS sync_history (
                id {pk},
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                prediction_id INTEGER PRIMARY KEY,
                game_id TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                home_score INTEGER, away_score INTEGER,
                players_json TEXT NOT NULL
            )
        """)
        db.commit()

def save_prediction(result):
    try:
        p = result["match_prediction"]
        with db_connection() as db:
            db.execute("""
                INSERT INTO predictions
                (created_at, match_name, home_team, away_team, home_xg, away_xg,
                 home_win, away_win, over_55, prediction_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), result["match"],
                result["home_team"], result["away_team"],
                p["expected_goals_home"], p["expected_goals_away"],
                p["home_win_pct_value"], p["away_win_pct_value"],
                p["over_5_5_goals_prob_value"], json.dumps(result, ensure_ascii=False)
            ))
            db.commit()
    except Exception:
        logger.exception("Erreur sauvegarde prédiction")

# ---------------------- TEAM DATA ---------------------------

def _weighted_mean(pairs, digits):
    """Moyenne pondérée par les matchs joués : un gardien remplaçant pèse moins que le titulaire."""
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return None
    return round(sum(value * weight for value, weight in pairs) / total, digits)

def fetch_team_stats_summary(team_str):
    abbr = get_nhl_abbr(team_str)
    data = cached_get(f"stats_{abbr}", lambda: nhl_get(f"club-stats/{abbr}/now"), 900)
    if not data:
        return None
    skaters = data.get("skaters") or []
    goalies = data.get("goalies") or []
    total_goals = sum(int(p.get("goals", 0) or 0) for p in skaters)
    games = max([int(p.get("gamesPlayed", 0) or 0) for p in skaters] or [0])
    gaa_values, save_values = [], []   # listes de (valeur, matchs joués)
    for goalie in goalies:
        gp = int(goalie.get("gamesPlayed", 0) or 0)
        if gp <= 0:
            continue
        try:
            if goalie.get("goalsAgainstAvg") is not None:
                gaa_values.append((float(goalie["goalsAgainstAvg"]), gp))
            if goalie.get("savePctg") is not None:
                save_values.append((float(goalie["savePctg"]), gp))
        except (TypeError, ValueError):
            pass
    return {
        "abbr": abbr, "goals": total_goals, "games": games,
        "goals_per_game": round(total_goals / max(1, games), 3),
        "skaters": skaters, "goalies": goalies,
        "average_goalie_gaa": _weighted_mean(gaa_values, 3),
        "average_goalie_save_pct": _weighted_mean(save_values, 4)
    }

def fetch_team_roster(team_str):
    abbr = get_nhl_abbr(team_str)
    data = cached_get(f"roster_{abbr}", lambda: nhl_get(f"roster/{abbr}/current"), 1800)
    if not data:
        return []
    players = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []) or []:
            first, last = p.get("firstName", {}), p.get("lastName", {})
            if isinstance(first, dict): first = first.get("default", "")
            if isinstance(last, dict): last = last.get("default", "")
            players.append({
                "player_id": p.get("id"), "name": f"{first} {last}".strip(),
                "position": p.get("positionCode", ""), "number": p.get("sweaterNumber"),
                "status": "Présent dans l'effectif NHL"
            })
    return players

def _toi_to_seconds(value):
    """Convertit 'MM:SS' (ou 'HH:MM:SS') ou un nombre en secondes. None si invalide."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    try:
        parts = [int(x) for x in str(value).split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds) if seconds > 0 else None

def fetch_player_recent_form(player_id):
    if not player_id:
        return None
    data = cached_get(f"form_{player_id}", lambda: nhl_get(f"player/{player_id}/game-log/now"), 1800)
    if not data:
        return None
    games = (data.get("gameLog") or [])[:10]
    if not games:
        return None
    count = len(games)
    goals = sum(int(g.get("goals", 0) or 0) for g in games)
    assists = sum(int(g.get("assists", 0) or 0) for g in games)
    shots = sum(int(g.get("shots", 0) or 0) for g in games)
    toi_values = [t for t in (_toi_to_seconds(g.get("toi")) for g in games) if t]
    return {
        "games": count, "goals": goals, "assists": assists, "shots": shots,
        "avg_toi_seconds": round(sum(toi_values) / len(toi_values), 1) if toi_values else None,
        "goals_per_game": round(goals / count, 3),
        "assists_per_game": round(assists / count, 3),
        "shots_per_game": round(shots / count, 2)
    }

# ---------------------- CONTEXTE D'ÉQUIPE ------------------

UPCOMING_STATES = ("FUT", "PRE", "LIVE", "CRIT")
PLAYED_STATES = ("FINAL", "OFF")

def fetch_team_schedule(abbr):
    return cached_get(f"sched_{abbr}", lambda: nhl_get(f"club-schedule-season/{abbr}/now"), 900)

def add_team_context(stats, abbr, game_date=None):
    """Ajoute à `stats` : jours de repos, back-to-back, forme sur les 10 derniers matchs.

    Si la date du match n'est pas donnée, on prend le prochain match non terminé de l'équipe.
    Sans calendrier disponible, les champs restent à None (aucun effet sur le modèle).
    """
    if not stats:
        return stats
    stats.update({"rest_days": None, "b2b": False, "recent_games": 0,
                  "recent_gf": None, "recent_ga": None, "effective_gaa": stats.get("average_goalie_gaa")})
    data = fetch_team_schedule(abbr)
    games = [g for g in (data or {}).get("games", []) if g.get("gameDate")
             and g.get("gameType", 2) in (2, 3)]
    if not games:
        return stats
    target = game_date
    if not target:
        upcoming = sorted(g["gameDate"] for g in games if g.get("gameState") in UPCOMING_STATES)
        target = upcoming[0] if upcoming else None
    if not target:
        return stats
    prior = sorted((g for g in games if g.get("gameState") in PLAYED_STATES and g["gameDate"] < target),
                   key=lambda g: g["gameDate"])
    if not prior:
        return stats
    stats["rest_days"] = (datetime.strptime(target, "%Y-%m-%d")
                          - datetime.strptime(prior[-1]["gameDate"], "%Y-%m-%d")).days
    stats["b2b"] = stats["rest_days"] == 1
    gf = ga = 0
    last = prior[-10:]
    for g in last:
        home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
        hs, as_ = int(home.get("score", 0) or 0), int(away.get("score", 0) or 0)
        if home.get("abbrev") == abbr:
            gf, ga = gf + hs, ga + as_
        else:
            gf, ga = gf + as_, ga + hs
    n = len(last)
    stats["recent_games"] = n
    stats["recent_gf"] = round(gf / n, 2)
    stats["recent_ga"] = round(ga / n, 2)
    base = stats.get("average_goalie_gaa")
    if n >= TEAM_FORM_MIN_GAMES:
        stats["effective_gaa"] = round(
            (1 - TEAM_FORM_WEIGHT) * base + TEAM_FORM_WEIGHT * stats["recent_ga"], 3
        ) if base is not None else stats["recent_ga"]
    return stats

def situational_factor(own_stats, opp_stats):
    """Fatigue : notre back-to-back réduit notre attaque, celui de l'adversaire l'augmente."""
    factor = 1.0
    if own_stats and own_stats.get("b2b"):
        factor *= OWN_B2B_FACTOR
    if opp_stats and opp_stats.get("b2b"):
        factor *= OPP_B2B_FACTOR
    return factor

def _effective_gaa(stats):
    if not stats:
        return None
    return stats.get("effective_gaa") if stats.get("effective_gaa") is not None else stats.get("average_goalie_gaa")

def prefetch_recent_forms(player_ids):
    """Récupère en parallèle la forme récente (résultats mis en cache), renvoie {id: forme}."""
    ids = [pid for pid in dict.fromkeys(player_ids) if pid]
    if not ids:
        return {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        forms = list(pool.map(fetch_player_recent_form, ids))
    return dict(zip(ids, forms))

def goalie_analysis(stats):
    if not stats:
        return []
    result = []
    for g in stats.get("goalies", []):
        games = int(g.get("gamesPlayed", 0) or 0)
        if games <= 0:
            continue
        first, last = g.get("firstName", {}), g.get("lastName", {})
        if isinstance(first, dict): first = first.get("default", "")
        if isinstance(last, dict): last = last.get("default", "")
        result.append({
            "name": f"{first} {last}".strip(), "games": games,
            "save_pct": g.get("savePctg"), "gaa": g.get("goalsAgainstAvg"),
            "wins": g.get("wins"), "losses": g.get("losses"), "shutouts": g.get("shutouts")
        })
    result.sort(key=lambda x: x.get("save_pct") or 0, reverse=True)
    return result[:3]

# ---------------------- PLAYER MODEL ------------------------

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def probability_to_odds(probability):
    return round(1 / probability, 2) if probability and probability > 0 else None

def regressed_shooting_pct(goals, shots):
    """% de réussite ramené vers la moyenne ligue : limite l'effet de la chance sur petit échantillon."""
    return (goals + SH_PRIOR_SHOTS * LEAGUE_SH_PCT) / (shots + SH_PRIOR_SHOTS)

def toi_factor(recent_form, season_toi_seconds):
    """Compare le temps de glace récent à celui de la saison (changement de rôle, ligne, blessures)."""
    recent_toi = recent_form.get("avg_toi_seconds") if recent_form else None
    season_toi = _toi_to_seconds(season_toi_seconds)
    if not recent_toi or not season_toi:
        return 1.0
    return clamp(recent_toi / season_toi, *TOI_FACTOR_RANGE)

def calculate_player_probability(goals, assists, games, recent_form=None,
                                 opponent_factor=1.0, team_factor=1.0, metric="goal",
                                 shots=None, season_toi_seconds=None):
    games = max(1, int(games or 1))
    season_total = goals if metric == "goal" else assists
    season_rate = season_total / games
    recent_rate = None
    if recent_form:
        recent_rate = recent_form.get("goals_per_game" if metric == "goal" else "assists_per_game")

    # Buts : modèle tirs x % de réussite régressé (plus stable que les buts bruts).
    if metric == "goal" and shots and shots > 0:
        sh_pct = regressed_shooting_pct(goals, shots)
        season_rate = (shots / games) * sh_pct
        if recent_form and recent_form.get("shots_per_game") is not None:
            recent_rate = recent_form["shots_per_game"] * sh_pct

    player_rate = 0.70 * season_rate + 0.30 * recent_rate if recent_rate is not None else season_rate
    ice_factor = toi_factor(recent_form, season_toi_seconds)
    expected_rate = max(0.0, player_rate * opponent_factor * team_factor * ice_factor)
    probability = clamp(1 - math.exp(-expected_rate), 0.001, 0.95)
    return {
        "probability": round(probability * 100, 1),
        "fair_odds": probability_to_odds(probability),
        "expected_rate": round(expected_rate, 3),
        "season_rate": round(season_rate, 3),
        "recent_rate": round(recent_rate, 3) if recent_rate is not None else None,
        "toi_factor": round(ice_factor, 3),
        "prob_2plus_goals": (round((1 - math.exp(-expected_rate) * (1 + expected_rate)) * 100, 1)
                             if metric == "goal" else None)
    }

def _player_name(p):
    first, last = p.get("firstName", {}), p.get("lastName", {})
    if isinstance(first, dict): first = first.get("default", "")
    if isinstance(last, dict): last = last.get("default", "")
    return f"{first} {last}".strip()

def calculate_security_profile(probability_pct, games, recent_form, metric):
    """Score de régularité statistique, pas une garantie de résultat."""
    games_factor = min(max(int(games or 0), 0) / 25.0, 1.0)
    if recent_form:
        recent_count = recent_form.get("goals" if metric == "goal" else "assists", 0)
        activity_factor = min(recent_count / (3.0 if metric == "goal" else 4.0), 1.0)
        if metric == "goal":
            shot_factor = min((recent_form.get("shots_per_game", 0) or 0) / 3.0, 1.0)
        else:
            shot_factor = min((recent_form.get("assists_per_game", 0) or 0) / 0.5, 1.0)
    else:
        activity_factor = 0.0
        shot_factor = 0.0
    score = (0.55 * float(probability_pct) + 20 * games_factor
             + 15 * activity_factor + 10 * shot_factor)
    score = round(clamp(score, 0, 100), 1)
    if score >= 70 and probability_pct >= 25 and games >= 15:
        level, label = "high", "Confiance statistique élevée"
    elif score >= 48 and probability_pct >= 15 and games >= 8:
        level, label = "medium", "Confiance statistique moyenne"
    else:
        level, label = "low", "Profil statistique risqué"
    return {"security_score": score, "confidence": level,
            "confidence_label": label}


def opponent_defense_factor(opponent_stats):
    """Facteur de faiblesse défensive de l'adversaire.

    Basé sur les buts encaissés (GAA moyen des gardiens adverses) : plus l'adversaire
    encaisse, plus le facteur est > 1. Remplace l'ancien calcul qui utilisait à tort
    les buts MARQUÉS par l'adversaire.
    """
    if not opponent_stats:
        return 1.0
    gaa = _effective_gaa(opponent_stats)
    if gaa is None or float(gaa) <= 0:
        return 1.0
    return clamp(float(gaa) / 3.0, 0.8, 1.2)


def goalie_adjustment(opponent_stats):
    """Ajustement modéré basé sur le % d'arrêts des gardiens adverses.

    Le GAA est déjà pris en compte par opponent_defense_factor : on n'utilise ici
    que le % d'arrêts pour éviter un double comptage.
    """
    if not opponent_stats:
        return 1.0
    save_pct = opponent_stats.get("average_goalie_save_pct")
    if save_pct is None:
        return 1.0
    # L'API NHL renvoie généralement un pourcentage décimal (ex. 0.910).
    return clamp(1.0 + (0.910 - float(save_pct)) * 1.5, 0.94, 1.06)


def fetch_nhl_players(team_str, metric="goal", opponent_factor=1.0,
                      team_factor=1.0, opponent_stats=None):
    stats = fetch_team_stats_summary(team_str)
    if not stats:
        return []
    output = []
    goalie_factor = goalie_adjustment(opponent_stats)

    # Sélection des joueurs éligibles, puis récupération parallèle de la forme récente.
    eligible = []
    for p in stats["skaters"]:
        pos = p.get("positionCode", "")
        if metric == "goal" and pos not in ("L", "R", "C", "F"):
            continue
        if metric == "assist" and pos not in ("L", "R", "C", "F", "D"):
            continue
        if int(p.get("gamesPlayed", 0) or 0) <= 0:
            continue
        eligible.append(p)
    forms = prefetch_recent_forms([p.get("playerId") or p.get("id") for p in eligible])

    for p in eligible:
        pos = p.get("positionCode", "")
        goals = int(p.get("goals", 0) or 0)
        assists = int(p.get("assists", 0) or 0)
        games = int(p.get("gamesPlayed", 0) or 0)
        player_id = p.get("playerId") or p.get("id")
        recent = forms.get(player_id)
        pred = calculate_player_probability(
            goals, assists, games, recent,
            opponent_factor * goalie_factor, team_factor, metric,
            shots=int(p.get("shots", 0) or 0),
            season_toi_seconds=p.get("avgTimeOnIcePerGame")
        )
        total = goals if metric == "goal" else assists
        label = "buts" if metric == "goal" else "passes décisives"
        recent_total = recent["goals" if metric == "goal" else "assists"] if recent else 0
        security = calculate_security_profile(pred["probability"], games, recent, metric)
        output.append({
            "player_id": player_id, "name": _player_name(p), "pos": pos,
            "stats": f"{total} {label} ({games} m)", "games_played": games,
            "recent_form": recent,
            "momentum": "🔥 Forme récente positive" if recent_total > 0 else "📊 Statistiques de saison",
            "prob": f'{pred["probability"]}%', "probability_value": pred["probability"],
            "odds": pred["fair_odds"], "fair_odds": pred["fair_odds"],
            "expected_rate": pred["expected_rate"], "toi_factor": pred["toi_factor"],
            "prob_2plus": pred["prob_2plus_goals"], **security
        })
    output.sort(key=lambda x: x["probability_value"], reverse=True)
    return output[:10]

# ---------------------- ELO ---------------------------------

def elo_win_prob(home_elo, away_elo):
    """Probabilité de victoire de l'équipe à domicile (avantage du domicile inclus)."""
    return 1.0 / (1.0 + 10 ** (-(home_elo - away_elo + ELO_HOME_ADVANTAGE) / 400.0))

def _fetch_past_schedule(abbr, season):
    return cached_get(f"sched_{abbr}_{season}",
                      lambda: nhl_get(f"club-schedule-season/{abbr}/{season}"), 86400)

def _collect_played_games(schedules):
    games = {}
    for data in schedules:
        for g in (data or {}).get("games", []) or []:
            if g.get("gameState") not in PLAYED_STATES or g.get("gameType", 2) not in (2, 3):
                continue
            home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
            if not g.get("id") or not home.get("abbrev") or not away.get("abbrev"):
                continue
            games[g["id"]] = (g.get("gameDate", ""), g["id"], home["abbrev"], away["abbrev"],
                              int(home.get("score", 0) or 0), int(away.get("score", 0) or 0))
    return sorted(games.values())

def _replay_elo(games, ratings):
    for _, _, home, away, hs, as_ in games:
        if hs == as_:
            continue
        rh = ratings.setdefault(home, ELO_START)
        ra = ratings.setdefault(away, ELO_START)
        expected = elo_win_prob(rh, ra)
        result = 1.0 if hs > as_ else 0.0
        margin = 1.0 + math.log(abs(hs - as_))   # victoire large = mise à jour un peu plus forte
        delta = ELO_K * margin * (result - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings

def build_elo_table():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        current = list(pool.map(fetch_team_schedule, ALL_TEAMS))
    games_now = _collect_played_games(current)
    ratings = {team: float(ELO_START) for team in ALL_TEAMS}

    # Saison précédente (si disponible) pour ne pas partir de zéro en début de saison.
    season_id = next((str(g.get("season")) for d in current for g in (d or {}).get("games", [])
                      if g.get("season")), None)
    if season_id and len(season_id) == 8:
        prev = f"{int(season_id[:4]) - 1}{int(season_id[4:]) - 1}"
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                past = list(pool.map(lambda t: _fetch_past_schedule(t, prev), ALL_TEAMS))
            _replay_elo(_collect_played_games(past), ratings)
            for team in list(ratings):
                ratings[team] = ELO_START + (ratings[team] - ELO_START) * ELO_SEASON_REGRESSION
        except Exception:
            logger.exception("Elo : saison précédente indisponible")
    _replay_elo(games_now, ratings)
    return {"ratings": {t: round(r, 1) for t, r in ratings.items()}, "games_used": len(games_now)}

def get_elo_table():
    return cached_get("elo_table", build_elo_table, 3600)

# ---------------------- MATCH MODEL -------------------------

def goals_pmf(mu, k):
    """Probabilité de marquer k buts : binomiale négative (légère sur-dispersion, plus réaliste que Poisson)."""
    return float(nbinom.pmf(k, GOALS_DISPERSION, GOALS_DISPERSION / (GOALS_DISPERSION + mu)))

def calculate_match_model(home_xg, away_xg, ot_home_share=0.5):
    home_xg = clamp(float(home_xg), 0.2, 8)
    away_xg = clamp(float(away_xg), 0.2, 8)
    home_pmf = [goals_pmf(home_xg, k) for k in range(16)]
    away_pmf = [goals_pmf(away_xg, k) for k in range(16)]
    home_reg = away_reg = draw = over55 = under55 = 0.0
    best_probability, best_home, best_away = -1, 0, 0
    for hg in range(16):
        for ag in range(16):
            prob = home_pmf[hg] * away_pmf[ag]
            if hg > ag: home_reg += prob
            elif ag > hg: away_reg += prob
            else: draw += prob
            if hg + ag >= 6: over55 += prob
            else: under55 += prob
            if prob > best_probability:
                best_probability, best_home, best_away = prob, hg, ag
    # Égalité en temps réglementaire -> prolongation / tirs au but, répartie selon ot_home_share
    ot_home_share = clamp(float(ot_home_share), 0.0, 1.0)
    home_final = home_reg + draw * ot_home_share
    away_final = away_reg + draw * (1 - ot_home_share)
    total = home_final + away_final or 1
    return {
        "home_win_pct_value": round(home_final / total * 100, 1),
        "away_win_pct_value": round(away_final / total * 100, 1),
        "draw_or_ot_pct_value": round(draw * 100, 1),
        "over_5_5_goals_prob_value": round(over55 * 100, 1),
        "under_5_5_goals_prob_value": round(under55 * 100, 1),
        "expected_goals_home": round(home_xg, 2),
        "expected_goals_away": round(away_xg, 2),
        "expected_total_goals": round(home_xg + away_xg, 2),
        "most_likely_score": {"home": best_home, "away": best_away}
    }

def calculate_value(probability_percent, market_odds):
    probability = probability_percent / 100
    fair_odds = probability_to_odds(probability)
    if market_odds is None or market_odds <= 1:
        return {"market_odds": None, "fair_odds": fair_odds, "value_percent": None,
                "value_detected": False, "message": "Cote bookmaker non renseignée"}
    edge = (probability - 1 / market_odds) * 100
    return {"market_odds": market_odds, "fair_odds": fair_odds,
            "value_percent": round(edge, 2), "value_detected": edge >= 3,
            "message": "Value potentielle" if edge >= 3 else "Pas de value suffisante"}

def calculate_team_expectation(team_stats, opponent_stats, home=False):
    if not team_stats or team_stats["games"] <= 0:
        return 3.0
    attack = team_stats["goals_per_game"]
    recent_gf = team_stats.get("recent_gf")
    if recent_gf is not None and team_stats.get("recent_games", 0) >= TEAM_FORM_MIN_GAMES:
        attack = (1 - TEAM_FORM_WEIGHT) * attack + TEAM_FORM_WEIGHT * recent_gf
    goalie_gaa = _effective_gaa(opponent_stats)
    defensive_factor = clamp(goalie_gaa / 3.0, 0.75, 1.25) if goalie_gaa is not None else 1.0
    home_factor = 1.04 if home else 0.96
    situation = situational_factor(team_stats, opponent_stats)
    return round(clamp(attack * defensive_factor * home_factor * situation, 1.5, 5.5), 2)

# ---------------------- PREDICTION --------------------------

class MatchRequest(BaseModel):
    league: str = "NHL"
    home_team: str = Field(min_length=2)
    away_team: str = Field(min_length=2)
    game_date: Optional[str] = None   # YYYY-MM-DD, sinon prochain match de l'équipe
    home_id: Optional[str] = None
    away_id: Optional[str] = None
    home_market_odds: Optional[float] = Field(default=None, gt=1, le=1000)
    away_market_odds: Optional[float] = Field(default=None, gt=1, le=1000)
    over_55_market_odds: Optional[float] = Field(default=None, gt=1, le=1000)

class LegacyMatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False

def build_match_prediction(req):
    home_abbr = get_nhl_abbr(req.home_team)
    away_abbr = get_nhl_abbr(req.away_team)
    if home_abbr == away_abbr:
        raise ValueError("Les deux équipes du match doivent être différentes.")
    if req.game_date:
        try:
            datetime.strptime(req.game_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("game_date invalide. Format YYYY-MM-DD.")
    home_stats = fetch_team_stats_summary(home_abbr)
    away_stats = fetch_team_stats_summary(away_abbr)
    try:
        add_team_context(home_stats, home_abbr, req.game_date)
        add_team_context(away_stats, away_abbr, req.game_date)
    except Exception:
        logger.exception("Contexte d'équipe indisponible (calendrier)")
    home_xg = calculate_team_expectation(home_stats, away_stats, True)
    away_xg = calculate_team_expectation(away_stats, home_stats, False)
    # Elo : probabilité de victoire indépendante des buts, mélangée au modèle de buts.
    elo_info = None
    elo_p = None
    try:
        table = get_elo_table()
        if table:
            home_elo, away_elo = table["ratings"].get(home_abbr), table["ratings"].get(away_abbr)
            if home_elo is not None and away_elo is not None:
                elo_p = elo_win_prob(home_elo, away_elo)
                elo_info = {"home_elo": home_elo, "away_elo": away_elo,
                            "home_win_pct": round(elo_p * 100, 1), "weight": ELO_WEIGHT,
                            "games_used": table["games_used"]}
    except Exception:
        logger.exception("Elo indisponible")
    ot_share = clamp(0.5 + (elo_p - 0.5) * OT_EDGE_SHARE, 0.45, 0.55) if elo_p is not None else 0.5
    model = calculate_match_model(home_xg, away_xg, ot_share)
    if elo_p is not None:
        model["home_win_pct_value"] = round((1 - ELO_WEIGHT) * model["home_win_pct_value"]
                                            + ELO_WEIGHT * elo_p * 100, 1)
        model["away_win_pct_value"] = round(100 - model["home_win_pct_value"], 1)

    # Facteur adverse : un attaquant domicile profite d'une défense VISITEUSE faible
    # (buts encaissés élevés), et inversement.
    home_factor = opponent_defense_factor(away_stats)
    away_factor = opponent_defense_factor(home_stats)

    # Le % d'arrêts adverse est intégré avec un poids limité (voir goalie_adjustment).
    home_team_factor = 1.04 * situational_factor(home_stats, away_stats)
    away_team_factor = 0.96 * situational_factor(away_stats, home_stats)
    home_scorers = fetch_nhl_players(home_abbr, "goal", home_factor, home_team_factor, away_stats)
    away_scorers = fetch_nhl_players(away_abbr, "goal", away_factor, away_team_factor, home_stats)
    home_playmakers = fetch_nhl_players(home_abbr, "assist", home_factor, home_team_factor, away_stats)
    away_playmakers = fetch_nhl_players(away_abbr, "assist", away_factor, away_team_factor, home_stats)
    home_goalies, away_goalies = goalie_analysis(home_stats), goalie_analysis(away_stats)

    # Top 5 MODE SÉCURITÉ : classement par score de régularité, sans promettre un résultat.
    safe = lambda players: sorted(
        players, key=lambda x: (x.get("security_score", 0), x.get("probability_value", 0)),
        reverse=True
    )[:5]
    result = {
        "version": VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "match": f"{home_abbr} vs {away_abbr}", "home_team": home_abbr, "away_team": away_abbr,
        "home_logo": f"https://assets.nhle.com/logos/nhl/svg/{home_abbr}_light.svg",
        "away_logo": f"https://assets.nhle.com/logos/nhl/svg/{away_abbr}_light.svg",
        "mode": "SECURITE",
        "match_prediction": {
            **model,
            "home_win_pct": f'{model["home_win_pct_value"]}%',
            "away_win_pct": f'{model["away_win_pct_value"]}%',
            "draw_or_ot_pct": f'{model["draw_or_ot_pct_value"]}%',
            "over_5_5_goals_prob": f'{model["over_5_5_goals_prob_value"]}%',
            "under_5_5_goals_prob": f'{model["under_5_5_goals_prob_value"]}%'
        },
        "top_goalscorers_home": home_scorers, "top_goalscorers_away": away_scorers,
        "top_playmakers_home": home_playmakers, "top_playmakers_away": away_playmakers,
        "safe_top5_goals_home": safe(home_scorers), "safe_top5_goals_away": safe(away_scorers),
        "safe_top5_assists_home": safe(home_playmakers), "safe_top5_assists_away": safe(away_playmakers),
        "goalies_home": home_goalies, "goalies_away": away_goalies,
        "roster_home": fetch_team_roster(home_abbr), "roster_away": fetch_team_roster(away_abbr),
        "team_stats": {
            "home": {"goals_per_game": home_stats["goals_per_game"], "games": home_stats["games"]} if home_stats else None,
            "away": {"goals_per_game": away_stats["goals_per_game"], "games": away_stats["games"]} if away_stats else None
        },
        "elo": elo_info,
        "context": {
            side: ({k: stats.get(k) for k in ("rest_days", "b2b", "recent_games", "recent_gf",
                                              "recent_ga", "effective_gaa")} if stats else None)
            for side, stats in (("home", home_stats), ("away", away_stats))
        },
        "goalie_adjustment": {
            "home_attack_factor": round(home_factor * goalie_adjustment(away_stats), 3),
            "away_attack_factor": round(away_factor * goalie_adjustment(home_stats), 3),
            "note": "Facteur défensif (GAA adverse) combiné à un ajustement modéré du % d'arrêts."
        },
        "value_analysis": {
            "home_win": calculate_value(model["home_win_pct_value"], req.home_market_odds),
            "away_win": calculate_value(model["away_win_pct_value"], req.away_market_odds),
            "over_5_5": calculate_value(model["over_5_5_goals_prob_value"], req.over_55_market_odds)
        },
        "data_quality": {
            "home_stats_available": home_stats is not None, "away_stats_available": away_stats is not None,
            "recent_player_form": "Données disponibles selon l'API NHL; indisponibles pour certains joueurs possibles.",
            "injuries": "Les blessures et compositions officielles ne sont pas confirmées par cette API. Vérification avant match nécessaire.",
            "safety_method": "Score statistique combinant probabilité estimée, volume de matchs et forme récente; ce n'est pas une garantie."
        }
    }
    save_prediction(result)
    return result

# ---------------------- SCHEDULE ----------------------------

def fetch_nhl_score(date_string):
    return cached_get(f"score_{date_string}", lambda: nhl_get(f"score/{date_string}"), 60)

def _localized_team_name(team_obj):
    place = team_obj.get("placeName", {})
    common = team_obj.get("commonName", {})
    if isinstance(place, dict): place = place.get("default", "")
    if isinstance(common, dict): common = common.get("default", "")
    full = f"{place} {common}".strip()
    if full:
        return full
    name = team_obj.get("name", {})
    if isinstance(name, dict): name = name.get("default", "")
    return (name or team_obj.get("abbrev", "") or "").strip()

def get_nhl_matches_for_date(date):
    try:
        requested = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date invalide. Format YYYY-MM-DD.")
    matches, seen = [], set()
    for offset in (-1, 0, 1):
        nhl_date = (requested + timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            data = fetch_nhl_score(nhl_date)
            for game in (data or {}).get("games", []):
                game_id = str(game.get("id", ""))
                start_utc = game.get("startTimeUTC")
                if not game_id or not start_utc or game_id in seen:
                    continue
                dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                local_dt = dt.astimezone(PARIS_TZ)
                if local_dt.strftime("%Y-%m-%d") != date:
                    continue
                seen.add(game_id)
                home, away = game.get("homeTeam", {}), game.get("awayTeam", {})
                state = game.get("gameState", "")
                if state in ("FINAL", "OFF"):
                    display = f'Final ({home.get("score", 0)} - {away.get("score", 0)})'
                elif state in ("LIVE", "CRIT"):
                    display = f'EN DIRECT ({home.get("score", 0)} - {away.get("score", 0)})'
                else:
                    display = local_dt.strftime("%H:%M")
                matches.append({
                    "game_id": game_id,
                    "home_team": _localized_team_name(home),
                    "away_team": _localized_team_name(away),
                    "home_abbr": home.get("abbrev", ""),
                    "away_abbr": away.get("abbrev", ""),
                    "home_logo": home.get("logo", ""),
                    "away_logo": away.get("logo", ""),
                    "game_state": display, "start_time": local_dt.isoformat()
                })
        except Exception as exc:
            logger.warning("Calendrier %s : %s", nhl_date, exc)
    matches.sort(key=lambda x: x["start_time"])
    return {"date": date, "timezone": "Europe/Paris", "matches": matches}

# ---------------------- SYNC -------------------------------

def perform_sync():
    success, errors = 0, 0
    for team in ALL_TEAMS:
        try:
            stats = fetch_team_stats_summary(team)
            roster = fetch_team_roster(team)
            if stats is not None and roster: success += 1
            else: errors += 1
        except Exception as exc:
            errors += 1
            logger.warning("Sync %s : %s", team, exc)
    status = "success" if errors == 0 else "partial"
    message = f"{success} équipes actualisées, {errors} erreurs"
    with db_connection() as db:
        db.execute("INSERT INTO sync_history (created_at,status,message) VALUES (?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), status, message))
        db.commit()
    return {"status": status, "teams_updated": success, "errors": errors,
            "message": message, "time": datetime.now(timezone.utc).isoformat()}

# (La boucle automatique background_nhl_sync a été retirée : Vercel ne
#  supporte pas les tâches de fond. Utiliser GitHub Actions + run_cron_job().)

# ---------------------- EVALUATION --------------------------

FINAL_STATES = ("FINAL", "OFF")

def _find_final_game(home_abbr, away_abbr, created_at_iso):
    """Cherche le match terminé correspondant à une prédiction (None si pas encore joué)."""
    created = datetime.fromisoformat(created_at_iso)
    for offset in (0, 1, 2):
        day = (created + timedelta(days=offset)).strftime("%Y-%m-%d")
        data = fetch_nhl_score(day)
        for game in (data or {}).get("games", []):
            if (game.get("homeTeam", {}).get("abbrev") != home_abbr
                    or game.get("awayTeam", {}).get("abbrev") != away_abbr):
                continue
            start_utc = game.get("startTimeUTC")
            if start_utc:
                start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                if start < created:
                    continue  # prédiction faite après le coup d'envoi : exclue du backtest
            if game.get("gameState") in FINAL_STATES:
                return game
    return None

def _parse_boxscore(box):
    players = {}
    stats = (box or {}).get("playerByGameStats") or {}
    for side in ("homeTeam", "awayTeam"):
        for group in ("forwards", "defense"):
            for pl in (stats.get(side) or {}).get(group, []) or []:
                pid = pl.get("playerId")
                if pid:
                    players[str(pid)] = [int(pl.get("goals", 0) or 0), int(pl.get("assists", 0) or 0)]
    return players

def evaluate_pending(limit=50):
    """Compare les prédictions passées aux résultats réels (matchs terminés uniquement)."""
    with db_connection() as db:
        rows = db.execute("""
            SELECT p.id, p.created_at, p.home_team, p.away_team
            FROM predictions p LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id
            WHERE o.prediction_id IS NULL ORDER BY p.id DESC LIMIT ?
        """, (int(clamp(limit, 1, 200)),)).fetchall()
    evaluated, pending, errors = 0, 0, 0
    for row in rows:
        try:
            game = _find_final_game(row["home_team"], row["away_team"], row["created_at"])
            if not game:
                pending += 1
                continue
            game_id = str(game.get("id"))
            box = nhl_get(f"gamecenter/{game_id}/boxscore")
            players = _parse_boxscore(box)
            with db_connection() as db:
                db.execute("""
                    INSERT INTO prediction_outcomes
                    (prediction_id, game_id, evaluated_at, home_score, away_score, players_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (prediction_id) DO UPDATE SET
                        game_id = EXCLUDED.game_id, evaluated_at = EXCLUDED.evaluated_at,
                        home_score = EXCLUDED.home_score, away_score = EXCLUDED.away_score,
                        players_json = EXCLUDED.players_json
                """, (row["id"], game_id, datetime.now(timezone.utc).isoformat(),
                      int(game.get("homeTeam", {}).get("score", 0) or 0),
                      int(game.get("awayTeam", {}).get("score", 0) or 0),
                      json.dumps(players)))
                db.commit()
            evaluated += 1
        except Exception as exc:
            errors += 1
            logger.warning("Évaluation prédiction %s : %s", row["id"], exc)
    return {"evaluated": evaluated, "pending": pending, "errors": errors}

def _calibration(pairs):
    """pairs = [(probabilité 0-1, résultat 0/1)] -> Brier, taux de réussite et calibration par tranche."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    base = sum(y for _, y in pairs) / n
    buckets = []
    for low, high in ((0, .15), (.15, .25), (.25, .35), (.35, 1.01)):
        sel = [(p, y) for p, y in pairs if low <= p < high]
        if sel:
            buckets.append({
                "range": f"{int(low * 100)}-{min(int(high * 100), 100)}%", "n": len(sel),
                "avg_predicted_pct": round(sum(p for p, _ in sel) / len(sel) * 100, 1),
                "actual_pct": round(sum(y for _, y in sel) / len(sel) * 100, 1)
            })
    return {"n": n, "brier": round(brier, 4), "brier_constant_baseline": round(base * (1 - base), 4),
            "actual_rate_pct": round(base * 100, 1), "buckets": buckets}

def compute_metrics():
    """Métriques sur les matchs évalués (une seule prédiction par match : la plus récente)."""
    with db_connection() as db:
        rows = db.execute("""
            SELECT p.id, p.prediction_json, o.game_id, o.home_score, o.away_score, o.players_json
            FROM prediction_outcomes o JOIN predictions p ON p.id = o.prediction_id
            ORDER BY p.id DESC
        """).fetchall()
    latest = {}
    for row in rows:
        latest.setdefault(row["game_id"], row)

    winner, over55 = [], []
    pools = {"goal": [], "assist": [], "safe_goal": [], "safe_assist": []}
    for row in latest.values():
        pred = json.loads(row["prediction_json"])
        mp = pred.get("match_prediction", {})
        actual = json.loads(row["players_json"])
        if mp.get("home_win_pct_value") is not None:
            winner.append((mp["home_win_pct_value"] / 100, 1 if row["home_score"] > row["away_score"] else 0))
        if mp.get("over_5_5_goals_prob_value") is not None:
            over55.append((mp["over_5_5_goals_prob_value"] / 100,
                           1 if row["home_score"] + row["away_score"] >= 6 else 0))
        for side in ("home", "away"):
            for key, metric, idx in (("top_goalscorers", "goal", 0), ("top_playmakers", "assist", 1)):
                for pl in pred.get(f"{key}_{side}", []) or []:
                    stat = actual.get(str(pl.get("player_id")))
                    if stat is not None:  # joueur absent du match : ignoré
                        pools[metric].append((pl["probability_value"] / 100, 1 if stat[idx] >= 1 else 0))
            for key, metric, idx in (("safe_top5_goals", "safe_goal", 0), ("safe_top5_assists", "safe_assist", 1)):
                for pl in pred.get(f"{key}_{side}", []) or []:
                    stat = actual.get(str(pl.get("player_id")))
                    if stat is not None:
                        pools[metric].append((pl["probability_value"] / 100, 1 if stat[idx] >= 1 else 0))

    def hit_rate(pairs):
        return {"n": len(pairs), "hit_rate_pct": round(sum(y for _, y in pairs) / len(pairs) * 100, 1)} if pairs else {"n": 0}

    return {
        "games_evaluated": len(latest),
        "match_winner_home": {**_calibration(winner),
                              "accuracy_pct": round(sum(1 for p, y in winner if (p >= .5) == bool(y)) / len(winner) * 100, 1) if winner else None},
        "over_5_5": _calibration(over55),
        "goalscorer_calibration": _calibration(pools["goal"]),
        "playmaker_calibration": _calibration(pools["assist"]),
        "safe_top5_goals": hit_rate(pools["safe_goal"]),
        "safe_top5_assists": hit_rate(pools["safe_assist"]),
        "notes": "Brier plus bas = mieux. Comparer à brier_constant_baseline : en dessous, le modèle apporte de l'information. "
                 "Il faut plusieurs centaines de joueurs évalués avant de tirer une conclusion."
    }

# ---------------------- AUTOMATISATION ----------------------

JOB_LOCK = threading.Lock()
JOB_STATE = {"running": False, "last_start": None, "last_end": None, "last_summary": None}

def predict_todays_games(hours_ahead=AUTO_PREDICT_HOURS_AHEAD):
    """Prédit les matchs qui commencent dans les `hours_ahead` prochaines heures, une seule fois par match."""
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=hours_ahead)
    days = sorted({now.astimezone(PARIS_TZ).strftime("%Y-%m-%d"),
                   horizon.astimezone(PARIS_TZ).strftime("%Y-%m-%d")})
    created = skipped = errors = 0
    for day in days:
        for match in get_nhl_matches_for_date(day)["matches"]:
            try:
                start = datetime.fromisoformat(match["start_time"]).astimezone(timezone.utc)
                home, away = match.get("home_abbr"), match.get("away_abbr")
                if not home or not away or not (now < start <= horizon):
                    continue
                with db_connection() as db:
                    recent = db.execute(
                        "SELECT created_at FROM predictions WHERE home_team=? AND away_team=? "
                        "ORDER BY id DESC LIMIT 10", (home, away)
                    ).fetchall()
                window_start = start - timedelta(hours=24)
                if any(window_start <= datetime.fromisoformat(r["created_at"]) < start for r in recent):
                    skipped += 1
                    continue
                build_match_prediction(MatchRequest(home_team=home, away_team=away))
                created += 1
            except Exception:
                errors += 1
                logger.exception("Prédiction automatique %s", match.get("game_id"))
    return {"created": created, "already_done": skipped, "errors": errors}

def _last_sync_age_hours():
    with db_connection() as db:
        row = db.execute("SELECT created_at FROM sync_history ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])).total_seconds() / 3600

def run_cron_job():
    """Tâche complète : synchro (si ancienne), Elo, prédictions du jour, évaluation des résultats."""
    if not JOB_LOCK.acquire(blocking=False):
        return False
    summary = {}
    JOB_STATE.update(running=True, last_start=datetime.now(timezone.utc).isoformat())
    try:
        age = _last_sync_age_hours()
        if age is None or age >= SYNC_MIN_INTERVAL_HOURS:
            summary["sync"] = perform_sync()["status"]
        summary["elo"] = bool(get_elo_table())
        summary["predictions"] = predict_todays_games()
        summary["evaluation"] = evaluate_pending(100)
    except Exception as exc:
        logger.exception("Erreur tâche automatique")
        summary["error"] = str(exc)
    finally:
        JOB_STATE.update(running=False, last_end=datetime.now(timezone.utc).isoformat(), last_summary=summary)
        JOB_LOCK.release()
    return True

# ---------------------- FASTAPI -----------------------------

try:
    init_database()
except Exception:
    logger.exception("Initialisation base de données")

app = FastAPI(
    title="NHL GoalScorer Analytics API", version=VERSION,
    description="Analyse NHL : buteurs, passeurs, gardiens, xG, probabilités et historique."
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"""<!doctype html><html lang="fr"><meta charset="utf-8">
        <title>NHL IA {VERSION}</title><body style="font-family:Arial;padding:30px">
        <h2>NHL GoalScorer Analytics {VERSION}</h2><p>API opérationnelle.</p>
        <a href="/docs">Documentation API</a></body></html>"""

@app.get("/api/health")
def health():
    return {"status": "online", "version": VERSION, "database": "postgresql" if USE_PG else DB_FILE,
            "time": datetime.now(timezone.utc).isoformat()}

@app.get("/api/matches/nhl")
def api_nhl_matches(date: str):
    return get_nhl_matches_for_date(date)

@app.post("/predict/goalscorers")
def predict_goalscorers(req: MatchRequest):
    if req.league.upper() != "NHL":
        raise HTTPException(status_code=400, detail="Cette route est réservée à la NHL.")
    try:
        return build_match_prediction(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.RequestException as exc:
        logger.exception("API NHL indisponible")
        raise HTTPException(status_code=503, detail=f"API NHL indisponible : {exc}")
    except Exception:
        logger.exception("Erreur prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne pendant la prédiction.")

@app.post("/predict/match")
def predict_match_legacy(req: LegacyMatchRequest):
    home_xg, away_xg = req.home_expected_goals, req.away_expected_goals
    if req.home_b2b: home_xg *= 0.92
    if req.away_b2b: away_xg *= 0.92
    model = calculate_match_model(home_xg, away_xg)
    return {
        "match": f"{req.home_team} vs {req.away_team}", "home_team": req.home_team,
        "away_team": req.away_team, "prediction": model, "match_prediction": model,
        "version": VERSION
    }

@app.get("/api/history")
def get_history(limit: int = 50):
    limit = int(clamp(limit, 1, 500))
    with db_connection() as db:
        rows = db.execute(
            "SELECT id, created_at, match_name, home_team, away_team, home_xg, away_xg, "
            "home_win, away_win, over_55 FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"count": len(rows), "predictions": [dict(row) for row in rows]}

@app.get("/api/history/{prediction_id}")
def get_history_detail(prediction_id: int):
    with db_connection() as db:
        row = db.execute("SELECT prediction_json FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Prédiction introuvable")
    return json.loads(row["prediction_json"])

@app.get("/api/predict/auto-sync")
def auto_sync():
    try:
        return perform_sync()
    except Exception:
        logger.exception("Erreur synchronisation")
        raise HTTPException(status_code=500, detail="Erreur pendant la synchronisation NHL.")

@app.get("/api/sync/history")
def get_sync_history():
    with db_connection() as db:
        rows = db.execute("SELECT * FROM sync_history ORDER BY id DESC LIMIT 30").fetchall()
    return {"synchronizations": [dict(row) for row in rows]}

@app.get("/api/cron")
def api_cron(key: Optional[str] = None):
    """À appeler par le cron (toutes les 10 min) : répond tout de suite, la tâche tourne en arrière-plan."""
    if CRON_SECRET and not (key and secrets.compare_digest(key, CRON_SECRET)):
        raise HTTPException(status_code=403, detail="Clé invalide.")
    if JOB_LOCK.locked():
        return {"status": "already_running", **JOB_STATE}
    threading.Thread(target=run_cron_job, daemon=True).start()
    return {"status": "started"}

@app.get("/api/cron/status")
def api_cron_status():
    return JOB_STATE

@app.get("/api/elo")
def api_elo():
    """Classement Elo des équipes (calculé sur la saison en cours + saison précédente)."""
    table = get_elo_table()
    if not table:
        raise HTTPException(status_code=503, detail="Classement Elo indisponible.")
    ranking = sorted(table["ratings"].items(), key=lambda kv: kv[1], reverse=True)
    return {"games_used": table["games_used"],
            "ranking": [{"team": t, "elo": r} for t, r in ranking]}

@app.get("/api/evaluate")
def api_evaluate(limit: int = 50):
    """Évalue les prédictions dont le match est terminé."""
    return evaluate_pending(limit)

@app.get("/api/metrics")
def api_metrics():
    """Précision et calibration du modèle sur les matchs déjà évalués."""
    return compute_metrics()

@app.get("/api/football/matches")
def football_matches(date: Optional[str] = None):
    return {
        "date": date, "matches": [], "status": "football_not_configured",
        "message": "Le module football n'est pas encore raccordé à une source de données."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
