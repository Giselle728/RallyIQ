"""
RallyIQ FastAPI REST API
Serves all 5 Big Data Tennis Analytics features:
1. Player Intelligence Lab
2. Serve Strategy Heatmap
3. Similar Playstyle Recommender (Cosine Similarity)
4. ATP Rivalry Network (Louvain Community Detection & PageRank)
5. ML Match Winner Predictor (Supervised Classification)
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import (
    DATASET_METRICS_REPORT,
    PROCESSED_PLAYER_STATS,
    PROCESSED_SERVE_STRATEGY
)
from backend.storage.mongo_store import mongo_store
from backend.algorithms.recommender import recommender
from backend.algorithms.graph_analytics import rivalry_network
from backend.ml.predictor import predictor

class MatchPredictionRequest(BaseModel):
    player_a: str = Field(..., description="First ATP player name")
    player_b: str = Field(..., description="Second ATP player name")
    surface: str = Field("Hard", description="Surface: Hard, Clay, Grass")
    best_of: int = Field(3, description="Best of 3 or 5 sets")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rallyiq.api")

app = FastAPI(
    title="RallyIQ: Big Data Tennis Intelligence Platform",
    description="Real ATP tennis big data analytics pipeline with PySpark, MongoDB, Machine Learning, and Graph Analytics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load datasets on startup
player_profiles_cache: Dict[str, Any] = {}
serve_strategy_cache: Dict[str, Any] = {}
dataset_report_cache: Dict[str, Any] = {}

def load_api_caches():
    global player_profiles_cache, serve_strategy_cache, dataset_report_cache
    
    # 1. Dataset Report
    if os.path.exists(DATASET_METRICS_REPORT):
        with open(DATASET_METRICS_REPORT, "r") as f:
            dataset_report_cache = json.load(f)
            
    # 2. Player Profiles
    if os.path.exists(PROCESSED_PLAYER_STATS):
        with open(PROCESSED_PLAYER_STATS, "r") as f:
            player_profiles_cache = json.load(f)
            
    # 3. Serve Strategy
    if os.path.exists(PROCESSED_SERVE_STRATEGY):
        with open(PROCESSED_SERVE_STRATEGY, "r") as f:
            serve_strategy_cache = json.load(f)
            
    logger.info("RallyIQ API data caches initialized.")

# Load immediately on import
load_api_caches()

@app.on_event("startup")
def startup_event():
    load_api_caches()

# =============================================================================
# 1. SYSTEM OVERVIEW & METRICS
# =============================================================================
@app.get("/api/overview")
@app.get("/api/dashboard/overview")
def get_overview():
    """Return dataset size, match/point counts, engine stats, and pipeline status."""
    mongo_stats = mongo_store.get_stats()
    mongo_health = mongo_store.test_connection()
    surfaces = dataset_report_cache.get("surfaces", {"Hard": 4851, "Clay": 1877, "Grass": 837})
    total_matches = dataset_report_cache.get("match_count", 7565)
    total_points = dataset_report_cache.get("point_count", 904128)
    raw_size = dataset_report_cache.get("raw_size_mb", 117.0)
    ml_status = predictor.get_status()
    
    return {
        "platform": "RallyIQ: Big Data Tennis Intelligence Platform",
        "dataset_report": dataset_report_cache,
        "raw_size_mb": raw_size,
        "processed_size_mb": dataset_report_cache.get("processed_size_mb", 12.17),
        "total_matches": total_matches,
        "total_points": total_points,
        "surfaces": surfaces,
        "storage": mongo_stats,
        "storage_health": mongo_health,
        "ml_predictor": ml_status,
        "engine_architecture": {
            "ingestion": "Python 3.10 Pandas (Schema validation, deduplication, type casting)",
            "batch_processing": "Apache PySpark local[2] (Distributed joins & aggregations)",
            "storage_layer": mongo_stats["engine"],
            "machine_learning": "Supervised Random Forest Classifier with Zero-Leakage Pre-Match Features",
            "similarity": "Normalized Multi-Feature Cosine Similarity",
            "graph_analytics": "Louvain Community Detection & PageRank Centrality"
        }
    }

@app.get("/api/dashboard/surfaces")
def get_dashboard_surfaces():
    """Return surface counts and distribution breakdown."""
    surfaces = dataset_report_cache.get("surfaces", {"Hard": 4851, "Clay": 1877, "Grass": 837})
    total_matches = dataset_report_cache.get("match_count", 7565)
    breakdown = []
    for s_name, count in surfaces.items():
        breakdown.append({
            "surface": s_name,
            "count": count,
            "percentage": round((count / max(1, total_matches)) * 100.0, 1)
        })
    return {
        "total_matches": total_matches,
        "surfaces": surfaces,
        "breakdown": breakdown
    }

@app.get("/api/dashboard/top-players")
def get_dashboard_top_players(limit: int = 10):
    """Return top ATP players sorted by matches played."""
    players = []
    for p_name, data in player_profiles_cache.items():
        players.append({
            "player": p_name,
            "total_matches": data.get("total_matches", 0),
            "overall": data.get("overall", {}),
            "surfaces": list(data.get("surfaces", {}).keys())
        })
    players.sort(key=lambda x: x["total_matches"], reverse=True)
    return {"top_players": players[:limit]}

@app.get("/api/matches")
def get_matches(
    limit: int = Query(50, ge=1, le=500, description="Max matches to return (1-500)"),
    offset: int = Query(0, ge=0, description="Pagination offset (must be >= 0)"),
    surface: Optional[str] = None,
    player: Optional[str] = None
):
    """Return paginated match records from clean_matches dataset."""
    query = {}
    if surface:
        query["Surface"] = surface
    if player:
        query["$or"] = [{"Player 1": player}, {"Player 2": player}]
        
    cursor = mongo_store.db.matches.find(query, {"_id": 0}).skip(offset).limit(limit)
    matches_list = list(cursor)
    total_count = mongo_store.db.matches.count_documents(query)
    
    return {
        "matches": matches_list,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }

# =============================================================================
# 2. FEATURE 1: PLAYER INTELLIGENCE LAB
# =============================================================================
@app.get("/api/players")
def list_players(query: Optional[str] = None, min_matches: int = 3):
    """List qualified players with match counts."""
    players = []
    for p_name, data in player_profiles_cache.items():
        tot = data.get("total_matches", 0)
        if tot >= min_matches:
            if not query or query.lower() in p_name.lower():
                players.append({
                    "player": p_name,
                    "total_matches": tot,
                    "surfaces": list(data.get("surfaces", {}).keys())
                })
    players.sort(key=lambda x: x["total_matches"], reverse=True)
    return {"players": players}

@app.get("/api/player-lab")
def compare_players(
    player_a: str = Query(..., description="First ATP player"),
    player_b: str = Query(..., description="Second ATP player"),
    surface: str = Query("Overall", description="Surface: Overall, Hard, Clay, Grass")
):
    """Compare two ATP players across surface, stats, and head-to-head records."""
    prof_a = player_profiles_cache.get(player_a)
    prof_b = player_profiles_cache.get(player_b)
    
    if not prof_a:
        raise HTTPException(status_code=404, detail=f"Player '{player_a}' not found.")
    if not prof_b:
        raise HTTPException(status_code=404, detail=f"Player '{player_b}' not found.")
        
    def get_stats_for_surface(prof, s_name):
        if s_name == "Overall" or s_name not in prof.get("surfaces", {}):
            return prof.get("overall", {})
        return prof.get("surfaces", {}).get(s_name, {})
        
    stats_a = get_stats_for_surface(prof_a, surface)
    stats_b = get_stats_for_surface(prof_b, surface)
    
    # Query H2H details
    h2h_data = rivalry_network.get_rivalry_details(player_a, player_b)
    
    return {
        "player_a": {
            "name": player_a,
            "total_matches": prof_a.get("total_matches", 0),
            "stats": stats_a
        },
        "player_b": {
            "name": player_b,
            "total_matches": prof_b.get("total_matches", 0),
            "stats": stats_b
        },
        "surface": surface,
        "head_to_head": h2h_data
    }

# =============================================================================
# 3. FEATURE 2: SERVE STRATEGY HEATMAP
# =============================================================================
@app.get("/api/serve-strategy")
def get_serve_strategy(
    player: str = Query(..., description="Player name"),
    surface: str = Query("Hard", description="Court surface: Hard, Clay, Grass")
):
    """Return court zone placement counts, percentages, and fault distribution."""
    prof = serve_strategy_cache.get(player)
    if not prof:
        raise HTTPException(status_code=404, detail=f"No serve data found for player '{player}'.")
        
    surf_data = prof.get(surface)
    if not surf_data:
        # Fall back to any available surface
        surf_data = next(iter(prof.values()), None)
        surface = next(iter(prof.keys()), "Hard")
        
    return {
        "player": player,
        "surface": surface,
        "available_surfaces": list(prof.keys()),
        "strategy": surf_data
    }

# =============================================================================
# 4. FEATURE 3: SIMILAR PLAYSTYLE RECOMMENDER
# =============================================================================
@app.get("/api/recommender")
def get_recommendations(
    player: str = Query(..., description="Target player name"),
    top_k: int = Query(5, ge=1, le=50, description="Number of recommendations (1-50)")
):
    """Find players with most similar statistical playstyle using Cosine Similarity."""
    player = player.strip()
    if not player:
        raise HTTPException(status_code=422, detail="player must not be empty.")
    res = recommender.find_similar_players(player, top_k=top_k)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

# =============================================================================
# 5. FEATURE 4: ATP RIVALRY NETWORK
# =============================================================================
@app.get("/api/rivalry-network")
def get_rivalry_network(
    min_meetings: int = Query(3,  ge=1, description="Minimum meetings for edge inclusion"),
    top_n: int = Query(35,ge=1, description="Top N players to display")
):
    """Return Louvain communities, PageRank centrality, and rivalry graph data."""
    return rivalry_network.get_network_data(min_meetings=min_meetings, top_n_nodes=top_n)

@app.get("/api/rivalry-details")
def get_rivalry_details(
    player_a: str = Query(..., description="First player"),
    player_b: str = Query(..., description="Second player")
):
    """Return head-to-head meeting history and rivalry stats."""
    res = rivalry_network.get_rivalry_details(player_a, player_b)
    if not res:
        raise HTTPException(status_code=404, detail=f"No rivalry record found between {player_a} and {player_b}.")
    return res

# =============================================================================
# 6. STORAGE HEALTH CHECK & MONGO DIAGNOSTICS
# =============================================================================
@app.get("/api/health/mongo")
@app.get("/api/health/storage")
def get_mongo_health():
    """Verify live MongoDB Atlas connectivity, ping, and cluster status safely."""
    return mongo_store.test_connection()

# =============================================================================
# 7. FEATURE 5: MACHINE LEARNING MATCH WINNER PREDICTION
# =============================================================================
@app.get("/api/predict/status")
def get_predictor_status():
    """Return ML model metadata, evaluation metrics, and operational readiness."""
    return predictor.get_status()

@app.get("/api/predict/evaluation")
def get_predictor_evaluation():
    """Return validated Google Colab evaluation report and production predictor metadata."""
    return predictor.get_evaluation_report()

@app.get("/api/predict/match")
def predict_match_get(
    player_a: str = Query(..., description="First ATP player name"),
    player_b: str = Query(..., description="Second ATP player name"),
    surface: str = Query("Hard", description="Surface: Hard, Clay, Grass"),
    best_of: int = Query(3, description="Best of 3 or 5 sets")
):
    """Predict ATP match winner and probability estimates using trained ML model."""
    try:
        return predictor.predict(player_a, player_b, surface=surface, best_of=best_of)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/match")
def predict_match_post(payload: MatchPredictionRequest):
    """Predict ATP match winner via JSON payload."""
    try:
        return predictor.predict(
            payload.player_a, payload.player_b, surface=payload.surface, best_of=payload.best_of
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# =============================================================================
# FRONTEND & EVALUATION REPORT STATIC FILES
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
REPORTS_ML_EVAL_DIR = os.path.join(PROJECT_ROOT, "reports", "ml_evaluation")

if os.path.exists(REPORTS_ML_EVAL_DIR):
    app.mount("/reports/ml_evaluation", StaticFiles(directory=REPORTS_ML_EVAL_DIR), name="ml_evaluation")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RallyIQ API active. Frontend index.html not yet installed."}
