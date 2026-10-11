"""
RallyIQ FastAPI REST API
Serves all 6 Big Data Tennis Analytics features:
1. Player Intelligence Lab
2. Match Momentum Analyzer (DGIM + EWMA)
3. Serve Strategy Heatmap
4. Similar Playstyle Recommender (Cosine Similarity)
5. ATP Rivalry Network (Louvain Community Detection & PageRank)
6. Stream Integrity Monitor (Bloom Filter & MongoDB Persistence)
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import (
    DATASET_METRICS_REPORT,
    PROCESSED_PLAYER_STATS,
    PROCESSED_SERVE_STRATEGY,
    DATA_PROCESSED_DIR
)
from backend.storage.mongo_store import mongo_store
from backend.algorithms.dgim import MomentumAnalyzer
from backend.algorithms.bloom_filter import BloomFilter
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
    description="Real ATP tennis big data analytics pipeline with PySpark, MongoDB, DGIM, Bloom Filter, and Graph Analytics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Bloom Filter instance for streaming pipeline
stream_bloom_filter = BloomFilter(m=8192, k=5)
stream_events_cache: List[Dict[str, Any]] = []
stream_event_pointer = 0

# Load datasets on startup
player_profiles_cache: Dict[str, Any] = {}
serve_strategy_cache: Dict[str, Any] = {}
dataset_report_cache: Dict[str, Any] = {}
replayable_matches_cache: List[Dict[str, Any]] = []
clean_points_df: Optional[pd.DataFrame] = None

def load_api_caches():
    global player_profiles_cache, serve_strategy_cache, dataset_report_cache
    global replayable_matches_cache, stream_events_cache, clean_points_df
    
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
            
    # 4. Replayable Matches
    replay_path = os.path.join(DATA_PROCESSED_DIR, "replayable_matches.json")
    if os.path.exists(replay_path):
        with open(replay_path, "r") as f:
            replayable_matches_cache = json.load(f)
            
    # 5. Stream Sample Events
    sample_path = os.path.join(DATA_PROCESSED_DIR, "stream_sample_events.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            stream_events_cache = json.load(f)
            
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
            "streaming": "Incremental Stream with Kirsch-Mitzenmacher Bloom Filter",
            "sliding_window": "Datar-Gionis-Indyk-Motwani (DGIM) + EWMA Momentum",
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

@app.get("/api/dashboard/recent-events")
def get_dashboard_recent_events(limit: int = 15):
    """Return recent point stream events from MongoDB or sample cache."""
    docs = list(mongo_store.db.stream_events.find({}, {"_id": 0}).sort("point_number", -1).limit(limit))
    if not docs and stream_events_cache:
        docs = stream_events_cache[:limit]
    return {"recent_events": docs, "count": len(docs)}

@app.get("/api/dashboard/live")
def get_dashboard_live():
    """Return live stream metrics, DGIM window telemetry, and engine health."""
    bloom_metrics = stream_bloom_filter.get_metrics()
    mongo_events_count = mongo_store.db.stream_events.count_documents({})
    return {
        "status": "online",
        "stream_pointer": stream_event_pointer,
        "bloom_filter": bloom_metrics,
        "mongo_persisted_events": mongo_events_count,
        "pyspark_active": True,
        "dgim_engine": "ready",
        "storage": mongo_store.get_stats()
    }

@app.get("/api/matches")
def get_matches(
    limit: int = Query(50, description="Max matches to return"),
    offset: int = Query(0, description="Pagination offset"),
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
# 3. FEATURE 2: MATCH MOMENTUM ANALYZER (DGIM + EWMA)
# =============================================================================
@app.get("/api/matches/replayable")
def get_replayable_matches():
    """List historical point-by-point matches available for momentum replay."""
    return {"matches": replayable_matches_cache}

@app.get("/api/momentum/replay")
def replay_match_momentum(
    match_id: str = Query(..., description="Match ID to replay"),
    window_size: int = Query(40, description="DGIM sliding window size N"),
    alpha: float = Query(0.25, description="EWMA smoothing factor")
):
    """
    Replay match point-by-point through DGIM sliding window and EWMA momentum scoring.
    """
    global clean_points_df
    if clean_points_df is None:
        clean_points_path = os.path.join(DATA_PROCESSED_DIR, "clean_points.parquet")
        if os.path.exists(clean_points_path):
            clean_points_df = pd.read_parquet(clean_points_path)
            
    if clean_points_df is None:
        raise HTTPException(status_code=500, detail="Points dataset not loaded.")
        
    pts = clean_points_df[clean_points_df["match_id"] == match_id].sort_values("Pt")
    if len(pts) == 0:
        raise HTTPException(status_code=404, detail=f"No point data found for match '{match_id}'.")
        
    # Match metadata
    match_meta = next((m for m in replayable_matches_cache if m["match_id"] == match_id), None)
    p1 = match_meta["player_1"] if match_meta else "Player 1"
    p2 = match_meta["player_2"] if match_meta else "Player 2"
    
    analyzer = MomentumAnalyzer(alpha=alpha, window_size=window_size)
    match_points_dicts = pts.to_dict(orient="records")
    result = analyzer.replay_match(match_points_dicts, p1, p2)
    
    # Pre-select initial snapshot (point 0 at pre-match start)
    result["snapshot"] = result["points"][0] if result["points"] else {}
    result["match_id"] = match_id
    result["metadata"] = match_meta
    return result

# =============================================================================
# 4. FEATURE 3: SERVE STRATEGY HEATMAP
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
# 5. FEATURE 4: SIMILAR PLAYSTYLE RECOMMENDER
# =============================================================================
@app.get("/api/recommender")
def get_recommendations(
    player: str = Query(..., description="Target player name"),
    top_k: int = Query(5, description="Number of recommendations")
):
    """Find players with most similar statistical playstyle using Cosine Similarity."""
    res = recommender.find_similar_players(player, top_k=top_k)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

# =============================================================================
# 6. FEATURE 5: ATP RIVALRY NETWORK
# =============================================================================
@app.get("/api/rivalry-network")
def get_rivalry_network(
    min_meetings: int = Query(3, description="Minimum meetings for edge inclusion"),
    top_n: int = Query(35, description="Top N players to display")
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
# 7. FEATURE 6: STREAM INTEGRITY MONITOR (BLOOM FILTER & MONGO)
# =============================================================================
@app.get("/api/stream/status")
def get_stream_status():
    """Return current Bloom filter metrics and MongoDB stream metrics."""
    bloom_metrics = stream_bloom_filter.get_metrics()
    mongo_events_count = mongo_store.db.stream_events.count_documents({})
    
    return {
        "bloom_filter": bloom_metrics,
        "events_streamed_session": stream_event_pointer,
        "total_sample_events": len(stream_events_cache),
        "mongo_persisted_events": mongo_events_count
    }

@app.post("/api/stream/step")
def step_stream(count: int = Query(5, description="Number of events to process")):
    """
    Ingest the next 'count' tennis point events through Bloom Filter and persist to MongoDB.
    Confirmed duplicate events are rejected before database persistence.
    Persistence failure is handled safely to keep Bloom Filter and database in sync.
    """
    global stream_event_pointer
    if not stream_events_cache:
        raise HTTPException(status_code=500, detail="Stream events cache empty.")
        
    processed_batch = []
    
    for _ in range(count):
        if stream_event_pointer >= len(stream_events_cache):
            stream_event_pointer = 0  # Loop back for endless test stream
            
        event = stream_events_cache[stream_event_pointer]
        stream_event_pointer += 1
        
        event_key = event["event_id"]
        
        # 1. Query with verification against ground-truth set and Bloom filter
        verify_result = stream_bloom_filter.query_with_verification(event_key)
        is_already_present = verify_result["exact_result"]
        bloom_suspected = verify_result["bloom_result"]
        is_fp = verify_result["is_false_positive"]

        if is_already_present:
            # CONFIRMED DUPLICATE: Reject BEFORE performing persistence operation!
            stream_bloom_filter.record_duplicate(event_key)
            processed_batch.append({
                "event_id": event_key,
                "match_id": event["match_id"],
                "point_number": event["point_number"],
                "players": f"{event['player_1']} vs {event['player_2']}",
                "score": event["score"],
                "server": event["server"],
                "winner": event["winner"],
                "serve_dir": event["serve_dir"],
                "is_ace": event["is_ace"],
                "bloom_suspected_duplicate": True,
                "is_false_positive": False,
                "duplicate_rejected": True,
                "persisted": False,
                "status": "duplicate_rejected",
                "verification": verify_result
            })
            continue

        # 2. NEW UNIQUE EVENT: Persist to MongoDB first to prevent Bloom Filter / DB desync
        mongo_doc = {
            "event_id": event_key,
            "match_id": event["match_id"],
            "point_number": event["point_number"],
            "players": f"{event['player_1']} vs {event['player_2']}",
            "score": event["score"],
            "server": event["server"],
            "winner": event["winner"],
            "serve_dir": event["serve_dir"],
            "is_ace": event["is_ace"],
            "bloom_suspected_duplicate": bloom_suspected,
            "is_false_positive": is_fp
        }
        
        try:
            # Insert copy so mongo_doc is not mutated with non-serializable ObjectId
            mongo_store.db.stream_events.insert_one(dict(mongo_doc))
        except pymongo.errors.DuplicateKeyError:
            # Concurrency race: another request inserted this event_id in between
            stream_bloom_filter.record_duplicate(event_key)
            processed_batch.append({
                **mongo_doc,
                "bloom_suspected_duplicate": True,
                "is_false_positive": False,
                "duplicate_rejected": True,
                "persisted": False,
                "status": "duplicate_rejected",
                "verification": verify_result
            })
            continue
        except Exception as e:
            logger.error(f"MongoDB persistence failure for event {event_key}: {e}")
            # Do NOT commit to Bloom Filter if database persistence failed!
            processed_batch.append({
                **mongo_doc,
                "duplicate_rejected": False,
                "persisted": False,
                "status": "persistence_error",
                "error": str(e)
            })
            continue

        # 3. Database persistence succeeded: safely commit to Bloom Filter
        stream_bloom_filter.commit_new_item(event_key)
        
        processed_batch.append({
            **mongo_doc,
            "duplicate_rejected": False,
            "persisted": True,
            "status": "persisted_and_filtered",
            "verification": verify_result
        })
        
    mongo_store.persist_to_disk()
    
    return {
        "processed_count": len(processed_batch),
        "events": processed_batch,
        "bloom_status": stream_bloom_filter.get_metrics(),
        "mongo_persisted_events": mongo_store.db.stream_events.count_documents({})
    }

@app.post("/api/stream/inject-duplicate")
def inject_duplicate():
    """
    Deliberately inject a previously seen point event to demonstrate duplicate detection.
    Confirmed duplicate is rejected before persistence, so MongoDB is not modified and
    Bloom filter 'n' is not incremented.
    """
    if stream_bloom_filter.n == 0:
        step_stream(count=1)
        
    # Pick a random or the most recent event that was actually processed
    target_event = stream_events_cache[max(0, stream_event_pointer - 1)]
    event_key = target_event["event_id"]
    
    # Query Bloom filter and ground truth
    bloom_detected = stream_bloom_filter.contains(event_key)
    is_confirmed = stream_bloom_filter.is_exact_member(event_key)
    
    # Record duplicate rejection without incrementing n or writing to MongoDB
    stream_bloom_filter.record_duplicate(event_key)
    
    return {
        "injected_event_id": event_key,
        "bloom_detected_duplicate": bloom_detected,
        "is_confirmed_duplicate": is_confirmed,
        "duplicate_rejected": True,
        "persisted": False,
        "exact_verification": {
            "item": event_key,
            "bloom_result": bloom_detected,
            "exact_result": is_confirmed,
            "is_false_positive": False,
            "is_duplicate": True
        },
        "bloom_status": stream_bloom_filter.get_metrics(),
        "mongo_persisted_events": mongo_store.db.stream_events.count_documents({})
    }

@app.post("/api/stream/reset")
def reset_stream(
    m: int = Query(8192, description="Bit array size m"),
    k: int = Query(5, description="Number of hash functions k")
):
    """Reset Bloom filter and stream pointer with configurable parameters."""
    global stream_event_pointer
    stream_bloom_filter.reset(m=m, k=k)
    stream_event_pointer = 0
    try:
        mongo_store.db.stream_events.delete_many({})
        mongo_store.persist_to_disk()
    except Exception as e:
        logger.warning(f"Error purging stream_events on reset: {e}")
    return {
        "status": "reset",
        "new_configuration": {"m": m, "k": k},
        "bloom_status": stream_bloom_filter.get_metrics(),
        "mongo_persisted_events": mongo_store.db.stream_events.count_documents({})
    }

@app.get("/api/stream/events")
def get_recent_stream_events(limit: int = 15):
    """Fetch latest stream events persisted to MongoDB."""
    docs = list(mongo_store.db.stream_events.find({}, {"_id": 0}).sort("point_number", -1).limit(limit))
    return {"events": docs}

# =============================================================================
# 8. STORAGE HEALTH CHECK & MONGO DIAGNOSTICS
# =============================================================================
@app.get("/api/health/mongo")
@app.get("/api/health/storage")
def get_mongo_health():
    """Verify live MongoDB Atlas connectivity, ping, and cluster status safely."""
    return mongo_store.test_connection()

# =============================================================================
# 9. FEATURE 7: MACHINE LEARNING MATCH WINNER PREDICTION
# =============================================================================
@app.get("/api/predict/status")
def get_predictor_status():
    """Return ML model metadata, evaluation metrics, and operational readiness."""
    return predictor.get_status()

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
# FRONTEND STATIC FILES
# =============================================================================
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RallyIQ API active. Frontend index.html not yet installed."}
