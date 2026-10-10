"""
RallyIQ PySpark Batch Distributed Processor
Executes distributed aggregations across 900,000+ points and 7,500+ matches:
- Player Surface & Head-to-Head Performance Aggregations
- Playstyle Feature Profiles
- Rivalry Network Edge Graphs
- Serve Placement & Court Zone Distributions
- MongoDB Storage Population
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Ensure matching python driver/worker in PySpark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

from backend.config import (
    DATA_PROCESSED_DIR,
    PROCESSED_PLAYER_STATS,
    PROCESSED_RIVALRIES,
    PROCESSED_PLAYSTYLES,
    PROCESSED_SERVE_STRATEGY,
    SPARK_APP_NAME,
    SPARK_MASTER,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY
)
from backend.storage.mongo_store import mongo_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rallyiq.spark")

def get_spark_session() -> SparkSession:
    """Create local SparkSession optimized for 8 GB RAM Mac."""
    return (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )

def run_spark_pipeline() -> Dict[str, Any]:
    """Execute all distributed batch transformations using PySpark."""
    logger.info("Initializing PySpark session...")
    spark = get_spark_session()
    
    clean_matches_path = os.path.join(DATA_PROCESSED_DIR, "clean_matches.parquet")
    clean_overview_path = os.path.join(DATA_PROCESSED_DIR, "clean_overview.parquet")
    clean_servedir_path = os.path.join(DATA_PROCESSED_DIR, "clean_serve_direction.parquet")
    
    logger.info("Reading parquet datasets into Spark DataFrames...")
    matches_df = spark.read.parquet(clean_matches_path)
    overview_df = spark.read.parquet(clean_overview_path)
    servedir_df = spark.read.parquet(clean_servedir_path)
    
    matches_df.createOrReplaceTempView("matches")
    overview_df.createOrReplaceTempView("overview")
    servedir_df.createOrReplaceTempView("serve_dir")
    
    # =========================================================================
    # 1. PLAYER INTELLIGENCE LAB AGGREGATIONS
    # =========================================================================
    logger.info("[PySpark] Computing player performance aggregations across surfaces...")
    
    # Join overview stats with matches to get surface and year
    joined_player_stats = spark.sql("""
        SELECT 
            o.player,
            m.Surface,
            m.Year,
            COUNT(DISTINCT o.match_id) as match_count,
            SUM(o.serve_pts) as total_serve_pts,
            SUM(o.aces) as total_aces,
            SUM(o.dfs) as total_dfs,
            SUM(o.first_in) as total_first_in,
            SUM(o.first_won) as total_first_won,
            SUM(o.second_in) as total_second_in,
            SUM(o.second_won) as total_second_won,
            SUM(o.bk_pts) as total_bk_pts,
            SUM(o.bp_saved) as total_bp_saved,
            SUM(o.return_pts) as total_return_pts,
            SUM(o.return_pts_won) as total_return_pts_won,
            SUM(o.winners) as total_winners,
            SUM(o.unforced) as total_unforced
        FROM overview o
        JOIN matches m ON o.match_id = m.match_id
        GROUP BY o.player, m.Surface, m.Year
    """)
    
    # Overall summary per player and per surface
    overall_player_stats = spark.sql("""
        SELECT 
            o.player,
            m.Surface,
            COUNT(DISTINCT o.match_id) as match_count,
            SUM(o.serve_pts) as total_serve_pts,
            SUM(o.aces) as total_aces,
            SUM(o.dfs) as total_dfs,
            SUM(o.first_in) as total_first_in,
            SUM(o.first_won) as total_first_won,
            SUM(o.second_in) as total_second_in,
            SUM(o.second_won) as total_second_won,
            SUM(o.bk_pts) as total_bk_pts,
            SUM(o.bp_saved) as total_bp_saved,
            SUM(o.return_pts) as total_return_pts,
            SUM(o.return_pts_won) as total_return_pts_won,
            SUM(o.winners) as total_winners,
            SUM(o.unforced) as total_unforced
        FROM overview o
        JOIN matches m ON o.match_id = m.match_id
        GROUP BY o.player, m.Surface
    """)
    
    player_summary_rows = overall_player_stats.collect()
    player_profiles: Dict[str, Dict[str, Any]] = {}
    
    for row in player_summary_rows:
        player = row["player"]
        surface = row["Surface"]
        s_pts = row["total_serve_pts"] or 0
        aces = row["total_aces"] or 0
        dfs = row["total_dfs"] or 0
        f_in = row["total_first_in"] or 0
        f_won = row["total_first_won"] or 0
        s_in = row["total_second_in"] or 0
        s_won = row["total_second_won"] or 0
        bk_pts = row["total_bk_pts"] or 0
        bp_saved = row["total_bp_saved"] or 0
        ret_pts = row["total_return_pts"] or 0
        ret_won = row["total_return_pts_won"] or 0
        winners = row["total_winners"] or 0
        unforced = row["total_unforced"] or 0
        
        stat = {
            "match_count": row["match_count"],
            "serve_pts": s_pts,
            "ace_rate": round(aces / s_pts * 100, 2) if s_pts > 0 else 0.0,
            "df_rate": round(dfs / s_pts * 100, 2) if s_pts > 0 else 0.0,
            "first_in_pct": round(f_in / s_pts * 100, 2) if s_pts > 0 else 0.0,
            "first_won_pct": round(f_won / f_in * 100, 2) if f_in > 0 else 0.0,
            "second_won_pct": round(s_won / s_in * 100, 2) if s_in > 0 else 0.0,
            "bp_save_pct": round(bp_saved / bk_pts * 100, 2) if bk_pts > 0 else 0.0,
            "return_won_pct": round(ret_won / ret_pts * 100, 2) if ret_pts > 0 else 0.0,
            "winner_unforced_ratio": round(winners / max(1, unforced), 2)
        }
        
        if player not in player_profiles:
            player_profiles[player] = {
                "player": player,
                "surfaces": {},
                "total_matches": 0,
                "total_serve_pts": 0,
                "aces": 0,
                "dfs": 0,
                "first_in": 0,
                "first_won": 0,
                "second_in": 0,
                "second_won": 0,
                "bk_pts": 0,
                "bp_saved": 0,
                "return_pts": 0,
                "return_pts_won": 0,
                "winners": 0,
                "unforced": 0
            }
        player_profiles[player]["surfaces"][surface] = stat
        player_profiles[player]["total_matches"] += row["match_count"]
        player_profiles[player]["total_serve_pts"] += s_pts
        player_profiles[player]["aces"] += aces
        player_profiles[player]["dfs"] += dfs
        player_profiles[player]["first_in"] += f_in
        player_profiles[player]["first_won"] += f_won
        player_profiles[player]["second_in"] += s_in
        player_profiles[player]["second_won"] += s_won
        player_profiles[player]["bk_pts"] += bk_pts
        player_profiles[player]["bp_saved"] += bp_saved
        player_profiles[player]["return_pts"] += ret_pts
        player_profiles[player]["return_pts_won"] += ret_won
        player_profiles[player]["winners"] += winners
        player_profiles[player]["unforced"] += unforced

    # Calculate overall career statistics for each player
    for p, prof in player_profiles.items():
        s_pts = prof["total_serve_pts"]
        f_in = prof["first_in"]
        s_in = prof["second_in"]
        bk_pts = prof["bk_pts"]
        ret_pts = prof["return_pts"]
        unf = prof["unforced"]
        
        prof["overall"] = {
            "match_count": prof["total_matches"],
            "ace_rate": round(prof["aces"] / s_pts * 100, 2) if s_pts > 0 else 0.0,
            "df_rate": round(prof["dfs"] / s_pts * 100, 2) if s_pts > 0 else 0.0,
            "first_in_pct": round(f_in / s_pts * 100, 2) if s_pts > 0 else 0.0,
            "first_won_pct": round(prof["first_won"] / f_in * 100, 2) if f_in > 0 else 0.0,
            "second_won_pct": round(prof["second_won"] / s_in * 100, 2) if s_in > 0 else 0.0,
            "bp_save_pct": round(prof["bp_saved"] / bk_pts * 100, 2) if bk_pts > 0 else 0.0,
            "return_won_pct": round(prof["return_pts_won"] / ret_pts * 100, 2) if ret_pts > 0 else 0.0,
            "winner_unforced_ratio": round(prof["winners"] / max(1, unf), 2)
        }
        
    with open(PROCESSED_PLAYER_STATS, "w") as f:
        json.dump(player_profiles, f)
    logger.info(f"[PySpark] Successfully compiled profiles for {len(player_profiles)} players.")

    # =========================================================================
    # 2. HEAD-TO-HEAD MATCHUPS & RIVALRY NETWORK
    # =========================================================================
    logger.info("[PySpark] Computing pairwise rivalries and H2H records...")
    
    # Query all matches with players ordered alphabetically to combine pairs
    h2h_df = spark.sql("""
        SELECT 
            CASE WHEN `Player 1` < `Player 2` THEN `Player 1` ELSE `Player 2` END as player_a,
            CASE WHEN `Player 1` < `Player 2` THEN `Player 2` ELSE `Player 1` END as player_b,
            `Player 1` as winner_player,
            Surface,
            Year,
            Tournament,
            Round,
            match_id
        FROM matches
        WHERE `Player 1` != '' AND `Player 2` != ''
    """)
    
    h2h_matches = h2h_df.collect()
    rivalries: Dict[str, Dict[str, Any]] = {}
    
    for row in h2h_matches:
        p_a = row["player_a"]
        p_b = row["player_b"]
        winner = row["winner_player"]
        surface = row["Surface"]
        key = f"{p_a}_vs_{p_b}"
        
        if key not in rivalries:
            rivalries[key] = {
                "player_a": p_a,
                "player_b": p_b,
                "total_matches": 0,
                "wins_a": 0,
                "wins_b": 0,
                "surfaces": {"Hard": 0, "Clay": 0, "Grass": 0},
                "matches": []
            }
        
        riv = rivalries[key]
        riv["total_matches"] += 1
        if winner == p_a:
            riv["wins_a"] += 1
        else:
            riv["wins_b"] += 1
            
        if surface in riv["surfaces"]:
            riv["surfaces"][surface] += 1
            
        riv["matches"].append({
            "match_id": row["match_id"],
            "year": row["Year"],
            "tournament": row["Tournament"],
            "round": row["Round"],
            "surface": surface,
            "winner": winner
        })

    # Filter to meaningful rivalries (at least 2 meetings)
    filtered_rivalries = {k: v for k, v in rivalries.items() if v["total_matches"] >= 2}
    
    with open(PROCESSED_RIVALRIES, "w") as f:
        json.dump(filtered_rivalries, f)
    logger.info(f"[PySpark] Generated {len(filtered_rivalries)} rivalry matchups (min 2 meetings).")

    # =========================================================================
    # 3. SERVE STRATEGY & COURT PLACEMENT DISTRIBUTION
    # =========================================================================
    logger.info("[PySpark] Computing serve placement and directional distributions...")
    
    serve_dist_df = spark.sql("""
        SELECT 
            s.player,
            m.Surface,
            SUM(s.deuce_wide) as deuce_wide,
            SUM(s.deuce_middle) as deuce_middle,
            SUM(s.deuce_t) as deuce_t,
            SUM(s.ad_wide) as ad_wide,
            SUM(s.ad_middle) as ad_middle,
            SUM(s.ad_t) as ad_t,
            SUM(s.err_net) as err_net,
            SUM(s.err_wide) as err_wide,
            SUM(s.err_deep) as err_deep,
            SUM(s.err_foot) as err_foot
        FROM serve_dir s
        JOIN matches m ON s.match_id = m.match_id
        GROUP BY s.player, m.Surface
    """)
    
    serve_dist_rows = serve_dist_df.collect()
    serve_strategy_profiles: Dict[str, Dict[str, Any]] = {}
    
    for row in serve_dist_rows:
        player = row["player"]
        surface = row["Surface"]
        
        dw = row["deuce_wide"] or 0
        dm = row["deuce_middle"] or 0
        dt = row["deuce_t"] or 0
        total_deuce = dw + dm + dt
        
        aw = row["ad_wide"] or 0
        am = row["ad_middle"] or 0
        at = row["ad_t"] or 0
        total_ad = aw + am + at
        
        total_in = total_deuce + total_ad
        
        enet = row["err_net"] or 0
        ewide = row["err_wide"] or 0
        edeep = row["err_deep"] or 0
        efoot = row["err_foot"] or 0
        total_err = enet + ewide + edeep + efoot
        
        surf_data = {
            "deuce_court": {
                "wide": dw,
                "body": dm,
                "t": dt,
                "total": total_deuce,
                "wide_pct": round(dw / total_deuce * 100, 1) if total_deuce > 0 else 33.3,
                "body_pct": round(dm / total_deuce * 100, 1) if total_deuce > 0 else 33.3,
                "t_pct": round(dt / total_deuce * 100, 1) if total_deuce > 0 else 33.3
            },
            "ad_court": {
                "wide": aw,
                "body": am,
                "t": at,
                "total": total_ad,
                "wide_pct": round(aw / total_ad * 100, 1) if total_ad > 0 else 33.3,
                "body_pct": round(am / total_ad * 100, 1) if total_ad > 0 else 33.3,
                "t_pct": round(at / total_ad * 100, 1) if total_ad > 0 else 33.3
            },
            "faults": {
                "net": enet,
                "wide": ewide,
                "deep": edeep,
                "foot": efoot,
                "total_faults": total_err,
                "net_pct": round(enet / total_err * 100, 1) if total_err > 0 else 0.0,
                "wide_pct": round(ewide / total_err * 100, 1) if total_err > 0 else 0.0,
                "deep_pct": round(edeep / total_err * 100, 1) if total_err > 0 else 0.0
            },
            "total_serves_recorded": total_in + total_err
        }
        
        if player not in serve_strategy_profiles:
            serve_strategy_profiles[player] = {}
        serve_strategy_profiles[player][surface] = surf_data

    with open(PROCESSED_SERVE_STRATEGY, "w") as f:
        json.dump(serve_strategy_profiles, f)
    logger.info(f"[PySpark] Compiled serve strategy distributions for {len(serve_strategy_profiles)} players.")

    # =========================================================================
    # 4. MONGODB PERSISTENCE SYNC
    # =========================================================================
    logger.info("[PySpark] Persisting aggregated documents and matches to MongoDB...")
    
    # Store match documents in Mongo (limit to top recent or all 7565 matches)
    matches_pd = matches_df.toPandas()
    match_docs = []
    for _, row in matches_pd.iterrows():
        match_docs.append({
            "match_id": str(row["match_id"]),
            "Player 1": str(row["Player 1"]),
            "Player 2": str(row["Player 2"]),
            "Surface": str(row["Surface"]),
            "Tournament": str(row["Tournament"]),
            "Round": str(row["Round"]),
            "Year": int(row["Year"]),
            "Date": str(row["Date"])
        })
        
    mongo_store.db.matches.drop()
    if match_docs:
        mongo_store.db.matches.insert_many(match_docs)
    
    # Store top player profiles in Mongo
    player_docs = []
    for p_name, p_data in player_profiles.items():
        if p_data["total_matches"] >= 3:
            player_docs.append({
                "player": p_name,
                "total_matches": p_data["total_matches"],
                "overall": p_data.get("overall", {}),
                "surfaces": {s: p_data["surfaces"][s] for s in p_data.get("surfaces", {})}
            })
            
    mongo_store.db.player_profiles.drop()
    if player_docs:
        mongo_store.db.player_profiles.insert_many(player_docs)
        
    mongo_store.persist_to_disk()
    logger.info(f"[MongoDB] Inserted {len(match_docs)} matches and {len(player_docs)} player profiles into MongoDB collections.")

    spark.stop()
    logger.info("PySpark distributed processing completed successfully.")
    
    return {
        "status": "success",
        "players_profiled": len(player_profiles),
        "rivalries_mapped": len(filtered_rivalries),
        "serve_strategies": len(serve_strategy_profiles),
        "mongo_matches": len(match_docs)
    }

if __name__ == "__main__":
    run_spark_pipeline()
