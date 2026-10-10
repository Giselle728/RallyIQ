"""
RallyIQ Replay Indexer
Indexes point-by-point matches for low-latency streaming and replay lookups.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List

from backend.config import DATA_PROCESSED_DIR

logger = logging.getLogger("rallyiq.indexer")

REPLAY_CACHE_PATH = os.path.join(DATA_PROCESSED_DIR, "replayable_matches.json")
STREAM_EVENTS_SAMPLE = os.path.join(DATA_PROCESSED_DIR, "stream_sample_events.json")

def generate_replay_index(top_n_matches: int = 50) -> Dict[str, Any]:
    clean_points_path = os.path.join(DATA_PROCESSED_DIR, "clean_points.parquet")
    clean_matches_path = os.path.join(DATA_PROCESSED_DIR, "clean_matches.parquet")
    
    if not os.path.exists(clean_points_path) or not os.path.exists(clean_matches_path):
        logger.warning("Processed parquet files not found.")
        return {}

    logger.info("Indexing replayable matches...")
    pts_df = pd.read_parquet(clean_points_path)
    matches_df = pd.read_parquet(clean_matches_path)
    
    matches_dict = matches_df.set_index("match_id").to_dict(orient="index")
    
    point_counts = pts_df["match_id"].value_counts()
    top_match_ids = point_counts.head(top_n_matches).index.tolist()
    
    indexed_matches = []
    for m_id in top_match_ids:
        meta = matches_dict.get(m_id, {})
        indexed_matches.append({
            "match_id": m_id,
            "player_1": meta.get("Player 1", "Player 1"),
            "player_2": meta.get("Player 2", "Player 2"),
            "tournament": meta.get("Tournament", "ATP Tour"),
            "round": meta.get("Round", "Main Draw"),
            "surface": meta.get("Surface", "Hard"),
            "year": int(meta.get("Year", 2020)),
            "points_count": int(point_counts[m_id]),
            "label": f"{meta.get('Year', '')} {meta.get('Tournament', '')} - {meta.get('Player 1', '')} vs {meta.get('Player 2', '')} ({point_counts[m_id]} pts)"
        })
        
    with open(REPLAY_CACHE_PATH, "w") as f:
        json.dump(indexed_matches, f, indent=2)
        
    # Also extract a sample of 2,000 real point events for Stream Integrity testing
    sample_stream_pts = pts_df[pts_df["match_id"].isin(top_match_ids[:10])].head(2000)
    stream_events = []
    for idx, row in sample_stream_pts.iterrows():
        m_id = str(row["match_id"])
        meta = matches_dict.get(m_id, {})
        stream_events.append({
            "event_id": f"{m_id}_pt_{row['Pt']}",
            "match_id": m_id,
            "point_number": int(row["Pt"]),
            "player_1": meta.get("Player 1", "Player 1"),
            "player_2": meta.get("Player 2", "Player 2"),
            "server": int(row["Svr"]),
            "winner": int(row["PtWinner"]),
            "score": str(row["Pts"]),
            "set1": int(row["Set1"]),
            "set2": int(row["Set2"]),
            "gm1": int(row["Gm1"]),
            "gm2": int(row["Gm2"]),
            "serve_dir": str(row.get("ServeDirection", "Unknown")),
            "is_ace": bool(row.get("IsAce", False))
        })
        
    with open(STREAM_EVENTS_SAMPLE, "w") as f:
        json.dump(stream_events, f)
        
    logger.info(f"Replay index generated with {len(indexed_matches)} matches and {len(stream_events)} stream events.")
    return {"matches": indexed_matches, "stream_sample_count": len(stream_events)}

if __name__ == "__main__":
    generate_replay_index()
