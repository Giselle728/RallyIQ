"""
RallyIQ Dataset Ingestion, Cleaning, Schema Validation, and Duplicate Detection
Processes raw ATP CSV files, validates schemas, eliminates duplicate records,
analyzes missing values, and outputs cleaned structured datasets.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from backend.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MATCHES_FILE,
    POINTS_2020S_FILE,
    POINTS_2010S_FILE,
    OVERVIEW_FILE,
    SERVE_DIR_FILE,
    RALLY_FILE,
    NET_FILE,
    DATASET_METRICS_REPORT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rallyiq.ingest")

# Text that means "no name", including values stringified upstream (NaN -> "nan").
MISSING_NAME_TOKENS = {"", "nan", "none", "null", "n/a", "na"}


def _clean_name_series(series: pd.Series) -> pd.Series:
    """Strip names; missing, blank, or stringified-null values become <NA>."""
    cleaned = series.astype("string").str.strip()
    invalid = cleaned.isna() | cleaned.str.lower().isin(MISSING_NAME_TOKENS)
    return cleaned.mask(invalid)

def validate_matches_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate ATP matches metadata."""
    initial_count = len(df)
    
    # 1. Deduplication on match_id
    dupes = int(df.duplicated(subset=["match_id"]).sum())
    df = df.drop_duplicates(subset=["match_id"]).copy()
    
    # 2. Schema validation & column standardization
    valid_surfaces = {"Hard", "Clay", "Grass", "Carpet"}
    df["Surface"] = df["Surface"].astype(str).str.strip().str.capitalize()
    df.loc[~df["Surface"].isin(valid_surfaces), "Surface"] = "Hard"  # Standardize anomalies
    
    # Standardize player names; drop rows where either name is missing/blank
    df["Player 1"] = _clean_name_series(df["Player 1"])
    df["Player 2"] = _clean_name_series(df["Player 2"])
    bad_names = df["Player 1"].isna() | df["Player 2"].isna()
    missing_names_dropped = int(bad_names.sum())
    df = df[~bad_names].copy()
    df["Player 1"] = df["Player 1"].astype(str)
    df["Player 2"] = df["Player 2"].astype(str)
    
    # Extract Year from Date (format YYYYMMDD or from match_id)
    def parse_year(row):
        val = str(row["Date"]).strip()
        if len(val) >= 4 and val[:4].isdigit():
            y = int(val[:4])
            if 1968 <= y <= 2026:
                return y
        # Try from match_id prefix
        m_id = str(row["match_id"]).strip()
        if len(m_id) >= 4 and m_id[:4].isdigit():
            y = int(m_id[:4])
            if 1968 <= y <= 2026:
                return y
        return 2020

    df["Year"] = df.apply(parse_year, axis=1)
    
    # Determine winner if available from match_id naming convention (Player1 vs Player2)
    # Jeff Sackmann convention: Player_1-Player_2; winner is typically Player 1 in charted format
    # But let's verify if score indicates winner
    df["Best of"] = pd.to_numeric(df["Best of"], errors="coerce").fillna(3).astype(int)
    
    missing_stats = {col: int(df[col].isna().sum()) for col in df.columns}
    
    metrics = {
        "initial_rows": initial_count,
        "clean_rows": len(df),
        "duplicates_removed": dupes,
        "missing_values": missing_stats,
        "surfaces": df["Surface"].value_counts().to_dict(),
        "missing_names_dropped": missing_names_dropped,
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max())
    }
    return df, metrics

def validate_points_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate point-by-point data."""
    initial_count = len(df)
    
    # 1. Deduplication on (match_id, Pt)
    dupes = int(df.duplicated(subset=["match_id", "Pt"]).sum())
    df = df.drop_duplicates(subset=["match_id", "Pt"]).copy()
    
    # 2. Type coercion & sanitization
    df["Pt"] = pd.to_numeric(df["Pt"], errors="coerce").fillna(1).astype(int)
    df["Svr"] = pd.to_numeric(df["Svr"], errors="coerce").fillna(1).astype(int)
    df["PtWinner"] = pd.to_numeric(df["PtWinner"], errors="coerce").fillna(1).astype(int)
    
    # Filter valid point winners (1 or 2)
    df = df[df["PtWinner"].isin([1, 2])].copy()
    
    # Ensure Set and Game columns
    df["Set1"] = pd.to_numeric(df["Set1"], errors="coerce").fillna(0).astype(int)
    df["Set2"] = pd.to_numeric(df["Set2"], errors="coerce").fillna(0).astype(int)
    df["Gm1"] = pd.to_numeric(df["Gm1"], errors="coerce").fillna(0).astype(int)
    df["Gm2"] = pd.to_numeric(df["Gm2"], errors="coerce").fillna(0).astype(int)
    
    # Clean text columns
    df["Pts"] = df["Pts"].astype(str).str.strip()
    df["1st"] = df["1st"].fillna("").astype(str).str.strip()
    df["2nd"] = df["2nd"].fillna("").astype(str).str.strip()
    
    # Serve direction parser: first char of 1st or 2nd (4=wide, 5=body, 6=T)
    def extract_serve_dir(row):
        code = row["1st"] or row["2nd"]
        if code and len(code) > 0 and code[0] in "456":
            dirs = {"4": "Wide", "5": "Body", "6": "T"}
            return dirs[code[0]]
        return "Unknown"
    
    df["ServeDirection"] = df.apply(extract_serve_dir, axis=1)
    
    # Is Ace or Service Winner
    df["IsAce"] = df["1st"].str.contains(r"\*", regex=True) | df["2nd"].str.contains(r"\*", regex=True)
    df["IsDoubleFault"] = (df["1st"] != "") & (df["2nd"] != "") & (df["2nd"].str.contains(r"[ndw]", regex=True)) & (df["PtWinner"] != df["Svr"])

    metrics = {
        "initial_rows": initial_count,
        "clean_rows": len(df),
        "duplicates_removed": dupes,
        "unique_matches": int(df["match_id"].nunique()),
        "aces_identified": int(df["IsAce"].sum())
    }
    return df, metrics

def validate_overview_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate match overview statistics."""
    initial_count = len(df)
    
    # Filter to Total set summary for overall match stats
    df = df[df["set"] == "Total"].copy()
    
    # Drop rows with a missing/blank player name
    df["player"] = _clean_name_series(df["player"])
    missing_players = int(df["player"].isna().sum())
    df = df[df["player"].notna()].copy()
    df["player"] = df["player"].astype(str)
    
    # Deduplicate on (match_id, player)
    dupes = int(df.duplicated(subset=["match_id", "player"]).sum())
    df = df.drop_duplicates(subset=["match_id", "player"]).copy()
    
    numeric_cols = [
        "serve_pts", "aces", "dfs", "first_in", "first_won",
        "second_in", "second_won", "bk_pts", "bp_saved",
        "return_pts", "return_pts_won", "winners", "unforced"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    metrics = {
        "initial_rows": initial_count,
        "clean_rows": len(df),
        "duplicates_removed": dupes,
        "missing_players_dropped": missing_players,
        "unique_players": int(df["player"].nunique())
    }
    return df, metrics

def run_ingestion_and_validation() -> Dict[str, Any]:
    """Run full ingestion, cleaning, validation, and profiling pipeline."""
    logger.info("Starting RallyIQ Real Dataset Ingestion & Validation...")
    
    # Raw file size calculations
    raw_files = [
        MATCHES_FILE, POINTS_2020S_FILE, POINTS_2010S_FILE,
        OVERVIEW_FILE, SERVE_DIR_FILE, RALLY_FILE, NET_FILE
    ]
    raw_size_bytes = sum(os.path.getsize(f) for f in raw_files if os.path.exists(f))
    raw_size_mb = raw_size_bytes / (1024 * 1024)
    logger.info(f"Verified raw data size: {raw_size_mb:.2f} MB")
    
    # 1. Ingest Matches
    logger.info(f"Ingesting {MATCHES_FILE}...")
    matches_raw = pd.read_csv(MATCHES_FILE)
    matches_clean, matches_meta = validate_matches_schema(matches_raw)
    logger.info(f"Matches: dropped {matches_meta['missing_names_dropped']} rows with missing player names")
    
    # 2. Ingest Points (2020s and 2010s)
    logger.info("Ingesting point-by-point records (2020s & 2010s)...")
    pts_2020 = pd.read_csv(POINTS_2020S_FILE, low_memory=False)
    pts_2010 = pd.read_csv(POINTS_2010S_FILE, low_memory=False)
    pts_combined = pd.concat([pts_2020, pts_2010], ignore_index=True)
    pts_clean, points_meta = validate_points_schema(pts_combined)
    
    # 3. Ingest Overview Stats
    logger.info("Ingesting match performance overview stats...")
    overview_raw = pd.read_csv(OVERVIEW_FILE)
    overview_clean, overview_meta = validate_overview_stats(overview_raw)
    logger.info(f"Overview: dropped {overview_meta['missing_players_dropped']} rows with missing player names")
    
    # 4. Ingest Serve Direction Stats
    logger.info("Ingesting serve direction stats...")
    serve_raw = pd.read_csv(SERVE_DIR_FILE)
    serve_clean = serve_raw[serve_raw["row"] == "Total"].copy().drop_duplicates(subset=["match_id", "player"])
    
    # 5. Ingest Rally & Net Stats
    logger.info("Ingesting rally & net play stats...")
    rally_raw = pd.read_csv(RALLY_FILE)
    net_raw = pd.read_csv(NET_FILE)
    
    # Determine verified match winners from terminal point records
    logger.info("Computing verified match winners from point terminal states...")
    last_pts = pts_clean.sort_values("Pt").groupby("match_id").last().reset_index()[["match_id", "PtWinner"]]
    last_pts.rename(columns={"PtWinner": "terminal_pt_winner"}, inplace=True)
    matches_clean = pd.merge(matches_clean, last_pts, on="match_id", how="left")
    matches_clean["verified_winner"] = np.where(
        matches_clean["terminal_pt_winner"] == 1, matches_clean["Player 1"],
        np.where(matches_clean["terminal_pt_winner"] == 2, matches_clean["Player 2"], None)
    )
    matches_clean["winner_status"] = np.where(
        matches_clean["verified_winner"].notna(), "Verified", "Unverified (Points Not Charted)"
    )

    # Save cleaned master datasets for PySpark batch jobs
    clean_matches_path = os.path.join(DATA_PROCESSED_DIR, "clean_matches.parquet")
    clean_points_path = os.path.join(DATA_PROCESSED_DIR, "clean_points.parquet")
    clean_overview_path = os.path.join(DATA_PROCESSED_DIR, "clean_overview.parquet")
    clean_servedir_path = os.path.join(DATA_PROCESSED_DIR, "clean_serve_direction.parquet")
    clean_rally_path = os.path.join(DATA_PROCESSED_DIR, "clean_rally.parquet")
    clean_net_path = os.path.join(DATA_PROCESSED_DIR, "clean_net_points.parquet")
    
    matches_clean.to_parquet(clean_matches_path, index=False)
    pts_clean.to_parquet(clean_points_path, index=False)
    overview_clean.to_parquet(clean_overview_path, index=False)
    serve_clean.to_parquet(clean_servedir_path, index=False)
    rally_raw.to_parquet(clean_rally_path, index=False)
    net_raw.to_parquet(clean_net_path, index=False)
    
    # Calculate processed data size
    processed_files = [
        clean_matches_path, clean_points_path, clean_overview_path,
        clean_servedir_path, clean_rally_path, clean_net_path
    ]
    processed_size_bytes = sum(os.path.getsize(f) for f in processed_files if os.path.exists(f))
    processed_size_mb = processed_size_bytes / (1024 * 1024)
    
    # Prepare comprehensive data quality & metrics report
    report = {
        "dataset_name": "ATP Match Charting Project (Jeff Sackmann)",
        "raw_size_mb": round(raw_size_mb, 2),
        "processed_size_mb": round(processed_size_mb, 2),
        "total_raw_records": len(matches_raw) + len(pts_combined) + len(overview_raw) + len(serve_raw) + len(rally_raw) + len(net_raw),
        "match_count": len(matches_clean),
        "point_count": len(pts_clean),
        "overview_rows": len(overview_clean),
        "serve_rows": len(serve_clean),
        "season_coverage": {
            "min_year": matches_meta["year_min"],
            "max_year": matches_meta["year_max"],
            "total_seasons": matches_meta["year_max"] - matches_meta["year_min"] + 1
        },
        "surfaces": matches_meta["surfaces"],
        "data_cleaning_metrics": {
            "matches_duplicates_removed": matches_meta["duplicates_removed"],
            "points_duplicates_removed": points_meta["duplicates_removed"],
            "overview_duplicates_removed": overview_meta["duplicates_removed"],
            "matches_missing_names_dropped": matches_meta["missing_names_dropped"],
            "overview_missing_players_dropped": overview_meta["missing_players_dropped"]
        },
        "missing_value_statistics": {
            "matches": matches_meta["missing_values"]
        },
        "winner_attribution": {
            "verified_matches": int(matches_clean["verified_winner"].notna().sum()),
            "unverified_matches": int(matches_clean["verified_winner"].isna().sum()),
            "verified_percentage": round(int(matches_clean["verified_winner"].notna().sum()) / len(matches_clean) * 100, 1)
        }
    }
    
    with open(DATASET_METRICS_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Ingestion & validation complete. Raw: {raw_size_mb:.2f} MB, Points: {len(pts_clean):,}, Matches: {len(matches_clean):,}")
    return report

if __name__ == "__main__":
    run_ingestion_and_validation()
