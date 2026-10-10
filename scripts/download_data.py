#!/usr/bin/env python3
"""
RallyIQ - Raw ATP Dataset Downloader
Downloads genuine ATP match and point-by-point data directly from the
Match Charting Project (Jeff Sackmann / Tennis Abstract).
Targeting >100 MB of real, non-duplicated raw tennis data.
"""

import os
import sys
import time
import requests
import certifi

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"

FILES_TO_DOWNLOAD = [
    {
        "filename": "charting-m-matches.csv",
        "description": "ATP Match Metadata (1968-Present: tournament, players, surface, court, round, date)",
        "expected_min_mb": 1.0,
    },
    {
        "filename": "charting-m-points-2020s.csv",
        "description": "ATP Point-by-Point Records 2020-Present (point sequence, game/set state, serve direction, rally, shot codes)",
        "expected_min_mb": 50.0,
    },
    {
        "filename": "charting-m-points-2010s.csv",
        "description": "ATP Point-by-Point Records 2010-2019 (Big Four golden era point-by-point data)",
        "expected_min_mb": 30.0,
    },
    {
        "filename": "charting-m-stats-Overview.csv",
        "description": "Match-level performance stats (aces, double faults, 1st/2nd serve won, break points saved/won)",
        "expected_min_mb": 5.0,
    },
    {
        "filename": "charting-m-stats-ServeDirection.csv",
        "description": "Serve Direction Analytics (Deuce Wide/Body/T, Ad Wide/Body/T, fault coordinates & outcomes)",
        "expected_min_mb": 4.0,
    },
    {
        "filename": "charting-m-stats-Rally.csv",
        "description": "Rally Length Analytics (1-3 shots, 4-6 shots, 7-9 shots, 10+ shots with win distributions)",
        "expected_min_mb": 9.0,
    },
    {
        "filename": "charting-m-stats-NetPoints.csv",
        "description": "Net Play Analytics (Approach shots, net points won, passing shots, net errors)",
        "expected_min_mb": 5.0,
    },
]

def download_file(filename: str, description: str):
    target_path = os.path.join(DATA_DIR, filename)
    url = f"{BASE_URL}/{filename}"
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[EXISTS] {filename} ({size_mb:.2f} MB) already downloaded. Skipping.")
        return target_path, size_mb

    print(f"\n[DOWNLOADING] {filename}...")
    print(f"  Description: {description}")
    print(f"  Source URL: {url}")
    
    start_time = time.time()
    
    try:
        response = requests.get(url, stream=True, verify=certifi.where(), timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    mb = downloaded / (1024 * 1024)
                    if total_size > 0:
                        tot_mb = total_size / (1024 * 1024)
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r  Progress: {mb:.1f} MB / {tot_mb:.1f} MB ({percent:.1f}%)")
                    else:
                        sys.stdout.write(f"\r  Downloaded: {mb:.1f} MB")
                    sys.stdout.flush()
                    
        duration = time.time() - start_time
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"\n  [DONE] {filename} saved ({size_mb:.2f} MB in {duration:.1f}s)")
        return target_path, size_mb
    except Exception as e:
        print(f"\n  [ERROR] Failed to download {filename}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        raise

def main():
    print("=" * 70)
    print("RallyIQ: Big Data Tennis Intelligence Platform")
    print("Authentic ATP Dataset Download Pipeline")
    print("=" * 70)
    
    total_mb = 0.0
    downloaded_files = []
    
    for item in FILES_TO_DOWNLOAD:
        path, size_mb = download_file(item["filename"], item["description"])
        total_mb += size_mb
        downloaded_files.append((item["filename"], size_mb))

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    for name, mb in downloaded_files:
        print(f"  • {name:<35} : {mb:>7.2f} MB")
    print("-" * 70)
    print(f"  TOTAL RAW DATASET SIZE           : {total_mb:>7.2f} MB")
    print(f"  STORAGE DIRECTORY                : {DATA_DIR}")
    print("=" * 70)
    
    if total_mb >= 50.0:
        print(f"✓ Requirement satisfied: Raw dataset size ({total_mb:.2f} MB) meets/exceeds 50-100 MB target.")
    else:
        print(f"⚠ Warning: Total size ({total_mb:.2f} MB) is below 50 MB.")

if __name__ == "__main__":
    main()
