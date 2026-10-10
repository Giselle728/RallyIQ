"""
RallyIQ Configuration
Paths, Big Data constants, PySpark settings, and MongoDB configs.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CACHE_DIR = os.path.join(BASE_DIR, "data", "stream_cache")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# File names
MATCHES_FILE = os.path.join(DATA_RAW_DIR, "charting-m-matches.csv")
POINTS_2020S_FILE = os.path.join(DATA_RAW_DIR, "charting-m-points-2020s.csv")
POINTS_2010S_FILE = os.path.join(DATA_RAW_DIR, "charting-m-points-2010s.csv")
OVERVIEW_FILE = os.path.join(DATA_RAW_DIR, "charting-m-stats-Overview.csv")
SERVE_DIR_FILE = os.path.join(DATA_RAW_DIR, "charting-m-stats-ServeDirection.csv")
RALLY_FILE = os.path.join(DATA_RAW_DIR, "charting-m-stats-Rally.csv")
NET_FILE = os.path.join(DATA_RAW_DIR, "charting-m-stats-NetPoints.csv")

# Processed Parquet / JSON paths
PROCESSED_PLAYER_STATS = os.path.join(DATA_PROCESSED_DIR, "player_aggregations.json")
PROCESSED_RIVALRIES = os.path.join(DATA_PROCESSED_DIR, "rivalry_network.json")
PROCESSED_PLAYSTYLES = os.path.join(DATA_PROCESSED_DIR, "playstyle_profiles.json")
PROCESSED_SERVE_STRATEGY = os.path.join(DATA_PROCESSED_DIR, "serve_strategy.json")
DATASET_METRICS_REPORT = os.path.join(DATA_PROCESSED_DIR, "dataset_report.json")
MONGO_DISK_CACHE = os.path.join(DATA_PROCESSED_DIR, "mongo_storage.json")

# PySpark local configuration (safe for 8 GB RAM Mac)
SPARK_APP_NAME = "RallyIQ-Tennis-BigData"
SPARK_MASTER = "local[2]"
SPARK_DRIVER_MEMORY = "1g"
SPARK_EXECUTOR_MEMORY = "1g"

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = "rallyiq_tennis"

# Stream & Bloom Filter Defaults
DEFAULT_BLOOM_CAPACITY = 10000
DEFAULT_BLOOM_BITS = 8192
DEFAULT_BLOOM_HASHES = 5
DEFAULT_DGIM_WINDOW = 40
DEFAULT_MOMENTUM_ALPHA = 0.25
