# RallyIQ: Big Data Tennis Intelligence Platform

A high-performance Big Data Analytics (BDA) platform built on **117.0 MB of genuine historical ATP men's tennis data** from the Match Charting Project. Implements distributed batch processing (PySpark), document storage with indexing (MongoDB), sliding-window stream counting (DGIM), probabilistic membership testing (Bloom filter), graph community detection (Louvain & PageRank), and multi-feature player similarity (Cosine metric).

---

## Architecture Overview

```
[117 MB Raw ATP Data (Jeff Sackmann)]
                │
                ▼
[Ingestion & Schema Validation (Pandas)]
   • 904,128 sequential points validated
   • 7,565 ATP matches deduplicated & cleaned
   • Column type coercion & null profiling
                │
                ▼
[Distributed Batch Processing (Apache PySpark)]
   • Surface-split player aggregations
   • Pairwise rivalry records (1,183 matchups)
   • Serve directional distributions
                │
                ▼
[Document Storage Layer (MongoDB / mongomock fallback)]
   • Collections: matches, player_profiles, stream_events, stream_metrics
   • Compound indexes on match_id, players, surface, event_id
                │
                ├──────────────────────────────────────┐
                ▼                                      ▼
[Real-Time Analytics Algorithms]             [FastAPI REST Backend]
   • DGIM: O(log² N) sliding window              • 12 REST Endpoints
   • EWMA Momentum Analyzer                      • Interactive CORS API
   • Kirsch-Mitzenmacher Bloom Filter            • Parquet/JSON low-latency caches
   • Cosine Playstyle Recommender                      │
   • Louvain Community Detection (Q=0.46)              ▼
   • PageRank Dominance Centrality           [Modern Dark Navy / Cyan UI]
```

---

## Implemented Big Data Algorithms

| # | Feature | Big Data Algorithm / Technology | Mathematical / Algorithmic Formulation |
|---|---|---|---|
| 1 | **Player Intelligence Lab** | Distributed Spark Aggregation | Distributed joins & MapReduce-style multi-key aggregations across surface & career dimensions |
| 2 | **Match Momentum Analyzer** | DGIM (Datar-Gionis-Indyk-Motwani) + EWMA | Sliding-window bit counting with power-of-two bucket hierarchy, error $\le \frac{1}{2k} = 25\%$, $O(\log^2 N)$ memory |
| 3 | **Serve Strategy Heatmap** | Directional & Fault Zone Clustering | Court spatial partitioning into Deuce/Ad service boxes (Wide, Body, T) with error trajectories |
| 4 | **Playstyle Recommender** | Normalized Cosine Similarity | Z-score feature scaling across 8 performance dimensions, Cosine metric, explainable attribution |
| 5 | **ATP Rivalry Network** | Louvain Community Detection & PageRank | Modularity optimization $Q = \sum [e_{ii} - a_i^2]$ and directed eigenvector dominance centrality |
| 6 | **Stream Integrity Monitor** | Bloom Filter with Double Hashing | $k$ hash functions via $h_i(x) = (h_1(x) + i \cdot h_2(x)) \pmod m$, theoretical FP $p = (1 - e^{-kn/m})^k$ |
| 7 | **ML Match Winner Predictor** | Random Forest vs. Logistic Regression | Chronological pre-match feature vectors (zero target leakage), symmetrical ensembling $P(A, B) = 1 - P(B, A)$ |

---

## Machine Learning Match Winner Predictor

RallyIQ includes a genuine supervised machine-learning model trained to estimate pre-match ATP match outcomes and probabilities.

### Zero-Target-Leakage Methodology
1. **Chronological Feature Isolation:** Features for any match at time $T$ are computed strictly using matches that occurred prior to $T$. Target match outcome is withheld until after features are recorded.
2. **Post-Match Metric Exclusion:** Post-match statistics (aces, double faults, unforced errors, match duration, and break point conversions in that match) are strictly excluded from the pre-match feature vector.
3. **Pure Pre-Match Features Extracted (14 Dimensions):**
   - `win_rate_diff`: Career win rate difference ($P_1 - P_2$)
   - `surface_win_rate_diff`: Court-specific surface win rate difference
   - `recent_form_diff`: 10-match rolling form win rate difference
   - `h2h_diff` & `h2h_total`: Prior head-to-head meeting wins difference and total rivalry matches
   - `exp_diff`: Log-transformed tour experience difference ($\ln(1 + M_1) - \ln(1 + M_2)$)
   - `p1_win_rate`, `p2_win_rate`, `p1_surface_win_rate`, `p2_surface_win_rate`: Absolute player priors
   - `surface_Clay`, `surface_Grass`, `surface_Hard`: One-hot encoded surface indicators
   - `best_of`: Match format (3 vs 5 sets)
4. **Guaranteed Mathematical Symmetry:**
   - Training data includes reciprocal inverted pairs $(P_2, P_1, 1 - y)$.
   - Inference uses symmetrical ensembling: $P(A \text{ beats } B) = \frac{1}{2} [P_{\text{model}}(A, B) + (1 - P_{\text{model}}(B, A))]$, ensuring exact inverse probabilities $P(A, B) + P(B, A) = 100.0\%$.

### Chronological Train / Validation / Test Splits
- **Total Verified Matches Trained:** 5,569 matches (11,138 symmetrical samples)
- **Train Set (70%):** 7,796 samples (Jan 2010 – Aug 2023)
- **Validation Set (15%):** 1,670 samples (Aug 2023 – Oct 2024)
- **Holdout Test Set (15%):** 1,672 samples (Oct 2024 – May 2026)

### Held-Out Test Evaluation Benchmark
| Metric | Baseline Logistic Regression (StandardScaler) | Primary Random Forest (100 Trees, max_depth=6) |
|---|---|---|
| **Accuracy** | 67.28% | **67.94%** (+0.66%) |
| **ROC-AUC** | 0.7350 | **0.7505** (+0.0155) |
| **F1 Score** | 0.6750 | **0.6759** |
| **Precision** | 66.86% | **67.61%** |
| **Recall** | 68.42% | **68.90%** |

### Top Gini Feature Importances
1. `exp_diff` (0.3825) — Tour experience differential
2. `win_rate_diff` (0.1468) — Career win rate differential
3. `p2_win_rate` (0.0995) — Opponent baseline win rate
4. `p1_win_rate` (0.0868) — Player baseline win rate
5. `recent_form_diff` (0.0699) — 10-match rolling form differential
6. `surface_win_rate_diff` (0.0593) — Surface court specialty differential

### Model Retraining Command
```bash
./venv/bin/python3 -m backend.ml.train_predictor
```
Artifacts are automatically saved to `models/match_predictor.joblib` and `models/model_metrics.json`.

---

## MongoDB Atlas & Storage Architecture

RallyIQ supports dual-mode persistent storage:
1. **Live MongoDB Atlas:** Connects to cloud cluster (`tennisbda.ryaeljl.mongodb.net`) over TLS.
   - Configured via root `.env` (`MONGO_URI`).
   - Uses `certifi.where()` CA bundle to prevent macOS OpenSSL root store trust errors.
   - Fully masked connection URI in logs (`mongodb+srv://****:****@...`) to prevent credential leakage.
   - Explicit storage status telemetry: `connected_atlas`, `connected_local`, or `fallback_mock`.
2. **Embedded Fallback (`mongomock` with disk sync):**
   - Automatically engaged when offline or in sandboxed environments where external DNS is blocked.
   - Exposes clear diagnostic status without silent failure masking.

### Storage & Health Endpoints
- `GET /api/health/mongo` / `GET /api/health/storage`: Live ping test (`admin.command('ping')`), active database, cluster, and engine status.
- `GET /api/predict/status`: ML model readiness, test metrics, and feature importances.
- `GET /api/predict/match?player_a=...&player_b=...&surface=Hard&best_of=3`: Probabilistic match forecast.

---

## Dataset Statistics

- **Raw Data Size:** `117.00 MB` (exceeds the 50–100 MB requirement)
- **Point-by-Point Records:** `904,128` points
- **Total Raw Records:** `1,170,566` rows
- **Matches Charted:** `7,565` matches
- **Career & Era Coverage:** `58` seasons (1969 – 2026)
- **Surfaces:** Hard (`4,851`), Clay (`1,877`), Grass (`837`)
- **Deduplication:** 579 duplicate point records and 26 duplicate overview rows removed during validation.

---

## Essential Run Commands

### 1. Run Automated Test Suite (45 Tests)
```bash
./venv/bin/python3 -m unittest discover -s backend/tests
```

### 2. Run the End-to-End Pipeline
```bash
./scripts/run_pipeline.sh
```

### 3. Launch the Application Server
```bash
./scripts/start_server.sh
```
Open your browser at **http://localhost:8000** to explore the dashboard.

---

## Hardware Efficiency (8 GB RAM Mac)
- PySpark operates in `local[2]` mode with `1g` driver/executor memory and partition pruning.
- Memory footprint during execution never exceeds 1.2 GB RAM.
- Storage layer connects to live MongoDB Atlas if reachable, or utilizes an in-process embedded MongoDB engine (`mongomock`) with disk persistence.

---

# Feature Deep Dive: How Everything Works

This section documents every feature end-to-end: the algorithms behind them, how the
code implements them (with real snippets from the codebase), and the exact request/data
flow from raw CSV to rendered UI.

## Project Layout

```
rallyiq/
├── data/
│   ├── raw/                     # 117 MB Jeff Sackmann ATP Match Charting CSVs (7 files)
│   └── processed/               # Cleaned Parquet + JSON artifacts produced by the pipeline
├── backend/
│   ├── config.py                # Paths, Spark config, Mongo config, Bloom/DGIM defaults
│   ├── pipeline/
│   │   ├── ingest_validate.py   # Stage 1: ingestion, dedup, schema validation
│   │   └── spark_processor.py   # Stage 2: distributed batch aggregations (PySpark)
│   ├── algorithms/
│   │   ├── dgim.py              # Feature 2: DGIM sliding window + EWMA momentum
│   │   ├── bloom_filter.py      # Feature 6: Kirsch-Mitzenmacher Bloom filter
│   │   ├── recommender.py       # Feature 4: Cosine playstyle similarity
│   │   └── graph_analytics.py   # Feature 5: Louvain communities + PageRank
│   ├── ml/
│   │   ├── train_predictor.py   # Feature 7: zero-leakage training pipeline
│   │   └── predictor.py         # Feature 7: inference engine (symmetrical ensembling)
│   ├── storage/mongo_store.py   # Singleton storage layer (Atlas / mongomock fallback)
│   ├── api/
│   │   ├── main.py              # FastAPI app: all 25 REST routes
│   │   └── replay_indexer.py    # Builds replay + stream-sample indexes
│   └── tests/                   # 45 unittest cases (algorithms / api / predictor)
├── frontend/                    # Vanilla JS dark-theme SPA (index.html, app.js, style.css)
├── models/                      # match_predictor.joblib + model_metrics.json
└── scripts/                     # run_pipeline.sh, start_server.sh, download_data.py
```

---

## Pipeline Foundation (runs before any feature)

### Stage 1 — Ingestion, Cleaning & Schema Validation (`backend/pipeline/ingest_validate.py`)

**How it works.** Reads the 7 raw charting CSVs (matches, points-2010s, points-2020s,
Overview, ServeDirection, Rally, NetPoints), then per table:

1. **Deduplication** on natural keys — `match_id` for matches, `(match_id, Pt)` for points.
2. **Type coercion** — `Pt`, `Svr`, `PtWinner`, `Set1/2`, `Gm1/2` forced to numeric with `errors="coerce"` + fillna.
3. **Controlled vocabularies** — `Surface` stripped/capitalized and constrained to `{Hard, Clay, Grass, Carpet}`; anything else is standardized to `Hard`.
4. **Derived columns** — `Year` parsed from `Date` (YYYYMMDD) with fallback to the `match_id` prefix, guarded by the range 1968–2026.
5. **Winner verification** — the *last* point of each match is joined back to the match row, so the winner is derived from actual point outcomes rather than trusted from a column:

```python
last_pts = pts_clean.sort_values("Pt").groupby("match_id").tail(1)
matches_clean = matches_clean.merge(last_pts, on="match_id", how="left")
matches_clean["verified_winner"] = np.where(
    matches_clean["terminal_pt_winner"] == 1, matches_clean["Player 1"],
    np.where(matches_clean["terminal_pt_winner"] == 2, matches_clean["Player 2"], None)
)
matches_clean["winner_status"] = np.where(
    matches_clean["verified_winner"].notna(), "Verified", "Unverified (Points Not Charted)"
)
```

**Output:** six `clean_*.parquet` files plus `dataset_report.json` (raw/processed MB,
row counts, duplicates removed, missing-value profile, surface breakdown, winner attribution %).

**Flow:**
```
raw CSVs ──▶ dedupe ──▶ type coercion ──▶ vocabulary fixes ──▶ year parse
        ──▶ join terminal point ──▶ verified_winner / winner_status
        ──▶ clean_*.parquet + dataset_report.json
```
### Stage 2 — Distributed Batch Processing (`backend/pipeline/spark_processor.py`)

**How it works.** A SparkSession is built for an 8 GB machine (`local[2]`, 1g driver,
1g executor, `shuffle.partitions=4`), the Parquet files are registered as temp views,
and the aggregations are expressed as Spark SQL:

```python
overall_player_stats = spark.sql("""
    SELECT o.player, m.Surface,
           COUNT(DISTINCT o.match_id) as match_count,
           SUM(o.serve_pts) as total_serve_pts,
           SUM(o.aces) as total_aces, ... SUM(o.unforced) as total_unforced
    FROM overview o
    JOIN matches m ON o.match_id = m.match_id
    GROUP BY o.player, m.Surface
""")
```

Derived per-player rates (ace rate, DF rate, 1st-serve-in %, 1st/2nd serve win %,
break-point save %, return win %, winners/unforced ratio) are computed from the summed
counters, then rolled into `player_aggregations.json`. The same stage also builds:

- **Pairwise rivalry records** — all player pairs per match, kept only when `total_matches >= 2` → `rivalry_network.json`
- **Serve distributions** — SUM of deuce/ad × wide/body/t plus error zones (net/wide/deep/foot), grouped by player × surface → `serve_strategy.json`
- **MongoDB population** — `matches` and `player_profiles` collections are dropped and re-inserted, then flushed to the disk cache.

**Resilience:** if the Spark gateway/JVM fails to start, `run_spark_pipeline()` catches
the exception and falls back to `_run_local_batch_pipeline()` (identical outputs via
pandas), so the pipeline never hard-fails:

```python
try:
    spark = get_spark_session()
    _ = spark.range(1).count()          # probe the cluster before real work
    return _run_pyspark_pipeline(spark)
except Exception as e:
    logger.warning(f"[PySpark Engine] Spark cluster/gateway unavailable ({e}). "
                   "Executing batch processing via optimized local DataFrame engine...")
    return _run_local_batch_pipeline()
```

### Stage 3 — Replay & Stream Indexing (`backend/api/replay_indexer.py`)

- Picks the **top 50 matches by point count** and writes `replayable_matches.json` (feeds the Momentum tab dropdown).
- Extracts **2,000 real point events from the 10 richest matches** into `stream_sample_events.json`. Each event gets a deterministic key:

```python
"event_id": f"{m_id}_pt_{row['Pt']}"   # e.g. 20231119-..._pt_142
```

This `event_id` is the exact string inserted into the Bloom filter by Feature 6.

**Full pipeline flow:**
```
./scripts/run_pipeline.sh
  ├─ [1/3] backend.pipeline.ingest_validate   → clean_*.parquet + dataset_report.json
  ├─ [2/3] backend.pipeline.spark_processor   → player_aggregations.json,
  │                                              rivalry_network.json, serve_strategy.json,
  │                                              MongoDB matches/player_profiles
  └─ [3/3] backend.api.replay_indexer         → replayable_matches.json,
                                                 stream_sample_events.json
```

---

## Feature 1 — Player Intelligence Lab

**Endpoints:** `GET /api/players`, `GET /api/player-lab`, `GET /api/matches`
**Code:** `backend/api/main.py` (lines 232–287), data produced by `spark_processor.py`
**UI tab:** 🧬 Player Intelligence Lab

**How it works.**
1. `/api/players?query=&min_matches=` filters the pre-computed `player_aggregations.json`
   cache, keeps players with ≥ `min_matches` charted matches, substring-matches the query
   (case-insensitive) and sorts descending by match count — this powers both dropdowns.
2. `/api/player-lab?player_a=&player_b=&surface=` resolves each player's profile and picks
   the stat block for the requested surface (falling back to the `overall` block when the
   player has no matches on that surface):

```python
def get_stats_for_surface(prof, s_name):
    if s_name == "Overall" or s_name not in prof.get("surfaces", {}):
        return prof.get("overall", {})
    return prof.get("surfaces", {}).get(s_name, {})
```

3. Head-to-head is fetched from the rivalry graph (`rivalry_network.get_rivalry_details`),
   which looks up both key orderings (`A_vs_B` and `B_vs_A`).
4. `/api/matches` is a paginated MongoDB query with optional `Surface` filter and
   `$or` player filter over the `matches` collection.

**Stats surfaced per player:** `ace_rate`, `df_rate`, `first_in_pct`, `first_won_pct`,
`second_won_pct`, `bp_save_pct`, `return_won_pct`, `winner_unforced_ratio`, plus raw
counters (serve pts, aces, DFs, break pts saved, winners, unforced errors).

**Flow:**
```
player_aggregations.json ──▶ API cache ──▶ GET /api/player-lab ──▶ surface block + H2H
rivalry_network.json ─────────────────────────────┘
```

---

## Feature 2 — Match Momentum Analyzer (DGIM + EWMA)

**Endpoints:** `GET /api/matches/replayable`, `GET /api/momentum/replay`
**Code:** `backend/algorithms/dgim.py` (407 lines)
**UI tab:** ⚡ Match Momentum — interactive point-by-point replay with scrubber

### Algorithm A: DGIM sliding-window counter (Datar–Gionis–Indyk–Motwani)

Counts "1" bits (e.g. "player 1 won the point") inside the last *N* stream positions
using only **O(log² N)** memory instead of storing the window.

**Data structure:** an ordered list of buckets, each holding `(timestamp, size)` where
**size is always a power of two**, with **at most k = 2 buckets of the same size**.

**Insert (`add_bit`):**
```python
# expire anything strictly outside the window
min_valid_time = t - self.window_size + 1
self.buckets = [b for b in self.buckets if b.timestamp >= min_valid_time]
if bit == 1:
    self.buckets.insert(0, DGIMBucket(t, 1))   # new size-1 bucket
    self._merge_buckets()                      # restore the invariant
```

**Merge rule** — when more than *k* buckets share a size, the two **oldest** merge into
one of double the size (keeping the newer timestamp, which is the conservative choice):

```python
if len(matching_indices) > self.k:
    idx1, idx2 = matching_indices[-2], matching_indices[-1]
    merged_bucket = DGIMBucket(self.buckets[idx1].timestamp, size * 2)
    self.buckets[idx1] = merged_bucket
    self.buckets.pop(idx2)
    size *= 2          # cascade to the next power of two
```

**Query (`count`)** — sum every fully-covered bucket, add **half of the oldest
overlapping bucket** (it may be partially outside the window):

```python
if min_time <= 1:                       # window covers the whole stream → exact
    est = sum(b.size for b in relevant_buckets)
else:
    est = sum(b.size for b in relevant_buckets[:-1])
    oldest = relevant_buckets[-1]
    est += oldest.size // 2 if oldest.size > 1 else oldest.size
```

**Error bound** returned by the code: `theoretical_max_error_pct = 100 / (2·(k−1))`,
i.e. **50% worst-case for standard k = 2** (the theoretical DGIM guarantee; the unit
tests additionally verify ≤ 25% on well-formed streams and 0% error while the stream is
shorter than the window). A `raw_window` ground-truth list is maintained alongside so
the API can report *estimated vs exact vs error %* live — the tests in
`test_algorithms.py` assert bucket sizes are powers of two, ≤ k buckets per size, and
that expired buckets never appear in a query.

### Algorithm B: EWMA momentum with leverage weighting

Each point gets a **leverage/impact weight** from real tennis situations
(`evaluate_point_impact`):

| Situation | Base weight |
|---|---|
| Match point won/saved | 4.0 |
| Set point | 3.0 |
| Break point (0-40/15-40/30-40/40-AD) | 2.5 |
| Tiebreak point | 2.0 |
| Game point (40-0/40-15/40-30/AD-40) | 1.5 |
| Standard point | 1.0 |
| + Ace or Double Fault bonus | +0.5 |

Sign convention: `impact = +weight` if Player 1 won, `−weight` if Player 2 won.
Then each player runs an independent **Exponentially Weighted Moving Average**:

```python
# M_{i,t} = alpha * impact_{i,t} + (1 - alpha) * M_{i,t-1}
self.p1_momentum = (self.alpha * p1_imp) + ((1.0 - self.alpha) * self.p1_momentum)
self.p2_momentum = (self.alpha * p2_imp) + ((1.0 - self.alpha) * self.p2_momentum)
# net differential momentum
self.momentum_score = self.p1_momentum - self.p2_momentum
```

Defaults: `alpha = 0.25` (editable in the UI, 0.05–0.9), `window_size = 40` points.
Simultaneously two DGIM instances consume the same stream — `dgim_p1` gets a `1` when
Player 1 wins the point, `dgim_p2` when Player 2 wins — giving approximate "points won
in the last 40" per player with sub-linear memory.

### Request flow

```
UI (match select, window, alpha)
   └─▶ GET /api/momentum/replay?match_id=...&window_size=40&alpha=0.25
         ├─▶ clean_points.parquet filtered by match_id, sorted by Pt
         ├─▶ MomentumAnalyzer(alpha, window_size).replay_match(points, p1, p2)
         │      ├─ point 0 = pre-match baseline snapshot
         │      └─ for each point: leverage → EWMA update → DGIM add_bit → snapshot
         ├─▶ momentum swings = count of sign changes in the momentum curve
         └─▶ JSON: points[] (per-point snapshots), curve[] (chart series), metrics
```

The response embeds `dgim_p1_window_wins` (estimate), `dgim_p1_exact_wins` (ground
truth) and `error_percentage` per point, so the UI can plot both series and show the
approximation error live. `replay_match()` also counts **momentum swings** — every time
the momentum curve crosses zero:

```python
if (curr_m > 0 and prev_m < 0) or (curr_m < 0 and prev_m > 0):
    shifts += 1
```

---

## Feature 3 — Serve Strategy Heatmap

**Endpoint:** `GET /api/serve-strategy?player=&surface=`
**Code:** zone aggregation in `spark_processor.py`, serving in `main.py` (lines 337–358)
**UI tab:** 🎾 Serve Strategy — court-zone heatmap + fault breakdown

**How it works.** The Match Charting Project tags every serve with a location in the
service box. `spark_processor.py` partitions the court into the classic 2 × 3 grid:

- **Deuce court** and **Ad court** × **Wide / Body (middle) / T** → 6 placement cells
- Fault trajectories are binned into 4 error zones: **net, wide, deep, foot fault**

```python
spark.sql("""
    SELECT s.player, m.Surface,
           SUM(s.deuce_wide) as deuce_wide, SUM(s.deuce_middle) as deuce_middle, SUM(s.deuce_t) as deuce_t,
           SUM(s.ad_wide) as ad_wide,       SUM(s.ad_middle) as ad_middle,       SUM(s.ad_t) as ad_t,
           SUM(s.err_net) as err_net, SUM(s.err_wide) as err_wide,
           SUM(s.err_deep) as err_deep, SUM(s.err_foot) as err_foot
    FROM serve_dir s JOIN matches m ON s.match_id = m.match_id
    GROUP BY s.player, m.Surface
""")
```

Per surface the raw counts are converted to percentages (with a 33.3% neutral default
when a side has zero recorded serves), producing the JSON shape the heatmap renders:

```python
surf_data = {
    "deuce_court": {"wide": dw, "body": dm, "t": dt, "total": total_deuce,
                    "wide_pct": ..., "body_pct": ..., "t_pct": ...},
    "ad_court":    {"wide": aw, "body": am, "t": at, ...},
    "faults":      {"net": enet, "wide": ewide, "deep": edeep, "foot": efoot,
                    "total_faults": total_err, "net_pct": ..., "wide_pct": ..., "deep_pct": ...}
}
```

**Flow:**
```
charting-m-stats-ServeDirection.csv ──▶ clean_serve_direction.parquet
      ──▶ Spark SQL GROUP BY player, Surface ──▶ serve_strategy.json ──▶ API cache
      └──▶ GET /api/serve-strategy ──▶ UI heatmap (6 zones + fault bars)
```
If the requested surface has no data for that player, the endpoint transparently falls
back to the first available surface and reports `available_surfaces` so the UI can relabel.

---

## Feature 4 — Similar Playstyle Recommender (Cosine Similarity)

**Endpoint:** `GET /api/recommender?player=&top_k=`
**Code:** `backend/algorithms/recommender.py`
**UI tab:** 🎯 Playstyle Recommender — ranked cards with radar comparison

**How it works.** Every qualified player (≥ `min_matches`, default 5) becomes an
**8-dimensional feature vector** drawn from `player_aggregations.json`:

```python
FEATURE_NAMES = ["ace_rate", "df_rate", "first_in_pct", "first_won_pct",
                 "second_won_pct", "bp_save_pct", "return_won_pct",
                 "winner_unforced_ratio"]
```

**Step 1 — honest imputation.** Zeros (missing stats) are replaced with the population
column median, treating missing as NaN so medians ignore them:

```python
col_medians = np.nanmedian(np.where(self.feature_matrix == 0, np.nan, self.feature_matrix), axis=0)
col_medians = np.nan_to_num(col_medians, nan=50.0)
zero_mask = self.feature_matrix[:, j] == 0
self.feature_matrix[zero_mask, j] = col_medians[j]
```

**Step 2 — Z-score scaling** via `sklearn.preprocessing.StandardScaler`, so a
percentage point in ace rate and a ratio in winners/unforced errors contribute fairly.

**Step 3 — Cosine Similarity** against all qualified players:

```python
target_vec = self.normalized_matrix[target_idx].reshape(1, -1)
similarities = cosine_similarity(target_vec, self.normalized_matrix)[0]
ranked_indices = np.argsort(similarities)[::-1]   # drop self, take top_k
```

$$\text{cos}(\vec{a}, \vec{b}) = \frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\,\|\vec{b}\|}$$

**Step 4 — explainability.** For each recommendation, the 3 features with the smallest
absolute normalized distance become `key_similar_traits` (feature label, both values,
raw difference), so the UI can say *why* the players are similar. Each player is also
assigned a tactical archetype via rule-based thresholds:

```python
if ace_rate >= 11.0 and f_won >= 75.0:      "Big Server / Serve-Dominant"
elif ret_won >= 40.0 and wu_ratio <= 1.15:  "Counterpuncher / Defensive Specialist"
elif wu_ratio >= 1.4 and f_won >= 73.0:     "Aggressive Baseliner / Power Hitter"
elif ret_won >= 38.0 and ace_rate >= 7.0:   "All-Court Elite / Complete Specialist"
else:                                        "Tactical Baseliner"
```

**Flow:**
```
player_aggregations.json ──▶ median imputation ──▶ StandardScaler (z-score)
   ──▶ cosine_similarity(target, all) ──▶ sort desc ──▶ top_k + attribution + archetype
   └──▶ GET /api/recommender ──▶ UI cards + radar charts
```

---
## Feature 5 — ATP Rivalry Network (Louvain Community Detection & PageRank)

**Endpoints:** `GET /api/rivalry-network`, `GET /api/rivalry-details`
**Code:** `backend/algorithms/graph_analytics.py`, edges built in `spark_processor.py`
**UI tab:** 🌐 Rivalry Network — interactive force graph with community colors

**How it works.** Every player pair with ≥ 2 meetings becomes an edge in an undirected
`networkx` graph. Edge attributes carry the head-to-head record plus two derived metrics:

```python
# 1.0 when perfectly even (50-50), lower if lopsided
competitiveness = round((2.0 * min(w_a, w_b)) / max(1, v_total), 2) if v_total > 0 else 0.5
intensity = round(total * (0.5 + 0.5 * competitiveness), 2)

self.G.add_edge(p_a, p_b, weight=total, meetings=total, wins_a=w_a, wins_b=w_b,
                competitiveness=competitiveness, intensity=intensity, surfaces=riv.get("surfaces", {}))
```

**Algorithm A — Louvain Community Detection** (`networkx.algorithms.community.louvain_communities`,
seeded for reproducibility). Louvain greedily moves nodes between communities to maximize
**modularity**:

$$Q = \frac{1}{2m}\sum_{ij}\left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i, c_j)$$

Two phases repeat until Q stops improving: (1) **local modularity optimization** — each
node is tentatively moved to the neighbouring community giving the largest ΔQ; (2)
**graph aggregation** — communities are contracted into super-nodes and the process
repeats on the coarsened graph. Weighted edges mean frequently-played rivalries pull
players into the same cluster; because ATP tour structure correlates with era and rank,
the resulting communities often correspond to competitive cohorts/eras.

```python
self.communities = louvain_communities(self.G, weight="weight", seed=42)
self.modularity_score = round(modularity(self.G, self.communities, weight="weight"), 4)
```

**Algorithm B — PageRank dominance centrality** (damping α = 0.85, weighted by meetings):

$$\text{PR}(u) = \frac{1-\alpha}{N} + \alpha \sum_{v \in \text{In}(u)} \frac{\text{PR}(v)}{|\text{Out}(v)|}$$

computed iteratively until convergence. High-PageRank players sit at the centre of many
/heavy rivalries — a data-driven "dominance centrality".

**Algorithm C — Betweenness centrality** (`nx.betweenness_centrality`) identifies
"bridge" players who connect otherwise separate eras/communities (useful for spotting
players who faced both cohorts).

**Rendering the view** (`get_network_data`):
- Nodes ranked by `pagerank × total_matches`, top N (default 40) shown.
- Edges kept only if `meetings >= min_meetings` (UI slider, default 3).
- Each community gets a color and a generated label from its top-2 PageRank members
  (e.g. `"Cluster 1: Djokovic, Nadal era"`), and key players per cluster are surfaced.

**Flow:**
```
rivalry_network.json ──▶ build weighted nx.Graph ──▶ Louvain (Q score) + PageRank + Betweenness
   ──▶ filter top-N nodes / min-meeting edges ──▶ nodes[] + edges[] + community_clusters[]
   └──▶ GET /api/rivalry-network ──▶ UI force-directed graph; click edge ──▶ /api/rivalry-details
```

---

## Feature 6 — Stream Integrity Monitor (Bloom Filter + MongoDB Persistence)

**Endpoints:** `GET /api/stream/status`, `POST /api/stream/step`, `POST /api/stream/inject-duplicate`,
`POST /api/stream/reset`, `GET /api/stream/events`, `GET /api/health/mongo`
**Code:** `backend/algorithms/bloom_filter.py`, stream loop in `main.py` (lines 412–588), storage in `mongo_store.py`
**UI tab:** 🌐 Stream Integrity — live counters, FP-rate gauges, MongoDB inspector

**How it works.** A pre-built index of 2,000 real point events (`event_id = "<match_id>_pt_<n>"`)
is replayed through a streaming dedup pipeline. The goal is to catch duplicate events
using sub-linear memory, persist each unique event, and *measure* how well the
approximate structure behaves versus ground truth.

### The Bloom filter (Kirsch–Mitzenmacher double hashing)

A bit-array of `m` bits with `k` hash functions. To test membership, compute `k` bit
positions and check whether all are set. RallyIQ derives the `k` positions from just
two base hashes (the KM optimization) instead of `k` independent hashes:

```python
h1 = int(hashlib.md5(data).hexdigest(), 16)
h2 = int(hashlib.sha256(data).hexdigest(), 16)
pos = (h1 + i * h2) % self.m      # for i in 0..k-1
```

Insert = set all `k` bits; membership = all `k` bits set. Defaults `m=8192`, `k=5`.

**Theoretical false-positive probability** (probability a *non-member* is wrongly reported
as present):

$$p = \left(1 - e^{-kn/m}\right)^{k}$$

```python
exponent = -(float(self.k) * float(self.n)) / float(self.m)
p = math.pow(1.0 - math.exp(exponent), self.k)
```

**Empirical false-positive rate** is tracked separately by checking every query against an
exact Python `set` ground truth, counting only *negative* queries (absent items):

```python
negative_queries = self.false_positive_count + self.true_negative_count
return float(self.false_positive_count / negative_queries)
```

### Safe streaming: Bloom + DB consistency

A naive pipeline commits to the filter *before* confirming the database write, which can
desync the two. RallyIQ uses an **optimistic-verify-commit** protocol per event
(`/api/stream/step`):

```
event = next_event
1. query_with_verification(event_id)   # bloom answer + exact-set ground truth
2. if exact_set confirms duplicate:
       record_duplicate();  status="duplicate_rejected";  continue   # NOT persisted
3. try persist to MongoDB stream_events:
       on DB failure -> DO NOT commit to Bloom; status="persistence_error"; continue
4. on DB success -> stream_bloom_filter.commit_new_item(event_id)    # only now set bits
```

Key invariant: **the Bloom filter is only updated after a successful DB write**, so the
filter can never claim an item that isn't actually stored. `commit_new_item` returns
whether the item *was* a false positive at insert time (bits set but item not in the
exact set), letting the UI distinguish a true duplicate from a coincidence.

**Reset** (`/api/stream/reset?m=&k=`) rebuilds the bit array with a caller-chosen `m`/`k`,
resets the pointer, and purges `stream_events` — useful for re-running saturation
experiments and watching the FP rate climb as `n` grows.

**Flow:**
```
UI "Step 5" / "Start Live Stream" / "Inject Duplicate"
   └─▶ POST /api/stream/step (count)
         ├─▶ Bloom query + exact-set verification
         ├─▶ duplicate? reject before write
         ├─▶ persist unique event ──▶ mongo stream_events (or disk fallback)
         ├─▶ commit_new_item (only on success)
         └─▶ return per-event status + live bloom metrics + mongo count
   └─▶ GET /api/health/mongo  (ping / engine / collection stats)
```

---

## Feature 7 — ML Match Winner Predictor (Random Forest vs Logistic Regression)


**Endpoints:** `GET /api/predict/status`, `GET /api/predict/match`, `POST /api/predict/match`
**Code:** `backend/ml/train_predictor.py` (training), `backend/ml/predictor.py` (inference)
**Artifacts:** `models/match_predictor.joblib`, `models/model_metrics.json`
**UI tab:** 🤖 ML Match Predictor

### Zero-target-leakage feature engineering

The central design constraint: **a match's features must only use information known
before that match started.** `extract_chronological_features()` sorts every *verified*
match by date and walks it forward, maintaining rolling per-player state in dictionaries:

```python
player_matches / player_wins            # career record
player_surface_matches / _wins          # per-surface record
player_recent_form                      # last-10 results (1=win, 0=loss)
h2h_wins[(p1, p2)]                      # head-to-head wins
```

For each match it reads the current state to build the feature vector, **records the
sample(s), and only then updates the state** with the result — so no future outcome leaks
into a past row:

```python
# ... build feature_vec from current state, append record(s) ...
# UPDATE ROLLING STATE ONLY AFTER RECORDING PRE-MATCH FEATURES
player_matches[p1] += 1; player_matches[p2] += 1
if y == 1:
    player_wins[p1] += 1; h2h_wins[(p1, p2)] += 1; ...
```

Each match yields a 14-feature differential vector (`win_rate_diff`,
`surface_win_rate_diff`, `recent_form_diff`, `h2h_diff`, `h2h_total`, `exp_diff`,
absolute baselines `p1/p2_win_rate` & surface rates, one-hot surface, `best_of`). It is
**duplicated in the reverse orientation** with the target flipped (`1 - y`), enforcing the
symmetry constraint P(A,B) = 1 − P(B,A) and doubling the sample count (11,138 samples
from 5,569 verified matches).

### Chronological split & models

Splits are by time, never random: **train** (2010–2023), **validation** (2023–2024),
**holdout test** (2024–2026). Two models are trained and compared:
- **Primary:** `RandomForestClassifier(n_estimators=100, max_depth=6)` → test accuracy **67.94%**, ROC-AUC **0.7505**
- **Baseline:** `LogisticRegression` on `StandardScaler`-scaled features → test accuracy **67.28%**, ROC-AUC **0.7350**

Retrain with `./venv/bin/python3 -m backend.ml.train_predictor`, which overwrites the
joblib + metrics artifacts.

### Symmetrical-averaging inference

`predictor.predict()` rebuilds the same differential feature vector from the stored final
rolling state, then averages the forward and inverted-model probabilities so the answer is
order-independent:

```python
probs_fwd = self.model.predict_proba(X_df)[0]
probs_rev = self.model.predict_proba(X_df_rev)[0]
prob_a = float(np.clip(0.5 * (probs_fwd[1] + (1.0 - probs_rev[1])), 0.01, 0.99))
```

The response includes both RF and baseline probabilities, a confidence label (High if the
probability margin ≥ 0.35, Moderate ≥ 0.15, else Low), the head-to-head prior, and an
interpretability block (`key_factors`). Unseen players get a neutral 0.5 prior and a
`coverage_notice` rather than an error. Each prediction is appended to a
`prediction_audit` MongoDB collection with a timestamp.

**Flow:**
```
clean_matches.parquet (Verified only)
   └─▶ chronological walk ──▶ 14-feature vectors + reverse copies
         └─▶ time split ──▶ RandomForest vs LogisticRegression ──▶ metrics + feature_importances
               └─▶ joblib artifact ──▶ predictor singleton
UI matchup ──▶ GET/POST /api/predict/match ──▶ build features from final rolling state
   ──▶ symmetrical probability average ──▶ winner + confidence + key_factors (+ audit log)
```

---

## Storage Layer (`backend/storage/mongo_store.py`)

A process-wide **singleton** (`MongoStorage`, `__new__` guards re-initialization) that
abstracts the document store behind one object, `mongo_store.db`. On init it:

1. **Attempts a live connection** to `MONGO_URI` from the root `.env` with short timeouts
   (`serverSelectionTimeoutMS=4000`) and a `certifi` CA bundle for Atlas TLS. A `ping`
   command authenticates the connection before it is accepted.
2. **Reports status honestly** — `connected_atlas`, `connected_local`, or
   `fallback_mock`. If the cluster is unreachable (e.g. sandboxed/offline), it transparently
   switches to an in-process **`mongomock`** store so the app never hard-fails.
3. **Masks credentials** in every log line via `mask_mongo_uri()` (`mongodb+srv://****:****@...`)
   so connection strings and error strings can't leak secrets.
4. **Persists to disk** (`persist_to_disk()` → `data/processed/mongo_storage.json`) so the
   mock engine survives restarts; `_load_from_disk()` rehydrates on boot.

Collections: `matches`, `points`, `player_profiles`, `stream_events`, `stream_metrics`,
`prediction_audit`. `get_stats()` returns engine + per-collection counts;
`test_connection()` (`/api/health/mongo`) performs a live ping and lists databases/collections.

## Configuration (`backend/config.py`)

Loads `.env` (python-dotenv) and centralizes every path and tuning constant:

```python
SPARK_MASTER = "local[2]"; SPARK_DRIVER_MEMORY = "1g"; SPARK_EXECUTOR_MEMORY = "1g"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_BLOOM_CAPACITY, DEFAULT_BLOOM_BITS, DEFAULT_BLOOM_HASHES = 10000, 8192, 5
DEFAULT_DGIM_WINDOW, DEFAULT_MOMENTUM_ALPHA = 40, 0.25
```

The `local[2]` / `1g` settings keep the whole pipeline under ~1.2 GB RAM on an 8 GB Mac.

## REST API Reference (25 routes)

| Method & Path | Purpose | Key params |
|---|---|---|
| `GET /api/overview` · `/api/dashboard/overview` | Dataset size, matches/points, engine + ML status | — |
| `GET /api/dashboard/surfaces` | Surface distribution + percentages | — |
| `GET /api/dashboard/top-players` | Top players by matches | `limit` |
| `GET /api/dashboard/recent-events` | Latest stream events | `limit` |
| `GET /api/dashboard/live` | Live Bloom/DGIM/Mongo telemetry | — |
| `GET /api/matches` | Paginated match records | `limit, offset, surface, player` |
| `GET /api/players` | Qualified player list (dropdowns) | `query, min_matches` |
| `GET /api/player-lab` | Two-player surface comparison + H2H | `player_a, player_b, surface` |
| `GET /api/matches/replayable` | Matches available for momentum replay | — |
| `GET /api/momentum/replay` | DGIM + EWMA point-by-point replay | `match_id, window_size, alpha` |
| `GET /api/serve-strategy` | Court-zone placement + fault stats | `player, surface` |
| `GET /api/recommender` | Cosine playstyle neighbours | `player, top_k` |
| `GET /api/rivalry-network` | Louvain/PageRank graph payload | `min_meetings, top_n` |
| `GET /api/rivalry-details` | Full head-to-head for a pair | `player_a, player_b` |
| `GET /api/stream/status` | Bloom + stream metrics | — |
| `POST /api/stream/step` | Ingest next N events (dedup + persist) | `count` |
| `POST /api/stream/inject-duplicate` | Re-inject a seen event to demo dedup | — |
| `POST /api/stream/reset` | Rebuild filter, purge stream events | `m, k` |
| `GET /api/stream/events` | Recent persisted stream events | `limit` |
| `GET /api/health/mongo` · `/api/health/storage` | Live connectivity + collection stats | — |
| `GET /api/predict/status` | Model readiness + test metrics | — |
| `GET /api/predict/match` | Predict winner (query string) | `player_a, player_b, surface, best_of` |
| `POST /api/predict/match` | Predict winner (JSON body) | `MatchPredictionRequest` |
| `GET /` · `/static/*` | Serves the frontend SPA + assets | — |

## Test Suite (45 tests)

Run with `./venv/bin/python3 -m unittest discover -s backend/tests`. Coverage:
- **`test_algorithms.py`** — DGIM counting/power-of-two buckets/window expiry & error
  bound; MomentumAnalyzer leverage/EWMA/momentum-swing behaviour; Bloom filter
  (theoretical vs empirical FP, duplicate vs false-positive distinction, negative-query
  accounting); recommender cosine ranking; Louvain + PageRank on the real graph.
- **`test_api.py`** — endpoint response contracts (overview, player-lab, recommender,
  serve-strategy, rivalry, stream, predict).
- **`test_predictor.py`** — predictor artifact loading, probability symmetry
  P(A,B)=1−P(B,A), confidence labelling; plus Mongo URI masking and health-check shape.

## End-to-End Data Flow (summary)

```
raw ATP CSVs ─▶ ingest_validate ─▶ clean Parquet
                    │                    │
                    │                    └─▶ spark_processor ─▶ player_aggregations /
                    │                                          rivalry_network / serve_strategy
                    │                                          └─▶ MongoDB (+ disk mirror)
                    └─▶ replay_indexer ─▶ replayable_matches + stream_sample_events
                                            │
        models/ (train_predictor) ──────────┤
                                            ▼
                          FastAPI (backend/api/main.py)  ──▶  vanilla-JS dark UI
                          DGIM / Bloom / Recommender / Graph / RF predictor
```


