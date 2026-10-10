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

| # | Feature | Big Data Algorithm | Mathematical / Algorithmic Formulation |
|---|---|---|---|
| 1 | **Player Intelligence Lab** | Distributed Spark Aggregation | Distributed joins & MapReduce-style multi-key aggregations across surface & career dimensions |
| 2 | **Match Momentum Analyzer** | DGIM (Datar-Gionis-Indyk-Motwani) + EWMA | Sliding-window bit counting with power-of-two bucket hierarchy, error $\le \frac{1}{2k} = 25\%$, $O(\log^2 N)$ memory |
| 3 | **Serve Strategy Heatmap** | Directional & Fault Zone Clustering | Court spatial partitioning into Deuce/Ad service boxes (Wide, Body, T) with error trajectories |
| 4 | **Playstyle Recommender** | Normalized Cosine Similarity | Z-score feature scaling across 8 performance dimensions, Cosine metric, explainable attribution |
| 5 | **ATP Rivalry Network** | Louvain Community Detection & PageRank | Modularity optimization $Q = \sum [e_{ii} - a_i^2]$ and directed eigenvector dominance centrality |
| 6 | **Stream Integrity Monitor** | Bloom Filter with Double Hashing | $k$ hash functions via $h_i(x) = (h_1(x) + i \cdot h_2(x)) \pmod m$, theoretical FP $p = (1 - e^{-kn/m})^k$ |

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

### 1. Run Automated Test Suite (18 Tests)
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
- Storage layer seamlessly connects to live MongoDB if running, or utilizes an in-process embedded MongoDB engine (`mongomock`) with disk persistence.
