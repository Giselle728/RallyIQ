"""
API Integration Tests for RallyIQ
Verifies all FastAPI REST endpoints across the 6 features.
"""

import unittest
from fastapi.testclient import TestClient
import mongomock
from backend.api.main import app
from backend.storage.mongo_store import mongo_store

class TestRallyIQApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Isolate stream_events collection so automated test runs never clear or overwrite real Atlas data
        cls._orig_stream_events = mongo_store.db.stream_events
        cls.mock_db = mongomock.MongoClient()["test_stream_db"]
        mongo_store.db.stream_events = cls.mock_db.stream_events

    @classmethod
    def tearDownClass(cls):
        mongo_store.db.stream_events = cls._orig_stream_events

    def test_overview_endpoint(self):
        res = self.client.get("/api/overview")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("dataset_report", data)
        self.assertIn("raw_size_mb", data["dataset_report"])
        self.assertGreaterEqual(data["dataset_report"]["raw_size_mb"], 50.0)
        self.assertIn("storage", data)
        self.assertIn("collections", data["storage"])

    def test_dashboard_compatibility_endpoints(self):
        # 1. /api/dashboard/overview
        res1 = self.client.get("/api/dashboard/overview")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertIn("total_matches", data1)
        self.assertIn("total_points", data1)
        
        # 2. /api/dashboard/surfaces
        res2 = self.client.get("/api/dashboard/surfaces")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertIn("surfaces", data2)
        self.assertIn("breakdown", data2)
        
        # 3. /api/dashboard/top-players
        res3 = self.client.get("/api/dashboard/top-players?limit=5")
        self.assertEqual(res3.status_code, 200)
        data3 = res3.json()
        self.assertIn("top_players", data3)
        self.assertEqual(len(data3["top_players"]), 5)
        
        # 4. /api/dashboard/recent-events
        res4 = self.client.get("/api/dashboard/recent-events?limit=5")
        self.assertEqual(res4.status_code, 200)
        data4 = res4.json()
        self.assertIn("recent_events", data4)
        
        # 5. /api/dashboard/live
        res5 = self.client.get("/api/dashboard/live")
        self.assertEqual(res5.status_code, 200)
        data5 = res5.json()
        self.assertEqual(data5["status"], "online")
        
        # 6. /api/matches
        res6 = self.client.get("/api/matches?limit=10&offset=0")
        self.assertEqual(res6.status_code, 200)
        data6 = res6.json()
        self.assertIn("matches", data6)

    def test_players_endpoint(self):
        res = self.client.get("/api/players?min_matches=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("players", data)
        self.assertGreater(len(data["players"]), 0)

    def test_player_lab_endpoint(self):
        res = self.client.get("/api/player-lab?player_a=Novak Djokovic&player_b=Rafael Nadal&surface=Overall")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["player_a"]["name"], "Novak Djokovic")
        self.assertEqual(data["player_b"]["name"], "Rafael Nadal")
        self.assertIn("head_to_head", data)

    def test_replayable_matches_endpoint(self):
        res = self.client.get("/api/matches/replayable")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("matches", data)
        self.assertGreater(len(data["matches"]), 0)

    def test_momentum_replay_endpoint(self):
        # Fetch first replayable match
        matches_res = self.client.get("/api/matches/replayable")
        self.assertEqual(matches_res.status_code, 200)
        matches = matches_res.json()["matches"]
        self.assertGreater(len(matches), 0)
        first_match = matches[0]
        m_id = first_match["match_id"]
        
        # Verify match metadata provenance
        self.assertIn("tournament", first_match)
        self.assertIn("year", first_match)
        self.assertIn("points_count", first_match)
        self.assertIn("player_1", first_match)
        self.assertIn("player_2", first_match)
        
        res = self.client.get(f"/api/momentum/replay?match_id={m_id}&window_size=30&alpha=0.25")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("curve", data)
        self.assertIn("points", data)
        self.assertIn("snapshot", data)
        self.assertGreater(len(data["points"]), 0)
        
        # Verify initial snapshot is point 0 with zero points won, zero momentum, zero DGIM count
        snap = data["snapshot"]
        self.assertEqual(snap["point_number"], 0)
        self.assertEqual(snap["winner"], 0)
        self.assertEqual(snap["dgim_p1_window_wins"], 0)
        self.assertEqual(snap["dgim_p1_exact_wins"], 0)
        self.assertEqual(snap["momentum"], 0.0)
        self.assertEqual(snap["dominant_player"], "Neutral")
        
        # Verify curve starts at point 0
        self.assertEqual(data["curve"][0]["pt"], 0)
        self.assertEqual(data["curve"][0]["momentum"], 0.0)
        
        # Verify point 1 has updated DGIM, momentum, and consistent fields
        pt1 = data["points"][1]
        self.assertEqual(pt1["point_number"], 1)
        self.assertIn(pt1["winner"], [1, 2])
        self.assertIn("p1_momentum", pt1)
        self.assertIn("p2_momentum", pt1)
        self.assertIn("dgim_p1_buckets", pt1)
        self.assertIn("theoretical_max_error_pct", pt1)

    def test_serve_strategy_endpoint(self):
        res = self.client.get("/api/serve-strategy?player=Roger Federer&surface=Hard")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["player"], "Roger Federer")
        self.assertIn("strategy", data)
        self.assertIn("deuce_court", data["strategy"])
        self.assertIn("ad_court", data["strategy"])
        self.assertIn("faults", data["strategy"])

    def test_recommender_endpoint(self):
        res = self.client.get("/api/recommender?player=Carlos Alcaraz&top_k=3")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["target_player"], "Carlos Alcaraz")
        self.assertIn("recommendations", data)
        self.assertEqual(len(data["recommendations"]), 3)
        self.assertIn("archetype", data["recommendations"][0])

    def test_rivalry_network_endpoint(self):
        res = self.client.get("/api/rivalry-network?min_meetings=3&top_n=25")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("community_clusters", data)
        self.assertGreater(len(data["nodes"]), 0)

    def test_stream_pipeline_endpoints(self):
        # 1. Reset filter and purge isolated mock stream collection
        reset_res = self.client.post("/api/stream/reset?m=4096&k=4")
        self.assertEqual(reset_res.status_code, 200)
        reset_data = reset_res.json()
        self.assertEqual(reset_data["bloom_status"]["inserted_items_n"], 0)
        self.assertEqual(reset_data["mongo_persisted_events"], 0)
        
        # 2. Step stream
        step_res = self.client.post("/api/stream/step?count=5")
        self.assertEqual(step_res.status_code, 200)
        step_data = step_res.json()
        self.assertEqual(step_data["processed_count"], 5)
        self.assertEqual(step_data["bloom_status"]["inserted_items_n"], 5)
        self.assertEqual(step_data["mongo_persisted_events"], 5)
        
        # 3. Verify persisted events returned
        events_res = self.client.get("/api/stream/events?limit=8")
        self.assertEqual(events_res.status_code, 200)
        events_data = events_res.json()
        self.assertEqual(len(events_data["events"]), 5)

        # 4. Inject duplicate
        dup_res = self.client.post("/api/stream/inject-duplicate")
        self.assertEqual(dup_res.status_code, 200)
        dup_data = dup_res.json()
        self.assertTrue(dup_data["bloom_detected_duplicate"])
        self.assertTrue(dup_data["is_confirmed_duplicate"])
        self.assertTrue(dup_data["duplicate_rejected"])
        self.assertEqual(dup_data["bloom_status"]["duplicate_rejections"], 1)
        # CRUCIAL: Unique n must NOT increase on duplicate injection
        self.assertEqual(dup_data["bloom_status"]["inserted_items_n"], 5)
        # CRUCIAL: Persisted document count in MongoDB must NOT increase
        self.assertEqual(dup_data["mongo_persisted_events"], 5)
        
        # 5. Status verification: check theoretical FP precision (not rounded to 0.0)
        status_res = self.client.get("/api/stream/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        bf_status = status_data["bloom_filter"]
        self.assertEqual(bf_status["inserted_items_n"], 5)
        self.assertEqual(status_data["mongo_persisted_events"], 5)
        self.assertGreater(bf_status["theoretical_fp_prob"], 0.0)
        self.assertLess(bf_status["theoretical_fp_prob"], 1e-4)

        # 6. Reset again and verify purge
        reset_res_2 = self.client.post("/api/stream/reset?m=4096&k=4")
        self.assertEqual(reset_res_2.json()["mongo_persisted_events"], 0)
        events_res_empty = self.client.get("/api/stream/events?limit=8")
        self.assertEqual(len(events_res_empty.json()["events"]), 0)

    def test_stream_duplicate_rejection_and_negative_query_semantics(self):
        # Reset isolated stream
        self.client.post("/api/stream/reset?m=8192&k=5")
        
        # Step 3 items
        r1 = self.client.post("/api/stream/step?count=3")
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertEqual(d1["processed_count"], 3)
        self.assertEqual(d1["bloom_status"]["inserted_items_n"], 3)
        self.assertEqual(d1["mongo_persisted_events"], 3)
        
        # Inject duplicate
        r_dup = self.client.post("/api/stream/inject-duplicate")
        d_dup = r_dup.json()
        self.assertTrue(d_dup["duplicate_rejected"])
        self.assertTrue(d_dup["bloom_detected_duplicate"])
        self.assertTrue(d_dup["is_confirmed_duplicate"])
        # MongoDB document count must NOT change
        self.assertEqual(d_dup["mongo_persisted_events"], 3)
        # Unique inserted items count n must NOT change
        self.assertEqual(d_dup["bloom_status"]["inserted_items_n"], 3)
        # Duplicate rejections must increment to 1
        self.assertEqual(d_dup["bloom_status"]["duplicate_rejections"], 1)

        # Status endpoint verification
        r_status = self.client.get("/api/stream/status")
        d_status = r_status.json()
        bf = d_status["bloom_filter"]
        self.assertEqual(bf["inserted_items_n"], 3)
        self.assertEqual(d_status["mongo_persisted_events"], 3)
        self.assertEqual(bf["exact_set_size"], 3)
        self.assertEqual(bf["duplicate_rejections"], 1)
        self.assertGreater(bf["negative_queries_tested"], 0)
        self.assertIsInstance(bf["theoretical_fp_prob"], float)
        self.assertGreater(bf["theoretical_fp_prob"], 0.0)

    def test_storage_health_endpoint(self):
        res = self.client.get("/api/health/mongo")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("engine", data)
        self.assertIn("database", data)
        self.assertEqual(data["database"], "rallyiq_tennis")

    def test_predict_status_endpoint(self):
        res = self.client.get("/api/predict/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("is_ready"))
        self.assertEqual(data.get("status"), "ready")

    def test_predict_match_endpoints(self):
        # GET endpoint
        res_get = self.client.get("/api/predict/match?player_a=Carlos Alcaraz&player_b=Jannik Sinner&surface=Hard")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertIn("prediction", data_get)
        self.assertIn("predicted_winner", data_get["prediction"])
        self.assertIn("baseline_comparison", data_get)
        self.assertEqual(data_get["baseline_comparison"]["metric_type"], "Held-Out Test Set")
        self.assertIn("model_metadata", data_get)
        self.assertIn("67.94%", data_get["model_metadata"]["test_accuracy"])

        # GET endpoint with historical unverified player (Pete Sampras)
        res_sampras = self.client.get("/api/predict/match?player_a=Pete Sampras&player_b=Novak Djokovic&surface=Grass")
        self.assertEqual(res_sampras.status_code, 200)
        data_sampras = res_sampras.json()
        self.assertTrue(data_sampras["has_fallback_player"])
        self.assertTrue(data_sampras["player_a"]["fallback_applied"])
        self.assertEqual(data_sampras["player_a"]["verified_matches"], 0)
        self.assertGreaterEqual(data_sampras["player_a"]["historical_matches"], 200)
        self.assertGreater(len(data_sampras["coverage_notices"]), 0)

        # POST endpoint
        res_post = self.client.post(
            "/api/predict/match",
            json={"player_a": "Novak Djokovic", "player_b": "Rafael Nadal", "surface": "Clay", "best_of": 3}
        )
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.json()
        self.assertIn("prediction", data_post)
        self.assertEqual(data_post["surface"], "Clay")

    def test_frontend_root_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"RallyIQ", res.content)

if __name__ == "__main__":
    unittest.main()
