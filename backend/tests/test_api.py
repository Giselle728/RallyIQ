"""
API Integration Tests for RallyIQ
Verifies all FastAPI REST endpoints across the 6 features.
"""

import unittest
from fastapi.testclient import TestClient
from backend.api.main import app

class TestRallyIQApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

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
        first_match = matches_res.json()["matches"][0]
        m_id = first_match["match_id"]
        
        res = self.client.get(f"/api/momentum/replay?match_id={m_id}&window_size=30&alpha=0.25")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("curve", data)
        self.assertIn("points", data)
        self.assertGreater(len(data["points"]), 0)
        self.assertIn("dgim_p1_window_wins", data["points"][0])

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
        # 1. Reset filter
        reset_res = self.client.post("/api/stream/reset?m=4096&k=4")
        self.assertEqual(reset_res.status_code, 200)
        
        # 2. Step stream
        step_res = self.client.post("/api/stream/step?count=5")
        self.assertEqual(step_res.status_code, 200)
        step_data = step_res.json()
        self.assertEqual(step_data["processed_count"], 5)
        
        # 3. Inject duplicate
        dup_res = self.client.post("/api/stream/inject-duplicate")
        self.assertEqual(dup_res.status_code, 200)
        dup_data = dup_res.json()
        self.assertTrue(dup_data["bloom_detected_duplicate"])
        
        # 4. Status
        status_res = self.client.get("/api/stream/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertGreater(status_data["bloom_filter"]["inserted_items_n"], 0)

    def test_frontend_root_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"RallyIQ", res.content)

if __name__ == "__main__":
    unittest.main()
