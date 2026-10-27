"""
API Integration Tests for RallyIQ
Verifies all FastAPI REST endpoints across the 5 features:
1. Player Intelligence Lab
2. Serve Strategy Heatmap
3. Similar Playstyle Recommender (Cosine Similarity)
4. ATP Rivalry Network (Louvain Community Detection & PageRank)
5. ML Match Winner Predictor (Supervised Classification)
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
        
        # 4. /api/matches
        res4 = self.client.get("/api/matches?limit=10&offset=0")
        self.assertEqual(res4.status_code, 200)
        data4 = res4.json()
        self.assertIn("matches", data4)

    def test_players_endpoint(self):
        res = self.client.get("/api/players?min_matches=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("players", data)
        self.assertGreater(len(data["players"]), 0)

    def test_predict_endpoint_rejects_invalid_surface_and_best_of(self):
        for query in ("surface=Carpet", "best_of=7"):
            with self.subTest(query=query):
                res = self.client.get(f"/api/predict/match?player_a=Novak Djokovic&player_b=Rafael Nadal&{query}")
                self.assertEqual(res.status_code, 400)

    def test_player_lab_endpoint(self):
        res = self.client.get("/api/player-lab?player_a=Novak Djokovic&player_b=Rafael Nadal&surface=Overall")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["player_a"]["name"], "Novak Djokovic")
        self.assertEqual(data["player_b"]["name"], "Rafael Nadal")
        self.assertIn("head_to_head", data)

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
    
    def test_recommender_rejects_invalid_top_k(self):
        for bad in (0, -3, 99999):
            res = self.client.get(
                "/api/recommender", params={"player": "Carlos Alcaraz", "top_k": bad}
            )
            self.assertEqual(res.status_code, 422, f"top_k={bad} should be rejected")

    def test_recommender_top_k_boundaries(self):
        for valid in (1, 50):
            res = self.client.get(
                "/api/recommender", params={"player": "Carlos Alcaraz", "top_k": valid}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(len(res.json()["recommendations"]), valid)

    def test_recommender_rejects_empty_player(self):
        for bad in ("", "   "):
            res = self.client.get(
                "/api/recommender", params={"player": bad, "top_k": 5}
            )
            self.assertEqual(res.status_code, 422, f"player={bad!r} should be rejected")

    def test_recommender_trims_player_name(self):
        res = self.client.get(
            "/api/recommender", params={"player": "  Carlos Alcaraz  ", "top_k": 3}
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["target_player"], "Carlos Alcaraz")

    def test_recommender_unknown_player_returns_404(self):
        res = self.client.get(
            "/api/recommender", params={"player": "Nobody Real", "top_k": 5}
        )
        self.assertEqual(res.status_code, 404)

    def test_rivalry_network_endpoint(self):
        res = self.client.get("/api/rivalry-network?min_meetings=3&top_n=25")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("community_clusters", data)
        self.assertGreater(len(data["nodes"]), 0)

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
        self.assertIn("colab_evaluation", data)
        self.assertTrue(data["colab_evaluation"]["available"])

    def test_predict_evaluation_and_charts_endpoints(self):
        res = self.client.get("/api/predict/evaluation")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertIn("colab_evaluation", payload)
        self.assertIn("production_predictor", payload)
        data = payload["colab_evaluation"]
        self.assertTrue(data.get("available"))
        self.assertEqual(data["dataset_and_split_summary"]["test_unique_matches"], 899)
        self.assertAlmostEqual(data["metrics"]["accuracy"], 0.6863, places=4)
        self.assertAlmostEqual(data["metrics"]["roc_auc"], 0.7491, places=4)

        # Verify static chart images are served properly
        for chart_url in data["charts"].values():
            chart_res = self.client.get(chart_url)
            self.assertEqual(chart_res.status_code, 200)
            self.assertEqual(chart_res.headers.get("content-type"), "image/png")

    def test_predict_match_endpoints(self):
        # GET endpoint
        res_get = self.client.get("/api/predict/match?player_a=Carlos Alcaraz&player_b=Jannik Sinner&surface=Hard")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertIn("prediction", data_get)
        self.assertIn("predicted_winner", data_get["prediction"])
        self.assertIn("baseline_comparison", data_get)
        self.assertTrue(data_get["baseline_comparison"]["baseline_available"])
        self.assertIn("model_metadata", data_get)
        self.assertEqual(data_get["model_metadata"]["training_cutoff_date"], "20230217")
        self.assertNotIn("disclaimer", data_get)

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
