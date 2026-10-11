"""
Unit and Integration Tests for RallyIQ ML Match Predictor and Mongo Health
"""

import unittest
from backend.ml.predictor import predictor
from backend.storage.mongo_store import mongo_store, mask_mongo_uri

class TestMatchPredictor(unittest.TestCase):
    def test_predictor_is_loaded_and_ready(self):
        status = predictor.get_status()
        self.assertTrue(status.get("is_ready"))
        self.assertEqual(status.get("status"), "ready")
        self.assertIn("RandomForestClassifier", status.get("algorithm", ""))
        self.assertIn("test_metrics", status)
        rf_metrics = status["test_metrics"]["random_forest"]
        self.assertGreater(rf_metrics["accuracy"], 0.60)
        self.assertGreater(rf_metrics["roc_auc"], 0.70)

    def test_prediction_known_matchup(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="Clay")
        self.assertIn("prediction", res)
        self.assertIn("predicted_winner", res["prediction"])
        prob_a = res["prediction"]["player_a_win_probability"]
        prob_b = res["prediction"]["player_b_win_probability"]
        self.assertAlmostEqual(prob_a + prob_b, 100.0, places=1)
        self.assertIn("key_factors", res)
        self.assertIn("baseline_comparison", res)
        self.assertIn("disclaimer", res)

    def test_exact_prediction_symmetry(self):
        res1 = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        res2 = predictor.predict("Jannik Sinner", "Carlos Alcaraz", surface="Hard")
        
        # Symmetrical ensembling guarantees exact inverse probabilities
        prob_a_1 = res1["prediction"]["player_a_win_probability"]
        prob_b_1 = res1["prediction"]["player_b_win_probability"]
        prob_a_2 = res2["prediction"]["player_a_win_probability"]
        prob_b_2 = res2["prediction"]["player_b_win_probability"]
        
        self.assertEqual(prob_a_1, prob_b_2)
        self.assertEqual(prob_b_1, prob_a_2)

    def test_same_player_validation_error(self):
        with self.assertRaises(ValueError):
            predictor.predict("Roger Federer", "Roger Federer", surface="Grass")

    def test_empty_player_validation_error(self):
        with self.assertRaises(ValueError):
            predictor.predict("", "Novak Djokovic", surface="Hard")

    def test_unknown_player_graceful_handling(self):
        # Should not crash; defaults to neutral 0.5 priors
        res = predictor.predict("Unknown Player XYZ", "Novak Djokovic", surface="Hard")
        self.assertEqual(res["prediction"]["predicted_winner"], "Novak Djokovic")
        self.assertGreater(res["prediction"]["player_b_win_probability"], 50.0)

    def test_historical_coverage_notice_for_unverified_player(self):
        # Pete Sampras has 237 historical matches in database but 0 in 2010-2026 point training data
        res = predictor.predict("Pete Sampras", "Novak Djokovic", surface="Grass")
        self.assertTrue(res["has_fallback_player"])
        self.assertTrue(res["player_a"]["fallback_applied"])
        self.assertEqual(res["player_a"]["verified_matches"], 0)
        self.assertGreaterEqual(res["player_a"]["historical_matches"], 200)
        self.assertGreater(len(res["coverage_notices"]), 0)
        self.assertIn("neutral prior", res["coverage_notices"][0].lower())
        
        # Verify factor notice is included
        notice_factors = [f for f in res["key_factors"] if f["factor"] == "Dataset Coverage Notice"]
        self.assertEqual(len(notice_factors), 1)

    def test_model_metadata_metrics_and_baseline_labels(self):
        res = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        meta = res["model_metadata"]
        base = res["baseline_comparison"]
        self.assertIn("%", meta["test_accuracy"])
        self.assertIn("0.", meta["test_roc_auc"])
        self.assertIn("Chronological", meta["split_type"])
        self.assertEqual(base["metric_type"], "Held-Out Test Set")
        self.assertIn("%", base["baseline_test_accuracy"])
        self.assertIn("0.", base["baseline_test_roc_auc"])
        self.assertIn("logistic_regression_prob_a", base)
        self.assertIn("logistic_regression_prob_b", base)

class TestMongoHealthAndMasking(unittest.TestCase):
    def test_mask_mongo_uri_removes_credentials(self):
        test_uri = "mongodb+srv://admin_user:SuperSecretPassword123@mycluster.mongodb.net/testdb?retryWrites=true"
        masked = mask_mongo_uri(test_uri)
        self.assertNotIn("SuperSecretPassword123", masked)
        self.assertNotIn("admin_user", masked)
        self.assertIn("****:****", masked)
        self.assertIn("mycluster.mongodb.net", masked)

    def test_mongo_test_connection_structure(self):
        health = mongo_store.test_connection()
        self.assertIn("status", health)
        self.assertIn("engine", health)
        self.assertIn("cluster", health)
        self.assertIn("database", health)
        self.assertEqual(health["database"], "rallyiq_tennis")

if __name__ == "__main__":
    unittest.main()
