"""
Unit and Integration Tests for RallyIQ ML Match Predictor, Feature Engineering,
Chronological Split Isolation, and Google Colab Evaluation Integration.
"""

import os
import tempfile
import unittest
import joblib
import pandas as pd

from backend.ml.predictor import MatchPredictor, predictor
from backend.ml.train_predictor import (
    extract_chronological_features,
    validate_and_clean_matches,
    MODEL_ARTIFACT_PATH
)
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
        self.assertIn("majority_class_baseline", status["test_metrics"])
        self.assertEqual(status["test_metrics"]["majority_class_baseline"]["accuracy"], 0.50)
        self.assertIn("colab_evaluation", status)
        self.assertTrue(status["colab_evaluation"]["available"])

    def test_schema_version_and_cutoff_date(self):
        status = predictor.get_status()
        self.assertEqual(status.get("schema_version"), "1.1.0")
        self.assertIsNotNone(status.get("training_cutoff_date"))
        self.assertEqual(status.get("training_cutoff_date"), "20230217")

    def test_prediction_known_matchup(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="Clay")
        self.assertIn("prediction", res)
        self.assertIn("predicted_winner", res["prediction"])
        prob_a = res["prediction"]["player_a_win_probability"]
        prob_b = res["prediction"]["player_b_win_probability"]
        self.assertAlmostEqual(prob_a + prob_b, 100.0, places=1)
        self.assertIn("key_factors", res)
        self.assertIn("baseline_comparison", res)
        self.assertNotIn("disclaimer", res)

    def test_exact_prediction_symmetry(self):
        res1 = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        res2 = predictor.predict("Jannik Sinner", "Carlos Alcaraz", surface="Hard")

        # Order-independent averaging guarantees exact inverse probabilities
        prob_a_1 = res1["prediction"]["player_a_win_probability"]
        prob_b_1 = res1["prediction"]["player_b_win_probability"]
        prob_a_2 = res2["prediction"]["player_a_win_probability"]
        prob_b_2 = res2["prediction"]["player_b_win_probability"]

        self.assertEqual(prob_a_1, prob_b_2)
        self.assertEqual(prob_b_1, prob_a_2)
        self.assertAlmostEqual(prob_a_1 + prob_b_1, 100.0, places=1)

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

    def test_model_metadata_and_baseline_labels(self):
        res = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        meta = res["model_metadata"]
        base = res["baseline_comparison"]
        self.assertIn("Chronological", meta["split_type"])
        self.assertEqual(meta["training_cutoff_date"], "20230217")
        self.assertNotIn("test_accuracy", meta)
        self.assertNotIn("test_roc_auc", meta)
        self.assertTrue(base["baseline_available"])
        self.assertIsNotNone(base["logistic_regression_prob_a"])
        self.assertIsNotNone(base["logistic_regression_prob_b"])
        self.assertAlmostEqual(
            base["logistic_regression_prob_a"] + base["logistic_regression_prob_b"],
            100.0,
            places=1
        )

    def test_baseline_failure_does_not_fallback_to_rf(self):
        orig_baseline = predictor.baseline_model
        try:
            predictor.baseline_model = None
            res = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
            base = res["baseline_comparison"]
            self.assertFalse(base["baseline_available"])
            self.assertIsNone(base["logistic_regression_prob_a"])
            self.assertIsNone(base["logistic_regression_prob_b"])
        finally:
            predictor.baseline_model = orig_baseline

    def test_colab_evaluation_report_integration(self):
        report_bundle = predictor.get_evaluation_report()
        self.assertIn("colab_evaluation", report_bundle)
        self.assertIn("production_predictor", report_bundle)

        report = report_bundle["colab_evaluation"]
        self.assertTrue(report["available"])
        self.assertEqual(report["source_directory"], "reports/ml_evaluation")
        self.assertEqual(report["dataset_and_split_summary"]["test_unique_matches"], 899)
        self.assertEqual(report["model_identity"]["feature_count"], 9)

        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["accuracy"], 0.6863, places=4)
        self.assertAlmostEqual(metrics["precision"], 0.6921, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.7158, places=4)
        self.assertAlmostEqual(metrics["f1_score"], 0.7038, places=4)
        self.assertAlmostEqual(metrics["roc_auc"], 0.7491, places=4)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.6851, places=4)
        self.assertAlmostEqual(metrics["specificity"], 0.6543, places=4)

        cm = report["confusion_matrix"]
        self.assertEqual(cm["tn"], 282)
        self.assertEqual(cm["fp"], 149)
        self.assertEqual(cm["fn"], 133)
        self.assertEqual(cm["tp"], 335)
        self.assertEqual(cm["total_unique_test_matches"], 899)

        self.assertIn("confusion_matrix", report["charts"])
        self.assertIn("roc_curve", report["charts"])
        self.assertIn("feature_importance", report["charts"])

    def test_surface_and_best_of_normalization(self):
        res_clay = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="  cLaY  ", best_of="5")
        self.assertEqual(res_clay["surface"], "Clay")
        self.assertEqual(res_clay["best_of"], 5)

    def test_invalid_surface_raises_error(self):
        with self.assertRaises(ValueError):
            predictor.predict("Novak Djokovic", "Rafael Nadal", surface="unknown_surface")

    def test_invalid_best_of_raises_error(self):
        for bad in ("invalid", 7, 0, None):
            with self.assertRaises(ValueError):
                predictor.predict("Novak Djokovic", "Rafael Nadal", best_of=bad)

    def test_none_player_name_raises_error(self):
        with self.assertRaises(ValueError):
            predictor.predict(None, "Rafael Nadal")
            
    def test_surface_is_case_and_whitespace_insensitive(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="  clay ")
        self.assertEqual(res["surface"], "Clay")
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="GRASS")
        self.assertEqual(res["surface"], "Grass")

    def test_best_of_numeric_string_is_accepted(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", best_of="5")
        self.assertEqual(res["best_of"], 5)

    def test_all_valid_surface_and_best_of_combinations(self):
        for surface in ("Hard", "Clay", "Grass"):
            for best_of in (3, 5):
                with self.subTest(surface=surface, best_of=best_of):
                    res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface=surface, best_of=best_of)
                    self.assertEqual(res["surface"], surface)
                    self.assertEqual(res["best_of"], best_of)

    def test_whitespace_only_player_name_raises_error(self):
        with self.assertRaises(ValueError):
            predictor.predict("   ", "Rafael Nadal")
        with self.assertRaises(ValueError):
            predictor.predict("Novak Djokovic", "\t")
    def test_unloaded_model_resilience(self):
        # Test isolated instance with nonexistent artifact path
        isolated = MatchPredictor(artifact_path="/tmp/nonexistent_model.joblib")
        self.assertFalse(isolated.is_loaded)
        self.assertTrue(hasattr(isolated, "historical_profiles"))
        self.assertIsInstance(isolated.historical_profiles, dict)

        status = isolated.get_status()
        self.assertEqual(status["status"], "not_trained")
        self.assertFalse(status["is_ready"])

        with self.assertRaises(RuntimeError):
            isolated.predict("Novak Djokovic", "Rafael Nadal")

    def test_invalid_artifact_payload_handling(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = f.name
            joblib.dump({"corrupted": "payload_without_required_keys"}, temp_path)

        try:
            isolated = MatchPredictor(artifact_path=temp_path)
            self.assertFalse(isolated.is_loaded)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_feature_schema_validation_error(self):
        # Save original feature names and temporarily add an unknown required feature
        orig_features = list(predictor.feature_names)
        try:
            predictor.feature_names = orig_features + ["unsupported_phantom_feature"]
            with self.assertRaises(ValueError) as ctx:
                predictor.predict("Carlos Alcaraz", "Jannik Sinner")
            self.assertIn("Feature schema mismatch", str(ctx.exception))
        finally:
            predictor.feature_names = orig_features


class TestPipelineLeakageAndSplits(unittest.TestCase):
    def test_zero_intra_day_leakage_in_date_groups(self):
        """
        Verify that multiple matches occurring on the SAME date do NOT leak into each other.
        Features for all same-day matches must be computed strictly from earlier dates.
        """
        synthetic_df = pd.DataFrame([
            {
                "match_id": "20240101-M-Tourney-R1-Player_A-Player_B",
                "Player 1": "Player A", "Player 2": "Player B",
                "Surface": "Hard", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Player A"
            },
            {
                "match_id": "20240101-M-Tourney-R1-Player_A-Player_C",
                "Player 1": "Player A", "Player 2": "Player C",
                "Surface": "Hard", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Player A"
            },
            {
                "match_id": "20240102-M-Tourney-R2-Player_A-Player_D",
                "Player 1": "Player A", "Player 2": "Player D",
                "Surface": "Hard", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Player A"
            }
        ])

        feat_df, _ = extract_chronological_features(synthetic_df)

        # For the two matches on 20240101, Player A must have neutral prior (0.5) in both
        match1_forward = feat_df.iloc[0]
        match2_forward = feat_df.iloc[2]
        self.assertEqual(match1_forward["p1_win_rate"], 0.5)
        self.assertEqual(match2_forward["p1_win_rate"], 0.5)

        # For match on 20240102, Player A's win rate reflects both earlier wins
        match3_forward = feat_df.iloc[4]
        self.assertEqual(match3_forward["p1_win_rate"], 1.0)

    def test_disjoint_date_splits_and_unique_vs_augmented_counts(self):
        """Verify date ranges for Train, Validation, and Test are disjoint and counts distinguish unique vs augmented."""
        self.assertTrue(os.path.exists(MODEL_ARTIFACT_PATH))
        metrics = predictor.metrics
        ds_meta = metrics.get("dataset_metadata", {})

        splits = metrics.get("splits", {})
        train_range = splits["train"]["date_range"].split(" to ")
        val_range = splits["validation"]["date_range"].split(" to ")
        test_range = splits["test"]["date_range"].split(" to ")

        train_max = train_range[1]
        val_min = val_range[0]
        val_max = val_range[1]
        test_min = test_range[0]

        # Chronological ordering without date collision
        self.assertLess(train_max, val_min)
        self.assertLess(val_max, test_min)

        # Verify unique match counts vs augmented sample counts
        self.assertEqual(ds_meta["total_unique_matches"], 5568)
        self.assertEqual(ds_meta["total_augmented_samples"], 11136)
        self.assertEqual(splits["train"]["unique_matches"], 3534)
        self.assertEqual(splits["train"]["augmented_samples"], 7068)
        self.assertEqual(splits["validation"]["unique_matches"], 1135)
        self.assertEqual(splits["validation"]["augmented_samples"], 2270)
        self.assertEqual(splits["test"]["unique_matches"], 899)
        self.assertEqual(splits["test"]["augmented_samples"], 1798)

    def test_inference_player_states_frozen_at_cutoff(self):
        """Verify predictor player states contain only data up to the training cutoff date."""
        cutoff_date = predictor.training_cutoff_date
        self.assertIsNotNone(cutoff_date)
        self.assertEqual(cutoff_date, "20230217")

        # In our verified dataset, Carlos Alcaraz has 59 matches on or before 20230217
        alcaraz_m = predictor.player_states["player_matches"].get("Carlos Alcaraz", 0)
        self.assertEqual(alcaraz_m, 59)

    def test_data_validation_filters_corrupted_rows(self):
        """Verify data cleaning rejects same-player, missing names, and invalid winner rows."""
        dirty_df = pd.DataFrame([
            {
                "match_id": "20240101-M-Test-R1-Roger_Federer-Roger_Federer",
                "Player 1": "Roger Federer", "Player 2": "Roger Federer",
                "Surface": "Grass", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Roger Federer"
            },
            {
                "match_id": "20240101-M-Test-R1-Empty-Novak_Djokovic",
                "Player 1": "   ", "Player 2": "Novak Djokovic",
                "Surface": "Hard", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Novak Djokovic"
            },
            {
                "match_id": "20240101-M-Test-R1-Rafael_Nadal-Novak_Djokovic",
                "Player 1": "Rafael Nadal", "Player 2": "Novak Djokovic",
                "Surface": "Clay", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Invalid Phantom Winner"
            },
            {
                "match_id": "20240101-M-Test-R1-Valid1-Valid2",
                "Player 1": "Player One", "Player 2": "Player Two",
                "Surface": "Hard", "Best of": 3,
                "winner_status": "Verified", "verified_winner": "Player One"
            }
        ])

        clean, summary = validate_and_clean_matches(dirty_df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(summary["same_player_rows_dropped"], 1)
        self.assertEqual(summary["missing_player_rows_dropped"], 1)
        self.assertEqual(summary["invalid_winner_rows_dropped"], 1)


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
