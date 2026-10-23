"""Tests for prediction response consistency and probability symmetry."""

import unittest

from backend.ml.predictor import predictor


class TestPredictionResponseConsistency(unittest.TestCase):
    def test_probabilities_sum_to_one_hundred(self):
        res = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        prob_a = res["prediction"]["player_a_win_probability"]
        prob_b = res["prediction"]["player_b_win_probability"]
        self.assertAlmostEqual(prob_a + prob_b, 100.0, places=1)

    def test_predicted_winner_matches_higher_probability(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="Clay")
        prob_a = res["prediction"]["player_a_win_probability"]
        prob_b = res["prediction"]["player_b_win_probability"]
        winner = res["prediction"]["predicted_winner"]
        if prob_a > prob_b:
            self.assertEqual(winner, "Novak Djokovic")
        elif prob_b > prob_a:
            self.assertEqual(winner, "Rafael Nadal")
        else:
            self.assertIn(winner, ("Novak Djokovic", "Rafael Nadal"))

    def test_winner_probability_is_maximum(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="Clay")
        prob_a = res["prediction"]["player_a_win_probability"]
        prob_b = res["prediction"]["player_b_win_probability"]
        self.assertEqual(
            res["prediction"]["winner_probability"], max(prob_a, prob_b)
        )

    def test_confidence_level_matches_margin(self):
        res = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        prob_a = res["prediction"]["player_a_win_probability"]
        prob_b = res["prediction"]["player_b_win_probability"]
        margin = abs(prob_a - prob_b) / 100.0
        level = res["prediction"]["confidence_level"]
        if margin >= 0.35:
            self.assertEqual(level, "High")
        elif margin >= 0.15:
            self.assertEqual(level, "Moderate")
        else:
            self.assertEqual(level, "Slight Edge / Close Match")

    def test_reverse_symmetry(self):
        res1 = predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
        res2 = predictor.predict("Jannik Sinner", "Carlos Alcaraz", surface="Hard")
        self.assertEqual(
            res1["prediction"]["player_a_win_probability"],
            res2["prediction"]["player_b_win_probability"],
        )
        self.assertEqual(
            res1["prediction"]["player_b_win_probability"],
            res2["prediction"]["player_a_win_probability"],
        )

    def test_baseline_probabilities_sum(self):
        res = predictor.predict("Novak Djokovic", "Rafael Nadal", surface="Clay")
        baseline = res["baseline_comparison"]
        if baseline["baseline_available"]:
            self.assertAlmostEqual(
                baseline["logistic_regression_prob_a"]
                + baseline["logistic_regression_prob_b"],
                100.0,
                places=1,
            )

if __name__ == "__main__":
    unittest.main()
