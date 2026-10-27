
import unittest

import pandas as pd

from backend.ml.train_predictor import extract_chronological_features


def make_matches():
    """Create a small deterministic dataset with known match outcomes."""
    return pd.DataFrame([
        {
            "match_id": "20240101-M-Test-R1-A-B",
            "Player 1": "Player A",
            "Player 2": "Player B",
            "Surface": "Clay",
            "Best of": 5,
            "winner_status": "Verified",
            "verified_winner": "Player A",
        },
        {
            "match_id": "20240102-M-Test-R2-A-B",
            "Player 1": "Player A",
            "Player 2": "Player B",
            "Surface": "Clay",
            "Best of": 3,
            "winner_status": "Verified",
            "verified_winner": "Player B",
        },
        {
            "match_id": "20240103-M-Test-R3-A-C",
            "Player 1": "Player A",
            "Player 2": "Player C",
            "Surface": "Grass",
            "Best of": 3,
            "winner_status": "Verified",
            "verified_winner": "Player A",
        },
    ])


class TestPredictionFeatureGeneration(unittest.TestCase):
    def test_generates_two_rows_per_match(self):
        features, metadata = extract_chronological_features(make_matches())

        self.assertEqual(len(features), 6)
        self.assertEqual(metadata["total_verified_matches_processed"], 3)
        self.assertEqual(features["is_reverse_augmented"].sum(), 3)

    def test_initial_features_use_neutral_priors(self):
        features, _ = extract_chronological_features(make_matches())

        forward = features.iloc[0]
        reverse = features.iloc[1]

        self.assertEqual(forward["p1_win_rate"], 0.5)
        self.assertEqual(forward["p2_win_rate"], 0.5)
        self.assertEqual(forward["win_rate_diff"], 0.0)
        self.assertEqual(forward["target"], 1)

        self.assertEqual(reverse["p1_win_rate"], 0.5)
        self.assertEqual(reverse["p2_win_rate"], 0.5)
        self.assertEqual(reverse["win_rate_diff"], 0.0)
        self.assertEqual(reverse["target"], 0)

    def test_reverse_features_are_symmetric(self):
        features, _ = extract_chronological_features(make_matches())

        forward = features.iloc[2]
        reverse = features.iloc[3]

        self.assertEqual(forward["p1"], reverse["p2"])
        self.assertEqual(forward["p2"], reverse["p1"])
        self.assertEqual(
            forward["win_rate_diff"],
            -reverse["win_rate_diff"],
        )
        self.assertEqual(
            forward["surface_win_rate_diff"],
            -reverse["surface_win_rate_diff"],
        )
        self.assertEqual(
            forward["recent_form_diff"],
            -reverse["recent_form_diff"],
        )
        self.assertEqual(forward["h2h_diff"], -reverse["h2h_diff"])
        self.assertEqual(forward["h2h_total"], reverse["h2h_total"])
        self.assertEqual(forward["exp_diff"], -reverse["exp_diff"])
        self.assertEqual(forward["target"] + reverse["target"], 1)

    def test_surface_and_best_of_features_are_consistent(self):
        features, _ = extract_chronological_features(make_matches())

        clay_best_of_five = features.iloc[0]
        grass_match = features.iloc[4]

        self.assertEqual(clay_best_of_five["surface_Clay"], 1)
        self.assertEqual(clay_best_of_five["surface_Grass"], 0)
        self.assertEqual(clay_best_of_five["surface_Hard"], 0)
        self.assertEqual(clay_best_of_five["best_of"], 5)

        self.assertEqual(grass_match["surface_Clay"], 0)
        self.assertEqual(grass_match["surface_Grass"], 1)
        self.assertEqual(grass_match["surface_Hard"], 0)
        self.assertEqual(grass_match["best_of"], 3)

        for _, row in features.iterrows():
            self.assertEqual(
                row["surface_Clay"]
                + row["surface_Grass"]
                + row["surface_Hard"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
