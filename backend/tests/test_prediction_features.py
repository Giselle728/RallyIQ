"""Tests for match prediction feature generation (chronological features)."""

import unittest
import pandas as pd

from backend.ml.train_predictor import extract_chronological_features


def _make_matches(rows):
    return pd.DataFrame(rows)


class TestPredictionFeatureGeneration(unittest.TestCase):
    def _fixture(self):
        return pd.DataFrame([
            {
                "match_id": "20230101-M-A-B",
                "Player 1": "Alice",
                "Player 2": "Bob",
                "Surface": "Hard",
                "Best of": 3,
                "Date": "20230101",
                "winner_status": "Verified",
                "verified_winner": "Alice",
            },
            {
                "match_id": "20230102-M-A-B",
                "Player 1": "Alice",
                "Player 2": "Bob",
                "Surface": "Hard",
                "Best of": 3,
                "Date": "20230102",
                "winner_status": "Verified",
                "verified_winner": "Bob",
            },
            {
                "match_id": "20230102-M-NEW-X",
                "Player 1": "Newcomer",
                "Player 2": "Bob",
                "Surface": "Clay",
                "Best of": 3,
                "Date": "20230102",
                "winner_status": "Verified",
                "verified_winner": "Bob",
            },
        ])

    def test_head_to_head_difference(self):
        df, meta = extract_chronological_features(self._fixture())
        # The second Alice vs Bob match should reflect prior H2H (Alice 1 win)
        rows = df[(df["p1"] == "Alice") & (df["p2"] == "Bob") & (df["date"] == "20230102")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["h2h_total"], 1)
        self.assertEqual(rows.iloc[0]["h2h_diff"], 1)

    def test_recent_form(self):
        df, meta = extract_chronological_features(self._fixture())
        rows = df[(df["p1"] == "Alice") & (df["p2"] == "Bob") & (df["date"] == "20230102")]
        self.assertEqual(len(rows), 1)
        # Alice won first match -> recent form 1.0; Bob lost -> 0.0
        self.assertAlmostEqual(rows.iloc[0]["recent_form_diff"], 1.0, places=3)

    def test_surface_win_rates(self):
        df, meta = extract_chronological_features(self._fixture())
        rows = df[(df["p1"] == "Alice") & (df["p2"] == "Bob") & (df["date"] == "20230102")]
        self.assertEqual(len(rows), 1)
        # Alice won on Hard previously -> surface win rate 1.0
        self.assertAlmostEqual(rows.iloc[0]["p1_surface_win_rate"], 1.0, places=3)

    def test_missing_player_history_neutral_prior(self):
        df, meta = extract_chronological_features(self._fixture())
        # Newcomer has no prior matches -> neutral 0.5 win rate
        rows = df[(df["p1"] == "Newcomer")]
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows.iloc[0]["p1_win_rate"], 0.5, places=3)
        self.assertAlmostEqual(rows.iloc[0]["p1_surface_win_rate"], 0.5, places=3)

    def test_missing_player_history_reverse_augmented(self):
        df, meta = extract_chronological_features(self._fixture())
        self.assertGreater(len(df), 0)
        # Each match yields two rows (primary + reverse)
        self.assertEqual(len(df) % 2, 0)


if __name__ == "__main__":
    unittest.main()