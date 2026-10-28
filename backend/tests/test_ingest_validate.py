import unittest

import numpy as np
import pandas as pd

from backend.pipeline.ingest_validate import (
    safe_percentage,
    validate_matches_schema,
    validate_overview_stats,
)


def _matches(rows):
    df = pd.DataFrame(rows, columns=["match_id", "Player 1", "Player 2"])
    df["Surface"] = "Hard"
    df["Date"] = "20200101"
    df["Best of"] = 3
    return df


class TestMissingPlayerNames(unittest.TestCase):
    def test_missing_blank_and_null_names_are_dropped(self):
        df = _matches([
            ("2020-A", "Carlos Alcaraz", "Jannik Sinner"),
            ("2020-B", np.nan, "Jannik Sinner"),
            ("2020-C", "", "Jannik Sinner"),
            ("2020-D", "Carlos Alcaraz", "   "),
            ("2020-E", "Carlos Alcaraz", None),
            ("2020-F", "nan", "Jannik Sinner"),
        ])
        out, meta = validate_matches_schema(df)
        self.assertEqual(list(out["match_id"]), ["2020-A"])
        self.assertEqual(meta["missing_names_dropped"], 5)

    def test_no_fake_names_remain(self):
        df = _matches([
            ("2020-A", "Carlos Alcaraz", "Jannik Sinner"),
            ("2020-B", None, "Jannik Sinner"),
        ])
        out, _ = validate_matches_schema(df)
        names = set(out["Player 1"]) | set(out["Player 2"])
        self.assertFalse(names & {"nan", "None", ""})

    def test_valid_names_are_trimmed_and_kept(self):
        df = _matches([("2020-A", "  Rafael Nadal  ", "Novak Djokovic")])
        out, meta = validate_matches_schema(df)
        self.assertEqual(out.iloc[0]["Player 1"], "Rafael Nadal")
        self.assertEqual(meta["missing_names_dropped"], 0)

    def test_overview_rows_with_missing_player_are_dropped(self):
        df = pd.DataFrame({
            "match_id": ["2020-A", "2020-A", "2020-B", "2020-C"],
            "player": ["Carlos Alcaraz", np.nan, "   ", "Jannik Sinner"],
            "set": ["Total"] * 4,
        })
        out, meta = validate_overview_stats(df)
        self.assertEqual(sorted(out["player"]), ["Carlos Alcaraz", "Jannik Sinner"])
        self.assertEqual(meta["missing_players_dropped"], 2)

class TestSafePercentage(unittest.TestCase):
    def test_zero_denominator_returns_zero(self):
        self.assertEqual(safe_percentage(0, 0), 0.0)
        self.assertEqual(safe_percentage(5, 0), 0.0)

    def test_empty_dataframe_gives_zero_verified_percentage(self):
        empty = pd.DataFrame({"verified_winner": pd.Series([], dtype="object")})
        verified = int(empty["verified_winner"].notna().sum())
        self.assertEqual(safe_percentage(verified, len(empty)), 0.0)

    def test_normal_ratio_is_rounded_percentage(self):
        self.assertEqual(safe_percentage(1, 3), 33.3)
        self.assertEqual(safe_percentage(5568, 7565), 73.6)
        self.assertEqual(safe_percentage(3, 3), 100.0)

    def test_missing_or_non_finite_denominator_returns_zero(self):
        for bad in (None, float("nan"), float("inf")):
            with self.subTest(total=bad):
                self.assertEqual(safe_percentage(1, bad), 0.0)

    def test_result_is_always_finite(self):
        for part, total in ((0, 0), (1, 0), (None, 5), (float("nan"), 5)):
            with self.subTest(part=part, total=total):
                self.assertTrue(np.isfinite(safe_percentage(part, total)))
                

if __name__ == "__main__":
    unittest.main()
