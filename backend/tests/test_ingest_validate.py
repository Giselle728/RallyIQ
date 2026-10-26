import unittest

import numpy as np
import pandas as pd

from backend.pipeline.ingest_validate import (
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


if __name__ == "__main__":
    unittest.main()
