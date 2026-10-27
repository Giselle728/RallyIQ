"""
Unit and Integration Tests for RallyIQ Big Data Algorithms:
1. Similar Playstyle Recommender (Cosine Similarity)
2. Louvain Community Detection & PageRank Centrality
"""

import unittest
from backend.algorithms.recommender import PlaystyleRecommender
from backend.algorithms.graph_analytics import RivalryNetwork

class TestPlaystyleRecommender(unittest.TestCase):
    def test_recommender_cosine_similarity(self):
        rec = PlaystyleRecommender(min_matches=3)
        if len(rec.qualified_players) >= 2:
            target = rec.qualified_players[0]
            res = rec.find_similar_players(target, top_k=3)
            self.assertEqual(res["target_player"], target)
            self.assertEqual(len(res["recommendations"]), 3)
            # Similarities must be <= 100% and >= 0%
            for r in res["recommendations"]:
                self.assertLessEqual(r["similarity_score"], 100.0)
                self.assertGreater(r["similarity_score"], 50.0)

class TestRivalryNetwork(unittest.TestCase):
    def test_louvain_communities_and_pagerank(self):
        rn = RivalryNetwork()
        if rn.G.number_of_nodes() > 0:
            self.assertGreater(len(rn.communities), 0)
            self.assertGreater(rn.modularity_score, 0.0)
            self.assertGreater(len(rn.pagerank_scores), 0)
            data = rn.get_network_data(min_meetings=2, top_n_nodes=20)
            self.assertGreater(len(data["nodes"]), 0)

    def test_empty_graph_returns_valid_empty_response(self):
        rn = RivalryNetwork(rivalries_path="does_not_exist.json")
        data = rn.get_network_data(min_meetings=3, top_n_nodes=35)
        self.assertEqual(data["nodes"], [])
        self.assertEqual(data["edges"], [])
        self.assertEqual(data["community_clusters"], [])
        self.assertEqual(data["metrics"]["total_players_in_graph"], 0)
        self.assertEqual(data["metrics"]["louvain_communities_count"], 0)

    def test_empty_response_has_message_and_echoes_min_meetings(self):
        rn = RivalryNetwork(rivalries_path="does_not_exist.json")
        data = rn.get_network_data(min_meetings=7, top_n_nodes=10)
        self.assertEqual(data["message"], "No rivalry data available.")
        self.assertEqual(data["metrics"]["min_meetings_threshold"], 7)
        self.assertEqual(data["metrics"]["total_rivalries_in_graph"], 0)

    def test_empty_rivalries_json_file_returns_empty_response(self):
        import os, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name
        try:
            rn = RivalryNetwork(rivalries_path=path)
            data = rn.get_network_data()
            self.assertEqual(data["nodes"], [])
            self.assertEqual(data["edges"], [])
        finally:
            os.remove(path)

    def test_very_high_min_meetings_returns_no_edges_without_crashing(self):
        rn = RivalryNetwork()
        data = rn.get_network_data(min_meetings=10**6, top_n_nodes=35)
        self.assertEqual(data["edges"], [])
        self.assertIn("metrics", data)

    def test_non_positive_top_n_nodes_is_safe(self):
        rn = RivalryNetwork()
        for n in (0, -5):
            with self.subTest(top_n_nodes=n):
                data = rn.get_network_data(min_meetings=3, top_n_nodes=n)
                self.assertEqual(data["nodes"], [])
                self.assertEqual(data["edges"], [])

if __name__ == "__main__":
    unittest.main()
