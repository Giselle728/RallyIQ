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

if __name__ == "__main__":
    unittest.main()
