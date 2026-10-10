"""
Unit and Integration Tests for RallyIQ Big Data Algorithms:
1. DGIM Sliding-Window Bit Counter
2. Momentum Analyzer (EWMA + Leverage)
3. Kirsch-Mitzenmacher Bloom Filter
4. Similar Playstyle Recommender (Cosine Similarity)
5. Louvain Community Detection & PageRank Centrality
"""

import unittest
import numpy as np
from backend.algorithms.dgim import DGIM, MomentumAnalyzer
from backend.algorithms.bloom_filter import BloomFilter
from backend.algorithms.recommender import PlaystyleRecommender
from backend.algorithms.graph_analytics import RivalryNetwork

class TestDGIMAlgorithm(unittest.TestCase):
    def test_dgim_basic_counting(self):
        # Window size 20, max 2 buckets per size
        dgim = DGIM(window_size=20, max_buckets_per_size=2)
        
        # Feed stream of 20 bits: alternating 1 and 0 (10 ones)
        for i in range(20):
            dgim.add_bit(1 if i % 2 == 0 else 0)
            
        res = dgim.count()
        self.assertEqual(res["exact_count"], 10)
        # DGIM theoretical error bound is <= 25% for k=2
        self.assertLessEqual(res["error_percentage"], 25.0)
        self.assertGreater(res["estimated_count"], 0)
        self.assertGreater(len(res["buckets"]), 0)

    def test_dgim_bucket_merging_powers_of_two(self):
        dgim = DGIM(window_size=50, max_buckets_per_size=2)
        # Feed 16 consecutive ones
        for i in range(16):
            dgim.add_bit(1)
            
        res = dgim.count()
        # Verify all bucket sizes are powers of 2
        for b in res["buckets"]:
            size = b["size"]
            self.assertTrue((size & (size - 1)) == 0, f"Bucket size {size} is not power of 2")
            
        # Verify at most k=2 buckets of any given size
        sizes = [b["size"] for b in res["buckets"]]
        for s in set(sizes):
            self.assertLessEqual(sizes.count(s), 2)

class TestMomentumAnalyzer(unittest.TestCase):
    def test_momentum_impact_and_ewma(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        
        # Test Break Point won by Player 1
        bp_record = {"Pts": "30-40", "Svr": 2, "PtWinner": 1, "IsAce": False, "IsDoubleFault": False}
        impact, label = analyzer.evaluate_point_impact(bp_record, "Djokovic", "Nadal")
        self.assertGreater(impact, 2.0)
        self.assertIn("Break Point", label)
        
        # Process point
        res = analyzer.process_point(bp_record, "Djokovic", "Nadal")
        self.assertGreater(res["momentum"], 0.0)
        self.assertEqual(res["dominant_player"], "Djokovic")

class TestBloomFilter(unittest.TestCase):
    def test_bloom_insertion_and_membership(self):
        bf = BloomFilter(m=1024, k=4)
        items = [f"event_{i}" for i in range(50)]
        
        for item in items:
            bf.add(item)
            
        # Check all inserted items return True
        for item in items:
            self.assertTrue(bf.contains(item))
            
        # Check non-inserted item verification
        res = bf.query_with_verification("non_existent_event_9999")
        self.assertFalse(res["exact_result"])

    def test_bloom_duplicate_detection(self):
        bf = BloomFilter(m=2048, k=5)
        # First insertion should not be detected as previously existing
        was_dup_1 = bf.add("match_100_pt_1")
        self.assertFalse(was_dup_1)
        
        # Second insertion of same item must flag True
        was_dup_2 = bf.add("match_100_pt_1")
        self.assertTrue(was_dup_2)
        self.assertEqual(bf.duplicate_reject_count, 1)

    def test_bloom_theoretical_false_positive_formula(self):
        bf = BloomFilter(m=1000, k=4)
        for i in range(100):
            bf.add(f"item_{i}")
        fp = bf.theoretical_fp_rate()
        self.assertGreater(fp, 0.0)
        self.assertLess(fp, 1.0)

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
