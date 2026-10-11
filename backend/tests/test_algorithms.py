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

    def test_dgim_boundary_full_window_exact_counting(self):
        # When stream length <= window_size, no bit precedes the window, count must be exact
        dgim = DGIM(window_size=40, max_buckets_per_size=2)
        # Feed 25 bits: 15 ones, 10 zeros
        bit_stream = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1]
        for b in bit_stream:
            dgim.add_bit(b)
        res = dgim.count()
        self.assertEqual(res["exact_count"], 15)
        self.assertEqual(res["estimated_count"], 15)
        self.assertEqual(res["error_percentage"], 0.0)

    def test_dgim_expiration_outside_window(self):
        # Stream length 60, window size 20: all buckets must have timestamp >= 41
        dgim = DGIM(window_size=20, max_buckets_per_size=2)
        for i in range(1, 61):
            dgim.add_bit(1 if i % 2 == 0 else 0, timestamp=i)
        res = dgim.count()
        for b in res["buckets"]:
            self.assertGreaterEqual(b["timestamp"], 41, "Expired bucket found in window")
        self.assertLessEqual(res["error_percentage"], 50.0)

    def test_dgim_power_of_two_and_error_bound_across_multiple_windows_and_lengths(self):
        import random
        window_sizes = [10, 25, 40, 60]
        stream_lengths = [20, 50, 100, 200]
        for w in window_sizes:
            for l in stream_lengths:
                random.seed(42 + w + l)
                dgim = DGIM(window_size=w, max_buckets_per_size=2)
                for _ in range(l):
                    dgim.add_bit(random.choice([0, 1]))
                res = dgim.count()
                # Verify powers of two
                for b in res["buckets"]:
                    self.assertTrue((b["size"] & (b["size"] - 1)) == 0)
                # Verify theoretical bound <= 50.0% for k=2
                self.assertLessEqual(res["error_percentage"], 50.0 + 1e-6)

class TestMomentumAnalyzer(unittest.TestCase):
    def test_momentum_impact_and_ewma(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        
        # Test Break Point won by Player 1 (Svr=2, so Player 1 is Receiver)
        bp_record = {"Pts": "30-40", "Svr": 2, "PtWinner": 1, "IsAce": False, "IsDoubleFault": False}
        impact, label = analyzer.evaluate_point_impact(bp_record, "Djokovic", "Nadal")
        self.assertGreater(impact, 2.0)
        self.assertIn("Break Point", label)
        self.assertIn("Converted", label)
        
        # Process point
        res = analyzer.process_point(bp_record, "Djokovic", "Nadal")
        self.assertGreater(res["momentum"], 0.0)
        self.assertEqual(res["dominant_player"], "Djokovic")
        self.assertGreater(res["p1_momentum"], 0.0)
        self.assertEqual(res["p2_momentum"], 0.0)

    def test_initial_state_does_not_count_points(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        match_pts = [
            {"Pt": 1, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "0-0", "Svr": 1, "PtWinner": 1},
            {"Pt": 2, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "15-0", "Svr": 1, "PtWinner": 1},
            {"Pt": 3, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "30-0", "Svr": 1, "PtWinner": 2},
        ]
        res = analyzer.replay_match(match_pts, "Federer", "Djokovic")
        # Point 0 is initial pre-match state
        pt0 = res["points"][0]
        self.assertEqual(pt0["point_number"], 0)
        self.assertEqual(pt0["winner"], 0)
        self.assertEqual(pt0["dgim_p1_window_wins"], 0)
        self.assertEqual(pt0["dgim_p1_exact_wins"], 0)
        self.assertEqual(pt0["momentum"], 0.0)
        self.assertEqual(pt0["dominant_player"], "Neutral")
        self.assertEqual(pt0["label"], "Match Start")
        self.assertEqual(len(pt0["dgim_p1_buckets"]), 0)
        
        # Curve point 0 is initial state
        self.assertEqual(res["curve"][0]["pt"], 0)
        self.assertEqual(res["curve"][0]["momentum"], 0.0)
        self.assertEqual(res["curve"][0]["winner"], 0)

    def test_single_step_advances_exactly_one_point(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        match_pts = [
            {"Pt": 1, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "0-0", "Svr": 1, "PtWinner": 1},
            {"Pt": 2, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "15-0", "Svr": 1, "PtWinner": 2},
        ]
        res = analyzer.replay_match(match_pts, "Alcaraz", "Sinner")
        self.assertEqual(len(res["points"]), 3)  # Pt 0, Pt 1, Pt 2
        
        # Step 1: advances to point 1
        pt1 = res["points"][1]
        self.assertEqual(pt1["point_number"], 1)
        self.assertEqual(pt1["winner"], 1)
        self.assertEqual(pt1["winner_name"], "Alcaraz")
        self.assertEqual(pt1["dgim_p1_window_wins"], 1)
        self.assertGreater(pt1["momentum"], 0.0)
        
        # Step 2: advances to point 2
        pt2 = res["points"][2]
        self.assertEqual(pt2["point_number"], 2)
        self.assertEqual(pt2["winner"], 2)
        self.assertEqual(pt2["winner_name"], "Sinner")
        self.assertGreater(pt2["p2_momentum"], 0.0)

    def test_ewma_momentum_consistency_between_players(self):
        analyzer = MomentumAnalyzer(alpha=0.3, window_size=30)
        pts = [
            {"Pt": 1, "Pts": "15-0", "Svr": 1, "PtWinner": 1},
            {"Pt": 2, "Pts": "30-0", "Svr": 1, "PtWinner": 1},
            {"Pt": 3, "Pts": "30-15", "Svr": 1, "PtWinner": 2},
            {"Pt": 4, "Pts": "30-30", "Svr": 1, "PtWinner": 2},
            {"Pt": 5, "Pts": "30-40", "Svr": 1, "PtWinner": 2},  # Break Point
        ]
        res = analyzer.replay_match(pts, "Nadal", "Federer")
        # For each point, momentum == p1_momentum - p2_momentum
        for pt in res["points"][1:]:
            expected_diff = round(pt["p1_momentum"] - pt["p2_momentum"], 3)
            self.assertAlmostEqual(pt["momentum"], expected_diff, places=3)
            self.assertGreaterEqual(pt["p1_momentum"], 0.0)
            self.assertGreaterEqual(pt["p2_momentum"], 0.0)

    def test_break_point_saved_by_server(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        # Svr=1 (Djokovic serving), Pts=15-40 (Nadal has double break point). Djokovic wins point.
        save_record = {"Pts": "15-40", "Svr": 1, "PtWinner": 1, "Gm1": 2, "Gm2": 3}
        impact, label = analyzer.evaluate_point_impact(save_record, "Djokovic", "Nadal")
        self.assertEqual(impact, 2.5)  # Favors Player 1 (Djokovic)
        self.assertIn("Break Point", label)
        self.assertIn("Saved", label)

    def test_game_point_not_misclassified_as_break_point(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        # Svr=1, Pts=40-15: Server leading 40-15. This is Game Point, NOT Break Point!
        gp_record = {"Pts": "40-15", "Svr": 1, "PtWinner": 1, "Gm1": 1, "Gm2": 1}
        impact, label = analyzer.evaluate_point_impact(gp_record, "Djokovic", "Nadal")
        self.assertEqual(impact, 1.5)
        self.assertIn("Game Point", label)
        self.assertNotIn("Break Point", label)

    def test_tiebreak_and_set_point_detection(self):
        analyzer = MomentumAnalyzer(alpha=0.25, window_size=30)
        # Tiebreak point at 6-6 in games
        tb_record = {"Pts": "2-2", "Svr": 1, "PtWinner": 2, "Gm1": 6, "Gm2": 6}
        impact, label = analyzer.evaluate_point_impact(tb_record, "Djokovic", "Nadal")
        self.assertEqual(impact, -2.0)  # Favors Player 2 (Nadal)
        self.assertIn("Tiebreak Point", label)

        # Set point in tiebreak: Server at 6-3
        sp_record = {"Pts": "6-3", "Svr": 1, "PtWinner": 1, "Gm1": 6, "Gm2": 6, "Set1": 0, "Set2": 0}
        impact_sp, label_sp = analyzer.evaluate_point_impact(sp_record, "Djokovic", "Nadal")
        self.assertEqual(impact_sp, 3.0)
        self.assertIn("Set Point", label_sp)

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
        self.assertEqual(bf.n, 1)
        self.assertEqual(len(bf.exact_set), 1)
        self.assertEqual(bf.duplicate_reject_count, 0)
        
        # Second insertion of same item must flag True
        was_dup_2 = bf.add("match_100_pt_1")
        self.assertTrue(was_dup_2)
        self.assertEqual(bf.duplicate_reject_count, 1)
        # Unique count n and exact_set must NOT increase on duplicate insertion
        self.assertEqual(bf.n, 1)
        self.assertEqual(len(bf.exact_set), 1)

    def test_insert_increments_n_exactly_once(self):
        bf = BloomFilter(m=4096, k=5)
        self.assertEqual(bf.n, 0)
        bf.add("event_alpha")
        self.assertEqual(bf.n, 1)
        bf.add("event_beta")
        self.assertEqual(bf.n, 2)
        # Duplicate attempts must not increment n
        bf.add("event_alpha")
        self.assertEqual(bf.n, 2)
        self.assertEqual(bf.duplicate_reject_count, 1)

    def test_false_positive_distinguished_from_confirmed_duplicate(self):
        # Create a small Bloom filter to easily produce a false positive collision
        bf = BloomFilter(m=32, k=3)
        candidate = None
        for i in range(100):
            item = f"key_{i}"
            bf.add(item)
            test_item = f"unseen_probe_{i}"
            if test_item not in bf.exact_set and bf.contains(test_item):
                candidate = test_item
                break

        self.assertIsNotNone(candidate, "Should find a false-positive candidate on small saturated filter")
        init_dups = bf.duplicate_reject_count
        init_fps = bf.false_positive_count
        init_n = bf.n

        # Adding a false-positive item:
        # Must NOT be classified as confirmed duplicate (must return False)
        was_dup = bf.add(candidate)
        self.assertFalse(was_dup)
        # Must increment false_positive_count
        self.assertEqual(bf.false_positive_count, init_fps + 1)
        # Must NOT increment duplicate_reject_count
        self.assertEqual(bf.duplicate_reject_count, init_dups)
        # Must increment unique inserted count n
        self.assertEqual(bf.n, init_n + 1)
        self.assertTrue(candidate in bf.exact_set)

    def test_theoretical_probability_raw_precision(self):
        bf = BloomFilter(m=8192, k=5)
        # n = 0 must return 0.0
        self.assertEqual(bf.theoretical_fp_rate(), 0.0)
        
        # Add 5 items
        for i in range(5):
            bf.add(f"item_{i}")
        fp = bf.theoretical_fp_rate()
        self.assertGreater(fp, 0.0)
        # For m=8192, k=5, n=5, theoretical FP is ~2.6e-13.
        # It must NOT be rounded to 0.0!
        self.assertLess(fp, 1e-10)
        self.assertIsInstance(fp, float)

    def test_empirical_false_positive_uses_only_negative_queries(self):
        bf = BloomFilter(m=4096, k=5)
        bf.add("known_event_1")
        bf.add("known_event_2")

        # Querying existing items (positive queries / duplicates)
        bf.query_with_verification("known_event_1")
        bf.query_with_verification("known_event_2")

        # No absent items queried yet, so empirical FP rate must be 0.0
        self.assertEqual(bf.false_positive_count, 0)
        self.assertEqual(bf.true_negative_count, 0)
        self.assertEqual(bf.empirical_fp_rate(), 0.0)

        # Query absent item
        res = bf.query_with_verification("absent_item_xyz")
        self.assertFalse(res["exact_result"])
        # Should be counted as a negative query (either TN or FP)
        metrics = bf.get_metrics()
        self.assertEqual(metrics["negative_queries_tested"], 1)

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
