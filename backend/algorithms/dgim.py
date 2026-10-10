"""
RallyIQ DGIM Algorithm & Exponential Match Momentum Analyzer
Implements:
1. Datar-Gionis-Indyk-Motwani (DGIM) algorithm for counting binary events
   in a sliding window of size N with O(log^2 N) space.
2. Exponentially Weighted Momentum Scoring (EWMA) with leverage-based point weighting.
"""

import math
from typing import List, Dict, Any, Tuple, Optional

class DGIMBucket:
    def __init__(self, timestamp: int, size: int):
        self.timestamp = timestamp
        self.size = size

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "size": self.size}

class DGIM:
    """
    DGIM Stream Window Counter
    Guarantees max error <= 1 / (2k) using power-of-two bucket hierarchy.
    """
    def __init__(self, window_size: int = 40, max_buckets_per_size: int = 2):
        self.window_size = window_size
        self.k = max_buckets_per_size
        self.buckets: List[DGIMBucket] = []  # Ordered from newest (index 0) to oldest
        self.current_time = 0
        self.raw_window: List[Tuple[int, int]] = []  # Ground truth (timestamp, bit) for empirical verification

    def add_bit(self, bit: int, timestamp: Optional[int] = None) -> None:
        if timestamp is not None:
            self.current_time = timestamp
        else:
            self.current_time += 1
            
        t = self.current_time
        
        # Ground truth window maintenance
        self.raw_window.append((t, bit))
        # Evict raw window items older than window_size
        min_valid_time = t - self.window_size + 1
        self.raw_window = [(ts, b) for (ts, b) in self.raw_window if ts >= min_valid_time]
        
        # Expire buckets strictly outside window
        self.buckets = [b for b in self.buckets if b.timestamp >= min_valid_time]
        
        if bit == 1:
            # Create new bucket of size 1 at current timestamp
            new_bucket = DGIMBucket(t, 1)
            self.buckets.insert(0, new_bucket)
            self._merge_buckets()

    def _merge_buckets(self) -> None:
        """Merge buckets when more than k buckets of the same size exist."""
        size = 1
        while True:
            # Find indices of buckets with current size
            matching_indices = [i for i, b in enumerate(self.buckets) if b.size == size]
            if len(matching_indices) > self.k:
                # Merge the two oldest buckets of this size (the last two in matching_indices)
                idx1 = matching_indices[-2]
                idx2 = matching_indices[-1]
                
                # The newer of the two keeps its timestamp, new size is 2 * size
                merged_bucket = DGIMBucket(self.buckets[idx1].timestamp, size * 2)
                
                # Replace idx1 with merged_bucket, delete idx2
                self.buckets[idx1] = merged_bucket
                self.buckets.pop(idx2)
                
                # Advance to check next power of 2
                size *= 2
            else:
                break

    def count(self, query_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Estimate count of 1s in the window.
        Returns estimated count, exact ground truth, error %, and active bucket metadata.
        """
        w = query_window or self.window_size
        min_time = self.current_time - w + 1
        
        # Ground truth exact count
        exact = sum(b for (ts, b) in self.raw_window if ts >= min_time)
        
        # DGIM estimated count
        relevant_buckets = [b for b in self.buckets if b.timestamp >= min_time]
        
        if not relevant_buckets:
            return {
                "estimated_count": 0,
                "exact_count": exact,
                "error_percentage": 0.0,
                "theoretical_max_error_pct": round(100.0 / (2 * self.k), 1),
                "bucket_count": 0,
                "buckets": []
            }
            
        # Sum all buckets except the oldest relevant bucket
        est = 0
        for b in relevant_buckets[:-1]:
            est += b.size
            
        # Add half of the oldest overlapping bucket
        oldest = relevant_buckets[-1]
        est += oldest.size // 2 if oldest.size > 1 else oldest.size
        
        abs_err = abs(est - exact)
        err_pct = round((abs_err / max(1, exact)) * 100, 2)
        
        return {
            "estimated_count": int(est),
            "exact_count": int(exact),
            "absolute_error": int(abs_err),
            "error_percentage": err_pct,
            "theoretical_max_error_pct": round(100.0 / (2 * self.k), 1),
            "bucket_count": len(relevant_buckets),
            "buckets": [b.to_dict() for b in relevant_buckets]
        }

class MomentumAnalyzer:
    """
    Computes point leverage, exponential momentum curve, and sliding window state.
    """
    def __init__(self, alpha: float = 0.25, window_size: int = 40):
        self.alpha = alpha
        self.window_size = window_size
        self.dgim_p1 = DGIM(window_size=window_size)
        self.dgim_p2 = DGIM(window_size=window_size)
        self.momentum_score = 0.0
        self.history: List[Dict[str, Any]] = []

    def evaluate_point_impact(self, pt_record: Dict[str, Any], p1_name: str, p2_name: str) -> Tuple[float, str]:
        """
        Calculates leverage/impact score for a single tennis point.
        Positive impact favors Player 1, negative favors Player 2.
        """
        score_str = str(pt_record.get("Pts", ""))
        is_break_point = "BP" in score_str or score_str in ["30-40", "40-AD", "15-40", "0-40", "AD-40", "40-30", "40-15", "40-0"]
        is_game_point = "40-" in score_str or "-40" in score_str or "AD" in score_str
        is_tiebreak = bool(pt_record.get("TbSet", False))
        
        server = int(pt_record.get("Svr", 1))
        winner = int(pt_record.get("PtWinner", 1))
        is_ace = bool(pt_record.get("IsAce", False))
        is_df = bool(pt_record.get("IsDoubleFault", False))
        
        base_weight = 1.0
        leverage_label = "Standard Point"
        
        if is_break_point:
            base_weight = 2.5
            leverage_label = "Break Point"
        elif is_tiebreak:
            base_weight = 2.0
            leverage_label = "Tiebreak Point"
        elif is_game_point:
            base_weight = 1.5
            leverage_label = "Game Point"
            
        if is_ace:
            base_weight += 0.5
            leverage_label += " (Ace)"
        elif is_df:
            base_weight += 0.5
            leverage_label += " (Double Fault)"
            
        # Direction: + if P1 won, - if P2 won
        impact = base_weight if winner == 1 else -base_weight
        return impact, leverage_label

    def process_point(self, pt_record: Dict[str, Any], p1_name: str, p2_name: str) -> Dict[str, Any]:
        """Process one point through momentum EWMA and DGIM window."""
        pt_num = int(pt_record.get("Pt", len(self.history) + 1))
        winner = int(pt_record.get("PtWinner", 1))
        
        impact, label = self.evaluate_point_impact(pt_record, p1_name, p2_name)
        
        # Exponential momentum update: M_t = alpha * impact + (1 - alpha) * M_{t-1}
        self.momentum_score = (self.alpha * impact) + ((1.0 - self.alpha) * self.momentum_score)
        
        # Update DGIM for player 1 and player 2 wins
        self.dgim_p1.add_bit(1 if winner == 1 else 0, timestamp=pt_num)
        self.dgim_p2.add_bit(1 if winner == 2 else 0, timestamp=pt_num)
        
        dgim_stats_p1 = self.dgim_p1.count()
        dgim_stats_p2 = self.dgim_p2.count()
        
        point_summary = {
            "point_number": pt_num,
            "set1": int(pt_record.get("Set1", 0)),
            "set2": int(pt_record.get("Set2", 0)),
            "gm1": int(pt_record.get("Gm1", 0)),
            "gm2": int(pt_record.get("Gm2", 0)),
            "score": str(pt_record.get("Pts", "")),
            "server": int(pt_record.get("Svr", 1)),
            "winner": winner,
            "winner_name": p1_name if winner == 1 else p2_name,
            "impact": round(impact, 2),
            "label": label,
            "momentum": round(self.momentum_score, 3),
            "dominant_player": p1_name if self.momentum_score >= 0 else p2_name,
            "dgim_p1_window_wins": dgim_stats_p1["estimated_count"],
            "dgim_p2_window_wins": dgim_stats_p2["estimated_count"],
            "dgim_p1_exact_wins": dgim_stats_p1["exact_count"],
            "dgim_p1_error_pct": dgim_stats_p1["error_percentage"],
            "dgim_p1_buckets": dgim_stats_p1["buckets"],
            "window_size": self.window_size
        }
        self.history.append(point_summary)
        return point_summary

    def replay_match(self, match_points: List[Dict[str, Any]], p1_name: str, p2_name: str) -> Dict[str, Any]:
        """Replay an entire point-by-point match."""
        self.momentum_score = 0.0
        self.history = []
        self.dgim_p1 = DGIM(window_size=self.window_size)
        self.dgim_p2 = DGIM(window_size=self.window_size)
        
        curve = []
        for pt in match_points:
            res = self.process_point(pt, p1_name, p2_name)
            curve.append({
                "pt": res["point_number"],
                "score": f"{res['gm1']}-{res['gm2']} ({res['score']})",
                "momentum": res["momentum"],
                "winner": res["winner"],
                "p1_wins_est": res["dgim_p1_window_wins"],
                "p1_wins_exact": res["dgim_p1_exact_wins"]
            })
            
        # Detect momentum shifts (sign changes in momentum)
        shifts = 0
        for i in range(1, len(curve)):
            if (curve[i]["momentum"] >= 0 and curve[i-1]["momentum"] < 0) or \
               (curve[i]["momentum"] < 0 and curve[i-1]["momentum"] >= 0):
                shifts += 1
                
        return {
            "p1": p1_name,
            "p2": p2_name,
            "total_points": len(match_points),
            "window_size": self.window_size,
            "momentum_swings": shifts,
            "points": self.history,
            "curve": curve
        }
