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
    DGIM Stream Window Counter (Datar-Gionis-Indyk-Motwani algorithm).
    Counts 1-bits in a sliding window of size N using O(log^2 N) memory.
    Guarantees theoretical max relative error <= 1 / (2 * (k - 1)) for k >= 2
    using a power-of-two bucket hierarchy where at most k buckets of the same
    size exist. For standard k=2 (at most 2 buckets per size), max error <= 50.0%.
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
        
        # DGIM estimated count: select buckets whose latest timestamp is >= min_time
        relevant_buckets = [b for b in self.buckets if b.timestamp >= min_time]
        theoretical_bound_pct = round(100.0 / (2 * max(1, self.k - 1)), 1)
        
        if not relevant_buckets:
            return {
                "estimated_count": 0,
                "exact_count": exact,
                "absolute_error": exact,
                "error_percentage": 0.0,
                "theoretical_max_error_pct": theoretical_bound_pct,
                "bucket_count": 0,
                "buckets": []
            }
            
        # When the window covers the entire stream (min_time <= 1), all buckets are inside the window
        if min_time <= 1:
            est = sum(b.size for b in relevant_buckets)
        else:
            # Sum all buckets except the oldest relevant bucket
            est = sum(b.size for b in relevant_buckets[:-1])
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
            "theoretical_max_error_pct": theoretical_bound_pct,
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
        self.p1_momentum = 0.0
        self.p2_momentum = 0.0
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset analyzer to initial pre-match state."""
        self.momentum_score = 0.0
        self.p1_momentum = 0.0
        self.p2_momentum = 0.0
        self.history = []
        self.dgim_p1 = DGIM(window_size=self.window_size)
        self.dgim_p2 = DGIM(window_size=self.window_size)

    def evaluate_point_impact(self, pt_record: Dict[str, Any], p1_name: str, p2_name: str) -> Tuple[float, str]:
        """
        Calculates leverage/impact score for a single tennis point.
        Positive impact favors Player 1, negative favors Player 2.
        
        Evaluates genuine tennis game situations:
        - Break Points (Receiver leading at 0-40, 15-40, 30-40, 40-AD)
        - Game Points (Server leading at 40-0, 40-15, 40-30, AD-40)
        - Tiebreak Points (at 6-6 in games)
        - Set Points / Match Points (decisive games/points)
        - Aces and Double Faults
        """
        score_str = str(pt_record.get("Pts", "")).strip()
        server = int(pt_record.get("Svr", 1))  # 1 = Player 1, 2 = Player 2
        receiver = 2 if server == 1 else 1
        winner = int(pt_record.get("PtWinner", 1))
        
        gm1 = int(pt_record.get("Gm1", 0))
        gm2 = int(pt_record.get("Gm2", 0))
        set1 = int(pt_record.get("Set1", 0))
        set2 = int(pt_record.get("Set2", 0))
        
        is_ace = bool(pt_record.get("IsAce", False))
        is_df = bool(pt_record.get("IsDoubleFault", False))
        
        # 1. Detect Tiebreak (games tied at 6-6 or higher)
        is_tiebreak = (gm1 >= 6 and gm2 >= 6 and gm1 == gm2)
        
        # 2. Break Points (Receiver has game point opportunity)
        # In Tennis Abstract charting, Pts is always Server-Score first, Receiver-Score second.
        # Break point occurs strictly when Receiver is at game point: 0-40, 15-40, 30-40, 40-AD.
        is_break_point = not is_tiebreak and (score_str in ["0-40", "15-40", "30-40", "40-AD"] or "BP" in score_str)
        
        # 3. Game Points for Server (40-0, 40-15, 40-30, AD-40)
        is_game_point = not is_tiebreak and (score_str in ["40-0", "40-15", "40-30", "AD-40"])
        
        # 4. Set Point / Match Point detection
        is_set_point = False
        is_match_point = False
        sp_player = 0
        
        if is_tiebreak and "-" in score_str:
            try:
                parts = score_str.split("-")
                s_pts, r_pts = int(parts[0]), int(parts[1])
                # First to 7 with 2-point margin
                if s_pts >= 6 and (s_pts - r_pts) >= 1:
                    is_set_point = True
                    sp_player = server
                elif r_pts >= 6 and (r_pts - s_pts) >= 1:
                    is_set_point = True
                    sp_player = receiver
            except (ValueError, IndexError):
                pass
        elif not is_tiebreak:
            # In a regular game, check if this game point or break point would win the set
            gm_svr = gm1 if server == 1 else gm2
            gm_rec = gm2 if server == 1 else gm1
            if is_game_point and gm_svr >= 5 and (gm_svr - gm_rec) >= 1:
                is_set_point = True
                sp_player = server
            elif is_break_point and gm_rec >= 5 and (gm_rec - gm_svr) >= 1:
                is_set_point = True
                sp_player = receiver

        if is_set_point:
            sets_sp = set1 if sp_player == 1 else set2
            if sets_sp >= 1:
                is_match_point = True

        # Assign leverage weights and descriptive labels
        p1_short = p1_name.split()[-1] if p1_name else "Player 1"
        p2_short = p2_name.split()[-1] if p2_name else "Player 2"
        sp_name = p1_short if sp_player == 1 else p2_short
        bp_name = p1_short if receiver == 1 else p2_short
        gp_name = p1_short if server == 1 else p2_short

        if is_match_point:
            base_weight = 3.5
            if winner == sp_player:
                leverage_label = f"Match Point - Won by {sp_name}"
            else:
                leverage_label = "Match Point - Saved"
        elif is_set_point:
            base_weight = 3.0
            if winner == sp_player:
                leverage_label = f"Set Point - Won by {sp_name}"
            else:
                leverage_label = "Set Point - Saved"
        elif is_break_point:
            base_weight = 2.5
            if winner == receiver:
                leverage_label = f"Break Point - Converted by {bp_name}"
            else:
                leverage_label = f"Break Point - Saved by {gp_name}"
        elif is_tiebreak:
            base_weight = 2.0
            leverage_label = "Tiebreak Point"
        elif is_game_point:
            base_weight = 1.5
            if winner == server:
                leverage_label = f"Game Point - Held by {gp_name}"
            else:
                leverage_label = "Game Point"
        else:
            base_weight = 1.0
            leverage_label = "Standard Point"

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
        pt_num = int(pt_record.get("Pt", len(self.history)))
        winner = int(pt_record.get("PtWinner", 1))
        
        impact, label = self.evaluate_point_impact(pt_record, p1_name, p2_name)
        base_weight = abs(impact)
        
        # Player-specific impact for EWMA:
        # Player 1 receives positive leverage weight on win; Player 2 receives positive leverage weight on win
        p1_imp = base_weight if winner == 1 else 0.0
        p2_imp = base_weight if winner == 2 else 0.0
        
        # Consistent EWMA update:
        # M_{i, t} = alpha * impact_{i, t} + (1 - alpha) * M_{i, t-1}
        self.p1_momentum = (self.alpha * p1_imp) + ((1.0 - self.alpha) * self.p1_momentum)
        self.p2_momentum = (self.alpha * p2_imp) + ((1.0 - self.alpha) * self.p2_momentum)
        
        # Net differential momentum: M_t = M_{1, t} - M_{2, t}
        self.momentum_score = self.p1_momentum - self.p2_momentum
        
        # Update DGIM for player 1 and player 2 wins
        self.dgim_p1.add_bit(1 if winner == 1 else 0, timestamp=pt_num)
        self.dgim_p2.add_bit(1 if winner == 2 else 0, timestamp=pt_num)
        
        dgim_stats_p1 = self.dgim_p1.count()
        dgim_stats_p2 = self.dgim_p2.count()
        
        dominant = "Neutral"
        if self.momentum_score > 0.001:
            dominant = p1_name
        elif self.momentum_score < -0.001:
            dominant = p2_name

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
            "p1_momentum": round(self.p1_momentum, 3),
            "p2_momentum": round(self.p2_momentum, 3),
            "dominant_player": dominant,
            "dgim_p1_window_wins": dgim_stats_p1["estimated_count"],
            "dgim_p2_window_wins": dgim_stats_p2["estimated_count"],
            "dgim_p1_exact_wins": dgim_stats_p1["exact_count"],
            "dgim_p1_error_pct": dgim_stats_p1["error_percentage"],
            "dgim_p1_buckets": dgim_stats_p1["buckets"],
            "theoretical_max_error_pct": dgim_stats_p1["theoretical_max_error_pct"],
            "window_size": self.window_size
        }
        self.history.append(point_summary)
        return point_summary

    def replay_match(self, match_points: List[Dict[str, Any]], p1_name: str, p2_name: str) -> Dict[str, Any]:
        """Replay an entire point-by-point match starting from pre-match initial state (point 0)."""
        self.reset()
        
        # Point 0 represents the initial pre-match state before any points are played
        first_svr = int(match_points[0].get("Svr", 1)) if match_points else 1
        initial_point = {
            "point_number": 0,
            "set1": 0,
            "set2": 0,
            "gm1": 0,
            "gm2": 0,
            "score": "0-0",
            "server": first_svr,
            "winner": 0,
            "winner_name": "None (Match Start)",
            "impact": 0.0,
            "label": "Match Start",
            "momentum": 0.0,
            "p1_momentum": 0.0,
            "p2_momentum": 0.0,
            "dominant_player": "Neutral",
            "dgim_p1_window_wins": 0,
            "dgim_p2_window_wins": 0,
            "dgim_p1_exact_wins": 0,
            "dgim_p1_error_pct": 0.0,
            "dgim_p1_buckets": [],
            "theoretical_max_error_pct": round(100.0 / (2 * max(1, self.dgim_p1.k - 1)), 1),
            "window_size": self.window_size
        }
        self.history.append(initial_point)
        
        curve = [{
            "pt": 0,
            "score": "0-0 (0-0)",
            "momentum": 0.0,
            "p1_momentum": 0.0,
            "p2_momentum": 0.0,
            "winner": 0,
            "p1_wins_est": 0,
            "p1_wins_exact": 0
        }]
        
        for pt in match_points:
            res = self.process_point(pt, p1_name, p2_name)
            curve.append({
                "pt": res["point_number"],
                "score": f"{res['gm1']}-{res['gm2']} ({res['score']})",
                "momentum": res["momentum"],
                "p1_momentum": res["p1_momentum"],
                "p2_momentum": res["p2_momentum"],
                "winner": res["winner"],
                "p1_wins_est": res["dgim_p1_window_wins"],
                "p1_wins_exact": res["dgim_p1_exact_wins"]
            })
            
        # Detect momentum shifts (sign changes in momentum)
        shifts = 0
        for i in range(2, len(curve)):
            prev_m = curve[i-1]["momentum"]
            curr_m = curve[i]["momentum"]
            if (curr_m > 0 and prev_m < 0) or (curr_m < 0 and prev_m > 0):
                shifts += 1
                
        return {
            "p1": p1_name,
            "p2": p2_name,
            "total_points": len(match_points),
            "window_size": self.window_size,
            "alpha": self.alpha,
            "momentum_swings": shifts,
            "initial_state": initial_point,
            "points": self.history,
            "curve": curve
        }
