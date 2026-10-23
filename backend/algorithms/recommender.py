"""
RallyIQ Similar Playstyle Recommender
Extracts multi-dimensional statistical feature vectors for ATP players,
applies feature scaling, computes Cosine Similarity, and generates
transparent feature-level explanations and archetype classifications.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import PROCESSED_PLAYER_STATS

logger = logging.getLogger("rallyiq.recommender")

FEATURE_NAMES = [
    "ace_rate",              # Aces per 100 serve pts
    "df_rate",               # Double faults per 100 serve pts
    "first_in_pct",          # 1st Serve in %
    "first_won_pct",         # 1st Serve win %
    "second_won_pct",        # 2nd Serve win %
    "bp_save_pct",           # Break points saved %
    "return_won_pct",        # Return points won %
    "winner_unforced_ratio"  # Winners to Unforced Error ratio
]

FEATURE_LABELS = {
    "ace_rate": "Ace Rate (%)",
    "df_rate": "Double Fault Rate (%)",
    "first_in_pct": "1st Serve In (%)",
    "first_won_pct": "1st Serve Win (%)",
    "second_won_pct": "2nd Serve Win (%)",
    "bp_save_pct": "Break Point Save (%)",
    "return_won_pct": "Return Win (%)",
    "winner_unforced_ratio": "Winner / Unforced Ratio"
}

class PlaystyleRecommender:
    def __init__(self, profiles_path: str = PROCESSED_PLAYER_STATS, min_matches: int = 5):
        self.profiles_path = profiles_path
        self.min_matches = min_matches
        self.player_profiles: Dict[str, Any] = {}
        self.qualified_players: List[str] = []
        self.feature_matrix: Optional[np.ndarray] = None
        self.normalized_matrix: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self._load_and_fit()

    def _load_and_fit(self) -> None:
        """Load player statistics, filter eligible players, and fit scaler."""
        if not os.path.exists(self.profiles_path):
            logger.warning(f"Player profiles file not found at {self.profiles_path}")
            return
            
        with open(self.profiles_path, "r") as f:
            self.player_profiles = json.load(f)
            
        # Filter players with at least min_matches
        eligible = []
        matrix_rows = []
        
        for player, data in self.player_profiles.items():
            tot_matches = data.get("total_matches", 0)
            overall = data.get("overall", {})
            if tot_matches >= self.min_matches and overall:
                row = []
                for feat in FEATURE_NAMES:
                    val = float(overall.get(feat, 0.0))
                    row.append(val)
                eligible.append(player)
                matrix_rows.append(row)
                
        if not matrix_rows:
            logger.warning("No eligible players found for recommender.")
            return

        self.qualified_players = eligible
        self.feature_matrix = np.array(matrix_rows, dtype=np.float64)
        
        # Honest missing / zero value imputation with population column median
        col_medians = np.nanmedian(np.where(self.feature_matrix == 0, np.nan, self.feature_matrix), axis=0)
        col_medians = np.nan_to_num(col_medians, nan=50.0)
        
        for j in range(self.feature_matrix.shape[1]):
            zero_mask = self.feature_matrix[:, j] == 0
            self.feature_matrix[zero_mask, j] = col_medians[j]
            
        # Z-score normalization across all features
        self.normalized_matrix = self.scaler.fit_transform(self.feature_matrix)
        logger.info(f"Recommender fitted with {len(self.qualified_players)} ATP players across {len(FEATURE_NAMES)} features.")

    def classify_archetype(self, features: Dict[str, float]) -> str:
        """Classify player into tactical tennis archetype based on statistical traits."""
        ace_rate = features.get("ace_rate", 0)
        f_won = features.get("first_won_pct", 0)
        ret_won = features.get("return_won_pct", 0)
        wu_ratio = features.get("winner_unforced_ratio", 1.0)
        
        if ace_rate >= 11.0 and f_won >= 75.0:
            return "Big Server / Serve-Dominant"
        elif ret_won >= 40.0 and wu_ratio <= 1.15:
            return "Counterpuncher / Defensive Specialist"
        elif wu_ratio >= 1.4 and f_won >= 73.0:
            return "Aggressive Baseliner / Power Hitter"
        elif ret_won >= 38.0 and ace_rate >= 7.0:
            return "All-Court Elite / Complete Specialist"
        else:
            return "Tactical Baseliner"

    def find_similar_players(self, target_player: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Find top K most similar players to target_player using Cosine Similarity.
        Returns recommendations with feature attribution and radar data.
        """
        if self.feature_matrix is None or self.normalized_matrix is None or not self.qualified_players:
            return {
                "error": "Recommender is not fitted or no eligible players are available.",
                "available_players": [],
            }

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, top_k)

        if target_player not in self.qualified_players:
            # Check case-insensitive match
            matches = [p for p in self.qualified_players if p.lower() == target_player.lower()]
            if matches:
                target_player = matches[0]
            else:
                return {
                    "error": f"Player '{target_player}' not found or has fewer than {self.min_matches} charted matches.",
                    "available_players": self.qualified_players[:30]
                }
                
        target_idx = self.qualified_players.index(target_player)
        target_vec = self.normalized_matrix[target_idx].reshape(1, -1)
        target_raw = self.feature_matrix[target_idx]
        
        # Compute Cosine Similarities against all qualified players
        similarities = cosine_similarity(target_vec, self.normalized_matrix)[0]
        
        # Sort descending, exclude self (target_idx)
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_indices = [i for i in ranked_indices if i != target_idx][:top_k]
        
        target_dict = {feat: round(float(target_raw[j]), 2) for j, feat in enumerate(FEATURE_NAMES)}
        target_archetype = self.classify_archetype(target_dict)
        
        recommendations = []
        for idx in ranked_indices:
            p_name = self.qualified_players[idx]
            sim_score = round(float(similarities[idx]) * 100.0, 1)
            p_raw = self.feature_matrix[idx]
            p_dict = {feat: round(float(p_raw[j]), 2) for j, feat in enumerate(FEATURE_NAMES)}
            
            # Feature contribution to similarity: smallest absolute normalized distance
            abs_diffs = np.abs(self.normalized_matrix[target_idx] - self.normalized_matrix[idx])
            top_align_indices = np.argsort(abs_diffs)[:3]
            top_aligned_features = [
                {
                    "feature": FEATURE_LABELS[FEATURE_NAMES[k]],
                    "target_value": target_dict[FEATURE_NAMES[k]],
                    "match_value": p_dict[FEATURE_NAMES[k]],
                    "difference": round(float(abs(target_dict[FEATURE_NAMES[k]] - p_dict[FEATURE_NAMES[k]])), 2)
                }
                for k in top_align_indices
            ]
            
            recommendations.append({
                "player": p_name,
                "similarity_score": sim_score,
                "archetype": self.classify_archetype(p_dict),
                "total_matches": self.player_profiles[p_name].get("total_matches", 0),
                "features": p_dict,
                "key_similar_traits": top_aligned_features
            })
            
        return {
            "target_player": target_player,
            "target_matches": self.player_profiles[target_player].get("total_matches", 0),
            "target_archetype": target_archetype,
            "target_features": target_dict,
            "recommendations": recommendations,
            "feature_labels": FEATURE_LABELS
        }

recommender = PlaystyleRecommender()
