"""
RallyIQ ATP Rivalry Network & Graph Analytics
Implements:
1. Weighted Graph modeling of ATP player rivalries
2. Louvain Community Detection algorithm for competitive clusters & eras
3. Graph centrality measures: PageRank, Betweenness, and Degree Centrality
4. Rivalry intensity & head-to-head dominance analytics
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

from backend.config import PROCESSED_RIVALRIES, PROCESSED_PLAYER_STATS

logger = logging.getLogger("rallyiq.graph")

class RivalryNetwork:
    def __init__(self, rivalries_path: str = PROCESSED_RIVALRIES, profiles_path: str = PROCESSED_PLAYER_STATS):
        self.rivalries_path = rivalries_path
        self.profiles_path = profiles_path
        self.raw_rivalries: Dict[str, Any] = {}
        self.player_profiles: Dict[str, Any] = {}
        self.G = nx.Graph()
        self.communities: List[set] = []
        self.modularity_score = 0.0
        self.pagerank_scores: Dict[str, float] = {}
        self.betweenness_scores: Dict[str, float] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct weighted rivalry graph and run graph algorithms."""
        if not os.path.exists(self.rivalries_path):
            logger.warning(f"Rivalries file not found at {self.rivalries_path}")
            return

        with open(self.rivalries_path, "r") as f:
            self.raw_rivalries = json.load(f)
            
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, "r") as f:
                self.player_profiles = json.load(f)

        self.G.clear()
        
        # Populate edges
        for key, riv in self.raw_rivalries.items():
            p_a = riv["player_a"]
            p_b = riv["player_b"]
            total = riv["total_matches"]
            v_total = riv.get("verified_matches", riv["wins_a"] + riv["wins_b"])
            w_a = riv["wins_a"]
            w_b = riv["wins_b"]
            
            # Competitiveness / rivalry intensity: 1.0 when perfectly even (50-50), lower if lopsided
            competitiveness = round((2.0 * min(w_a, w_b)) / max(1, v_total), 2) if v_total > 0 else 0.5
            intensity = round(total * (0.5 + 0.5 * competitiveness), 2)
            
            self.G.add_edge(
                p_a, p_b,
                weight=total,
                meetings=total,
                verified_meetings=v_total,
                unverified_meetings=riv.get("unverified_matches", total - v_total),
                wins_a=w_a,
                wins_b=w_b,
                competitiveness=competitiveness,
                intensity=intensity,
                surfaces=riv.get("surfaces", {})
            )

        # Run Louvain Community Detection
        if len(self.G.nodes) > 0 and len(self.G.edges) > 0:
            self.communities = louvain_communities(self.G, weight="weight", seed=42)
            try:
                self.modularity_score = round(modularity(self.G, self.communities, weight="weight"), 4)
            except Exception:
                self.modularity_score = 0.0
                
            # Run PageRank
            self.pagerank_scores = nx.pagerank(self.G, weight="weight", alpha=0.85)
            # Run Betweenness Centrality
            self.betweenness_scores = nx.betweenness_centrality(self.G, weight="weight")
            
            # Label nodes with community ID and centrality
            community_map = {}
            for c_id, members in enumerate(self.communities):
                for member in members:
                    community_map[member] = c_id
                    
            for node in self.G.nodes:
                self.G.nodes[node]["community"] = community_map.get(node, 0)
                self.G.nodes[node]["pagerank"] = round(self.pagerank_scores.get(node, 0.0), 5)
                self.G.nodes[node]["betweenness"] = round(self.betweenness_scores.get(node, 0.0), 5)
                self.G.nodes[node]["degree"] = self.G.degree(node)
                self.G.nodes[node]["total_matches"] = self.player_profiles.get(node, {}).get("total_matches", self.G.degree(node))

        logger.info(f"Rivalry graph created: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges, {len(self.communities)} communities (Modularity: {self.modularity_score}).")

    def get_network_data(self, min_meetings: int = 3, top_n_nodes: int = 40) -> Dict[str, Any]:
        """
        Generate filtered subgraph JSON for interactive visual representation.
        """
        # Filter nodes by degree / total matches
        nodes_ranked = sorted(
            self.G.nodes(data=True),
            key=lambda x: x[1].get("pagerank", 0) * x[1].get("total_matches", 1),
            reverse=True
        )[:top_n_nodes]
        
        node_names = {n[0] for n in nodes_ranked}
        
        nodes_list = []
        for n_name, attrs in nodes_ranked:
            nodes_list.append({
                "id": n_name,
                "label": n_name,
                "community": attrs.get("community", 0),
                "pagerank": attrs.get("pagerank", 0.0),
                "betweenness": attrs.get("betweenness", 0.0),
                "degree": attrs.get("degree", 0),
                "total_matches": attrs.get("total_matches", 0)
            })

        edges_list = []
        for u, v, data in self.G.edges(data=True):
            if u in node_names and v in node_names and data.get("meetings", 0) >= min_meetings:
                edges_list.append({
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1),
                    "meetings": data.get("meetings", 1),
                    "wins_source": data.get("wins_a", 0) if u < v else data.get("wins_b", 0),
                    "wins_target": data.get("wins_b", 0) if u < v else data.get("wins_a", 0),
                    "competitiveness": data.get("competitiveness", 0.5),
                    "intensity": data.get("intensity", 1.0)
                })

        # Summarize communities
        community_summaries = []
        colors = ["#00F2FE", "#FF007F", "#7928CA", "#00FFA3", "#FF8000", "#FFD700", "#4FACFE"]
        for c_id, members in enumerate(self.communities):
            active_members = [m for m in members if m in node_names]
            if active_members:
                top_members = sorted(active_members, key=lambda x: self.pagerank_scores.get(x, 0), reverse=True)[:4]
                community_summaries.append({
                    "community_id": c_id,
                    "color": colors[c_id % len(colors)],
                    "total_members": len(members),
                    "visible_members": len(active_members),
                    "key_players": top_members,
                    "label": f"Cluster {c_id + 1}: {', '.join(top_members[:2])} era"
                })

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "metrics": {
                "total_players_in_graph": self.G.number_of_nodes(),
                "total_rivalries_in_graph": self.G.number_of_edges(),
                "louvain_communities_count": len(self.communities),
                "graph_modularity_score": self.modularity_score,
                "min_meetings_threshold": min_meetings
            },
            "community_clusters": community_summaries
        }

    def get_rivalry_details(self, player_a: str, player_b: str) -> Optional[Dict[str, Any]]:
        """Get full head-to-head match details for two players."""
        key1 = f"{player_a}_vs_{player_b}"
        key2 = f"{player_b}_vs_{player_a}"
        
        riv = self.raw_rivalries.get(key1) or self.raw_rivalries.get(key2)
        if not riv:
            return None
        return riv

rivalry_network = RivalryNetwork()
