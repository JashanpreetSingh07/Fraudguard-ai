import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class GraphInvestigationEngine:
    """
    Forensic Graph Analysis Engine for uncovering financial crime networks,
    mule syndicates, device farms, and shared proxy infrastructure.
    """

    NODE_COLORS = {
        "user": "#3b82f6",       # Blue
        "card": "#10b981",       # Green
        "device": "#f59e0b",     # Amber
        "ip": "#8b5cf6",         # Purple
        "merchant": "#64748b",   # Slate Gray
        "fraud_hub": "#ef4444",   # Red Alert
    }

    def __init__(self, graph: Optional[nx.Graph] = None):
        self.graph = graph or nx.Graph()

    def build_from_dataframe(self, df: pd.DataFrame) -> "GraphInvestigationEngine":
        """Builds heterogeneous entity graph from transaction records."""
        self.graph.clear()

        for _, row in df.iterrows():
            u = f"USR_{row['user_id']}"
            c = f"CARD_{row['card_id']}"
            d = f"DEV_{row['device_id']}"
            ip = f"IP_{row['ip_address']}"
            m = f"MERCH_{row['merchant_id']}"

            is_fraud = bool(row.get("is_fraud", 0) == 1 or row.get("composite_risk_score", 0) >= config.THRESHOLD_HIGH)

            # Add nodes
            self.graph.add_node(u, node_type="user", label=str(row["user_id"]), is_fraud=is_fraud)
            self.graph.add_node(c, node_type="card", label=str(row["card_id"]), is_fraud=is_fraud)
            self.graph.add_node(d, node_type="device", label=str(row["device_id"]), is_fraud=is_fraud)
            self.graph.add_node(ip, node_type="ip", label=str(row["ip_address"]), is_fraud=is_fraud)
            self.graph.add_node(m, node_type="merchant", label=str(row.get("merchant_category", row["merchant_id"])), is_fraud=False)

            # Add edges with attributes
            self.graph.add_edge(u, c, rel="OWNS_CARD")
            self.graph.add_edge(u, d, rel="USED_DEVICE")
            self.graph.add_edge(u, ip, rel="ACCESSED_VIA_IP")
            self.graph.add_edge(u, m, rel="TRANSACTED_AT", amount=float(row["amount"]))

        return self

    def extract_subgraph(self, center_id: str, hops: int = 2, max_nodes: int = 40) -> Dict[str, Any]:
        """
        Extracts ego-network around a target entity for forensic inspection.
        Returns serialized nodes and edges formatted for visualization engines.
        """
        # Resolve target node id
        target = None
        for n in self.graph.nodes():
            if center_id.upper() in n.upper() or n.upper().endswith(center_id.upper()):
                target = n
                break

        if target is None:
            # Fallback to most connected high-degree node
            degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
            target = degrees[0][0] if degrees else None

        if target is None:
            return {"nodes": [], "edges": [], "summary": "Graph is empty."}

        # Ego subgraph extraction
        sub_nodes = set([target])
        current_layer = set([target])
        for _ in range(hops):
            next_layer = set()
            for n in current_layer:
                next_layer.update(self.graph.neighbors(n))
            sub_nodes.update(next_layer)
            current_layer = next_layer
            if len(sub_nodes) >= max_nodes:
                break

        subgraph = self.graph.subgraph(list(sub_nodes)[:max_nodes])

        nodes_list = []
        for n in subgraph.nodes():
            attrs = subgraph.nodes[n]
            ntype = attrs.get("node_type", "entity")
            is_fraud = attrs.get("is_fraud", False)

            color = self.NODE_COLORS["fraud_hub"] if is_fraud else self.NODE_COLORS.get(ntype, "#64748b")
            size = 35 if n == target else (25 if is_fraud else 18)

            nodes_list.append({
                "id": n,
                "label": attrs.get("label", n),
                "type": ntype,
                "color": color,
                "size": size,
                "is_fraud": is_fraud,
                "degree": subgraph.degree[n],
            })

        edges_list = []
        for u, v, data in subgraph.edges(data=True):
            edges_list.append({
                "source": u,
                "target": v,
                "label": data.get("rel", "CONNECTED_TO"),
                "color": "#94a3b8",
            })

        summary = f"Forensic subgraph for '{target}': {len(nodes_list)} entities, {len(edges_list)} connections."

        return {
            "center_node": target,
            "nodes": nodes_list,
            "edges": edges_list,
            "summary": summary,
        }

    def detect_collusion_rings(self, min_users: int = 3) -> List[Dict[str, Any]]:
        """
        Discovers shared-infrastructure syndicates:
        Finds devices or IPs shared across multiple user accounts.
        """
        syndicates = []

        for node, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("node_type")
            if ntype in ["device", "ip"]:
                # Ignore generic POS terminals
                if "POS_" in node:
                    continue

                neighbors = list(self.graph.neighbors(node))
                users = [nb for nb in neighbors if self.graph.nodes[nb].get("node_type") == "user"]

                if len(users) >= min_users:
                    syndicates.append({
                        "infrastructure_node": node,
                        "type": ntype,
                        "user_count": len(users),
                        "linked_users": [self.graph.nodes[u].get("label", u) for u in users],
                        "risk_level": "CRITICAL" if len(users) >= 4 else "HIGH",
                    })

        return sorted(syndicates, key=lambda x: x["user_count"], reverse=True)
