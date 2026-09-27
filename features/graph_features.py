import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FraudGraphBuilder:
    """
    Constructs a multi-entity graph of transactions, users, devices, cards, and IPs
    to extract topological risk indicators, community structures, and mule rings.
    """

    def __init__(self):
        self.graph = nx.Graph()

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.Graph:
        """Constructs an entity interaction graph from transactions."""
        self.graph.clear()

        for _, row in df.iterrows():
            user_node = f"USER_{row['user_id']}"
            card_node = f"CARD_{row['card_id']}"
            dev_node = f"DEV_{row['device_id']}"
            ip_node = f"IP_{row['ip_address']}"
            merch_node = f"MERCH_{row['merchant_id']}"

            # Add nodes with types
            self.graph.add_node(user_node, node_type="user", label=row["user_id"])
            self.graph.add_node(card_node, node_type="card", label=row["card_id"])
            self.graph.add_node(dev_node, node_type="device", label=row["device_id"])
            self.graph.add_node(ip_node, node_type="ip", label=row["ip_address"])
            self.graph.add_node(merch_node, node_type="merchant", label=row["merchant_id"])

            # Add edges
            self.graph.add_edge(user_node, card_node, relationship="owns_card")
            self.graph.add_edge(user_node, dev_node, relationship="used_device")
            self.graph.add_edge(user_node, ip_node, relationship="connected_via")
            self.graph.add_edge(user_node, merch_node, relationship="transacted_at", amount=row["amount"])

        return self.graph


def compute_graph_features(df: pd.DataFrame, graph: nx.Graph = None) -> Tuple[pd.DataFrame, nx.Graph]:
    """
    Computes graph linkage metrics:
    - Device user count (shared device indicator)
    - IP user count (shared proxy/IP indicator)
    - User card count
    - Entity degree centrality
    - Shared infrastructure risk index
    """
    df = df.copy()

    # Pre-compute distinct mappings using pandas for high performance
    device_users = df.groupby("device_id")["user_id"].nunique().to_dict()
    ip_users = df.groupby("ip_address")["user_id"].nunique().to_dict()
    user_devices = df.groupby("user_id")["device_id"].nunique().to_dict()
    user_cards = df.groupby("user_id")["card_id"].nunique().to_dict()
    card_users = df.groupby("card_id")["user_id"].nunique().to_dict()

    df["device_user_count"] = df["device_id"].map(device_users).fillna(1).astype(int)
    df["ip_user_count"] = df["ip_address"].map(ip_users).fillna(1).astype(int)
    df["user_device_count"] = df["user_id"].map(user_devices).fillna(1).astype(int)
    df["user_card_count"] = df["user_id"].map(user_cards).fillna(1).astype(int)
    df["card_user_count"] = df["card_id"].map(card_users).fillna(1).astype(int)

    # Exclude common POS infrastructure from false alarms
    df.loc[df["device_id"].str.startswith("POS_"), "device_user_count"] = 1
    df.loc[df["ip_address"].str.startswith("POS_"), "ip_user_count"] = 1

    # Shared entity flags
    df["is_shared_device"] = (df["device_user_count"] > 1).astype(int)
    df["is_shared_ip"] = (df["ip_user_count"] > 2).astype(int)
    df["is_shared_card"] = (df["card_user_count"] > 1).astype(int)

    # Composite graph risk score (0 to 1 scale)
    graph_risk = (
        (df["device_user_count"] - 1).clip(lower=0, upper=5) * 0.45
        + (df["ip_user_count"] - 1).clip(lower=0, upper=10) * 0.25
        + (df["card_user_count"] - 1).clip(lower=0, upper=3) * 0.30
    )
    df["graph_risk_score"] = np.clip(graph_risk / 3.0, 0.0, 1.0)

    # Build or maintain the graph
    if graph is None:
        builder = FraudGraphBuilder()
        graph = builder.build_from_dataframe(df)

    return df, graph
