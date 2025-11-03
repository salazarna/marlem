"""
Base classes for grid components.

This module provides the core classes for representing grid elements
like locations and grid network topologies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import networkx as nx
import numpy as np


class GridTopology(Enum):
    """Different grid topologies."""

    IEEE34 = "IEEE34"  # IEEE 34-node test feeder
    IEEE13 = "IEEE13"  # IEEE 13-node test feeder
    MESH = "MESH"  # Mesh topology
    RING = "RING"  # Ring topology
    LINE = "LINE"  # Line topology


@dataclass
class Location:
    """Represents a location in the grid."""
    node_id: Optional[str]  # Unique identifier for the location in the network
    x: float  # Grid x-coordinate
    y: float  # Grid y-coordinate
    zone: Optional[str] = None  # Grid zone identifier

    def distance_to(self,
                    other: 'Location',
                    graph: nx.Graph) -> float:
        """
        Calculate distance to another location.

        If a network graph is provided, calculates the shortest path distance
        along the network. Otherwise, falls back to Euclidean distance.

        Args:
            other: Another Location object
            graph: NetworkX graph representing the network topology
                   Nodes should match node_id values

        Returns:
            Distance between nodes along network
        """
        try:
            return nx.shortest_path_length(graph,
                                           source=self.node_id,
                                           target=other.node_id,
                                           weight='weight')
        except:
            return self._euclidean_distance(other.x, other.y)

    def _euclidean_distance(self, x: float, y: float) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            x: x-coordinate of the other point
            y: y-coordinate of the other point

        Returns:
            Euclidean distance between the two points
        """
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)
