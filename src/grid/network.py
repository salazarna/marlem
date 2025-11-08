"""
Grid network topology module for Local Energy Market.

This module provides functions to create and manage different grid network topologies
that can be used for realistic distance calculations in market mechanisms.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch

from ..agent.der import DERAgent
from .base import GridTopology, Location
from .config import get_predefined_config


class GridNetwork:
    """Grid network topology for Local Energy Market."""

    def __init__(self,
                 topology: GridTopology = GridTopology.MESH,
                 num_nodes: int = 5,
                 capacity: float = 1e6,
                 seed: Optional[int] = None) -> None:
        """Initialize the grid network.

        Args:
            topology: Type of grid network topology to create
            num_nodes: Number of nodes to include in the network (for procedural topologies)
            capacity: Total grid capacity (kW)
            seed: Random seed for reproducible topology generation
        """
        self.topology = topology
        self.num_nodes = num_nodes
        self.capacity = max(0.0, capacity)

        # Initialize grid network
        self.graph: nx.Graph = self._create_grid_network(seed)
        self.agent_to_node: Dict[str, str] = {}

        # Edge-specific congestion tracking
        self.edge_flows: Dict[Tuple[str, str], float] = {}
        self.edge_capacities: Dict[Tuple[str, str], float] = {}

        # Initialize grid network state
        self.reset(seed)

        # Validate configuration
        self._check_init()

    def reset(self, seed: Optional[int] = None) -> None:
        """Initialize edge capacities and flows for congestion modeling.

        Args:
            seed: Random seed for reproducible topology generation
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset dictionaries
        self.edge_capacities = {}
        self.edge_flows = {}

        # Calculate inverse weight sum for normalization
        inverse_weight_sum = 0.0
        for u, v, data in self.graph.edges(data=True):
            distance = data.get('weight', 1.0)
            inverse_weight_sum += 1.0 / distance

        # Allocate capacity to each edge based on inverse distance
        for u, v, data in self.graph.edges(data=True):
            # Get edge distance (weight)
            distance = data.get('weight', 1.0)

            # Calculate normalized edge capacity (shorter edges get more capacity) to ensure total capacity is preserved
            edge_capacity = self.capacity * (1.0 / distance) / inverse_weight_sum

            # Store capacity
            self.edge_capacities[(u, v)] = edge_capacity

            # Initialize flow to a small random value (0-30% of capacity)
            self.edge_flows[(u, v)] = np.random.uniform(0, 0.3) * edge_capacity

    def _check_init(self) -> None:
        """Validate GridNetwork configuration parameters.

        Raises:
            ValueError: If capacity is negative
            ValueError: If topology is invalid
            ValueError: If num_nodes is invalid for procedural topologies
        """
        if self.capacity < 0:
            raise ValueError(f"Grid capacity must be non-negative, got {self.capacity}")

        if self.topology not in GridTopology:
            raise ValueError(f"Invalid topology '{self.topology}'. Available: {list(GridTopology)}")

        # For procedural topologies, validate num_nodes
        if self.topology in GridTopology._value2member_map_:
            if self.num_nodes < 2:
                raise ValueError(f"Procedural topology '{self.topology}' requires at least 2 nodes, got {self.num_nodes}")

        # Validate graph was created successfully
        if not self.graph or len(self.graph.nodes) == 0:
            raise ValueError(f"Failed to create grid network with topology '{self.topology}'")

    def _create_grid_network(self, seed: Optional[int] = None) -> nx.Graph:
        """Create a network grid topology for use in market simulations.

        Args:
            seed: Random seed for reproducibility

        Returns:
            NetworkX graph representing the grid network
        """
        if self.topology not in GridTopology:
            raise ValueError(f"Topology '{self.topology}' not defined. Available topologies: {GridTopology._member_names_}.")

        if seed is not None:
            np.random.seed(seed)

        graph = nx.Graph()

        # Pre-defined topologies
        if self.topology in [GridTopology.IEEE34, GridTopology.IEEE13]:
            config = get_predefined_config(self.topology)

            # Add edges to graph
            for u, v, w in config["edges"]:
                graph.add_edge(u, v, weight=w)

            # Store node positions and zone information as graph attributes
            graph.pos = config["positions"]
            graph.zones = config["zones"]

            # Node to zone mapping
            graph.node_to_zone = {}
            for zone_name, node_list in graph.zones.items():
                for node_id in node_list:
                    graph.node_to_zone[node_id] = zone_name

        # Procedural topologies
        else:
            # Mesh network (all nodes interconnected)
            if self.topology == GridTopology.MESH:
                for i in range(self.num_nodes):
                    for j in range(i+1, self.num_nodes):
                        weight = 0.5 + np.random.rand()  # Random weight between 0.5 and 1.5
                        graph.add_edge(str(i), str(j), weight=weight)

                # Create positions in a circle
                graph.pos = {}
                for i in range(self.num_nodes):
                    angle = 2 * np.pi * i / self.num_nodes
                    graph.pos[str(i)] = (5 + 3 * np.cos(angle), 5 + 3 * np.sin(angle))

                # Create basic zones (4 quadrants)
                graph.zones = {"zone1": [str(i) for i in range(self.num_nodes) if i < self.num_nodes/4],
                               "zone2": [str(i) for i in range(self.num_nodes) if self.num_nodes/4 <= i < self.num_nodes/2],
                               "zone3": [str(i) for i in range(self.num_nodes) if self.num_nodes/2 <= i < 3*self.num_nodes/4],
                               "zone4": [str(i) for i in range(self.num_nodes) if i >= 3*self.num_nodes/4]}

                # Create node to zone mapping
                graph.node_to_zone = {}
                for zone_name, node_list in graph.zones.items():
                    for node_id in node_list:
                        graph.node_to_zone[node_id] = zone_name

            # Ring network (circular topology)
            elif self.topology == GridTopology.RING:
                for i in range(self.num_nodes):
                    next_node = (i + 1) % self.num_nodes
                    weight = 0.5 + np.random.rand()
                    graph.add_edge(str(i), str(next_node), weight=weight)

                # Create positions in a circle
                graph.pos = {}
                for i in range(self.num_nodes):
                    angle = 2 * np.pi * i / self.num_nodes
                    graph.pos[str(i)] = (5 + 3 * np.cos(angle), 5 + 3 * np.sin(angle))

                # Zones can be segments of the ring
                segment_size = max(1, self.num_nodes // 4)
                graph.zones = {f"segment{j+1}": [str(i) for i in range(self.num_nodes) if j*segment_size <= i < (j+1)*segment_size] for j in range(4)}

                # Add any remaining nodes to the last segment
                if self.num_nodes > 4 * segment_size:
                    graph.zones["segment4"].extend([str(i) for i in range(self.num_nodes) if i >= 4*segment_size])

                # Create node to zone mapping
                graph.node_to_zone = {}
                for zone_name, node_list in graph.zones.items():
                    for node_id in node_list:
                        graph.node_to_zone[node_id] = zone_name

            # Linear network (nodes in sequence)
            elif self.topology == GridTopology.LINE:
                for i in range(self.num_nodes - 1):
                    weight = 0.5 + np.random.rand()
                    graph.add_edge(str(i), str(i + 1), weight=weight)

                # Create positions in a line
                graph.pos = {str(i): (i, 0) for i in range(self.num_nodes)}

                # Zones can be segments of the line
                segment_size = max(1, self.num_nodes // 4)
                graph.zones = {f"segment{j+1}": [str(i) for i in range(self.num_nodes) if j*segment_size <= i < (j+1)*segment_size] for j in range(4)}

                # Add any remaining nodes to the last segment
                if self.num_nodes > 4 * segment_size:
                    graph.zones["segment4"].extend([str(i) for i in range(self.num_nodes) if i >= 4*segment_size])

                # Create node to zone mapping
                graph.node_to_zone = {}
                for zone_name, node_list in graph.zones.items():
                    for node_id in node_list:
                        graph.node_to_zone[node_id] = zone_name

        return graph

    def assign_agents_to_graph(self, agents: List[DERAgent]) -> None:
        """
        Assign agents to grid network locations, optionally specifying locations for specific agents.

        Args:
            agents: List of DERAgent objects to be assigned locations
        """
        # Extract agent IDs and node IDs from agents
        agent_ids = [a.id for a in agents]
        node_ids = [a.node_id for a in agents]

        # Get list of nodes from grid graph
        positions = list(self.graph.pos.keys())

        if len(agents) > len(positions):
            raise ValueError(f"Not enough nodes ({len(positions)}) to assign all agents ({len(agents)}).")

        if node_ids is None:
            if len(positions) == len(agents):
                assigned_nodes = positions
            else:
                assigned_nodes = np.random.choice(positions, size=len(agents), replace=False)

            # Create mapping for all agents (random assignment)
            self.agent_to_node = {agent_id: str(node) for agent_id, node in zip(agent_ids, assigned_nodes)}

        else:
            # Assign locations based on agent_locations dictionary and random for the rest
            assigned_nodes = set()
            for agent_id, node_id in zip(agent_ids, node_ids):
                if node_id is None:
                    continue
                if node_id not in positions:
                    raise ValueError(f"Node ID '{node_id}' does not exist in the graph.")

                self.agent_to_node[agent_id] = str(node_id)
                assigned_nodes.add(node_id)

            # Assign remaining agents randomly to unassigned nodes
            remaining_agents = [a for a in agent_ids if a not in self.agent_to_node.keys()]
            available_nodes = [n for n in positions if n not in assigned_nodes]

            if len(remaining_agents) > 0:
                if len(remaining_agents) > len(available_nodes):
                    raise ValueError("Not enough available nodes to assign all remaining agents.")

                randomly_assigned_nodes = np.random.choice(available_nodes, size=len(remaining_agents), replace=False)

                for agent_id, node_id in zip(remaining_agents, randomly_assigned_nodes):
                    self.agent_to_node[agent_id] = str(node_id)

        # Update agent locations
        for agent in agents:
            agent.update_location(self.get_location(agent.id))

    def update_flow_from_trade(self,
                               buyer_node: str,
                               seller_node: str,
                               quantity: float) -> None:
        """Update power flow based on a trade between two nodes.

        This simulates the impact of power flow on the network by increasing
        the flow along the shortest path between buyer and seller. The flow
        decreases along the path due to transmission losses.

        Args:
            buyer_node: Node ID of the buyer
            seller_node: Node ID of the seller
            quantity: Trade quantity (energy units) at the source
        """
        try:
            # Find shortest path between buyer and seller
            path = nx.shortest_path(self.graph, seller_node, buyer_node, weight='weight')

            # Initialize remaining power (starts with full quantity at seller node)
            remaining_power = quantity

            # Update flow along the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]

                # Ensure edge is in the correct orientation
                key = (u, v) if (u, v) in self.edge_flows else (v, u)

                # Skip if edge not found (shouldn't happen with valid path)
                if key not in self.edge_flows:
                    continue

                # Get edge length
                edge_length = self.graph[u][v]['weight']

                # Calculate power at the start of this edge
                power_at_edge = remaining_power

                # Update remaining power after this edge
                remaining_power -= self.transmission_loss(edge_length, power_at_edge)

                # Increase flow on this edge (use power at start of edge)
                self.edge_flows[key] += power_at_edge

                # Cap at capacity to avoid unrealistic values
                self.edge_flows[key] = float(min(self.edge_flows[key], self.edge_capacities[key]))

        # No path exists between nodes
        except nx.NetworkXNoPath:
            pass

    def transmission_loss(self,
                          distance: float,
                          quantity: float,
                          loss_factor: float = 0.03) -> float:
        """Calculate transmission losses based on distance between two locations.

        Transmission losses are proportional to the distance between nodes and the
        quantity of energy being transmitted.

        Args:
            distance: Distance between nodes
            quantity: Energy quantity being transmitted (in energy units)
            loss_factor: Transmission loss factor (dimensionless)

        Returns:
            Transmission loss in energy units
        """
        return distance * quantity * np.clip(loss_factor, 0.0, 1.0)

    def get_edge_congestion(self,
                            node1: str,
                            node2: str) -> float:
        """Calculate congestion level for a specific edge.

        Args:
            node1: First node of the edge
            node2: Second node of the edge

        Returns:
            Congestion level (0-1 scale, where 1 is fully congested)
        """
        # Check if edge exists in either direction
        if (node1, node2) in self.edge_flows:
            flow = self.edge_flows[(node1, node2)]
            capacity = self.edge_capacities[(node1, node2)]
            return min(1.0, flow / capacity) if capacity > 0 else 0.0

        elif (node2, node1) in self.edge_flows:
            flow = self.edge_flows[(node2, node1)]
            capacity = self.edge_capacities[(node2, node1)]
            return min(1.0, flow / capacity) if capacity > 0 else 0.0

        # Edge not found
        return 0.0

    def get_path_congestion(self,
                            buyer_node: str,
                            seller_node: str) -> float:
        """Calculate the maximum congestion level along the path between two nodes.

        Args:
            buyer_node: Node ID of the buyer
            seller_node: Node ID of the seller

        Returns:
            Maximum congestion level along the path (0-1 scale)
        """
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, seller_node, buyer_node, weight='weight')

            # Calculate maximum congestion along the path
            max_congestion = 0.0

            for i in range(len(path) - 1):
                edge_congestion = self.get_edge_congestion(path[i], path[i+1])
                max_congestion = max(max_congestion, edge_congestion)

            return max_congestion

        # No path exists
        except nx.NetworkXNoPath:
            return 0.0

    def calculate_congestion_level(self) -> float:
        """Calculate the current grid congestion level.

        Returns:
            Congestion level (0-1 scale, where 1 is fully congested)
        """
        # Calculate average congestion across all edges
        try:
            edge_congestions = []
            for u, v in self.edge_flows.keys():
                edge_congestions.append(self.get_edge_congestion(u, v))

            # Return average congestion if there are edges
            return sum(edge_congestions) / len(edge_congestions) if edge_congestions else 0.0

        # Fallback to simple flow-based congestion calculation
        except Exception:
            total_flow = sum(self.edge_flows.values())
            total_capacity = sum(self.edge_capacities.values())
            return min(1.0, total_flow / total_capacity) if total_capacity > 0 else 0.0

    def get_location(self, agent_id: str) -> Location:
        """Convert a node position and metadata to a Location object.

        Args:
            agent_id: Agent ID

        Returns:
            Location object with node_id, x, y, and zone.
        """
        if self.agent_to_node == {}:
            raise ValueError("No agent to node mapping available.")

        node = self.agent_to_node[agent_id]
        x, y = self.graph.pos[node]
        zone = self.graph.node_to_zone[node] if hasattr(self.graph, 'node_to_zone') and node in self.graph.node_to_zone else None

        return Location(node, x, y, zone)

    def get_node_zone(self, node_id: str) -> str:
        """Get the zone for a specific node.

        Args:
            node_id: ID of the node

        Returns:
            Zone name for the node
        """
        if hasattr(self.graph, 'node_to_zone') and node_id in self.graph.node_to_zone:
            return self.graph.node_to_zone[node_id]
        return "default"

    def visualize(self,
                  agents: Optional[List[DERAgent]] = None,
                  save_path: Optional[str] = None) -> None:
        """
        Visualize the grid network with agent positions and zones.

        Args:
            agents: Optional list of DERAgent objects
            save_path: If provided, saves the figure to this path
        """
        plt.figure(figsize=(12, 10))

        # Create a color map for zones
        unique_zones = set(self.graph.node_to_zone.values())
        zone_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_zones)))
        zone_color_map = {zone: zone_colors[i] for i, zone in enumerate(unique_zones)}
        node_colors = [zone_color_map[self.graph.node_to_zone[node]] if node in self.graph.node_to_zone else 'lightgray' for node in self.graph.nodes()]

        # Draw the network
        nx.draw(self.graph,
                self.graph.pos,
                with_labels=True,
                node_color=node_colors,
                node_size=700,
                edge_color='gray',
                width=1.5,
                font_size=10)

        # If agent information is provided, visualize their positions
        if agents:
            agent_nodes = {}
            for agent_id, node_id in self.agent_to_node.items():
                if node_id in agent_nodes:
                    agent_nodes[node_id].append(agent_id)
                else:
                    agent_nodes[node_id] = [agent_id]

            # Add agent labels
            for node_id, agent_list in agent_nodes.items():
                if node_id in self.graph.pos:
                    x, y = self.graph.pos[node_id]

                    # Draw a slightly larger red circle for the node with agents
                    plt.scatter(x, y, s=800, facecolors='none', edgecolors='red', linewidths=2)

                    # Add label with agent count if more than one
                    if len(agent_list) > 1:
                        plt.text(x, y+0.15, f"{len(agent_list)} agents", horizontalalignment='center', size=8)

        # Add a legend for zones if showing them
        legend_elements = [Patch(facecolor=zone_color_map[zone], label=zone) for zone in unique_zones]
        plt.legend(handles=legend_elements, loc='best')

        # Set title and tight layout
        plt.title(f"Grid Network Topology ({self.topology.value})")
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
