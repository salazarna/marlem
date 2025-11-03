"""
GridNetwork: Network topology creation, locations and management.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_agents, create_grid_network

from src.grid.base import GridTopology
from src.grid.network import GridNetwork


def test_grid_network(visualize: bool = False) -> None:
    """Test GridNetwork and Location classes thoroughly."""
    # Test different grid topologies
    topologies_to_test = [GridTopology.IEEE34, GridTopology.IEEE13, GridTopology.MESH, GridTopology.RING, GridTopology.LINE]

    for topology in topologies_to_test:
        print(f"--- STEP 1. Testing {topology.value} topology ---")

        if topology == GridTopology.IEEE34 or topology == GridTopology.IEEE13:
            grid = GridNetwork(topology=topology, capacity=1000.0)
        else:
            grid = GridNetwork(topology=topology, num_nodes=8, capacity=1000.0)

        print(f"✓ Grid created: {len(grid.graph.nodes)} nodes, {len(grid.graph.edges)} edges")

        # Test agent assignment using shared utility
        if topology == GridTopology.IEEE34:
            agents = create_agents(num_agents=3,
                                   capacity=50.0,
                                   node_ids=["800", "830", None])  # None for random assignment

            agents[1].capacity = 60.0  # Override capacity for agent_2
            agents[2].capacity = 55.0  # Override capacity for agent_3

        elif topology == GridTopology.IEEE13:
            agents = create_agents(num_agents=3,
                                   capacity=50.0,
                                   node_ids=["650", "632", None])  # None for random assignment

            agents[1].capacity = 60.0  # Override capacity for agent_2
            agents[2].capacity = 55.0  # Override capacity for agent_3

        else:  # For procedural topologies (MESH, RING, LINE)
            agents = create_agents(num_agents=3,
                                   capacity=50.0,
                                   node_ids=["0", "1", None])  # None for random assignment

            agents[1].capacity = 60.0  # Override capacity for agent_2
            agents[2].capacity = 55.0  # Override capacity for agent_3

        grid.assign_agents_to_graph(agents)
        print(f"✓ Agents assigned: {grid.agent_to_node}")

        # Test locations and distances
        loc1 = grid.get_location("agent_0")
        loc2 = grid.get_location("agent_1")
        loc3 = grid.get_location("agent_2")

        distance_a1_a2 = loc1.distance_to(loc2, grid.graph)
        distance_a1_a3 = loc1.distance_to(loc3, grid.graph)
        print(f"✓ Distance calculation (a1, a2): {distance_a1_a2:.3f} units")
        print(f"✓ Distance calculation (a1, a3): {distance_a1_a3:.3f} units")

        # Test congestion tracking
        initial_congestion = grid.get_edge_congestion(loc1.node_id, loc2.node_id)
        grid.update_flow_from_trade(loc1.node_id, loc2.node_id, 50.0)
        final_congestion = grid.get_edge_congestion(loc1.node_id, loc2.node_id)
        print(f"✓ Congestion tracking: {initial_congestion:.3f} → {final_congestion:.3f}")

        # Test transmission loss
        for agent, reputation in zip(agents, [0.5, 0.5, 0.5]):
            agent.reputation = reputation

        loss = grid.transmission_loss(distance_a1_a2, 50.0)
        print(f"✓ Transmission loss: {loss:.3f} Wh for 50 Wh trade")

        # Test preferred partner selection
        preferred_partner = [agent.select_preferred_partner(agents, grid.graph, False) for agent in agents]
        print(f"✓ Preferred partner: {preferred_partner}")

        # Test visualization
        if visualize:
            grid.visualize(agents)

    # Test with grid network for congestion tracking
    grid = create_grid_network()
    agents = create_agents(num_agents=2,
                           capacity=100.0,
                           node_ids=["800", "830"])
    grid.assign_agents_to_graph(agents)

    # Show basic congestion tracking
    initial_congestion = grid.get_path_congestion("800", "830")
    grid.update_flow_from_trade("800", "830", 100.0)
    final_congestion = grid.get_path_congestion("800", "830")
    print(f"✓ Grid congestion tracking: {initial_congestion:.3f} → {final_congestion:.3f}")
    print(f"  - Edge flows: {grid.edge_flows}")
    print(f"  - Edge capacities: {grid.edge_capacities}")


def run_tests() -> bool :
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING GridNetwork TESTS")

    try:
        test_grid_network(visualize=False)
        print("🎉 GridNetwork TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
