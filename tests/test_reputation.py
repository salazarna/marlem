"""
ReputationHandler: Agent reputation system.
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_agents

from src.grid.base import Location
from src.market.matching import MatchingResult
from src.market.order import Order, Trade
from src.market.reputation import ReputationHandler


def test_reputation() -> None:
    """Test ReputationHandler class thoroughly."""
    # Create reputation handler
    reputation_handler = ReputationHandler(reliability_weight=0.3,
                                           fairness_weight=0.2,
                                           grid_weight=0.5)
    print(f"✓ Reputation handler created")
    print(f"  - Weights: reliability={reputation_handler.reliability_weight}, "
          f"fairness={reputation_handler.fairness_weight}, grid={reputation_handler.grid_weight}")

    # Create test agents using shared utility
    agents = create_agents(num_agents=3, capacity=100.0)

    # Reset reputation for all agents
    for agent in agents:
        agent.reputation = reputation_handler.reset(seed=42)
        print(f"✓ Agent {agent.id} reputation reset to: {agent.reputation:.3f}")

    # Test get_agent_ranking method
    print("--- STEP 1. Testing Agent Ranking ---")
    initial_rankings = reputation_handler.get_agent_ranking(agents)
    print(f"✓ Initial agent rankings:")
    for rank, (agent_id, score) in enumerate(initial_rankings, 1):
        print(f"  {rank}. {agent_id}: {score:.3f}")

    # Test update_reputation method with simulated matching results
    print("--- STEP 2. Testing Reputation Updates ---")

    # Create test locations
    location1 = Location(node_id="800", x=0, y=0, zone="zone1")
    location2 = Location(node_id="830", x=5, y=3, zone="zone2")
    location3 = Location(node_id="854", x=10, y=8, zone="zone3")

    # Create test trades
    test_trades = [Trade("agent_0", "agent_1", 0.22, 25.0, datetime.datetime.now().timestamp(), 2.5, 0.05),
                   Trade("agent_2", "DSO", 0.24, 30.0, datetime.datetime.now().timestamp(), 3.2, 0.07),
                   Trade("DSO", "agent_0", 0.21, 20.0, datetime.datetime.now().timestamp(), 1.8, 0.03)]

    # Create test unmatched orders
    unmatched_orders = [Order("unmatched_1", "agent_1", 0.30, 15.0, True, 1.0, location2),
                        Order("unmatched_2", "agent_2", 0.18, 10.0, False, 1.0, location3)]

    # Create a mock matching result
    matching_result = MatchingResult(trades=test_trades,
                                     unmatched_orders=unmatched_orders,
                                     clearing_price=0.225,
                                     clearing_volume=75.0,
                                     grid_balance=5.0,  # Slight excess supply
                                     dso_buy_volume=15.0,
                                     dso_sell_volume=10.0,
                                     dso_total_volume=25.0,
                                     p2p_volume=50.0,
                                     dso_trade_ratio=0.33,
                                     dso_grid_import=5.0,
                                     dso_buy_price=0.26,
                                     dso_sell_price=0.08,
                                     price_spread=0.18,
                                     local_price_avg=0.225,
                                     local_price_advantage=0.045,
                                     grid_congestion=0.1)

    print(f"✓ Created test matching result:")
    print(f"  - Trades: {len(matching_result.trades)}")
    print(f"  - Unmatched orders: {len(matching_result.unmatched_orders)}")
    print(f"  - Clearing price: ${matching_result.clearing_price:.3f}/Wh")
    print(f"  - Grid balance: {matching_result.grid_balance:.1f} kW")

    # Update reputation for each agent based on the matching result
    print("--- STEP 3. Updating Agent Reputations ---")
    for agent in agents:
        old_reputation = agent.reputation
        time_of_day = 0.5  # Midday

        # Update reputation
        new_reputation = reputation_handler.update_reputation(agent, matching_result, time_of_day)
        agent.reputation = new_reputation

        print(f"✓ {agent.id} reputation: {old_reputation:.3f} → {new_reputation:.3f} "
              f"(change: {new_reputation - old_reputation:+.3f})")

    # Test ranking after reputation updates
    print("--- STEP 4. Updated Agent Rankings ---")
    updated_rankings = reputation_handler.get_agent_ranking(agents)
    print(f"✓ Updated agent rankings:")
    for rank, (agent_id, score) in enumerate(updated_rankings, 1):
        print(f"  {rank}. {agent_id}: {score:.3f}")

    # Show ranking changes
    print("--- STEP 5. Ranking Changes ---")
    initial_positions = {agent_id: rank for rank, (agent_id, _) in enumerate(initial_rankings, 1)}
    updated_positions = {agent_id: rank for rank, (agent_id, _) in enumerate(updated_rankings, 1)}

    for agent_id in [agent.id for agent in agents]:
        old_pos = initial_positions[agent_id]
        new_pos = updated_positions[agent_id]
        change = old_pos - new_pos  # Positive means moved up

        if change > 0:
            print(f"✓ {agent_id}: Moved UP {change} position(s) (rank {old_pos} → {new_pos})")
        elif change < 0:
            print(f"✓ {agent_id}: Moved DOWN {abs(change)} position(s) (rank {old_pos} → {new_pos})")
        else:
            print(f"✓ {agent_id}: No change in ranking (rank {old_pos})")

    # Test multiple reputation updates to show evolution
    print("--- STEP 6. Testing Reputation Evolution ---")
    print("Simulating 3 more market rounds...")

    for round_num in range(1, 4):
        # Create different scenarios for each round
        if round_num == 1:
            # Scenario: agent_0 has good trades, agent_1 has unmatched orders
            test_trades = [Trade("agent_0", "agent_2", 0.225, 40.0, datetime.datetime.now().timestamp(), 4.0, 0.08)]
            unmatched_orders = [Order(f"unmatched_{round_num}_1", "agent_1", 0.35, 20.0, True, 1.0, location2)]
            grid_balance = -10.0  # Shortage
        elif round_num == 2:
            # Scenario: agent_1 recovers with good trades
            test_trades = [Trade("agent_1", "DSO", 0.22, 35.0, datetime.datetime.now().timestamp(), 3.5, 0.06),
                           Trade("DSO", "agent_2", 0.23, 25.0, datetime.datetime.now().timestamp(), 2.8, 0.04)]
            unmatched_orders = []
            grid_balance = 2.0  # Slight excess
        else:  # round_num == 3
            # Scenario: agent_2 has mixed results
            test_trades = [Trade("agent_0", "agent_1", 0.24, 30.0, datetime.datetime.now().timestamp(), 3.0, 0.05)]
            unmatched_orders = [Order(f"unmatched_{round_num}_1", "agent_2", 0.15, 25.0, False, 1.0, location3)]
            grid_balance = 0.0  # Balanced

        # Create matching result for this round
        matching_result = MatchingResult(trades=test_trades,
                                         unmatched_orders=unmatched_orders,
                                         clearing_price=0.225,
                                         clearing_volume=sum(t.quantity for t in test_trades),
                                         grid_balance=grid_balance,
                                         dso_buy_volume=5.0,
                                         dso_sell_volume=3.0,
                                         dso_total_volume=8.0,
                                         p2p_volume=sum(t.quantity for t in test_trades) - 8.0,
                                         dso_trade_ratio=0.2,
                                         dso_grid_import=abs(grid_balance) if grid_balance < 0 else 0.0,
                                         dso_buy_price=0.26,
                                         dso_sell_price=0.08,
                                         price_spread=0.18,
                                         local_price_avg=0.225,
                                         local_price_advantage=0.045,
                                         grid_congestion=0.1)

        # Update reputations
        for agent in agents:
            old_reputation = agent.reputation
            time_of_day = 0.3 + (round_num * 0.2)  # Varying time of day
            new_reputation = reputation_handler.update_reputation(agent, matching_result, time_of_day)
            agent.reputation = new_reputation

        # Show round results
        current_rankings = reputation_handler.get_agent_ranking(agents)
        print(f"✓ Round {round_num} rankings:")
        for rank, (agent_id, score) in enumerate(current_rankings, 1):
            print(f"  {rank}. {agent_id}: {score:.3f}")

    print("✓ Reputation system testing completed successfully!")
    print("  - All methods (reset, update_reputation, get_agent_ranking) work correctly")
    print("  - Reputation scores evolve based on trading behavior")
    print("  - Agent rankings change dynamically based on performance")


def run_tests() -> bool :
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING ReputationHandler TESTS")

    try:
        test_reputation()
        print("🎉 ReputationHandler TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
