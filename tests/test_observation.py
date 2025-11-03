"""
ObservationHandler: Observation management and execution.
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_agents, create_dso_agent, create_grid_network, create_market_config

from src.coordination.implicit_cooperation import ImplicitCooperation
from src.environment.observation import ObservationHandler
from src.market.matching import MatchingHistory, MatchingResult
from src.market.order import Order, Trade


def test_observation() -> None:
    """Test ObservationHandler class thoroughly."""
    # Use shared utilities to create test components
    market_config = create_market_config(min_price=0.05, max_price=0.50)
    grid = create_grid_network()
    dso_agent = create_dso_agent(agent_id="DSO_obs", grid_network=grid)
    agents = create_agents(num_agents=3, seed=42)

    # Assign agents to grid
    grid.assign_agents_to_graph(agents)

    # Create observation handler
    max_steps = 24
    obs_handler = ObservationHandler(max_steps, agents, dso_agent, market_config, grid)
    print(f"✓ Observation handler created for {len(agents)} agents")

    # Test observation space creation
    print(f"✓ Observation spaces created:")
    for agent in agents:
        obs_space = obs_handler.observation_space[agent.id]
        print(f"  - {agent.id}: {obs_space}")

    # Test initial observation reset
    print(f"--- STEP 1. Testing Initial Observation Reset ---")
    initial_obs = obs_handler.reset_observation_space()

    for agent_id, obs in initial_obs.items():
        print(f"✓ {agent_id} initial observation:")
        print(f"  - Shape: {obs.shape}")
        print(f"  - Current step: {obs[0]}")
        print(f"  - Time of day: {obs[1]}")
        print(f"  - Reputation: {obs[24]}")  # Reputation is at index 24
        print(f"  - Battery SOC: {obs[26]}")  # Battery SOC at index 26

    # Test observation update
    print(f"--- STEP 2. Testing Observation Update ---")

    # Create mock matching result
    trades = [Trade("agent_0", "agent_1", 0.25, 30.0, datetime.datetime.now().timestamp(), 3.0, 0.06),
              Trade("agent_2", "DSO", 0.20, 20.0, datetime.datetime.now().timestamp(), 2.0, 0.04)]

    matching_result = MatchingResult(trades=trades,
                                     unmatched_orders=[],
                                     clearing_price=0.225,
                                     clearing_volume=50.0,
                                     grid_balance=10.0,
                                     dso_buy_volume=20.0,
                                     dso_sell_volume=0.0,
                                     dso_total_volume=20.0,
                                     p2p_volume=30.0,
                                     dso_trade_ratio=0.4,
                                     dso_grid_import=0.0,
                                     dso_buy_price=0.08,
                                     dso_sell_price=0.25,
                                     price_spread=0.17,
                                     local_price_avg=0.225,
                                     local_price_advantage=0.145,
                                     grid_congestion=0.1)

    # Create implicit cooperation handler
    implicit_cooperation = ImplicitCooperation(grid.capacity)

    # Create test orders (all orders including matched ones)
    test_orders = [Order("order_1", "agent_0", 0.25, 30.0, True, 1.0, grid.get_location("agent_0")),
                   Order("order_2", "agent_1", 0.24, 25.0, True, 1.0, grid.get_location("agent_1")),
                   Order("order_3", "agent_2", 0.20, 20.0, False, 1.0, grid.get_location("agent_2"))]

    # Create matching history and add the result
    matching_history = MatchingHistory()
    matching_history.update(matching_result)

    # Update observations
    current_step = 5
    time_of_day = 0.208  # 5/24
    updated_obs = obs_handler.update_observation_space(current_step, time_of_day, test_orders, matching_history, implicit_cooperation)

    for agent_id, obs in updated_obs.items():
        print(f"✓ {agent_id} updated observation:")
        print(f"  - Current step: {obs[0]}")
        print(f"  - Time of day: {obs[1]:.3f}")
        print(f"  - Clearing price: {obs[2]:.3f}")
        print(f"  - Clearing volume: {obs[3]:.1f}")
        print(f"  - Grid balance: {obs[4]:.1f}")
        print(f"  - P2P volume: {obs[8]:.1f}")
        print(f"  - DSO trade ratio: {obs[9]:.3f}")

    print(f"✓ Observation handler testing completed successfully!")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING ObservationHandler TESTS")

    try:
        test_observation()
        print("🎉 ObservationHandler TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
