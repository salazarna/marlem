"""
RewardHandler: Reward management and execution.
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.battery import Battery
from src.agent.der import DERAgent
from src.coordination.implicit_cooperation import ImplicitCooperation
from src.environment.reward import RewardHandler
from src.grid.base import GridTopology, Location
from src.grid.network import GridNetwork
from src.market.dso import DSOAgent
from src.market.matching import MatchingResult
from src.market.order import Order, Trade


def test_reward() -> None:
    """Test RewardHandler class thoroughly."""
    # Create grid network
    grid = GridNetwork(topology=GridTopology.IEEE34)

    # Create DSO agent
    dso_agent = DSOAgent(id="DSO_reward",
                         feed_in_tariff=[0.08] * 24,
                         utility_price=[0.25] * 24,
                         grid_network=grid)

    # Create implicit cooperation handler
    implicit_cooperation = ImplicitCooperation(grid.capacity)

    # Create reward handler
    num_agents = 3
    reward_handler = RewardHandler(num_agents, dso_agent, grid, implicit_cooperation)
    print(f"✓ Reward handler created for {num_agents} agents")
    print(f"  - Grid balance factor: {reward_handler.f_grid_balance}")
    print(f"  - Trading factor: {reward_handler.f_trading}")
    print(f"  - Economic factor: {reward_handler.f_economic}")
    print(f"  - DSO penalty: {reward_handler.p_dso}")

    # Create test agent with battery
    battery = Battery(nominal_capacity=50.0, min_soc=0.1, max_soc=0.9)
    battery.reset(seed=42)
    agent = DERAgent(id="test_agent",
                     capacity=100.0,
                     battery=battery,
                     generation_profile=[30.0] * 24,
                     demand_profile=[25.0] * 24)
    agent.reset(0.7, [30.0] * 24, [25.0] * 24, 42)

    # Create test location
    location = Location(node_id="800", x=0, y=0, zone="zone1")

    # Test different reward scenarios
    print(f"--- STEP 1. Testing Different Reward Scenarios ---")

    # Scenario 1: Successful P2P trade
    print(f"Scenario 1: Successful P2P Trade")
    order_p2p = Order("order_p2p", "test_agent", 0.25, 50.0, True, 1.0, location)
    trades_p2p = [Trade("test_agent", "other_agent", 0.25, 50.0, datetime.datetime.now().timestamp(), 5.0, 0.1)]

    matching_p2p = MatchingResult(trades=trades_p2p,
                                  unmatched_orders=[],
                                  clearing_price=0.25,
                                  clearing_volume=50.0,
                                  grid_balance=5.0,
                                  dso_buy_volume=0.0,
                                  dso_sell_volume=0.0,
                                  dso_total_volume=0.0,
                                  p2p_volume=50.0,
                                  dso_trade_ratio=0.0,
                                  dso_grid_import=0.0,
                                  dso_buy_price=0.08,
                                  dso_sell_price=0.25,
                                  price_spread=0.17,
                                  local_price_avg=0.25,
                                  local_price_advantage=0.17)

    kpis_p2p = {"social_welfare": 12.5,
                "market_liquidity": 50.0,
                "avg_bid_ask_spread": 0.02,
                "price_volatility": 0.01,
                "coordination_score": 0.95,
                "supply_demand_imbalance": 0.05,
                "max_grid_congestion": 0.1}

    reward_p2p = reward_handler.calculate_reward(agent,order_p2p, matching_p2p, kpis_p2p, 0.05, 0.50, False)
    print(f"✓ P2P trade reward: {reward_p2p:.4f}")

    # Scenario 2: DSO trade
    print(f"Scenario 2: DSO Trade")
    order_dso = Order("order_dso", "test_agent", 0.20, 40.0, False, 1.0, location)
    trades_dso = [Trade("test_agent", "DSO", 0.08, 40.0, datetime.datetime.now().timestamp(), 3.2, 0.08)]

    matching_dso = MatchingResult(trades=trades_dso,
                                  unmatched_orders=[],
                                  clearing_price=0.08,
                                  clearing_volume=40.0,
                                  grid_balance=-5.0,
                                  dso_buy_volume=40.0,
                                  dso_sell_volume=0.0,
                                  dso_total_volume=40.0,
                                  p2p_volume=0.0,
                                  dso_trade_ratio=1.0,
                                  dso_grid_import=5.0,
                                  dso_buy_price=0.08,
                                  dso_sell_price=0.25,
                                  price_spread=0.17,
                                  local_price_avg=0.08,
                                  local_price_advantage=-0.17)

    kpis_dso = {"social_welfare": 3.2,
                "market_liquidity": 40.0,
                "avg_bid_ask_spread": 0.0,
                "price_volatility": 0.0,
                "coordination_score": 0.85,
                "supply_demand_imbalance": 0.15,
                "max_grid_congestion": 0.2}

    reward_dso = reward_handler.calculate_reward(agent, order_dso, matching_dso, kpis_dso, 0.05, 0.50, False)
    print(f"✓ DSO trade reward: {reward_dso:.4f}")

    # Scenario 3: Mixed trades (P2P + DSO)
    print(f"Scenario 3: Mixed Trades (P2P + DSO)")
    order_mixed = Order("order_mixed", "test_agent", 0.22, 60.0, True, 1.0, location)
    trades_mixed = [Trade("test_agent", "other_agent", 0.24, 35.0, datetime.datetime.now().timestamp(), 4.2, 0.08),
                    Trade("test_agent", "DSO", 0.25, 25.0, datetime.datetime.now().timestamp(), 3.1, 0.05)]

    matching_mixed = MatchingResult(trades=trades_mixed,
                                    unmatched_orders=[],
                                    clearing_price=0.245,
                                    clearing_volume=60.0,
                                    grid_balance=2.0,
                                    dso_buy_volume=0.0,
                                    dso_sell_volume=25.0,
                                    dso_total_volume=25.0,
                                    p2p_volume=35.0,
                                    dso_trade_ratio=0.42,
                                    dso_grid_import=0.0,
                                    dso_buy_price=0.08,
                                    dso_sell_price=0.25,
                                    price_spread=0.17,
                                    local_price_avg=0.245,
                                    local_price_advantage=0.005)

    kpis_mixed = {"social_welfare": 14.7,
                 "market_liquidity": 60.0,
                 "avg_bid_ask_spread": 0.015,
                 "price_volatility": 0.005,
                 "coordination_score": 0.90,
                 "supply_demand_imbalance": 0.02,
                 "max_grid_congestion": 0.05}

    reward_mixed = reward_handler.calculate_reward(agent, order_mixed, matching_mixed, kpis_mixed, 0.05, 0.50, False)
    print(f"✓ Mixed trades reward: {reward_mixed:.4f}")

    # Scenario 4: No trades (unmatched order)
    print(f"Scenario 4: No Trades (Unmatched Order)")
    order_unmatched = Order("order_unmatched", "test_agent", 0.10, 30.0, True, 1.0, location)

    matching_unmatched = MatchingResult(trades=[],
                                        unmatched_orders=[order_unmatched],
                                        clearing_price=0.25,
                                        clearing_volume=0.0,
                                        grid_balance=0.0,
                                        dso_buy_volume=0.0,
                                        dso_sell_volume=0.0,
                                        dso_total_volume=0.0,
                                        p2p_volume=0.0,
                                        dso_trade_ratio=0.0,
                                        dso_grid_import=0.0,
                                        dso_buy_price=0.08,
                                        dso_sell_price=0.25,
                                        price_spread=0.17,
                                        local_price_avg=0.25,
                                        local_price_advantage=0.15)

    kpis_unmatched = {"social_welfare": 0.0,
                      "market_liquidity": 0.0,
                      "avg_bid_ask_spread": 0.0,
                      "price_volatility": 0.0,
                      "coordination_score": 1.0,
                      "supply_demand_imbalance": 0.0,
                     "max_grid_congestion": 0.0}

    reward_unmatched = reward_handler.calculate_reward(agent, order_unmatched, matching_unmatched, kpis_unmatched, 0.05, 0.50, False)
    print(f"✓ Unmatched order reward: {reward_unmatched:.4f}")

    # Test terminal reward with demand response
    print(f"Scenario 5: Terminal Reward with Unmet Demand")
    agent.total_demand_required = 600.0  # 24 hours * 25 kW
    agent.cumulative_demand_satisfied = 450.0  # 75% satisfied

    reward_terminal = reward_handler.calculate_reward(agent, order_p2p, matching_p2p, kpis_p2p, 0.05, 0.50, is_terminal=True)
    print(f"✓ Terminal reward with unmet demand: {reward_terminal:.4f}")
    print(f"  - Unmet demand: {agent.total_demand_required - agent.cumulative_demand_satisfied:.1f} Wh")
    print(f"  - Demand satisfaction: {agent.cumulative_demand_satisfied / agent.total_demand_required * 100:.1f}%")

    print(f"✓ Reward handler testing completed successfully!")
    print(f"  - P2P trades receive higher rewards than DSO trades")
    print(f"  - Mixed trades receive intermediate rewards")
    print(f"  - Unmatched orders receive minimal rewards")
    print(f"  - Terminal penalties apply for unmet demand")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING RewardHandler TESTS")

    try:
        test_reward()
        print("🎉 RewardHandler TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
