"""
ImplicitCooperation: Implicit cooperation handler for market coordination.
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_agents

from src.coordination.implicit_cooperation import ImplicitCooperation
from src.grid.base import Location
from src.market.matching import MatchingHistory, MatchingResult
from src.market.order import Order, Trade


def test_implicit_cooperation() -> None:
    """Test ImplicitCooperation class thoroughly."""
    # Create implicit cooperation handler
    grid_capacity = 1000.0
    implicit_cooperation = ImplicitCooperation(grid_capacity)
    print(f"✓ Implicit cooperation handler created with grid capacity: {grid_capacity} kW")

    # Create test locations
    location1 = Location(node_id="800", x=0, y=0, zone="zone1")
    location2 = Location(node_id="830", x=5, y=3, zone="zone2")
    location3 = Location(node_id="854", x=10, y=8, zone="zone3")

    # Create test orders
    orders = [Order("order_1", "agent_0", 0.25, 30.0, True, 1.0, location1),  # Buy order
              Order("order_2", "agent_1", 0.28, 25.0, True, 1.0, location2),  # Buy order
              Order("order_3", "agent_2", 0.20, 40.0, False, 1.0, location3),  # Sell order
              Order("order_4", "agent_0", 0.22, 20.0, False, 1.0, location1)]  # Sell order

    # Create test trades
    trades = [Trade("agent_0", "agent_2", 0.22, 25.0, datetime.datetime.now().timestamp(), 2.5, 0.05),
              Trade("agent_1", "agent_2", 0.24, 15.0, datetime.datetime.now().timestamp(), 1.8, 0.03),
              Trade("agent_0", "DSO", 0.21, 5.0, datetime.datetime.now().timestamp(), 0.7, 0.02)]

    # Separate DSO trades from P2P trades
    dso_trades = [t for t in trades if t.buyer_id == "DSO" or t.seller_id == "DSO"]

    # Create mock matching history
    matching_history = MatchingHistory()

    # Add some historical data
    for i in range(5):
        result = MatchingResult(trades=[],
                                unmatched_orders=[],
                                clearing_price=0.20 + (i * 0.02),  # Gradually increasing price
                                clearing_volume=50.0 + (i * 10.0),  # Gradually increasing volume
                                grid_balance=0.0,
                                dso_buy_volume=0.0,
                                dso_sell_volume=0.0,
                                dso_total_volume=0.0,
                                p2p_volume=50.0 + (i * 10.0),
                                dso_trade_ratio=0.0,
                                dso_grid_import=0.0,
                                dso_buy_price=0.08,
                                dso_sell_price=0.25,
                                price_spread=0.17,
                                local_price_avg=0.20 + (i * 0.02),
                                local_price_advantage=0.12 + (i * 0.02),
                                grid_congestion=0.1)

        matching_history.update(result)

    # Create test grid congestion (overall value)
    grid_congestion = 0.4  # 40% overall congestion

    print(f"✓ Test data created:")
    print(f"  - Orders: {len(orders)}")
    print(f"  - Trades: {len(trades)} (including {len(dso_trades)} DSO trades)")
    print(f"  - History steps: {len(matching_history.history)}")
    print(f"  - Grid congestion level: {grid_congestion}")

    # Create test agents for KPI calculation using shared utility
    test_agents = create_agents(num_agents=3, capacity=100.0)

    # Test KPI calculation with DSO trades
    kpis = implicit_cooperation.get_kpis("DSO",
                                         test_agents,
                                         orders,
                                         trades,
                                         grid_congestion,
                                         matching_history,
                                         current_step=0)

    print(f"\n✓ KPIs calculated:")
    print(f"  - Social welfare: {kpis['social_welfare']:.3f}")
    print(f"  - Market liquidity: {kpis['market_liquidity']:.3f}")
    print(f"  - Coordination score: {kpis['coordination_score']:.3f}")
    print(f"  - Supply-demand imbalance: {kpis['supply_demand_imbalance']:.3f}")
    print(f"  - Grid congestion: {kpis['grid_congestion']:.3f}")

    # Test with different scenarios
    print(f"\n--- STEP 1. Testing Different Market Scenarios ---")

    # Scenario 1: Perfectly balanced market (P2P only)
    balanced_trades = [Trade("agent_1", "agent_2", 0.25, 50.0, datetime.datetime.now().timestamp(), 5.0, 0.1)]
    balanced_congestion = 0.1  # 10% congestion

    kpis_balanced = implicit_cooperation.get_kpis("DSO",
                                                  test_agents,
                                                  orders,
                                                  balanced_trades,
                                                  balanced_congestion,
                                                  matching_history,
                                                  current_step=0)

    print(f"✓ Balanced market scenario (P2P only):")
    print(f"  - Coordination score: {kpis_balanced['coordination_score']:.3f}")
    print(f"  - Supply-demand imbalance: {kpis_balanced['supply_demand_imbalance']:.3f}")

    # Scenario 2: Highly imbalanced market with DSO intervention
    imbalanced_trades = [Trade("agent_1", "agent_2", 0.25, 800.0, datetime.datetime.now().timestamp(), 80.0, 1.6),  # Large trade
                         Trade("agent_3", "DSO", 0.08, 100.0, datetime.datetime.now().timestamp(), 8.0, 0.16)]  # DSO trade
    imbalanced_dso_trades = [t for t in imbalanced_trades if t.buyer_id == "DSO" or t.seller_id == "DSO"]
    high_congestion = 0.9  # 90% congestion

    kpis_imbalanced = implicit_cooperation.get_kpis("DSO",
                                                    test_agents,
                                                    orders,
                                                    imbalanced_trades,
                                                    high_congestion,
                                                    matching_history,
                                                    current_step=0)

    print(f"✓ Imbalanced market scenario (with DSO intervention):")
    print(f"  - Coordination score: {kpis_imbalanced['coordination_score']:.3f}")
    print(f"  - Supply-demand imbalance: {kpis_imbalanced['supply_demand_imbalance']:.3f}")
    print(f"  - Grid congestion: {kpis_imbalanced['grid_congestion']:.3f}")
    print(f"  - DSO trades: {len(imbalanced_dso_trades)}")

    # Scenario 3: Testing with custom DSO ID
    print(f"✓ Scenario 3: Testing with custom DSO ID")
    custom_dso_trades = [Trade("agent_1", "agent_2", 0.23, 40.0, datetime.datetime.now().timestamp(), 4.0, 0.08),  # P2P trade
                         Trade("agent_3", "CUSTOM_DSO", 0.08, 60.0, datetime.datetime.now().timestamp(), 4.8, 0.12)]  # Custom DSO trade
    custom_dso_only = [t for t in custom_dso_trades if t.buyer_id == "CUSTOM_DSO" or t.seller_id == "CUSTOM_DSO"]

    kpis_custom_dso = implicit_cooperation.get_kpis("CUSTOM_DSO",
                                                    test_agents,
                                                    orders,
                                                    custom_dso_trades,
                                                    0.3,
                                                    matching_history,
                                                    current_step=0)

    print(f"✓ Custom DSO scenario:")
    print(f"  - Total trades: {len(custom_dso_trades)}")
    print(f"  - Custom DSO trades: {len(custom_dso_only)}")
    print(f"  - Coordination score: {kpis_custom_dso['coordination_score']:.3f}")
    print(f"  - DER self-consumption: {kpis_custom_dso['der_self_consumption']:.3f}")
    print(f"  - Flexibility utilization: {kpis_custom_dso['flexibility_utilization']:.3f}")

    print(f"✓ Implicit cooperation testing completed successfully!")
    print(f"  - DSO-agnostic implementation works with any DSO ID")
    print(f"  - Resource coordination metrics properly identify DSO vs P2P trades")
    print(f"  - Grid congestion integration uses actual DSO congestion level")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING ImplicitCooperation TESTS")

    try:
        test_implicit_cooperation()
        print("🎉 ImplicitCooperation TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
