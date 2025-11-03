"""
OrderMatcher: Order matching and market clearing.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_agents, create_dso_agent, create_grid_network, create_market_config

from src.market.matching import OrderMatcher
from src.market.order import Order


def test_matching() -> None:
    """Test OrderMatcher class thoroughly."""
    # Use shared utilities to create test components
    market_config = create_market_config(min_price=0.05,
                                         max_price=0.50,
                                         visualize_blockchain=True,
                                         enable_partner_preference=False)
    grid = create_grid_network()
    agents = create_agents(num_agents=3,
                           node_ids=["800", "830", "854"])
    grid.assign_agents_to_graph(agents)
    dso_agent = create_dso_agent(agent_id="DSO_matcher", grid_network=grid)

    # Create order matcher
    matcher = OrderMatcher(len(agents),
                           market_config,
                           grid,
                           dso_agent)
    print(f"✓ Order matcher created for {len(agents)} agents")

    # Create test orders
    orders = [Order("order_1", "agent_0", 0.25, 30.0, True, 1.0, grid.get_location("agent_0")),  # Buy order
              Order("order_2", "agent_1", 0.23, 20.0, True, 1.0, grid.get_location("agent_1")),  # Buy order
              Order("order_3", "agent_2", 0.20, 25.0, False, 1.0, grid.get_location("agent_2"))]  # Sell order

    print(f"✓ Created {len(orders)} test orders")

    # Match orders
    reputation_scores = {"agent_0": 0.8, "agent_1": 0.7, "agent_2": 0.9}
    result = matcher.match_orders(current_step=1,
                                  orders=orders,
                                  reputation_scores=reputation_scores,
                                  grid_balance=0.0)

    print(f"✓ Matching completed:")
    print(f"  - Trades executed: {len(result.trades)}")
    print(f"  - Clearing price: ${result.clearing_price:.3f}/Wh")
    print(f"  - Clearing volume: {result.clearing_volume:.2f} Wh")
    print(f"  - Grid balance: {result.grid_balance:.2f} kW")

    # Display trades
    for i, trade in enumerate(result.trades):
        print(f"  - Trade {i+1}: {trade.buyer_id} ← {trade.seller_id}, "
              f"{trade.quantity:.2f} Wh @ ${trade.price:.3f}/Wh")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING OrderMatcher TESTS")

    try:
        test_matching()
        print("🎉 OrderMatcher TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
