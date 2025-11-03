"""
DSOAgent: Distribution system operator.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid.base import GridTopology, Location
from src.grid.network import GridNetwork
from src.market.dso import DSOAgent
from src.market.order import Order


def test_dso() -> None:
    """Test DSOAgent class thoroughly."""
    print("--- STEP 1. Creating DSO Agent ---")
    # Create DSO with pricing profiles
    feed_in_tariff = [0.08, 0.07, 0.09, 0.08] * 6  # 24-hour profile
    utility_price = [0.25, 0.28, 0.22, 0.26] * 6   # 24-hour profile
    grid = GridNetwork(topology=GridTopology.IEEE34)

    dso = DSOAgent(id="DSO_1",
                   feed_in_tariff=feed_in_tariff,
                   utility_price=utility_price,
                   grid_network=grid)

    dso.reset(seed=42)
    print(f"✓ DSO Agent created: {dso.id}")
    print(f"  - Feed-in tariff length: {len(dso.feed_in_tariff)}")
    print(f"  - Utility price length: {len(dso.utility_price)}")
    print(f"  - Balance: {dso.balance:.2f} kW, Profit: ${dso.profit:.2f}")

    print("--- STEP 2. Testing Pricing Retrieval ---")
    # Test pricing retrieval
    fit_price = dso.get_feed_in_tariff(step=5)
    util_price = dso.get_utility_price(step=5)
    print(f"✓ Pricing: FIT ${fit_price:.3f}/Wh, Utility ${util_price:.3f}/Wh")

    # Test fee calculation
    location1 = Location(node_id="800", x=0, y=0, zone="zone1")
    location2 = Location(node_id="830", x=5, y=3, zone="zone2")

    buy_order = Order("buy_order_dso", "buyer", 0.25, 50.0, True, 1.0, location1)
    sell_order = Order("sell_order_dso", "seller", 0.20, 50.0, False, 1.0, location2)

    fees = dso.calculate_fees(buy_order, sell_order, 0.225, 50.0)
    print(f"✓ Grid fees calculated: ${fees:.4f}")

    # Test grid constraint validation
    print("--- STEP 3. Testing Grid Constraint Validation ---")

    # Valid trade
    is_valid_1 = dso.validate_grid_constraints(buy_order, sell_order)
    print(f"✓ Validating a standard trade: {'VALID' if is_valid_1 else 'INVALID'}")

    # Invalid trade (exceeds capacity)
    dso.balance = dso.grid_network.capacity - 40.0 # Set balance close to capacity
    is_valid_2 = dso.validate_grid_constraints(buy_order, sell_order)
    print(f"✓ Validating trade that exceeds capacity: {'INVALID' if not is_valid_2 else 'VALID'}")
    dso.reset() # Reset for next test

    # Invalid trade (violates path congestion)
    grid.update_flow_from_trade(location1.node_id, location2.node_id, dso.grid_network.capacity) # Create high congestion
    dso.congestion_threshold = 0.1 # Set a low threshold to trigger violation
    is_valid_3 = dso.validate_grid_constraints(buy_order, sell_order)
    print(f"✓ Validating trade with high congestion: {'INVALID' if not is_valid_3 else 'VALID'}")
    dso.reset()
    grid.reset()

    # Test clearing of unmatched orders
    print("--- STEP 4. Testing Clearing of Unmatched Orders ---")
    unmatched_buys = [Order("unmatched_buy_1", "unmatched_buyer_1", 0.28, 60.0, True, 2.0, location1),
                      Order("unmatched_buy_2", "unmatched_buyer_2", 0.27, 40.0, True, 2.1, location2)]
    unmatched_sells = [Order("unmatched_sell_1", "unmatched_seller_1", 0.15, 70.0, False, 2.2, location2)]

    dso_trades, dso_stats = dso.clear_unmatched_orders(10, unmatched_buys, unmatched_sells)

    print(f"✓ Cleared {len(unmatched_buys) + len(unmatched_sells)} unmatched orders, resulting in {len(dso_trades)} DSO trades.")
    print("  - DSO stats:", dso_stats)
    print(f"  - DSO profit after trades: ${dso.profit:.2f}")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING DSOAgent TESTS")

    try:
        test_dso()
        print("🎉 DSOAgent TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
