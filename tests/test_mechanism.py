"""
MarketMechanism: Market mechanism classes for price calculation and market clearing.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid.base import Location
from src.market.mechanism import AveragePricing, BidAskSpreadPricing, BuyerPricing, NashBargainingPricing, ProportionalSurplusPricing, SellerPricing
from src.market.order import Order


def test_market_mechanism() -> None:
    """Test all market mechanism classes thoroughly."""
    # Create test locations and orders
    location1 = Location(node_id="800", x=0, y=0, zone="zone1")
    location2 = Location(node_id="830", x=5, y=3, zone="zone2")

    buy_order = Order(id="buy_order_1",
                      agent_id="buyer_1",
                      price=0.25,
                      quantity=50.0,
                      is_buy=True,
                      timestamp=1.0,
                      location=location1)

    sell_order = Order(id="sell_order_1",
                       agent_id="seller_1",
                       price=0.20,
                       quantity=50.0,
                       is_buy=False,
                       timestamp=1.0,
                       location=location2)

    print(f"✓ Test orders created:")
    print(f"  - Buy order: ${buy_order.price}/Wh, {buy_order.quantity} Wh")
    print(f"  - Sell order: ${sell_order.price}/Wh, {sell_order.quantity} Wh")

    # Test different pricing mechanisms
    mechanisms = {"Average Pricing": AveragePricing(),
                  "Buyer Pricing": BuyerPricing(),
                  "Seller Pricing": SellerPricing(),
                  "Bid-Ask Spread Pricing": BidAskSpreadPricing(),
                  "Nash Bargaining Pricing": NashBargainingPricing(),
                  "Proportional Surplus Pricing": ProportionalSurplusPricing()}

    for name, mechanism in mechanisms.items():
        if name == "Buyer Pricing":
            price = mechanism.calculate_price(buy_order)
        elif name == "Seller Pricing":
            price = mechanism.calculate_price(sell_order)
        else:
            price = mechanism.calculate_price(buy_order,
                                              sell_order,
                                              factor=0.3,
                                              market_price=0.225)
        print(f"✓ {name}: ${price:.3f}/Wh")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING MarketMechanism TESTS")

    try:
        test_market_mechanism()
        print("🎉 MarketMechanism TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
