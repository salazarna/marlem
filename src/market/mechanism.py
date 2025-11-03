"""
Market mechanism implementation for the Local Energy Market.
"""

from enum import Enum
from typing import Protocol

from .order import Order


class BasePricing(Protocol):
    """Protocol defining the interface for pricing mechanisms.

    All pricing mechanisms must implement this protocol, but can accept
    different parameters based on their specific needs.
    """

    def calculate_price(self, *args, **kwargs) -> float:
        """Calculate the trade price for a matched pair of orders.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Calculated trade price
        """
        ...


class AveragePricing(BasePricing):
    """Calculates trade price as average of buy and sell prices."""

    def calculate_price(self,
                        buy_order: Order,
                        sell_order: Order,
                        *args,
                        **kwargs) -> float:
        """Calculate average of buy and sell prices.

        Args:
            buy_order: Buy order
            sell_order: Sell order

        Returns:
            Average price
        """
        return (buy_order.price + sell_order.price) / 2


class BuyerPricing(BasePricing):
    """Sets trade price to buyer's bid price (i.e., pay-as-bid pricing)."""

    def calculate_price(self,
                        buy_order: Order,
                        *args,
                        **kwargs) -> float:
        """Calculate price based on pay-as-bid principle.

        Args:
            buy_order: Buy order

        Returns:
            Buyer's price
        """
        return buy_order.price


class SellerPricing(BasePricing):
    """Sets trade price to seller's ask price."""

    def calculate_price(self,
                        sell_order: Order,
                        *args,
                        **kwargs) -> float:
        """Use seller's price as trade price.

        Args:
            sell_order: Sell order

        Returns:
            Seller's price
        """
        return sell_order.price


class BidAskSpreadPricing(BasePricing):
    """Implement a pricing mechanism that sets the price at a point within the bid-ask spread."""

    def calculate_price(self,
                        buy_order: Order,
                        sell_order: Order,
                        factor: float = 0.5,
                        *args,
                        **kwargs) -> float:
        """Calculate price based on continuous double auction principles.

        The price is set at a point within the bid-ask spread, determined by an
        improvement factor. This creates a "price improvement" for both parties.

        Args:
            buy_order: Buy order
            sell_order: Sell order
            factor: Factor (0-1) determining where in the bid-ask spread the clearing price is set

        Returns:
            Calculated continuous double auction price
        """
        bid_ask_spread = buy_order.price - sell_order.price

        return sell_order.price + bid_ask_spread * max(0.0, min(factor, 1.0))


class NashBargainingPricing(BasePricing):
    """Implements Nash bargaining solution for surplus division.

    The Nash solution maximizes the product of utility gains over disagreement payoffs,
    typically resulting in equal surplus sharing under symmetric conditions.
    This promotes fair and efficient coordination between trading agents.
    """

    def calculate_price(self,
                        buy_order: Order,
                        sell_order: Order,
                        *args,
                        **kwargs) -> float:
        """Calculate price based on Nash bargaining solution.

        The solution maximizes (buyer_gain - d1) * (seller_gain - d2) where
        d1, d2 are disagreement payoffs. Under symmetric conditions, this
        results in equal surplus sharing.

        Args:
            buy_order: Buy order
            sell_order: Sell order

        Returns:
            Nash bargaining solution price
        """
        # Calculate total surplus from the trade
        bid_ask_spread = buy_order.price - sell_order.price

        if bid_ask_spread <= 0:
            # No positive surplus, use average pricing as fallback
            return AveragePricing().calculate_price(buy_order, sell_order)

        # Under symmetric disagreement payoffs, Nash solution is equal split
        return sell_order.price + bid_ask_spread / 2


class ProportionalSurplusPricing(BasePricing):
    """Implements proportional surplus sharing based on bid distances.

    Allocates surplus proportionally to each agent's distance from a reference price,
    incentivizing truthful bidding and rewarding agents based on their willingness
    to trade.
    """

    def calculate_price(self,
                        buy_order: Order,
                        sell_order: Order,
                        market_price: float,
                        *args,
                        **kwargs) -> float:
        """Calculate price based on proportional surplus sharing.

        Surplus is allocated proportionally to each agent's bid distance from
        the reference price, rewarding agents who bid more aggressively.

        Args:
            buy_order: Buy order
            sell_order: Sell order
            market_price: Reference market price. If None, falls back to average pricing.

        Returns:
            Proportional surplus sharing price
        """
        # Calculate distances from reference price
        buyer_distance = max(0, buy_order.price - market_price)
        seller_distance = max(0, market_price - sell_order.price)
        total_distance = buyer_distance + seller_distance

        # Calculate surplus
        bid_ask_spread = buy_order.price - sell_order.price

        # If no meaningful distances, fall back to average pricing
        if total_distance == 0 or bid_ask_spread <= 0:
            return AveragePricing().calculate_price(buy_order, sell_order)

        # Allocate surplus proportionally
        return sell_order.price + (buyer_distance / total_distance) * bid_ask_spread


class ClearingMechanism(Enum):
    """Different clearing mechanisms."""

    AVERAGE = "average"
    BUYER = "buyer"
    SELLER = "seller"
    BID_ASK_SPREAD = "bid_ask_spread"
    NASH_BARGAINING = "nash_bargaining"
    PROPORTIONAL_SURPLUS = "proportional_surplus"

    def get_instance(self) -> BasePricing:
        """Get the pricing instance for this mechanism."""
        mechanisms = {"average": AveragePricing,
                      "buyer": BuyerPricing,
                      "seller": SellerPricing,
                      "bid_ask_spread": BidAskSpreadPricing,
                      "nash_bargaining": NashBargainingPricing,
                      "proportional_surplus": ProportionalSurplusPricing}

        return mechanisms[self.value]()
