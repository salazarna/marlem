"""
Distribution System Operator (DSO) agent implementation.

This module implements the Distribution System Operator (DSO) agent for the Local Energy Market.
The DSO is responsible for maintaining grid balance, processing unmatched orders, and
collecting grid fees.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..grid.network import GridNetwork
from .order import Order, Trade


class DSOAgent:
    """
    Distribution System Operator (DSO) agent for the Local Energy Market.

    The DSO is responsible for:
    1. Grid balance management
    2. Fallback market mechanism for unmatched orders
    3. Grid fee collection (congestion, transmission, imbalance)
    4. Market services (grid constraint validation, dynamic pricing)

    This class centralizes all grid-related operations.
    """

    def __init__(self,
                 id: str = "DSO",
                 feed_in_tariff: List[float] = [],
                 utility_price: List[float] = [],
                 grid_network: Optional[GridNetwork] = None) -> None:
        """
        Initialize the DSO agent.

        Args:
            id: DSO agent identifier
            feed_in_tariff: Price DSO pays for excess generation ($/kWh)
            utility_price: Price DSO charges for additional demand ($/kWh)
            grid_network: Grid network for congestion and constraint management
        """
        # Initialization
        self.id = id
        self.feed_in_tariff = feed_in_tariff  # Price DSO pays for excess generation ($/kWh)
        self.utility_price = utility_price  # Price DSO charges for additional demand ($/kWh)
        self.grid_network = grid_network  # Grid network for operations

        # Grid state tracking
        self.balance: float = 0.0
        self.congestion_level: float = 0.0

        # Financial tracking
        self.profit: float = 0.0
        self.fees: float = 0.0

        # Market state tracking
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0
        self.total_volume: float = 0.0
        self.grid_import: float = 0.0

        # Intervention thresholds
        self.congestion_threshold: float = 0.8  # Threshold for congestion intervention
        self.voltage_threshold: float = 0.05  # Threshold for voltage drop
        self.instability_threshold: float = 0.2  # Threshold for grid stability

        # Grid fee factors
        self.congestion_factor: float = 0.5  # Factor for congestion fees
        self.distance_factor: float = 0.35  # Factor for distance-based fees
        self.zone_factor: float = 0.15  # Factor for cross-zone fees
        self.voltage_factor: float = 0.15  # Factor for voltage drop fees
        self.imbalance_factor: float = 0.15  # Factor for grid stability fees

        # Check initialization
        self._check_init()

    def _check_init(self) -> None:
        """Validate the DSO configuration parameters.

        This method checks for consistency of configuration parameters, especially when
        using time-varying price profiles.

        Raises:
            ValueError: If feed_in_tariff is non-positive
            ValueError: If utility_price is non-positive
            ValueError: If both feed_in_tariff and utility_price are time-varying and have different lengths
            ValueError: If feed_in_tariff is greater than utility_price
        """
        # Allow empty lists for profile management - they will be set by ProfileHandler
        if len(self.feed_in_tariff) == 0:
            raise ValueError("The attribute <feed_in_tariff> cannot be empty.")

        if len(self.utility_price) == 0:
            raise ValueError("The attribute <utility_price> cannot be empty.")

        # Check feed_in_tariff and utility_price have the same length
        if len(self.feed_in_tariff) != len(self.utility_price):
            raise ValueError(f"When both <feed_in_tariff> and <utility_price> are time-varying, they must have the same length (feed_in_tariff: {len(self.feed_in_tariff)}, utility_price: {len(self.utility_price)}).")

        for i, (fit, utility) in enumerate(zip(self.feed_in_tariff, self.utility_price)):
            if fit < 0:
                raise ValueError(f"The <feed_in_tariff[{i}]> ({fit}) must be non-negative.")

            if utility < 0:
                raise ValueError(f"The <utility_price[{i}]> ({utility}) must be non-negative.")

            if fit > utility:
                raise ValueError(f"The <feed_in_tariff[{i}]> ({fit}) must be less than or equal to <utility_price[{i}]> ({utility}).")

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the DSO agent state.
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset grid state
        self.balance = 0.0
        self.congestion_level = 0.0

        # Financial tracking
        self.profit = 0.0
        self.fees = 0.0

        # Market state tracking
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.total_volume = 0.0
        self.grid_import = 0.0

    def get_feed_in_tariff(self, step: int) -> float:
        """Get the feed-in tariff for a specific time step.

        This method handles both fixed and time-varying feed-in tariff values.
        For time-varying values, it will return the appropriate value for the given time step.

        Args:
            step: Current time step in the simulation

        Returns:
            Feed-in tariff for the specified time step

        Raises:
            IndexError: If using time-varying tariff and step is out of range
        """
        try:
            return self.feed_in_tariff[step]
        except IndexError:
            raise IndexError(f"Time step {step} is out of range for <feed_in_tariff> with length {len(self.feed_in_tariff)}.")

    def get_utility_price(self, step: int) -> float:
        """Get the utility price for a specific time step.

        This method handles both fixed and time-varying utility price values.
        For time-varying values, it will return the appropriate value for the given time step.

        Args:
            step: Current time step in the simulation

        Returns:
            Utility price for the specified time step

        Raises:
            IndexError: If using time-varying price and step is out of range
        """
        try:
            return self.utility_price[step]
        except IndexError:
            raise IndexError(f"Time step {step} is out of range for <utility_price> with length {len(self.utility_price)}.")

    def get_market_price(self, step: int) -> float:
        """Get the market price for a specific time step.

        This method handles both fixed and time-varying market price values.
        For time-varying values, it will return the appropriate value for the given time step.

        Args:
            step: Current time step in the simulation

        Returns:
            Market price for the specified time step

        Raises:
            IndexError: If using time-varying market price and step is out of range
        """
        return (self.get_feed_in_tariff(step) + self.get_utility_price(step)) / 2

    def clear_unmatched_orders(self,
                               step: int,
                               unmatched_buy_orders: List[Order],
                               unmatched_sell_orders: List[Order]) -> Tuple[List[Trade], Dict]:
        """Process unmatched orders through the DSO fallback mechanism.

        This function handles unmatched orders by creating trades with the DSO
        using the feed-in tariff for selling excess generation and the utility price
        for buying additional energy.

        Args:
            step: Current time step in the simulation
            unmatched_buy_orders: Buy orders that couldn't be matched in the market
            unmatched_sell_orders: Sell orders that couldn't be matched in the market

        Returns:
            Tuple of (dso_trades, dso_statistics)
        """
        # Initialize DSO statistics
        dso_buy_volume = 0.0
        dso_sell_volume = 0.0
        dso_trades = []

        # DSO buys excess generation at feed-in tariff
        for sell_order in unmatched_sell_orders:
            if sell_order.quantity <= 0:
                continue

            # Create a DSO purchase trade
            dso_trades.append(Trade(self.id,
                                    sell_order.agent_id,
                                    self.get_feed_in_tariff(step),
                                    sell_order.quantity,
                                    sell_order.timestamp,
                                    0.0,  # DSO is assumed to be at zero distance
                                    0.0,   # No transmission loss with DSO
                                    0.0))  # No grid fees for DSO trades

            # Update totals
            dso_buy_volume += sell_order.quantity

            # Mark as matched
            sell_order.quantity = 0

        # DSO sells to meet unmet demand at utility price
        for buy_order in unmatched_buy_orders:
            if buy_order.quantity <= 0:
                continue

            # Create a DSO sale trade
            dso_trades.append(Trade(buy_order.agent_id,
                                    self.id,
                                    self.get_utility_price(step),
                                    buy_order.quantity,
                                    buy_order.timestamp,
                                    0.0,  # DSO is assumed to be at zero distance
                                    0.0,  # No transmission loss with DSO
                                    0.0))  # No grid fees for DSO trades

            # Update totals
            dso_sell_volume += buy_order.quantity

            # Mark as matched
            buy_order.quantity = 0

        # Update DSO statistics
        self.buy_volume += dso_buy_volume
        self.sell_volume += dso_sell_volume
        self.total_volume += dso_buy_volume + dso_sell_volume
        self.grid_import += dso_sell_volume - dso_buy_volume

        # Calculate DSO profit from these trades
        dso_profit = (dso_sell_volume * self.get_utility_price(step)) - (dso_buy_volume * self.get_feed_in_tariff(step))
        self.profit += dso_profit

        # Create statistics dictionary
        dso_stats = {"dso_buy_volume": dso_buy_volume,
                     "dso_sell_volume": dso_sell_volume,
                     "dso_total_volume": dso_buy_volume + dso_sell_volume,
                     "dso_grid_import": dso_sell_volume - dso_buy_volume,
                     "dso_profit": dso_profit}

        return dso_trades, dso_stats

    def calculate_fees(self,
                       buy_order: Order,
                       sell_order: Order,
                       trade_price: float,
                       quantity: float) -> float:
        """Calculate and collect grid-related fees based on grid conditions.

        Args:
            buy_order: Buy order (for single trade fee calculation)
            sell_order: Sell order (for single trade fee calculation)
            trade_price: Base price of the trade ($/kWh) (for single trade fee calculation)
            quantity: Quantity of energy being traded (kWh) (for single trade fee calculation)
            grid_network: GridNetwork object containing grid state information

        Returns:
            Total fees for the trade
        """
        # Initialize fees
        congestion_fee = 0.0
        transmission_fee = 0.0
        imbalance_fee = 0.0
        voltage_fee = 0.0
        thermal_fee = 0.0
        zone_bonus = 0.0

        # Get buyer and seller node IDs if available
        buyer_node = buy_order.location.node_id if buy_order.location else None
        seller_node = sell_order.location.node_id if sell_order.location else None

        # Calculate path-specific congestion if possible, otherwise use global congestion
        if buyer_node and seller_node and self.grid_network:
            try:
                congestion_level = self.grid_network.get_path_congestion(buyer_node, seller_node)
            except Exception:
                congestion_level = self.grid_network.calculate_congestion_level()
        else:
            # No location information available, use global congestion
            congestion_level = self.grid_network.calculate_congestion_level() if self.grid_network else 0.0

        # Calculate transmission distance and fee
        distance = buy_order.location.distance_to(sell_order.location, self.grid_network.graph) if buy_order.location and sell_order.location and self.grid_network else 0.0

        # Check if buyer and seller are in the same zone and apply bonus
        if buy_order.location and sell_order.location and buy_order.location.zone == sell_order.location.zone:
            # Apply a bonus for same-zone trades (negative fee = bonus)
            zone_bonus = -quantity * trade_price * self.zone_factor  # Negative because it's a bonus

        # Grid congestion fee based on path congestion or global congestion
        if congestion_level > self.congestion_threshold:
            congestion_fee = self.congestion_factor * (congestion_level - self.congestion_threshold) * quantity * trade_price

        # Voltage drop calculation and fee
        voltage_drop = distance * self.voltage_factor
        if voltage_drop > self.voltage_threshold:
            voltage_fee = min(0.2, voltage_drop) * quantity * trade_price

        # Thermal limit fee
        if congestion_level > self.instability_threshold:
            thermal_factor = (congestion_level - self.instability_threshold) / (1.0 - self.instability_threshold)
            thermal_fee = min(0.3, thermal_factor) * quantity * trade_price

        # Calculate imbalance fee based on how the trade affects grid balance
        balance_impact = self._calculate_balance_impact(buy_order.agent_id, sell_order.agent_id, quantity)

        if balance_impact < 0:  # Trade worsens grid balance
            imbalance_fee = abs(balance_impact) * quantity * trade_price * self.imbalance_factor

        # Total all fees
        total_fee = congestion_fee + transmission_fee + imbalance_fee + voltage_fee + thermal_fee + zone_bonus
        self.fees += total_fee

        return total_fee

    def validate_grid_constraints(self,
                                  buy_order: Order,
                                  sell_order: Order) -> bool:
        """Validate if a potential trade satisfies grid constraints.

        Args:
            buy_order: Buy order
            sell_order: Sell order
            grid_network: GridNetwork object containing grid state information

        Returns:
            Boolean indicating if the trade is valid from a grid perspective
        """
        # Price compatibility check
        if buy_order.price < sell_order.price:
            return False

        # Calculate potential match quantity
        proposed_quantity = min(buy_order.quantity, sell_order.quantity)

        if proposed_quantity <= 0:
            return False

        # Check if trade would exceed grid capacity (using DSO's balance)
        if self.grid_network and abs(self.balance + proposed_quantity) > self.grid_network.capacity:
            return False

        # Get buyer and seller node IDs if available
        buyer_node = buy_order.location.node_id if buy_order.location else None
        seller_node = sell_order.location.node_id if sell_order.location else None

        # Skip path-specific checks if location information is missing
        if not buyer_node or not seller_node or not self.grid_network:
            return True

        # Check path congestion
        try:
            path_congestion = self.grid_network.get_path_congestion(buyer_node, seller_node)
        except Exception:
            # Fallback to global congestion if path calculation fails
            path_congestion = self.grid_network.calculate_congestion_level()

        if path_congestion > self.congestion_threshold:
            return False

        return True

    def _calculate_balance_impact(self,
                                  buyer_id: str,
                                  seller_id: str,
                                  quantity: float) -> float:
        """Calculate the impact of a trade on grid balance.

        Positive values mean the trade helps balance the grid.
        Negative values mean the trade worsens grid imbalance.

        Args:
            buyer_id: ID of the buyer agent
            seller_id: ID of the seller agent
            quantity: Quantity of energy being traded (kWh)

        Returns:
            Balance impact score (-1 to 1 scale)
        """
        # If grid has excess supply (positive balance)
        if self.balance > 0:
            # Any consumption helps balance
            if buyer_id != self.id and seller_id != self.id:
                # P2P trade doesn't directly affect grid balance
                return 0.0
            elif buyer_id == self.id:
                # DSO buying energy (consuming excess) helps balance
                impact = min(quantity / abs(self.balance), 1.0)
                return impact
            else:
                # DSO selling energy (adding to excess) worsens balance
                impact = -min(quantity / abs(self.balance), 1.0)
                return impact

        # If grid has excess demand (negative balance)
        elif self.balance < 0:
            # Any generation helps balance
            if buyer_id != self.id and seller_id != self.id:
                # P2P trade doesn't directly affect grid balance
                return 0.0
            elif seller_id == self.id:
                # DSO selling energy (providing shortage) helps balance
                impact = min(quantity / abs(self.balance), 1.0)
                return impact
            else:
                # DSO buying energy (adding to shortage) worsens balance
                impact = -min(quantity / abs(self.balance), 1.0)
                return impact

        # Grid is perfectly balanced
        return 0.0
