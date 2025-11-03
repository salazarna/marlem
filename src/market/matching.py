"""
Order matching implementation for the Local Energy Market.

This module handles the matching of buy and sell orders in a decentralized manner,
considering grid constraints and market rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..grid.network import GridNetwork
from .dso import DSOAgent
from .mechanism import ClearingMechanism
from .order import Order, Trade
from .validation import Validator


@dataclass
class MarketConfig:
    """Market configuration parameters.

    This class defines the configuration parameters for the local energy market (LEM).
    It includes settings for market rules, pricing mechanisms, grid constraints,
    DSO interaction, and preference-based matching.

    Preference-based matching allows agents to express preferences for specific trading partners,
    enabling the study of emergent trading relationships and decentralized coordination
    in multi-agent systems with limited information sharing.
    """

    min_price: float = 0.0  # Minimum allowed price ($/Wh)
    max_price: float = 1000.0  #  Maximum allowed price($/Wh)
    min_quantity: float = 0.0  # Minimum allowed quantity (Wh)
    max_quantity: float = 1000.0  # Maximum allowed quantity (Wh)
    price_mechanism: ClearingMechanism = ClearingMechanism.AVERAGE  # Type of pricing mechanism to use
    blockchain_difficulty: int = 2  # Difficulty of the blockchain
    visualize_blockchain: bool = False  # Enable/disable blockchain visualization
    enable_partner_preference: bool = False  # Use RL-based (vs rule-based) partner selection
    _threshold: float = 1e-6  # Minimum quantity threshold for matching orders

    def __post_init__(self) -> None:
        """Validate the market configuration parameters."""
        self._check_init()

    def _check_init(self) -> None:
        """Validate the market configuration parameters."""
        if self.max_price <= self.min_price:
            raise ValueError(f"Maximum price must be greater than minimum price, got <max_price = {self.max_price}> and <min_price = {self.min_price}>.")

        if self.min_quantity < 0:
            raise ValueError(f"Minimum quantity must be positive, got <min_quantity = {self.min_quantity}>.")

        if self.max_quantity <= self.min_quantity:
            raise ValueError(f"Maximum quantity must be greater than minimum quantity, got <max_quantity = {self.max_quantity}> and <min_quantity = {self.min_quantity}>.")

        if self.blockchain_difficulty < 1:
            raise ValueError(f"Blockchain difficulty must be at least 1, got <blockchain_difficulty = {self.blockchain_difficulty}>.")

        if self.price_mechanism not in ClearingMechanism:
            raise ValueError(f"Unknown pricing mechanism: <price_mechanism = {self.price_mechanism}>. Valid options are: {list(ClearingMechanism)}.")

        if self._threshold < 0:
            raise ValueError(f"Threshold must be positive, got <_threshold = {self._threshold}>.")

@dataclass
class MatchingResult:
    """Result of order matching process."""
    # Matching result
    trades: List[Trade]
    unmatched_orders: List[Order]
    clearing_price: float
    clearing_volume: float

    # Market metrics
    dso_buy_volume: float
    dso_sell_volume: float
    dso_total_volume: float
    p2p_volume: float
    dso_trade_ratio: float
    dso_grid_import: float
    dso_buy_price: float
    dso_sell_price: float
    price_spread: float
    local_price_avg: float
    local_price_advantage: float

    # Grid metrics
    grid_balance: float
    grid_congestion: float

    def __post_init__(self) -> None:
        """Convert all numpy numeric types to standard float."""
        for k, v in self.__annotations__.items():
            if v is float:
                setattr(self, k, float(getattr(self, k)))


class MatchingHistory:
    """History of order matching process."""

    def __init__(self):
        """Initialize the matching history."""
        self.history: List[MatchingResult] = []

    def reset(self):
        """Reset the matching history."""
        self.history = []

    def update(self, matching_result: MatchingResult):
        """Update the matching history.

        Args:
            matching_result: Matching result to add to the history
        """
        self.history.append(matching_result)

    def get_matching_result(self, step: int) -> MatchingResult:
        """Get the matching result at a specific step.

        Args:
            step: Step to get the matching result for

        Returns:
            Matching result at the specified step
        """
        return self.history[step]


class OrderMatcher:
    """Matches buy and sell orders while considering grid constraints."""

    def __init__(self,
                 num_agents: int,
                 config: MarketConfig,
                 grid_network: GridNetwork,
                 dso_agent: DSOAgent) -> None:
        """Initialize the order matcher.

        Args:
            num_agents: Number of agents in the market
            config: Market configuration parameters
            grid_network: GridNetwork object representing the network topology
            dso_agent: DSOAgent object representing the DSO agent
        """
        # Initialize configurations
        self.num_agents = num_agents
        self.config = config
        self.grid_network = grid_network
        self.dso = dso_agent

        # Initialize the decentralized validator
        self.validator = Validator(self.config.blockchain_difficulty)

        # Initialize market state
        self.matching_history: MatchingHistory = MatchingHistory()
        self.grid_capacity: float = config.max_quantity * self.num_agents
        self.trades: List[Trade] = []

        # Check initialization
        self._check_init()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the order matcher."""
        if seed is not None:
            np.random.seed(seed)

        # Reset the DSO
        self.dso.reset(seed)

        # Reset the decentralized validator
        self.validator.reset()

        # Reset market state
        self.matching_history.reset()
        self.trades = []

    def _check_init(self) -> None:
        """Validate the market configuration parameters.

        This method checks for consistency of configuration parameters, especially when
        using time-varying price profiles.

        Raises:
            ValueError: If min_price is greater than or equal to max_price
            ValueError: If min_quantity is non-positive
            ValueError: If max_quantity is less than or equal to min_quantity
            ValueError: If num_agents is non-positive
            ValueError: If grid_network is None
            ValueError: If dso is None
        """
        # Validate num_agents
        if self.num_agents <= 0:
            raise ValueError(f"Number of agents must be positive, got {self.num_agents}")

        # Validate grid_network
        if self.grid_network is None:
            raise ValueError("Grid network cannot be None")

        # Validate DSO
        if self.dso is None:
            raise ValueError("DSO agent cannot be None")

        # Validate min/max price relationship
        if self.config.min_price >= self.config.max_price:
            raise ValueError(f"Minimum price must be less than maximum price, got <min_price = {self.config.min_price}> and <max_price = {self.config.max_price}>.")

        # Validate quantity bounds
        if self.config.min_quantity < 0:
            raise ValueError(f"Minimum quantity must be positive, got <min_quantity = {self.config.min_quantity}>.")

        if self.config.max_quantity <= self.config.min_quantity:
            raise ValueError(f"Maximum quantity must be greater than minimum quantity, got <max_quantity = {self.config.max_quantity}> and <min_quantity = {self.config.min_quantity}>.")

        # Check if the pricing mechanism is valid
        if self.config.price_mechanism not in ClearingMechanism:
            raise ValueError(f"Unknown pricing mechanism: <price_mechanism = {self.config.price_mechanism}>. Valid options are: {list(ClearingMechanism)}.")

        # Validate blockchain difficulty
        if self.config.blockchain_difficulty < 1:
            raise ValueError(f"Blockchain difficulty must be at least 1, got <blockchain_difficulty = {self.config.blockchain_difficulty}>.")

    def match_orders(self,
                     current_step: int,
                     orders: List[Order],
                     reputation_scores: Dict[str, float],
                     grid_balance: float = 0.0) -> MatchingResult:
        """Match orders while respecting grid constraints.

        Args:
            current_step: Current simulation step
            orders: List of orders to match
            reputation_scores: Dictionary mapping agent IDs to reputation scores
            grid_balance: Current grid balance (positive means excess supply)

        Returns:
            MatchingResult containing trades and matching statistics
        """
        # Initialize trades and other tracking variables
        trades = []
        total_volume = 0.0
        total_value = 0.0
        remaining_orders = orders.copy()

        # STAGE 1: Preference-based matching
        market_price = self.matching_history.history[-1].clearing_price if self.matching_history.history else self.dso.get_market_price(current_step)
        preference_trades, remaining_orders = self._match_preferred_partners(current_step,
                                                                             orders,
                                                                             grid_balance,
                                                                             market_price)

        # Update totals with preference trades
        trades = preference_trades.copy()
        total_volume = sum(trade.quantity for trade in preference_trades)
        total_value = sum(trade.quantity * trade.price for trade in preference_trades)

        # Update grid state based on preference trades
        for trade in preference_trades:
            # DSO buying (consuming excess) reduces grid balance
            if trade.buyer_id == self.dso.id:
                grid_balance -= trade.quantity

            # DSO selling (providing shortage) increases grid balance
            elif trade.seller_id == self.dso.id:
                grid_balance += trade.quantity

        # STAGE 2: Price-based matching for remaining orders
        # Separate buy and sell orders
        buy_orders = [o for o in remaining_orders if o.is_buy]
        sell_orders = [o for o in remaining_orders if not o.is_buy]

        # Consider reputation in sorting if provided
        if reputation_scores:
            buy_orders.sort(key=lambda x: (-x.price, -(reputation_scores.get(x.agent_id, 0.5)), x.timestamp)) # Higher reputation and price for buyers
            sell_orders.sort(key=lambda x: (x.price, -(reputation_scores.get(x.agent_id, 0.5)), x.timestamp)) # Lower price and higher reputation for sellers
        else:
            buy_orders.sort(key=lambda x: (-x.price, x.timestamp))  # Highest price first, then earliest
            sell_orders.sort(key=lambda x: (x.price, x.timestamp))  # Lowest price first, then earliest

        # Match orders considering grid constraints
        buy_idx = sell_idx = 0
        while (buy_idx < len(buy_orders)) and (sell_idx < len(sell_orders)) and (buy_orders[buy_idx].price >= sell_orders[sell_idx].price):
            buy_order = buy_orders[buy_idx]
            sell_order = sell_orders[sell_idx]

            # Skip orders with negligible quantities
            if buy_order.quantity < self.config._threshold:
                buy_idx += 1
                continue
            if sell_order.quantity < self.config._threshold:
                sell_idx += 1
                continue

            # Calculate potential match quantity
            proposed_quantity = min(buy_order.quantity, sell_order.quantity)

            # Check grid constraints
            if abs(grid_balance) > self.grid_capacity:
                # Grid is already at capacity, skip this match
                if grid_balance > 0:
                    # Too much supply, try next seller
                    sell_idx += 1
                else:
                    # Too much demand, try next buyer
                    buy_idx += 1
                continue

            # Calculate trade price using selected mechanism
            trade_price = self.config.price_mechanism.get_instance().calculate_price(buy_order,
                                                                                     sell_order,
                                                                                     market_price)

            # Calculate transmission loss for this trade
            distance = buy_order.location.distance_to(sell_order.location, self.grid_network.graph) if buy_order.location and sell_order.location else 0.0
            transmission_loss = self.grid_network.transmission_loss(distance, proposed_quantity)

            # Calculate grid-related fees
            grid_fees = self.dso.calculate_fees(buy_order,
                                                sell_order,
                                                trade_price,
                                                proposed_quantity)

            # Record the trade
            trades.append(Trade(buy_order.agent_id,
                                sell_order.agent_id,
                                trade_price,
                                proposed_quantity,
                                max(buy_order.timestamp, sell_order.timestamp),
                                distance,
                                transmission_loss,
                                grid_fees))

            # Update totals
            total_volume += proposed_quantity
            total_value += proposed_quantity * trade_price

            # DSO buying (consuming excess) reduces grid balance
            if buy_order.agent_id == self.dso.id:
                grid_balance -= proposed_quantity

            # DSO selling (providing shortage) increases grid balance
            elif sell_order.agent_id == self.dso.id:
                grid_balance += proposed_quantity

            # Update order quantities
            buy_order.quantity -= (proposed_quantity - transmission_loss)  # Buyer receives less due to losses
            sell_order.quantity -= proposed_quantity  # Seller provides the full amount

            # Move to next order if current is fully matched
            if buy_order.quantity <= self.config._threshold:
                buy_idx += 1
            if sell_order.quantity <= self.config._threshold:
                sell_idx += 1

        # Collect unmatched orders with remaining quantity
        unmatched_buy_orders = [o for o in buy_orders if o.quantity > 0]
        unmatched_sell_orders = [o for o in sell_orders if o.quantity > 0]

        # STAGE 3: Process unmatched orders with DSO using shared utility
        dso_trades, dso_stats = self.dso.clear_unmatched_orders(current_step,
                                                                unmatched_buy_orders,
                                                                unmatched_sell_orders)

        # Add DSO trades to the trade list
        trades.extend(dso_trades)

        # Update totals with DSO trades
        for trade in dso_trades:
            total_volume += trade.quantity
            total_value += trade.quantity * trade.price

        # Update DSO with local market information
        self.dso.balance = grid_balance
        self.dso.congestion_level = self.grid_network.calculate_congestion_level()
        self.dso_stats = self.calculate_market_statistics(current_step,
                                                          trades,
                                                          dso_stats)

        # Handle unmatched orders
        # DSO is always active, so all orders should be matched
        unmatched_orders = []

        # Calculate clearing price
        clearing_price = total_value / total_volume if total_volume > 0 else self.dso.get_market_price(current_step)

        # Store trades for later reference
        self.trades = trades

        # Add all trades to validator for decentralized validation
        for trade in trades:
            self.validator.add_trade(trade)

        # Force block creation at the end of the market step to ensure all trades in this step are validated as one block
        if self.validator.pending_trades:
            self.validator._create_new_block()

        # Visualize blockchain if enabled
        if self.config.visualize_blockchain:
            self.validator.visualize_blockchain()

        # Matching result
        matching_result = MatchingResult(trades,
                                         unmatched_orders,
                                         clearing_price,
                                         total_volume,
                                         self.dso_stats["dso_buy_volume"],
                                         self.dso_stats["dso_sell_volume"],
                                         self.dso_stats["dso_total_volume"],
                                         self.dso_stats["p2p_volume"],
                                         self.dso_stats["dso_trade_ratio"],
                                         self.dso_stats["dso_grid_import"],
                                         self.dso_stats["dso_buy_price"],
                                         self.dso_stats["dso_sell_price"],
                                         self.dso_stats["price_spread"],
                                         self.dso_stats["local_price_avg"],
                                         self.dso_stats["local_price_advantage"],
                                         self.dso.balance,
                                         self.dso.congestion_level)

        # Update matching history
        self.matching_history.update(matching_result)

        return matching_result

    def get_agent_position(self, agent_id: str) -> float:
        """Calculate the net energy position of an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Net energy position (positive = net seller, negative = net buyer)
        """
        position = 0.0
        agent_trades = [trade for trade in self.trades if trade.buyer_id == agent_id or trade.seller_id == agent_id]

        for trade in agent_trades:
            if trade.seller_id == agent_id:
                position += trade.quantity  # Positive when selling

            if trade.buyer_id == agent_id:
                position -= trade.quantity  # Negative when buying

        return position

    def calculate_market_statistics(self,
                                    step: int,
                                    trades: List[Trade],
                                    dso_stats: Dict) -> Dict:
        """Calculate comprehensive market statistics including DSO and peer-to-peer trading.

        This method analyzes trading activity in both the DSO fallback mechanism and the
        local peer-to-peer market, calculating volumes, prices, and efficiency metrics.

        Args:
            step: Current time step in the simulation
            trades: List of all trades executed in this step
            dso_stats: Dictionary with basic DSO statistics from clear_unmatched_orders method

        Returns:
            Dictionary with comprehensive market statistics
        """
        # Separate DSO and local trades
        local_trades = [t for t in trades if t.buyer_id != self.dso.id and t.seller_id != self.dso.id]

        # Calculate local market metrics
        p2p_volume = sum(t.quantity for t in local_trades)
        local_value = sum(t.price * t.quantity for t in local_trades)

        # Calculate total market volume
        total_volume = dso_stats["dso_total_volume"] + p2p_volume

        # Update the statistics dictionary
        stats = dso_stats.copy()
        stats["p2p_volume"] = p2p_volume
        stats["total_volume"] = total_volume

        # Calculate DSO ratio (proportion of total volume handled by DSO)
        stats["dso_trade_ratio"] = dso_stats["dso_total_volume"] / total_volume if total_volume > 0 else 0.0

        # Get DSO prices for current step
        dso_buy_price = self.dso.get_feed_in_tariff(step)
        dso_sell_price = self.dso.get_utility_price(step)

        # Add DSO price information
        stats["dso_buy_price"] = dso_buy_price
        stats["dso_sell_price"] = dso_sell_price
        stats["price_spread"] = dso_sell_price - dso_buy_price

        # Calculate average local trade price
        if p2p_volume > 0:
            stats["local_price_avg"] = local_value / p2p_volume

            # Calculate local price advantage (positive values indicate local trading is beneficial)
            dso_avg_price = (dso_buy_price + dso_sell_price) / 2
            stats["local_price_advantage"] = dso_avg_price - stats["local_price_avg"]
        else:
            stats["local_price_avg"] = 0.0
            stats["local_price_advantage"] = 0.0

        return stats

    def _create_trade(self,
                      buy_order: Order,
                      sell_order: Order,
                      grid_balance: float,
                      market_price: float) -> Tuple[Trade, float]:
        """Create a trade object from matching buy and sell orders.

        This method handles the creation of a Trade object, calculating the trade price
        and updating the order quantities appropriately. It also calculates grid-related fees
        that are collected by the DSO and affect the final earnings/expenses of the agents.

        Args:
            buy_order: Buy order
            sell_order: Sell order
            grid_balance: Current grid balance (for updating grid state)
            market_price: Market reference price for pricing mechanisms

        Returns:
            Tuple containing (trade, new_grid_balance)
        """
        # Calculate potential match quantity
        proposed_quantity = min(buy_order.quantity, sell_order.quantity)

        # Calculate base trade price using selected mechanism
        base_price = self.config.price_mechanism.get_instance().calculate_price(buy_order,
                                                                                sell_order,
                                                                                market_price)

        # Calculate transmission loss for this trade
        distance = buy_order.location.distance_to(sell_order.location, self.grid_network.graph) if buy_order.location and sell_order.location else 0.0
        transmission_loss = self.grid_network.transmission_loss(distance, proposed_quantity) if self.grid_network else 0.0

        # Calculate grid-related fees
        grid_fees = self.dso.calculate_fees(buy_order,
                                            sell_order,
                                            base_price,
                                            proposed_quantity)

        # Create the trade object with fees information
        trade = Trade(buy_order.agent_id,
                      sell_order.agent_id,
                      base_price,
                      proposed_quantity,
                      max(buy_order.timestamp, sell_order.timestamp),
                      distance,
                      transmission_loss,
                      grid_fees)

        # Update order quantities
        buy_order.quantity -= (proposed_quantity - transmission_loss)  # Buyer receives less due to losses
        sell_order.quantity -= proposed_quantity  # Seller provides the full amount

        # Update grid state
        # DSO buying (consuming excess) reduces grid balance
        if buy_order.agent_id == self.dso.id:
            new_grid_balance = grid_balance - proposed_quantity

        # DSO selling (providing shortage) increases grid balance
        elif sell_order.agent_id == self.dso.id:
            new_grid_balance = grid_balance + proposed_quantity

        # P2P trades don't change grid balance (buyer and seller cancel out)
        else:
            new_grid_balance = grid_balance

        # Update edge congestion in the grid network if available
        if self.grid_network:
            buyer_node = buy_order.location.node_id if buy_order.location else None
            seller_node = sell_order.location.node_id if sell_order.location else None

            if buyer_node and seller_node:
                self.grid_network.update_flow_from_trade(buyer_node, seller_node, proposed_quantity)

        return trade, new_grid_balance

    def _match_preferred_partners(self,
                                  current_step: int,
                                  orders: List[Order],
                                  grid_balance: float,
                                  market_price: float) -> Tuple[List[Trade], List[Order]]:
        """Match orders based on explicit partner preferences.

        This method identifies orders with mutual preferences and creates trades between agents
        that have explicitly selected each other as preferred trading partners. These matches
        are processed first, before the price-based matching occurs.

        The strength of preference matching is controlled by the preference_weight parameter
        in the market configuration. Higher values prioritize preferences over price efficiency.

        Args:
            current_step: Current simulation step
            orders: List of orders to match
            grid_balance: Current grid balance (positive means excess supply)
            market_price: Current market price

        Returns:
            Tuple containing (preference_trades, remaining_orders)
        """
        # Initialize result lists
        preference_trades = []
        matched_order_ids = set()

        # STAGE 1: Handle DSO preferences first
        # Separate orders that prefer DSO from others
        dso_preferred_orders = [o for o in orders if o.partner_id == self.dso.id]
        p2p_preferred_orders = [o for o in orders if o.partner_id != self.dso.id]

        # Process DSO-preferred orders through DSO
        if dso_preferred_orders:
            # Separate buy and sell orders that prefer DSO
            dso_preferred_buys = [o for o in dso_preferred_orders if o.is_buy]
            dso_preferred_sells = [o for o in dso_preferred_orders if not o.is_buy]

            # Process through DSO mechanism
            dso_trades, _ = self.dso.clear_unmatched_orders(current_step,
                                                            dso_preferred_buys,
                                                            dso_preferred_sells)

            # Add DSO trades to preference trades
            preference_trades.extend(dso_trades)

            # Mark these orders as matched
            for order in dso_preferred_orders:
                matched_order_ids.add(order.id)

                # Update grid balance based on the trades
                if order.is_buy:
                    grid_balance -= order.quantity  # Buying reduces grid balance
                else:
                    grid_balance += order.quantity  # Selling increases grid balance

        # STAGE 2: Handle peer-to-peer preferences between DER agents
        # Group remaining orders by agent_id to find complementary preferences
        orders_by_agent_id = {}
        for order in p2p_preferred_orders:
            if order.agent_id not in orders_by_agent_id:
                orders_by_agent_id[order.agent_id] = []
            orders_by_agent_id[order.agent_id].append(order)

        # Create a list of potential matches with their scores
        potential_matches = []

        # Find mutual preferences among DER agents
        for order in p2p_preferred_orders:
            # Skip if order has already been matched, or no preference specified, or preferred partner is not in orders_by_agent_id
            if (order.id in matched_order_ids) or (order.partner_id is None) or (order.partner_id not in orders_by_agent_id):
                continue

            # Find complementary orders (buy-sell pairs)
            partner_orders = orders_by_agent_id[order.partner_id]
            complementary_orders = [o for o in partner_orders if o.is_buy != order.is_buy and o.partner_id == order.agent_id]

            # If no matching preferences found, continue to next order
            if not complementary_orders:
                continue

            # Determine buy and sell orders for each potential match
            for partner_order in complementary_orders:
                if order.is_buy:
                    buy_order, sell_order = order, partner_order
                else:
                    buy_order, sell_order = partner_order, order

                # Check if this is a valid match
                if not self.dso.validate_grid_constraints(buy_order, sell_order):
                    continue

                # Calculate price efficiency component (lower difference between buy and sell prices = higher efficiency)
                price_difference = buy_order.price - sell_order.price
                max_possible_difference = self.config.max_price - self.config.min_price
                price_score = 1.0 - (price_difference / max_possible_difference) if max_possible_difference > 0 else 0.5

                potential_matches.append({"buy_order": buy_order,
                                          "sell_order": sell_order,
                                          "score": price_score,
                                          "grid_balance": grid_balance})

        # Sort potential matches by descending score
        potential_matches.sort(key=lambda x: x["score"], reverse=True)

        # Create trades for the best matches
        for match in potential_matches:
            buy_order = match["buy_order"]
            sell_order = match["sell_order"]

            # Skip if either order is already fully matched
            if (buy_order.id in matched_order_ids) or (sell_order.id in matched_order_ids):
                continue

            # Skip if quantities are too small
            if buy_order.quantity < self.config._threshold or sell_order.quantity < self.config._threshold:
                matched_order_ids.add(buy_order.id)
                matched_order_ids.add(sell_order.id)
                continue

            # Create trade and update grid state
            trade, new_grid_balance = self._create_trade(buy_order,
                                                         sell_order,
                                                         match["grid_balance"],
                                                         market_price)
            preference_trades.append(trade)

            # Update grid state for subsequent matches
            match["grid_balance"] = new_grid_balance

            # Mark both orders as matched if fully satisfied or if remaining quantity is negligible
            if buy_order.quantity <= self.config._threshold:
                matched_order_ids.add(buy_order.id)

            if sell_order.quantity <= self.config._threshold:
                matched_order_ids.add(sell_order.id)

        # Collect remaining orders (those that weren't completely matched)
        remaining_orders = []
        for order in orders:
            if (order.id not in matched_order_ids) and order.quantity > 0:
                remaining_orders.append(order)

        return preference_trades, remaining_orders
