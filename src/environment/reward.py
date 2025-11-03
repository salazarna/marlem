"""
Reward mechanism for the Local Energy Market environment.

This module implements the reward calculation considering:
- Grid balance contribution
- Economic efficiency
- Resource allocation efficiency
- Implicit cooperation
- System stability
- DSO reliance penalties
"""

from typing import Dict, List

import numpy as np

from ..agent.der import DERAgent
from ..grid.network import GridNetwork
from ..market.dso import DSOAgent
from ..market.matching import MatchingResult
from ..market.order import Order, Trade


class RewardHandler:
    """Calculates rewards for market participants in the Local Energy Market.

    This handler implements a comprehensive reward structure that incentivizes:
    1. Successful trading with preference for P2P over DSO
    2. Efficient price formation and profit maximization relative to DSO prices
    3. Grid balancing with agent-specific impact assessment
    4. Implicit cooperation among agents
    5. System stability and resilience
    6. Using DSO only as last resort (dynamic penalty based on grid imbalance)
    """

    def __init__(self,
                 dso: DSOAgent,
                 grid_network: GridNetwork) -> None:
        """Initialize the reward calculator.

        Args:
            dso: DSO agent that manages grid balance
            grid_network: Grid network for capacity information
        """
        self.dso = dso
        self.grid_network = grid_network

        # Check initialization
        self._check_init()

        # Base reward factors (adjusted to emphasize grid balance and cooperation)
        self.f_grid_balance: float = 10.0  # Grid stability importance
        self.f_resource: float = 1.5  # Resource allocation importance
        self.f_trading: float = 5.0  # Trading success
        self.f_stability: float = 2.0  # Market stability importance
        self.f_economic: float = 3.0  # Economic efficiency importance

        # Implicit cooperation reward factors
        self.f_social_welfare: float = 7.0  # Social welfare importance
        self.f_market_liquidity: float = 1.5  # Market liquidity importance
        self.f_avg_bid_ask_spread: float = 1.5  # Average bid-ask spread importance
        self.f_price_volatility: float = 3.0  # Price volatility importance
        self.f_coordination_score: float = 5.0  # Coordination score importance
        self.f_supply_demand_imbalance: float = 10.0  # Supply-demand imbalance importance
        self.f_grid_congestion: float = 9.0  # Maximum grid congestion importance
        self.f_coordination_convergence: float = 8.0  # Coordination convergence importance
        self.f_der_self_consumption: float = 5.0  # DER self-consumption importance
        self.f_flexibility_utilization: float = 6.0  # Flexibility utilization importance

        # Penalty factors
        self.p_dso: float = 5.0  # Penalty factor for DSO trades in reward calculation
        self.p_grid_imbalance: float = 10.0  # Increased penalty for grid imbalance
        self.p_price_deviation: float = 2.0  # Stronger penalty for price manipulation
        self.p_unutilized_resource: float = 1.5  # Higher penalty for waste
        self.p_volatility: float = 2.0  # Penalty for causing market volatility

        # Trade weights
        self._w_local: float = 0.7  # Higher weight for local trades
        self._w_dso: float = 0.3  # Lower weight for DSO trades

        # Price weights
        self._w_efficiency: float = 0.4  # Weight for price efficiency
        self._w_profit: float = 0.6  # Weight for profit relative to DSO

        # Market and grid weights
        self._w_volatility: float = 0.3  # Weight for price volatility
        self._w_imbalance: float = 0.7  # Weight for grid imbalance

        # Contribution weights
        self._w_grid: float = 0.5  # Grid contribution
        self._w_price: float = 0.2  # Price contribution
        self._w_local: float = 0.3  # Local trading contribution

    def _check_init(self) -> None:
        """Check if the reward handler is initialized correctly."""
        if self.dso is None:
            raise ValueError(f"DSO agent must be provided in the config. Current value: <dso = {self.dso}>.")

        if self.grid_network is None:
            raise ValueError(f"Grid network must be provided in the config. Current value: <grid_network = {self.grid_network}>.")

    def calculate_reward(self,
                         agent: DERAgent,
                         order: Order,
                         matching: MatchingResult,
                         kpis: Dict[str, float],
                         min_price: float,
                         max_price: float,
                         is_terminal: bool = False) -> float:
        """Calculate the reward for an agent's market action.

        This method implements a hierarchical reward structure that combines:
        1. Base reward: Agent-specific performance metrics
        2. Cooperation factor: Market-wide health metrics
        3. Contribution factor: How much the agent helped improve the system
        4. DSO penalty: Penalizes excessive reliance on DSO

        Args:
            agent: DERAgent object representing the agent
            order: Agent's submitted order
            matching: MatchingResult object representing the matching result
            min_price: Minimum price in the market
            max_price: Maximum price in the market
            is_terminal: Whether this is the terminal step

        Returns:
            float: The total reward value with adaptive scaling
        """
        # STEP 1. Estimate trades
        # Get agent's trades (those where agent is buyer or seller)
        agent_trades = [trade for trade in matching.trades if (trade.buyer_id == agent.id) or (trade.seller_id == agent.id)]

        # Return penalty if no trades (scaled appropriately)
        if not agent_trades:
            return -5.0

        # Separate agent's trades between local and DSO
        agent_local_trades = [t for t in agent_trades if t.buyer_id != self.dso.id and t.seller_id != self.dso.id]
        agent_dso_trades = [t for t in agent_trades if t.buyer_id == self.dso.id or t.seller_id == self.dso.id]

        # Calculate volumes for trading ratios and DSO penalty
        local_quantity = sum(trade.quantity for trade in agent_local_trades)
        dso_quantity = sum(trade.quantity for trade in agent_dso_trades)
        total_quantity = local_quantity + dso_quantity

        # STEP 2. Calculate base reward (individual performance)
        r_grid_balance = self._calculate_grid_balance_reward(agent, order)
        r_resource = self._calculate_resource_allocation_reward(order, agent_trades)
        r_trading = self._calculate_trading_reward(order, local_quantity, dso_quantity)
        r_stability = self._calculate_stability_reward(order, agent_trades, matching.clearing_price)
        r_economic = self._calculate_economic_reward(order,
                                                     agent_local_trades,
                                                     agent_dso_trades,
                                                     matching.clearing_price,
                                                     min_price,
                                                     max_price,
                                                     matching.dso_buy_price,
                                                     matching.dso_sell_price)

        # Base reward (individual performance)
        base_reward = ((self.f_grid_balance * r_grid_balance) +
                       (self.f_resource * r_resource) +
                       (self.f_trading * r_trading) +
                       (self.f_stability * r_stability) +
                       (self.f_economic * r_economic))

        # STEP 3. Calculate cooperation factor (market-wide performance)
        # Invert metrics where lower is better (penalties become rewards)
        inverted_kpis = kpis.copy()
        for metric in ["supply_demand_imbalance", "grid_congestion", "price_volatility", "avg_bid_ask_spread"]:
            if metric in inverted_kpis:
                inverted_kpis[metric] = 1.0 - float(inverted_kpis[metric])

        # Calculate normalized cooperation factor using available KPIs
        cooperation_factor = ((self.f_social_welfare * inverted_kpis.get("social_welfare", 0.0)) +
                              (self.f_market_liquidity * inverted_kpis.get("market_liquidity", 0.0)) +
                              (self.f_avg_bid_ask_spread * inverted_kpis.get("avg_bid_ask_spread", 0.0)) +
                              (self.f_price_volatility * inverted_kpis.get("price_volatility", 0.0)) +
                              (self.f_supply_demand_imbalance * inverted_kpis.get("supply_demand_imbalance", 0.0)) +
                              (self.f_grid_congestion * inverted_kpis.get("grid_congestion", 0.0)) +
                              (self.f_coordination_score * inverted_kpis.get("coordination_score", 0.0)) +
                              (self.f_coordination_convergence * inverted_kpis.get("coordination_convergence", 0.0)) +
                              (self.f_der_self_consumption * inverted_kpis.get("der_self_consumption", 0.0)) +
                              (self.f_flexibility_utilization * inverted_kpis.get("flexibility_utilization", 0.0)))

        total_weight = sum([self.f_social_welfare, self.f_market_liquidity, self.f_avg_bid_ask_spread, self.f_price_volatility,
                            self.f_supply_demand_imbalance, self.f_grid_congestion, self.f_coordination_score,
                            self.f_coordination_convergence, self.f_der_self_consumption, self.f_flexibility_utilization])

        # Normalize to [0, 1] range
        cooperation_factor = cooperation_factor / total_weight if total_weight > 0 else 0.0

        # STEP 4. Calculate agent's contribution to system improvement
        contribution_factor = self._calculate_contribution_factor(agent,
                                                                  matching,
                                                                  agent_trades,
                                                                  agent_local_trades)

        # STEP 5. Calculate DSO penalty (negative impact)
        dso_ratio = dso_quantity / total_quantity if total_quantity > 0 else 0.0
        dso_penalty = dso_ratio * self.p_dso * abs(self.dso.balance) / max(self.grid_network.capacity, 1.0)

        # STEP 6. Apply terminal penalty for unmet demand (negative impact)
        demand_response_penalty = 0.0

        if is_terminal:
            # Calculate unmet demand at episode end
            unmet_demand = max(0.0, agent.total_demand_required - agent.cumulative_demand_satisfied)

            # Normalize by total demand required to get a ratio between 0 and 1
            if agent.total_demand_required > 0:
                unmet_demand_ratio = unmet_demand / agent.total_demand_required
                demand_response_penalty = self.p_dso * (unmet_demand_ratio ** 2)

        # STEP 7. Combine all components
        # Calculate cooperation boost
        cooperation_boost = (1.0 + cooperation_factor * contribution_factor)
        boosted_base_reward = base_reward * cooperation_boost
        base_magnitude = max(abs(boosted_base_reward), 1.0)

        # Scale penalties proportionally to base reward magnitude
        normalized_dso_penalty = (dso_penalty / max(self.grid_network.capacity, 1.0)) * base_magnitude
        normalized_demand_penalty = demand_response_penalty * base_magnitude / self.p_dso

        # Final reward calculation
        return boosted_base_reward - normalized_dso_penalty - normalized_demand_penalty

    def _calculate_trading_reward(self,
                                  order: Order,
                                  local_quantity: float,
                                  dso_quantity: float) -> float:
        """Calculate reward component for trading success.

        Args:
            order: Agent's submitted order
            local_quantity: Local trading volume
            dso_quantity: DSO trading volume

        Returns:
            Trading success reward component
        """
        local_quantity_ratio = min(local_quantity / order.quantity, 1.0) if order.quantity > 0 else 0.0
        dso_quantity_ratio = min(dso_quantity / order.quantity, 1.0) if order.quantity > 0 else 0.0

        return (self._w_local * local_quantity_ratio) + (self._w_dso * dso_quantity_ratio)

    def _calculate_grid_balance_reward(self,
                                       agent: DERAgent,
                                       order: Order) -> float:
        """Calculate reward component for grid balance contribution.

        Rewards actions that help balance the grid and manage congestion.
        Incorporates agent-specific impact assessment for proper credit assignment.

        Args:
            agent: DERAgent object
            order: Agent's submitted order

        Returns:
            Grid balance reward component
        """

        # Normalize grid state to [-1, 1]
        normalized_grid_state = np.clip(self.dso.balance / (self.grid_network.capacity), -1.0, 1.0)

        # Calculate base contribution to grid balance
        if order.is_buy:
            # Buying helps when grid_state > 0 (excess supply)
            contribution = order.quantity if self.dso.balance > 0 else -order.quantity
        else:
            # Selling helps when grid_state < 0 (excess demand)
            contribution = order.quantity if self.dso.balance < 0 else -order.quantity

        # Normalize contribution
        normalized_contribution = contribution / self.grid_network.capacity

        # Calculate imbalance penalty (negative impact for high imbalance)
        imbalance_penalty = abs(normalized_grid_state) * self.p_grid_imbalance

        # Calculate base balance contribution
        if self.grid_network.capacity is not None and self.grid_network.capacity > 0:
            imbalance_ratio = abs(self.dso.balance) / self.grid_network.capacity
            balance_contribution = 1.0 - min(imbalance_ratio, 1.0)
        else:
            balance_contribution = 1.0 - min(abs(normalized_grid_state), 1.0)

        # Base reward includes contribution minus penalties
        base_reward = normalized_contribution - imbalance_penalty

        # Apply agent-specific impact assessment
        prev_balance = self.dso.balance - agent.balance

        # Calculate impact (positive if improved balance)
        agent_impact = abs(prev_balance) - abs(self.dso.balance)

        # Normalize impact to [-1, 1] range
        agent_impact_normalized = min(max(agent_impact / (self.grid_network.capacity * 0.1), -1.0), 1.0)

        # Apply impact as additive bonus/penalty
        final_reward = base_reward + (agent_impact_normalized * 0.5)

        # Combine with balance contribution
        return balance_contribution + final_reward

    def _calculate_economic_reward(self,
                                   order: Order,
                                   agent_local_trades: List[Trade],
                                   agent_dso_trades: List[Trade],
                                   clearing_price: float,
                                   min_price: float,
                                   max_price: float,
                                   dso_buy_price: float,
                                   dso_sell_price: float) -> float:
        """Calculate reward component for economic efficiency.

        Rewards orders that:
        - Are close to market price (price efficiency)
        - Result in successful trades (execution efficiency)
        - Maintain market stability (volatility control)
        - Prioritize local trades over DSO trades
        - Outperform DSO pricing (profit maximization relative to DSO)

        Args:
            order: Agent's submitted order
            agent_local_trades: List of agent's local trades
            agent_dso_trades: List of agent's DSO trades
            clearing_price: Current market clearing price
            min_price: Minimum price in the market
            max_price: Maximum price in the market
            dso_buy_price: Price at which DSO buys energy (feed-in tariff)
            dso_sell_price: Price at which DSO sells energy (utility price)

        Returns:
            Economic efficiency reward component
        """
        agent_trades = agent_local_trades + agent_dso_trades

        if clearing_price == 0 or not agent_trades:
            return 0.0

        # STEP 1. Local price efficiency
        local_quantity = sum(trade.quantity for trade in agent_local_trades)
        dso_quantity = sum(trade.quantity for trade in agent_dso_trades)
        total_quantity = local_quantity + dso_quantity

        # Calculate average trade price
        avg_trade_price = sum(trade.price * trade.quantity for trade in agent_trades) / total_quantity

        # Calculate price efficiency
        price_deviation_ratio = abs(avg_trade_price - clearing_price) / clearing_price if clearing_price > 0 else 1.0
        price_efficiency = 1.0 - min(price_deviation_ratio * self.p_price_deviation, 1.0)

        # STEP 2. DSO price outperformance
        if total_quantity > 0 and order.quantity > 0:
            local_ratio = local_quantity / total_quantity
            dso_ratio = dso_quantity / total_quantity
            execution_efficiency = ((self._w_local * local_ratio) + (self._w_dso * dso_ratio)) * (total_quantity / order.quantity)
        else:
            execution_efficiency = 0.0

        # Clamp execution efficiency to [0,1] range
        execution_efficiency = min(1.0, max(0.0, execution_efficiency))

        # Calculate profit relative to DSO based on agent's role
        if order.is_buy:
            # Higher profit when agent buys cheaper than DSO selling price
            price_diff = dso_sell_price - avg_trade_price
            dso_max_profit = dso_sell_price - min_price
        else:
            # Higher profit when agent sells higher than DSO buying price
            price_diff = avg_trade_price - dso_buy_price
            dso_max_profit = max_price - dso_buy_price

        # Normalize to [0, 1] range
        if dso_max_profit > 0:
            dso_profit_score = min(max(price_diff / dso_max_profit, 0.0), 1.0)
        else:
            dso_profit_score = 0.0

        return ((self._w_efficiency * price_efficiency) + (self._w_profit * dso_profit_score))

    def _calculate_resource_allocation_reward(self,
                                              order: Order,
                                              trades: List[Trade]) -> float:
        """Calculate reward component for resource allocation efficiency.

        Rewards efficient use of resources:
        - Full execution of orders
        - Minimal unused capacity

        Args:
            order: Agent's submitted order
            trades: List of executed trades

        Returns:
            Resource allocation reward component
        """
        # Calculate execution efficiency
        executed_quantity = sum(trade.quantity for trade in trades if (order.is_buy and trade.buyer_id == order.agent_id) or (not order.is_buy and trade.seller_id == order.agent_id))
        execution_ratio = executed_quantity / order.quantity if order.quantity > 0 else 0

        # Calculate unutilized resource penalty (negative impact)
        unutilized_ratio = 1 - execution_ratio
        resource_penalty = unutilized_ratio * self.p_unutilized_resource

        return execution_ratio - resource_penalty

    def _calculate_stability_reward(self,
                                    order: Order,
                                    trades: List[Trade],
                                    clearing_price: float) -> float:
        """Calculate reward component for market stability.

        Rewards behaviors that contribute to long-term market stability:
        - Price consistency
        - Grid state improvement

        Args:
            order: Agent's submitted order
            trades: List of executed trades
            clearing_price: Current market clearing price

        Returns:
            Market stability reward component
        """
        if not trades:
            return 0.0

        # Calculate price stability contribution
        price_volatility = np.std([t.price for t in trades]) / clearing_price if clearing_price > 0 else 1.0
        price_stability_penalty = price_volatility * self.p_volatility
        price_stability = 1.0 - price_stability_penalty

        # Calculate grid state improvement
        initial_imbalance = abs(self.dso.balance)
        trade_impact = sum(t.quantity for t in trades if t.buyer_id == order.agent_id) - sum(t.quantity for t in trades if t.seller_id == order.agent_id)
        new_imbalance = abs(self.dso.balance + trade_impact)
        grid_improvement = max(0.0, (initial_imbalance - new_imbalance) / initial_imbalance) if initial_imbalance > 0 else 0.0

        return (self._w_volatility * price_stability) + (self._w_imbalance * grid_improvement)

    def _calculate_contribution_factor(self,
                                       agent: DERAgent,
                                       matching: MatchingResult,
                                       agent_trades: List[Trade],
                                       agent_local_trades: List[Trade]) -> float:
        """Calculate how much an agent contributed to system improvement.

        This method evaluates an agent's contribution to overall market health by analyzing:
        1. Grid balance contribution - Did the agent help reduce grid imbalance?
        2. Price stability contribution - Did the agent's trades stabilize prices?
        3. Local trading contribution - Did the agent prioritize local trades over DSO?

        Returns:
            float: A value between 0 and 1, where:
            - 0 means the agent worked against system goals
            - 0.5 means neutral contribution
            - 1 means the agent significantly helped system goals
        """
        NEUTRAL = 0.5 # If no trades or by default

        if not agent_trades:
            return NEUTRAL

        # STEP 1. Grid balance contribution
        grid_contribution = NEUTRAL

        if matching.grid_balance != 0:
            # Calculate agent's net contribution (positive = selling, negative = buying)
            net_contribution = sum(t.quantity if t.seller_id == agent.id else -t.quantity for t in agent_trades)

            # If grid has excess (positive balance) and agent buys, or grid has shortage and agent sells, that's helpful
            if (matching.grid_balance > 0 and net_contribution < 0) or (matching.grid_balance < 0 and net_contribution > 0):
                impact_ratio = min(1.0, abs(net_contribution) / abs(matching.grid_balance))
                grid_contribution = 0.5 + (0.5 * impact_ratio)  # [0.5-1.0] range for helpful actions
            else:
                impact_ratio = min(1.0, abs(net_contribution) / abs(matching.grid_balance))
                grid_contribution = 0.5 - (0.5 * impact_ratio)  # [0.0-0.5] range for unhelpful actions

        # STEP 2. Price stability contribution
        price_contribution = NEUTRAL

        if matching.clearing_price > 0:
            # Calculate average price deviation from market price
            avg_price_deviation = np.mean([abs(t.price - matching.clearing_price) / matching.clearing_price for t in agent_trades])

            # Lower deviation = higher contribution ([0-1] range)
            price_contribution = max(0.0, min(1.0, 1.0 - avg_price_deviation * 2))

        # STEP 3. Local trading contribution
        local_contribution = len(agent_local_trades) / len(agent_trades) if agent_trades else 0 # [0-1] range

        return (self._w_grid * grid_contribution) + (self._w_price * price_contribution) + (self._w_local * local_contribution)
