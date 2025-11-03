"""
Comprehensive Market Analytics for Local Energy Markets.

This module provides all market analysis functionality including economic metrics,
market structure analysis, and agent behavior evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.stats import linregress

from ..market.matching import MatchingHistory


@dataclass
class MarketMetrics:
    """Comprehensive market metrics for the Local Energy Market."""

    # Core Economic Indicators
    social_welfare: float = 0.0
    consumer_surplus: float = 0.0
    producer_surplus: float = 0.0

    # Price Metrics
    avg_clearing_price: float = 0.0
    price_volatility: float = 0.0
    price_trend: float = 0.0

    # Market Efficiency
    allocation_efficiency: float = 0.0
    price_discovery_efficiency: float = 0.0

    # Trading Volume Metrics
    total_trading_volume: float = 0.0
    transaction_count: int = 0
    avg_trade_size: float = 0.0

    # Market Structure
    market_concentration: float = 0.0

    # Welfare Distribution
    welfare_distribution_gini: float = 0.0
    price_fairness_index: float = 0.0

    # Market structure
    p2p_trade_ratio: float = 0.0
    dso_dependency_ratio: float = 0.0
    market_liquidity: float = 0.0

    # Temporal trends
    volume_trend: float = 0.0
    efficiency_trend: float = 0.0

    # Agent behavior
    avg_agent_rewards: Dict[str, float] = field(default_factory=dict)
    trading_frequency_by_agent: Dict[str, int] = field(default_factory=dict)


class MarketMetricsHandler:
    """Comprehensive market analyzer for Local Energy Markets.

    This class consolidates all market analysis functionality including:
    - Economic efficiency metrics
    - Market structure analysis
    - Agent behavior evaluation
    - Temporal trend analysis
    """

    def __init__(self,
                 agent_rewards: Dict[str, List[float]],
                 matching_history: MatchingHistory) -> None:
        """Initialize the market analyzer.

        Args:
            matching_history: History of market matching results
        """
        self.agent_rewards = agent_rewards
        self.matching_history = matching_history
        self.trades = [t for r in matching_history.history for t in r.trades]

    def get_metrics(self) -> MarketMetrics:
        """Perform comprehensive market analysis.

        Args:
            agent_rewards: Optional agent reward histories for behavior analysis

        Returns:
            MarketAnalytics with all analytics results
        """
        economic = self.get_economic_metrics()
        market_structure = self.get_market_structure()
        agent_behavior = self.get_agent_behavior()

        # Handle empty data gracefully
        if not economic or not market_structure or not agent_behavior:
            return MarketMetrics()  # Return default MarketMetrics with all zeros

        return MarketMetrics(social_welfare=economic["social_welfare"],
                             consumer_surplus=economic["consumer_surplus"],
                             producer_surplus=economic["producer_surplus"],
                             avg_clearing_price=economic["avg_clearing_price"],
                             price_volatility=economic["price_volatility"],
                             price_trend=economic["price_trend"],
                             allocation_efficiency=economic["allocation_efficiency"],
                             price_discovery_efficiency=economic["price_discovery_efficiency"],
                             total_trading_volume=economic["total_trading_volume"],
                             transaction_count=economic["transaction_count"],
                             avg_trade_size=economic["avg_trade_size"],
                             market_concentration=economic["market_concentration"],
                             welfare_distribution_gini=economic["welfare_distribution_gini"],
                             price_fairness_index=economic["price_fairness_index"],
                             p2p_trade_ratio=market_structure["p2p_trade_ratio"],
                             dso_dependency_ratio=market_structure["dso_dependency_ratio"],
                             market_liquidity=market_structure["market_liquidity"],
                             volume_trend=market_structure["volume_trend"],
                             efficiency_trend=market_structure["efficiency_trend"],
                             avg_agent_rewards=agent_behavior["avg_agent_rewards"],
                             trading_frequency_by_agent=agent_behavior["trading_frequency_by_agent"])

    def get_economic_metrics(self) -> Dict[str, float]:
        """Analyze economic efficiency and welfare metrics.

        Returns:
            EconomicMetrics with comprehensive economic analysis
        """
        if not self.trades:
            return {}

        # Extract trade data
        trade_prices = [t.price for t in self.trades]
        trade_quantities = [t.quantity for t in self.trades]

        # Core economic indicators
        social_welfare = self._calculate_social_welfare(trade_prices, trade_quantities)
        consumer_surplus, producer_surplus = self._calculate_surpluses()

        # Price metrics
        avg_clearing_price = np.mean(trade_prices)
        price_volatility = self._calculate_price_volatility(trade_prices)
        price_trend = self._calculate_price_trend(trade_prices)

        # Market efficiency
        allocation_efficiency = self._calculate_allocation_efficiency()
        price_discovery_efficiency = self._calculate_price_discovery_efficiency()

        # Trading volume metrics
        total_trading_volume = sum(trade_quantities)
        transaction_count = len(self.trades)
        avg_trade_size = total_trading_volume / transaction_count if transaction_count > 0 else 0.0

        # Market structure
        market_concentration = self._calculate_market_concentration()

        # Welfare distribution
        welfare_distribution_gini = self._calculate_welfare_gini()
        price_fairness_index = self._calculate_price_fairness(avg_clearing_price)

        return {"social_welfare": social_welfare,
                "consumer_surplus": consumer_surplus,
                "producer_surplus": producer_surplus,
                "avg_clearing_price": avg_clearing_price,
                "price_volatility": price_volatility,
                "price_trend": price_trend,
                "allocation_efficiency": allocation_efficiency,
                "price_discovery_efficiency": price_discovery_efficiency,
                "total_trading_volume": total_trading_volume,
                "transaction_count": transaction_count,
                "avg_trade_size": avg_trade_size,
                "market_concentration": market_concentration,
                "welfare_distribution_gini": welfare_distribution_gini,
                "price_fairness_index": price_fairness_index}

    def get_market_structure(self) -> Dict[str, float]:
        """Analyze market structure and dynamics.

        Returns:
            MarketMetrics with market structure analysis
        """
        if not self.matching_history.history:
            return {}

        # Extract data from matching results
        clearing_prices = [r.clearing_price for r in self.matching_history.history]
        trading_volumes = [r.clearing_volume for r in self.matching_history.history]
        p2p_volumes = [r.p2p_volume for r in self.matching_history.history]
        dso_volumes = [r.dso_total_volume for r in self.matching_history.history]

        # Market structure metrics
        total_p2p = np.sum(p2p_volumes)
        total_dso = np.sum(dso_volumes)
        total_volume = total_p2p + total_dso

        p2p_trade_ratio = total_p2p / total_volume if total_volume > 0 else 0.0
        dso_dependency_ratio = total_dso / total_volume if total_volume > 0 else 1.0

        # Market liquidity (inverse of price volatility)
        price_volatility = np.std(clearing_prices) / np.mean(clearing_prices) if np.mean(clearing_prices) > 0 else 0.0
        market_liquidity = 1.0 / (1.0 + price_volatility)

        # Calculate trends
        volume_trend = self._calculate_trend(trading_volumes)
        efficiency_ratios = [r.clearing_volume / max(r.clearing_price, 0.01) for r in self.matching_history.history]
        efficiency_trend = self._calculate_trend(efficiency_ratios)

        return {"p2p_trade_ratio": p2p_trade_ratio,
                "dso_dependency_ratio": dso_dependency_ratio,
                "market_liquidity": market_liquidity,
                "volume_trend": volume_trend,
                "efficiency_trend": efficiency_trend}

    def get_agent_behavior(self) -> Dict[str, float]:
        """Analyze agent behavior and coordination effectiveness.

        Args:
            agent_rewards: Dictionary mapping agent IDs to reward histories

        Returns:
            AgentMetrics with agent behavior analysis
        """
        if not self.matching_history.history:
            return {}

        return {"avg_agent_rewards": {agent_id: np.mean(rewards) for agent_id, rewards in self.agent_rewards.items()},
                "trading_frequency_by_agent": self._calculate_trading_frequency()}

    def _calculate_social_welfare(self,
                                  prices: List[float],
                                  quantities: List[float]) -> float:
        """Calculate social welfare as total economic value created.

        Args:
            prices: List of prices
            quantities: List of quantities

        Returns:
            Social welfare as total economic value created
        """
        if len(prices) == 0 or len(quantities) == 0:
            return 0.0
        return np.sum([p * q for p, q in zip(prices, quantities)])

    def _calculate_surpluses(self) -> tuple[float, float]:
        """Calculate consumer and producer surplus.

        Returns:
            Consumer and producer surplus
        """
        if not self.trades:
            return 0.0, 0.0

        # Simplified surplus calculation
        total_value = sum(t.price * t.quantity for t in self.trades)
        baseline_value = np.mean([t.price for t in self.trades]) * sum(t.quantity for t in self.trades)

        # Split surplus evenly between consumers and producers (approximation)
        total_surplus = max(0.0, total_value - baseline_value)
        return total_surplus / 2, total_surplus / 2

    def _calculate_price_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility as coefficient of variation.

        Args:
            prices: List of prices

        Returns:
            Price volatility as coefficient of variation
        """
        if len(prices) < 2:
            return 0.0
        mean_price = np.mean(prices)
        return np.std(prices) / mean_price if mean_price > 0 else 0.0

    def _calculate_price_trend(self, prices: List[float]) -> float:
        """Calculate price trend using linear regression.

        Args:
            prices: List of prices

        Returns:
            Price trend using linear regression
        """
        if len(prices) < 2:
            return 0.0
        try:
            slope, _, r_value, _, _ = linregress(np.arange(len(prices)), prices)
            return slope * r_value
        except Exception:
            return 0.0

    def _calculate_allocation_efficiency(self) -> float:
        """Calculate allocation efficiency based on trade distribution.

        Returns:
            Allocation efficiency based on trade distribution
        """
        if not self.trades:
            return 0.0

        quantities = [t.quantity for t in self.trades]
        total_quantity = sum(quantities)

        # Perfect allocation would have uniform distribution
        expected_share = 1.0 / len(self.trades)
        actual_shares = [q / total_quantity for q in quantities]

        # Calculate allocation efficiency
        deviations = [abs(share - expected_share) for share in actual_shares]
        allocation_deviation = sum(deviations) / len(deviations)

        return max(0.0, 1.0 - allocation_deviation * len(self.trades))

    def _calculate_price_discovery_efficiency(self) -> float:
        """Calculate price discovery efficiency.

        Returns:
            Price discovery efficiency
        """
        if not self.trades:
            return 0.0

        prices = [t.price for t in self.trades]
        if len(prices) < 2:
            return 1.0

        # Lower variance in prices indicates better price discovery
        price_variance = np.var(prices)
        mean_price = np.mean(prices)

        normalized_variance = price_variance / (mean_price ** 2) if mean_price > 0 else 0.0
        return max(0.0, 1.0 - min(normalized_variance, 1.0))

    def _calculate_market_concentration(self) -> float:
        """Calculate market concentration using Herfindahl index.

        Returns:
            Market concentration using Herfindahl index
        """
        if not self.trades:
            return 0.0

        # Calculate volume by trader
        trader_volumes = {}
        total_volume = 0.0

        for trade in self.trades:
            trader_volumes[trade.buyer_id] = trader_volumes.get(trade.buyer_id, 0.0) + trade.quantity
            trader_volumes[trade.seller_id] = trader_volumes.get(trade.seller_id, 0.0) + trade.quantity
            total_volume += 2 * trade.quantity

        if total_volume == 0:
            return 0.0

        # Calculate Herfindahl index
        market_shares = [volume / total_volume for volume in trader_volumes.values()]
        return sum(share ** 2 for share in market_shares)

    def _calculate_welfare_gini(self) -> float:
        """Calculate Gini coefficient for welfare distribution.

        Returns:
            Gini coefficient for welfare distribution
        """
        if not self.trades:
            return 0.0

        # Calculate welfare by agent
        agent_welfare = {}
        for trade in self.trades:
            welfare_per_trade = trade.price * trade.quantity
            agent_welfare[trade.buyer_id] = agent_welfare.get(trade.buyer_id, 0.0) + welfare_per_trade
            agent_welfare[trade.seller_id] = agent_welfare.get(trade.seller_id, 0.0) + welfare_per_trade

        if not agent_welfare:
            return 0.0

        # Calculate Gini coefficient
        welfare_values = sorted(agent_welfare.values())
        n = len(welfare_values)
        cumulative_welfare = np.cumsum(welfare_values)

        if cumulative_welfare[-1] == 0:
            return 0.0

        return (n + 1 - 2 * np.sum(cumulative_welfare) / cumulative_welfare[-1]) / n

    def _calculate_price_fairness(self, clearing_price: float) -> float:
        """Calculate price fairness index.

        Args:
            clearing_price: Clearing price

        Returns:
            Price fairness index
        """
        if not self.trades or clearing_price <= 0:
            return 0.0

        # Measure how close individual trade prices are to the clearing price
        price_deviations = [abs(t.price - clearing_price) / clearing_price for t in self.trades]
        avg_deviation = np.mean(price_deviations)

        return max(0.0, 1.0 - min(avg_deviation, 1.0))

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of a time series.

        Args:
            values: List of values

        Returns:
            Trend of a time series
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        try:
            slope, _, r_value, _, _ = linregress(x, values)
            return slope * r_value
        except Exception:
            return 0.0

    def _calculate_trading_frequency(self) -> Dict[str, int]:
        """Calculate trading frequency by agent.

        Returns:
            Trading frequency by agent
        """
        frequency = {}
        for result in self.matching_history.history:
            for trade in result.trades:
                frequency[trade.buyer_id] = frequency.get(trade.buyer_id, 0) + 1
                frequency[trade.seller_id] = frequency.get(trade.seller_id, 0) + 1
        return frequency
