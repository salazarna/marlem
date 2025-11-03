"""
DSOMetricsHandler: Consolidates all DSO-related metric calculations for the Local Energy Market.
"""

from dataclasses import dataclass

import numpy as np

from ..market.matching import MatchingHistory, MatchingResult


@dataclass
class DSOMetrics:
    """Comprehensive DSO performance metrics for the Local Energy Market."""

    # Trading Volume Metrics
    dso_buy_volume: float = 0.0
    dso_sell_volume: float = 0.0
    dso_total_volume: float = 0.0
    p2p_volume: float = 0.0

    # Market Share and Dependency
    dso_trade_ratio: float = 0.0
    p2p_trade_ratio: float = 0.0

    # Financial Metrics
    dso_buy_price_avg: float = 0.0
    dso_sell_price_avg: float = 0.0
    price_spread: float = 0.0
    local_price_avg: float = 0.0
    local_price_advantage: float = 0.0

    # Grid Import/Export
    net_grid_import: float = 0.0
    grid_import_ratio: float = 0.0
    self_sufficiency_ratio: float = 0.0

    # Cost and Savings
    dso_profit: float = 0.0
    avoided_dso_cost: float = 0.0
    local_trading_benefit: float = 0.0

    # Performance Indicators
    dso_utilization_efficiency: float = 0.0
    market_balance_quality: float = 0.0
    fallback_effectiveness: float = 0.0


class DSOMetricsHandler:
    """Handler for DSO performance metrics in the Local Energy Market."""

    def __init__(self,
                 grid_capacity: float,
                 matching_history: MatchingHistory,
                 _window: int = 2) -> None:
        """Initialize the DSO metrics handler.

        Args:
            grid_capacity: Grid capacity for normalization purposes
            matching_history: MatchingHistory object
            _window: Window size for the analysis
        """
        self.grid_capacity = grid_capacity
        self.history = matching_history.history
        self._window = _window

    def get_metrics(self) -> DSOMetrics:
        """Calculate aggregated DSO metrics from the stored history.

        Returns:
            DSOMetrics object with aggregated metrics
        """
        if not self.history:
            return DSOMetrics()

        # Aggregate volume metrics
        total_dso_buy = sum(r.dso_buy_volume for r in self.history)
        total_dso_sell = sum(r.dso_sell_volume for r in self.history)
        total_dso_volume = total_dso_buy + total_dso_sell
        total_p2p_volume = sum(r.p2p_volume for r in self.history)
        total_volume = total_dso_volume + total_p2p_volume

        # Market share metrics
        dso_trade_ratio = total_dso_volume / total_volume if total_volume > 0 else 0.0
        p2p_trade_ratio = total_p2p_volume / total_volume if total_volume > 0 else 0.0

        # Average financial metrics
        dso_buy_prices = [r.dso_buy_price for r in self.history if r.dso_buy_price > 0]
        dso_sell_prices = [r.dso_sell_price for r in self.history if r.dso_sell_price > 0]
        local_prices = [r.local_price_avg for r in self.history if r.local_price_avg > 0]

        dso_buy_price_avg = np.mean(dso_buy_prices) if dso_buy_prices else 0.0
        dso_sell_price_avg = np.mean(dso_sell_prices) if dso_sell_prices else 0.0
        price_spread = dso_sell_price_avg - dso_buy_price_avg
        local_price_avg = np.mean(local_prices) if local_prices else 0.0
        local_price_advantage = np.mean([r.local_price_advantage for r in self.history])

        # Grid import/export
        net_grid_import = sum(r.dso_grid_import for r in self.history)
        grid_import_ratio = abs(net_grid_import) / total_volume if total_volume > 0 else 0.0
        self_sufficiency_ratio = total_p2p_volume / total_volume if total_volume > 0 else 0.0

        # Aggregate cost calculations
        dso_profit = sum(self._calculate_dso_profit(r) for r in self.history)
        avoided_dso_cost = sum(self._calculate_avoided_dso_cost(r) for r in self.history)
        local_trading_benefit = sum(self._calculate_local_trading_benefit(r) for r in self.history)

        # Performance indicators (averages)
        dso_utilization_efficiency = np.mean([self._calculate_dso_utilization_efficiency(r) for r in self.history])
        market_balance_quality = np.mean([self._calculate_market_balance_quality(r) for r in self.history])
        fallback_effectiveness = np.mean([self._calculate_fallback_effectiveness(r) for r in self.history])

        return DSOMetrics(dso_buy_volume=total_dso_buy,
                          dso_sell_volume=total_dso_sell,
                          dso_total_volume=total_dso_volume,
                          p2p_volume=total_p2p_volume,
                          dso_trade_ratio=dso_trade_ratio,
                          p2p_trade_ratio=p2p_trade_ratio,
                          dso_buy_price_avg=dso_buy_price_avg,
                          dso_sell_price_avg=dso_sell_price_avg,
                          price_spread=price_spread,
                          local_price_avg=local_price_avg,
                          local_price_advantage=local_price_advantage,
                          net_grid_import=net_grid_import,
                          grid_import_ratio=grid_import_ratio,
                          self_sufficiency_ratio=self_sufficiency_ratio,
                          dso_profit=dso_profit,
                          avoided_dso_cost=avoided_dso_cost,
                          local_trading_benefit=local_trading_benefit,
                          dso_utilization_efficiency=dso_utilization_efficiency,
                          market_balance_quality=market_balance_quality,
                          fallback_effectiveness=fallback_effectiveness)

    def _calculate_dso_profit(self, result: MatchingResult) -> float:
        """Calculate DSO profit from a matching result.

        Args:
            result: MatchingResult object

        Returns:
            DSO profit
        """
        # DSO profit = (sell volume * sell price) - (buy volume * buy price)
        sell_revenue = result.dso_sell_volume * result.dso_sell_price
        buy_cost = result.dso_buy_volume * result.dso_buy_price
        return sell_revenue - buy_cost

    def _calculate_avoided_dso_cost(self, result: MatchingResult) -> float:
        """Calculate cost avoided by trading locally instead of with DSO.

        Args:
            result: MatchingResult object

        Returns:
            Cost avoided by trading locally instead of with DSO
        """
        if result.p2p_volume <= 0 or result.local_price_avg <= 0:
            return 0.0

        # Approximate cost if all P2P volume had been traded with DSO
        dso_avg_price = np.mean([result.dso_buy_price, result.dso_sell_price])
        dso_cost = result.p2p_volume * dso_avg_price
        local_cost = result.p2p_volume * result.local_price_avg

        return max(0.0, dso_cost - local_cost)

    def _calculate_local_trading_benefit(self, result: MatchingResult) -> float:
        """Calculate economic benefit of local trading.

        Args:
            result: MatchingResult object

        Returns:
            Economic benefit of local trading
        """
        return result.local_price_advantage * result.p2p_volume

    def _calculate_dso_utilization_efficiency(self, result: MatchingResult) -> float:
        """Calculate DSO capacity utilization efficiency.

        Args:
            result: MatchingResult object

        Returns:
            DSO capacity utilization efficiency
        """
        total_volume = result.dso_total_volume + result.p2p_volume
        if total_volume <= 0:
            return 1.0  # Perfect efficiency when no trading

        # Efficiency = how much of the total volume DSO had to handle (lower is better for decentralization)
        dso_utilization = result.dso_total_volume / total_volume

        # Convert to efficiency score (higher is better)
        return 1.0 - dso_utilization

    def _calculate_market_balance_quality(self, result: MatchingResult) -> float:
        """Calculate how well the DSO balances the market.

        Args:
            result: MatchingResult object

        Returns:
            How well the DSO balances the market
        """
        # Perfect balance would have grid balance close to 0
        grid_balance_normalized = abs(result.grid_balance) / self.grid_capacity
        return max(0.0, 1.0 - min(grid_balance_normalized, 1.0))

    def _calculate_fallback_effectiveness(self, result: MatchingResult) -> float:
        """Calculate effectiveness of DSO as fallback mechanism.

        Args:
            result: MatchingResult object

        Returns:
            Effectiveness of DSO as fallback mechanism
        """
        # Effectiveness = ability to handle unmatched orders
        total_orders_handled = len(result.trades) + len(result.unmatched_orders)
        if total_orders_handled <= 0:
            return 1.0

        # All orders should be matched when DSO is active (perfect fallback)
        matched_orders = len(result.trades)
        return matched_orders / total_orders_handled
