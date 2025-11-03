"""
GridMetricsHandler: This module consolidates all grid-related metric calculations for the Local Energy Market
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.stats import linregress

from ..grid.network import GridNetwork
from ..market.matching import MatchingHistory


@dataclass
class GridMetrics:
    """Comprehensive grid performance metrics for the Local Energy Market."""

    # Stability Metrics
    grid_stability_index: float = 0.0
    supply_demand_balance: float = 0.0

    # Congestion Metrics
    avg_congestion_level: float = 0.0

    # Efficiency Metrics
    transmission_loss_ratio: float = 0.0
    grid_utilization_efficiency: float = 0.0

    # Operational Metrics
    transmission_losses: float = 0.0
    load_factor: float = 0.0

    # Network Performance
    capacity_utilization: float = 0.0

    # Temporal Patterns
    stability_trend: float = 0.0
    congestion_trend: float = 0.0
    efficiency_trend: float = 0.0

    # Time Series Data (for enhanced visualization)
    grid_balance_over_time: List[float] = field(default_factory=list)
    grid_congestion_over_time: List[float] = field(default_factory=list)
    grid_stability_over_time: List[float] = field(default_factory=list)
    grid_utilization_over_time: List[float] = field(default_factory=list)


class GridMetricsHandler:
    """Handler for grid performance metrics in the Local Energy Market."""

    def __init__(self,
                 grid_network: GridNetwork,
                 matching_history: MatchingHistory,
                 _window: int = 2) -> None:
        """Initialize the grid metrics handler.

        Args:
            grid_network: Grid network
            matching_history: MatchingHistory object
            _window: Window size for the analysis
        """
        self.grid_network = grid_network
        self.history = matching_history.history
        self._window = _window

    def get_metrics(self) -> GridMetrics:
        """Calculate grid metrics from the stored history.

        Returns:
            GridMetrics object with calculated metrics
        """
        if not self.history:
            return GridMetrics()

        # Calculate stability metrics
        supply_demand_balance = self._calculate_supply_demand_balance()
        grid_stability_index = self._calculate_grid_stability_index()

        # Calculate congestion metrics
        congestion = [r.grid_congestion for r in self.history]
        avg_congestion_level = np.mean(congestion) if congestion else 0.0

        # Calculate efficiency metrics
        transmission_loss_ratio = self._calculate_transmission_loss_ratio()
        grid_utilization_efficiency = self._calculate_grid_utilization_efficiency()

        # Calculate operational metrics
        transmission_losses = self._calculate_transmission_losses()
        load_factor = self._calculate_load_factor()

        # Calculate network performance
        capacity_utilization = self._calculate_capacity_utilization()

        # Calculate temporal trends
        stability_trend = self._calculate_stability_trend()
        congestion_trend = self._calculate_congestion_trend()
        efficiency_trend = self._calculate_efficiency_trend()

        return GridMetrics(grid_stability_index=grid_stability_index,
                           supply_demand_balance=supply_demand_balance,
                           avg_congestion_level=avg_congestion_level,
                           transmission_loss_ratio=transmission_loss_ratio,
                           grid_utilization_efficiency=grid_utilization_efficiency,
                           transmission_losses=transmission_losses,
                           load_factor=load_factor,
                           capacity_utilization=capacity_utilization,
                           stability_trend=stability_trend,
                           congestion_trend=congestion_trend,
                           efficiency_trend=efficiency_trend)

    def _calculate_supply_demand_balance(self) -> float:
        """Calculate supply-demand balance quality.

        Returns:
            Supply-demand balance quality (0-1)
        """
        if not self.history:
            return 0.0

        # Balance based on grid balance deviations
        grid_balances = [abs(r.grid_balance) for r in self.history]
        avg_imbalance = np.mean(grid_balances)

        # Normalize by grid capacity
        normalized_imbalance = avg_imbalance / self.grid_network.capacity
        return max(0.0, 1.0 - normalized_imbalance)

    def _calculate_grid_stability_index(self) -> float:
        """Calculate overall grid stability index.

        Returns:
            Grid stability index (0-1)
        """
        if not self.history:
            return 0.0

        # Balance component
        balance_component = self._calculate_supply_demand_balance()

        # Congestion component
        congestion = [r.grid_congestion for r in self.history]
        avg_congestion = np.mean(congestion) if congestion else 0.0
        congestion_component = 1.0 - avg_congestion

        # Price stability component
        prices = [r.clearing_price for r in self.history if r.clearing_price > 0]
        price_volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.0
        price_component = max(0.0, 1.0 - price_volatility)

        # Combine components
        return np.mean([balance_component, congestion_component, price_component])

    def _calculate_transmission_loss_ratio(self) -> float:
        """Calculate transmission loss ratio.

        Returns:
            Transmission loss ratio (0-1)
        """
        if not self.history:
            return 0.0

        total_losses = 0.0
        total_flow = 0.0

        for result in self.history:
            for trade in result.trades:
                total_losses += trade.transmission_loss
                total_flow += trade.quantity

        return total_losses / total_flow if total_flow > 0 else 0.0

    def _calculate_grid_utilization_efficiency(self) -> float:
        """Calculate grid utilization efficiency.

        Returns:
            Grid utilization efficiency (0-1)
        """
        if not self.history:
            return 0.0

        # Efficiency based on successful local trading vs total volume
        total_p2p = sum(r.p2p_volume for r in self.history)
        total_volume = sum(r.p2p_volume + r.dso_total_volume for r in self.history)

        return total_p2p / total_volume if total_volume > 0 else 0.0

    def _calculate_transmission_losses(self) -> float:
        """Calculate total grid losses.

        Returns:
            Total grid losses (0-1)
        """
        total_losses = 0.0
        for result in self.history:
            for trade in result.trades:
                total_losses += trade.transmission_loss
        return total_losses

    def _calculate_load_factor(self) -> float:
        """Calculate load factor (average/peak).

        Returns:
            Load factor (0-1)
        """
        if not self.history:
            return 0.0

        volumes = [r.clearing_volume for r in self.history]
        if not volumes:
            return 0.0

        return np.mean(volumes) / max(volumes) if max(volumes) > 0 else 0.0

    def _calculate_capacity_utilization(self) -> float:
        """Calculate capacity utilization.

        Returns:
            Capacity utilization (0-1)
        """
        if not self.history:
            return 0.0

        # Utilization based on total flow vs capacity
        total_flow = sum(self.grid_network.edge_flows.values())
        total_capacity = sum(self.grid_network.edge_capacities.values())

        return total_flow / total_capacity if total_capacity > 0 else 0.0

    def _calculate_stability_trend(self) -> float:
        """Calculate stability improvement trend.

        Returns:
            Stability improvement trend (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Calculate stability scores over time
        stability_scores = []
        for result in self.history:
            # Calculate balance score for this specific time step
            grid_balance = abs(result.grid_balance)
            normalized_imbalance = grid_balance / self.grid_network.capacity
            balance_score = max(0.0, 1.0 - normalized_imbalance)
            stability_scores.append(balance_score)

        return self._calculate_trend(stability_scores)

    def _calculate_congestion_trend(self) -> float:
        """Calculate congestion evolution trend.

        Returns:
            Congestion evolution trend (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Negative trend = improving (decreasing congestion)
        congestion = [r.grid_congestion for r in self.history]
        return -self._calculate_trend(congestion)

    def _calculate_efficiency_trend(self) -> float:
        """Calculate efficiency improvement trend.

        Returns:
            Efficiency improvement trend (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        efficiency_scores = []
        for result in self.history:
            # Calculate efficiency score for this specific time step
            total_volume = result.p2p_volume + result.dso_total_volume
            if total_volume > 0:
                eff_score = result.p2p_volume / total_volume
            else:
                eff_score = 0.0
            efficiency_scores.append(eff_score)

        return self._calculate_trend(efficiency_scores)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of a time series.

        Returns:
            Trend (0-1)
        """
        if len(values) < self._window:
            return 0.0

        x = np.arange(len(values))
        try:
            slope, _, r_value, _, _ = linregress(x, values)
            return slope * r_value
        except Exception:
            return 0.0
