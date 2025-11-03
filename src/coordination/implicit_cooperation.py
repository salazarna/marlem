"""
Implicit Cooperation Model for Local Energy Markets.

This module implements coordination KPIs for scientific reporting and evaluation
of decentralized multi-agent coordination effectiveness in local energy markets.
"""

from typing import Dict, List, Optional

import numpy as np

from ..agent.der import DERAgent
from ..market.matching import MatchingHistory
from ..market.order import Order, Trade


class ImplicitCooperation:
    """
    Handles implicit coordination KPIs for evaluating market performance.

    This class focuses on measuring coordination effectiveness through:
    - Economic efficiency metrics
    - Grid stability and balance metrics
    - Coordination emergence indicators
    - Resource utilization metrics

    These KPIs are used for analysis of how well decentralized agents
    coordinate without explicit communication.
    """

    def __init__(self, grid_capacity: float) -> None:
        """
        Initialize the implicit cooperation model.

        Args:
            grid_capacity: Maximum grid capacity for normalization
        """
        self.grid_capacity = grid_capacity

        # Check initialization
        self._check_init()

    def _check_init(self) -> None:
        """Check if the implicit cooperation model is initialized correctly."""
        if self.grid_capacity <= 0:
            raise ValueError(f"Grid capacity must be greater than 0. Current value: <grid_capacity = {self.grid_capacity}>.")

    def get_kpis(self,
                 current_step: int,
                 dso_id: str,
                 agents: List[DERAgent],
                 orders: List[Order],
                 trades: List[Trade],
                 grid_congestion: float,
                 matching_history: MatchingHistory) -> Dict[str, float]:
        """
        Calculate all coordination KPIs for scientific reporting.

        Args:
            current_step: Current step
            dso_id: ID of the DSO agent
            agents: List of DER agents (optional, for advanced metrics)
            orders: List of orders in the current step
            trades: List of executed trades in the current step
            grid_congestion: Overall grid congestion level (0-1 scale)
            matching_history: Historical matching results

        Returns:
            Dictionary with all KPIs at the top level
        """
        kpis = {}
        dso_trades = [t for t in trades if t.buyer_id == dso_id or t.seller_id == dso_id]

        # Core coordination metrics (always calculated)
        kpis.update(self._get_economic_efficiency_metrics(orders, trades, matching_history))
        kpis.update(self._get_grid_stability_metrics(grid_congestion, trades))
        kpis.update(self._get_coordination_effectiveness_metrics(trades, matching_history))
        kpis.update(self._get_resource_coordination_metrics(current_step, agents, trades, dso_trades))

        return kpis

    def _get_economic_efficiency_metrics(self,
                                         orders: List[Order],
                                         trades: List[Trade],
                                         matching_history: MatchingHistory) -> Dict[str, float]:
        """
        Calculate economic efficiency metrics showing market performance.

        Returns:
            Dictionary containing economic efficiency KPIs
        """
        # Social welfare (total economic value created)
        social_welfare = sum(t.price * abs(t.quantity) for t in trades)

        # Market liquidity (total trading volume)
        market_liquidity = sum(abs(t.quantity) for t in trades)

        # Price formation efficiency
        bids = [o.price for o in orders if not o.is_buy]
        asks = [o.price for o in orders if o.is_buy]
        avg_bid_ask_spread = (np.mean(asks) - np.mean(bids)) if bids and asks else 0.0

        # Price stability over time
        price_history = [m.clearing_price for m in matching_history.history if m.clearing_price > 0]
        price_volatility = np.std(price_history) if len(price_history) > 1 else 0.0

        return {"social_welfare": social_welfare,
                "market_liquidity": market_liquidity,
                "avg_bid_ask_spread": avg_bid_ask_spread,
                "price_volatility": price_volatility}

    def _get_grid_stability_metrics(self,
                                    grid_congestion: float,
                                    trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate grid stability metrics showing system balance.

        Returns:
            Dictionary containing grid stability KPIs
        """
        # Supply-demand balance (normalized by grid capacity)
        total_imbalance = abs(sum(t.quantity for t in trades)) / (self.grid_capacity or 1.0)
        supply_demand_imbalance = min(1.0, total_imbalance)

        # Grid congestion metrics (overall grid congestion level)
        overall_grid_congestion = max(0.0, min(1.0, grid_congestion))  # Ensure 0-1 range

        return {"supply_demand_imbalance": supply_demand_imbalance,
                "grid_congestion": overall_grid_congestion}

    def _get_coordination_effectiveness_metrics(self,
                                                trades: List[Trade],
                                                matching_history: MatchingHistory) -> Dict[str, float]:
        """
        Calculate coordination effectiveness showing how well agents coordinate.

        Returns:
            Dictionary containing coordination effectiveness KPIs
        """
        # Coordination score based on supply-demand balance
        imbalance = abs(sum(t.quantity for t in trades)) / (self.grid_capacity or 1.0)
        coordination_score = max(0.0, min(1.0, 1.0 - imbalance))

        # Market efficiency trend (coordination convergence)
        coordination_convergence = 0.0
        if len(matching_history.history) > 2:
            recent_volumes = [m.clearing_volume for m in matching_history.history[-3:]]

            if len(recent_volumes) > 1 and np.mean(recent_volumes) > 0:
                volume_stability = 1.0 - min(1.0, np.std(recent_volumes) / np.mean(recent_volumes))
                coordination_convergence = max(0.0, volume_stability)

        return {"coordination_score": coordination_score,
                "coordination_convergence": coordination_convergence}

    def _get_resource_coordination_metrics(self,
                                           current_step: int,
                                           agents: List[DERAgent],
                                           trades: List[Trade],
                                           dso_trades: Optional[List[Trade]] = None) -> Dict[str, float]:
        """
        Calculate resource coordination metrics showing DER utilization.

        Args:
            current_step: Current step
            agents: List of DER agents
            trades: All trades in the current step
            dso_trades: Optional list of DSO trades (for more accurate identification)

        Returns:
            Dictionary containing resource coordination KPIs
        """
        # Get agent IDs for identifying local trades
        agent_ids = {agent.id for agent in agents}

        # Separate local and DSO trades
        if dso_trades is not None:
            # Use provided DSO trades for accurate identification
            dso_trade_ids = {(t.buyer_id, t.seller_id, t.timestamp) for t in dso_trades}
            local_trades = [t for t in trades if (t.buyer_id, t.seller_id, t.timestamp) not in dso_trade_ids]
        else:
            # Fall back to heuristic: trades where both parties are agents are local
            local_trades = [t for t in trades if t.buyer_id in agent_ids and t.seller_id in agent_ids]

        total_volume = sum(t.quantity for t in trades)
        local_volume = sum(t.quantity for t in local_trades)

        der_self_consumption = local_volume / total_volume if total_volume > 0 else 0.0

        # Flexibility utilization: proportion of available flexible energy that is utilized
        # Available flexibility = sum of all agents' potential flexible energy
        total_available_flexibility = 0.0
        for agent in agents:
            try:
                # Get current generation and demand
                current_generation = agent.get_generation(current_step)
                current_demand = agent.get_demand(current_step)

                # Get battery available energy
                available_charge = 0.0
                available_discharge = 0.0
                if agent.battery:
                    available_charge, available_discharge = agent.battery.estimate_available_energy()

                # Calculate sellable flexibility (surplus generation + battery discharge)
                net_generation = current_generation - current_demand
                sellable_flexibility = max(0.0, net_generation + available_discharge)

                # Calculate buyable flexibility (deficit demand + battery charge)
                buyable_flexibility = max(0.0, -net_generation + available_charge)

                # Total flexibility = sum of both directions (agent can participate in either direction)
                agent_flexibility = sellable_flexibility + buyable_flexibility
                total_available_flexibility += agent_flexibility
            except (IndexError, AttributeError):
                # If agent doesn't have profile or step is out of range, skip
                continue

        # Utilized flexibility = energy actually traded locally (P2P trades)
        utilized_flexibility = sum(abs(t.quantity) for t in local_trades) if local_trades else 0.0

        # Flexibility utilization = utilized / available (energy-based metric)
        flexibility_utilization = utilized_flexibility / total_available_flexibility if total_available_flexibility > 0 else 0.0

        return {"der_self_consumption": der_self_consumption,
                "flexibility_utilization": flexibility_utilization}
