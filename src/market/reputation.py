"""Reputation system for tracking agent behavior in the Local Energy Market.

This module implements a decentralized reputation system that evaluates agent
reliability, price fairness, and grid contribution based on market outcomes.
"""

from typing import List, Tuple

import numpy as np

from ..agent.der import DERAgent
from .matching import MatchingResult
from .order import Trade


class ReputationHandler:
    """Manages agent reputation in the Local Energy Market.

    This system calculates reputation scores based on market matching results,
    evaluating agents on reliability, price fairness, and grid contribution.
    """

    def __init__(self,
                 reliability_weight: float = 0.3,
                 fairness_weight: float = 0.2,
                 grid_weight: float = 0.5) -> None:
        """Initialize the reputation system.

        Args:
            reliability_weight: Weight for reliability in overall score
            fairness_weight: Weight for price fairness in overall score
            grid_weight: Weight for grid contribution in overall score
        """
        self.reliability_weight = reliability_weight
        self.fairness_weight = fairness_weight
        self.grid_weight = grid_weight
        self._neutral = 0.5

        # Check initialization
        self._check_init()

    def _check_init(self) -> None:
        """Check if the reputation handler is initialized correctly."""
        if self.reliability_weight < 0.0 or self.reliability_weight > 1.0:
            raise ValueError(f"Reliability weight must be between 0.0 and 1.0. Current value: <reliability_weight = {self.reliability_weight}>.")

        if self.fairness_weight < 0.0 or self.fairness_weight > 1.0:
            raise ValueError(f"Fairness weight must be between 0.0 and 1.0. Current value: <fairness_weight = {self.fairness_weight}>.")

        if self.grid_weight < 0.0 or self.grid_weight > 1.0:
            raise ValueError(f"Grid weight must be between 0.0 and 1.0. Current value: <grid_weight = {self.grid_weight}>.")

        if self.reliability_weight + self.fairness_weight + self.grid_weight != 1.0:
            raise ValueError(f"Weights must sum to 1.0. Current value: <reliability_weight = {self.reliability_weight}, fairness_weight = {self.fairness_weight}, grid_weight = {self.grid_weight}>.")

    def reset(self, seed: int = None) -> float:
        """Generate initial reputation score for an agent.

        Args:
            seed: Optional random seed for reproducible reputation initialization

        Returns:
            Initial reputation score between 0.0 and 1.0
        """
        if seed is not None:
            np.random.seed(seed)

        return np.random.uniform(0.0, 1.0)

    def update_reputation(self,
                          agent: DERAgent,
                          matching_result: MatchingResult,
                          time_of_day: float,
                          dso_id: str = "DSO") -> float:
        """Update reputation metrics for an agent based on matching results.

        Args:
            agent: DERAgent object
            matching_result: Result of the order matching process
            time_of_day: Current time of day (0-1)
            dso_id: ID of the DSO agent for identifying DSO trades

        Returns:
            Updated reputation score
        """
        new_reputation = self._calculate_reputation(agent, matching_result, dso_id)
        return (time_of_day * agent.reputation) + ((1.0 - time_of_day) * new_reputation)

    def get_agent_ranking(self, agents: List[DERAgent]) -> List[Tuple[str, float]]:
        """Get ranked list of agents by overall reputation.

        Args:
            agents: List of DERAgent objects to rank

        Returns:
            List of (agent_id, score) tuples, sorted by score (highest first)
        """
        rankings = [(agent.id, agent.reputation) for agent in agents]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def _calculate_reputation(self,
                              agent: DERAgent,
                              matching_result: MatchingResult,
                              dso_id: str) -> float:
        """Calculate reputation metrics for an agent based on matching results.

        This method calculates reputation considering three key components:
        1. Reliability - Did the agent's orders get matched successfully?
        2. Price fairness - How close were the agent's prices to the market clearing price?
        3. Grid contribution - Did the agent's trades help balance the grid?

        Args:
            agent: DERAgent object
            matching_result: Result of the order matching process
            dso_id: ID of the DSO agent for identifying DSO trades

        Returns:
            Calculated reputation score [0-1]
        """
        # If there are no trades in the matching result, return existing reputation
        if not matching_result.trades:
            return agent.reputation

        # STEP 1. Reliability score
        agent_trades = [t for t in matching_result.trades if t.buyer_id == agent.id or t.seller_id == agent.id]
        agent_unmatched = [o for o in matching_result.unmatched_orders if o.agent_id == agent.id]

        # Higher reliability if more orders were matched
        reliability = len(agent_trades) / (len(agent_trades) + len(agent_unmatched)) if (agent_trades or agent_unmatched) else self._neutral

        # STEP 2. Price fairness
        if agent_trades:
            price_deviations = [abs(t.price - matching_result.clearing_price) / matching_result.clearing_price for t in agent_trades if matching_result.clearing_price > 0]
            fairness = 1.0 - min(1.0, np.mean(price_deviations) if price_deviations else 0.0)
        else:
            fairness = self._neutral

        # STEP 3. Grid contribution
        if agent_trades:
            grid_contributions = []

            for trade in agent_trades:
                # Only consider trades where this agent is directly involved
                if (trade.buyer_id == agent.id) or (trade.seller_id == agent.id):
                    # Calculate the impact of this trade on grid stability
                    impact = self._calculate_grid_impact(agent.id, trade, matching_result.grid_balance, dso_id)
                    grid_contributions.append(impact)

            # Average the contributions across all trades
            grid_contribution = np.mean(grid_contributions) if grid_contributions else 0.0

            # Normalize to 0-1 range (since _calculate_grid_impact returns -1 to 1)
            grid_contribution = (grid_contribution + 1) / 2
        else:
            grid_contribution = self._neutral

        # Calculate overall score with weights
        overall_score = (self.reliability_weight * reliability) + (self.fairness_weight * fairness) + (self.grid_weight * grid_contribution)

        return max(0.0, min(1.0, overall_score))

    def _calculate_grid_impact(self,
                               agent_id: str,
                               trade: Trade,
                               grid_balance: float,
                               dso_id: str) -> float:
        """Calculate trade's impact on grid stability.

        This method evaluates how a trade impacts grid stability based on the current grid state.
        - Positive impact: Agent helps balance the grid (sells during shortage, buys during excess)
        - Negative impact: Agent worsens grid imbalance (buys during shortage, sells during excess)

        Args:
            agent_id: ID of the DERAgent whose reputation is being calculated
            trade: Completed trade
            grid_balance: Current grid balance (positive = excess supply, negative = shortage)
            dso_id: ID of the DSO agent for identifying DSO trades

        Returns:
            Impact score between -1 and 1 (higher is better for grid stability)
        """
        impact = 0.0

        # If grid is balanced, neutral impact
        if abs(grid_balance) < 0.1:
            return 0.0

        # For DSO trades, evaluate differently
        if (trade.buyer_id == dso_id) or (trade.seller_id == dso_id):
            # DSO trades are always stabilizing by definition
            return self._neutral

        # Calculate impact based on grid state and agent's role
        is_buyer = (trade.buyer_id == agent_id)
        if grid_balance > 0:  # Excess supply
            if is_buyer:
                # Buying during excess is good (+)
                impact = min(1.0, trade.quantity / abs(grid_balance))
            else:
                # Selling during excess is bad (-)
                impact = -min(1.0, trade.quantity / abs(grid_balance))
        else:  # grid_balance < 0, Shortage
            if is_buyer:
                # Buying during shortage is bad (-)
                impact = -min(1.0, trade.quantity / abs(grid_balance))
            else:
                # Selling during shortage is good (+)
                impact = min(1.0, trade.quantity / abs(grid_balance))

        return impact
