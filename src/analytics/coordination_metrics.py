"""
Coordination Metrics Handler for Local Energy Markets.

This module provides specialized tools for analyzing coordination signals,
measuring the effectiveness of the implicit cooperation model, and
identifying emergent behaviors in decentralized energy markets.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats

from ..market.matching import MatchingHistory


@dataclass
class CoordinationMetrics:
    """Comprehensive coordination metrics for the Local Energy Market."""

    # Core Coordination Indicators
    coordination_score: float = 0.0
    coordination_convergence: float = 0.0

    # Emergent Behavior Patterns
    strategy_alignment: float = 0.0
    emergent_efficiency: float = 0.0

    # Agent Responsiveness
    information_efficiency: float = 0.0

    # Resource Coordination
    resource_coordination_index: float = 0.0

    # Market Balance Coordination
    supply_demand_coordination: float = 0.0

    # Temporal Coordination Patterns
    coordination_trend: float = 0.0
    coordination_stability: float = 0.0

    # Implicit Cooperation Validation Metrics
    signal_impact_score: float = 0.0
    agent_responsiveness: float = 0.0
    emergent_coordination_strength: float = 0.0

    # DER Management Specific Metrics
    battery_coordination_efficiency: float = 0.0
    peak_reduction_coordination: float = 0.0
    energy_waste_reduction: float = 0.0


class CoordinationMetricsHandler:
    """Calculator for coordination metrics in the Local Energy Market."""

    def __init__(self,
                 grid_capacity: float,
                 matching_history: MatchingHistory,
                 _window: int = 2) -> None:
        """Initialize the coordination metrics calculator.

        Args:
            grid_capacity: Grid capacity for normalization purposes
            matching_history: MatchingHistory object
            _window: Window size for the analysis
        """
        self.grid_capacity = grid_capacity
        self.history = matching_history.history
        self._window = _window

    def get_metrics(self, agents_reward: Dict[str, List[float]]) -> CoordinationMetrics:
        """Calculate coordination metrics from the stored history.

        Args:
            agents_reward: Agent reward histories for behavior analysis

        Returns:
            CoordinationMetrics object with calculated metrics
        """
        if not self.history:
            return CoordinationMetrics()

        # Core coordination indicators
        coordination_score = self._calculate_coordination_score()
        coordination_convergence = self._calculate_coordination_convergence()

        # Emergent behavior patterns
        strategy_alignment = self._calculate_strategy_alignment(agents_reward)
        emergent_efficiency = self._calculate_emergent_efficiency()

        # Agent responsiveness
        information_efficiency = self._calculate_information_efficiency(agents_reward)

        # Resource coordination
        resource_coordination_index = self._calculate_resource_coordination()

        # Market balance coordination
        supply_demand_coordination = self._calculate_supply_demand_coordination()

        # Temporal coordination patterns
        coordination_trend = self._calculate_coordination_trend()
        coordination_stability = self._calculate_coordination_stability()

        # Implicit Cooperation Validation Metrics
        signal_impact_score = self._calculate_signal_impact_score(agents_reward)
        agent_responsiveness = self._calculate_agent_responsiveness(agents_reward)
        emergent_coordination_strength = self._calculate_emergent_coordination_strength()

        # DER Management Specific Metrics
        battery_coordination_efficiency = self._calculate_battery_coordination_efficiency()
        peak_reduction_coordination = self._calculate_peak_reduction_coordination()
        energy_waste_reduction = self._calculate_energy_waste_reduction()

        return CoordinationMetrics(coordination_score=coordination_score,
                                   coordination_convergence=coordination_convergence,
                                   strategy_alignment=strategy_alignment,
                                   emergent_efficiency=emergent_efficiency,
                                   information_efficiency=information_efficiency,
                                   resource_coordination_index=resource_coordination_index,
                                   supply_demand_coordination=supply_demand_coordination,
                                   coordination_trend=coordination_trend,
                                   coordination_stability=coordination_stability,
                                   signal_impact_score=signal_impact_score,
                                   agent_responsiveness=agent_responsiveness,
                                   emergent_coordination_strength=emergent_coordination_strength,
                                   battery_coordination_efficiency=battery_coordination_efficiency,
                                   peak_reduction_coordination=peak_reduction_coordination,
                                   energy_waste_reduction=energy_waste_reduction)

    def _calculate_coordination_score(self) -> float:
        """Calculate overall coordination effectiveness score.

        Returns:
            Coordination effectiveness score (0-1)
        """
        if not self.history:
            return 0.0

        # Coordination based on multiple factors
        balance_scores = []
        efficiency_scores = []

        for result in self.history:
            # Grid balance coordination (better balance = higher score)
            balance_normalized = abs(result.grid_balance) / self.grid_capacity
            balance_score = max(0.0, 1.0 - balance_normalized)
            balance_scores.append(balance_score)

            # Market efficiency coordination (higher P2P ratio = better coordination)
            total_volume = result.p2p_volume + result.dso_total_volume
            if total_volume > 0:
                efficiency_score = result.p2p_volume / total_volume
            else:
                efficiency_score = 0.0
            efficiency_scores.append(efficiency_score)

        # Combine balance and efficiency
        all_scores = balance_scores + efficiency_scores
        return float(np.mean(all_scores)) if all_scores else 0.0

    def _calculate_coordination_convergence(self) -> float:
        """Calculate coordination convergence over time.

        Returns:
            Coordination convergence over time (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Calculate coordination scores over time
        coord_scores = []
        for result in self.history:
            # Grid balance coordination (better balance = higher score)
            balance_normalized = abs(result.grid_balance) / self.grid_capacity
            balance_score = max(0.0, 1.0 - balance_normalized)

            # Market efficiency coordination (higher P2P ratio = better coordination)
            total_volume = result.p2p_volume + result.dso_total_volume
            if total_volume > 0:
                efficiency_score = result.p2p_volume / total_volume
            else:
                efficiency_score = 0.0

            # Combine balance and efficiency for this time step
            coord_scores.append(float(np.mean([balance_score, efficiency_score])))

        # Check for convergence (decreasing variance over time)
        early_variance = np.var(coord_scores[:self._window])
        late_variance = np.var(coord_scores[-self._window:])

        if early_variance == 0:
            return 1.0 if late_variance == 0 else 0.0

        # Calculate convergence
        convergence = max(0.0, (early_variance - late_variance) / early_variance)

        return min(1.0, convergence)

    def _calculate_strategy_alignment(self, agents_reward: Dict[str, List[float]]) -> float:
        """Calculate strategy alignment between agents.

        Args:
            agents_reward: Agent reward histories for behavior analysis

        Returns:
            Strategy alignment between agents (0-1)
        """
        if len(agents_reward) < self._window:
            return 0.0

        # Calculate pairwise correlations between agent reward patterns
        correlations = []
        agent_ids = list(agents_reward.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                rewards_i = agents_reward[agent_ids[i]]
                rewards_j = agents_reward[agent_ids[j]]

                if len(rewards_i) >= 2 and len(rewards_j) >= 2:
                    min_len = min(len(rewards_i), len(rewards_j))
                    try:
                        corr = np.corrcoef(rewards_i[:min_len], rewards_j[:min_len])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except Exception:
                        pass

        return np.mean(correlations) if correlations else 0.0

    def _calculate_emergent_efficiency(self) -> float:
        """Calculate efficiency from emergent behaviors.

        Returns:
            Efficiency from emergent behaviors (0-1)
        """
        if not self.history:
            return 0.0

        # Efficiency = improvement in market metrics over time
        p2p_ratios = [r.p2p_volume / (r.p2p_volume + r.dso_total_volume) if r.p2p_volume + r.dso_total_volume > 0 else 0.0 for r in self.history]

        if len(p2p_ratios) < 2:
            return np.mean(p2p_ratios) if p2p_ratios else 0.0

        # Trend toward higher P2P ratio indicates emergent efficiency
        trend = self._calculate_trend(p2p_ratios)

        return max(0.0, min(1.0, trend + np.mean(p2p_ratios)))

    def _calculate_information_efficiency(self, agents_reward: Dict[str, List[float]]) -> float:
        """Calculate information propagation efficiency.

        Args:
            agents_reward: Agent reward histories for behavior analysis

        Returns:
            Information propagation efficiency (0-1)
        """
        if len(agents_reward) < self._window:
            return 0.0

        # Information efficiency = uniformity in agent performance (less asymmetry)
        all_rewards = []
        for rewards in agents_reward.values():
            all_rewards.extend(rewards)

        if not all_rewards:
            return 0.0

        # Lower variance = better information efficiency
        reward_variance = np.var(all_rewards)
        reward_mean = np.mean(all_rewards)

        if reward_mean == 0:
            return 1.0 if reward_variance == 0 else 0.0

        # Coefficient of variation
        cv = np.sqrt(reward_variance) / abs(reward_mean)

        return max(0.0, 1.0 - min(cv, 1.0))

    def _calculate_resource_coordination(self) -> float:
        """Calculate resource coordination index.

        Returns:
            Resource coordination index (0-1)
        """
        if not self.history:
            return 0.0

        # Resource coordination = successful P2P matching relative to total volume
        total_p2p = sum(r.p2p_volume for r in self.history)
        total_volume = sum(r.p2p_volume + r.dso_total_volume for r in self.history)

        return total_p2p / total_volume if total_volume > 0 else 0.0

    def _calculate_supply_demand_coordination(self) -> float:
        """Calculate supply-demand balance coordination.

        Returns:
            Supply-demand balance coordination (0-1)
        """
        if not self.history:
            return 0.0

        # Coordination = ability to maintain grid balance
        grid_balances = [abs(r.grid_balance) for r in self.history]
        avg_imbalance = np.mean(grid_balances)

        # Normalize by grid capacity
        normalized_imbalance = avg_imbalance / self.grid_capacity
        return max(0.0, 1.0 - normalized_imbalance)

    def _calculate_coordination_trend(self) -> float:
        """Calculate coordination improvement trend.

        Returns:
            Coordination improvement trend (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Calculate coordination scores over time
        coord_scores = []
        for result in self.history:
            # Grid balance coordination (better balance = higher score)
            balance_normalized = abs(result.grid_balance) / self.grid_capacity
            balance_score = max(0.0, 1.0 - balance_normalized)

            # Market efficiency coordination (higher P2P ratio = better coordination)
            total_volume = result.p2p_volume + result.dso_total_volume
            if total_volume > 0:
                efficiency_score = result.p2p_volume / total_volume
            else:
                efficiency_score = 0.0

            # Combine balance and efficiency for this time step
            coord_scores.append(float(np.mean([balance_score, efficiency_score])))

        return self._calculate_trend(coord_scores)

    def _calculate_coordination_stability(self) -> float:
        """Calculate stability of coordination.

        Returns:
            Stability of coordination (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        coord_scores = []
        for result in self.history:
            # Grid balance coordination (better balance = higher score)
            balance_normalized = abs(result.grid_balance) / self.grid_capacity
            balance_score = max(0.0, 1.0 - balance_normalized)

            # Market efficiency coordination (higher P2P ratio = better coordination)
            total_volume = result.p2p_volume + result.dso_total_volume
            if total_volume > 0:
                efficiency_score = result.p2p_volume / total_volume
            else:
                efficiency_score = 0.0

            # Combine balance and efficiency for this time step
            coord_scores.append(float(np.mean([balance_score, efficiency_score])))

        if not coord_scores:
            return 0.0

        # Stability = 1 - coefficient of variation
        mean_score = np.mean(coord_scores)
        if mean_score == 0:
            return 1.0 if np.std(coord_scores) == 0 else 0.0

        cv = np.std(coord_scores) / mean_score
        return max(0.0, 1.0 - cv)

    # Implicit Cooperation Validation Methods
    def _calculate_signal_impact_score(self, agents_reward: Dict[str, List[float]]) -> float:
        """Calculate how coordination signals influence market outcomes.

        Args:
            agents_reward: Agent reward histories for behavior analysis

        Returns:
            Signal impact score (0-1)
        """
        if len(self.history) < self._window or not agents_reward:
            return 0.0

        # Extract coordination signals (P2P ratio as proxy for coordination)
        coordination_signals = []
        market_outcomes = []

        for result in self.history:
            total_volume = result.p2p_volume + result.dso_total_volume
            if total_volume > 0:
                p2p_ratio = result.p2p_volume / total_volume
                coordination_signals.append(p2p_ratio)

                # Market outcome: grid balance quality
                balance_quality = max(0.0, 1.0 - abs(result.grid_balance) / self.grid_capacity)
                market_outcomes.append(balance_quality)

        if len(coordination_signals) < 2 or len(market_outcomes) < 2:
            return 0.0

        try:
            correlation = np.corrcoef(coordination_signals, market_outcomes)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _calculate_agent_responsiveness(self, agents_reward: Dict[str, List[float]]) -> float:
        """Calculate how quickly agents respond to coordination signals.

        Args:
            agents_reward: Agent reward histories for behavior analysis

        Returns:
            Agent responsiveness score (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Calculate response time as correlation between signal changes and reward changes
        responsiveness_scores = []

        for agent_id, rewards in agents_reward.items():
            if len(rewards) < self._window:
                continue

            # Calculate reward changes (responses)
            reward_changes = [rewards[i] - rewards[i-1] for i in range(1, len(rewards))]

            # Calculate coordination signal changes (P2P ratio changes)
            signal_changes = []
            for i in range(1, len(self.history)):
                prev_total = self.history[i-1].p2p_volume + self.history[i-1].dso_total_volume
                curr_total = self.history[i].p2p_volume + self.history[i].dso_total_volume

                if prev_total > 0 and curr_total > 0:
                    prev_ratio = self.history[i-1].p2p_volume / prev_total
                    curr_ratio = self.history[i].p2p_volume / curr_total
                    signal_changes.append(curr_ratio - prev_ratio)

            # Calculate correlation between signal changes and reward changes
            min_len = min(len(reward_changes), len(signal_changes))
            if min_len >= 2:
                try:
                    corr = np.corrcoef(reward_changes[:min_len], signal_changes[:min_len])[0, 1]
                    if not np.isnan(corr):
                        responsiveness_scores.append(abs(corr))
                except Exception:
                    pass

        return np.mean(responsiveness_scores) if responsiveness_scores else 0.0

    def _calculate_emergent_coordination_strength(self) -> float:
        """Calculate the strength of emergent coordination patterns.

        Returns:
            Emergent coordination strength (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Measure coordination strength through multiple indicators
        coordination_indicators = []

        for result in self.history:
            # Indicator 1: P2P efficiency
            total_volume = result.p2p_volume + result.dso_total_volume
            p2p_efficiency = result.p2p_volume / total_volume if total_volume > 0 else 0.0

            # Indicator 2: Grid balance quality
            balance_quality = max(0.0, 1.0 - abs(result.grid_balance) / self.grid_capacity)

            # Indicator 3: Market clearing efficiency
            clearing_efficiency = result.clearing_volume / max(result.clearing_price, 0.01) if result.clearing_price > 0 else 0.0
            clearing_efficiency = min(1.0, clearing_efficiency / 100.0)  # Normalize

            # Combine indicators
            combined_strength = np.mean([p2p_efficiency, balance_quality, clearing_efficiency])
            coordination_indicators.append(combined_strength)

        # Calculate trend and consistency
        if len(coordination_indicators) >= 2:
            trend = self._calculate_trend(coordination_indicators)
            consistency = 1.0 - (np.std(coordination_indicators) / max(np.mean(coordination_indicators), 0.01))
            return max(0.0, min(1.0, (trend + consistency) / 2.0))

        return np.mean(coordination_indicators) if coordination_indicators else 0.0

    def _calculate_battery_coordination_efficiency(self) -> float:
        """Calculate how well storage systems are coordinated.

        Returns:
            Battery coordination efficiency (0-1)
        """
        if not self.history:
            return 0.0

        # Battery coordination = efficient use of storage for grid balance
        coordination_scores = []

        for result in self.history:
            # Measure how well battery usage contributes to grid balance
            grid_balance = abs(result.grid_balance)

            # If grid is balanced, battery coordination is good
            if grid_balance < self.grid_capacity * 0.1:  # Within 10% of capacity
                battery_score = 1.0
            else:
                # Penalize excessive grid imbalance
                battery_score = max(0.0, 1.0 - (grid_balance / self.grid_capacity))

            coordination_scores.append(battery_score)

        return np.mean(coordination_scores) if coordination_scores else 0.0

    def _calculate_peak_reduction_coordination(self) -> float:
        """Calculate peak reduction achieved through coordination.

        Returns:
            Peak reduction coordination score (0-1)
        """
        if len(self.history) < self._window:
            return 0.0

        # Calculate load factor improvement over time
        volumes = [r.clearing_volume for r in self.history]
        if not volumes:
            return 0.0

        # Load factor = average / peak (higher is better for peak reduction)
        load_factor = np.mean(volumes) / max(volumes) if max(volumes) > 0 else 0.0

        # Calculate trend in load factor (improving trend = better peak reduction)
        if len(volumes) >= self._window:
            # Calculate rolling load factors
            window_size = min(5, len(volumes) // 2)
            rolling_load_factors = []

            for i in range(len(volumes) - window_size + 1):
                window_volumes = volumes[i:i + window_size]
                window_load_factor = np.mean(window_volumes) / max(window_volumes) if max(window_volumes) > 0 else 0.0
                rolling_load_factors.append(window_load_factor)

            if len(rolling_load_factors) >= 2:
                trend = self._calculate_trend(rolling_load_factors)
                return max(0.0, min(1.0, load_factor + trend))

        return load_factor

    def _calculate_energy_waste_reduction(self) -> float:
        """Calculate energy waste reduction through coordination.

        Returns:
            Energy waste reduction score (0-1)
        """
        if not self.history:
            return 0.0

        # Energy waste = unused generation + excessive grid imports/exports
        waste_scores = []

        for result in self.history:
            # Calculate waste indicators
            total_volume = result.p2p_volume + result.dso_total_volume

            # Waste 1: Low P2P ratio (energy not used locally)
            p2p_ratio = result.p2p_volume / total_volume if total_volume > 0 else 0.0

            # Waste 2: High grid imbalance (excessive imports/exports)
            grid_imbalance = abs(result.grid_balance) / self.grid_capacity

            # Waste 3: Low clearing efficiency
            clearing_efficiency = result.clearing_volume / max(result.clearing_price, 0.01) if result.clearing_price > 0 else 0.0
            clearing_efficiency = min(1.0, clearing_efficiency / 100.0)  # Normalize

            # Combine waste indicators (lower waste = higher score)
            waste_score = np.mean([p2p_ratio, 1.0 - grid_imbalance, clearing_efficiency])
            waste_scores.append(waste_score)

        return np.mean(waste_scores) if waste_scores else 0.0

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of a time series.

        Args:
            values: List of values to calculate trend for

        Returns:
            Trend of the time series (0-1)
        """
        if len(values) < self._window:
            return 0.0

        x = np.arange(len(values))
        try:
            slope, _, r_value, _, _ = stats.linregress(x, values)
            return slope * r_value
        except Exception:
            return 0.0
