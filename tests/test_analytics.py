"""
Analytics: Validates the functionality of the updated analytics module.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.coordination_metrics import CoordinationMetricsHandler
from src.analytics.dso_metrics import DSOMetricsHandler
from src.analytics.grid_metrics import GridMetricsHandler
from src.analytics.market_analytics import MarketMetricsHandler
from src.grid.network import GridNetwork
from src.market.matching import MatchingHistory, MatchingResult
from src.market.order import Trade


def _create_matching_history_and_rewards() -> Tuple[MatchingHistory, Dict[str, List[float]]]:
    """Create a matching history and rewards for testing.

    Returns:
        Tuple[MatchingHistory, Dict[str, List[float]]]: A tuple containing the matching history and agent rewards.
    """
    matching_history = MatchingHistory()

    # Sample matching results
    for i in range(5):
        matching_history.update(MatchingResult(trades=[Trade(f"buyer{j}", f"seller{j}", 100 + i*5, 10 + i, timestamp=time.time()) for j in range(3)],
                                               unmatched_orders=[],
                                               clearing_price=100 + i*5,
                                               clearing_volume=30 + i*3,
                                               grid_balance=i*2 - 5,
                                               grid_congestion=0.1 + i*0.05,
                                               dso_buy_volume=5 + i,
                                               dso_sell_volume=8 + i,
                                               dso_total_volume=13 + i*2,
                                               p2p_volume=20 + i*2,
                                               dso_trade_ratio=0.4 - i*0.05,
                                               dso_grid_import=3 + i,
                                               dso_buy_price=90 + i*2,
                                               dso_sell_price=110 + i*3,
                                               price_spread=20 + i,
                                               local_price_avg=105 + i*2,
                                               local_price_advantage=5 + i))

    # Sample agent rewards
    agent_rewards = {"agent1": [10, 15, 12, 18, 20],
                     "agent2": [8, 12, 14, 16, 18],
                     "agent3": [5, 8, 10, 12, 15]}

    return matching_history, agent_rewards


def test_market_metrics() -> None:
    """Test market analytics functionality."""
    print("MARKET ANALYTICS SHOWCASE")

    print("--- STEP 1. Market Performance Analysis ---")
    matching_history, agent_rewards  = _create_matching_history_and_rewards()
    metrics = MarketMetricsHandler(agent_rewards, matching_history).get_metrics()

    # Core Economic Indicators
    print("--- STEP 2. Core Economic Indicators ---")
    print(f"  - Social welfare: ${metrics.social_welfare:.2f}")
    print(f"  - Consumer surplus: ${metrics.consumer_surplus:.2f}")
    print(f"  - Producer surplus: ${metrics.producer_surplus:.2f}")

    # Price Metrics
    print("--- STEP 3. Price Metrics ---")
    print(f"  - Average clearing price: ${metrics.avg_clearing_price:.2f}")
    print(f"  - Price volatility: {metrics.price_volatility:.4f}")
    print(f"  - Price trend: {metrics.price_trend:.4f}")

    # Market Efficiency
    print("--- STEP 4. Market Efficiency ---")
    print(f"  - Allocation efficiency: {metrics.allocation_efficiency:.4f}")
    print(f"  - Price discovery efficiency: {metrics.price_discovery_efficiency:.4f}")

    # Trading Volume Metrics
    print("--- STEP 5. Trading Volume Metrics ---")
    print(f"  - Total trading volume: {metrics.total_trading_volume:.2f} kWh")
    print(f"  - Transaction count: {metrics.transaction_count}")
    print(f"  - Average trade size: {metrics.avg_trade_size:.2f}")

    # Market Structure
    print("--- STEP 6. Market Structure ---")
    print(f"  - Market concentration: {metrics.market_concentration:.4f}")
    print(f"  - P2P trade ratio: {metrics.p2p_trade_ratio:.4f}")
    print(f"  - DSO dependency ratio: {metrics.dso_dependency_ratio:.4f}")
    print(f"  - Market liquidity: {metrics.market_liquidity:.4f}")

    # Welfare Distribution
    print("--- STEP 7. Welfare Distribution ---")
    print(f"  - Welfare distribution Gini: {metrics.welfare_distribution_gini:.4f}")
    print(f"  - Price fairness index: {metrics.price_fairness_index:.4f}")

    # Temporal trends
    print("--- STEP 8. Temporal Trends ---")
    print(f"  - Volume trend: {metrics.volume_trend:.4f}")
    print(f"  - Efficiency trend: {metrics.efficiency_trend:.4f}")

    # Agent behavior
    print("--- STEP 9. Agent Behavior ---")
    print(f"  - Average agent rewards: {metrics.avg_agent_rewards}")
    print(f"  - Trading frequency by agent: {metrics.trading_frequency_by_agent}")


def test_dso_metrics() -> None:
    """Test metrics extraction functions."""
    print("METRICS EXTRACTION SHOWCASE")

    print("--- STEP 1. Creating Test Data ---")
    matching_history, _  = _create_matching_history_and_rewards()
    metrics = DSOMetricsHandler(1000.0, matching_history).get_metrics()

    print("--- STEP 2. Comprehensive Metrics Extraction ---")
    print(f"  - DSO buy volume: {metrics.dso_buy_volume}")
    print(f"  - DSO sell volume: {metrics.dso_sell_volume}")
    print(f"  - DSO total volume: {metrics.dso_total_volume}")
    print(f"  - P2P volume: {metrics.p2p_volume}")

    # Market Share and Dependency
    print("--- STEP 3. Market Share and Dependency ---")
    print(f"  - DSO trade ratio: {metrics.dso_trade_ratio}")
    print(f"  - P2P trade ratio: {metrics.p2p_trade_ratio}")

    # Financial Metrics
    print("--- STEP 4. Financial Metrics ---")
    print(f"  - DSO buy price average: {metrics.dso_buy_price_avg}")
    print(f"  - DSO sell price average: {metrics.dso_sell_price_avg}")
    print(f"  - Price spread: {metrics.price_spread}")
    print(f"  - Local price average: {metrics.local_price_avg}")
    print(f"  - Local price advantage: {metrics.local_price_advantage}")

    # Grid Import/Export
    print("--- STEP 5. Grid Import/Export ---")
    print(f"  - Net grid import: {metrics.net_grid_import}")
    print(f"  - Grid import ratio: {metrics.grid_import_ratio}")
    print(f"  - Self-sufficiency ratio: {metrics.self_sufficiency_ratio}")

    # Cost and Savings
    print("--- STEP 6. Cost and Savings ---")
    print(f"  - DSO profit: {metrics.dso_profit}")
    print(f"  - Avoided DSO cost: {metrics.avoided_dso_cost}")
    print(f"  - Local trading benefit: {metrics.local_trading_benefit}")

    # Performance Indicators
    print("--- STEP 7. Performance Indicators ---")
    print(f"  - DSO utilization efficiency: {metrics.dso_utilization_efficiency}")
    print(f"  - Market balance quality: {metrics.market_balance_quality}")
    print(f"  - Fallback effectiveness: {metrics.fallback_effectiveness}")


def test_coordination_metrics() -> None:
    """Test coordination analysis functionality."""
    print("COORDINATION ANALYSIS SHOWCASE")

    print("--- STEP 1. Creating Coordination Metrics Handler ---")
    matching_history, agent_rewards  = _create_matching_history_and_rewards()
    metrics = CoordinationMetricsHandler(1000.0, matching_history).get_metrics(agent_rewards)

    # Core Coordination Indicators
    print("--- STEP 2. Core Coordination Indicators ---")
    print(f"  - Coordination score: {metrics.coordination_score}")
    print(f"  - Coordination convergence: {metrics.coordination_convergence}")

    # Emergent Behavior Patterns
    print("--- STEP 3. Emergent Behavior Patterns ---")
    print(f"  - Strategy alignment: {metrics.strategy_alignment}")
    print(f"  - Emergent efficiency: {metrics.emergent_efficiency}")

    # Agent Responsiveness
    print("--- STEP 4. Agent Responsiveness ---")
    print(f"  - Information efficiency: {metrics.information_efficiency}")

    # Resource Coordination
    print("--- STEP 5. Resource Coordination ---")
    print(f"  - Resource coordination index: {metrics.resource_coordination_index}")

    # Market Balance Coordination
    print("--- STEP 6. Market Balance Coordination ---")
    print(f"  - Supply-demand coordination: {metrics.supply_demand_coordination}")

    # Temporal Coordination Patterns
    print("--- STEP 7. Temporal Coordination Patterns ---")
    print(f"  - Coordination trend: {metrics.coordination_trend}")
    print(f"  - Coordination stability: {metrics.coordination_stability}")

    # Implicit Cooperation Validation Metrics
    print("--- STEP 8. Implicit Cooperation Validation Metrics ---")
    print(f"  - Signal impact score: {metrics.signal_impact_score}")
    print(f"  - Agent responsiveness: {metrics.agent_responsiveness}")
    print(f"  - Emergent coordination strength: {metrics.emergent_coordination_strength}")

    # DER Management Specific Metrics
    print("--- STEP 9. DER Management Specific Metrics ---")
    print(f"  - Battery coordination efficiency: {metrics.battery_coordination_efficiency}")
    print(f"  - Peak reduction coordination: {metrics.peak_reduction_coordination}")
    print(f"  - Energy waste reduction: {metrics.energy_waste_reduction}")


def test_grid_metrics() -> None:
    """Test grid analytics functionality."""
    print("GRID ANALYTICS SHOWCASE")

    print("--- STEP 1. Creating Grid Metrics Handler ---")
    grid_network = GridNetwork(num_nodes=3, capacity=1000.0)
    matching_history, _  = _create_matching_history_and_rewards()
    metrics = GridMetricsHandler(grid_network, matching_history).get_metrics()

    # Stability Metrics
    print("--- STEP 2. Stability Metrics ---")
    print(f"  - Grid stability index: {metrics.grid_stability_index:.4f}")
    print(f"  - Supply-demand balance: {metrics.supply_demand_balance:.4f}")

    # Congestion Metrics
    print("--- STEP 3. Congestion Metrics ---")
    print(f"  - Average congestion level: {metrics.avg_congestion_level:.4f}")

    # Efficiency Metrics
    print("--- STEP 4. Efficiency Metrics ---")
    print(f"  - Transmission loss ratio: {metrics.transmission_loss_ratio:.4f}")
    print(f"  - Grid utilization efficiency: {metrics.grid_utilization_efficiency:.4f}")

    # Operational Metrics
    print("--- STEP 5. Operational Metrics ---")
    print(f"  - Transmission losses: {metrics.transmission_losses:.4f}")
    print(f"  - Load factor: {metrics.load_factor:.4f}")

    # Network Performance
    print("--- STEP 6. Network Performance ---")
    print(f"  - Capacity utilization: {metrics.capacity_utilization:.4f}")

    # Temporal Patterns
    print("--- STEP 7. Temporal Patterns ---")
    print(f"  - Stability trend: {metrics.stability_trend:.4f}")
    print(f"  - Congestion trend: {metrics.congestion_trend:.4f}")
    print(f"  - Efficiency trend: {metrics.efficiency_trend:.4f}")


def run_analytics_tests() -> bool:
    """Run all analytics tests."""
    print("🚀 STARTING ANALYTICS MODULE SHOWCASE")

    try:
        # Test market analytics
        test_market_metrics()

        # Test metrics extraction
        test_dso_metrics()

        # Test coordination analysis
        test_coordination_metrics()

        # Test grid analytics
        test_grid_metrics()

        print("🎉 ANALYTICS SHOWCASE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == '__main__':
    success = run_analytics_tests()
    exit(0 if success else 1)
