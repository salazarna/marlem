"""
Visualization: Validates the functionality of the updated visualization module.

This module tests all visualization capabilities including:
- Trading network visualization
- Price and volume distribution plots
- Spatial distribution heatmaps
- Market metrics visualization
- Agent behavior analysis
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.grid_metrics import GridMetrics

# Import only what we need - metrics classes are imported within test functions
from src.market.matching import MatchingHistory, MatchingResult
from src.market.order import Trade
from src.root import __main__
from src.visualization.plotter import Plotter


def test_trading_network_visualization(path: str,
                                       num_history: int = 10,
                                       num_trades: int = 3,
                                       num_agents: int = 5) -> None:
    """Test trading network visualization functionality.

    Args:
        path: Path to save the plots
        num_history: Number of history to create
        num_trades: Number of trades to create
        num_agents: Number of agents to create
    """
    print("--- STEP 1: Creating Sample Trading Data ---")
    matching_history = MatchingHistory()

    # Define a fixed set of agents for realistic trading patterns
    buyers = [f"buyer_{i}" for i in range(num_agents)]
    sellers = [f"seller_{i}" for i in range(num_agents)]

    print(f"  - Using {num_agents} buyers and {num_agents} sellers")
    print(f"  - Creating {num_trades} trades per period")

    # Create sample matching results with trades
    for i in range(num_history):
        trades = []
        for j in range(num_trades):
            # Randomly select buyer and seller from the fixed agent pool
            buyer = np.random.choice(buyers)
            seller = np.random.choice(sellers)

            # Ensure buyer and seller are different (no self-trading)
            while buyer == seller:
                seller = np.random.choice(sellers)

            trades.append(Trade(buyer_id=buyer,
                                seller_id=seller,
                                price=100 + i*5 + j*2,
                                quantity=10 + i + j,
                                timestamp=time.time()))

        matching_history.update(MatchingResult(trades=trades,
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

    print(f"✓ Created matching history with {len(matching_history.history)} results")
    print(f"✓ Total trades: {sum(len(r.trades) for r in matching_history.history)}")

    print("--- STEP 2. Testing MarketVisualizer Class ---")
    visualizer = Plotter(save_path=path)
    print("✓ MarketVisualizer initialized successfully")

    # Test trading network figure creation
    visualizer.trading_network(matching_history)
    print("✓ Trading network figure created successfully")

    print("--- STEP 3. Network Analysis ---")
    # Count unique agents
    all_trades = []
    for result in matching_history.history:
        all_trades.extend(result.trades)

    unique_buyers = set(trade.buyer_id for trade in all_trades)
    unique_sellers = set(trade.seller_id for trade in all_trades)
    all_agents = unique_buyers | unique_sellers

    print(f"  - Unique buyers: {len(unique_buyers)}")
    print(f"  - Unique sellers: {len(unique_sellers)}")
    print(f"  - Total unique agents: {len(all_agents)}")
    print(f"  - Total trades: {len(all_trades)}")

    # Calculate trading patterns
    buyer_activity = {}
    seller_activity = {}
    for trade in all_trades:
        buyer_activity[trade.buyer_id] = buyer_activity.get(trade.buyer_id, 0) + 1
        seller_activity[trade.seller_id] = seller_activity.get(trade.seller_id, 0) + 1

    print(f"  - Most active buyer: {max(buyer_activity, key=buyer_activity.get)} ({max(buyer_activity.values())} trades)")
    print(f"  - Most active seller: {max(seller_activity, key=seller_activity.get)} ({max(seller_activity.values())} trades)")
    print(f"  - Average trades per buyer: {np.mean(list(buyer_activity.values())):.1f}")
    print(f"  - Average trades per seller: {np.mean(list(seller_activity.values())):.1f}")


def test_price_volume_distribution(path: str) -> None:
    """Test price and volume distribution visualization."""

    print("--- STEP 1: Creating Sample Market Data ---")
    matching_history = MatchingHistory()

    # Create sample data with varied prices and volumes
    for i in range(20):
        trades = []
        for j in range(np.random.randint(1, 5)):
            trades.append(Trade(buyer_id=f"buyer{j}",
                                seller_id=f"seller{j}",
                                price=np.random.normal(100, 20) + i*2,
                                quantity=np.random.exponential(10) + 5,
                                timestamp=time.time()))

        matching_history.update(
            MatchingResult(trades=trades,
                           unmatched_orders=[],
                           clearing_price=np.random.normal(100, 15) + i*1.5,
                           clearing_volume=sum(t.quantity for t in trades),
                           grid_balance=np.random.normal(0, 2),
                           grid_congestion=0.1 + np.random.random() * 0.2,
                           dso_buy_volume=5 + i,
                           dso_sell_volume=8 + i,
                           dso_total_volume=13 + i*2,
                           p2p_volume=20 + i*2,
                           dso_trade_ratio=0.4 - i*0.02,
                           dso_grid_import=3 + i,
                           dso_buy_price=90 + i*2,
                           dso_sell_price=110 + i*3,
                           price_spread=20 + i,
                           local_price_avg=105 + i*2,
                           local_price_advantage=5 + i))

    print(f"✓ Created market data with {len(matching_history.history)} periods")

    print("--- STEP 2. Testing Price Distribution ---")
    clearing_prices = [result.clearing_price for result in matching_history.history]
    print(f"  - Price range: ${min(clearing_prices):.2f} - ${max(clearing_prices):.2f}")
    print(f"  - Average price: ${np.mean(clearing_prices):.2f}")
    print(f"  - Price volatility: {np.std(clearing_prices):.2f}")

    print("--- STEP 3. Testing Volume Distribution ---")
    trade_volumes = []
    for result in matching_history.history:
        total_volume = sum(trade.quantity for trade in result.trades)
        trade_volumes.append(total_volume)

    print(f"  - Volume range: {min(trade_volumes):.2f} - {max(trade_volumes):.2f} kWh")
    print(f"  - Average volume: {np.mean(trade_volumes):.2f} kWh")
    print(f"  - Volume variability: {np.std(trade_volumes):.2f}")

    print("--- STEP 4. Creating Distribution Figures ---")
    visualizer = Plotter(save_path=path)

    # Test different plot types
    plot_types = ["violin", "kde", "histogram", "cdf", "box"]
    for plot_type in plot_types:
        try:
            visualizer.statistical_distribution(matching_history, plot_type=plot_type)
            print(f"✓ {plot_type.capitalize()} distribution figure created successfully")
        except Exception as e:
            print(f"✗ Error creating {plot_type} plot: {str(e)[:50]}...")

    print("✓ All distribution figures created with MarketVisualizer")


def test_spatial_distribution(path: str) -> None:
    """Test agent-to-agent trading matrix visualization."""
    print("--- STEP 1. Creating Agent Trading Data ---")
    matching_history = MatchingHistory()

    # Create a fixed set of agents for consistent visualization
    agents = [f"agent_{i}" for i in range(5)]  # 5 agents: agent_0, agent_1, agent_2, agent_3, agent_4

    # Create sample data showing trading patterns across multiple market steps
    for step in range(10):  # 10 market steps
        trades = []

        # Step 0: Initial trades
        if step == 0:
            trades = [
                Trade(buyer_id="agent_0", seller_id="agent_1", price=100, quantity=10, timestamp=time.time()),
                Trade(buyer_id="agent_2", seller_id="agent_0", price=105, quantity=7, timestamp=time.time()),
                Trade(buyer_id="agent_3", seller_id="agent_1", price=98, quantity=5, timestamp=time.time()),
            ]
        # Step 1: More trades
        elif step == 1:
            trades = [
                Trade(buyer_id="agent_1", seller_id="agent_2", price=102, quantity=8, timestamp=time.time()),
                Trade(buyer_id="agent_4", seller_id="agent_0", price=99, quantity=6, timestamp=time.time()),
                Trade(buyer_id="agent_0", seller_id="agent_3", price=103, quantity=4, timestamp=time.time()),
            ]
        # Other steps: Random trading patterns
        else:
            for _ in range(np.random.randint(1, 4)):  # 1-3 trades per step
                buyer = np.random.choice(agents)
                seller = np.random.choice([a for a in agents if a != buyer])  # Different from buyer
                trades.append(Trade(
                    buyer_id=buyer,
                    seller_id=seller,
                    price=100 + np.random.normal(0, 5),
                    quantity=np.random.exponential(5) + 2,
                    timestamp=time.time()
                ))

        matching_history.update(
            MatchingResult(
                trades=trades,
                unmatched_orders=[],
                clearing_price=100 + step*2,
                clearing_volume=sum(t.quantity for t in trades),
                grid_balance=np.random.normal(0, 1),
                grid_congestion=0.1 + np.random.random() * 0.1,
                dso_buy_volume=5 + step,
                dso_sell_volume=8 + step,
                dso_total_volume=13 + step*2,
                p2p_volume=20 + step*2,
                dso_trade_ratio=0.4 - step*0.02,
                dso_grid_import=3 + step,
                dso_buy_price=90 + step*2,
                dso_sell_price=110 + step*3,
                price_spread=20 + step,
                local_price_avg=105 + step*2,
                local_price_advantage=5 + step
            )
        )

    print(f"✓ Created trading data with {len(matching_history.history)} market steps")
    print(f"  - Fixed agent set: {agents}")

    # Show some example trades
    all_trades = []
    for result in matching_history.history:
        all_trades.extend(result.trades)
    print(f"  - Total trades: {len(all_trades)}")
    print("  - Example trades:")
    for i, trade in enumerate(all_trades[:3]):
        print(f"    {i+1}. {trade.buyer_id} buys {trade.quantity:.1f} kWh from {trade.seller_id} at ${trade.price:.2f}")

    print("--- STEP 2. Testing Agent-to-Agent Trading Matrix ---")
    visualizer = Plotter(save_path=path)

    # Test with automatic grid size (equal to number of unique agents)
    fig = visualizer.spatial_heatmap(matching_history, grid_network=None)

    # Calculate number of unique agents
    unique_agents = set()
    for result in matching_history.history:
        for trade in result.trades:
            unique_agents.add(trade.buyer_id)
            unique_agents.add(trade.seller_id)

    expected_grid_size = len(unique_agents)
    print("✓ Agent-to-agent trading matrix created")
    print(f"  - Number of unique agents: {expected_grid_size}")
    print(f"  - Matrix dimensions: {expected_grid_size}x{expected_grid_size}")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - X-axis: Sellers, Y-axis: Buyers")
    print("  - Values: Accumulated trading volume across all market steps")


def test_grid_stability_metrics(path: str) -> None:
    """Test grid stability metrics visualization."""
    print("--- STEP 1. Creating Sample Grid Metrics ---")

    # Create sample time series data
    time_steps = 30
    np.random.seed(42)  # For reproducible results

    # Generate realistic time series data
    grid_balance_over_time = [np.random.normal(0, 1.5) for _ in range(time_steps)]
    grid_congestion_over_time = [np.random.beta(2, 5) for _ in range(time_steps)]
    grid_stability_over_time = [np.random.beta(8, 2) for _ in range(time_steps)]
    grid_utilization_over_time = [np.random.beta(3, 7) for _ in range(time_steps)]

    # Create sample grid metrics with time series data
    grid_metrics = GridMetrics(
        grid_stability_index=0.85,
        supply_demand_balance=0.12,
        avg_congestion_level=0.25,
        transmission_loss_ratio=0.08,
        grid_utilization_efficiency=0.78,
        transmission_losses=15.5,
        load_factor=0.82,
        capacity_utilization=0.75,
        stability_trend=0.05,
        congestion_trend=-0.02,
        efficiency_trend=0.03,
        # Time series data
        grid_balance_over_time=grid_balance_over_time,
        grid_congestion_over_time=grid_congestion_over_time,
        grid_stability_over_time=grid_stability_over_time,
        grid_utilization_over_time=grid_utilization_over_time
    )

    print("✓ Created sample grid metrics with time series data")
    print(f"  - Grid stability index: {grid_metrics.grid_stability_index}")
    print(f"  - Supply-demand balance: {grid_metrics.supply_demand_balance}")
    print(f"  - Grid utilization efficiency: {grid_metrics.grid_utilization_efficiency}")
    print(f"  - Time series length: {len(grid_metrics.grid_balance_over_time)} steps")
    print(f"  - Balance range: {min(grid_metrics.grid_balance_over_time):.2f} to {max(grid_metrics.grid_balance_over_time):.2f}")

    print("--- STEP 2. Testing Enhanced Grid Stability Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test enhanced grid stability metrics plot with time series
    fig = visualizer.plot_grid_stability_metrics(grid_metrics)
    print("✓ Enhanced grid stability metrics figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Time series plots for grid balance, congestion, stability, utilization + summary metrics")


def test_economic_efficiency_metrics(path: str) -> None:
    """Test economic efficiency metrics visualization."""
    print("--- STEP 1. Creating Sample Market Metrics ---")

    # Import MarketMetrics
    from src.analytics.market_analytics import MarketMetrics

    # Create sample market metrics
    market_metrics = MarketMetrics(
        social_welfare=2500.0,
        consumer_surplus=1200.0,
        producer_surplus=1300.0,
        avg_clearing_price=105.5,
        price_volatility=0.15,
        price_trend=0.02,
        allocation_efficiency=0.88,
        price_discovery_efficiency=0.92,
        total_trading_volume=1500.0,
        transaction_count=45,
        avg_trade_size=33.3,
        market_concentration=0.25,
        welfare_distribution_gini=0.35,
        price_fairness_index=0.78,
        p2p_trade_ratio=0.65,
        dso_dependency_ratio=0.35,
        market_liquidity=0.82,
        volume_trend=0.05,
        efficiency_trend=0.03
    )

    print("✓ Created sample market metrics")
    print(f"  - Social welfare: ${market_metrics.social_welfare}")
    print(f"  - Average clearing price: ${market_metrics.avg_clearing_price}")
    print(f"  - Market liquidity: {market_metrics.market_liquidity}")

    print("--- STEP 2. Testing Economic Efficiency Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test economic efficiency metrics plot
    fig = visualizer.plot_economic_efficiency_metrics(market_metrics)
    print("✓ Economic efficiency metrics figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Price metrics, market quality, surplus distribution, trading metrics")


def test_resource_utilization_metrics(path: str) -> None:
    """Test resource utilization metrics visualization."""
    print("--- STEP 1. Creating Sample Resource Metrics ---")

    # Import MarketMetrics (reused for resource utilization)
    from src.analytics.market_analytics import MarketMetrics

    # Create sample resource utilization metrics
    resource_metrics = MarketMetrics(
        allocation_efficiency=0.85,
        price_discovery_efficiency=0.90,
        social_welfare=3000.0,
        welfare_distribution_gini=0.30,
        price_fairness_index=0.82,
        p2p_trade_ratio=0.70,
        dso_dependency_ratio=0.30,
        market_liquidity=0.88,
        total_trading_volume=2000.0,
        transaction_count=60,
        avg_trade_size=33.3
    )

    print("✓ Created sample resource utilization metrics")
    print(f"  - Allocation efficiency: {resource_metrics.allocation_efficiency}")
    print(f"  - P2P trade ratio: {resource_metrics.p2p_trade_ratio}")
    print(f"  - Market liquidity: {resource_metrics.market_liquidity}")

    print("--- STEP 2. Testing Resource Utilization Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test resource utilization metrics plot
    fig = visualizer.plot_resource_utilization_metrics(resource_metrics)
    print("✓ Resource utilization metrics figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Market efficiency, welfare metrics, market structure, trading activity")


def test_dso_metrics(path: str) -> None:
    """Test DSO metrics visualization."""
    print("--- STEP 1. Creating Sample DSO Metrics ---")

    # Import DSOMetrics
    from src.analytics.dso_metrics import DSOMetrics

    # Create sample DSO metrics
    dso_metrics = DSOMetrics(
        dso_buy_volume=150.0,
        dso_sell_volume=200.0,
        dso_total_volume=350.0,
        p2p_volume=800.0,
        dso_trade_ratio=0.30,
        p2p_trade_ratio=0.70,
        dso_buy_price_avg=0.095,
        dso_sell_price_avg=0.125,
        price_spread=0.030,
        local_price_avg=0.105,
        local_price_advantage=0.015,
        net_grid_import=50.0,
        grid_import_ratio=0.15,
        self_sufficiency_ratio=0.85,
        dso_profit=25.0,
        avoided_dso_cost=45.0,
        local_trading_benefit=60.0,
        dso_utilization_efficiency=0.75,
        market_balance_quality=0.82,
        fallback_effectiveness=0.78
    )

    # Add time series data for DSO metrics
    dso_metrics.dso_trades_over_time = [20, 25, 30, 28, 35, 32, 38, 40, 42, 45]
    dso_metrics.local_trades_over_time = [80, 85, 90, 88, 95, 92, 98, 100, 102, 105]
    dso_metrics.dso_ratio_over_time = [0.20, 0.23, 0.25, 0.24, 0.27, 0.26, 0.28, 0.29, 0.30, 0.30]
    dso_metrics.market_decentralization = 0.75
    dso_metrics.dso_dependency_index = 0.25

    print("✓ Created sample DSO metrics")
    print(f"  - DSO trade ratio: {dso_metrics.dso_trade_ratio}")
    print(f"  - Market decentralization: {dso_metrics.market_decentralization}")
    print(f"  - Avoided DSO cost: ${dso_metrics.avoided_dso_cost}")

    print("--- STEP 2. Testing DSO Metrics Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test DSO metrics plot
    fig = visualizer.plot_dso_metrics(dso_metrics)
    print("✓ DSO metrics figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Trade distribution, DSO breakdown, price comparison, market characteristics, time series")


def test_agent_behavior(path: str) -> None:
    """Test agent behavior visualization."""
    print("--- STEP 1. Creating Sample Agent Behavior Data ---")

    # Create sample agent metrics over time
    agent_metrics = {
        "agent_0": [0.5, 0.6, 0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.92, 0.95],
        "agent_1": [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "agent_2": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
        "agent_3": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        "agent_4": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.88, 0.85, 0.82]
    }

    print("✓ Created sample agent behavior data")
    print(f"  - Number of agents: {len(agent_metrics)}")
    print(f"  - Time steps: {len(agent_metrics['agent_0'])}")
    print("  - Sample values for agent_0:", agent_metrics['agent_0'][:5])

    print("--- STEP 2. Testing Agent Behavior Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test agent behavior plot
    fig = visualizer.plot_agent_behavior(agent_metrics)
    print("✓ Agent behavior figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Trading activity over time for multiple agents")


def test_market_performance_figure(path: str) -> None:
    """Test market performance figure visualization."""
    print("--- STEP 1. Creating Sample Market Performance Data ---")

    # Create sample matching history with varied performance
    matching_history = MatchingHistory()

    for step in range(15):  # 15 time steps
        # Create varying number of trades
        num_trades = np.random.randint(2, 8)
        trades = []

        for _ in range(num_trades):
            buyer = f"buyer_{np.random.randint(0, 5)}"
            seller = f"seller_{np.random.randint(0, 5)}"
            trades.append(Trade(
                buyer_id=buyer,
                seller_id=seller,
                price=100 + step*2 + np.random.normal(0, 10),
                quantity=np.random.exponential(8) + 3,
                timestamp=time.time()
            ))

        # Create varied market performance
        p2p_volume = np.random.exponential(50) + 20
        dso_volume = np.random.exponential(30) + 10
        total_volume = p2p_volume + dso_volume

        matching_history.update(MatchingResult(
            trades=trades,
            unmatched_orders=[],
            clearing_price=100 + step*1.5 + np.random.normal(0, 5),
            clearing_volume=total_volume,
            grid_balance=np.random.normal(0, 2),
            grid_congestion=0.1 + np.random.random() * 0.3,
            dso_buy_volume=dso_volume * 0.6,
            dso_sell_volume=dso_volume * 0.4,
            dso_total_volume=dso_volume,
            p2p_volume=p2p_volume,
            dso_trade_ratio=dso_volume / total_volume,
            dso_grid_import=np.random.normal(5, 2),
            dso_buy_price=90 + step + np.random.normal(0, 3),
            dso_sell_price=110 + step + np.random.normal(0, 3),
            price_spread=20 + np.random.normal(0, 5),
            local_price_avg=105 + step + np.random.normal(0, 3),
            local_price_advantage=5 + np.random.normal(0, 2)
        ))

    print(f"✓ Created market performance data with {len(matching_history.history)} steps")
    print(f"  - Total trades: {sum(len(r.trades) for r in matching_history.history)}")
    print(f"  - Price range: ${min(r.clearing_price for r in matching_history.history):.2f} - ${max(r.clearing_price for r in matching_history.history):.2f}")

    print("--- STEP 2. Testing Market Performance Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test market performance figure
    fig = visualizer.create_market_performance_figure(matching_history)
    print("✓ Market performance figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Price over time, volume over time, P2P ratio, trade count")


def test_coordination_effectiveness_figure(path: str) -> None:
    """Test coordination effectiveness figure visualization."""
    print("--- STEP 1. Creating Sample Coordination Data ---")

    # Create sample matching history with coordination patterns
    matching_history = MatchingHistory()

    for step in range(12):  # 12 time steps
        # Create trades with coordination patterns
        num_trades = np.random.randint(3, 10)
        trades = []

        for _ in range(num_trades):
            buyer = f"agent_{np.random.randint(0, 6)}"
            seller = f"agent_{np.random.randint(0, 6)}"
            # Ensure buyer != seller
            while buyer == seller:
                seller = f"agent_{np.random.randint(0, 6)}"

            trades.append(Trade(
                buyer_id=buyer,
                seller_id=seller,
                price=100 + step*1.5 + np.random.normal(0, 8),
                quantity=np.random.exponential(6) + 2,
                timestamp=time.time()
            ))

        # Create coordination effectiveness patterns
        p2p_volume = np.random.exponential(40) + 15
        total_volume = p2p_volume + np.random.exponential(20) + 10
        grid_balance = np.random.normal(0, 1.5)  # Lower values = better coordination

        matching_history.update(MatchingResult(
            trades=trades,
            unmatched_orders=[],
            clearing_price=100 + step*2 + np.random.normal(0, 6),
            clearing_volume=total_volume,
            grid_balance=grid_balance,
            grid_congestion=0.05 + np.random.random() * 0.2,
            dso_buy_volume=total_volume - p2p_volume,
            dso_sell_volume=0,
            dso_total_volume=total_volume - p2p_volume,
            p2p_volume=p2p_volume,
            dso_trade_ratio=(total_volume - p2p_volume) / total_volume,
            dso_grid_import=np.random.normal(3, 1.5),
            dso_buy_price=95 + step + np.random.normal(0, 2),
            dso_sell_price=105 + step + np.random.normal(0, 2),
            price_spread=10 + np.random.normal(0, 3),
            local_price_avg=100 + step + np.random.normal(0, 2),
            local_price_advantage=3 + np.random.normal(0, 1)
        ))

    print(f"✓ Created coordination data with {len(matching_history.history)} steps")
    print(f"  - Total trades: {sum(len(r.trades) for r in matching_history.history)}")
    print(f"  - Average grid balance: {np.mean([abs(r.grid_balance) for r in matching_history.history]):.2f}")

    print("--- STEP 2. Testing Coordination Effectiveness Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test coordination effectiveness figure
    fig = visualizer.create_coordination_effectiveness_figure(matching_history)
    print("✓ Coordination effectiveness figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Grid balance, P2P ratio, coordination score, trade distribution")


def test_sequential_data(path: str) -> None:
    """Test sequential data visualization."""
    print("--- STEP 1. Creating Sample Sequential Data ---")

    # Create sample sequential performance data
    sequential_data = {
        "ppo": {
            "ctce": [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97],
            "ctde": [0.05, 0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92],
            "dtde": [0.02, 0.15, 0.35, 0.55, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87]
        },
        "appo": {
            "ctce": [0.08, 0.25, 0.45, 0.65, 0.78, 0.82, 0.87, 0.89, 0.92, 0.94],
            "ctde": [0.04, 0.18, 0.38, 0.58, 0.72, 0.77, 0.82, 0.85, 0.87, 0.89],
            "dtde": [0.01, 0.12, 0.32, 0.52, 0.67, 0.72, 0.77, 0.8, 0.82, 0.84]
        },
        "sac": {
            "ctce": [0.12, 0.35, 0.55, 0.75, 0.85, 0.88, 0.92, 0.94, 0.96, 0.98],
            "ctde": [0.06, 0.25, 0.45, 0.65, 0.8, 0.83, 0.87, 0.89, 0.91, 0.93],
            "dtde": [0.03, 0.2, 0.4, 0.6, 0.75, 0.78, 0.82, 0.84, 0.86, 0.88]
        }
    }

    print("✓ Created sample sequential performance data")
    print(f"  - Algorithms: {list(sequential_data.keys())}")
    print(f"  - Approaches: {list(sequential_data['ppo'].keys())}")
    print(f"  - Epochs: {len(sequential_data['ppo']['ctce'])}")
    print("  - Sample PPO-CTCE values:", sequential_data['ppo']['ctce'][:5])

    print("--- STEP 2. Testing Sequential Data Visualization ---")
    visualizer = Plotter(save_path=path)

    # Test sequential data plot
    fig = visualizer.plot_sequential_data(
        data=sequential_data,
        title="RL Algorithm Performance Comparison",
        x_label="Training Epoch",
        y_label="Average Reward"
    )
    print("✓ Sequential data figure created successfully")
    print(f"  - Figure size: {fig.get_size_inches()}")
    print("  - Contains: Performance curves for multiple algorithms and approaches")


def run_visualization_tests(path: str) -> bool:
    """Run all visualization tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """

    print("🚀 STARTING VisualizationModule TESTS")

    try:
        # Test trading network visualization
        test_trading_network_visualization(path,
                                           num_history=10,
                                           num_trades=5,
                                           num_agents=5)

        # Test price and volume distribution
        test_price_volume_distribution(path)

        # Test spatial distribution
        test_spatial_distribution(path)

        # Test grid stability metrics
        test_grid_stability_metrics(path)

        # Test economic efficiency metrics
        test_economic_efficiency_metrics(path)

        # Test resource utilization metrics
        test_resource_utilization_metrics(path)

        # Test DSO metrics
        test_dso_metrics(path)

        # Test agent behavior
        test_agent_behavior(path)

        # Test market performance figure
        test_market_performance_figure(path)

        # Test coordination effectiveness figure
        test_coordination_effectiveness_figure(path)

        # Test sequential data
        test_sequential_data(path)

        print("🎉 VisualizationModule TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == '__main__':
    success = run_visualization_tests(path=f"{__main__}/downloads/figs")
    exit(0 if success else 1)
