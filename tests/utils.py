"""
Shared environment configuration and utilities for testing.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.battery import Battery
from src.agent.der import DERAgent
from src.grid.base import GridTopology
from src.grid.network import GridNetwork
from src.market.dso import DSOAgent
from src.market.matching import ClearingMechanism, MarketConfig
from src.profile.der import DERProfileHandler
from src.profile.dso import DSOProfileHandler


def create_market_config(min_price=0.05,
                         max_price=0.50,
                         min_quantity=1.0,
                         max_quantity=100.0,
                         price_mechanism=ClearingMechanism.AVERAGE,
                         blockchain_difficulty=2,
                         visualize_blockchain=False,
                         enable_partner_preference=True) -> MarketConfig:
    """Create a test market configuration with customizable parameters.

    Args:
        min_price: Minimum price ($/Wh)
        max_price: Maximum price ($/Wh)
        min_quantity: Minimum quantity (Wh)
        max_quantity: Maximum quantity (Wh)
        price_mechanism: Price mechanism
        blockchain_difficulty: Blockchain difficulty
        visualize_blockchain: Visualize blockchain
        enable_partner_preference: Enable partner preference

    Returns:
        MarketConfig: Market configuration
    """
    return MarketConfig(min_price=min_price,
                        max_price=max_price,
                        min_quantity=min_quantity,
                        max_quantity=max_quantity,
                        price_mechanism=price_mechanism,
                        blockchain_difficulty=blockchain_difficulty,
                        visualize_blockchain=visualize_blockchain,
                        enable_partner_preference=enable_partner_preference)


def create_agents(num_agents=3,
                  capacity=100.0,
                  battery_capacity=50.0,
                  min_soc=0.1,
                  max_soc=0.9,
                  node_ids=None,
                  seed=42) -> list[DERAgent]:
    """Create test DER agents with batteries and profiles.

    Args:
        num_agents: Number of agents
        capacity: Capacity (kW)
        battery_capacity: Battery capacity (Wh)
        min_soc: Minimum SOC
        max_soc: Maximum SOC
        node_ids: Node IDs
        seed: Seed

    Returns:
        List[DERAgent]: List of DER agents
    """
    agents = []

    for i in range(num_agents):
        battery = Battery(nominal_capacity=battery_capacity,
                          min_soc=min_soc,
                          max_soc=max_soc)

        # Create realistic generation and demand profiles
        generation_profile = [max(1.0, np.random.uniform(20, 80)) for _ in range(24)]
        demand_profile = [max(1.0, np.random.uniform(10, 60)) for _ in range(24)]

        agent_kwargs = {"id": f"agent_{i}",
                        "capacity": capacity,
                        "battery": battery,
                        "generation_profile": generation_profile,
                        "demand_profile": demand_profile}

        # Add node_id if provided
        if node_ids and i < len(node_ids):
            agent_kwargs["node_id"] = node_ids[i]

        agent = DERAgent(**agent_kwargs)
        agents.append(agent)

    return agents


def create_dso_agent(agent_id="DSO",
                     feed_in_tariff=0.08,
                     utility_price=0.25,
                     grid_network=None,
                     num_hours=24) -> DSOAgent:
    """Create a test DSO agent.

    Args:
        agent_id: Agent ID
        feed_in_tariff: Feed-in tariff
        utility_price: Utility price
        grid_network: Grid network
        num_hours: Number of hours

    Returns:
        DSOAgent: DSO agent
    """
    return DSOAgent(id=agent_id,
                    feed_in_tariff=[feed_in_tariff] * num_hours,
                    utility_price=[utility_price] * num_hours,
                    grid_network=grid_network)


def create_grid_network(topology=GridTopology.IEEE34) -> GridNetwork:
    """Create a test grid network.

    Args:
        topology: Grid topology

    Returns:
        GridNetwork: Grid network
    """
    return GridNetwork(topology=topology)


def create_env_config(seed=42,
                      max_steps=24,
                      num_agents=3,
                      market_config=None,
                      grid_network=None) -> dict:
    """Create a comprehensive test environment configuration.

    Args:
        seed: Seed
        max_steps: Number of steps
        num_agents: Number of agents
        market_config: Market configuration
        grid_network: Grid network

    Returns:
        Environment configuration dictionary
    """
    if market_config is None:
        market_config = create_market_config()

    if grid_network is None:
        grid_network = create_grid_network()

    agents = create_agents(num_agents=num_agents, seed=seed)

    env_config = {"seed": seed,
                  "max_steps": max_steps,
                  "agents": agents,
                  "market_config": market_config,
                  "grid_network": grid_network,
                  "der_profile_handler": DERProfileHandler(min_quantity=market_config.min_quantity,
                                                           max_quantity=market_config.max_quantity,
                                                           seed=seed),
                  "dso_profile_handler": DSOProfileHandler(min_price=market_config.min_price,
                                                           max_price=market_config.max_price,
                                                           seed=seed),
                  "enable_asynchronous_order": False,
                  "enable_reset_dso_profiles": False,
                  "max_error": 0.3,
                  "num_anchor": 4}

    return env_config


def create_simple_env_config() -> dict:
    """Create a simple test environment configuration for basic tests.

    Returns:
        Environment configuration dictionary
    """
    return create_env_config(seed=42,
                             max_steps=24,
                             num_agents=3,
                             market_config=create_market_config(min_price=0.0,
                                                                max_price=100.0,
                                                                min_quantity=0.0,
                                                                max_quantity=100.0))
