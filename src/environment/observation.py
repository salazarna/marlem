"""
Observation handling for the Local Energy Market environment.

This module implements the partial observability aspects of the Dec-POMDP,
managing what information is available to each agent based on their local
perspective and limited knowledge of the market state.
"""

from typing import Dict, List

import numpy as np
from gymnasium import spaces

from ..agent.der import DERAgent
from ..coordination.implicit_cooperation import ImplicitCooperation
from ..grid.network import GridNetwork
from ..market.dso import DSOAgent
from ..market.matching import MarketConfig, MatchingHistory
from ..market.order import Order
from .base import RLProtection


class ObservationHandler:
    """Handles partial observability in the Dec-POMDP environment.

    This class manages what information is available to each agent, implementing
    the partial observability aspect of the Dec-POMDP framework. This is critical
    is to model realistic information constraints in decentralized energy markets,
    where agents only have access to local information and limited market signals.
    """

    def __init__(self,
                 max_steps: int,
                 agents: List[DERAgent],
                 dso: DSOAgent,
                 market_config: MarketConfig,
                 grid_network: GridNetwork) -> None:
        """Initialize the observation handler.

        Args:
            agents: List of DER agents in the market
            max_steps: Maximum number of steps in the market
            market_config: Market configuration parameters
            grid_network: Grid network for zone-specific calculations
        """
        self.max_steps = max_steps  # Maximum number of steps in the market
        self.agents = agents  # List of DER agents in the market
        self.dso = dso  # DSO agent
        self.market_config = market_config  # Market configuration parameters
        self.grid_network = grid_network  # Grid network for zone-specific calculations

        # Check initialization
        self._check_init()

        # Define the observation space for each agent
        self.observation_space = spaces.Dict({})

        # Safe calculation of high values to prevent overflow
        _max_quantity = RLProtection.clip_value(value=market_config.max_quantity * len(self.agents), min_value=0.0)
        _max_dso_quantity = RLProtection.clip_value(value=_max_quantity * 2, min_value=0.0)
        _max_demand = RLProtection.clip_value(value=self.max_steps * market_config.max_quantity, min_value=0.0)
        _max_social_welfare = RLProtection.clip_value(value=market_config.max_price * market_config.max_quantity * len(self.agents), min_value=0.0)

        for agent in self.agents:
            low = np.array([
                # Market signals
                0.0,  # current_step
                0.0,  # time_of_day
                market_config.min_price,  # clearing_price
                market_config.min_quantity,  # clearing_volume
                -grid_network.capacity,  # grid_balance
                0.0,  # dso_buy_volume
                0.0,  # dso_sell_volume
                0.0,  # dso_total_volume
                0.0,  # p2p_volume
                0.0,  # dso_trade_ratio
                -grid_network.capacity,  # net_grid_import
                min(dso.feed_in_tariff),  # dso_buy_price
                min(dso.utility_price),  # dso_sell_price
                market_config.min_price,  # local_price_avg
                0.0,  # price_spread
                -market_config.max_price,  # local_price_advantage

                # Agent signals
                0.0,  # energy_generation
                0.0,  # energy_demand
                0.0,  # cumulative_demand_satisfied
                0.0,  # cumulative_demand_deferred
                0.0,  # remaining_demand
                0.0,  # cumulative_supply_satisfied
                0.0,  # cumulative_supply_deferred
                0.0,  # remaining_supply
                -RLProtection.MAX_SAFE_VALUE.value,  # average_profit
                0.0,  # reputation
                agent.battery.min_soc * agent.battery.nominal_capacity if agent.battery else 0.0,  # battery_energy_level
                agent.battery.min_soc if agent.battery else 0.0,  # battery_soc
                0.0,  # battery_available_charge
                0.0,  # battery_available_discharge
                0.0,  # battery_cumulative_charge
                0.0,  # battery_cumulative_discharge

                # Implicit cooperation KPIs - Economic Efficiency
                0.0,  # social_welfare (sum of price * quantity, always >= 0)
                0.0,  # market_liquidity (total trading volume, always >= 0)
                -market_config.max_price,  # avg_bid_ask_spread (can be negative if bids > asks)
                0.0,  # price_volatility (standard deviation, always >= 0)
                0.0,  # supply_demand_imbalance (normalized ratio)
                0.0,  # grid_congestion (normalized ratio)
                0.0,  # coordination_score (normalized score)
                0.0,  # coordination_convergence
                0.0,  # der_self_consumption
                0.0   # flexibility_utilization
            ])

            # Safe calculation of high values to prevent overflow (agent-specific)
            _max_supply = RLProtection.clip_value(value=self.max_steps * agent.capacity, min_value=0.0)
            _battery_max_energy = RLProtection.clip_value(value=agent.battery.max_soc * agent.battery.nominal_capacity if agent.battery else 0.0, min_value=0.0)
            _battery_max_cumulative = RLProtection.clip_value(value=self.max_steps * _battery_max_energy, min_value=0.0)

            high = np.array([
                # Market signals
                self.max_steps,  # current_step
                1.0,  # time_of_day
                market_config.max_price,  # clearing_price
                _max_quantity,  # clearing_volume (can be sum of all agents)
                grid_network.capacity,  # grid_balance
                _max_quantity,  # dso_buy_volume (can be sum of all agents)
                _max_quantity,  # dso_sell_volume (can be sum of all agents)
                _max_dso_quantity,  # dso_total_volume (buy + sell)
                _max_quantity,  # p2p_volume (can be sum of all agents)
                1.0,  # dso_trade_ratio
                grid_network.capacity,  # net_grid_import
                max(dso.feed_in_tariff),  # dso_buy_price
                max(dso.utility_price),  # dso_sell_price
                market_config.max_price,  # local_price_avg
                market_config.max_price,  # price_spread
                market_config.max_price,  # local_price_advantage

                # Agent signals
                agent.capacity,  # energy_generation
                max(agent.demand_profile) if len(agent.demand_profile) > 0 else market_config.max_quantity,  # energy_demand
                sum(agent.demand_profile) if len(agent.demand_profile) > 0 else _max_demand,  # cumulative_demand_satisfied
                sum(agent.demand_profile) if len(agent.demand_profile) > 0 else _max_demand,  # cumulative_demand_deferred
                sum(agent.demand_profile) if len(agent.demand_profile) > 0 else _max_demand,  # remaining_demand
                sum(agent.generation_profile) if len(agent.generation_profile) > 0 else _max_supply,  # cumulative_supply_satisfied
                sum(agent.generation_profile) if len(agent.generation_profile) > 0 else _max_supply,  # cumulative_supply_deferred
                sum(agent.generation_profile) if len(agent.generation_profile) > 0 else _max_supply,  # remaining_supply
                RLProtection.MAX_SAFE_VALUE.value,  # average_profit
                1.0,  # reputation
                _battery_max_energy,  # battery_energy_level
                agent.battery.max_soc if agent.battery else 0.0,  # battery_soc
                _battery_max_energy,  # battery_available_charge
                _battery_max_energy,  # battery_available_discharge
                _battery_max_cumulative,  # battery_cumulative_charge
                _battery_max_cumulative,  # battery_cumulative_discharge

                # Implicit cooperation KPIs - Economic Efficiency
                _max_social_welfare,  # social_welfare (max price * max total volume)
                _max_quantity,  # market_liquidity (max total trading volume)
                market_config.max_price,  # avg_bid_ask_spread (max possible spread)
                market_config.max_price,  # price_volatility (max possible volatility)
                1.0,  # supply_demand_imbalance (normalized ratio)
                1.0,  # grid_congestion (normalized ratio)
                1.0,  # coordination_score (normalized score)
                1.0,  # coordination_convergence
                1.0,  # der_self_consumption
                1.0   # flexibility_utilization
            ])

            # If partner preference is enabled, add partner reputation to observation space
            if self.market_config.enable_partner_preference:
                low = np.concatenate([low, np.zeros(len(self.agents))])
                high = np.concatenate([high, np.ones(len(self.agents))])

            # Add to observation space
            self.observation_space[agent.id] = spaces.Box(low=low, high=high, dtype=np.float32)

    def _check_init(self) -> None:
        """Check if the observation handler is initialized correctly."""
        if self.max_steps <= 0:
            raise ValueError(f"Max steps must be greater than 0. Current value: <max_steps = {self.max_steps}>.")

        if self.agents is None:
            raise ValueError(f"Agents must be provided in the config. Current value: <agents = {self.agents}>.")

        if self.dso is None:
            raise ValueError(f"DSO agent must be provided in the config. Current value: <dso = {self.dso}>.")

        if self.market_config is None:
            raise ValueError(f"Market configuration must be provided in the config. Current value: <market_config = {self.market_config}>.")

        if self.grid_network is None:
            raise ValueError(f"Grid network must be provided in the config. Current value: <grid_network = {self.grid_network}>.")

    def reset_observation_space(self) -> Dict[str, np.array]:
        """Create initial observation space."""
        obs = {}

        # Ensure consistent ordering by iterating in agent order
        for agent in self.agents:
            battery_status = agent.battery.get_state() if agent.battery else self._battery_null_state()

            obs[agent.id] = np.array([
                # Market signals
                0.0,  # current_step
                0.0,  # time_of_day
                self.market_config.min_price,  # clearing_price
                self.market_config.min_quantity,  # clearing_volume
                0.0,  # grid_balance
                0.0,  # dso_buy_volume
                0.0,  # dso_sell_volume
                0.0,  # dso_total_volume
                0.0,  # p2p_volume
                0.0,  # dso_trade_ratio
                0.0,  # net_grid_import
                self.dso.feed_in_tariff[0],  # dso_buy_price
                self.dso.utility_price[0],  # dso_sell_price
                0.0,  # local_price_avg
                0.0,  # price_spread
                0.0,  # local_price_advantage

                # Agent signals
                0.0,  # energy_generation
                0.0,  # energy_demand
                0.0,  # cumulative_demand_satisfied
                0.0,  # cumulative_demand_deferred
                0.0,  # remaining_demand
                0.0,  # cumulative_supply_satisfied
                0.0,  # cumulative_supply_deferred
                0.0,  # remaining_supply
                0.0,  # average_profit
                agent.reputation,  # reputation
                battery_status["energy_level"],  # battery_energy_level
                battery_status["soc"],  # battery_soc
                battery_status["available_charge"],  # battery_available_charge
                battery_status["available_discharge"],  # battery_available_discharge
                0.0,  # battery_cumulative_charge
                0.0,  # battery_cumulative_discharge

                # Implicit cooperation KPIs - Economic Efficiency
                0.0,  # social_welfare (sum of price * quantity, always >= 0)
                0.0,  # market_liquidity (total trading volume, always >= 0)
                0.0,  # avg_bid_ask_spread (can be negative if bids > asks)
                0.0,  # price_volatility (standard deviation, always >= 0)
                0.0,  # supply_demand_imbalance (normalized ratio)
                0.0,  # grid_congestion (normalized ratio)
                0.0,  # coordination_score (normalized score)
                0.0,  # coordination_convergence
                0.0,  # der_self_consumption
                0.0,  # flexibility_utilization
            ], dtype=np.float32)

            if self.market_config.enable_partner_preference:
                agents_reputation = np.array([other_agent.reputation for other_agent in self.agents], dtype=np.float32)
                obs[agent.id] = np.concatenate([obs[agent.id], agents_reputation])

        return obs

    def update_observation_space(self,
                                 current_step: int,
                                 time_of_day: float,
                                 orders: List[Order],
                                 matching_history: MatchingHistory,
                                 implicit_cooperation: ImplicitCooperation) -> Dict[str, np.array]:
        """Update the observation space with the latest environment state.

        Args:
            current_step: Current time step
            time_of_day: Current time of day
            orders: Orders submitted in this step (matched + unmatched)
            matching_history: Matching history
            implicit_cooperation: Implicit cooperation system

        Returns:
            Dict[str, np.array]: Updated observation space
        """
        matching = matching_history.history[-1]

        # Market signals
        market_signals = np.array([
            current_step,
            time_of_day,
            matching.clearing_price,
            matching.clearing_volume,
            matching.grid_balance,
            matching.dso_buy_volume,
            matching.dso_sell_volume,
            matching.dso_total_volume,
            matching.p2p_volume,
            matching.dso_trade_ratio,
            matching.dso_grid_import,
            matching.dso_buy_price,
            matching.dso_sell_price,
            matching.local_price_avg,
            matching.price_spread,
            matching.local_price_advantage
        ], dtype=np.float32)

        # Implicit cooperation KPIs
        kpis = implicit_cooperation.get_kpis(current_step,
                                             self.dso.id,
                                             self.agents,
                                             orders,
                                             matching.trades,
                                             self.dso.congestion_level,
                                             matching_history)

        implicit_cooperation_kpis = np.array([
            # Economic Efficiency KPIs
            kpis["social_welfare"],
            kpis["market_liquidity"],
            kpis["avg_bid_ask_spread"],
            kpis["price_volatility"],

            # Grid Stability KPIs
            kpis["supply_demand_imbalance"],
            kpis["grid_congestion"],

            # Coordination Effectiveness KPIs
            kpis["coordination_score"],
            kpis["coordination_convergence"],

            # Resource Coordination KPIs
            kpis["der_self_consumption"],
            kpis["flexibility_utilization"]
        ], dtype=np.float32)

        # Agent signals
        obs = {}
        for agent in self.agents:
            battery_status = agent.battery.get_state() if agent.battery else self._battery_null_state()

            agent_signals = np.array([
                agent.get_generation(current_step),
                agent.get_demand(current_step),
                agent.cumulative_demand_satisfied,
                agent.cumulative_demand_deferred,
                agent.get_remaining_demand(current_step),
                agent.cumulative_supply_satisfied,
                agent.cumulative_supply_deferred,
                agent.get_remaining_supply(current_step),
                agent.profit,
                agent.reputation,
                battery_status["energy_level"],
                battery_status["soc"],
                battery_status["available_charge"],
                battery_status["available_discharge"],
                battery_status["cumulative_charge"],
                battery_status["cumulative_discharge"]
            ], dtype=np.float32)

            # Add to observation space
            obs[agent.id] = np.concatenate([market_signals, agent_signals, implicit_cooperation_kpis])

            if self.market_config.enable_partner_preference:
                agents_reputation = np.array([other_agent.reputation for other_agent in self.agents], dtype=np.float32)
                obs[agent.id] = np.concatenate([obs[agent.id], agents_reputation])

        return obs

    @staticmethod
    def _battery_null_state() -> Dict:
        """Get battery state when battery is not installed.

        Returns:
            Dictionary containing battery state information
        """
        return {"energy_level": 0.0,
                "soc": 0.0,
                "available_charge": 0.0,
                "available_discharge": 0.0,
                "cumulative_charge": 0.0,
                "cumulative_discharge": 0.0}
