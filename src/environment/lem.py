"""
Local Energy Market (LEM) environment implementation using Gymnasium.

This environment implements a decentralized marketplace where agents can submit buy/sell
offers for energy trading. The environment handles:
- Non-stationary market dynamics
- Continuous action spaces for bid/ask prices and quantities
- Partial observability of market state
- Decentralized market clearing
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from ..agent.der import DERAgent
from ..coordination.implicit_cooperation import ImplicitCooperation
from ..grid.base import GridTopology
from ..grid.network import GridNetwork
from ..market.dso import DSOAgent
from ..market.matching import MarketConfig, OrderMatcher
from ..market.order import Order
from ..market.reputation import ReputationHandler
from ..profile.der import DERProfileHandler
from ..profile.dso import DSOProfileHandler
from .action import ActionHandler
from .observation import ObservationHandler
from .reward import RewardHandler


class LocalEnergyMarket(MultiAgentEnv):
    """
    Local Energy Market environment implementing the MultiAgentEnv interface.

    This environment implements a decentralized marketplace where agents can trade
    energy through buy/sell offers, following a Dec-POMDP framework. The agents have
    partial observability of the market state and make decisions based on their own
    local observations.
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, env_config: Dict[str, Any]) -> None:
        """Initialize the environment.

        Args:
            config: RLlib env_config dict.
        """
        super().__init__()

        # Check environment configuration
        self._check_env_config(env_config)

        # Environment configuration
        self.max_steps: int = env_config["max_steps"]
        self.market_config: MarketConfig = env_config["market_config"]
        self.grid_network: GridNetwork = env_config.get("grid_network", None)
        self.dso: DSOAgent = env_config.get("dso", None)
        self.der_profile_handler: DERProfileHandler = env_config.get("der_profile_handler", None)
        self.dso_profile_handler: DSOProfileHandler = env_config.get("dso_profile_handler", None)
        self.enable_reset_dso_profiles: bool = env_config.get("enable_reset_dso_profiles", False)
        self.enable_asynchronous_order: bool = env_config.get("enable_asynchronous_order", True)
        self.max_error: float = env_config.get("max_error", 0.3)
        self.num_anchor: int = env_config.get("num_anchor", 4)
        self.seed: Optional[int] = env_config.get("seed", None)

        # Agents configuration
        _config_agents = env_config["agents"]
        self._agents: Dict[str, DERAgent] = {agent.id: agent for agent in _config_agents}
        self.agents_id: List[str] = [*self._agents.keys()]
        self.agents = self.possible_agents = set(self.agents_id)

        # Check environment configuration
        self._set_default_values()

        # Check initialization
        self._check_init()

        # Assign agents to grid network
        self.grid_network.assign_agents_to_graph(_config_agents)

        # Environment attributes
        self.render_mode: Optional[str] = None
        self.rewards: MultiAgentDict = {id: [] for id in self.agents_id}

        # Initialize environment state
        self.current_step: int = 0
        self.time_of_day: float = 0.0

        # Initialize market
        self.market = OrderMatcher(len(self.agents_id),
                                   self.market_config,
                                   self.grid_network,
                                   self.dso)

        # Initialize reputation system
        self.reputation_handler = ReputationHandler()

        # Initialize implicit cooperation model
        self.implicit_cooperation = ImplicitCooperation(self.grid_network.capacity)

        # Initialize observation handler with enhanced features
        self.observation_handler = ObservationHandler(self.max_steps,
                                                      _config_agents,
                                                      self.dso,
                                                      self.market_config,
                                                      self.grid_network)
        self.observation_spaces: spaces.Dict[str, spaces.Box] = self.observation_handler.observation_space

        # Initialize action handler
        self.action_handler = ActionHandler(_config_agents,
                                            self.dso,
                                            self.market_config)
        self.action_spaces: spaces.Dict[str, spaces.Box] = self.action_handler.action_space

        # Initialize reward handler
        self.reward_handler = RewardHandler(self.dso,
                                            self.grid_network)

    @staticmethod
    def _check_env_config(env_config: Dict[str, Any]) -> None:
        """Check environment configuration.

        Args:
            env_config: Environment configuration dictionary.

        Raises:
            ValueError: If required attributes are not provided in the <env_config> dictionary.
            ValueError: If agents is not a list of DERAgent objects.
            ValueError: If max_steps is not an integer.
            ValueError: If market_config is not a MarketConfig object.
            ValueError: If grid_network is not a GridNetwork object.
            ValueError: If dso is not a DSOAgent object.
            ValueError: If der_profile_handler is not a DERProfileHandler object.
            ValueError: If dso_profile_handler is not a DSOProfileHandler object.
        """
        # Check required attributes
        for key in ["max_steps", "market_config", "agents"]:
            if key not in env_config.keys():
                raise ValueError(f"The <env_config> is missing the required key: <key = {key}>.")

        # Check agents configuration
        if env_config.get("agents") is None:
            raise ValueError("Agents must be provided in the config.")

        # Validate agents
        if not isinstance(env_config.get("agents"), list) or len(env_config.get("agents")) == 0:
            raise ValueError(f"The <agents> must be a non-empty list, got <agents = {env_config.get("agents")}>.")

        # Check for duplicate agent IDs
        if len(env_config.get("agents")) != len(set(agent.id for agent in env_config.get("agents"))):
            raise ValueError("Duplicate agent IDs found in agents config.")

        # Check max_steps
        if not isinstance(env_config.get("max_steps"), int) or env_config.get("max_steps") <= 0:
            raise ValueError(f"The <max_steps> must be a positive integer, got <max_steps = {env_config.get("max_steps")}>.")

        # Check market_config
        if not isinstance(env_config.get("market_config"), MarketConfig):
            raise ValueError(f"The attribute <market_config> must be a MarketConfig object. Got <market_config = {env_config.get("market_config")}>.")

        # Check grid_network
        if (env_config.get("grid_network") is not None) and (not isinstance(env_config.get("grid_network"), GridNetwork)):
            raise ValueError(f"The attribute <grid_network> must be a GridNetwork object or None. Got <grid_network = {env_config.get("grid_network")}>.")

        # Check dso
        if (env_config.get("dso") is not None) and (not isinstance(env_config.get("dso"), DSOAgent)):
            raise ValueError(f"The attribute <dso> must be a DSOAgent object or None. Got <dso = {env_config.get("dso")}>.")

        # Check der_profile_handler
        if (env_config.get("der_profile_handler") is not None) and (not isinstance(env_config.get("der_profile_handler"), DERProfileHandler)):
            raise ValueError(f"The attribute <der_profile_handler> must be a DERProfileHandler object or None. Got <der_profile_handler = {env_config.get("der_profile_handler")}>.")

        # Check dso_profile_handler
        if (env_config.get("dso_profile_handler") is not None) and (not isinstance(env_config.get("dso_profile_handler"), DSOProfileHandler)):
            raise ValueError(f"The attribute <dso_profile_handler> must be a DSOProfileHandler object or None. Got <dso_profile_handler = {env_config.get("dso_profile_handler")}>.")

    def _set_default_values(self) -> None:
        """Check environment configuration and set default values if not provided."""
        # Set default values if not provided
        if self.grid_network is None:
            self.grid_network = GridNetwork(GridTopology.IEEE34,
                                            len(self.agents),
                                            self.market_config.max_quantity * len(self.agents),
                                            seed=self.seed)

        # Set DER profile handler
        if self.der_profile_handler is None:
            self.der_profile_handler = DERProfileHandler(self.market_config.min_quantity,
                                                         self.market_config.max_quantity,
                                                         seed=self.seed)

        # Set DSO profile handler
        if self.dso_profile_handler is None:
            self.dso_profile_handler = DSOProfileHandler(self.market_config.min_price,
                                                         self.market_config.max_price,
                                                         seed=self.seed)

        # Set DSO agent
        if self.dso is None:
            feed_in_tariff, utility_price = self.dso_profile_handler.get_price_profiles(self.max_steps)
            self.dso = DSOAgent("DSO",
                                feed_in_tariff,
                                utility_price,
                                self.grid_network)

    def _check_init(self) -> None:
        """ Validate DERAgent and DSOAgent parameters against market constraints.

        Raises:
            ValueError: If generation_profile or demand_profile have different lengths
            ValueError: If generation_profile length is not equal to max_steps
            ValueError: If generation_profile or demand_profile values are outside market quantity bounds
            ValueError: If feed_in_tariff or utility_price values are outside market price bounds
        """
        # Check DERAgent parameters
        for agent in self._agents.values():
            # Check profile lengths
            if len(agent.generation_profile) != len(agent.demand_profile):
                raise ValueError(f"Length mismatch: <generation_profile> ({len(agent.generation_profile)}), <demand_profile> ({len(agent.demand_profile)}).")
            if len(agent.generation_profile) != self.max_steps:
                raise ValueError(f"Length mismatch: <generation_profile> ({len(agent.generation_profile)}), <max_steps> ({self.max_steps}).")

            # Check profile values only if profiles are not empty
            if agent.generation_profile:
                if min(agent.generation_profile) < self.market_config.min_quantity:
                    raise ValueError(f"The <generation_profile> of agent <{agent.id}> is below market <min_quantity> ({self.market_config.min_quantity}).")
                if max(agent.generation_profile) > self.market_config.max_quantity:
                    raise ValueError(f"The <generation_profile> of agent <{agent.id}> exceeds market <max_quantity> ({self.market_config.max_quantity}).")

            if agent.demand_profile:
                if min(agent.demand_profile) < self.market_config.min_quantity:
                    raise ValueError(f"The <demand_profile> of agent <{agent.id}> is below market <min_quantity> ({self.market_config.min_quantity}).")
                if max(agent.demand_profile) > self.market_config.max_quantity:
                    raise ValueError(f"The <demand_profile> of agent <{agent.id}> exceeds market <max_quantity> ({self.market_config.max_quantity}).")

        # Check DSOAgent parameters
        if len(self.dso.feed_in_tariff) != self.max_steps:
            raise ValueError(f"Length mismatch: <feed_in_tariff> ({len(self.dso.feed_in_tariff)}), <max_steps> ({self.max_steps}).")
        if len(self.dso.utility_price) != self.max_steps:
            raise ValueError(f"Length mismatch: <utility_price> ({len(self.dso.utility_price)}), <max_steps> ({self.max_steps}).")

        # Check pricing profile values only if profiles are not empty
        if self.dso.feed_in_tariff:
            if min(self.dso.feed_in_tariff) < self.market_config.min_price:
                raise ValueError(f"DSO <feed_in_tariff> is below market <min_price> ({self.market_config.min_price}).")
            if max(self.dso.feed_in_tariff) > self.market_config.max_price:
                raise ValueError(f"DSO <feed_in_tariff> exceeds market <max_price> ({self.market_config.max_price}).")

        if self.dso.utility_price:
            if min(self.dso.utility_price) < self.market_config.min_price:
                raise ValueError(f"DSO <utility_price> is below market <min_price> ({self.market_config.min_price}).")
            if max(self.dso.utility_price) > self.market_config.max_price:
                raise ValueError(f"DSO <utility_price> exceeds market <max_price> ({self.market_config.max_price}).")

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (initial observations dict, info dictionary)
        """
        # Use provided seed or fallback to environment seed
        seed = seed if seed is not None else self.seed
        if seed is not None:
            np.random.seed(seed)

        # Reset environment state
        self.current_step = 0
        self.time_of_day = 0.0

        # Reset agents
        for agent in self._agents.values():
            # Reset agent's reputation
            reputation = self.reputation_handler.reset(seed)

            # Reset agent's profiles
            generation, demand = self.der_profile_handler.reset(agent.capacity,
                                                                self.max_steps)

            # Reset agent
            agent.reset(reputation,
                        generation,
                        demand,
                        seed)

        # Reset market
        self.market.reset(seed)

        # Reset DSO profiles
        if self.enable_reset_dso_profiles:
            self.dso.feed_in_tariff, self.dso.utility_price = self.dso_profile_handler.reset(self.max_steps)

        # Reset grid network
        self.grid_network.reset(seed)

        # Reset environment attributes
        self.rewards = {id: [] for id in self.agents_id}

        # Reset environment parameters
        observations = self.observation_handler.reset_observation_space()
        infos = {agent_id: {} for agent_id in self.agents_id}

        return observations, infos

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Execute one time step in the environment.

        This method handles:
        1. Processing agent actions into orders
        2. Matching orders and executing trades
        3. Tracking market state (prices, volumes, trades)
        4. Tracking grid state (global balance and zone-specific metrics)
        5. Computing rewards for each agent
        6. Providing observations back to agents

        The implementation maintains detailed zone-specific grid metrics when
        grid_network is enabled, providing richer state representation for the
        Dec-POMDP framework.

        Args:
            action_dict: Dictionary of actions for each agent

        Returns:
            Tuple of (observations dict, rewards dict, terminateds dict, truncateds dict, infos dict)
        """
        # Adjust and validate actions for each agent in consistent order
        for id in self.agents_id:
            action = action_dict[id]
            action_dict[id] = self._agents[id].adjust_action_for_battery(self.current_step,
                                                                         action,
                                                                         self.market_config)

            if not self.action_handler.is_valid_action(id, action_dict[id]):
                raise ValueError(f"Invalid action for {id}: {action_dict[id]}")

        # Initialize observation, reward, terminated, truncated, and info dictionaries
        observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        # STEP 1. Format actions into orders
        orders: MultiAgentDict = {}

        for id in self.agents_id:
            # Extract action components
            bid_price, bid_quantity, is_buy, preferred_partner = action_dict[id]

            # Constrain values to valid range
            bid_price = max(self.market_config.min_price, min(self.market_config.max_price, bid_price))
            bid_quantity = max(self.market_config.min_quantity, min(self.market_config.max_quantity, bid_quantity))
            is_buy = bool(is_buy > 0.5)
            preferred_partner = self.action_handler.get_partner_id(int(preferred_partner))

            # Fall back to rule-based selection if disabled RL-based selection
            if not self.market_config.enable_partner_preference:
                preferred_partner = self._agents[id].select_preferred_partner([*self._agents.values()],
                                                                              self.grid_network.graph,
                                                                              randomize=True)

            # Simulate asynchronous order arrival
            if self.enable_asynchronous_order:
                order_time = self.time_of_day + np.random.uniform(0, 1.0 / self.max_steps)
            else:
                order_time = self.time_of_day

            # Create order
            orders[id] = Order(f"{id}_{self.current_step}_{order_time}",
                               id,
                               bid_price,
                               bid_quantity,
                               is_buy,
                               order_time,
                               self._agents[id].location,
                               preferred_partner)

        # STEP 2. Execute orders and clear market
        reputation_scores = {agent.id: agent.reputation for agent in self._agents.values()}

        matching_result = self.market.match_orders(self.current_step,
                                                   [*orders.values()],
                                                   reputation_scores,
                                                   self.dso.balance)

        # Update energy tracking and battery states based on market outcomes
        for id in self.agents_id:
            # Calculate energy bought and sold for this agent
            energy_bought = 0.0
            energy_sold = 0.0

            for trade in matching_result.trades:
                if trade.buyer_id == id:
                    energy_bought += trade.quantity
                elif trade.seller_id == id:
                    energy_sold += trade.quantity

            # Update cumulative tracking based on market results
            self._agents[id].update_energy_tracking(self.current_step,
                                                    energy_bought,
                                                    energy_sold)

            # Update battery state based on actual trade outcomes
            self._agents[id].update_battery_from_trades(self.current_step,
                                                        energy_bought,
                                                        energy_sold)

        # STEP 3. Rewards
        # Get coordination KPIs for reward calculation and scientific reporting
        kpis = self.implicit_cooperation.get_kpis(self.current_step,
                                                  self.dso.id,
                                                  [*self._agents.values()],
                                                  [*orders.values()],
                                                  matching_result.trades,
                                                  self.dso.congestion_level,
                                                  self.market.matching_history)

        # Calculate rewards
        rewards = {id: self.reward_handler.calculate_reward(agent,
                                                            orders[id],
                                                            matching_result,
                                                            kpis,
                                                            self.market_config.min_price,
                                                            self.market_config.max_price,
                                                            is_terminal=self.current_step >= self.max_steps - 1)
                   for id, agent in self._agents.items()}

        # STEP 4. Update environment state
        # Time attributes
        self.current_step += 1
        self.time_of_day = (self.current_step % self.max_steps) / self.max_steps

        # Agent attributes
        for id, agent in self._agents.items():
            agent.balance = self.market.get_agent_position(id)
            agent.reputation = self.reputation_handler.update_reputation(agent, matching_result, self.time_of_day)

        # STEP 5. Check if episode is done
        if self.current_step >= self.max_steps - 1:
            terminateds = {agent_id: True for agent_id in self.agents_id}
            terminateds["__all__"] = len([terminateds[agent] for agent in self.agents]) == len(self.agents)
            truncateds = {agent_id: False for agent_id in self.agents_id}
            truncateds["__all__"] = len([truncateds[agent] for agent in self.agents]) == len(self.agents)
        else:
            terminateds = {agent_id: False for agent_id in self.agents_id}
            terminateds["__all__"] = False
            truncateds = {agent_id: False for agent_id in self.agents_id}
            truncateds["__all__"] = False


        # STEP 6. Additional info
        dso_stats = {"dso_buy_volume": matching_result.dso_buy_volume,
                     "dso_sell_volume": matching_result.dso_sell_volume,
                     "dso_total_volume": matching_result.dso_total_volume,
                     "p2p_volume": matching_result.p2p_volume,
                     "dso_trade_ratio": matching_result.dso_trade_ratio,
                     "dso_grid_import": matching_result.dso_grid_import,
                     "dso_buy_price": matching_result.dso_buy_price,
                     "dso_sell_price": matching_result.dso_sell_price,
                     "price_spread": matching_result.price_spread,
                     "local_price_avg": matching_result.local_price_avg,
                     "local_price_advantage": matching_result.local_price_advantage}

        self.info = {"current_step": self.current_step,
                     "time_of_day": self.time_of_day,
                     "trades": matching_result.trades,
                     "grid_balance": self.dso.balance,
                     "market_price": matching_result.clearing_price,
                     "market_volume": matching_result.clearing_volume,
                     "matching_history": self.market.matching_history,
                     "dso_stats": dso_stats}

        # STEP 7. Update observation space and infos
        observations = self.observation_handler.update_observation_space(self.current_step,
                                                                         self.time_of_day,
                                                                         [*orders.values()],
                                                                         self.market.matching_history,
                                                                         self.implicit_cooperation)

        # Initialize infos for all agents
        infos = {agent_id: self.info for agent_id in self.agents_id}

        # Return observation, reward, terminated, truncated, and info dictionaries
        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment.

        Returns:
            Rendered frame as numpy array (rgb_array mode) or None
        """
        if self.render_mode is None:
            return None

        # Implement visualization (to be added)
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        if seed is not None:
            np.random.seed(seed)
