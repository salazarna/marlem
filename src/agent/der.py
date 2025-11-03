"""
Distributed Energy Resource (DER) agent implementation.

This module implements agents representing distributed energy resources (DERs) in the
Local Energy Market. Agents can have generation and demand profiles, battery storage,
and the ability to participate in market trading with demand response capabilities.
"""

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from networkx import Graph

from ..grid.base import Location
from .battery import Battery

if TYPE_CHECKING:
    from ..market.matching import MarketConfig


class DERAgent:
    """Agent representing a Distributed Energy Resource in the Local Energy Market."""

    def __init__(self,
                 id: str,
                 capacity: float,
                 battery: Optional[Battery] = None,
                 node_id: Optional[str] = None,
                 generation_profile: List[float] = [],
                 demand_profile: List[float] = []) -> None:
        """Initialize the DER agent.

        Args:
            id: Unique identifier for the agent
            capacity: Maximum capacity of the DER (W)
            battery: Optional battery object
            node_id: Optional node ID in the grid network
            generation_profile: Optional time series (list/array) of generation values per timestep
            demand_profile: Optional time series (list/array) of demand values per timestep
        """
        self.id = id
        self.capacity = capacity
        self.battery = battery
        self.node_id = node_id
        self.generation_profile = generation_profile
        self.demand_profile = demand_profile

        # Demand response tracking
        self.cumulative_demand_satisfied: float = 0.0
        self.cumulative_demand_deferred: float = 0.0
        self.total_demand_required: float = sum(self.demand_profile)
        self.cumulative_supply_satisfied: float = 0.0
        self.cumulative_supply_deferred: float = 0.0
        self.total_supply_required: float = sum(self.generation_profile)

        # Grid attributes
        self.location: Optional[Location] = None

        # Attributes from the environment
        self.balance: float = 0.0
        self.profit: float = 0.0
        self.reputation: float = 0.5

        # Market state tracking
        self.price_history: List[float] = []

        # Mechanism factors
        self._reputation_factor: float = 0.5
        self._cooperation_factor: float = 0.3

        # Check initialization
        self._check_init()

    def reset(self,
              reputation: float,
              generation_profile: List[float],
              demand_profile: List[float],
              seed: Optional[int] = None) -> None:
        """Reset the DER agent state.

        Args:
            reputation: Reputation score
            generation_profile: Generation profile
            demand_profile: Demand profile
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset agent attributes
        self.reputation = reputation
        self.generation_profile = generation_profile
        self.demand_profile = demand_profile

        # Reset battery
        if self.battery:
            self.battery.reset(seed)

        # Reset demand response tracking
        self.cumulative_demand_satisfied = 0.0
        self.cumulative_demand_deferred = 0.0
        self.total_demand_required = sum(self.demand_profile)
        self.cumulative_supply_satisfied = 0.0
        self.cumulative_supply_deferred = 0.0
        self.total_supply_required = sum(self.generation_profile)

        # Reset attributes from the environment
        self.balance = np.random.uniform(-self.capacity, self.capacity)
        self.profit = 0.0

        # Reset market state tracking
        self.price_history = []

    def _check_init(self) -> None:
        """ Validate DERAgent parameters.

        Raises:
            ValueError: If id is not a string
            ValueError: If generation_profile or demand_profile have different lengths
        """

        if not isinstance(self.id, str):
            raise ValueError(f"id must be a string, got {type(self.id)}")

        if not (len(self.generation_profile) == len(self.demand_profile)):
            raise ValueError(f"Length mismatch: generation_profile ({len(self.generation_profile)}), demand_profile ({len(self.demand_profile)})")

    def update_location(self, location: Location) -> None:
        """Update the agent's location.

        Args:
            location: New location of the agent
        """
        self.location = location

    def select_preferred_partner(self,
                                 candidates: List['DERAgent'],
                                 graph: Graph,
                                 randomize: bool = True) -> str:
        """Select a preferred trading partner based on the agent"s strategy and observations.

        This method implements different strategies for partner selection.
        The selected partner ID will be included in the agent"s market orders.

        Args:
            candidates: List of known agents
            graph: Optional graph for distance calculation
            randomize: Whether to randomize the selection of a partner

        Returns:
            ID of the preferred trading partner
        """
        # Exclude self from candidates
        candidates = [agent for agent in candidates if agent.id != self.id]

        # Sort candidates by reputation and distance
        candidates.sort(key=lambda x: (-x.reputation, self.location.distance_to(x.location, graph)))

        return np.random.choice(candidates[:max(1, len(candidates) // 2 + 1)]).id if randomize else candidates[0].id

    def adjust_action_for_battery(self,
                                  step: int,
                                  action: np.array,
                                  market_config: "MarketConfig") -> np.array:
        """Adjust action to respect battery constraints.

        This method enforces physical constraints on the agent's actions based on:
        1. Battery state of charge limits
        2. Available energy for charging/discharging
        3. Market minimum and maximum quantity constraints

        Note: Cumulative demand/supply tracking is handled after market clearing
        when actual trades are known, not during action adjustment.

        Args:
            step: Current time step
            action: Raw action from policy containing [price, quantity, is_buy, preferred_partner]
            market_config: Market configuration parameters

        Returns:
            Adjusted action that respects physical constraints and market rules
        """
        # Unpack action components
        price, quantity, is_buy, preferred_partner = action

        # Get available energy from battery (if available)
        if self.battery:
            available_charge, available_discharge = self.battery.estimate_available_energy()
        else:
            available_charge, available_discharge = 0.0, 0.0

        # Current step's generation and base demand
        current_generation = self.generation_profile[step]
        current_demand = self.demand_profile[step]

        if is_buy:
            # Calculate how much energy we need to buy after using our own generation
            energy_deficit = quantity - current_generation

            # If we have excess generation, we can charge the battery
            if energy_deficit < 0:
                potential_charge = min(abs(energy_deficit), available_charge)
                energy_deficit += potential_charge

            # If we still have energy deficit, we need to buy or discharge battery
            if energy_deficit > 0:
                potential_discharge = min(energy_deficit, available_discharge)
                energy_deficit -= potential_discharge

            # Final quantity to buy is the remaining deficit (if positive)
            quantity = max(0.0, energy_deficit)

        else:
            # Calculate how much energy we can export after using our own generation
            energy_surplus = quantity - current_demand

            # If we have excess generation, we can charge the battery
            if energy_surplus > 0:
                potential_charge = min(energy_surplus, available_charge)
                energy_surplus -= potential_charge

            # If we have a deficit, we can discharge the battery to meet the export
            if energy_surplus < 0:
                potential_discharge = min(abs(energy_surplus), available_discharge)
                energy_surplus += potential_discharge

            # Final quantity to export is the remaining surplus (if positive)
            quantity = max(0.0, energy_surplus)

        # Ensure price and quantity respects market constraints
        price = max(market_config.min_price, min(market_config.max_price, price))
        quantity = max(market_config.min_quantity, min(market_config.max_quantity, quantity))

        return np.array([price, quantity, is_buy, preferred_partner], dtype=np.float32)

    def update_energy_tracking(self,
                               step: int,
                               energy_bought: float = 0.0,
                               energy_sold: float = 0.0) -> None:
        """Update cumulative energy tracking based on actual market outcomes.

        This method should be called after market clearing to track how much demand
        and supply was actually satisfied vs deferred based on market results.

        Args:
            step: Current time step
            energy_bought: Energy bought in the market this step
            energy_sold: Energy sold in the market this step
        """
        current_generation = self.generation_profile[step]
        current_demand = self.demand_profile[step]

        # Track demand satisfaction
        # Demand can be satisfied by: own generation + market purchases + battery discharge
        energy_available_for_demand = current_generation + energy_bought

        # What portion of demand was satisfied this step
        demand_satisfied_this_step = min(current_demand, energy_available_for_demand)
        demand_deferred_this_step = max(0.0, current_demand - demand_satisfied_this_step)

        self.cumulative_demand_satisfied += demand_satisfied_this_step
        self.cumulative_demand_deferred += demand_deferred_this_step

        # Track supply utilization
        # Supply can be used for: own demand + market sales + battery charging
        energy_used_from_supply = min(current_demand, current_generation) + energy_sold

        # What portion of generation was utilized this step
        supply_satisfied_this_step = min(current_generation, energy_used_from_supply)
        supply_deferred_this_step = max(0.0, current_generation - supply_satisfied_this_step)

        self.cumulative_supply_satisfied += supply_satisfied_this_step
        self.cumulative_supply_deferred += supply_deferred_this_step

    def update_battery_from_trades(self,
                                   step: int,
                                   energy_bought: float = 0.0,
                                   energy_sold: float = 0.0) -> None:
        """Update battery state based on actual market trade outcomes.

        This method calculates the agent's energy balance after trading and updates
        the battery state accordingly. It should be called after market clearing.

        Args:
            step: Current time step
            energy_bought: Energy bought in trades this step
            energy_sold: Energy sold in trades this step
        """
        # Skip if no battery
        if not self.battery:
            return

        # Calculate total energy balance: generation - demand + net trading
        current_generation = self.generation_profile[step]
        current_demand = self.demand_profile[step]
        net_trade_energy = energy_sold - energy_bought

        energy_balance = current_generation - current_demand + net_trade_energy

        if energy_balance > 0:
            # Surplus energy - charge battery
            self.battery.charge(energy_balance)
        elif energy_balance < 0:
            # Energy deficit - discharge battery
            self.battery.discharge(abs(energy_balance))
        else:
            # Perfect balance - set battery to idle
            self.battery.idle()

    def get_generation(self, step: int) -> float:
        """Get generation value for a specific step.

        Args:
            step: Time step

        Returns:
            Generation value at the specified step
        """
        try:
            return self.generation_profile[step]
        except IndexError:
            raise IndexError(f"Time step {step} is out of range for <generation_profile> with length {len(self.generation_profile)}.")

    def get_demand(self, step: int) -> float:
        """Get demand value for a specific step.

        Args:
            step: Time step

        Returns:
            Demand value at the specified step
        """
        try:
            return self.demand_profile[step]
        except IndexError:
            raise IndexError(f"Time step {step} is out of range for <demand_profile> with length {len(self.demand_profile)}.")

    def get_remaining_demand(self, step: int) -> float:
        """Get remaining demand from current step to end of episode.

        Args:
            step: Current time step

        Returns:
            Sum of remaining demand
        """
        try:
            return sum(self.demand_profile[step:])
        except IndexError:
            raise IndexError(f"Time step {step} is out of range for <demand_profile> with length {len(self.demand_profile)}.")

    def get_remaining_supply(self, step: int) -> float:
        """Get remaining supply from current step to end of episode.

        Args:
            step: Current time step

        Returns:
            Sum of remaining generation
        """
        try:
            return sum(self.generation_profile[step:])
        except IndexError:
            raise IndexError(f"Time step {step} is out of range for <generation_profile> with length {len(self.generation_profile)}.")
