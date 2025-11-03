"""
Battery storage model for DER agents in the Local Energy Market.

This module implements a simplified battery model that focuses on energy level
tracking and efficiency losses during charging and discharging operations.
"""

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class BatteryState(Enum):
    """Possible states of a battery."""

    CHARGING = "charging"
    DISCHARGING = "discharging"
    IDLE = "idle"


class Battery:
    """Battery storage model focusing on energy level and efficiency."""

    def __init__(self,
                 nominal_capacity: float,
                 min_soc: float = 0.0,
                 max_soc: float = 1.0,
                 charge_efficiency: float = 1.0,
                 discharge_efficiency: float = 1.0) -> None:
        """Initialize the battery model.

        Args:
            nominal_capacity: Nominal capacity of the battery in kWh
            min_soc: Minimum state of charge (0-1)
            max_soc: Maximum state of charge (0-1)
            charge_efficiency: Charging efficiency (0-1)
            discharge_efficiency: Discharging efficiency (0-1)
        """
        self.nominal_capacity = nominal_capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

        # Initialize state
        self.state: BatteryState = None
        self.energy_level: float = None
        self.cumulative_charge: float = None
        self.cumulative_discharge: float = None
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the battery state.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset state
        self.state = BatteryState.IDLE
        self.energy_level = np.random.uniform(self.min_soc * self.nominal_capacity, self.max_soc * self.nominal_capacity)
        self.cumulative_charge = 0.0
        self.cumulative_discharge = 0.0

        # Check reset
        self._check_init()

    def _check_init(self) -> None:
        """Validate battery specifications after initialization.

        Raises:
            ValueError: If nominal capacity is less than or equal to zero
            ValueError: If min SOC is not between 0 and 1
            ValueError: If max SOC is not between 0 and 1
            ValueError: If min SOC is greater than max SOC
        """
        if self.nominal_capacity <= 0:
            raise ValueError("Nominal capacity must be greater than zero.")
        if not 0 <= self.min_soc <= 1:
            raise ValueError("Min SOC must be between 0 and 1.")
        if not 0 <= self.max_soc <= 1:
            raise ValueError("Max SOC must be between 0 and 1.")
        if self.min_soc >= self.max_soc:
            raise ValueError("Min SOC must be less than Max SOC.")
        if not 0 <= self.charge_efficiency <= 1:
            raise ValueError("Charge efficiency must be between 0 and 1.")
        if not 0 <= self.discharge_efficiency <= 1:
            raise ValueError("Discharge efficiency must be between 0 and 1.")
        if self.energy_level < self.min_soc * self.nominal_capacity or self.energy_level > self.max_soc * self.nominal_capacity:
            raise ValueError("Energy level must be between min SOC and max SOC.")

    def _can_charge(self, energy: float) -> bool:
        """Check if charging is possible.

        Args:
            energy: Charging energy in kWh

        Returns:
            True if charging is possible
        """
        if energy <= 0:
            return False

        # Check if battery is already full
        if self.energy_level + energy >= self.max_soc * self.nominal_capacity:
            return False

        return True

    def _can_discharge(self, energy: float) -> bool:
        """Check if discharging is possible.

        Args:
            energy: Discharge energy in kWh

        Returns:
            True if discharging is possible
        """
        if energy <= 0:
            return False

        # Check if battery is already empty
        if self.energy_level - energy <= self.min_soc * self.nominal_capacity:
            return False

        return True

    def charge(self, energy: float) -> float:
        """Charge the battery.

        Args:
            energy: Charging energy in Wh

        Returns:
            Energy charged to the battery
        """
        if self._can_charge(energy):
            # Calculate maximum possible charge considering efficiency
            max_energy = min(energy,  # Energy from requested charge
                             self.max_soc * self.nominal_capacity - self.energy_level)  # Energy until full

            energy_stored = max_energy * self.charge_efficiency

            # Update state
            self.energy_level += energy_stored
            self.cumulative_charge += energy_stored
            self.state = BatteryState.CHARGING

            return energy_stored

        else:
            self.state = BatteryState.IDLE
            return 0.0

    def discharge(self, energy: float) -> float:
        """Discharge the battery.

        Args:
            energy: Discharging energy in Wh

        Returns:
            Energy discharged from the battery
        """
        if self._can_discharge(energy):
            # Calculate maximum possible discharge considering efficiency
            max_energy = min(energy,  # Energy from requested discharge
                             self.energy_level - self.min_soc * self.nominal_capacity)  # Energy until empty

            energy_discharged = max_energy * self.discharge_efficiency

            # Update state
            self.energy_level -= energy_discharged
            self.cumulative_discharge += energy_discharged
            self.state = BatteryState.DISCHARGING

            return energy_discharged

        else:
            self.state = BatteryState.IDLE
            return 0.0

    def idle(self) -> None:
        """Update battery state for a time step."""
        self.state = BatteryState.IDLE

    def get_state(self) -> Dict:
        """Get current battery state.

        Returns:
            Dictionary containing battery state information
        """
        available_charge, available_discharge = self.estimate_available_energy()

        return {"state": self.state,
                "energy_level": self.energy_level,
                "soc": self.energy_level / self.nominal_capacity,
                "available_charge": available_charge,
                "available_discharge": available_discharge,
                "cumulative_charge": self.cumulative_charge,
                "cumulative_discharge": self.cumulative_discharge}

    def estimate_available_energy(self) -> Tuple[float, float]:
        """Estimate available energy for charging and discharging.

        Returns:
            Tuple of (available charge capacity, available discharge capacity)
        """
        available_charge = self.max_soc * self.nominal_capacity - self.energy_level
        available_discharge = self.energy_level - self.min_soc * self.nominal_capacity

        return (available_charge, available_discharge)
