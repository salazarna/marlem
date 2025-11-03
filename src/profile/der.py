"""
Profile management for the DER agent.

This module handles loading, generation, and management of time-varying profiles
for the DER agent, including data validation and random profile generation.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import ProfileHandler


class DERProfileHandler(ProfileHandler):
    """Manages generation and demand profiles for DER agents."""

    def __init__(self,
                 min_quantity: float,
                 max_quantity: float,
                 generation_file_path: Optional[str] = None,
                 demand_file_path: Optional[str] = None,
                 decimals: int = 1,
                 seed: Optional[int] = None) -> None:
        """Initialize the DER ProfileHandler.

        Args:
            min_quantity: Minimum energy quantity (Wh)
            max_quantity: Maximum energy quantity (Wh)
            generation_file_path: Path to the generation data CSV file
            demand_file_path: Path to the demand data CSV file
            decimals: Number of decimal places for rounding (default: 1)
            seed: Random seed for reproducibility
        """
        super().__init__(seed, decimals)

        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.generation_file_path: Optional[Path] = Path(generation_file_path) if generation_file_path else None
        self.demand_file_path: Optional[Path] = Path(demand_file_path) if demand_file_path else None
        self.decimals = decimals

        # Check initialization
        self._check_init()

    def reset(self,
              capacity: float,
              max_steps: int) -> Tuple[List[float], List[float]]:
        """Reset an agent with new generation and demand profiles.

        Args:
            capacity: Maximum capacity of the DER (W)
            max_steps: Maximum number of steps in the simulation

        Returns:
            Tuple of (generation, demand)
        """
        return self.get_energy_profiles(max_steps, capacity)

    def _check_init(self) -> None:
        """Check initialization of file paths."""
        # Check generation file path
        if self.generation_file_path is not None:
            if not self.generation_file_path.exists():
                raise FileNotFoundError(f"Generation file not found at <{self.generation_file_path}>.")
            if self.generation_file_path.suffix.lower() != '.csv':
                raise ValueError(f"Generation file must be a CSV file: <{self.generation_file_path}>.")

        # Check demand file path
        if self.demand_file_path is not None:
            if not self.demand_file_path.exists():
                raise FileNotFoundError(f"Demand file not found at <{self.demand_file_path}>.")
            if self.demand_file_path.suffix.lower() != '.csv':
                raise ValueError(f"Demand file must be a CSV file: <{self.demand_file_path}>.")

        # Validate min_quantity and max_quantity
        if not isinstance(self.min_quantity, float) or self.min_quantity < 0:
            raise ValueError(f"The <min_quantity> must be a positive integer, got <min_quantity = {self.min_quantity}>.")
        if not isinstance(self.max_quantity, float) or self.max_quantity < 0:
            raise ValueError(f"The <max_quantity> must be a positive integer, got <max_quantity = {self.max_quantity}>.")

        # Validate decimals
        if not isinstance(self.decimals, int) or self.decimals < 0:
            raise ValueError(f"The <decimals> must be a positive integer, got <decimals = {self.decimals}>.")

    def get_energy_profiles(self,
                            steps: int,
                            capacity: float,
                            constant: bool = False) -> Tuple[List[float], List[float]]:
        """Get generation and demand profiles for a specific agent.

        Args:
            steps: Number of steps needed
            capacity: Agent capacity for scaling from per-unit

        Returns:
            Tuple of (generation_profile, demand_profile)
        """
        generation = self._clip(self._get_generation_profile(steps, capacity, constant))
        demand = self._clip(self._get_demand_profile(steps, capacity, constant))

        return generation, demand

    def _get_generation_profile(self,
                                steps: int,
                                capacity: float,
                                constant: bool) -> List[float]:
        """Get generation profile for an agent.

        Args:
            steps: Number of steps needed
            capacity: Agent capacity for scaling from per-unit
            constant: Whether to use a constant profile

        Returns:
            Generation profile
        """
        if self.generation_file_path:
            base_profile = self._load_profile(self.generation_file_path)
            scale_factor = np.random.uniform(1.0, 1.3)
            noise_factor = 0.05
            varied_profile = []

            for value in base_profile:
                noise = np.random.uniform(-noise_factor, noise_factor)
                varied_value = value * scale_factor * (1 + noise)
                scaled_value = max(0.0, varied_value * capacity)
                varied_profile.append(round(float(scaled_value), self.decimals))

            return self._apply_smoothing(varied_profile)

        else:
            if constant:
                return [capacity] * steps
            else:
                return self._apply_smoothing(self._random_generation_profile(steps, capacity), window_size=max(3, steps // 10))

    def _get_demand_profile(self,
                            steps: int,
                            capacity: float,
                            constant: bool) -> List[float]:
        """Get demand profile for an agent.

        Args:
            steps: Number of steps needed
            capacity: Agent capacity for scaling from per-unit
            constant: Whether to use a constant profile

        Returns:
            Demand profile
        """
        if self.demand_file_path:
            base_profile = self._load_profile(self.demand_file_path)
            scale_factor = np.random.uniform(0.6, 1.0)
            noise_factor = 0.1
            varied_profile = []

            for value in base_profile:
                noise = np.random.uniform(-noise_factor, noise_factor)
                varied_value = value * scale_factor * (1 + noise)
                scaled_value = max(0.0, varied_value * capacity)
                varied_profile.append(round(float(scaled_value), self.decimals))

            return self._apply_smoothing(varied_profile)
        else:
            if constant:
                return [capacity] * steps
            else:
                return self._apply_smoothing(self._random_demand_profile(steps, capacity), window_size=max(3, steps // 10))

    def _random_generation_profile(self,
                                   steps: int,
                                   capacity: float,
                                   max_value: float = 1.0) -> List[float]:
        """Generate a random generation profile (solar-like).

        Args:
            steps: Number of steps needed
            capacity: Agent capacity for scaling from per-unit
            max_value: Maximum value for the profile

        Returns:
            Generation profile
        """
        profile = []
        noise_magnitude = 0.1 if steps == 24 else 0.05
        cloudiness = np.random.uniform(0.0, 1.0)
        effective_max_value = max_value * (1 - 0.5 * cloudiness)

        for i in range(steps):
            hour = (i / steps) * 24
            if 6 <= hour <= 18:
                normalized_hour = (hour - 12) / 6
                base_value = effective_max_value * np.exp(-normalized_hour**2)
                cloud_noise = np.random.uniform(-0.2 * cloudiness, 0.2 * cloudiness)
                noise = np.random.uniform(-noise_magnitude, noise_magnitude)
                value = max(0.0, (base_value + noise + cloud_noise) * capacity)
            else:
                value = 0.0
            profile.append(round(float(value), self.decimals))

        return profile

    def _random_demand_profile(self,
                               steps: int,
                               capacity: float,
                               min_value: float = 0.2,
                               max_value: float = 1.0) -> List[float]:
        """Generate a random demand profile (typical residential pattern).

        Args:
            steps: Number of steps needed
            capacity: Agent capacity for scaling from per-unit
            min_value: Minimum value for the profile
            max_value: Maximum value for the profile

        Returns:
            Demand profile
        """
        profile = []
        noise_magnitude = 0.1 if steps == 24 else 0.05

        for i in range(steps):
            hour = (i / steps) * 24
            if (7 <= hour < 9) or (19 <= hour < 21):
                base_value = np.random.uniform(0.7, max_value)
            elif (1 <= hour < 6):
                base_value = np.random.uniform(min_value, 0.4)
            else:
                base_value = np.random.uniform(0.4, 0.7)

            noise = np.random.uniform(-noise_magnitude, noise_magnitude)
            value = max(0.0, (base_value + noise) * capacity)
            profile.append(round(float(value), self.decimals))

        return profile

    def _clip(self, values: List[float]) -> List[float]:
        """Clip values to the minimum and maximum quantity.

        Args:
            values: List of values to clip

        Returns:
            List of clipped values
        """
        return [round(float(val), self.decimals) for val in np.clip(values, self.min_quantity, self.max_quantity).tolist()]
