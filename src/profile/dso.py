"""
Profile management for the DSO agent.

This module handles loading, generation, and management of time-varying profiles
for the DSO agent, including data validation and random profile generation.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import ProfileHandler


class DSOProfileHandler(ProfileHandler):
    """Manages pricing profiles for DSO agents."""

    def __init__(self,
                 min_price: float,
                 max_price: float,
                 feed_in_tariff_file_path: Optional[str] = None,
                 utility_price_file_path: Optional[str] = None,
                 decimals: int = 1,
                 seed: Optional[int] = None) -> None:
        """Initialize the DSO ProfileHandler.

        Args:
            min_price: Minimum price for profile generation
            max_price: Maximum price for profile generation
            feed_in_tariff_file_path: Path to feed-in tariff data CSV file
            utility_price_file_path: Path to utility price data CSV file
            decimals: Number of decimal places for rounding (default: 1)
            seed: Random seed for reproducibility
        """
        super().__init__(seed, decimals)

        self.min_price = min_price
        self.max_price = max_price
        self.feed_in_tariff_file_path = Path(feed_in_tariff_file_path) if feed_in_tariff_file_path else None
        self.utility_price_file_path = Path(utility_price_file_path) if utility_price_file_path else None
        self.decimals = decimals

        # Check initialization
        self._check_init()

    def reset(self, max_steps: int) -> Tuple[List[float], List[float]]:
        """Reset a DSO agent with new pricing profiles.

        Args:
            max_steps: Maximum number of steps in the simulation

        Returns:
            Tuple of (feed_in_tariff_profile, utility_price_profile)
        """
        return self.get_price_profiles(max_steps)

    def _check_init(self) -> None:
        """Check initialization."""
        # Check feed-in tariff file path
        if self.feed_in_tariff_file_path is not None:
            if not self.feed_in_tariff_file_path.exists():
                raise FileNotFoundError(f"Feed-in tariff file not found at <{self.feed_in_tariff_file_path}>.")
            if self.feed_in_tariff_file_path.suffix.lower() != '.csv':
                raise ValueError(f"Feed-in tariff file must be a CSV file: <{self.feed_in_tariff_file_path}>.")

        # Check utility price file path
        if self.utility_price_file_path is not None:
            if not self.utility_price_file_path.exists():
                raise FileNotFoundError(f"Utility price file not found at <{self.utility_price_file_path}>.")
            if self.utility_price_file_path.suffix.lower() != '.csv':
                raise ValueError(f"Utility price file must be a CSV file: <{self.utility_price_file_path}>.")

        # Validate min_price and max_price
        if not isinstance(self.min_price, float) or self.min_price < 0:
            raise ValueError(f"The <min_price> must be a positive integer, got <min_price = {self.min_price}>.")
        if not isinstance(self.max_price, float) or self.max_price < 0:
            raise ValueError(f"The <max_price> must be a positive integer, got <max_price = {self.max_price}>.")

        # Validate decimals
        if not isinstance(self.decimals, int) or self.decimals < 0:
            raise ValueError(f"The <decimals> must be a positive integer, got <decimals = {self.decimals}>.")

    def get_price_profiles(self, steps: int) -> Tuple[List[float], List[float]]:
        """Get feed-in tariff and utility price profiles.

        Args:
            steps: Number of steps needed

        Returns:
            Tuple of (feed_in_tariff_profile, utility_price_profile)
        """
        feed_in_tariff_profile = self._get_feed_in_tariff_profile(steps)
        utility_price_profile = self._get_utility_price_profile(steps)

        # Ensure utility prices are always higher than feed-in tariffs
        return self._enforce_price_relationship(feed_in_tariff_profile, utility_price_profile)

    def _enforce_price_relationship(self,
                                    feed_in_tariff: List[float],
                                    utility_price: List[float]) -> Tuple[List[float], List[float]]:
        """Ensure utility prices are always higher than feed-in tariffs.

        Args:
            feed_in_tariff: Feed-in tariff profile
            utility_price: Utility price profile

        Returns:
            Tuple of (adjusted_feed_in_tariff, adjusted_utility_price)
        """
        adjusted_feed_in = []
        adjusted_utility = []

        # Minimum gap between utility and feed-in tariff (as percentage of max_price)
        min_gap = (self.max_price - self.min_price) * 0.05  # 5% of price range as minimum gap

        for fit, utility in zip(feed_in_tariff, utility_price):
            # If utility price is not sufficiently higher than feed-in tariff
            if utility <= fit + min_gap:
                # Increase utility price to maintain gap
                required_utility = fit + min_gap

                # If this would exceed max_price, reduce feed-in tariff instead
                if required_utility > self.max_price:
                    # Reduce feed-in tariff to create room for utility price
                    adjusted_fit = max(self.min_price, self.max_price - min_gap)
                    adjusted_util = min(self.max_price, adjusted_fit + min_gap)
                else:
                    adjusted_fit = fit
                    adjusted_util = required_utility
            else:
                # Relationship is already correct
                adjusted_fit = fit
                adjusted_util = utility

            # Final clipping to ensure bounds
            adjusted_fit = np.clip(adjusted_fit, self.min_price, self.max_price)
            adjusted_util = np.clip(adjusted_util, adjusted_fit + min_gap, self.max_price)

            adjusted_feed_in.append(round(float(adjusted_fit), self.decimals))
            adjusted_utility.append(round(float(adjusted_util), self.decimals))

        return adjusted_feed_in, adjusted_utility

    def _get_feed_in_tariff_profile(self, steps: int) -> List[float]:
        """Get feed-in tariff profile.

        Args:
            steps: Number of steps needed

        Returns:
            Feed-in tariff profile
        """
        if self.feed_in_tariff_file_path:
            base_profile = self._load_profile(self.feed_in_tariff_file_path)

            # Scale profile to price range
            min_val, max_val = min(base_profile), max(base_profile)
            scaled_profile = []

            if max_val > min_val:
                for value in base_profile:
                    normalized = (value - min_val) / (max_val - min_val)
                    base_scaled_value = self.min_price + normalized * (self.max_price - self.min_price) * 0.6
                    scaled_value = base_scaled_value * np.random.uniform(0.7, 1.3)
                    scaled_profile.append(round(float(np.clip(scaled_value, self.min_price, self.max_price * 0.9)), self.decimals))

                return self._apply_smoothing(scaled_profile)

            else:
                for _ in base_profile:
                    scaled_value = self.min_price * np.random.uniform(0.7, 1.5)
                    scaled_profile.append(round(float(np.clip(scaled_value, self.min_price, self.max_price * 0.9)), self.decimals))

                return self._apply_smoothing(scaled_profile)

        else:
            return self._apply_smoothing(self._random_feed_in_tariff_profile(steps))

    def _get_utility_price_profile(self, steps: int) -> List[float]:
        """Get utility price profile.

        Args:
            steps: Number of steps needed

        Returns:
            Utility price profile
        """
        if self.utility_price_file_path:
            base_profile = self._load_profile(self.utility_price_file_path)

            # Scale profile to price range
            min_val, max_val = min(base_profile), max(base_profile)
            scaled_profile = []

            if max_val > min_val:
                for value in base_profile:
                    normalized = (value - min_val) / (max_val - min_val)
                    base_scaled_value = self.min_price * 1.5 + normalized * (self.max_price - self.min_price * 1.5)
                    random_factor = np.random.uniform(0.75, 1.25)
                    scaled_value = base_scaled_value * random_factor
                    scaled_profile.append(round(float(np.clip(scaled_value, self.min_price * 1.1, self.max_price)), self.decimals))

                return self._apply_smoothing(scaled_profile)

            else:
                base_val = self.max_price

                for _ in base_profile:
                    random_factor = np.random.uniform(0.75, 1.25)
                    scaled_value = base_val * random_factor
                    scaled_profile.append(round(float(np.clip(scaled_value, self.min_price * 1.1, self.max_price)), self.decimals))

                return self._apply_smoothing(scaled_profile)

        else:
            return self._apply_smoothing(self._random_utility_price_profile(steps))

    def _random_feed_in_tariff_profile(self, steps: int) -> List[float]:
        """Generate a step function feed-in tariff profile with randomness.

        Args:
            steps: Number of steps needed

        Returns:
            Feed-in tariff profile
        """
        profile = []

        # Feed-in tariff is typically lower and has step function behavior
        base_price = self.min_price + (self.max_price - self.min_price) * np.random.uniform(0.2, 0.4)

        # Define step periods with randomized timing and factors
        # Base periods with random shifts in timing (±1-2 hours) and factors (±20%)
        base_periods = [
            (0, 6, 0.9),    # Early morning - medium price
            (6, 10, 0.7),   # Morning - lower price
            (10, 16, 0.6),  # Midday - lowest price (high solar generation)
            (16, 18, 0.8),  # Late afternoon - medium-low price
            (18, 22, 1.0),  # Evening - higher price
            (22, 24, 0.9)   # Night - medium price
        ]

        # Randomize the step periods
        step_periods = []
        for start, end, factor in base_periods:
            # Add random shifts to period boundaries (±1-2 hours)
            start_shift = np.random.uniform(-1.5, 1.5)
            end_shift = np.random.uniform(-1.5, 1.5)
            new_start = max(0, start + start_shift)
            new_end = min(24, end + end_shift)

            # Ensure periods don't overlap incorrectly
            if new_end <= new_start:
                new_end = new_start + 1

            # Randomize the price factor (±30%)
            new_factor = factor * np.random.uniform(0.7, 1.3)
            step_periods.append((new_start, new_end, new_factor))

        for i in range(steps):
            hour = (i / steps) * 24

            # Find which step period this hour belongs to
            price_factor = 1.0  # default
            for start_hour, end_hour, factor in step_periods:
                if start_hour <= hour < end_hour:
                    price_factor = factor
                    break

            # Add additional randomness to individual steps (+/- 25%)
            random_factor = np.random.uniform(0.75, 1.25)
            value = max(self.min_price, base_price * price_factor * random_factor)

            # Clip to ensure we don't exceed max bounds
            value = min(value, self.max_price * 0.9)
            profile.append(round(float(value), self.decimals))

        return profile

    def _random_utility_price_profile(self, steps: int) -> List[float]:
        """Generate a step function utility price profile with randomness.

        Args:
            steps: Number of steps needed

        Returns:
            Utility price profile
        """
        profile = []

        # Utility price is typically higher and has step function behavior
        base_price = self.min_price * 2 + (self.max_price - self.min_price * 2) * np.random.uniform(0.6, 0.8)

        # Define step periods with randomized timing and factors
        # Base periods with random shifts in timing (±1-2 hours) and factors (±20%)
        base_periods = [
            (0, 1, 0.8),    # Midnight - medium-low price
            (1, 6, 0.6),    # Early morning - off-peak (lowest)
            (6, 9, 1.3),    # Morning peak - high price
            (9, 12, 1.0),   # Late morning - standard price
            (12, 16, 1.1),  # Afternoon - medium-high price
            (16, 18, 1.2),  # Late afternoon - higher price
            (18, 22, 1.4),  # Evening peak - highest price
            (22, 24, 1.0)   # Night - standard price
        ]

        # Randomize the step periods
        step_periods = []
        for start, end, factor in base_periods:
            # Add random shifts to period boundaries (±1-2 hours)
            start_shift = np.random.uniform(-1.5, 1.5)
            end_shift = np.random.uniform(-1.5, 1.5)
            new_start = max(0, start + start_shift)
            new_end = min(24, end + end_shift)

            # Ensure periods don't overlap incorrectly
            if new_end <= new_start:
                new_end = new_start + 1

            # Randomize the price factor (±25%)
            new_factor = factor * np.random.uniform(0.75, 1.25)
            step_periods.append((new_start, new_end, new_factor))

        for i in range(steps):
            hour = (i / steps) * 24

            # Find which step period this hour belongs to
            price_factor = 1.0  # default
            for start_hour, end_hour, factor in step_periods:
                if start_hour <= hour < end_hour:
                    price_factor = factor
                    break

            # Add additional randomness to individual steps (+/- 20%)
            random_factor = np.random.uniform(0.8, 1.2)
            value = min(self.max_price, base_price * price_factor * random_factor)

            # Clip to ensure we don't go below min bounds
            value = max(value, self.min_price * 1.1)
            profile.append(round(float(value), self.decimals))

        return profile
