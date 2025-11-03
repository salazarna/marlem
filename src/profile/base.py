"""
Base class for profile management.

This module handles loading, generation, and management of time-varying profiles.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class ProfileHandler:
    """Base class for profile management with common functionality."""

    def __init__(self,
                 seed: Optional[int] = None,
                 decimals: int = 1) -> None:
        """Initialize the base profile handler.

        Args:
            seed: Random seed for reproducibility
            decimals: Number of decimal places for rounding (default: 1)
        """
        if seed is not None:
            np.random.seed(seed)

        self.decimals = decimals

    def _load_profile(self, path: Path) -> List[float]:
        """Load a single, random profile from a CSV file on-demand.

        Args:
            path: Path to the CSV file

        Returns:
            Profile list
        """
        df = pd.read_csv(path)
        selected_col = np.random.choice(df.columns)

        return df[selected_col].tolist()

    def _apply_smoothing(self,
                         profile: List[float],
                         window_size: int = 3) -> List[float]:
        """Applies a simple moving average filter to smooth a profile.

        Args:
            profile: The profile data to smooth.
            window_size: The size of the moving average window.

        Returns:
            The smoothed profile.
        """
        if (window_size <= 1) or (len(profile) < window_size):
            return profile

        kernel = np.ones(window_size) / window_size
        smoothed_arr = np.convolve(profile, kernel, mode='same')

        return [round(float(x), self.decimals) for x in smoothed_arr]
