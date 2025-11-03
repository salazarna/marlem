"""
Base classes for the Local Energy Market environment.
"""

from enum import Enum

from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
from ray.rllib.core.rl_module.torch import TorchRLModule


class RLProtection(Enum):
    """Supported RL protection methods."""

    MAX_SAFE_VALUE = 1e5

    @staticmethod
    def clip_value(value: float,
                   min_value: float = -MAX_SAFE_VALUE,
                   max_value: float = MAX_SAFE_VALUE) -> float:
        """Clip a value to the maximum safe value.

        Args:
            value: The value to clip.
            min_value: Minimum value for clipping (defaults to -1e5).
            max_value: Maximum value for clipping (defaults to 1e5).

        Returns:
            The clipped value.
        """
        return max(min_value, min(max_value, value))


class RLAlgorithm(Enum):
    """Supported RL algorithms."""

    PPO = "ppo"
    APPO = "appo"
    SAC = "sac"

    def get_module_class(self) -> TorchRLModule:
        """Get the module class for this algorithm."""
        modules = {"ppo": DefaultPPOTorchRLModule,
                   "appo": DefaultAPPOTorchRLModule,
                   "sac": DefaultSACTorchRLModule}

        return modules[self.value]


class TrainingMode(Enum):
    """Supported training modes."""

    CTCE = "ctce"
    CTDE = "ctde"
    DTDE = "dtde"
