"""
Multi-agent reinforcement learning inference implementation using Ray RLlib.
"""

from os import path
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_LEARNER_GROUP, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.numpy import convert_to_numpy

from .lem import LocalEnergyMarket
from .train import GroupedLEM


class RLInference:
    """Manages training using Ray RLlib with support for multiple algorithms."""

    def __init__(self,
                 env_config: Dict[str, Any],
                 exploration: bool = False,
                 checkpoint_path: Optional[str] = None,
                 storage_path: Optional[str] = None):
        """Initialize the RL trainer.

        Args:
            env_config: Environment configuration dictionary
            exploration: Whether to use stochastic actions
            checkpoint_path: Path to trained experiment checkpoint
            storage_path: Path to store training results
        """
        self.env_config = env_config
        self.exploration = exploration
        self.checkpoint_path = checkpoint_path
        self.storage_path = storage_path

        # Validate initialization parameters
        self._check_init()

        # Initialize RL module
        self.rl_module: RLModule | MultiRLModule = self._restore_from_checkpoint(checkpoint_path)

        # Initialize environment
        self.env: MultiAgentEnv = LocalEnergyMarket(env_config) if isinstance(self.rl_module, MultiRLModule) else GroupedLEM(env_config)

    def _check_init(self) -> None:
        """Validate initialization parameters to ensure proper configuration."""
        # Validate checkpoint path if provided
        if self.checkpoint_path is None:
            raise ValueError("No checkpoint provided to restore the RL module. Please provide <checkpoint_path>.")

        if self.checkpoint_path is not None:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.parent.exists():
                raise ValueError(f"The <checkpoint_path> parent directory does not exist: <{checkpoint_path.parent}>.")

        # Validate storage path if provided
        if self.storage_path is not None:
            storage_path = Path(self.storage_path)
            if not storage_path.parent.exists():
                raise ValueError(f"The <storage_path> parent directory does not exist: <{storage_path.parent}>.")

        # Validate environment configuration
        if not isinstance(self.env_config, dict):
            raise ValueError(f"The <env_config> must be a dictionary, got <env_config = {self.env_config}>.")

        # Check required environment configuration keys
        for key in ["max_steps", "market_config", "agents"]:
            if key not in self.env_config:
                raise ValueError(f"The <env_config> is missing the required key: <key = {key}>.")

        # Validate max_steps
        if not isinstance(self.env_config["max_steps"], int) or self.env_config["max_steps"] <= 0:
            raise ValueError(f"The <max_steps> must be a positive integer, got <max_steps = {self.env_config["max_steps"]}>.")

        # Validate agents
        if not isinstance(self.env_config["agents"], list) or len(self.env_config["agents"]) == 0:
            raise ValueError(f"The <agents> must be a non-empty list, got <agents = {self.env_config["agents"]}>.")

    def _restore_from_checkpoint(self, checkpoint_path: str) -> None:
        """Restore RL module from checkpoint directory for inference.

        Args:
            checkpoint_path: Path to the checkpoint
        """
        try:
            rl_module_path = path.join(checkpoint_path, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE)
            default_policy_path = path.join(rl_module_path, DEFAULT_MODULE_ID)

            # CTCE or CTDE
            if path.exists(default_policy_path):
                return RLModule.from_checkpoint(default_policy_path)
            # DTDE
            else:
                return MultiRLModule.from_checkpoint(rl_module_path)

        except Exception as e:
            raise RuntimeError(f"Failed to restore the RL module from experiment <{checkpoint_path}>: {e}.")

    def inference(self, iters: int = 10) -> Dict[str, float]:
        """Inference of a trained policy.

        Args:
            iters: Number of inference episodes.

        Returns:
            Dictionary of inference metrics.
        """
        # Validate iters
        if not isinstance(iters, int) or iters <= 0:
            raise ValueError(f"The <iters> must be a positive integer, got <iters = {iters}>.")

        # Metrics to track
        total_rewards = {}
        episode_lengths = []
        price_history = []
        volume_history = []
        grid_balance_history = []
        dso_stats_history = []

        # Inference
        for episode in range(iters):
            print(f"Running episode {episode + 1}/{iters} (RL Module inference)")

            # STEP 1. Reset environment
            obs, info = self.env.reset()

            # STEP 2. Initialize episode tracking
            episode_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
            episode_step = 0
            done = False

            # STEP 3. Run episode
            while not done:
                # STEP 4. Get actions using the restored algorithm
                actions = {}

                # DTDE: Each agent has its own RL module
                if isinstance(self.rl_module, MultiRLModule):
                    for agent_id, agent_obs in obs.items():
                        # Convert to torch tensor and add batch dimension
                        obs_batch = {Columns.OBS: torch.from_numpy(agent_obs).unsqueeze(0)}

                        # Forward pass for this agent using the individual agent RLModule
                        model_outputs = self.rl_module[agent_id].forward_inference(obs_batch)

                        # For continuous actions the action_dist_inputs contains distribution parameters
                        action_dist_params = convert_to_numpy(model_outputs[Columns.ACTION_DIST_INPUTS][0])

                        # Get the action space for this agent
                        action_space = self.env.action_spaces[agent_id]
                        action_dim = action_space.shape[0]

                        # The model outputs mean + log_std for each action dimension
                        means = action_dist_params[..., :action_dim]

                        # Get the stochastic (exploration) or deterministic (exploitation) action
                        if self.exploration:
                            log_stds = action_dist_params[..., action_dim:]
                            sampled_actions = torch.distributions.Normal(torch.from_numpy(means),
                                                                            torch.exp(torch.from_numpy(log_stds))).sample().numpy()
                        else:
                            sampled_actions = means

                        # Clip actions to valid range
                        actions[agent_id] = np.clip(sampled_actions,
                                                    a_min=action_space.low,
                                                    a_max=action_space.high)

                # CTCE/CTDE: Single RL module with grouped agents
                else:
                    # Convert grouped observations to the expected format for CTDE
                    obs_batch = {Columns.OBS: {"grouped_agents": tuple(torch.from_numpy(agent_obs) for agent_obs in obs["grouped_agents"])}}

                    # Forward pass through the single CTDE RLModule
                    model_outputs = self.rl_module.forward_inference(obs_batch)

                    # For continuous actions the action_dist_inputs contains distribution parameters
                    action_dist_params = convert_to_numpy(model_outputs[Columns.ACTION_DIST_INPUTS])  # TODO (Add [0] or not)

                    # Split the action parameters for each agent
                    start_idx = 0

                    # Use the original agent IDs from the environment config
                    for i, agent_id in enumerate(self.env.original_agents_id):
                        # Get the action space for this agent from the grouped action space
                        action_space = self.env.action_space["grouped_agents"].spaces[i]
                        action_dim = action_space.shape[0]

                        # Extract this agent's action parameters (mean + log_std)
                        agent_action_params = action_dist_params[start_idx:start_idx + action_dim * 2]

                        # Split into means and log_stds
                        means = agent_action_params[..., :action_dim]

                        # Get the stochastic (exploration) or deterministic (exploitation) action
                        if self.exploration:
                            log_stds = agent_action_params[..., action_dim:]
                            sampled_actions = torch.distributions.Normal(torch.from_numpy(means),
                                                                            torch.exp(torch.from_numpy(log_stds))).sample().numpy()
                        else:
                            sampled_actions = means

                        # Clip actions to valid range
                        actions[agent_id] = np.clip(sampled_actions,
                                                    a_min=action_space.low,
                                                    a_max=action_space.high)

                        # Move to next agent's action parameters
                        start_idx += action_dim * 2

                # STEP 5. Step environment
                obs, rewards, terminateds, truncateds, info = self.env.step(actions)

                # STEP 6. Update episode rewards
                if isinstance(rewards, dict):
                    for agent_id, reward in rewards.items():
                        episode_rewards[agent_id] += reward
                else:
                    episode_rewards["grouped_agents"] = episode_rewards.get("grouped_agents", 0.0) + rewards

                # STEP 7. Track metrics
                if info and isinstance(info, dict):
                    price_history.append(info.get("market_price", 0.0))
                    volume_history.append(info.get("market_volume", 0.0))
                    grid_balance_history.append(info.get("grid_balance", 0.0))

                    if "dso_stats" in info:
                        dso_stats_history.append(info["dso_stats"])

                # STEP 8. Check termination
                episode_step += 1
                if isinstance(terminateds, dict):
                    done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
                else:
                    done = terminateds or truncateds

                # STEP 9. Check termination
                done = done or episode_step >= self.env_config["max_steps"]

                if done:
                    print(f"Episode {episode + 1} | Done: {done} | Reward = {episode_rewards}")
                    obs, info = self.env.reset()
                    break

            # STEP 10. Record episode metrics
            for agent_id, reward in episode_rewards.items():
                if agent_id not in total_rewards:
                    total_rewards[agent_id] = []
                total_rewards[agent_id].append(reward)

            episode_lengths.append(episode_step)

        return self._calculate_metrics(total_rewards,
                                       episode_lengths,
                                       price_history,
                                       volume_history,
                                       grid_balance_history,
                                       dso_stats_history)

    def _calculate_metrics(self, total_rewards: Dict, episode_lengths: list,
                          price_history: list, volume_history: list,
                          grid_balance_history: list, dso_stats_history: list) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}

        # Agent-specific metrics
        for agent_id, rewards in total_rewards.items():
            if rewards:  # Only calculate if we have data
                metrics[f"{agent_id}_mean_reward"] = np.mean(rewards)
                metrics[f"{agent_id}_std_reward"] = np.std(rewards)
                metrics[f"{agent_id}_min_reward"] = np.min(rewards)
                metrics[f"{agent_id}_max_reward"] = np.max(rewards)

        # Overall metrics
        all_rewards = [reward for rewards in total_rewards.values() for reward in rewards]
        if all_rewards:
            metrics["mean_reward"] = np.mean(all_rewards)
            metrics["std_reward"] = np.std(all_rewards)
            metrics["min_reward"] = np.min(all_rewards)
            metrics["max_reward"] = np.max(all_rewards)

        # Episode metrics
        if episode_lengths:
            metrics["mean_episode_length"] = np.mean(episode_lengths)
            metrics["std_episode_length"] = np.std(episode_lengths)

        # Market metrics
        if price_history:
            metrics["mean_price"] = np.mean(price_history)
            metrics["std_price"] = np.std(price_history)
        if volume_history:
            metrics["mean_volume"] = np.mean(volume_history)
            metrics["std_volume"] = np.std(volume_history)
        if grid_balance_history:
            metrics["mean_grid_balance"] = np.mean(np.abs(grid_balance_history))
            metrics["std_grid_balance"] = np.std(grid_balance_history)

        # DSO metrics
        if dso_stats_history:
            try:
                metrics["mean_dso_trade_ratio"] = np.mean([stats["dso_trade_ratio"] for stats in dso_stats_history])
                metrics["mean_dso_volume"] = np.mean([stats["dso_total_volume"] for stats in dso_stats_history])
                metrics["mean_p2p_volume"] = np.mean([stats["p2p_volume"] for stats in dso_stats_history])
            except (KeyError, TypeError):
                # Handle case where dso_stats structure is different
                pass

        return metrics
