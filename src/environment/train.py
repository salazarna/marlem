"""
Multi-agent reinforcement learning training implementation using Ray RLlib.
"""

from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune import CheckpointConfig, ResultGrid, RunConfig, TuneConfig, Tuner, loguniform
from ray.tune.result import DONE, TRAINING_ITERATION
from ray.tune.schedulers import PopulationBasedTraining
from torch.cuda import device_count, is_available

from .base import RLAlgorithm, TrainingMode
from .ctce import CTCEModule
from .ctde import CTDEModule
from .io import EnvConfigHandler
from .lem import LocalEnergyMarket


class RLTrainer:
    """Manages training using Ray RLlib with support for multiple algorithms."""

    def __init__(self,
                 env_config: Dict[str, Any],
                 algorithm: RLAlgorithm = RLAlgorithm.PPO,
                 training: TrainingMode = TrainingMode.DTDE,
                 iters: int = 100,
                 tune_samples: int = 1,
                 checkpoint_freq: int = 10,
                 evaluation_interval: int = 10,
                 evaluation_duration: int = 30,
                 cpus: Optional[int] = None,
                 gpus: Optional[int] = None,
                 storage_path: Optional[str] = None):
        """Initialize the RL trainer.

        Args:
            env_config: Environment configuration dictionary
            algorithm: RL algorithm to use (PPO, APPO, SAC)
            training: Training mode (CTCE, CTDE, DTDE)
            iters: Number of training iterations
            tune_samples: Number of samples for tuning
            checkpoint_freq: Frequency of checkpoints
            evaluation_interval: Frequency of evaluation
            evaluation_duration: Duration of evaluation
            cpus: Number of CPUs for parallel sampling
            gpus: Number of GPUs per learner
            storage_path: Path to store training results
        """
        self.env_config = env_config
        self.algorithm = algorithm
        self.training = training
        self.iters = iters
        self.tune_samples = tune_samples
        self.checkpoint_freq = checkpoint_freq
        self.evaluation_interval = evaluation_interval
        self.evaluation_duration = evaluation_duration
        self.cpus = self._get_cpus_count() if cpus is None else cpus
        self.gpus = self._get_cuda_gpu_count() if gpus is None else gpus
        self.storage_path = storage_path

        # Validate initialization parameters
        self._check_init()

        # Initialize environment
        self.env: MultiAgentEnv = LocalEnergyMarket(env_config) if self.training == TrainingMode.DTDE else GroupedLEM(env_config)
        # self.env: MultiAgentEnv = RockPaperScissors(env_config) if self.training == TrainingMode.DTDE else GroupedRockPaperScissors(env_config)

    def _check_init(self) -> None:
        """Validate initialization parameters to ensure proper configuration."""
        # Validate algorithm
        if not isinstance(self.algorithm, RLAlgorithm):
            raise ValueError(f"The <algorithm> must be an RLAlgorithm enum, got <algorithm = {self.algorithm.value}>, supported algorithms: ['PPO', 'APPO', 'SAC'].")

        # Validate training mode
        if not isinstance(self.training, TrainingMode):
            raise ValueError(f"The <training> must be a TrainingMode enum, got <training = {self.training.value}>, supported training modes: ['CTCE', 'CTDE', 'DTDE'].")

        # Validate positive integers
        if not isinstance(self.cpus, int) or self.cpus <= 0:
            raise ValueError(f"The <cpus> must be a positive integer, got <cpus = {self.cpus}>.")

        if not isinstance(self.gpus, int) or self.gpus < 0:
            raise ValueError(f"The <gpus> must be a non-negative integer, got <gpus = {self.gpus}>.")

        if not isinstance(self.iters, int) or self.iters <= 0:
            raise ValueError(f"The <iters> must be a positive integer, got <iters = {self.iters}>.")

        if not isinstance(self.tune_samples, int) or self.tune_samples <= 0:
            raise ValueError(f"The <tune_samples> must be a positive integer, got <tune_samples = {self.tune_samples}>.")

        if not isinstance(self.checkpoint_freq, int) or self.checkpoint_freq <= 0:
            raise ValueError(f"The <checkpoint_freq> must be a positive integer, got <checkpoint_freq = {self.checkpoint_freq}>.")

        if not isinstance(self.evaluation_interval, int) or self.evaluation_interval <= 0:
            raise ValueError(f"The <evaluation_interval> must be a positive integer, got <evaluation_interval = {self.evaluation_interval}>.")

        if not isinstance(self.evaluation_duration, int) or self.evaluation_duration <= 0:
            raise ValueError(f"The <evaluation_duration> must be a positive integer, got <evaluation_duration = {self.evaluation_duration}>.")

        # Validate checkpoint frequency doesn't exceed iterations
        if self.checkpoint_freq > self.iters:
            raise ValueError(f"The <checkpoint_freq> ({self.checkpoint_freq}) cannot be greater than <iters> ({self.iters}).")

        # Validate storage path if provided
        if self.storage_path is not None:
            storage_path = Path(self.storage_path)
            if not storage_path.parent.exists():
                raise ValueError(f"The storage path parent directory does not exist: <storage_path = {storage_path.parent}>.")

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

    @staticmethod
    def _get_cpus_count() -> int:
        """
        Get the number of CPUs available.

        Returns:
            int: The number of CPUs available.
        """
        return max(1, cpu_count() - 4)

    @staticmethod
    def _get_cuda_gpu_count() -> int:
        """
        Get the number of CUDA GPUs available.

        Returns:
            int: The number of CUDA GPUs available.
        """
        return device_count() if is_available() else 0

    def _setup_config(self, embeddings_dim: int = 128) -> Tuple[AlgorithmConfig, Dict[str, Any]]:
        """
        Setup the algorithm configuration based on the specified algorithm type,
        and dynamically select valid hyperparameters for PBT based on algorithm.

        Args:
            embeddings_dim: The dimension of the embeddings.

        Returns:
            Tuple[AlgorithmConfig, Dict[str, Any]]: Tuple containing the algorithm configuration
            and the hyperparameter mutations.
        """
        # Setup the hyperparameter mutations based on the algorithm type
        if self.algorithm == RLAlgorithm.PPO:
            hyperparam_mutations = {"lr": loguniform(0.0001, 0.01),}
                                    # "gamma": loguniform(0.9, 0.99),
                                    # "entropy_coeff": loguniform(0.01, 10.0),
                                    # "grad_clip": loguniform(0.1, 10.0)}
                                    # "num_epochs": grid_search([30, 50, 100]),
                                    # "minibatch_size": grid_search([128, 512, 2048])}

        elif self.algorithm == RLAlgorithm.APPO:
            hyperparam_mutations = {"lr": loguniform(0.0001, 0.01),}
                                    # "entropy_coeff": loguniform(0.01, 10.0),
                                    # "grad_clip": loguniform(0.1, 10.0)}

        elif self.algorithm == RLAlgorithm.SAC:
            hyperparam_mutations = {"actor_lr": loguniform(0.0001, 0.01),
                                    "critic_lr": loguniform(0.0001, 0.01),}
                                    # "gamma": loguniform(0.9, 0.99),
                                    # "grad_clip": loguniform(0.1, 10.0)}

        else:
            raise ValueError(f"The <algorithm> must be an RLAlgorithm enum, got <algorithm = {self.algorithm.value}>, supported algorithms: ['PPO', 'APPO', 'SAC'].")

        # Setup the complete configuration
        config = self._setup_algorithm_config()
        config = self._setup_extra_config(config)
        config = self._setup_training_config(config, embeddings_dim)

        return config, hyperparam_mutations

    def _setup_algorithm_config(self) -> AlgorithmConfig:
        """
        Setup the algorithm configuration.

        Returns:
            AlgorithmConfig: The algorithm configuration.
        """
        if self.algorithm == RLAlgorithm.PPO:
            return (PPOConfig()
                    .training(lr=1e-5,
                              gamma=0.99,
                              use_critic=True,
                              use_gae=True,
                              lambda_=0.95,
                              num_epochs=30,
                              minibatch_size=128,
                              kl_coeff=0.0,
                              entropy_coeff=0.01,
                              clip_param=0.2,
                              grad_clip=0.5))

        elif self.algorithm == RLAlgorithm.APPO:
            return (APPOConfig()
                    .training(lr=1e-5,
                              vtrace=True,
                              lambda_=0.95,
                              kl_coeff=0.0,
                              entropy_coeff=0.01,
                              grad_clip=0.5))

        elif self.algorithm == RLAlgorithm.SAC:
            if self.training in [TrainingMode.CTCE, TrainingMode.CTDE]:
                # For centralized modes: sum all agents' action dimensions
                target_entropy = -sum(space.shape[0] for space in self.env.action_space["grouped_agents"])
                replay_buffer_type = "EpisodeReplayBuffer"
            else:
                # For decentralized mode: each agent has individual action space
                target_entropy = -self.env.action_spaces[self.env.agents_id[0]].shape[0]
                replay_buffer_type = "MultiAgentEpisodeReplayBuffer"

            return (SACConfig()
                    .training(actor_lr=1e-5,
                              critic_lr=1e-5,
                              gamma=0.99,
                              twin_q=True,
                              target_entropy=target_entropy,
                              grad_clip=0.5,
                              replay_buffer_config={"type": replay_buffer_type,
                                                    "capacity": int(1e3),
                                                    "replay_sequence_length": 1,
                                                    "replay_burn_in": 0,
                                                    "replay_zero_init_states": True})
                    .env_runners(batch_mode="complete_episodes"))

        else:
            raise ValueError(f"The <algorithm> must be an RLAlgorithm enum, got <algorithm = {self.algorithm.value}>, supported algorithms: ['PPO', 'APPO', 'SAC'].")

    def _setup_extra_config(self, config: AlgorithmConfig) -> AlgorithmConfig:
        """
        Setup the extra configuration.

        Args:
            config: The algorithm configuration.

        Returns:
            AlgorithmConfig: The algorithm configuration with the extra configuration.
        """
        if self.algorithm == RLAlgorithm.PPO:
            evaluation_config = PPOConfig.overrides(exploration=False)
        elif self.algorithm == RLAlgorithm.APPO:
            evaluation_config = APPOConfig.overrides(exploration=False)
        elif self.algorithm == RLAlgorithm.SAC:
            evaluation_config = SACConfig.overrides(exploration=False)
        else:
            raise ValueError(f"The <algorithm> must be an RLAlgorithm enum, got <algorithm = {self.algorithm.value}>, supported algorithms: ['PPO', 'APPO', 'SAC'].")

        return (config
                .framework("torch")
                .env_runners(num_env_runners=self.cpus,
                             sample_timeout_s=None)
                .learners(num_learners=1,
                          num_gpus_per_learner=self.gpus)
                .evaluation(evaluation_interval=self.evaluation_interval,
                            evaluation_duration=self.evaluation_duration,
                            evaluation_duration_unit=StorageUnit.EPISODES,
                            evaluation_num_env_runners=self.cpus,
                            evaluation_parallel_to_training=True,
                            evaluation_config=evaluation_config)
                # .callbacks(RewardCallbacks)
                )

    def _setup_training_config(self,
                               config: AlgorithmConfig,
                               embeddings_dim: int = 128) -> AlgorithmConfig:
        """
        Setup the training mode configuration.

        Args:
            config: The algorithm configuration.
            embeddings_dim: The dimension of the embeddings.

        Returns:
            AlgorithmConfig: The algorithm configuration with the mode configuration.
        """
        if self.training == TrainingMode.CTCE:
            return (config
                    .environment(GroupedLEM,
                                 env_config=self.env_config)
                    .rl_module(rl_module_spec=RLModuleSpec(module_class=CTCEModule,
                                                           observation_space=self.env.observation_space,
                                                           action_space=self.env.action_space,
                                                           model_config={"algorithm": self.algorithm.value,
                                                                         "embeddings_dim": embeddings_dim})))

        elif self.training == TrainingMode.CTDE:
            return (config
                    .environment(GroupedLEM,
                                 env_config=self.env_config)
                    .rl_module(rl_module_spec=RLModuleSpec(module_class=CTDEModule,
                                                           observation_space=self.env.observation_space,
                                                           action_space=self.env.action_space,
                                                           model_config={"algorithm": self.algorithm.value,
                                                                         "agents_id": self.env.original_agents_id,
                                                                         "embeddings_dim": embeddings_dim})))

        elif self.training == TrainingMode.DTDE:
            # Setup the policy of each agent
            agents_policy = {agent_id: PolicySpec(observation_space=self.env.observation_spaces[agent_id],
                                                  action_space=self.env.action_spaces[agent_id])
                             for agent_id in self.env.agents_id}

            # Setup the RL module of each agent
            agents_rl_module = {agent_id: RLModuleSpec(module_class=self.algorithm.get_module_class(),
                                                       observation_space=agents_policy[agent_id].observation_space,
                                                       action_space=agents_policy[agent_id].action_space)
                                for agent_id in self.env.agents_id}

            return (config
                    .environment(LocalEnergyMarket,
                                 env_config=self.env_config)
                    .multi_agent(policies=agents_policy,
                                 policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id)
                    .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=agents_rl_module)))

        else:
            raise ValueError(f"The <training> must be a TrainingMode enum, got <training = {self.training.value}>, supported training modes: ['CTCE', 'CTDE', 'DTDE'].")

    def _setup_custom_algo_config(self, config: AlgorithmConfig) -> AlgorithmConfig:
        """
        Setup the custom algorithm config.

        Args:
            config: The algorithm configuration.

        Returns:
            AlgorithmConfig: The algorithm configuration with the custom algorithm config.
        """
        class CustomAlgorithm(PPOConfig if self.algorithm == RLAlgorithm.PPO else APPOConfig if self.algorithm == RLAlgorithm.APPO else SACConfig):
            """
            Custom algorithm config that inherits from the appropriate base class.
            """
            def __init__(self, algo_config):
                """
                Initialize the custom algorithm config.

                Args:
                    algo_config: The algorithm configuration.
                """
                self.algo_config = algo_config

            def reset_config(self, new_config) -> bool:
                """
                Reset the configuration of the algorithm.

                Args:
                    new_config: The new configuration.
                """
                for k, v in new_config.items():
                    setattr(self.algo_config, k, v)
                return True

        return CustomAlgorithm(config)

    def _save_env_config(self, results: ResultGrid) -> None:
        """Save environment configuration to experiment directory.

        Args:
            results: The results of the training and evaluation.
        """
        EnvConfigHandler.save(self.env_config,
                              storage_path=str(Path(results.get_best_result().path).parent),
                              name="env_config")

    def train(self,
              embeddings_dim: int = 128,
              _checkpoint_path: Optional[str] = None,
              _iters: Optional[int] = None) -> Tuple[ResultGrid, Dict[str, Any]]:
        """
        Train the RL algorithm.

        Args:
            embeddings_dim: The dimension of the embeddings.
            _checkpoint_path: The path of the checkpoint to restore an experiment.
            _iters: The number of iterations to train for restored experiments.

        Returns:
            ResultGrid: The results of the training and evaluation.
            Dict[str, Any]: The metrics of the training and evaluation.
        """
        # Setup configuration
        algo_config, hyperparam_mutations = self._setup_config(embeddings_dim)
        algo_config = self._setup_custom_algo_config(algo_config).algo_config

        if _checkpoint_path:
            algo_config.callbacks(on_algorithm_init=lambda algorithm, **kwargs: algorithm.restore_from_path(_checkpoint_path))

        # Initialize ray
        if not ray.is_initialized():
            ray.init()

        # Initialize the tuner
        tuner = Tuner(trainable=algo_config.algo_class,
                      param_space=algo_config,
                      tune_config=TuneConfig(num_samples=self.tune_samples,
                                             scheduler=PopulationBasedTraining(time_attr=TRAINING_ITERATION,
                                                                               perturbation_interval=self.checkpoint_freq,
                                                                               hyperparam_mutations=hyperparam_mutations),
                                             metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                                             mode="max",
                                             reuse_actors=True),
                      run_config=RunConfig(storage_path=self.storage_path,
                                           name=f"lem_{self.algorithm.value}_{self.training.value}_{datetime.now().strftime('%d%B%H%M')}",
                                           stop={TRAINING_ITERATION: self.iters if _iters is None else abs(int(_iters)),
                                                 #  "time_total_s": None,
                                                 DONE: True},
                                           checkpoint_config=CheckpointConfig(checkpoint_score_attribute=f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                                                                              checkpoint_frequency=self.checkpoint_freq,
                                                                              checkpoint_at_end=True,
                                                                              num_to_keep=3)))

        # Run the tuner and capture the results
        results = tuner.fit()

        # Save environment configuration
        self._save_env_config(results)

        # Extract metrics from the best result
        metrics = self.get_best_metrics(results)

        # Shutdown ray
        ray.shutdown()

        return results, metrics

    def restore_experiment(self,
                           experiment_path: str,
                           embeddings_dim: int = 128) -> Tuple[ResultGrid, Dict[str, Any]]:
        """Restore tuner from experiment directory to continue training.

        Args:
            experiment_path: Path to the Ray RLlib experiment.
            embeddings_dim: The dimension of the embeddings.

        Returns:
            ResultGrid: The results of the training and evaluation.
            Dict[str, Any]: The metrics of the training and evaluation.
        """
        if not Tuner.can_restore(experiment_path):
            raise ValueError(f"Cannot restore tuner from path: {experiment_path}.")

        # Setup configuration
        algo_config, _ = self._setup_config(embeddings_dim)
        algo_config = self._setup_custom_algo_config(algo_config).algo_config

        try:
            tuner = Tuner.restore(path=experiment_path,
                                  trainable=algo_config.algo_class)

        except Exception as e:
            raise ValueError(f"Failed to restore tuner from experiment {experiment_path}: {e}.")

        # Initialize ray
        if not ray.is_initialized():
            ray.init()

        # Run the restored tuner and capture the results
        results = tuner.fit()

        # Save environment configuration to experiment directory
        self._save_env_config(results)

        # Extract metrics from the best result
        metrics = self.get_best_metrics(results)

        # Shutdown ray
        ray.shutdown()

        return results, metrics

    def train_checkpoint(self,
                         checkpoint_path: str,
                         iters: Optional[int] = None,
                         embeddings_dim: int = 128) -> Tuple[ResultGrid, Dict[str, Any]]:
        """Continue training an RL algorithm from a checkpoint.

        Args:
            checkpoint_path: The path of the checkpoint to restore.
            iters: The number of iterations to train for restored checkpoint.
            embeddings_dim: The dimension of the embeddings.

        Returns:
            ResultGrid: The results of the training and evaluation.
            Dict[str, Any]: The metrics of the training and evaluation.
        """
        return self.train(embeddings_dim,
                          checkpoint_path,
                          iters)

    @staticmethod
    def get_best_results(results: ResultGrid) -> Tuple[ResultGrid, ResultGrid]:
        """Get the best results of an experiment using as principal metric the
        `episode_return_mean`.

        Args:
            results: The results of an Ray Tune experiment.

        Returns:
            Tuple[ResultGrid, ResultGrid]: The best results of the training and evaluation.
        """
        train = results.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                                        mode="max")

        eval = results.get_best_result(metric=f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                                       mode="max")

        return train, eval

    def get_best_metrics(self,
                         results: ResultGrid,
                         as_dataframe: bool = False) -> Dict[str, Any]:
        """Get the metrics for the best result of an experiment.

        Args:
            results: The results of an Ray Tune experiment.
            as_dataframe: Whether to return the metrics as a pandas DataFrame.

        Returns:
            Dict[str, Any]: The training and evaluation metrics of an experiment.
        """
        train, eval = self.get_best_results(results)

        return {"train": train.metrics_dataframe if as_dataframe else train.metrics,
                "eval": eval.metrics_dataframe if as_dataframe else eval.metrics}

    def get_best_hyperparameters(self, results: ResultGrid) -> Dict[str, Any]:
        """Get the hyperparameters for the best result of an experiment.

        Args:
            results: The results of an Ray Tune experiment.

        Returns:
            Dict[str, Any]: The training and evaluation hyperparameters of an experiment.
        """
        train, eval = self.get_best_results(results)

        return {"train": train.config,
                "eval": eval.config}

    def get_best_experiment(self, results: ResultGrid) -> Dict[str, Any]:
        """Get the path of the best result of an experiment.

        Args:
            results: The results of an Ray Tune experiment.

        Returns:
            Dict[str, Any]: The training and evaluation path of an experiment.
        """
        train, eval = self.get_best_results(results)

        return {"train": train.path,
                "eval": eval.path}

    def get_best_checkpoint(self, results: ResultGrid) -> Dict[str, Any]:
        """Get the checkpoint for the best result of an experiment.

        Args:
            results: The results of an Ray Tune experiment.

        Returns:
            Dict[str, Any]: The training and evaluation checkpoint of an experiment.
        """
        train, eval = self.get_best_results(results)

        return {"train": train.checkpoint,
                "eval": eval.checkpoint}

    def get_best_trial(self, results: ResultGrid) -> Dict[str, Any]:
        """Get the trial for the best result of an experiment.

        Args:
            results: The results of an Ray Tune experiment.

        Returns:
            Dict[str, Any]: The training and evaluation trial of an experiment.
        """
        # Setup configuration
        _, hyperparam_mutations = self._setup_config()

        # Get the best results
        train, eval = self.get_best_results(results)

        return {"train": {k: v for k, v in train.config.items() if k in hyperparam_mutations},
                "eval": {k: v for k, v in eval.config.items() if k in hyperparam_mutations}}


class GroupedLEM(MultiAgentEnv):
    """
    Grouped environment for the local energy market.
    """

    def __init__(self, env_config: Dict[str, Any]) -> None:
        """Initialize the grouped environment.

        Args:
            env_config: Environment configuration dictionary
        """
        super().__init__()

        # Multi-agent environment
        env = LocalEnergyMarket(env_config)

        _tuple_obs_space = self._dict_to_tuple_space(env.observation_spaces)
        _tuple_act_space = self._dict_to_tuple_space(env.action_spaces)

        # Original `agents_id`
        self.original_agents_id = env.agents_id

        # Grouped environment
        self.env = env.with_agent_groups(groups={"grouped_agents": env.agents_id},
                                         obs_space=_tuple_obs_space,
                                         act_space=_tuple_act_space)
        self.agents_id = ["grouped_agents"]
        self.agents = self.possible_agents = self.agents_id
        self.observation_space = spaces.Dict({"grouped_agents": _tuple_obs_space})
        self.action_space = spaces.Dict({"grouped_agents": _tuple_act_space})

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset the environment.

        Args:
            seed: The seed for the environment.
            options: The options for the environment.

        Returns:
            obs, infos = self.env.reset(seed, options)
        """
        # Reset the environment
        obs, infos = self.env.reset(seed=seed,
                                    options=options)

        # Pack the observations into a group
        grouped_obs = {k: tuple(v) for k, v in obs.items()}

        return grouped_obs, infos

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, float, bool, bool, MultiAgentDict]:
        # Unpack the actions from the group to individual agent actions
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)

        # Pack the observations into a group
        grouped_obs = {k: tuple(v) for k, v in obs.items()}

        # The reward for the group is the sum of individual rewards
        grouped_reward = sum(rewards.values())

        return grouped_obs, grouped_reward, terminateds["__all__"], truncateds["__all__"], infos

    @staticmethod
    def _dict_to_tuple_space(dict_space: spaces.Dict) -> spaces.Tuple:
        """Converts a gymnasium Dict space to a Tuple space.

        It sorts the dictionary keys to ensure the resulting tuple has a
        consistent order.

        Args:
            dict_space: The Dict space to convert.

        Returns:
            A Tuple space containing the subspaces from the original Dict.
        """
        # Sort keys to ensure the order of spaces is always the same
        sorted_keys = sorted(dict_space.keys())

        # Create a tuple of the subspaces in the sorted order
        tuple_of_spaces = tuple(dict_space[key] for key in sorted_keys)

        return spaces.Tuple(tuple_of_spaces)


class RewardCallbacks(RLlibCallback):
    """Custom callbacks for tracking per-agent rewards during training and evaluation."""

    def on_episode_created(self, *, episode, env_runner, metrics_logger, env, env_index, rl_module):
        episode.custom_data["ENERGY_TRADES"] = []
        episode.custom_data["MARKET_PRICES"] = []
        episode.custom_data["AGENT_PROFITS"] = {agent_id: [] for agent_id in env.agents_id}

    def on_episode_step(self, *, episode, env_runner, metrics_logger, env, env_index, rl_module):
        # Track market metrics
        info = episode.get_info()
        if info:
            episode.custom_data["ENERGY_TRADES"].append(len(info.get('trades', [])))
            episode.custom_data["MARKET_PRICES"].append(info.get('market_price', 0.0))

            # Track individual agent profits
            for agent_id in env.agents_id:
                agent = env._agents.get(agent_id)
                if agent:
                    episode.custom_data["AGENT_PROFITS"][agent_id].append(agent.profit)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        # Log market-specific metrics
        total_trades = sum(episode.custom_data["ENERGY_TRADES"])
        avg_market_price = np.mean(episode.custom_data["MARKET_PRICES"]) if episode.custom_data["MARKET_PRICES"] else 0.0

        # Calculate agent-specific metrics
        agent_returns = episode.get_rewards()
        total_return = sum([sum(rewards) for rewards in agent_returns.values()])

        # Log aggregated metrics
        in_evaluation = kwargs.get("evaluation") or kwargs.get("in_evaluation")

        metrics_logger.log_value("market/total_trades", total_trades, reduce="mean", window=100)
        metrics_logger.log_value("market/avg_price", avg_market_price, reduce="mean", window=100)
        metrics_logger.log_value("market/total_return", total_return, reduce="mean", window=100)
        metrics_logger.log_value("evaluation_mode", in_evaluation, reduce="mean", window=100)

        # Log individual agent profits if available
        for agent_id, profits in episode.custom_data["AGENT_PROFITS"].items():
            if profits:
                metrics_logger.log_value(f"agent/{agent_id}/avg_profit", np.mean(profits), reduce="mean", window=100)

        # Calculate and log additional reward statistics
        rewards = list(episode.agent_rewards.values())
        if rewards:
            metrics_logger.log_value("agent/min_reward", np.min(rewards), reduce="mean", window=100)
            metrics_logger.log_value("agent/max_reward", np.max(rewards), reduce="mean", window=100)
            metrics_logger.log_value("agent/mean_reward", np.mean(rewards), reduce="mean", window=100)
            metrics_logger.log_value("agent/std_reward", np.std(rewards), reduce="mean", window=100)

        # Log cumulative rewards for each agent at episode end
        for agent_id, total_reward in episode.agent_rewards.items():
            episode.custom_metrics[f"{agent_id}_episode_reward"] = total_reward
            metrics_logger.log_value(f"agent/{agent_id}/episode_reward", total_reward, reduce="mean", window=100)
