"""
A custom RLModule for Centralized Training, Centralized Execution (CTCE)
with PPO, APPO, and SAC.
"""

from typing import Any, List, Optional, Tuple

import torch
from ray.rllib.algorithms.sac.sac_learner import QF_PREDS, QF_TWIN_PREDS
from ray.rllib.core import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchDiagGaussian, TorchMultiDistribution
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis import TARGET_NETWORK_ACTION_DIST_INPUTS, QNetAPI, TargetNetworkAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import NetworkType, StateDict, TensorType

from .base import RLAlgorithm


class CTCEModule(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI, QNetAPI):
    """
    A custom RLModule for Centralized Training, Centralized Execution (CTCE)
    with PPO, APPO, and SAC.
    """

    @override(TorchRLModule)
    def setup(self):
        """Initializes the neural networks for the CTCE module."""
        # Get attributes from model_config
        self.algorithm = self.model_config["algorithm"]
        embeddings_dim = self.model_config["embeddings_dim"]

        # Get the observation and action space (both are Dict({"grouped_agents": Tuple(...)}))
        tuple_obs_space = self.observation_space["grouped_agents"]
        tuple_act_space = self.action_space["grouped_agents"]

        # The observation/action space is the concatenated observations/actions of all agents
        concat_obs_dim = sum(space.shape[0] for space in tuple_obs_space)
        concat_act_dim = sum(space.shape[0] for space in tuple_act_space)

        # Shared encoder and policy networks for processing the concatenated observations/actions
        self.pi_encoder = self._get_encoder_network(concat_obs_dim, embeddings_dim)
        self.pi = self._get_head_network(embeddings_dim, concat_act_dim * 2)  # action_dim * 2 for TorchDiagGaussian (i.e., mean + log_std)

        # Value function network that outputs a single value for the centralized state
        self.vf_head = self._get_head_network(embeddings_dim, 1)

        if self.algorithm == RLAlgorithm.SAC.value:
            # Input to Q-networks is [concat_obs_dim, concat_act_dim]
            qf_encoder_input_dim = concat_obs_dim + concat_act_dim

            # Q-networks encoders for embeddings
            self.qf_encoder = self._get_encoder_network(qf_encoder_input_dim, embeddings_dim)
            self.qf_twin_encoder = self._get_encoder_network(qf_encoder_input_dim, embeddings_dim)

            # Q-networks heads for action logits
            self.qf = self._get_head_network(embeddings_dim, 1)
            self.qf_twin = self._get_head_network(embeddings_dim, 1)

        # For a tuple action space, Ray RLlib needs to know how to split the flat action_logits tensor
        self.action_dist_cls = self._get_distribution_class()

    @override(TorchRLModule)
    def _forward_inference(self,
                           batch: SampleBatch,
                           **kwargs) -> StateDict:
        """The forward pass for inference (exploitation).

        Args:
            batch: The batch received in the forward pass.

        Returns:
            A dictionary with action logits. The learner will use these
            to create the action distribution.
        """
        embeddings = self._compute_embeddings(batch)
        action_logits, _ = self._compute_logits(embeddings)

        return {Columns.ACTION_DIST_INPUTS: action_logits}

    @override(TorchRLModule)
    def _forward_exploration(self,
                             batch: SampleBatch,
                             **kwargs) -> StateDict:
        """The forward pass for exploration (data collection).

        Args:
            batch: The batch received in the forward pass.

        Returns:
            A dictionary with action logits. The learner will use these
            to create the action distribution.
        """
        return self._forward_inference(batch, **kwargs)

    @override(TorchRLModule)
    def _forward_train(self,
                       batch: SampleBatch,
                       **kwargs) -> StateDict:
        """
        The forward pass for training.

        Args:
            batch: The batch received in the forward pass.

        Returns:
            A dictionary with action logits. The learner will use these
            to create the action distribution.
        """
        embeddings = self._compute_embeddings(batch)
        action_logits, _ = self._compute_logits(embeddings)

        fwd_out = {Columns.ACTION_DIST_INPUTS: action_logits,
                   Columns.EMBEDDINGS: embeddings}

        if self.algorithm == RLAlgorithm.SAC.value:
            # FIRST PASS: Compute Q-values for actions sampled from replay buffer to train the critic
            # Concatenate the tuple of observations/actions into a single tensor
            concat_obs = torch.cat(batch[Columns.OBS]["grouped_agents"], dim=-1)
            concat_act = torch.cat(batch[Columns.ACTIONS]["grouped_agents"], dim=-1)

            # Embeddings from Q-network encoders with the concatenated observations/actions
            qf_input = torch.cat([concat_obs, concat_act], dim=-1)
            qf_embeddings = self.qf_encoder(qf_input)
            qf_twin_embeddings = self.qf_twin_encoder(qf_input)

            # Q-values from both Q-networks
            fwd_out[QF_PREDS] = self.qf(qf_embeddings).squeeze(-1)
            fwd_out[QF_TWIN_PREDS] = self.qf_twin(qf_twin_embeddings).squeeze(-1)

            # SECOND PASS: Compute Q-values for actions sampled from current policy to train the actor
            # Action distribution from the current policy action_logits
            curr_action_dist = self.action_dist_cls.from_logits(action_logits)
            curr_actions_resampled = curr_action_dist.sample()

            # Embeddings from Q-network encoders with the current concatenated observations/actions
            curr_qf_input = torch.cat([concat_obs, *curr_actions_resampled["grouped_agents"]], dim=-1)
            curr_qf_embeddings = self.qf_encoder(curr_qf_input)
            curr_qf_twin_embeddings = self.qf_twin_encoder(curr_qf_input)

            # Get q_curr values from both Q-networks
            fwd_out["q_curr"] = torch.min(self.qf(curr_qf_embeddings).squeeze(-1),
                                          self.qf_twin(curr_qf_twin_embeddings).squeeze(-1))
            fwd_out["logp_resampled"] = curr_action_dist.logp(curr_actions_resampled)

            # THIRD PASS: Compute Q-values for actions sampled from target policy to train the critic
            # Concatenate the tuple of next observations into a single tensor
            concat_next_obs = torch.cat(batch[Columns.NEXT_OBS]["grouped_agents"], dim=-1)

            # Get embeddings and action_logits from next observations
            next_obs_embeddings = self.target_pi_encoder(concat_next_obs)
            next_obs_action_logits = self.target_pi(next_obs_embeddings)

            # Target action distribution from the next observation's policy action_logits
            target_action_dist = self.action_dist_cls.from_logits(next_obs_action_logits)
            target_actions_resampled = target_action_dist.sample()

            # Embeddings from target Q-network encoders with the concatenated next observations/actions
            target_qf_input = torch.cat([concat_next_obs, *target_actions_resampled["grouped_agents"]], dim=-1)
            target_qf_embeddings = self.target_qf_encoder(target_qf_input)
            target_qf_twin_embeddings = self.target_qf_twin_encoder(target_qf_input)

            # Get Q-values from target Q-networks
            fwd_out["q_target_next"] = torch.min(self.target_qf(target_qf_embeddings).squeeze(-1),
                                                 self.target_qf_twin(target_qf_twin_embeddings).squeeze(-1))
            fwd_out["logp_next_resampled"] = target_action_dist.logp(target_actions_resampled)

        return fwd_out

    @override(ValueFunctionAPI)
    def compute_values(self,
                       batch: SampleBatch,
                       embeddings: Optional[Any] = None) -> TensorType:
        """
        Computes the value function predictions for the critic.

        Args:
            batch: The batch received in the forward pass.
            embeddings: The embeddings received in the forward pass.

        Returns:
            A tensor with the value function predictions.
        """
        if embeddings is None:
            embeddings = self._compute_embeddings(batch)

        _, vf_preds = self._compute_logits(embeddings)

        return vf_preds

    @override(TargetNetworkAPI)
    def make_target_networks(self) -> None:
        """Creates the required target networks for this RLModule."""
        self.target_pi_encoder = make_target_network(self.pi_encoder)
        self.target_pi = make_target_network(self.pi)

        if self.algorithm == RLAlgorithm.SAC.value:
            self.target_qf_encoder = make_target_network(self.qf_encoder)
            self.target_qf = make_target_network(self.qf)
            self.target_qf_twin_encoder = make_target_network(self.qf_twin_encoder)
            self.target_qf_twin = make_target_network(self.qf_twin)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        """Returns a list of (main_net, target_net) tuples for the target networks.

        Returns:
            A list of (main_net, target_net) tuples for the target networks.
        """
        target_pairs = [(self.pi_encoder, self.target_pi_encoder),
                        (self.pi, self.target_pi)]

        if self.algorithm == RLAlgorithm.SAC.value:
            target_pairs.extend([(self.qf_encoder, self.target_qf_encoder),
                                 (self.qf, self.target_qf),
                                 (self.qf_twin_encoder, self.target_qf_twin_encoder),
                                 (self.qf_twin, self.target_qf_twin)])

        return target_pairs

    @override(TargetNetworkAPI)
    def forward_target(self, batch: StateDict) -> StateDict:
        """
        Performs the forward pass through the target networks.

        Args:
            batch: The batch received in the forward pass.

        Returns:
            A dictionary with action logits. The learner will use these
            to create the action distribution.
        """
        # Concatenate the tuple of observations into a single tensor
        concat_obs = torch.cat(batch[Columns.OBS]["grouped_agents"], dim=-1)

        # Get embeddings from target policy encoder
        target_embeddings = self.target_pi_encoder(concat_obs)

        # Get action logits from target policy head
        return {TARGET_NETWORK_ACTION_DIST_INPUTS: self.target_pi(target_embeddings)}

    @override(QNetAPI)
    def compute_q_values(self, batch: StateDict) -> StateDict:
        """
        Computes Q-values for SAC.

        Args:
            batch: The batch received in the forward pass.

        Returns:
            A dictionary containing the Q-value predictions.
        """
        # Concatenate the tuple of observations into a single tensor
        concat_obs = torch.cat(batch[Columns.OBS]["grouped_agents"], dim=-1)
        concat_act = torch.cat(batch[Columns.ACTIONS]["grouped_agents"], dim=-1)

        # Concatenate observations and actions for Q-network input
        qf_input = torch.cat([concat_obs, concat_act], dim=-1)

        # Get embeddings from Q-network encoders
        qf_embeddings = self.qf_encoder(qf_input)
        qf_twin_embeddings = self.qf_twin_encoder(qf_input)

        # Get Q-values from both Q-network heads
        return {QF_PREDS: self.qf(qf_embeddings).squeeze(-1),
                QF_TWIN_PREDS: self.qf_twin(qf_twin_embeddings).squeeze(-1)}

    @override(QNetAPI)
    def compute_advantage_distribution(self, batch: StateDict) -> StateDict:
        """
        Computes the advantage distribution for SAC.

        Note: This method is required by QNetAPI but for non-distributional
        Q-learning (standard SAC), it's a simpler implementation.

        Args:
            batch: The batch containing forward pass outputs.

        Returns:
            A dictionary with Q-values.
        """
        q_values = self.compute_q_values(batch)

        return {QF_PREDS: q_values[QF_PREDS],
                QF_TWIN_PREDS: q_values[QF_TWIN_PREDS]}

    @staticmethod
    def _get_encoder_network(input_dim: int,
                             embedding_dim: int = 64,
                             bias: bool = True) -> NetworkType:
        """Creates a shared encoder network for the policy and Q-networks.

        Args:
            input_dim: The dimension of the input.
            embedding_dim: The dimension of the embedding.
            bias: Whether to use bias in the linear layers.

        Returns:
            A shared encoder network for the policy and Q-networks.
        """
        return torch.nn.Sequential(torch.nn.Linear(input_dim, embedding_dim, bias=bias),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(embedding_dim, embedding_dim, bias=bias),
                                   torch.nn.ReLU())

    @staticmethod
    def _get_head_network(embedding_dim: int,
                          output_dim: int,
                          bias: bool = True) -> NetworkType:
        """Creates a head network for the policy and Q-networks.

        Args:
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output.
            bias: Whether to use bias in the linear layer.

        Returns:
            A head network for the policy and Q-networks.
        """
        return torch.nn.Linear(embedding_dim, output_dim, bias=bias)

    def _compute_embeddings(self, batch: SampleBatch) -> TensorType:
        """Computes the embeddings for the policy and Q-networks.

        Args:
            batch: The batch received in the forward pass.

        Returns:
            A tensor with the embeddings.
        """
        # Concatenate the tuple of observations into a single tensor
        concat_obs = torch.cat(batch[Columns.OBS]["grouped_agents"], dim=-1)

        # Get embeddings from policy encoder with the concatenated observations
        embeddings = self.pi_encoder(concat_obs)

        return embeddings

    def _compute_logits(self, embeddings: TensorType) -> Tuple[TensorType, TensorType]:
        """Computes the logits for the policy and Q-networks.

        Args:
            embeddings: The embeddings received in the forward pass.

        Returns:
            A tuple with the action logits and value predictions.
        """
        # Get action_logits from policy head
        action_logits = self.pi(embeddings)

        # Get value predictions from value function head
        vf_preds = self.vf_head(embeddings).squeeze(-1)

        return action_logits, vf_preds

    def _get_distribution_class(self) -> TorchMultiDistribution:
        """Creates a TorchMultiDistribution class with a baked-in dict structure.

        Returns:
            A TorchMultiDistribution class with a baked-in dict structure.
        """
        tuple_space = self.action_space["grouped_agents"]
        child_dist_cls = TorchDiagGaussian

        class CustomTorchMultiDistribution(TorchMultiDistribution):
            @classmethod
            def from_logits(cls, logits, **kwargs):
                return super().from_logits(logits=logits,
                                           child_distribution_cls_struct={"grouped_agents": tuple([child_dist_cls] * len(tuple_space.spaces))},
                                           input_lens={"grouped_agents": tuple([child_dist_cls.required_input_dim(s) for s in tuple_space.spaces])},
                                           **kwargs)

        return CustomTorchMultiDistribution
