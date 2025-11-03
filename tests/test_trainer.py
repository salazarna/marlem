"""
RLTrainer: Reinforcement learning trainer
"""

import os
import sys

from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_simple_env_config

from src.environment.train import RLAlgorithm, RLTrainer, TrainingMode
from src.root import __main__


def test_trainer(algo: str,
                 mode: str) -> None:
    """Test RLTrainer class.

    Args:
        algo: Algorithm to use
        mode: Mode to use
    """
    # STEP 1. Create environment configuration
    print("--- STEP 1. Create Environment Configuration ---")

    # Use shared environment configuration
    env_config = create_simple_env_config()

    # STEP 2. Initialize trainer
    print("--- STEP 2. Initialize RLTrainer ---")

    trainer = RLTrainer(env_config=env_config,
                        algorithm=RLAlgorithm.PPO if algo == "ppo" else RLAlgorithm.APPO if algo == "appo" else RLAlgorithm.SAC if algo == "sac" else None,
                        training=TrainingMode.CTDE if mode == "ctde" else TrainingMode.CTCE if mode == "ctce" else TrainingMode.DTDE if mode == "dtde" else None,
                        iters=5,
                        tune_samples=1,
                        checkpoint_freq=2,
                        evaluation_interval=1,
                        cpus=1,
                        gpus=0,
                        storage_path=f"{__main__}/downloads")

    config, hyperparam_mutations = trainer._setup_config()
    print(f"✓ Algorithm configured: {config.algo_class}")

        # STEP 3. Test basic environment functionality
    print("--- STEP 3. Basic Environment Test ---")

    # Test that we can create and run the environment without Ray
    obs, info = trainer.env.reset(seed=42)
    print("✓ Environment reset successful")
    print(f"  - Agents: {len(trainer.env.agents)}")
    print(f"  - Observation keys: {list(obs.keys())}")
    print(f"  - Action keys: {list(trainer.env.action_spaces) if mode == 'dtde' else list(trainer.env.action_space)}")

    # Test a few environment steps
    for step in range(3):
        # Generate random valid actions
        actions = {}
        for agent_id in trainer.env.agents:
            # Use action_spaces instead of action_space for DTDE mode
            if mode == 'dtde':
                action_space = trainer.env.action_spaces[agent_id]
            else:
                action_space = trainer.env.action_space[agent_id]
            actions[agent_id] = action_space.sample()

        # Step the environment
        obs, rewards, terminated, truncated, info = trainer.env.step(actions)

        print(f"✓ Step {step + 1} completed")
        # Handle both single reward (grouped) and dict rewards (DTDE)
        if isinstance(rewards, dict):
            total_reward = sum(rewards.values())
            print(f"  - Total rewards: {total_reward:.2f} (individual: {rewards})")
        else:
            print(f"  - Total rewards: {rewards:.2f}")
        print(f"  - Market price: ${info.get('market_price', 0):.2f}/kWh")

        if terminated:
            break

    print("✓ Basic environment test completed successfully!")

    # STEP 4. Configuration validation
    print("--- STEP 4. Configuration Validation ---")

    config, hyperparam_mutations = trainer._setup_config()
    print("✓ Configuration setup successful")
    print(f"  - Algorithm: {config.algo_class}")
    print(f"  - Hyperparameters: {list(hyperparam_mutations.keys())}")
    print(f"  - Number of agents: {len(trainer.env.agents)}")

    # STEP 5. Try short training to test serialization
    print("--- STEP 5. Short Training Test ---")
    try:
        results, training_metrics = trainer.train()
        print("✓ Ray Tune training completed successfully!")

        # Get basic results
        training_metrics = ['episode_len_mean', 'episode_return_max', 'episode_return_mean', 'episode_return_min', 'num_env_steps_sampled', 'num_episodes']
        evaluation_metrics = ['evaluation/episode_len_mean', 'evaluation/episode_return_max', 'evaluation/episode_return_mean', 'evaluation/episode_return_min']

        best_result = results.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
        best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
        best_trial = {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}

        # Extract training metrics
        best_training_metrics = {k: v for k, v in best_result.metrics.items() if k in training_metrics}
        # Extract evaluation metrics
        best_evaluation_metrics = {k: v for k, v in best_result.metrics.items() if k in evaluation_metrics}

        # Also check for any evaluation metrics with different naming patterns
        all_eval_metrics = {k: v for k, v in best_result.metrics.items() if 'evaluation' in k.lower()}

        # Print comprehensive metrics
        print(f"  - Best checkpoint: {best_checkpoint}")
        print(f"  - Best training metrics: {best_training_metrics}")
        print(f"  - Best evaluation metrics: {best_evaluation_metrics}")
        if all_eval_metrics and not best_evaluation_metrics:
            print(f"  - All evaluation metrics found: {all_eval_metrics}")
        print(f"  - Best trial: {best_trial}")
        # print(f"  - Best trial metrics: {best_trial_metrics}")

    except Exception as e:
        print(f"✗ Training failed: {e}")


def run_tests() -> bool:
    """Run RLTrainer showcase.

    Returns:
        bool: True if tests passed, False otherwise

    Raises:
        Exception: If tests fail
    """
    print("🚀 STARTING RLTrainer SHOWCASE...")

    try:
        test_trainer(algo="ppo",  # ppo, appo, sac
                     mode="dtde")  # ctce, ctde, dtde
        print("🎉 RLTrainer SHOWCASE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
