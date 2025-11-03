"""
Test experiment restoration and train checkpoint functionality for RLTrainer.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_simple_env_config

from src.environment.train import RLAlgorithm, RLTrainer, TrainingMode
from src.root import __main__


def test_restore_experiment(path: str) -> bool:
    """Test the new checkpoint restoration and inference functionality.

    Args:
        path: Path to the experiment to restore.

    Returns:
        bool: True if the experiment was restored successfully, False otherwise.
    """

    # Use shared environment configuration
    env_config = create_simple_env_config()

    # STEP 2. Initialize trainer
    trainer = RLTrainer(env_config=env_config,
                        algorithm=RLAlgorithm.PPO,
                        training=TrainingMode.DTDE,
                        iters=5,
                        tune_samples=1,
                        checkpoint_freq=2,
                        evaluation_interval=1,
                        evaluation_duration=3,
                        cpus=1,
                        gpus=0,
                        storage_path=f"{__main__}/downloads")

    try:
        results, metrics = trainer.restore_experiment(experiment_path=path,
                                                      embeddings_dim=128)
        print(trainer.get_best_metrics(results, as_dataframe=True))

    except Exception as e:
        print(f"✗ Checkpoint restoration failed: {e}")
        return False

    return True


def test_train_checkpoint(path: str) -> bool:
    """Test the train_checkpoint method for continuing training from a checkpoint.

    Args:
        path: Path to the checkpoint to restore.

    Returns:
        bool: True if the checkpoint was restored successfully, False otherwise.
    """

    # Use shared environment configuration
    env_config = create_simple_env_config()

    # Initialize trainer
    trainer = RLTrainer(env_config=env_config,
                        algorithm=RLAlgorithm.PPO,
                        training=TrainingMode.DTDE,
                        iters=5,
                        tune_samples=1,
                        checkpoint_freq=2,
                        evaluation_interval=1,
                        evaluation_duration=3,
                        cpus=1,
                        gpus=0,
                        storage_path=f"{__main__}/downloads")

    try:
        # Test continuing training from a checkpoint
        results, metrics = trainer.train_checkpoint(checkpoint_path=path,
                                                    iters=3,
                                                    embeddings_dim=128)

        print("✓ Train checkpoint completed successfully!")
        print(f"  - Best metrics: {trainer.get_best_metrics(results)}")

    except Exception as e:
        print(f"✗ Train checkpoint failed: {e}")
        return False

    return True


def run_tests(experiment_path: str,
              checkpoint_path: str) -> bool:
    """Run all checkpoint restoration and inference tests.

    Args:
        experiment_path: Path to the experiment to restore.
        checkpoint_path: Path to the checkpoint to restore.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING CHECKPOINT RESTORE AND TRAIN CHECKPOINT TESTS")

    try:
        test_restore_experiment(experiment_path)
        test_train_checkpoint(checkpoint_path)

        print("🎉 Restore Experiment and Train Checkpoint SHOWCASE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        return False


if __name__ == "__main__":
    success = run_tests(experiment_path=f"{__main__}/downloads/TRAIN/lem_ppo_dtde_06September1412",
                        checkpoint_path=f"{__main__}/downloads/TRAIN/lem_ppo_dtde_06September1412/PPO_LocalEnergyMarket_73c9e_00000_0_2025-09-06_14-12-37/checkpoint_000000")
    exit(0 if success else 1)
