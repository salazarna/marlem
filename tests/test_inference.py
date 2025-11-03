"""
RLInference: Test inference functionality for trained RL models.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_simple_env_config

from src.environment.inference import RLInference
from src.root import __main__


def test_inference_dtde(path: str) -> bool:
    """Test DTDE inference functionality.

    Args:
        path: Path to the checkpoint to restore.

    Returns:
        bool: True if the inference was successful, False otherwise.
    """
    # Create environment configuration
    env_config = create_simple_env_config()

    # Test DTDE inference
    print("--- STEP 1. Testing DTDE Inference ---")
    try:
        model = RLInference(env_config=env_config,
                            exploration=False,
                            checkpoint_path=path,
                            storage_path=f"{__main__}/downloads")

        print("✓ DTDE model loaded successfully")
        print(f"  - Environment type: {type(model.env).__name__}")
        print(f"  - RL Module type: {type(model.rl_module).__name__}")
        print(f"  - Exploration mode: {model.exploration}")

        # Run inference
        iters = 3
        print(f"\n--- STEP 2. Running {iters} inference episodes ---")
        model.inference(iters)

        print("✓ DTDE inference completed successfully!")

    except Exception as e:
        print(f"✗ DTDE inference failed: {e}")
        return False

    return True


def test_inference_ctde(path: str) -> bool:
    """Test CTDE inference functionality.

    Args:
        path: Path to the checkpoint to restore.

    Returns:
        bool: True if the inference was successful, False otherwise.
    """
    # Create environment configuration
    env_config = create_simple_env_config()

    # Test CTDE inference
    print("--- STEP 1. Testing CTDE Inference ---")
    try:
        model = RLInference(env_config=env_config,
                            exploration=False,
                            checkpoint_path=path,
                            storage_path=f"{__main__}/downloads")

        print("✓ CTDE model loaded successfully")
        print(f"  - Environment type: {type(model.env).__name__}")
        print(f"  - RL Module type: {type(model.rl_module).__name__}")
        print(f"  - Exploration mode: {model.exploration}")

        # Run inference
        iters = 3
        print(f"\n--- STEP 2. Running {iters} inference episodes ---")
        model.inference(iters)

        print("✓ CTDE inference completed successfully!")

    except Exception as e:
        print(f"✗ CTDE inference failed: {e}")
        return False

    return True


def test_inference_ctce(path: str) -> bool:
    """Test CTCE inference functionality.

    Args:
        path: Path to the checkpoint to restore.

    Returns:
        bool: True if the inference was successful, False otherwise.
    """
    # Create environment configuration
    env_config = create_simple_env_config()

    # Test CTCE inference
    print("--- STEP 1. Testing CTCE Inference ---")
    try:
        model = RLInference(env_config=env_config,
                            exploration=False,
                            checkpoint_path=path,
                            storage_path=f"{__main__}/downloads")

        print("✓ CTCE model loaded successfully")
        print(f"  - Environment type: {type(model.env).__name__}")
        print(f"  - RL Module type: {type(model.rl_module).__name__}")
        print(f"  - Exploration mode: {model.exploration}")

        # Run inference
        iters = 3
        print(f"\n--- STEP 2. Running {iters} inference episodes ---")
        model.inference(iters)

        print("✓ CTCE inference completed successfully!")

    except Exception as e:
        print(f"✗ CTCE inference failed: {e}")
        return False

    return True


def run_tests(ctce_path: str,
              ctde_path: str,
              dtde_path: str) -> bool:
    """Run all inference tests.

    Args:
        ctce_path: Path to the CTCE checkpoint.
        ctde_path: Path to the CTDE checkpoint.
        dtde_path: Path to the DTDE checkpoint.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING CHECKPOINT RESTORATION TESTS")

    try:
        test_inference_dtde(dtde_path)
        test_inference_ctde(ctde_path)
        test_inference_ctce(ctce_path)

        print("🎉 Restore Experiment and Train Checkpoint SHOWCASE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        return False


if __name__ == "__main__":
    success = run_tests(ctce_path=f"{__main__}/downloads/INFERENCE/lem_ctce_ppo_06September1355/PPO_GroupedLEM_1db39_00000_0_2025-09-06_13-55-54/checkpoint_000000",
                        ctde_path=f"{__main__}/downloads/INFERENCE/lem_ctde_ppo_06September1358/PPO_GroupedLEM_7c262_00000_0_2025-09-06_13-58-33/checkpoint_000000",
                        dtde_path=f"{__main__}/downloads/INFERENCE/lem_ppo_dtde_06September1412/PPO_LocalEnergyMarket_73c9e_00000_0_2025-09-06_14-12-37/checkpoint_000000")
    exit(0 if success else 1)
