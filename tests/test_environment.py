"""
LocalEnergyMarket: Main environment for multi-agent RL.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_env_config, create_market_config

from src.environment.lem import LocalEnergyMarket


def test_environment() -> None:
    """Test LocalEnergyMarket environment thoroughly."""
    # Use shared environment configuration with custom parameters
    market_config = create_market_config(min_price=0.05, max_price=10.50)
    env_config = create_env_config(seed=42,
                                   max_steps=24,
                                   num_agents=5,
                                   market_config=market_config)

    # Create and reset environment
    env = LocalEnergyMarket(env_config=env_config)
    observations, info = env.reset(seed=42)

    print(f"✓ Environment initialized:")
    print(f"  - Agents: {len(env.agents)}")
    print(f"  - Max steps: {env.max_steps}")
    print(f"  - Observation space keys: {list(env.observation_spaces.keys())}")
    print(f"  - Action space type: {type(env.action_spaces[list(env.agents)[0]])}")

    # Run a few simulation steps
    total_rewards = {agent_id: 0.0 for agent_id in env.agents}

    for step in range(3):  # Run 3 steps
        # Generate simple valid actions for all agents
        actions = {}
        for agent_id in env.agents:
            # Create a simple, definitely valid action
            actions[agent_id] = np.array([np.random.uniform(market_config.min_price, market_config.max_price),  # price (mid-range)
                                          np.random.uniform(market_config.min_quantity, market_config.max_quantity),  # quantity (ensure above minimum)
                                          1.0 if step % 2 == 0 else 0.0,  # alternating buy/sell
                                          np.random.randint(0, len(env.agents))],  # preferred_partner (first agent index)
                                         dtype=np.float32)

        # Step environment
        observations, rewards, terminated, truncated, info = env.step(actions)

        # Accumulate rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        print(f"✓ Step {step + 1} completed:")
        print(f"  - Market price: ${env.info.get('market_price', 0):.3f}/Wh")
        print(f"  - Market volume: {env.info.get('market_volume', 0):.2f} Wh")
        print(f"  - Grid balance: {env.info.get('grid_balance', 0):.2f} kW")
        print(f"  - Number of trades: {len(env.info.get('trades', []))}")

        # Check if simulation should end
        if all(terminated.values()) or all(truncated.values()):
            break

    # Display final results
    print(f"✓ Simulation summary:")
    print(f"  - Total steps completed: {step + 1}")
    for agent_id, total_reward in total_rewards.items():
        print(f"  - {agent_id} total reward: ${total_reward:.3f}")

    env.close()


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING LocalEnergyMarket TESTS")

    try:
        test_environment()
        print("🎉 LocalEnergyMarket TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
