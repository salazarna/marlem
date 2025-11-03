"""
ActionHandler: Action management and execution.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_agents, create_dso_agent, create_market_config

from src.environment.action import ActionHandler


def test_action() -> None:
    """Test ActionHandler class thoroughly."""
    # Use shared utilities to create test components
    market_config = create_market_config(min_price=0.05,
                                         max_price=0.50)
    agents = create_agents(num_agents=3,
                           capacity=100.0)

    # Override capacities for specific test
    agents[0].capacity = 80.0
    agents[1].capacity = 100.0
    agents[2].capacity = 120.0
    dso_agent = create_dso_agent(agent_id="DSO")

    # Create action handler
    action_handler = ActionHandler(agents,
                                   dso_agent,
                                   market_config)
    print(f"✓ Action handler created for {len(agents)} agents")

    # Test action space creation
    print(f"✓ Action spaces created:")
    for agent in agents:
        action_space = action_handler.action_space[agent.id]
        print(f"  - {agent.id}: {action_space}")

    # Test action validation
    print(f"\n--- Testing Action Validation ---")

    # Valid actions
    valid_actions = {"agent_0": np.array([0.25, 50.0, 1.0, 0.0], dtype=np.float32),
                     "agent_1": np.array([0.30, 75.0, 0.0, 1.0], dtype=np.float32),
                     "agent_2": np.array([0.20, 100.0, 1.0, 2.0], dtype=np.float32)}

    for agent_id, action in valid_actions.items():
        is_valid = action_handler.is_valid_action(agent_id, action)
        print(f"✓ {agent_id} valid action: {'VALID' if is_valid else 'INVALID'}")

    # Invalid actions
    invalid_actions = {"agent_0": np.array([0.60, 50.0, 1.0, 0.0], dtype=np.float32),  # Price too high
                       "agent_1": np.array([0.30, 150.0, 0.0, 1.0], dtype=np.float32),  # Quantity too high
                       "agent_2": np.array([0.20, 100.0, 1.0, 5.0], dtype=np.float32)}   # Partner index too high

    for agent_id, action in invalid_actions.items():
        is_valid = action_handler.is_valid_action(agent_id, action)
        print(f"✓ {agent_id} invalid action: {'VALID' if is_valid else 'INVALID'}")

    # Test partner ID retrieval
    print(f"\n--- Testing Partner ID Retrieval ---")
    for i in range(len(agents)):
        partner_id = action_handler.get_partner_id(i)
        print(f"✓ Partner index {i} → {partner_id}")

    print(f"\n✓ Action handler testing completed successfully!")


def run_tests() -> bool:
    """Run ActionHandler showcase.

    Returns:
        bool: True if tests passed, False otherwise

    Raises:
        Exception: If tests fail
    """
    print("🚀 STARTING ActionHandler SHOWCASE...")

    try:
        test_action()
        print("🎉 ActionHandler SHOWCASE COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
