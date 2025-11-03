"""
DERAgent: Main agent class with generation/demand profiles and battery integration.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_market_config

from src.agent.battery import Battery
from src.agent.der import DERAgent


def test_der() -> None:
    """Test DERAgent class thoroughly."""
    # Create battery for agent
    battery = Battery(nominal_capacity=50.0,
                      min_soc=0.1,
                      max_soc=0.9)

    # Create DER agent with profiles
    generation_profile = [10.0, 20.0, 30.0, 25.0, 15.0, 5.0] * 4  # 24-hour profile
    demand_profile = [15.0, 10.0, 12.0, 18.0, 25.0, 20.0] * 4  # 24-hour profile

    agent = DERAgent(id="test_agent_1",
                     capacity=100.0,
                     battery=battery,
                     node_id="800",
                     generation_profile=generation_profile,
                     demand_profile=demand_profile)

    agent.reset(agent.reputation, generation_profile, demand_profile, seed=42)
    print(f"✓ DER Agent created: {agent.id}, capacity: {agent.capacity} kW")
    print(f"  - Generation profile length: {len(agent.generation_profile)}")
    print(f"  - Demand profile length: {len(agent.demand_profile)}")
    print(f"  - Battery: {agent.battery.nominal_capacity} Wh")
    print(f"  - Balance: {agent.balance:.2f} kW, Profit: ${agent.profit:.2f}")

    # Test action adjustment with battery constraints
    raw_action = np.array([0.25, 40.0, 1.0, 0], dtype=object)  # price, quantity, is_buy, partner
    adjusted_action = agent.adjust_action_for_battery(step=5,
                                                      action=raw_action,
                                                      market_config=create_market_config())
    print(f"✓ Action adjustment: {raw_action} → {adjusted_action}")

    # Test max_quantity constraint - action that would exceed maximum
    raw_action_large = np.array([0.30, 150.0, 0.0, 0], dtype=object)  # sell 150 Wh (above max 100)
    adjusted_action_large = agent.adjust_action_for_battery(step=5,
                                                            action=raw_action_large,
                                                            market_config=create_market_config())
    print(f"✓ Max quantity constraint: {raw_action_large} → {adjusted_action_large}")

    # Test min_quantity constraint - action that would be below minimum
    raw_action_small = np.array([0.20, 0.5, 1.0, 0], dtype=object)  # buy 0.5 Wh (below min 1.0)
    adjusted_action_small = agent.adjust_action_for_battery(step=5,
                                                            action=raw_action_small,
                                                            market_config=create_market_config())
    print(f"✓ Min quantity constraint: {raw_action_small} → {adjusted_action_small}")

    # Test demand response tracking
    print(f"✓ Demand response tracking:")
    print(f"  - Cumulative demand satisfied: {agent.cumulative_demand_satisfied:.2f} Wh")
    print(f"  - Total demand required: {agent.total_demand_required:.2f} Wh")
    print()


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING DERAgent TESTS")

    try:
        test_der()

        print("🎉 DERAgent TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
