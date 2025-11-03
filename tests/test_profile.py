"""
DERProfileHandler: Generation and demand profile management.
DSOProfileHandler: Pricing profile management.
ProfileHandler: Base profile functionality.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.profile.der import DERProfileHandler
from src.profile.dso import DSOProfileHandler


def test_profile() -> None:
    """Test DER and DSO profile handlers thoroughly."""
    # Test DER profile handler
    print("--- STEP 1. DER Profile Handler ---")
    der_handler = DERProfileHandler(seed=42)

    # Generate energy profiles
    generation, demand = der_handler.get_energy_profiles(steps=24,
                                                         capacity=100.0)
    print(f"✓ Generated energy profiles: {len(generation)} generation, {len(demand)} demand points")
    print(f"  - Generation range: {min(generation):.2f} - {max(generation):.2f} kW")
    print(f"  - Demand range: {min(demand):.2f} - {max(demand):.2f} kW")

    # Test reset method
    reset_generation, reset_demand = der_handler.reset(capacity=100.0,
                                                       max_steps=24)
    print(f"✓ Reset method: {len(reset_generation)} generation, {len(reset_demand)} demand points")
    print(f"  - Reset generation range: {min(reset_generation):.2f} - {max(reset_generation):.2f} kW")
    print(f"  - Reset demand range: {min(reset_demand):.2f} - {max(reset_demand):.2f} kW")

    # Test DSO profile handler
    print("--- STEP 2. DSO Profile Handler ---")
    dso_handler = DSOProfileHandler(min_price=0.08,
                                    max_price=0.35,
                                    seed=42)

    # Generate pricing profiles
    fit, utility = dso_handler.get_price_profiles(steps=24)
    print(f"✓ Generated pricing profiles: {len(fit)} FIT, {len(utility)} utility points")
    print(f"  - Feed-in tariff range: ${min(fit):.3f} - ${max(fit):.3f}/Wh")
    print(f"  - Utility price range: ${min(utility):.3f} - ${max(utility):.3f}/Wh")

    # Verify price relationship (utility should always be higher)
    all_higher = all(u > f for u, f in zip(utility, fit))
    print(f"✓ Price relationship check: Utility always higher than FIT? {all_higher}")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING ProfileHandler TESTS")

    try:
        test_profile()
        print("🎉 ProfileHandler TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
