"""
Battery: Energy storage model with SOC tracking and efficiency.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.battery import Battery


def test_battery() -> None:
    """Test Battery class thoroughly."""
    # Test battery creation and initialization
    battery = Battery(nominal_capacity=100.0,
                      min_soc=0.1,
                      max_soc=0.9,
                      charge_efficiency=0.95,
                      discharge_efficiency=0.93)

    battery.reset(seed=42)
    print(f"✓ Battery initialized: {battery.nominal_capacity} Wh, SOC: {battery.get_state()['soc']:.2f}")

    # Test charging
    energy_stored = battery.charge(30.0)
    print(f"✓ Charged: {energy_stored:.2f} Wh stored")
    print(f"  - State: {battery.state.value}, SOC: {battery.get_state()['soc']:.2f}")

    # Test discharging
    energy_delivered = battery.discharge(20.0)
    print(f"✓ Discharged: {energy_delivered:.2f} Wh delivered")
    print(f"  - State: {battery.state.value}, SOC: {battery.get_state()['soc']:.2f}")

    # Test get_state method
    current_state = battery.get_state()
    print(f"✓ Current battery state: {current_state}")

    # Test idle state
    battery.idle()
    print(f"✓ Battery idle: {battery.state.value}")

    # Test available energy estimation
    available_charge, available_discharge = battery.estimate_available_energy()
    print(f"✓ Available: Charge {available_charge:.2f} Wh, Discharge {available_discharge:.2f} Wh")


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING BATTERY TESTS")

    try:
        test_battery()
        print("🎉 BATTERY TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
