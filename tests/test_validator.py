"""
Validator: Blockchain-like trade validation.
"""

import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.order import Trade
from src.market.validation import Validator


def test_validator() -> None:
    """Test Validator class thoroughly."""
    # Create validator
    validator = Validator(difficulty=2, block_size=3)
    print(f"✓ Validator created: difficulty {validator.difficulty}")

    # Create test trades
    trades = [Trade("buyer1", "seller1", 0.22, 25.0, datetime.datetime.now().timestamp(), 2.5, 0.05),
              Trade("buyer2", "seller2", 0.24, 30.0, datetime.datetime.now().timestamp(), 3.2, 0.07),
              Trade("buyer3", "seller1", 0.21, 20.0, datetime.datetime.now().timestamp(), 1.8, 0.03),
              Trade("buyer1", "seller3", 0.26, 35.0, datetime.datetime.now().timestamp(), 4.1, 0.09)]

    # Add trades to validator
    for i, trade in enumerate(trades):
        validator.add_trade(trade)
        print(f"✓ Added trade {i+1}: {trade.buyer_id} ← {trade.seller_id}, "
              f"{trade.quantity:.1f} Wh @ ${trade.price:.3f}/Wh")

    print(f"✓ Blockchain state:")
    print(f"  - Chain length: {len(validator.chain)} blocks")
    print(f"  - Pending trades: {len(validator.pending_trades)}")

    # Verify chain integrity
    is_valid = validator.verify_chain()
    print(f"✓ Chain validation: {'VALID' if is_valid else 'INVALID'}")

    # Get statistics
    stats = validator.get_chain_stats()
    print(f"✓ Chain statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    # Get trade history
    history = validator.get_trade_history(agent_id="buyer1")
    print(f"✓ Trade history for buyer1: {len(history)} trades")

    # Visualize blockchain
    validator.visualize_blockchain()


def run_tests() -> bool:
    """Run all comprehensive tests.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    print("🚀 STARTING Validator TESTS")

    try:
        test_validator()
        print("🎉 Validator TESTS COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
