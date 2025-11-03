"""
Decentralized transaction validation for Local Energy Market.

This module implements a blockchain-inspired validation mechanism for market transactions,
ensuring decentralized verification without requiring a central authority.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .order import Trade


@dataclass
class TransactionBlock:
    """Represents a block of validated market transactions."""
    timestamp: float
    previous_hash: str
    trades: List[Trade]
    nonce: int
    hash: str = ""

    def calculate_hash(self) -> str:
        """Calculate block hash using trades and previous hash."""
        block_content = {"timestamp": self.timestamp,
                         "previous_hash": self.previous_hash,
                         "trades": [{"buyer_id": trade.buyer_id,
                                     "seller_id": trade.seller_id,
                                     "price": float(trade.price),
                                     "quantity": float(trade.quantity),
                                     "timestamp": float(trade.timestamp)} for trade in self.trades],
                         "nonce": self.nonce}

        block_string = json.dumps(block_content, sort_keys=True)

        return hashlib.sha256(block_string.encode()).hexdigest()


class Validator:
    """Handles decentralized validation of market transactions."""

    def __init__(self,
                 difficulty: int = 2,
                 block_size: int = 5) -> None:
        """Initialize the validator.

        Args:
            difficulty: Number of leading zeros required for proof of work
        """
        # Initialize the validator
        self.chain: List[TransactionBlock] = []
        self.difficulty = difficulty
        self.pending_trades: List[Trade] = []
        self.block_size = block_size

        # Create genesis block
        self._create_genesis_block()

    def reset(self) -> None:
        """Reset the validator."""
        # Reset the validator
        self.chain = []
        self.pending_trades = []

        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create the genesis block of the chain."""
        genesis_block = TransactionBlock(timestamp=0,
                                         previous_hash="0",
                                         trades=[],
                                         nonce=0)
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to pending transactions.

        Args:
            trade: Trade to be validated
        """
        self.pending_trades.append(trade)

        # Create new block if enough pending trades
        if len(self.pending_trades) >= self.block_size:
            self._create_new_block()

    def _create_new_block(self) -> None:
        """Create and validate a new block with pending trades."""
        if not self.pending_trades:
            block_timestamp = 0
        else:
            # Use the timestamp of the earliest trade in the block
            block_timestamp = min(trade.timestamp for trade in self.pending_trades)

        new_block = TransactionBlock(timestamp=block_timestamp,
                                     previous_hash=self.chain[-1].hash,
                                     trades=self.pending_trades.copy(),
                                     nonce=0)

        # Proof of work
        while True:
            new_block.hash = new_block.calculate_hash()
            if new_block.hash.startswith("0" * self.difficulty):
                break
            new_block.nonce += 1

        # Add validated block to chain
        self.chain.append(new_block)
        self.pending_trades.clear()

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire chain.

        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]

            # Verify block hash
            if current.hash != current.calculate_hash():
                return False

            # Verify chain continuity
            if current.previous_hash != previous.hash:
                return False

            # Verify proof of work
            if not current.hash.startswith("0" * self.difficulty):
                return False

        return True

    def get_trade_history(self, agent_id: Optional[str] = None) -> List[Trade]:
        """Get validated trade history, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter trades

        Returns:
            List of validated trades
        """
        trades = []
        for block in self.chain[1:]:  # Skip genesis block
            if agent_id:
                trades.extend([trade for trade in block.trades if trade.buyer_id == agent_id or trade.seller_id == agent_id])
            else:
                trades.extend(block.trades)
        return trades

    def get_chain_stats(self) -> Dict:
        """Get statistics about the validation chain.

        Returns:
            Dictionary of chain statistics
        """
        return {"total_blocks": len(self.chain),
                "total_trades": sum(len(block.trades) for block in self.chain),
                "average_block_size": sum(len(block.trades) for block in self.chain) / max(len(self.chain) - 1, 1),
                "chain_valid": self.verify_chain()}

    def visualize_blockchain(self):
        """Prints a simple text-based visualization of the blockchain to the console."""
        if not self.chain:
            print("Blockchain is empty.")

        print("\n----- BLOCKCHAIN VISUALIZATION -----")
        for i, block in enumerate(self.chain):
            print(f"\nBlock Index: {i}")
            print(f"  Timestamp: {block.timestamp}")
            print(f"  Previous Hash: {block.previous_hash[:8]}... (truncated)") # Truncate for readability
            print(f"  Number of Trades: {len(block.trades)}")
            print(f"  Nonce: {block.nonce}")
            print(f"  Hash: {block.hash[:8]}... (truncated)") # Truncate for readability
        print("----- END BLOCKCHAIN VISUALIZATION -----\n")
