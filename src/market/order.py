"""
Order and trade data classes for the Local Energy Market.
"""

from dataclasses import dataclass
from typing import Optional

from ..grid.base import Location


@dataclass
class Order:
    """Represents a market order."""
    id: str # Unique identifier for the order
    agent_id: str
    price: float
    quantity: float
    is_buy: bool
    timestamp: float
    location: Optional[Location] = None  # Agent's location in the grid
    partner_id: Optional[str] = None  # ID of the preferred trading partner. If specified, the agent prefers to trade with this partner when possible.


@dataclass
class Trade:
    """Represents a completed trade between two agents."""
    buyer_id: str
    seller_id: str
    price: float
    quantity: float
    timestamp: float
    distance: float = 0.0  # Distance between trading agents
    transmission_loss: float = 0.0  # Energy loss due to transmission
    fees: float = 0.0  # Grid-related fees collected by the DSO
