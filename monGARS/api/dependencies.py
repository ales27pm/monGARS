from __future__ import annotations

from monGARS.core.hippocampus import Hippocampus
from monGARS.core.peer import PeerCommunicator

hippocampus = Hippocampus()
peer_communicator = PeerCommunicator()


def get_hippocampus() -> Hippocampus:
    """Return the shared Hippocampus instance."""
    return hippocampus


def get_peer_communicator() -> PeerCommunicator:
    """Return the shared PeerCommunicator instance."""
    return peer_communicator
