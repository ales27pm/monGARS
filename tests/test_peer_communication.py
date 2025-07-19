import os

import httpx
import pytest

os.environ.setdefault("SECRET_KEY", "test-peer")

from monGARS.api.web_api import app, peer_comm
from monGARS.core.peer import PeerCommunicator


@pytest.fixture
async def client():
    async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
        peer_comm.peers = ["http://test/api/v1/peer/message"]
        peer_comm._client = async_client
        yield async_client


def test_peer_message_roundtrip(event_loop, client):
    communicator = PeerCommunicator(
        peers=["http://test/api/v1/peer/message"], client=client
    )
    message = {"hello": "world"}
    results = event_loop.run_until_complete(communicator.send(message))
    assert results == [True]
