import pytest

from monGARS.core.hippocampus import Hippocampus


@pytest.mark.asyncio
async def test_hippocampus_store_and_history():
    hippocampus = Hippocampus()
    await hippocampus.store("u1", "hello", "hi")
    history = await hippocampus.history("u1")
    assert history[0].query == "hello"
    assert history[0].response == "hi"
