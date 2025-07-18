def test_hippocampus_store_and_history():
    import asyncio

    from monGARS.core.hippocampus import Hippocampus

    hippocampus = Hippocampus()
    asyncio.run(hippocampus.store("u1", "hello", "hi"))
    history = asyncio.run(hippocampus.history("u1"))
    assert history[0].query == "hello"
    assert history[0].response == "hi"
