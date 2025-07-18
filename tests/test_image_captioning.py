import pytest

from monGARS.core.mains_virtuelles import ImageCaptioning


@pytest.mark.asyncio
async def test_generate_caption_returns_none_without_model(monkeypatch):
    captioner = ImageCaptioning()
    # Force missing model to ensure stable test behaviour
    captioner.model = None
    captioner.processor = None
    result = await captioner.generate_caption(b"fake")
    assert result is None
