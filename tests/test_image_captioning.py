import builtins
import io

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


@pytest.mark.asyncio
async def test_generate_caption_success(monkeypatch):
    captioner = ImageCaptioning()
    captioner.model = object()
    captioner.processor = object()
    monkeypatch.setattr(captioner, "_sync_generate_caption", lambda data: "cap")
    result = await captioner.generate_caption(b"img")
    assert result == "cap"


@pytest.mark.asyncio
async def test_generate_caption_error(monkeypatch):
    captioner = ImageCaptioning()
    captioner.model = object()
    captioner.processor = object()

    def raise_error(data):
        raise ValueError("bad image")

    monkeypatch.setattr(captioner, "_sync_generate_caption", raise_error)
    result = await captioner.generate_caption(b"img")
    assert result is None


@pytest.mark.asyncio
async def test_process_image_file_success(monkeypatch):
    captioner = ImageCaptioning()

    async def fake_generate(data):
        return "ok"

    monkeypatch.setattr(captioner, "generate_caption", fake_generate)

    class DummyFile(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: DummyFile(b"img"),
    )

    result = await captioner.process_image_file("path")
    assert result == "ok"


@pytest.mark.asyncio
async def test_process_image_file_not_found(monkeypatch):
    captioner = ImageCaptioning()
    monkeypatch.setattr(
        builtins, "open", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    result = await captioner.process_image_file("missing")
    assert result is None
