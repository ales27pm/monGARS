import asyncio
import io
import logging
from typing import Optional

try:  # heavy deps may be unavailable during testing
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
except ImportError:  # pragma: no cover - optional dependencies
    torch = None
    Image = None
    BlipForConditionalGeneration = None
    BlipProcessor = None

logger = logging.getLogger(__name__)


class ImageCaptioning:
    """Generate captions for images using a pretrained BLIP model."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None,
    ) -> None:
        if not torch or not BlipProcessor or not BlipForConditionalGeneration:
            logger.warning("Image captioning dependencies unavailable.")
            self.processor = None
            self.model = None
            return
        try:
            self.processor = BlipProcessor.from_pretrained(
                model_name, clean_up_tokenization_spaces=True
            )
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            tokenizer = getattr(self.processor, "tokenizer", None)
            if (
                tokenizer
                and getattr(tokenizer, "clean_up_tokenization_spaces", None) is None
            ):
                tokenizer.clean_up_tokenization_spaces = True
        except Exception as e:  # pragma: no cover - model may not be available
            logger.error("Failed to load image captioning model: %s", e)
            self.processor = None
            self.model = None

    def _sync_generate_caption(self, image_data: bytes) -> Optional[str]:
        try:
            image = Image.open(io.BytesIO(image_data))
            inputs = self.processor(image, return_tensors="pt", truncation=True).to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            return self.processor.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        except Exception as e:  # pragma: no cover - PIL/torch errors
            logger.error("Error generating caption: %s", e)
            return None

    async def generate_caption(self, image_data: bytes) -> Optional[str]:
        """Return a caption for the provided image bytes."""
        if not self.model or not self.processor:
            logger.warning("Image captioning model not loaded.")
            return None
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, self._sync_generate_caption, image_data
            )
        except Exception as e:
            logger.error("Error generating caption: %s", e)
            return None

    async def process_image_file(self, image_path: str) -> Optional[str]:
        """Load an image from disk and produce a caption."""
        loop = asyncio.get_running_loop()
        try:

            def read_file() -> bytes:
                with open(image_path, "rb") as f:
                    return f.read()

            data = await loop.run_in_executor(None, read_file)
            return await self.generate_caption(data)
        except FileNotFoundError:
            logger.error("Image file not found: %s", image_path)
            return None
        except Exception as e:  # pragma: no cover
            logger.error("Error processing image file: %s", e)
            return None
