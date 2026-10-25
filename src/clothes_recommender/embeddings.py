"""CLIP encoder that maps product photos and free-text queries into one vector space."""

from collections.abc import Sequence
from functools import cached_property
from itertools import batched
from typing import Any

import numpy as np
from PIL.Image import Image

# Activations grow with batch size; one 15k-text call used ~6 GB, so inputs are chunked.
MAX_BATCH = 256


class ClipEncoder:
    """`Encoder` backed by a Hugging Face CLIP checkpoint.

    The model is loaded on first use, so constructing an encoder (and importing this
    module) stays cheap for code paths that never embed anything.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model_name = model_name
        self._requested_device = device

    def load(self) -> None:
        """Load the model now instead of inside the first request (slow, and not thread-safe)."""
        _ = self._model, self._processor

    @property
    def dimension(self) -> int:
        return self._model.config.projection_dim

    def encode_images(self, images: Sequence[Image]) -> np.ndarray:
        # CLIP expects 3 channels; uploads are often RGBA PNGs or greyscale.
        chunks = [
            self._embed(
                self._model.get_image_features,
                self._processor(images=[i.convert("RGB") for i in batch], return_tensors="pt"),
            )
            for batch in batched(images, MAX_BATCH, strict=False)
        ]
        return self._stack(chunks)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        # Truncate to CLIP's 77-token context instead of failing on long chat messages.
        chunks = [
            self._embed(
                self._model.get_text_features,
                self._processor(
                    text=list(batch), padding=True, truncation=True, return_tensors="pt"
                ),
            )
            for batch in batched(texts, MAX_BATCH, strict=False)
        ]
        return self._stack(chunks)

    def _stack(self, chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack(chunks)

    def _embed(self, get_features: Any, inputs: Any) -> np.ndarray:
        import torch

        with torch.inference_mode():
            # transformers 5 returns a model output; the projected embedding is its pooler_output.
            features = get_features(**inputs.to(self._device)).pooler_output
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)

    @cached_property
    def _device(self) -> str:
        import torch

        if self._requested_device is not None:
            return self._requested_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def _model(self) -> Any:
        from transformers import CLIPModel

        return CLIPModel.from_pretrained(self._model_name).to(self._device).eval()

    @cached_property
    def _processor(self) -> Any:
        from transformers import CLIPProcessor

        return CLIPProcessor.from_pretrained(self._model_name)
