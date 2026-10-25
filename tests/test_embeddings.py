from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from clothes_recommender.config import Settings
from clothes_recommender.embeddings import MAX_BATCH, ClipEncoder


def test_constructing_encoder_does_not_load_model():
    # A bogus name would fail on load, so success proves loading is deferred.
    ClipEncoder("not-a-real/model")


@pytest.fixture
def offline_encoder():
    import torch

    processor_calls = []
    model_calls = []

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    def processor(**kwargs):
        processor_calls.append(kwargs)
        if "images" in kwargs:
            assert set(kwargs) == {"images", "return_tensors"}
            assert all(image.mode == "RGB" for image in kwargs["images"])
            values = [image.getpixel((0, 0))[0] for image in kwargs["images"]]
        else:
            assert set(kwargs) == {"text", "padding", "truncation", "return_tensors"}
            assert kwargs["padding"] is True
            assert kwargs["truncation"] is True
            values = [int(text.split()[0]) for text in kwargs["text"]]
        assert kwargs["return_tensors"] == "pt"
        return Inputs(values=torch.tensor(values, dtype=torch.float64))

    def features(values):
        assert torch.is_inference_mode_enabled()
        model_calls.append(len(values))
        return SimpleNamespace(
            pooler_output=torch.stack((values, torch.ones_like(values), values + 2), dim=1)
        )

    encoder = ClipEncoder("not-a-real/model", device="cpu")
    encoder._processor = processor
    encoder._model = SimpleNamespace(
        config=SimpleNamespace(projection_dim=3),
        get_image_features=features,
        get_text_features=features,
    )
    return encoder, processor_calls, model_calls


@pytest.mark.parametrize("count", [0, 1, MAX_BATCH, MAX_BATCH + 1])
@pytest.mark.parametrize("kind", ["images", "texts"])
def test_clip_batches_preserve_vectors_and_input_order(offline_encoder, kind, count):
    encoder, processor_calls, model_calls = offline_encoder
    values = np.arange(count) % 251
    if kind == "images":
        inputs = [
            Image.new("RGBA", (1, 1), (int(value), 0, 0, 0))
            if index % 2
            else Image.new("L", (1, 1), int(value))
            for index, value in enumerate(values)
        ]
        modes = [image.mode for image in inputs]
        vectors = encoder.encode_images(inputs)
        assert [image.mode for image in inputs] == modes
    else:
        inputs = [f"{value} " + "long query " * 100 for value in values]
        vectors = encoder.encode_texts(inputs)
        assert [text for call in processor_calls for text in call["text"]] == inputs

    batch_sizes = [min(MAX_BATCH, count - start) for start in range(0, count, MAX_BATCH)]
    assert [len(call[kind if kind == "images" else "text"]) for call in processor_calls] == (
        batch_sizes
    )
    assert model_calls == batch_sizes
    assert vectors.shape == (count, 3)
    assert vectors.dtype == np.float32
    expected = np.column_stack((values, np.ones(count), values + 2)).astype(np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(vectors, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


@pytest.mark.integration
def test_clip_embeds_images_and_texts_into_shared_normalised_space():
    encoder = ClipEncoder(Settings().clip_model_name, device="cpu")
    red_square = Image.new("RGBA", (224, 224), color=(220, 20, 20, 255))

    image_vectors = encoder.encode_images([red_square])
    text_vectors = encoder.encode_texts(["a plain red square", "a plain blue square"])

    assert image_vectors.shape == (1, encoder.dimension)
    assert text_vectors.shape == (2, encoder.dimension)
    assert image_vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(text_vectors, axis=1), 1.0, atol=1e-5)
    red_score, blue_score = text_vectors @ image_vectors[0]
    assert red_score > blue_score
