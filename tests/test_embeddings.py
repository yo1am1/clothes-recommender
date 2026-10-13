import numpy as np
import pytest
from PIL import Image

from clothes_recommender.config import Settings
from clothes_recommender.embeddings import ClipEncoder


def test_constructing_encoder_does_not_load_model():
    # A bogus name would fail on load, so success proves loading is deferred.
    ClipEncoder("not-a-real/model")


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
