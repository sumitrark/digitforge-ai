import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate as G
from model import build_generator, load_generator

HAS = os.path.exists("artifacts/dcgan_generator.h5") or os.path.exists("artifacts/generator.keras")


def test_missing_model_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="artifacts"):
        load_generator(str(tmp_path / "nope.h5"))


def test_architecture_shape():
    out = G.generate(build_generator(), 16)
    assert out.shape == (16, 28, 28, 1) and np.isfinite(out).all()


@pytest.mark.skipif(not HAS, reason="trained model not in artifacts/")
def test_trained_generator():
    out = G.generate(load_generator(), 16, seed=0)
    assert out.shape == (16, 28, 28, 1) and np.isfinite(out).all()
    assert out.min() >= -1.0001 and out.max() <= 1.0001
    png = G.png_bytes(G.to_pil(G.to_uint8(out)[0]))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_latent_seed_is_reproducible():
    first = G.make_latent(4, seed=123)
    second = G.make_latent(4, seed=123)
    assert first.shape == (4, G.LATENT_DIM)
    assert np.array_equal(first, second)


def test_interpolation_latents_have_expected_shape_and_endpoints():
    start = G.make_latent(1, seed=7)[0]
    end = G.make_latent(1, seed=42)[0]
    path = G.interpolation_latents(steps=9, seed_start=7, seed_end=42)
    assert path.shape == (9, G.LATENT_DIM)
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], end)
    assert np.isfinite(path).all()


def test_interpolation_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="at least 2"):
        G.interpolation_latents(steps=1, seed_start=7, seed_end=42)
