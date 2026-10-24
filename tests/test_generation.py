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
