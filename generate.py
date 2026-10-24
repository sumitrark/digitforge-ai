"""Inference helpers: latent vectors -> Generator -> displayable images."""
import io
import numpy as np
from PIL import Image

LATENT_DIM = 100


def make_latent(n, seed=None):
    rng = np.random.default_rng(seed)  # seed=None -> fresh randomness
    return rng.standard_normal((n, LATENT_DIM)).astype("float32")


def generate(model, n, seed=None):
    """Returns float array (n, 28, 28, 1) in the model's [-1, 1] range."""
    return np.asarray(model(make_latent(n, seed), training=False))


def to_uint8(images):
    """[-1, 1] -> [0, 255] (same mapping as the notebook: x*127.5+127.5)."""
    return np.clip(images[..., 0] * 127.5 + 127.5, 0, 255).astype("uint8")


def to_pil(img28, scale=1):
    im = Image.fromarray(img28, mode="L")
    return im.resize((28 * scale, 28 * scale), Image.NEAREST) if scale > 1 else im


def grid_image(u8, cols, scale=6, pad=4):
    n = len(u8)
    rows = -(-n // cols)
    s = 28 * scale
    canvas = Image.new("L", (cols * (s + pad) + pad, rows * (s + pad) + pad), 0)
    for i, a in enumerate(u8):
        canvas.paste(to_pil(a, scale), (pad + (i % cols) * (s + pad), pad + (i // cols) * (s + pad)))
    return canvas


def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
