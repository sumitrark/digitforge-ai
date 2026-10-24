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
    return generate_from_latent(model, make_latent(n, seed))


def generate_from_latent(model, latent_vectors):
    """Run the Generator on explicit latent vectors."""
    latent_vectors = np.asarray(latent_vectors, dtype="float32")
    if latent_vectors.ndim != 2 or latent_vectors.shape[1] != LATENT_DIM:
        raise ValueError(f"Expected latent vectors shaped (n, {LATENT_DIM}).")
    return np.asarray(model(latent_vectors, training=False))


def interpolation_latents(steps=9, seed_start=7, seed_end=42):
    """Create a linear interpolation between two reproducible latent vectors."""
    if steps < 2:
        raise ValueError("Interpolation requires at least 2 steps.")
    start = make_latent(1, seed_start)[0]
    end = make_latent(1, seed_end)[0]
    amounts = np.linspace(0.0, 1.0, steps, dtype="float32")[:, None]
    return ((1.0 - amounts) * start + amounts * end).astype("float32")


def generate_interpolation(model, steps=9, seed_start=7, seed_end=42):
    """Generate images along a linear path between two latent vectors."""
    latents = interpolation_latents(steps, seed_start, seed_end)
    return generate_from_latent(model, latents)


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
