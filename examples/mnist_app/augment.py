"""Data augmentation utilities for MNIST images (JAX compatible)."""

from functools import partial

import jax
import jax.numpy as jnp


def _bilinear_sample(img: jnp.ndarray, xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    """Bilinear sampling from image (JAX compatible).

    Args:
        img: (H, W) input image
        xs, ys: (H, W) source coordinates (float)

    Returns:
        (H, W) sampled image
    """
    H, W = img.shape

    x0 = jnp.floor(xs).astype(jnp.int32)
    y0 = jnp.floor(ys).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    def safe_get(x, y):
        x_clipped = jnp.clip(x, 0, W - 1)
        y_clipped = jnp.clip(y, 0, H - 1)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        val = img[y_clipped, x_clipped]
        return jnp.where(ok, val, 0.0)

    Ia = safe_get(x0, y0)
    Ib = safe_get(x0, y1)
    Ic = safe_get(x1, y0)
    Id = safe_get(x1, y1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def affine_warp(
    img: jnp.ndarray, A: jnp.ndarray, out_h: int = 28, out_w: int = 28
) -> jnp.ndarray:
    """Apply affine transformation to image (JAX compatible).

    Args:
        img: (H, W) input image
        A: 2x3 affine matrix (output coords -> input coords)
        out_h, out_w: output size

    Returns:
        (out_h, out_w) transformed image
    """
    yy, xx = jnp.meshgrid(
        jnp.arange(out_h, dtype=jnp.float32),
        jnp.arange(out_w, dtype=jnp.float32),
        indexing="ij",
    )
    ones = jnp.ones_like(xx)
    coords = jnp.stack([xx, yy, ones], axis=0).reshape(3, -1)

    src = (A @ coords).reshape(2, out_h, out_w)
    xs = src[0]
    ys = src[1]

    return _bilinear_sample(img, xs, ys)


def rotate_28(
    img: jnp.ndarray, degrees: float | jnp.ndarray, center: tuple[float, float] = (13.5, 13.5)
) -> jnp.ndarray:
    """Rotate 28x28 image (JAX compatible).

    Args:
        img: (28, 28) input image
        degrees: rotation angle (+ is counter-clockwise)
        center: rotation center

    Returns:
        (28, 28) rotated image
    """
    theta = jnp.deg2rad(degrees).astype(jnp.float32)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    cx, cy = center
    cos_i = cos_t
    sin_i = -sin_t

    a = cos_i
    b = -sin_i
    c = cx - a * cx - b * cy
    d = sin_i
    e = cos_i
    f = cy - d * cx - e * cy
    A = jnp.array([[a, b, c], [d, e, f]], dtype=jnp.float32)

    return affine_warp(img, A, 28, 28)


def shift_28(img: jnp.ndarray, dx: float | jnp.ndarray, dy: float | jnp.ndarray) -> jnp.ndarray:
    """Shift 28x28 image (JAX compatible).

    Args:
        img: (28, 28) input image
        dx: shift in x direction (+ is right)
        dy: shift in y direction (+ is down)

    Returns:
        (28, 28) shifted image
    """
    A = jnp.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=jnp.float32)
    return affine_warp(img, A, 28, 28)


def cutout_28(img: jnp.ndarray, x0: int | jnp.ndarray, y0: int | jnp.ndarray, size: int | jnp.ndarray) -> jnp.ndarray:
    """Apply cutout (random masking) to 28x28 image (JAX compatible).

    Args:
        img: (28, 28) input image
        x0, y0: top-left corner of mask
        size: mask square size

    Returns:
        (28, 28) image with cutout applied
    """
    H, W = img.shape
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    mask = (xx >= x0) & (xx < x0 + size) & (yy >= y0) & (yy < y0 + size)
    return jnp.where(mask, 0.0, img)


def _augment_single(
    img: jnp.ndarray,
    key: jax.Array,
    rot_deg: float,
    shift_px: float,
    cutout_size_range: tuple[int, int],
    p_rotate: float,
    p_shift: float,
    p_cutout: float,
) -> jnp.ndarray:
    """Apply random augmentations to a single image."""
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    # Rotation - always compute, select with probability
    do_rotate = jax.random.uniform(k1) < p_rotate
    deg = jax.random.uniform(k2, minval=-rot_deg, maxval=rot_deg)
    rotated = rotate_28(img, deg)
    img = jnp.where(do_rotate, rotated, img)

    # Shift - always compute, select with probability
    do_shift = jax.random.uniform(k3) < p_shift
    dx = jax.random.uniform(k4, minval=-shift_px, maxval=shift_px)
    dy = jax.random.uniform(k5, minval=-shift_px, maxval=shift_px)
    shifted = shift_28(img, dx, dy)
    img = jnp.where(do_shift, shifted, img)

    # Cutout - always compute, select with probability
    do_cutout = jax.random.uniform(k6) < p_cutout
    lo, hi = cutout_size_range
    size = jax.random.randint(k7, (), lo, hi + 1)
    x0 = jax.random.randint(k8, (), 0, jnp.maximum(1, 28 - size))
    y0 = jax.random.randint(k8, (), 0, jnp.maximum(1, 28 - size))
    cutout_img = cutout_28(img, x0, y0, size)
    img = jnp.where(do_cutout, cutout_img, img)

    return jnp.clip(img, 0.0, 1.0)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7), backend="cpu")
def _augment_batch_jit(
    X_in: jnp.ndarray,
    key: jax.Array,
    rot_deg: float,
    shift_px: float,
    cutout_size_range: tuple[int, int],
    p_rotate: float,
    p_shift: float,
    p_cutout: float,
) -> jnp.ndarray:
    """JIT-compiled augmentation for a batch of 2D images."""
    N = X_in.shape[0]
    keys = jax.random.split(key, N)

    augment_fn = lambda img, k: _augment_single(
        img, k, rot_deg, shift_px, cutout_size_range, p_rotate, p_shift, p_cutout
    )
    return jax.vmap(augment_fn)(X_in, keys)


def augment_batch(
    X: jnp.ndarray,
    key: jax.Array,
    rot_deg: float = 15.0,
    shift_px: float = 3.0,
    cutout_size_range: tuple[int, int] = (0, 10),
    p_rotate: float = 0.7,
    p_shift: float = 0.7,
    p_cutout: float = 0.5,
) -> jnp.ndarray:
    """Apply random augmentations to a batch of images (JAX compatible).

    Args:
        X: (N, 28, 28) or (N, 1, 28, 28) input batch
        key: JAX PRNG key
        rot_deg: max rotation angle in degrees
        shift_px: max shift in pixels
        cutout_size_range: (min, max) cutout size
        p_rotate: probability of applying rotation
        p_shift: probability of applying shift
        p_cutout: probability of applying cutout

    Returns:
        Augmented batch with same shape as input
    """
    squeeze_channel = False
    if X.ndim == 4:
        X_in = X[:, 0]
        squeeze_channel = True
    else:
        X_in = X

    out = _augment_batch_jit(
        X_in, key, rot_deg, shift_px, cutout_size_range, p_rotate, p_shift, p_cutout
    )

    if squeeze_channel:
        out = out[:, None, :, :]
    return out
