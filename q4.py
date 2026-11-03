"""
CENG 466 - THE1 - Question 4: Downsampling, upsampling and enhancement order

Student 1: Name Surname, ID
Student 2: Name Surname, ID

Usage:
    python q4.py
Reads   data/lol/high, data/uieb/reference   (Q4.2: the 10 + 10 given references)
        data/lol/low_res, data/lol/low       (Q4.3, optional: the 10 given LOL inputs, x4 downsampled / full size)
Writes
    outputs/q4/roundtrip/<down>_<up>/<dataset>/*.png   (Q4.2, <down> in {decim, gauss},
                                                 <up> in {nearest, bilinear, bicubic},
                                                 <dataset> in {lol, uieb})
    outputs/q4/enhance_then_up/*.png            (Q4.3 order A, optional)
    outputs/q4/up_then_enhance/*.png            (Q4.3 order B, optional)

Rules:
  * Interpolation and downsampling must be implemented YOURSELF with NumPy.
    cv2.resize, skimage.transform.resize/rescale, PIL resize, scipy.ndimage.zoom
    and similar functions are NOT allowed in this question.

Conventions (explained in the report):
  * Pixel-centre grid: low-res pixel i covers high-res pixels f*i ... f*i+f-1, so its
    centre lies at high-res coordinate f*i + (f-1)/2.  Upsampling therefore maps the
    high-res pixel x to the low-res coordinate  u = (x + 0.5)/f - 0.5.
  * Borders: indices are clamped to the image (edge replication), also in the prefilter.
"""

import os

import cv2
import numpy as np

from q1 import enhance_low_light  # must be used unchanged

FACTOR = 4
LOL_HIGH = os.path.join("data", "lol", "high")
UIEB_REF = os.path.join("data", "uieb", "reference")
LOW_RES = os.path.join("data", "lol", "low_res")
LOL_LOW = os.path.join("data", "lol", "low")
OUT_ROOT = os.path.join("outputs", "q4")

GAUSS_SIGMA_PER_F = 0.5   # prefilter sigma = 0.5 * factor pixels (about -5 dB at the new Nyquist frequency)
CUBIC_A = -0.5            # Keys cubic convolution parameter (-0.5 = Catmull-Rom; OpenCV uses -0.75)


def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------
def _gauss_kernel(factor):
    """
    Even-length Gaussian (3*factor taps) whose centre is the middle of a factor x factor block,
    i.e. exactly the position of the low-res pixel on the pixel-centre grid (no shift).
    """
    length = 3 * factor
    d = np.arange(length) - (1.5 * factor - 0.5)
    w = np.exp(-d ** 2 / (2.0 * (GAUSS_SIGMA_PER_F * factor) ** 2))
    return (w / w.sum()).astype(np.float32)


def _smooth_decimate_axis(x, factor, axis):
    """Gaussian smoothing and decimation along one axis in one pass (edge replication at borders)."""
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0] // factor
    xp = np.pad(x, [(factor, factor)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    w = _gauss_kernel(factor)
    out = sum(w[j] * xp[j::factor][:n] for j in range(len(w)))
    return np.moveaxis(out, 0, axis)


def downsample(img, factor, prefilter):
    """
    img : uint8 (H, W, 3); H and W are divisible by factor.
    prefilter : False -> plain decimation (keep every factor-th pixel)
                True  -> spatial low-pass (smoothing) filter, then decimation
    returns : uint8 (H // factor, W // factor, 3)

    Plain decimation keeps pixel (factor-1)//2 of every block (the closest integer to the block centre
    (factor-1)/2, so it is still 0.5 pixel off for factor 4); the Gaussian prefilter is centred exactly.
    """
    f = img.astype(np.float32)
    if not prefilter:
        o = (factor - 1) // 2
        return _to_uint8(f[o::factor, o::factor])
    f = _smooth_decimate_axis(f, factor, axis=0)
    f = _smooth_decimate_axis(f, factor, axis=1)
    return _to_uint8(f)


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------
def _coords(n_in, factor):
    """For every output index x: floor(u) and the fraction t = u - floor(u), u = (x + 0.5)/f - 0.5."""
    x = np.arange(n_in * factor)
    u = (x + 0.5) / factor - 0.5
    i0 = np.floor(u).astype(np.int64)
    return i0, (u - i0).astype(np.float32)


def _apply_axis(x, idx, w, axis):
    """out[o] = sum_k w[o, k] * x[idx[o, k]] along `axis` (idx is clamped to the image: edge replication)."""
    x = np.moveaxis(x, axis, 0)
    idx = np.clip(idx, 0, x.shape[0] - 1)
    out = sum(w[:, k].reshape((-1,) + (1,) * (x.ndim - 1)) * x[idx[:, k]] for k in range(idx.shape[1]))
    return np.moveaxis(out, 0, axis)


def upsample_nearest(img, factor):
    """uint8 (h, w, 3) -> uint8 (h * factor, w * factor, 3)"""
    # round(u) = floor((x + 0.5)/f) = x // f: every pixel is repeated factor x factor times
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def _bilinear_taps(n_in, factor):
    i0, t = _coords(n_in, factor)
    return np.stack([i0, i0 + 1], 1), np.stack([1 - t, t], 1)


def upsample_bilinear(img, factor):
    """uint8 (h, w, 3) -> uint8 (h * factor, w * factor, 3)"""
    f = img.astype(np.float32)
    idx, w = _bilinear_taps(f.shape[0], factor)
    f = _apply_axis(f, idx, w, 0)
    idx, w = _bilinear_taps(img.shape[1], factor)
    f = _apply_axis(f, idx, w, 1)
    return _to_uint8(f)


def _keys(s, a=CUBIC_A):
    """Keys cubic convolution kernel."""
    s = np.abs(s)
    near = (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1
    far = a * s ** 3 - 5 * a * s ** 2 + 8 * a * s - 4 * a
    return np.where(s <= 1, near, np.where(s < 2, far, 0.0)).astype(np.float32)


def _bicubic_taps(n_in, factor):
    i0, t = _coords(n_in, factor)
    idx = np.stack([i0 - 1, i0, i0 + 1, i0 + 2], 1)
    w = np.stack([_keys(t + 1), _keys(t), _keys(1 - t), _keys(2 - t)], 1)   # the four weights sum to 1
    return idx, w


def upsample_bicubic(img, factor):
    """uint8 (h, w, 3) -> uint8 (h * factor, w * factor, 3)"""
    f = img.astype(np.float32)
    idx, w = _bicubic_taps(f.shape[0], factor)
    f = _apply_axis(f, idx, w, 0)
    idx, w = _bicubic_taps(img.shape[1], factor)
    f = _apply_axis(f, idx, w, 1)
    return _to_uint8(f)       # clipped: bicubic overshoots at edges


DOWN = {"decim": False, "gauss": True}
UP = {"nearest": upsample_nearest, "bilinear": upsample_bilinear, "bicubic": upsample_bicubic}


# ---------------------------------------------------------------------------
def main():
    # Q4.2: round trip of the given references, 2 prefilter settings x 3 upsamplers
    for dataset, folder in (("lol", LOL_HIGH), ("uieb", UIEB_REF)):
        for name in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for dname, prefilter in DOWN.items():
                low = downsample(img, FACTOR, prefilter)
                for uname, up_fn in UP.items():
                    out_dir = os.path.join(OUT_ROOT, "roundtrip", f"{dname}_{uname}", dataset)
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, name), up_fn(low, FACTOR))
            print(f"round trip done: {dataset}/{name}")
    # TODO (optional): Q4.3 both orders on LOW_RES, compare with LOL_HIGH


if __name__ == "__main__":
    main()